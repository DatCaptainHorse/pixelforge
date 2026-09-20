//! The codec-generic encoder.
//!
//! Everything that is identical across H.264, H.265 and AV1 lives here:
//! the shared per-encoder state ([`EncoderCommon`]), the generic driver
//! ([`CodecEncoder`]) that owns the public `encode`/`flush`/`set_color_description`
//! flow, the shared initialization scaffolding ([`build_encoder_common`]) and the
//! shared rate-control decision ([`RateControlPlan`]).
//!
//! A codec is anything that implements [`VideoCodec`]: a small state struct that
//! plugs its *differences* into the generic flow — building the codec-specific
//! StdVideo* graph for a frame, tracking its reference pictures, and emitting its
//! parameter sets. Reading a codec's folder shows only those differences; the
//! scaffolding is here.

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::{debug, warn};

use crate::encoder::dpb::MAX_DPB_SLOTS;
use crate::encoder::gop::{GopFrameType, GopPosition, GopStructure};
use crate::encoder::pipeline::{EncodeFuture, EncodePipeline, PipelineConfig, SlotPacketMetadata};
use crate::encoder::resources::{
    EncoderTeardown, UploadParams, align_up, allocate_session_memory, create_command_resources,
    create_dpb_images, destroy_encoder_resources, get_video_format, lcm,
    query_supported_video_formats, upload_image_to_input,
};
use crate::encoder::{ColorDescription, EncodeConfig, FrameType, RateControlMode};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::vulkan::VideoContext;

/// Per-encoder state shared by every codec.
///
/// Holds the Vulkan video session, the DPB images, the async [`EncodePipeline`],
/// the GOP structure and the upload path — none of which differ between codecs.
/// Codec-specific state (reference lists, syntax counters, parameter caches)
/// lives in the [`VideoCodec`] implementor instead.
pub(crate) struct EncoderCommon {
    pub context: VideoContext,
    pub config: EncodeConfig,

    pub video_queue_fn: ash::khr::video_queue::Device,
    pub video_encode_fn: ash::khr::video_encode_queue::Device,
    pub session: vk::VideoSessionKHR,
    pub session_params: vk::VideoSessionParametersKHR,
    pub session_memory: Vec<vk::DeviceMemory>,

    /// Coded extent (display dimensions aligned to the codec block size and the
    /// device's picture-access granularity).
    pub aligned_width: u32,
    pub aligned_height: u32,

    /// Async slot rotation + bitstream readback.
    pub pipeline: EncodePipeline,
    /// Frame-type/POC schedule.
    pub gop: GopStructure,
    /// Monotonic display-order counter (presentation order).
    pub input_frame_num: u64,
    /// Monotonic encode-order counter (decode order / DTS; `0` => first frame).
    pub encode_frame_num: u64,
    /// Set when the rate control state in `config` no longer matches what the
    /// video session was last told, so the next recorded frame re-issues it
    /// through a coding-control command.
    ///
    /// Rate control is session state, not per-frame state: changing
    /// `config.target_bitrate` alone would leave the value chained to
    /// `vkCmdBeginVideoCodingKHR` disagreeing with the session, which is
    /// undefined rather than merely ineffective. The flag is what makes a live
    /// retune a state change instead of a lie.
    pub rate_control_dirty: bool,
    /// Intra refresh, when the device supports it and the config asked.
    ///
    /// `None` means key frames, which is what every stream did before this
    /// existed and what a device without the extension still does.
    pub intra_refresh: Option<IntraRefreshState>,

    pub dpb_images: Vec<vk::Image>,
    pub dpb_image_memories: Vec<vk::DeviceMemory>,
    pub dpb_image_views: Vec<vk::ImageView>,
    pub dpb_slot_count: usize,
    /// Whether each DPB slot has been written at least once (governs the
    /// UNDEFINED-vs-DPB old layout in the pre-encode barrier).
    pub dpb_slot_active: Vec<bool>,
    /// Single layered DPB image (true) vs one image per slot (false).
    pub use_layered_dpb: bool,
    /// DPB slot the current frame reconstructs into.
    pub current_dpb_slot: u8,

    pub command_pool: vk::CommandPool,
    pub upload_command_pool: vk::CommandPool,
    pub upload_command_buffer: vk::CommandBuffer,
    pub upload_fence: vk::Fence,
    /// Caller work this frame's first submission must wait for. The upload
    /// takes them when it copies, otherwise the encode submission does.
    pub pending_waits: Vec<TimelinePoint>,
}

impl EncoderCommon {
    /// Prologue: wait until the slot we are about to record over has been read
    /// back, pull the next GOP position, and snapshot the counters this frame
    /// will encode with.
    pub fn begin_frame(&mut self) -> FramePlan {
        self.pipeline.wait_current_free();
        let gop = self.gop.get_next_frame();
        let display_order = self.input_frame_num;
        self.input_frame_num += 1;
        FramePlan {
            gop,
            display_order,
            encode_index: self.encode_frame_num,
        }
    }

    /// Copy a source image into the current slot's input image. No-op when the
    /// source already *is* the slot image (e.g. converted in place).
    pub fn upload(&mut self, src_image: vk::Image) -> Result<()> {
        let (dst_image, input_image_layout) = {
            let slot = self.pipeline.current();
            (slot.input_image, slot.input_image_layout)
        };
        if src_image == dst_image {
            return Ok(());
        }
        let waits = std::mem::take(&mut self.pending_waits);

        let params = UploadParams {
            upload_command_buffer: self.upload_command_buffer,
            upload_fence: self.upload_fence,
            src_image,
            dst_image,
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
            pixel_format: self.config.pixel_format,
            input_image_layout,
            rgb: self.config.rgb_input.is_some(),
            upload_queue: self.context.transfer_queue(),
            waits: &waits,
        };
        upload_image_to_input(&self.context, &params)?;
        self.pipeline.current_mut().input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;
        Ok(())
    }

    /// Record the metadata for the packet the current slot will produce.
    pub fn set_pending_metadata(&mut self, metadata: SlotPacketMetadata) {
        self.pipeline.set_pending_metadata(metadata);
    }

    /// Submit the recorded command buffer for the current slot, mark its DPB
    /// slot active, and return the future that resolves with its packet.
    pub fn submit_frame(&mut self) -> Result<EncodeFuture> {
        let encode_queue = self.context.video_encode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue available".to_string())
        })?;
        let waits = std::mem::take(&mut self.pending_waits);
        let future = self.pipeline.submit_current(
            self.context.device(),
            self.context.sync2(),
            encode_queue,
            &waits,
        )?;
        self.dpb_slot_active[self.current_dpb_slot as usize] = true;
        Ok(future)
    }

    /// Epilogue: advance to the next pipeline slot.
    pub fn advance(&mut self) {
        self.pipeline.advance();
    }

    /// The `ash` device handle.
    pub fn device(&self) -> &ash::Device {
        self.context.device()
    }
}

/// Everything a single frame needs to know about its schedule, snapshotted by
/// [`EncoderCommon::begin_frame`] so the codec hooks all see a consistent view.
pub(crate) struct FramePlan {
    pub gop: GopPosition,
    /// Presentation order (PTS).
    pub display_order: u64,
    /// Encode order (DTS); `0` marks the very first encoded frame.
    pub encode_index: u64,
}

impl FramePlan {
    pub fn is_idr(&self) -> bool {
        self.gop.frame_type.is_idr()
    }
    pub fn is_reference(&self) -> bool {
        self.gop.is_reference
    }
    pub fn is_b_frame(&self) -> bool {
        self.gop.frame_type == GopFrameType::B
    }
    pub fn is_first_frame(&self) -> bool {
        self.encode_index == 0
    }
    pub fn pic_order_cnt(&self) -> i32 {
        self.gop.pic_order_cnt
    }
    /// The stream-level frame type for packet metadata.
    pub fn frame_type(&self) -> FrameType {
        match self.gop.frame_type {
            GopFrameType::Idr | GopFrameType::I => FrameType::I,
            GopFrameType::P => FrameType::P,
            GopFrameType::B => FrameType::B,
        }
    }
}

/// The codec header (if any) and stream frame type produced by
/// [`VideoCodec::begin_picture`].
pub(crate) struct PictureSetup {
    pub frame_type: FrameType,
    /// Codec header to prepend (SPS/PPS, VPS/SPS/PPS, AV1 sequence header).
    /// `Some` only for frames that carry one (typically the IDR/key frame).
    pub header: Option<Vec<u8>>,
}

/// The resolved rate-control decision for a frame.
///
/// The CQP/CBR/VBR selection logic is identical across codecs; only the default
/// QP the controller starts from differs (H.264/H.265 use 26, AV1 uses 128), so
/// the codec passes that in. Each codec then wires these values into its own
/// `VideoEncode*RateControl*InfoKHR` structs.
pub(crate) struct RateControlPlan {
    pub mode: vk::VideoEncodeRateControlModeFlagsKHR,
    pub average_bitrate: u32,
    pub max_bitrate: u32,
    /// QP/q-index: the configured quality level for CQP/Disabled, otherwise the
    /// codec's default starting point for the bitrate controller.
    pub qp: u32,
    /// Inclusive QP/q-index bounds to hand the encoder, on the codec's own
    /// scale, or `None` to leave the rate controller unconstrained.
    ///
    /// CQP pins both ends to the requested QP -- that is what constant-QP
    /// means. A bitrate mode gets whatever the caller asked for and, by
    /// default, nothing: **a QP floor under a bitrate target is a floor on how
    /// few bits a frame may spend, so any target below what that quality costs
    /// is unreachable.** H.264 used to hardcode a floor of 18 here and H.265
    /// one of 26, which made a CBR target under roughly 8-10 Mbps at 1080p
    /// impossible to hit; the encoder ignored it and emitted what QP 18 cost.
    /// AV1 never had the bug -- it already disabled its bounds under a bitrate
    /// mode, and this is the other two brought in line with it.
    pub qp_bounds: Option<(u32, u32)>,
}

/// Warn when the device cannot do the rate control the caller asked for.
///
/// Vulkan does not fail an encode for this: the session is created, frames come
/// out, and the target bitrate is quietly ignored -- which is the worst possible
/// shape for the failure, because everything downstream reports success while
/// the stream ignores its budget. Intel's ANV advertises `DISABLED` alone and
/// `maxBitrate = 0`, so every CBR encode on it is really constant-QP wearing a
/// bitrate's name, and nothing said so.
/// The GOP length to advertise to the rate controller, from a config's
/// `gop_size`.
///
/// Zero means "no periodic key frames" in this crate's config, and Vulkan spells
/// that `UINT32_MAX`: for `gopFrameCount`, `idrPeriod` and AV1's
/// `keyFramePeriod` alike, zero means *the implementation may assume a period
/// of its choosing* and `UINT32_MAX` means infinite. Those are opposite
/// instructions, and this crate's zero meant the second one.
///
/// Neither codec path said so. AV1 clamped zero to 1, which budgets for a key
/// frame every single frame, while H.264 and H.265 passed zero through and let
/// each driver invent a period. Three codecs, three answers, none of them the
/// one the caller asked for.
///
/// This only shapes the rate controller's bit allocation. Where IDRs actually
/// land is [`GopStructure`](crate::encoder::gop::GopStructure)'s decision and is
/// unaffected either way.
pub(crate) fn advisory_gop_length(gop_size: u32) -> u32 {
    if gop_size == 0 { u32::MAX } else { gop_size }
}

pub(crate) fn rate_control_is_supported(
    mode: RateControlMode,
    supported: vk::VideoEncodeRateControlModeFlagsKHR,
) -> bool {
    let wanted = match mode {
        RateControlMode::Cbr => vk::VideoEncodeRateControlModeFlagsKHR::CBR,
        RateControlMode::Vbr => vk::VideoEncodeRateControlModeFlagsKHR::VBR,
        // Both map to DISABLED, which every implementation must offer.
        RateControlMode::Cqp | RateControlMode::Disabled => return true,
    };
    supported.contains(wanted)
}

pub(crate) fn warn_unsupported_rate_control(
    config: &EncodeConfig,
    supported: vk::VideoEncodeRateControlModeFlagsKHR,
) {
    if !rate_control_is_supported(config.rate_control_mode, supported) {
        tracing::warn!(
            "device does not support {:?} rate control (it offers {supported:?}); \
             the {} bps target will be ignored and the stream will be encoded at \
             a constant quality instead",
            config.rate_control_mode,
            config.target_bitrate,
        );
    }
}

impl RateControlPlan {
    pub fn new(config: &EncodeConfig, controller_default_qp: u32) -> Self {
        match config.rate_control_mode {
            RateControlMode::Cqp | RateControlMode::Disabled => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
                average_bitrate: 0,
                max_bitrate: 0,
                qp: config.quality_level,
                qp_bounds: Some((config.quality_level, config.quality_level)),
            },
            RateControlMode::Cbr => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                average_bitrate: config.target_bitrate,
                max_bitrate: config.target_bitrate,
                qp: controller_default_qp,
                qp_bounds: config.qp_bounds,
            },
            RateControlMode::Vbr => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                average_bitrate: config.target_bitrate,
                max_bitrate: config.max_bitrate,
                qp: controller_default_qp,
                qp_bounds: config.qp_bounds,
            },
        }
    }

    /// The bounds as `(use_bounds, min, max)`, ready for the `use_min_qp` /
    /// `min_qp` pair every codec's rate-control layer info wants.
    ///
    /// Returning the flag alongside the values is what stops the two drifting:
    /// H.265 set `min_qp` and `max_qp` without ever setting `use_min_qp`, so
    /// its bounds were written into the struct and ignored by the driver.
    pub fn qp_bound_fields(&self) -> (bool, i32, i32) {
        match self.qp_bounds {
            Some((min, max)) => (true, min as i32, max as i32),
            None => (false, 0, 0),
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
    }
}

/// A video codec plugged into the generic [`CodecEncoder`].
///
/// Implementors are small state structs (reference lists, syntax counters,
/// parameter caches). The generic driver calls these hooks in order around each
/// frame; everything else (slot rotation, upload, readback, teardown) is shared.
pub(crate) trait VideoCodec: Sized + Send {
    /// Per-frame prologue: codec IDR/key-frame resets and header retrieval.
    /// Runs before recording; returns the packet's frame type and header.
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup>;

    /// Record the codec-specific encode commands and submit the frame. Builds
    /// the StdVideo* graph on its own stack (so the FFI pointers stay valid until
    /// `cmd_encode_video`) and finishes via [`EncoderCommon::submit_frame`].
    fn record_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture>;

    /// Per-frame epilogue: advance reference lists, syntax counters and the next
    /// DPB slot. Runs after submission.
    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan);

    /// Reference frame invalidation: drop every held reference the client can no
    /// longer decode and arrange for the next frame to predict from a surviving
    /// one.
    ///
    /// `first_lost_display_order` is the display order (the `pts` reported on
    /// encoded packets) of the earliest frame the client lost. In a linear P
    /// chain every reference at or after that frame is transitively undecodable,
    /// so all of them are dropped. Returns `true` if a usable reference survives
    /// (the next P-frame can be predicted from it); `false` if none remain and
    /// the caller must fall back to an IDR.
    fn invalidate_references(
        &mut self,
        common: &mut EncoderCommon,
        first_lost_display_order: u64,
    ) -> bool;

    /// Build the codec parameter sets and create Vulkan session parameters.
    /// Used at init and by `set_color_description`.
    fn create_session_params(
        &self,
        common: &EncoderCommon,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR>;

    /// Drop any cached header after the session parameters change.
    fn invalidate_header_cache(&mut self);
}

/// The codec-generic encoder: shared state plus a codec.
///
/// This owns the entire public encode flow; the codec only supplies its
/// differences through [`VideoCodec`].
pub struct CodecEncoder<C: VideoCodec> {
    pub(crate) common: EncoderCommon,
    pub(crate) codec: C,
}

// SAFETY: the only non-Send state is the persistently-mapped bitstream pointer
// inside the pipeline slots, which is synchronized via Vulkan fences and only
// read on the dedicated readback thread (see `pipeline`). The codec state is
// `Send` by the trait bound.
unsafe impl<C: VideoCodec> Send for CodecEncoder<C> {}

impl<C: VideoCodec> CodecEncoder<C> {
    /// The internal input image for the current slot (a `ColorConverter::convert`
    /// target that avoids an intermediate copy).
    pub fn input_image(&self) -> vk::Image {
        self.common.pipeline.input_image()
    }

    /// Encode one frame once `wait` is reached, returning a future for its
    /// packet. See [`crate::Encoder::encode_after`].
    pub fn encode_after(
        &mut self,
        src_image: vk::Image,
        wait: &[TimelinePoint],
    ) -> Result<EncodeFuture> {
        let plan = self.common.begin_frame();
        self.common.pending_waits.clear();
        self.common.pending_waits.extend_from_slice(wait);
        self.common.upload(src_image)?;

        let setup = self.codec.begin_picture(&mut self.common, &plan)?;
        self.common.set_pending_metadata(SlotPacketMetadata {
            frame_type: setup.frame_type,
            is_key_frame: plan.is_idr(),
            pts: plan.display_order,
            dts: plan.encode_index,
            header: setup.header,
            timestamps: [0; 2],
            now: std::time::Instant::now(),
        });

        let future = self.codec.record_picture(&mut self.common, &plan)?;
        self.common.encode_frame_num += 1;
        self.codec.end_picture(&mut self.common, &plan);
        self.common.advance();
        Ok(future)
    }

    /// End-of-stream barrier: wait for all in-flight frames to be read back.
    pub fn flush(&mut self) -> Result<()> {
        self.common.pipeline.flush();
        Ok(())
    }

    /// Force the next frame to be an IDR/key frame.
    pub fn request_idr(&mut self) {
        self.common.gop.request_idr();
    }

    /// Reference frame invalidation: drop references the client lost and predict
    /// the next frame from a surviving reference, falling back to an IDR when no
    /// reference survives. See [`crate::Encoder::invalidate_reference_frames`].
    pub fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        if !self
            .codec
            .invalidate_references(&mut self.common, first_lost_display_order)
        {
            self.common.gop.request_idr();
        }
    }

    /// Change how often an IDR is emitted, live.
    ///
    /// `None` stops periodic IDRs; the stream then carries key frames only when
    /// something asks for one. Takes effect on the next frame, with no session
    /// reset and no encoder rebuild.
    pub fn set_gop_size(&mut self, gop_size: Option<u32>) {
        let size = gop_size.unwrap_or(0);
        if self.common.config.gop_size == size {
            return;
        }
        self.common.config.gop_size = size;
        self.common.gop.set_gop_size(gop_size);
        // The GOP length rides the same per-frame rate-control struct as the
        // bitrate, so it needs the same control command to take effect.
        self.common.rate_control_dirty = true;
    }

    /// Retarget a bitrate-controlled encode, live.
    ///
    /// Takes effect on the next recorded frame and costs nothing else: no
    /// session reset, no encoder rebuild, **and no forced IDR**. That last part
    /// is the point. A caller adapting to a congested path would otherwise emit
    /// a keyframe every time it adjusted -- the largest frame there is, onto the
    /// path least able to carry it -- and the adaptation would cost more than it
    /// saved.
    ///
    /// # Errors
    /// If the encode is not under a bitrate mode. Under
    /// [`RateControlMode::Cqp`] or [`RateControlMode::Disabled`] there is no
    /// bitrate to retarget, and silently accepting one would leave the caller
    /// believing it had changed something.
    pub fn set_target_bitrate(&mut self, bits_per_second: u32) -> Result<()> {
        match self.common.config.rate_control_mode {
            RateControlMode::Cbr | RateControlMode::Vbr => {}
            mode => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "set_target_bitrate needs a bitrate rate-control mode, this encode is {mode:?}"
                )));
            }
        }
        if self.common.config.target_bitrate == bits_per_second {
            return Ok(());
        }
        self.common.config.target_bitrate = bits_per_second;
        self.common.rate_control_dirty = true;
        Ok(())
    }

    /// Rebuild session parameters with a new color description; the next frame is
    /// an IDR/key frame carrying the updated header.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        // Drain in-flight encodes before mutating shared session parameters.
        self.common.pipeline.wait_all_free();

        let old_session_params = self.common.session_params;
        let new_session_params = self.codec.create_session_params(&self.common, &desc)?;
        unsafe {
            self.common
                .video_queue_fn
                .destroy_video_session_parameters(old_session_params, None);
        }

        self.common.session_params = new_session_params;
        self.common.config.color_description = Some(desc);
        self.codec.invalidate_header_cache();
        self.common.gop.request_idr();
        Ok(())
    }
}

impl<C: VideoCodec> Drop for CodecEncoder<C> {
    fn drop(&mut self) {
        unsafe {
            let common = &mut self.common;
            let device = common.context.device();
            // Wait on just the queues this encoder used, not the whole device.
            let _ = device.queue_wait_idle(common.context.transfer_queue());
            if let Some(q) = common.context.video_encode_queue() {
                let _ = device.queue_wait_idle(q);
            }

            common.pipeline.destroy(device);

            destroy_encoder_resources(
                device,
                &common.video_queue_fn,
                &EncoderTeardown {
                    command_pool: common.command_pool,
                    upload_command_pool: common.upload_command_pool,
                    upload_fence: common.upload_fence,
                    dpb_images: &common.dpb_images,
                    dpb_image_views: &common.dpb_image_views,
                    dpb_image_memories: &common.dpb_image_memories,
                    session: common.session,
                    session_params: common.session_params,
                    session_memory: &common.session_memory,
                },
            );
        }
    }
}

/// What a codec passes to [`build_encoder_common`]; the codec owns the parts that
/// genuinely differ (its profile, block size, reference cap), the builder owns
/// the rest.
/// What a device can do with intra refresh, for this profile.
///
/// Copied out of the Vulkan query so it outlives that call's pointer chain,
/// the same reason [`DeviceVideoCaps`] exists.
#[derive(Clone, Copy, Default)]
pub(crate) struct IntraRefreshCaps {
    pub modes: vk::VideoEncodeIntraRefreshModeFlagsKHR,
    pub max_cycle_duration: u32,
    pub max_active_reference_pictures: u32,
    pub partition_independent_refresh_regions: u32,
    pub non_rectangular_refresh_regions: u32,
}

/// Intra refresh as it will actually be used, once the device has been asked.
#[derive(Clone, Debug)]
pub(crate) struct IntraRefreshState {
    /// Pictures in a full cycle.
    pub cycle_duration: u32,
    /// Where the next picture falls in the cycle, `0..cycle_duration`.
    pub index: u32,
    pub mode: vk::VideoEncodeIntraRefreshModeFlagsKHR,
    /// Whether to stop refreshed regions predicting from unrefreshed ones.
    ///
    /// The spec calls this optional -- "applications *may want to* limit the
    /// set of intra refresh regions of the reference picture" -- and what it
    /// buys is convergence for a decoder that joins mid-cycle, or after loss,
    /// without a key frame. What it costs is prediction.
    ///
    /// The cost is not small. Under the restriction, region `i` may predict
    /// only from regions below the current refresh index, so anything moving
    /// vertically out of that band cannot be predicted at all: content drifts
    /// until the refresh reaches it and then snaps into place. Measured at
    /// 1080p with a 240-picture cycle each region is about four pixel rows, so
    /// *any* vertical motion breaks prediction and the snapping is continuous
    /// -- seen as a band crawling down the picture. A 30-picture cycle gives
    /// regions of about 36 rows and is much better, but slow-moving content
    /// still pulses as the band passes.
    ///
    /// So it is off unless asked for. A client that joins with a real key
    /// frame -- which it must, since intra refresh cannot start a decoder --
    /// has nothing to converge from, and pays that cost for nothing.
    pub restrict_prediction: bool,
}

impl IntraRefreshState {
    /// Dirty regions to declare for every active reference.
    ///
    /// Not tracked per reference picture, because the spec does not define it
    /// that way: VUID-vkCmdEncodeVideoKHR-pNext-10843 requires this to equal
    /// the cycle duration minus *the encoded picture's* refresh index, for
    /// every reference declaring a non-zero count. A number derived from the
    /// reference's own history instead happens to agree while the reference is
    /// the immediately preceding picture and is invalid usage the moment it is
    /// not -- which in video encode shows up as a corrupt picture rather than
    /// an error.
    ///
    /// Zero means no restriction, which is the legal way to decline it.
    pub fn dirty_regions(&self) -> u32 {
        if self.restrict_prediction {
            self.cycle_duration.saturating_sub(self.index)
        } else {
            0
        }
    }

    /// Whether the next picture opens a refresh cycle.
    ///
    /// AV1 needs this: the spec asks for `error_resilient_mode` on the first
    /// picture of a cycle, so that CDF data -- the adaptive entropy state,
    /// carried forward between pictures and not covered by refreshing
    /// *samples* -- cannot propagate an error across the boundary.
    pub fn starts_cycle(&self) -> bool {
        self.index == 0
    }

    /// Move to the next picture in the cycle.
    pub fn advance(&mut self) {
        self.index = (self.index + 1) % self.cycle_duration.max(1);
    }

    /// Start the cycle over, because a key frame refreshed everything at once.
    pub fn restart(&mut self) {
        self.index = 0;
    }
}

/// The CTB size to reckon refresh regions in, from what the device offers.
///
/// The largest offered, deliberately. It gives the fewest regions, and the
/// fewest regions is the only bound that holds whichever size the encoder
/// actually ends up coding with -- guessing small would let a cycle through
/// that the picture cannot supply regions for, which is the failure this is
/// here to stop. Erring the other way merely shortens a sweep.
///
/// A device that reports no size has told us nothing, so the largest legal CTB
/// is assumed for the same reason.
pub(crate) fn h265_refresh_block(ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR) -> u32 {
    use vk::VideoEncodeH265CtbSizeFlagsKHR as Ctb;
    for (flag, size) in [(Ctb::TYPE_64, 64), (Ctb::TYPE_32, 32), (Ctb::TYPE_16, 16)] {
        if ctb_sizes.contains(flag) {
            return size;
        }
    }
    64
}

/// The superblock size to reckon refresh regions in. See
/// [`h265_refresh_block`] for why this takes the largest.
pub(crate) fn av1_refresh_block(sizes: vk::VideoEncodeAV1SuperblockSizeFlagsKHR) -> u32 {
    use vk::VideoEncodeAV1SuperblockSizeFlagsKHR as Sb;
    for (flag, size) in [(Sb::TYPE_128, 128), (Sb::TYPE_64, 64)] {
        if sizes.contains(flag) {
            return size;
        }
    }
    128
}

/// How many refresh regions a picture can be divided into under `mode`, or
/// `None` when that is not a question this can answer.
///
/// This is the bound the cycle duration has to respect. A cycle is a schedule
/// for handing out regions one picture at a time, so a cycle longer than the
/// picture has regions is asking for regions that do not exist: the sweep
/// either covers nothing on some pictures or is rejected outright, and neither
/// failure announces itself -- what shows up is a band that crawls unevenly
/// down the picture, which reads as an encoder quality problem rather than a
/// configuration one.
///
/// `block` is the codec's own block size -- macroblock, CTB, superblock --
/// because that is the unit refresh regions are built from. It is emphatically
/// *not* `encodeInputPictureGranularity`, which describes how input images may
/// be laid out and which differs: measured on RADV, H.265 reports a 64x16
/// input granularity while its CTBs are 64x64, and AV1 reports 8x2 against
/// 64x64 superblocks. Using the granularity overstates the row count by more
/// than thirty times for AV1.
fn refresh_region_limit(
    mode: vk::VideoEncodeIntraRefreshModeFlagsKHR,
    width: u32,
    height: u32,
    block: u32,
) -> Option<u32> {
    let block = block.max(1);
    let columns = width.div_ceil(block).max(1);
    let rows = height.div_ceil(block).max(1);
    match mode {
        vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_ROW_BASED => Some(rows),
        vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_COLUMN_BASED => Some(columns),
        // The implementation divides the picture however it likes and does not
        // say which way, so only a bound that holds for either is safe.
        vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_BASED => Some(rows.min(columns)),
        // Regions follow the slice or tile layout, which this does not set.
        _ => None,
    }
}

/// Decide whether intra refresh is on, and how, given what was asked for and
/// what the device offers.
///
/// Returns `None` when it is off, having said why if it was wanted. Silence
/// would leave a stream that still emits key frames while its caller believes
/// otherwise -- and the whole point of asking was to stop emitting them.
pub(crate) fn resolve_intra_refresh(
    context: &VideoContext,
    config: &EncodeConfig,
    caps: IntraRefreshCaps,
    // The codec's own block size -- macroblock, CTB or superblock -- which is
    // the unit refresh regions are built from.
    refresh_block: u32,
    active_reference_pictures: u32,
) -> Option<IntraRefreshState> {
    let cycle = config.intra_refresh_cycle?;
    debug!(
        "intra refresh caps: modes {:?}, max cycle {}, max active refs {}",
        caps.modes, caps.max_cycle_duration, caps.max_active_reference_pictures
    );
    if !context.has_video_encode_intra_refresh() {
        warn!("intra refresh requested, but this device does not support it; using key frames");
        return None;
    }
    // Block-based first, because we have no preference about how the picture
    // is divided and the spec says so explicitly: row-based and column-based
    // are block-based with an extra guarantee about granularity, so anything
    // offering either inherently offers block-based, and asking for the
    // general mode leaves the division to the implementation that knows its
    // own hardware. Picking row-based here would be choosing on the
    // implementation's behalf for no reason we can defend.
    //
    // The specific modes follow only as a fallback for a device that somehow
    // offers one without the general bit. Per-picture partition is last: it
    // ties the refresh region to the slice layout, which is a separate
    // decision this does not control.
    // An explicit preference is honoured or refused, never silently
    // substituted: asking for a vertical sweep and getting a horizontal one
    // would make the next comparison meaningless.
    let mode = match config.intra_refresh_mode {
        Some(shape) => {
            let wanted = match shape {
                crate::encoder::IntraRefreshShape::Blocks => {
                    vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_BASED
                }
                crate::encoder::IntraRefreshShape::Rows => {
                    vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_ROW_BASED
                }
                crate::encoder::IntraRefreshShape::Columns => {
                    vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_COLUMN_BASED
                }
                crate::encoder::IntraRefreshShape::Partitions => {
                    vk::VideoEncodeIntraRefreshModeFlagsKHR::PER_PICTURE_PARTITION
                }
            };
            if !caps.modes.contains(wanted) {
                warn!(
                    "intra refresh shape {shape:?} is not offered by this device (it offers \
                     {:?}); using key frames",
                    caps.modes
                );
                return None;
            }
            Some(wanted)
        }
        // No preference, so the general mode, which the spec says to prefer
        // in exactly that case: row- and column-based are block-based with an
        // added granularity guarantee, so anything offering either offers
        // this, and the implementation knows its own hardware.
        None => [
            vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_BASED,
            vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_ROW_BASED,
            vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_COLUMN_BASED,
            vk::VideoEncodeIntraRefreshModeFlagsKHR::PER_PICTURE_PARTITION,
        ]
        .into_iter()
        .find(|m| caps.modes.contains(*m)),
    };
    let Some(mode) = mode else {
        // Said, not swallowed. The device advertises the extension per device
        // and the modes per *profile*, so one codec having intra refresh says
        // nothing about another: measured on RADV, H.265 offers a mode and
        // H.264 and AV1 offer none. Declining quietly leaves a stream still
        // emitting key frames while its caller believes it is not, and the
        // only visible difference is a burst nobody is expecting.
        warn!(
            "intra refresh requested, but {:?} offers no refresh mode on this device; using key frames",
            config.codec
        );
        return None;
    };

    // A device may support fewer active references under intra refresh than
    // it does otherwise. Exceeding it is invalid usage, and invalid usage in
    // video encode tends to surface as a corrupt picture rather than an error,
    // so this refuses rather than trying its luck.
    if active_reference_pictures > caps.max_active_reference_pictures {
        warn!(
            "intra refresh requested, but it allows {} active reference picture(s) and this \
             encode uses {}; using key frames",
            caps.max_active_reference_pictures, active_reference_pictures
        );
        return None;
    }
    // Prediction inside a cycle is restricted to already-refreshed regions of
    // a reference, which has the practical effect that a picture can usefully
    // reference only the picture before it in the cycle, or one from outside
    // the cycle. Extra references are not invalid -- they are just mostly
    // unusable, and paid for in DPB memory and bandwidth either way.
    if active_reference_pictures > 1 {
        debug!(
            "intra refresh with {active_reference_pictures} active references: prediction inside \
             a cycle can only use the preceding picture, so the rest will contribute little"
        );
    }
    if caps.max_cycle_duration == 0 {
        warn!("intra refresh requested, but the device reports no usable cycle; using key frames");
        return None;
    }
    // Two bounds, and the picture's is the one that used to be missing. The
    // device maximum is generous -- 256 pictures, measured on RADV -- while a
    // 1080p H.265 picture is only 17 CTB rows tall, so a cycle set in seconds
    // clears the device bound easily and overruns the picture's silently.
    let region_limit = refresh_region_limit(
        mode,
        config.dimensions.width,
        config.dimensions.height,
        refresh_block,
    );
    let bound = caps
        .max_cycle_duration
        .min(region_limit.unwrap_or(u32::MAX));
    if bound < 2 {
        warn!(
            "intra refresh requested, but this picture divides into {bound} refresh region(s) \
             under {mode:?}; using key frames"
        );
        return None;
    }
    let cycle_duration = cycle.clamp(2, bound);
    if cycle_duration != cycle {
        match region_limit {
            // Named apart because they mean different things to whoever reads
            // it: the device bound is a property of the hardware, the region
            // bound a property of this resolution and codec, and only the
            // second one moves when the stream is reconfigured.
            Some(limit) if limit < caps.max_cycle_duration => debug!(
                "intra refresh cycle {cycle} exceeds the {limit} refresh region(s) a \
                 {}x{} picture has under {mode:?}; using {cycle_duration}",
                config.dimensions.width, config.dimensions.height,
            ),
            _ => debug!(
                "intra refresh cycle {cycle} exceeds the device maximum {}; using {cycle_duration}",
                caps.max_cycle_duration
            ),
        }
    }
    debug!("intra refresh on: {cycle_duration} pictures per cycle, mode {mode:?}");
    Some(IntraRefreshState {
        cycle_duration,
        index: 0,
        restrict_prediction: config.intra_refresh_recovery,
        mode,
    })
}

/// The per-picture intra refresh structs, and the reference slots updated to
/// declare how stale each of them is.
///
/// Returned together because the reference structs must stay alive as long as
/// the slots that point at them, and separating the two invites one being
/// dropped while the other is still being read by the driver.
pub(crate) struct IntraRefreshFrame {
    pub info: vk::VideoEncodeIntraRefreshInfoKHR<'static>,
    pub dirty: Vec<vk::VideoReferenceIntraRefreshInfoKHR<'static>>,
}

/// Note that a picture has been committed, and move the cycle on.
///
/// Called after the encode is submitted rather than before, for the same
/// reason the H.264 unmark queue is cleared there: a failure on the way to the
/// queue leaves the picture unencoded, and a cycle that advanced anyway would
/// have every later picture claiming a refresh position one ahead of what the
/// stream actually contains -- which a decoder cannot detect and cannot
/// recover from, because every region would appear refreshed a picture before
/// it really was.
pub(crate) fn intra_refresh_committed(common: &mut EncoderCommon, was_idr: bool) {
    let Some(ir) = common.intra_refresh.as_mut() else {
        return;
    };
    if was_idr {
        // A key frame refreshed the whole picture at once, so the cycle has
        // nothing left to do and starts again.
        ir.restart();
        return;
    }
    ir.advance();
}

/// Build this picture's intra refresh structs, if refresh is on.
///
/// The caller must chain `dirty[i]` onto reference slot `i` and `info` onto
/// the encode info, and set [`vk::VideoEncodeFlagsKHR::INTRA_REFRESH`].
pub(crate) fn intra_refresh_frame(
    common: &EncoderCommon,
    ref_slots: &[vk::VideoReferenceSlotInfoKHR],
) -> Option<IntraRefreshFrame> {
    let ir = common.intra_refresh.as_ref()?;
    // The same count for every reference, because that is what the spec
    // requires: it is a function of the encoded picture's refresh index, not
    // of any reference's history.
    let regions = ir.dirty_regions();
    let dirty = ref_slots
        .iter()
        .map(|_| {
            vk::VideoReferenceIntraRefreshInfoKHR::default().dirty_intra_refresh_regions(regions)
        })
        .collect();
    Some(IntraRefreshFrame {
        info: vk::VideoEncodeIntraRefreshInfoKHR::default()
            .intra_refresh_cycle_duration(ir.cycle_duration)
            .intra_refresh_index(ir.index),
        dirty,
    })
}

pub(crate) struct CommonInitRequest<'a> {
    pub context: &'a VideoContext,
    pub config: &'a EncodeConfig,
    /// The codec profile, with its `VideoEncode*ProfileInfoKHR` already chained.
    pub profile_info: &'a vk::VideoProfileInfoKHR<'a>,
    /// Device capabilities for this profile, resolved by [`query_video_caps`].
    pub caps: &'a DeviceVideoCaps,
    /// Codec block size for coded-extent alignment (macroblock / CTB / superblock).
    pub align_unit: u32,
    /// Codec block size to reckon intra refresh regions in.
    ///
    /// Separate from `align_unit` because that one is what this crate codes
    /// with while this one is what the *device* says it divides pictures into,
    /// and they disagree: H.265 aligns to 32 here while RADV offers only 64x64
    /// CTBs. A refresh bound computed from the smaller of the two is twice as
    /// permissive as the picture actually allows, which puts the cycle back
    /// over the limit this exists to keep it under.
    pub refresh_block: u32,
    /// Upper bound on active reference pictures the codec's syntax allows.
    pub max_active_refs_cap: usize,
    pub bitstream_buffer_size: usize,
    /// Whether the codec can use a layered DPB image when the driver lacks
    /// `SEPARATE_REFERENCE_IMAGES` (H.264/H.265 yes; AV1 no).
    pub allow_layered_dpb: bool,
    /// What the hardware RGB conversion supports for this profile, queried
    /// alongside the codec capabilities when [`EncodeConfig::rgb_input`] is
    /// set.
    pub rgb_caps: Option<RgbConversionCaps>,
    /// What the device said about intra refresh for this profile, read from
    /// the codec's own capability query.
    pub intra_refresh_caps: IntraRefreshCaps,
}

/// The profile struct that turns on RGB input. When
/// [`EncodeConfig::rgb_input`] is set it goes on the one profile the codec
/// builds, which every capability query, image, query pool and the session
/// then share: they must all name the same profile.
pub(crate) fn rgb_conversion_profile() -> vk::VideoEncodeProfileRgbConversionInfoVALVE<'static> {
    vk::VideoEncodeProfileRgbConversionInfoVALVE::default().perform_encode_rgb_conversion(true)
}

/// `VkVideoEncodeRgbConversionCapabilitiesVALVE`, without its `pNext`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RgbConversionCaps {
    models: vk::VideoEncodeRgbModelConversionFlagsVALVE,
    ranges: vk::VideoEncodeRgbRangeCompressionFlagsVALVE,
    x_chroma_offsets: vk::VideoEncodeRgbChromaOffsetFlagsVALVE,
    y_chroma_offsets: vk::VideoEncodeRgbChromaOffsetFlagsVALVE,
}

impl From<&vk::VideoEncodeRgbConversionCapabilitiesVALVE<'_>> for RgbConversionCaps {
    fn from(caps: &vk::VideoEncodeRgbConversionCapabilitiesVALVE<'_>) -> Self {
        Self {
            models: caps.rgb_models,
            ranges: caps.rgb_ranges,
            x_chroma_offsets: caps.x_chroma_offsets,
            y_chroma_offsets: caps.y_chroma_offsets,
        }
    }
}

impl RgbConversionCaps {
    /// The session settings that make the hardware produce what `desc`
    /// describes, or why it cannot.
    fn session_info(
        &self,
        desc: &ColorDescription,
    ) -> Result<vk::VideoEncodeSessionRgbConversionCreateInfoVALVE<'static>> {
        use vk::VideoEncodeRgbChromaOffsetFlagsVALVE as Offset;
        use vk::VideoEncodeRgbModelConversionFlagsVALVE as Model;
        use vk::VideoEncodeRgbRangeCompressionFlagsVALVE as Range;

        // H.273 matrix coefficients.
        let model = match desc.matrix_coefficients {
            1 => Model::YCBCR_709,
            5 | 6 => Model::YCBCR_601,
            9 => Model::YCBCR_2020,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "RGB input: no hardware conversion for matrix coefficients {other}"
                )));
            }
        };
        let range = if desc.full_range {
            Range::FULL_RANGE
        } else {
            Range::NARROW_RANGE
        };
        if !self.models.contains(model) || !self.ranges.contains(range) {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "RGB input: the driver converts {:?} in {:?}, not {:?} in {:?}",
                self.models, self.ranges, model, range
            )));
        }
        // Chroma sited midway between luma samples on both axes, which is what
        // the colour converter's 2x2 average produces, so a stream looks the
        // same whichever path made it. Where the driver cannot, it gets
        // whichever siting it can do.
        let pick = |supported: Offset| {
            if supported.contains(Offset::MIDPOINT) {
                Offset::MIDPOINT
            } else {
                Offset::COSITED_EVEN
            }
        };
        Ok(
            vk::VideoEncodeSessionRgbConversionCreateInfoVALVE::default()
                .rgb_model(model)
                .rgb_range(range)
                .x_chroma_offset(pick(self.x_chroma_offsets))
                .y_chroma_offset(pick(self.y_chroma_offsets)),
        )
    }
}

/// Result of [`build_encoder_common`]: the assembled common state plus the
/// negotiated active-reference count the codec needs for its parameter sets.
pub(crate) struct CommonInit {
    pub common: EncoderCommon,
    pub active_reference_count: u32,
}

/// The device-capability fields the generic init needs, resolved once by the
/// codec. Copied out of the Vulkan query so they outlive its pointer chain.
pub(crate) struct DeviceVideoCaps {
    pub picture_access_granularity: vk::Extent2D,
    pub min_coded_extent: vk::Extent2D,
    pub max_coded_extent: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub flags: vk::VideoCapabilityFlagsKHR,
    pub std_header_version: vk::ExtensionProperties,
}

/// Run `vkGetPhysicalDeviceVideoCapabilitiesKHR` and resolve the fields the
/// generic init needs.
///
/// The driver *requires* the codec's `VideoEncode*CapabilitiesKHR` chained into
/// `pNext`, so the codec builds the `capabilities` chain (the only codec-specific
/// part) and passes it in already populated with its encode/codec caps structs.
pub(crate) fn query_video_caps(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    capabilities: &mut vk::VideoCapabilitiesKHR,
) -> Result<DeviceVideoCaps> {
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let result = unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            capabilities,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to query Vulkan Video encode capabilities: {:?}",
            result
        )));
    }
    Ok(DeviceVideoCaps {
        picture_access_granularity: capabilities.picture_access_granularity,
        min_coded_extent: capabilities.min_coded_extent,
        max_coded_extent: capabilities.max_coded_extent,
        max_dpb_slots: capabilities.max_dpb_slots,
        max_active_reference_pictures: capabilities.max_active_reference_pictures,
        flags: capabilities.flags,
        std_header_version: capabilities.std_header_version,
    })
}

/// Query capabilities, create the video session, DPB images, command resources,
/// the encode pipeline and the GOP structure — the ~85% of encoder
/// initialization that is identical across codecs.
pub(crate) fn build_encoder_common(req: &CommonInitRequest) -> Result<CommonInit> {
    let context = req.context;
    let config = req.config;
    let width = config.dimensions.width;
    let height = config.dimensions.height;

    let video_queue_fn = ash::khr::video_queue::Device::load(context.instance(), context.device());
    let video_encode_fn =
        ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

    // Capabilities were queried by the codec (which chains its codec-specific
    // capability struct, required by the driver) via [`query_video_caps`].
    let capabilities = req.caps;

    // Align the coded extent to lcm(codec block size, device granularity), then
    // clamp up to the device minimum and re-align.
    let gran_w = capabilities.picture_access_granularity.width.max(1);
    let gran_h = capabilities.picture_access_granularity.height.max(1);
    let align_w = lcm(req.align_unit, gran_w);
    let align_h = lcm(req.align_unit, gran_h);
    let aligned_width = align_up(
        align_up(width, align_w).max(capabilities.min_coded_extent.width),
        align_w,
    );
    let aligned_height = align_up(
        align_up(height, align_h).max(capabilities.min_coded_extent.height),
        align_h,
    );
    if aligned_width > capabilities.max_coded_extent.width
        || aligned_height > capabilities.max_coded_extent.height
    {
        return Err(PixelForgeError::InvalidInput(format!(
            "Requested coded extent {}x{} (aligned to {}x{} with granularity {}x{}) exceeds device max {}x{} for this profile",
            width,
            height,
            aligned_width,
            aligned_height,
            gran_w,
            gran_h,
            capabilities.max_coded_extent.width,
            capabilities.max_coded_extent.height
        )));
    }
    tracing::info!(
        "Using coded extent {}x{} (granularity {}x{}, min {}x{}, max {}x{})",
        aligned_width,
        aligned_height,
        gran_w,
        gran_h,
        capabilities.min_coded_extent.width,
        capabilities.min_coded_extent.height,
        capabilities.max_coded_extent.width,
        capabilities.max_coded_extent.height
    );

    // With RGB input the encoder converts, so the session needs its settings
    // and the input image is RGB; the DPB stays YUV either way.
    let rgb_session = match config.rgb_input {
        Some(_) if !context.has_video_encode_rgb_conversion() => {
            return Err(PixelForgeError::NoSuitableDevice(
                "RGB input needs VK_VALVE_video_encode_rgb_conversion, which this device \
                 does not have enabled"
                    .to_string(),
            ));
        }
        Some(_) => {
            let caps = req.rgb_caps.ok_or_else(|| {
                PixelForgeError::NoSuitableDevice(
                    "RGB input: the driver reported no conversion capabilities".to_string(),
                )
            })?;
            Some(
                caps.session_info(
                    &config
                        .color_description
                        .unwrap_or_else(ColorDescription::bt709),
                )?,
            )
        }
        None => None,
    };

    // Pick input (SRC) and reference (DPB) formats.
    let preferred_src_format = get_video_format(config.pixel_format, config.bit_depth);
    let supported_src_formats = query_supported_video_formats(
        context,
        req.profile_info,
        vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
    )?;
    let supported_dpb_formats = query_supported_video_formats(
        context,
        req.profile_info,
        vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
    )?;
    if supported_src_formats.is_empty() {
        return Err(PixelForgeError::NoSuitableDevice(
            "No supported Vulkan Video SRC formats for this profile".to_string(),
        ));
    }
    if supported_dpb_formats.is_empty() {
        return Err(PixelForgeError::NoSuitableDevice(
            "No supported Vulkan Video DPB formats for this profile".to_string(),
        ));
    }
    let picture_format = config
        .rgb_input
        .map_or(preferred_src_format, |format| format.vk_format());
    if !supported_src_formats
        .iter()
        .any(|f| f.format == picture_format)
    {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Input format {:?} is not supported for VIDEO_ENCODE_SRC_KHR. Supported: {:?}",
            picture_format, supported_src_formats
        )));
    }
    let reference_picture_format = supported_dpb_formats
        .iter()
        .map(|f| f.format)
        .find(|f| *f == preferred_src_format)
        .unwrap_or(supported_dpb_formats[0].format);

    // Negotiate DPB slots and active references.
    let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
    let max_active_supported = capabilities.max_active_reference_pictures as usize;
    if max_dpb_slots_supported < 2 {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Device reports max_dpb_slots={} for this profile; need at least 2",
            max_dpb_slots_supported
        )));
    }
    let mut target_active_refs = (config.max_reference_frames as usize)
        .min(max_active_supported)
        .min(req.max_active_refs_cap);
    // A device may allow fewer active references under intra refresh than
    // otherwise -- measured on RADV, one for H.264 and AV1 -- and exceeding it
    // is invalid usage. Clamped rather than treated as a conflict, because
    // refresh cannot use the extra ones regardless: inside a cycle, prediction
    // is restricted to a reference's already-refreshed regions, so only the
    // preceding picture is usefully referenceable. The references given up
    // here were going to contribute nothing and cost DPB memory for it.
    if config.intra_refresh_cycle.is_some()
        && context.has_video_encode_intra_refresh()
        && req.intra_refresh_caps.modes != vk::VideoEncodeIntraRefreshModeFlagsKHR::NONE
    {
        let allowed = req.intra_refresh_caps.max_active_reference_pictures as usize;
        if allowed >= 1 && allowed < target_active_refs {
            debug!(
                "intra refresh allows {allowed} active reference picture(s), not \
                 {target_active_refs}; using {allowed}"
            );
            target_active_refs = allowed;
        }
    }
    if target_active_refs < 1 && max_active_supported >= 1 {
        target_active_refs = 1;
    }
    // B-frames are not yet supported, so a reconstructed-frame slot plus the
    // active references is enough.
    let dpb_slot_count = (target_active_refs + 1)
        .min(max_dpb_slots_supported)
        .min(MAX_DPB_SLOTS);
    let max_active_reference_pictures = target_active_refs.min(dpb_slot_count.saturating_sub(1));

    let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
        PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
    })?;

    // Use the driver-reported std header version for this profile.
    let std_header_version = capabilities.std_header_version;
    // Resolved before the session is made, because the mode is session state:
    // a session not created for intra refresh cannot be told to do it later.
    let intra_refresh = resolve_intra_refresh(
        context,
        config,
        req.intra_refresh_caps,
        req.refresh_block,
        max_active_reference_pictures as u32,
    );
    let mut intra_refresh_create = intra_refresh.as_ref().map(|ir| {
        vk::VideoEncodeSessionIntraRefreshCreateInfoKHR::default().intra_refresh_mode(ir.mode)
    });

    let mut rgb_session_info = rgb_session.unwrap_or_default();
    let mut session_create_info = vk::VideoSessionCreateInfoKHR::default()
        .queue_family_index(encode_queue_family)
        .flags(vk::VideoSessionCreateFlagsKHR::empty())
        .video_profile(req.profile_info)
        .picture_format(picture_format)
        .max_coded_extent(vk::Extent2D {
            width: aligned_width,
            height: aligned_height,
        })
        .reference_picture_format(reference_picture_format)
        .max_dpb_slots(dpb_slot_count as u32)
        .max_active_reference_pictures(max_active_reference_pictures as u32)
        .std_header_version(&std_header_version);
    if rgb_session.is_some() {
        session_create_info = session_create_info.push(&mut rgb_session_info);
    }
    if let Some(ir) = intra_refresh_create.as_mut() {
        session_create_info = session_create_info.push(ir);
    }

    let mut session = vk::VideoSessionKHR::null();
    let result = unsafe {
        (video_queue_fn.fp().create_video_session_khr)(
            context.device().handle(),
            &session_create_info,
            std::ptr::null(),
            &mut session,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::VideoSessionCreation(format!(
            "{:?}",
            result
        )));
    }
    let session_memory = allocate_session_memory(context, session, &video_queue_fn)?;

    // Use a layered DPB only when allowed and the driver lacks separate
    // reference images (AMD RADV).
    let supports_separate_dpb = capabilities
        .flags
        .contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);
    let use_layered_dpb = req.allow_layered_dpb && !supports_separate_dpb;
    if use_layered_dpb {
        tracing::info!("Using layered DPB (driver does not support separate reference images)");
    }

    let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
        context,
        aligned_width,
        aligned_height,
        reference_picture_format,
        dpb_slot_count,
        req.profile_info,
        use_layered_dpb,
    )?;

    let upload_queue_family = context.transfer_queue_family();
    let cmd = create_command_resources(context, encode_queue_family, upload_queue_family)?;

    let pipeline = EncodePipeline::new(&PipelineConfig {
        context,
        aligned_width,
        aligned_height,
        picture_format,
        rgb_input: config.rgb_input.is_some(),
        pixel_format: config.pixel_format,
        bit_depth: config.bit_depth,
        bitstream_buffer_size: req.bitstream_buffer_size,
        profile_info: req.profile_info,
        command_pool: cmd.command_pool,
        upload_command_buffer: cmd.upload_command_buffer,
        upload_fence: cmd.upload_fence,
    })?;

    // B-frames are not yet supported, so an I-P GOP. Set the SPS-matching
    // counters (no-ops for AV1, which keys off order hints).
    let mut gop = GopStructure::new_ip_only(config.gop_size);
    if intra_refresh.is_some() {
        // The refresh cycle *is* the recovery point, so a periodic key frame
        // on top of it is the cost intra refresh exists to avoid, paid twice:
        // once as the burst and again as the rate-control excursion that burst
        // forces. The first picture is still an IDR -- a stream has to start
        // somewhere -- and an explicitly requested one still works.
        gop.set_gop_size(None);
    }
    gop.set_max_frame_num(4);
    gop.set_max_poc_lsb(4);

    let common = EncoderCommon {
        intra_refresh,
        context: context.clone(),
        config: config.clone(),
        video_queue_fn,
        video_encode_fn,
        session,
        session_params: vk::VideoSessionParametersKHR::null(),
        session_memory,
        aligned_width,
        aligned_height,
        pipeline,
        gop,
        input_frame_num: 0,
        encode_frame_num: 0,
        rate_control_dirty: false,
        dpb_images,
        dpb_image_memories,
        dpb_image_views,
        dpb_slot_count,
        dpb_slot_active: vec![false; dpb_slot_count],
        use_layered_dpb,
        current_dpb_slot: 0,
        command_pool: cmd.command_pool,
        upload_command_pool: cmd.upload_command_pool,
        upload_command_buffer: cmd.upload_command_buffer,
        upload_fence: cmd.upload_fence,
        pending_waits: Vec::new(),
    };

    Ok(CommonInit {
        common,
        active_reference_count: max_active_reference_pictures as u32,
    })
}

#[cfg(test)]
mod advisory_gop_tests {
    use super::*;

    #[test]
    fn no_periodic_keyframes_is_reported_as_infinite() {
        // UINT32_MAX is Vulkan's own word for infinite here. Zero is not a
        // synonym for it -- zero lets the implementation pick a period, which is
        // the opposite instruction.
        assert_eq!(advisory_gop_length(0), u32::MAX);
    }

    #[test]
    fn a_real_period_is_reported_as_itself() {
        assert_eq!(advisory_gop_length(1), 1);
        assert_eq!(advisory_gop_length(240), 240);
    }
}

#[cfg(test)]
mod rate_control_support_tests {
    use super::*;

    const NONE: vk::VideoEncodeRateControlModeFlagsKHR =
        vk::VideoEncodeRateControlModeFlagsKHR::DISABLED;

    #[test]
    fn a_device_offering_only_disabled_cannot_do_bitrates() {
        // Intel's ANV reports exactly this, and every CBR encode on it silently
        // became constant-QP.
        assert!(!rate_control_is_supported(RateControlMode::Cbr, NONE));
        assert!(!rate_control_is_supported(RateControlMode::Vbr, NONE));
    }

    #[test]
    fn constant_qp_always_works() {
        // DISABLED is mandatory, so these must never warn -- a spurious warning
        // on every CQP encode would teach people to ignore the real one.
        assert!(rate_control_is_supported(RateControlMode::Cqp, NONE));
        assert!(rate_control_is_supported(RateControlMode::Disabled, NONE));
    }

    #[test]
    fn a_device_offering_the_mode_is_accepted() {
        let both = vk::VideoEncodeRateControlModeFlagsKHR::CBR
            | vk::VideoEncodeRateControlModeFlagsKHR::VBR;
        assert!(rate_control_is_supported(RateControlMode::Cbr, both));
        assert!(rate_control_is_supported(RateControlMode::Vbr, both));
    }

    #[test]
    fn offering_one_bitrate_mode_does_not_imply_the_other() {
        let cbr_only = vk::VideoEncodeRateControlModeFlagsKHR::CBR;
        assert!(rate_control_is_supported(RateControlMode::Cbr, cbr_only));
        assert!(!rate_control_is_supported(RateControlMode::Vbr, cbr_only));
    }
}

#[cfg(test)]
mod intra_refresh_tests {
    use super::*;

    fn state(cycle: u32, restrict: bool) -> IntraRefreshState {
        IntraRefreshState {
            cycle_duration: cycle,
            index: 0,
            mode: vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_ROW_BASED,
            restrict_prediction: restrict,
        }
    }

    /// VUID-vkCmdEncodeVideoKHR-pNext-10843: a non-zero count must equal the
    /// cycle duration minus *the encoded picture's* refresh index. A value
    /// derived from the reference's own history instead agrees only while the
    /// reference is the immediately preceding picture, and is invalid usage
    /// otherwise -- which shows up as a corrupt picture, not an error.
    #[test]
    fn the_dirty_count_is_the_one_the_spec_requires() {
        let mut ir = state(30, true);
        for index in 0..30 {
            ir.index = index;
            assert_eq!(
                ir.dirty_regions(),
                30 - index,
                "cycle duration minus index, at index {index}"
            );
        }
    }

    /// Declining the restriction is legal and is how it is declined: zero
    /// means no limit, and the structure carrying zero imposes nothing.
    #[test]
    fn declining_the_restriction_declares_nothing_dirty() {
        let mut ir = state(30, false);
        for index in 0..30 {
            ir.index = index;
            assert_eq!(ir.dirty_regions(), 0);
        }
    }

    /// VUID-vkCmdEncodeVideoKHR-pEncodeInfo-10841: the index must be less than
    /// the cycle duration, always.
    #[test]
    fn the_index_stays_inside_the_cycle() {
        let mut ir = state(3, true);
        for _ in 0..7 {
            ir.advance();
            assert!(ir.index < 3, "index {} left the cycle", ir.index);
        }
    }

    /// AV1 needs to know when a cycle opens, to reset the entropy model.
    #[test]
    fn the_first_picture_of_a_cycle_is_recognisable() {
        let mut ir = state(3, true);
        assert!(ir.starts_cycle());
        ir.advance();
        assert!(!ir.starts_cycle());
        ir.advance();
        assert!(!ir.starts_cycle());
        ir.advance();
        assert!(ir.starts_cycle(), "wrapping round begins a new cycle");
    }

    /// A key frame refreshed everything at once, so the cycle it interrupted
    /// is finished rather than paused.
    #[test]
    fn a_key_frame_restarts_the_cycle() {
        let mut ir = state(4, true);
        ir.advance();
        ir.advance();
        assert_eq!(ir.index, 2);
        ir.restart();
        assert_eq!(ir.index, 0);
    }
}

#[cfg(test)]
mod refresh_block_tests {
    use super::*;

    /// The coarsest division the device offers is the one that gives the
    /// fewest regions, and the fewest regions is the bound that holds however
    /// the encoder ends up dividing the picture.
    #[test]
    fn the_largest_offered_ctb_is_taken() {
        use vk::VideoEncodeH265CtbSizeFlagsKHR as Ctb;
        assert_eq!(h265_refresh_block(Ctb::TYPE_64), 64);
        assert_eq!(
            h265_refresh_block(Ctb::TYPE_16 | Ctb::TYPE_32 | Ctb::TYPE_64),
            64
        );
        assert_eq!(h265_refresh_block(Ctb::TYPE_16 | Ctb::TYPE_32), 32);
        assert_eq!(h265_refresh_block(Ctb::TYPE_16), 16);
    }

    /// A device reporting no CTB size at all has told us nothing, and a guess
    /// that is too small under-bounds the cycle -- which is the failure this
    /// whole path exists to prevent. The largest legal CTB is the safe guess.
    #[test]
    fn an_empty_ctb_report_falls_back_to_the_largest() {
        assert_eq!(
            h265_refresh_block(vk::VideoEncodeH265CtbSizeFlagsKHR::empty()),
            64
        );
    }

    #[test]
    fn the_largest_offered_superblock_is_taken() {
        use vk::VideoEncodeAV1SuperblockSizeFlagsKHR as Sb;
        assert_eq!(av1_refresh_block(Sb::TYPE_64), 64);
        assert_eq!(av1_refresh_block(Sb::TYPE_64 | Sb::TYPE_128), 128);
        assert_eq!(
            av1_refresh_block(vk::VideoEncodeAV1SuperblockSizeFlagsKHR::empty()),
            128
        );
    }
}

#[cfg(test)]
mod intra_refresh_cycle_tests {
    use super::*;

    const ROWS: vk::VideoEncodeIntraRefreshModeFlagsKHR =
        vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_ROW_BASED;
    const COLUMNS: vk::VideoEncodeIntraRefreshModeFlagsKHR =
        vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_COLUMN_BASED;
    const BLOCKS: vk::VideoEncodeIntraRefreshModeFlagsKHR =
        vk::VideoEncodeIntraRefreshModeFlagsKHR::BLOCK_BASED;
    const PARTITIONS: vk::VideoEncodeIntraRefreshModeFlagsKHR =
        vk::VideoEncodeIntraRefreshModeFlagsKHR::PER_PICTURE_PARTITION;

    /// 1080p in 64x64 blocks is 30 across and 17 down, so a row sweep has 17
    /// regions to give out and a column sweep 30. A cycle longer than that
    /// asks for regions the picture does not have.
    #[test]
    fn a_sweep_is_bounded_by_the_blocks_in_its_direction() {
        assert_eq!(refresh_region_limit(ROWS, 1920, 1080, 64), Some(17));
        assert_eq!(refresh_region_limit(COLUMNS, 1920, 1080, 64), Some(30));
    }

    /// H.264's 16x16 macroblocks divide the same picture far more finely, which
    /// is why the same cycle can be legal for one codec and not another on one
    /// device.
    #[test]
    fn a_smaller_block_gives_a_longer_usable_sweep() {
        assert_eq!(refresh_region_limit(ROWS, 1920, 1080, 16), Some(68));
        assert_eq!(refresh_region_limit(COLUMNS, 1920, 1080, 16), Some(120));
    }

    /// Rounded up. 1080 is not a whole number of 64-block rows, and the
    /// remainder is still a row that has to be refreshed -- rounding down
    /// would leave a strip that the sweep never reaches.
    #[test]
    fn a_partial_block_row_still_counts_as_a_row() {
        assert_eq!(refresh_region_limit(ROWS, 1920, 1080, 64), Some(17));
        assert_eq!(refresh_region_limit(ROWS, 1920, 1088, 64), Some(17));
        assert_eq!(refresh_region_limit(ROWS, 1920, 1089, 64), Some(18));
    }

    /// Block-based leaves the division to the implementation, so which way it
    /// sweeps is not knowable here. The smaller of the two bounds is the only
    /// one safe against both.
    #[test]
    fn block_based_takes_the_tighter_of_the_two_directions() {
        assert_eq!(refresh_region_limit(BLOCKS, 1920, 1080, 64), Some(17));
        // A tall picture reverses which direction is the tight one.
        assert_eq!(refresh_region_limit(BLOCKS, 1080, 1920, 64), Some(17));
    }

    /// Per-picture partition ties regions to the slice or tile layout, which
    /// is a different bound entirely and not one this controls. No limit is
    /// claimed rather than a wrong one imposed.
    #[test]
    fn a_partition_sweep_has_no_bound_this_can_compute() {
        assert_eq!(refresh_region_limit(PARTITIONS, 1920, 1080, 64), None);
    }

    /// A block size of zero cannot come from a real device, but dividing by it
    /// would panic, and a capability query that returned nothing is exactly
    /// when that would happen.
    #[test]
    fn a_zero_block_size_does_not_divide_by_zero() {
        assert_eq!(refresh_region_limit(ROWS, 1920, 1080, 0), Some(1080));
    }

    /// A picture smaller than one block has one region, which is not a sweep.
    /// The caller refuses at that point; the limit just has to be honest.
    #[test]
    fn a_picture_under_one_block_has_a_single_region() {
        assert_eq!(refresh_region_limit(ROWS, 32, 32, 64), Some(1));
    }
}
