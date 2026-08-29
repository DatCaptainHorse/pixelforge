//! The codec-generic decoder.
//!
//! Everything that is identical across H.264, H.265 and AV1 lives here: the
//! shared per-decoder state ([`DecoderCommon`]), the Vulkan video session and
//! its DPB images ([`DecodeSession`]), coded-data staging, and the readback
//! path that copies a decoded picture back to the host.
//!
//! A codec supplies only its differences: parsing its bitstream into pictures,
//! reconstructing the reference state the encoder implied, and building the
//! codec-specific `StdVideo*` graph for a picture. Reading a codec's folder
//! shows those differences; the scaffolding is here.
//!
//! There is deliberately no `VideoDecodeCodec` trait yet. The per-picture hook
//! shape is only knowable once a second codec exists to constrain it; H.265 is
//! the forcing function. Until then a codec owns a `DecoderCommon` directly.
//!
//! This mirrors the encoder's [`crate::encoder::codec`] split deliberately, so
//! the two directions read the same way.

use std::sync::Arc;

use ash::vk;

use crate::decoder::DecodedFrame;
use crate::decoder::frames::SlotPins;
use crate::decoder::pipeline::{DecodeFuture, DecodePipeline};
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::{
    VideoImageParams, allocate_command_buffers, allocate_session_memory, create_command_pool,
    create_dpb_images, create_fence, create_video_image,
};
use crate::vulkan::VideoContext;

/// Annex B start code prefixed to each slice handed to the driver.
///
/// Three bytes, not four. The leading zero byte of a 4-byte start code is legal
/// Annex B, and RADV decodes either form, but Intel's ANV decodes almost nothing
/// when a slice offset points at that extra zero: it reports no error and
/// returns a picture with a handful of macroblocks in it. ffmpeg's Vulkan
/// decoder uses three bytes for the same reason.
pub(crate) const START_CODE: [u8; 3] = [0, 0, 1];

/// Query decode capabilities for a profile.
///
/// `caps` is supplied by the caller already chained with the codec-specific
/// capability struct, since only the codec knows which one applies.
pub(crate) fn query_decode_caps(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    caps: &mut vk::VideoCapabilitiesKHR,
) -> Result<()> {
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let result = unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            caps,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::CodecNotSupported(format!(
            "decode profile not supported: {:?}",
            result
        )));
    }
    Ok(())
}

/// What a codec decided the session should look like.
///
/// The codec derives these from its own parameter sets and the device caps; the
/// generic layer turns them into Vulkan objects.
pub(crate) struct SessionPlan {
    /// Coded dimensions, already aligned to the device's picture granularity.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub pixel_format: PixelFormat,
    /// DPB slots: references, the picture being decoded, and `spare_slots`.
    pub slot_count: usize,
    /// How many of `slot_count` are spare capacity for pictures pinned past the
    /// decode that produced them: those held for display-order reordering, and
    /// those handed to the caller. Zero means every emitted picture has to be
    /// copied out instead of pinned in place.
    pub spare_slots: usize,
    /// References the driver may use for a single picture, which is
    /// `slot_count` minus the current picture and the output reservation.
    pub max_active_references: u32,
    /// Driver decodes straight into the DPB image (`DPB_AND_OUTPUT_COINCIDE`).
    pub coincide: bool,
    /// `dpb_usage` includes `SAMPLED`, so handed-out frames can be sampled.
    pub sampleable: bool,
    /// One layered DPB image rather than one image per slot.
    pub use_layered_dpb: bool,
    pub dpb_usage: vk::ImageUsageFlags,
}

/// The Vulkan video session and the images it decodes into.
///
/// Recreated whenever the stream's parameter sets change in a way that matters
/// (a new resolution, most often).
pub(crate) struct DecodeSession {
    pub session: vk::VideoSessionKHR,
    pub session_memory: Vec<vk::DeviceMemory>,
    pub session_params: vk::VideoSessionParametersKHR,

    /// Coded dimensions the session was created for.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub pixel_format: PixelFormat,

    /// DPB images. One image per slot, or a single layered image.
    pub dpb_images: Vec<vk::Image>,
    pub dpb_memories: Vec<vk::DeviceMemory>,
    /// One view per slot (indexes slots even when layered).
    pub dpb_views: Vec<vk::ImageView>,
    /// Whether each slot has been written at least once (governs the
    /// UNDEFINED-vs-DPB old layout in the pre-decode barrier).
    pub dpb_slot_active: Vec<bool>,
    pub use_layered_dpb: bool,

    /// True when the driver decodes into the DPB image directly; false when a
    /// distinct output image is required.
    pub coincide: bool,
    /// Spare DPB slots available for pinning pictures past their decode.
    pub spare_slots: usize,
    /// Whether decoded pictures can be sampled without being copied first.
    pub sampleable: bool,
    /// Distinct decode output image, only when `!coincide`.
    pub output_image: Option<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
}
impl DecodeSession {
    /// The image and array layer holding DPB slot `slot`.
    /// For layered DPBs all slots share `dpb_images[0]` and are distinguished
    /// by layer; for separate images each slot has its own image at layer 0.
    pub fn dpb_image_for_slot(&self, slot: u8) -> (vk::Image, u32) {
        if self.use_layered_dpb {
            (self.dpb_images[0], slot as u32)
        } else {
            (self.dpb_images[slot as usize], 0)
        }
    }
}

/// Per-decoder state shared by every codec.
///
/// Holds the Vulkan video session, the DPB images, the coded-data staging
/// buffer and the readback path — none of which differ between codecs.
/// Codec-specific state (parameter sets, POC, reference marking) lives in the
/// codec instead.
pub(crate) struct DecoderCommon {
    pub context: VideoContext,
    pub video_queue_fn: ash::khr::video_queue::Device,
    pub video_decode_fn: ash::khr::video_decode_queue::Device,
    pub decode_queue: vk::Queue,
    pub decode_queue_family: u32,

    pub command_pool: vk::CommandPool,

    /// Readback resources. The video decode queue family is a dedicated engine
    /// that does not advertise `TRANSFER_BIT` (true on RADV, among others), so
    /// copies must be recorded and submitted on the transfer queue instead.
    pub transfer_queue: vk::Queue,
    pub transfer_pool: vk::CommandPool,
    /// Command buffer for the synchronous, caller-driven copies (`download`,
    /// `copy_frame_to_planes`). Pipelined reorder copies use the slot's own.
    pub transfer_command_buffer: vk::CommandBuffer,
    pub transfer_fence: vk::Fence,

    /// In-flight decode submissions, their per-picture resources and the
    /// completion thread that resolves their futures.
    pub pipeline: DecodePipeline,

    /// Required alignment of bitstream buffer offsets/ranges.
    pub bitstream_offset_alignment: u64,
    pub bitstream_size_alignment: u64,

    /// Readback buffer for `download`, allocated on first use.
    pub readback: Option<(vk::Buffer, vk::DeviceMemory, usize)>,

    /// The active session, created lazily from the stream's parameter sets.
    pub session: Option<DecodeSession>,

    /// DPB slots reserved by frames the caller still holds. Shared with those
    /// frames, which release their slot when dropped.
    pub slot_pins: Arc<SlotPins>,

    /// Queue family that reads decoded frames, when it is neither the decode
    /// nor the transfer family. Added to every picture image's sharing set so
    /// a consumer can use frames without a queue family ownership transfer.
    pub consumer_queue_family: Option<u32>,
}

// `bitstream_ptr` is a persistently mapped host-visible allocation owned by
// this struct; it is never aliased or shared across threads.
unsafe impl Send for DecoderCommon {}

impl DecoderCommon {
    pub fn new(context: VideoContext, consumer_queue_family: Option<u32>) -> Result<Self> {
        let decode_queue_family = context.video_decode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("Device has no video decode queue family".to_string())
        })?;
        let decode_queue = context.video_decode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("Device has no video decode queue".to_string())
        })?;

        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_decode_fn =
            ash::khr::video_decode_queue::Device::load(context.instance(), context.device());

        let command_pool = create_command_pool(&context, decode_queue_family, "decode")?;
        let (transfer_pool, transfer_command_buffer, transfer_fence) =
            create_command_resources(&context, context.transfer_queue_family(), "decode transfer")?;
        let pipeline = DecodePipeline::new(&context, command_pool, transfer_pool)?;

        Ok(Self {
            transfer_queue: context.transfer_queue(),
            context,
            video_queue_fn,
            video_decode_fn,
            decode_queue,
            decode_queue_family,
            command_pool,
            transfer_pool,
            transfer_command_buffer,
            transfer_fence,
            pipeline,
            bitstream_offset_alignment: 1,
            bitstream_size_alignment: 1,
            readback: None,
            session: None,
            slot_pins: Arc::new(SlotPins::default()),
            consumer_queue_family,
        })
    }

    /// Every queue family that may touch a decoded picture.
    ///
    /// The decode family writes it, the transfer family copies from it, and a
    /// consumer named by
    /// [`DecodeConfig::with_consumer_queue_family`](crate::decoder::DecodeConfig::with_consumer_queue_family)
    /// reads it. `create_video_image` reduces this to `EXCLUSIVE` when the
    /// families turn out to be the same one.
    pub fn picture_sharing_families(&self) -> Vec<u32> {
        let mut families = vec![
            self.decode_queue_family,
            self.context.transfer_queue_family(),
        ];
        if let Some(consumer) = self.consumer_queue_family {
            families.push(consumer);
        }
        families
    }

    /// The active session, or an error if no parameter sets have been seen yet.
    pub fn session(&self) -> Result<&DecodeSession> {
        self.session
            .as_ref()
            .ok_or_else(|| PixelForgeError::InvalidInput("decode: no active session".to_string()))
    }

    /// Create the session, its DPB images and (when needed) a distinct output
    /// image, replacing any existing one.
    ///
    /// `make_params` builds the codec's session parameters once the session
    /// exists, since parameter sets are entirely codec-specific.
    pub fn create_session(
        &mut self,
        plan: &SessionPlan,
        profile_info: &vk::VideoProfileInfoKHR,
        std_header_version: &vk::ExtensionProperties,
        make_params: impl FnOnce(&Self, vk::VideoSessionKHR) -> Result<vk::VideoSessionParametersKHR>,
    ) -> Result<()> {
        self.destroy_session();

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.decode_queue_family)
            .flags(vk::VideoSessionCreateFlagsKHR::empty())
            .video_profile(profile_info)
            .picture_format(plan.picture_format)
            .max_coded_extent(vk::Extent2D {
                width: plan.coded_width,
                height: plan.coded_height,
            })
            .reference_picture_format(plan.picture_format)
            .max_dpb_slots(plan.slot_count as u32)
            .max_active_reference_pictures(plan.max_active_references)
            .std_header_version(std_header_version);

        let session = unsafe {
            self.video_queue_fn
                .create_video_session(&session_create_info, None)
        }
        .map_err(|e| PixelForgeError::VideoSessionCreation(format!("{:?}", e)))?;

        let session_memory = allocate_session_memory(&self.context, session, &self.video_queue_fn)?;

        // When the DPB doubles as the decode output it may also be copied from,
        // so the transfer queue needs access too, and a consumer reading frames
        // from a third family has to be named or its reads are undefined.
        let families = self.picture_sharing_families();

        let dpb_params = VideoImageParams {
            width: plan.coded_width,
            height: plan.coded_height,
            format: plan.picture_format,
            usage: plan.dpb_usage,
            sharing_families: &families,
        };

        let (dpb_images, dpb_memories, dpb_views) = create_dpb_images(
            &self.context,
            &dpb_params,
            profile_info,
            plan.slot_count,
            plan.use_layered_dpb,
        )?;

        let output_image = if plan.coincide {
            None
        } else {
            let output_params = VideoImageParams {
                usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                ..dpb_params
            };
            Some(create_video_image(
                &self.context,
                &output_params,
                profile_info,
            )?)
        };

        let session_params = make_params(self, session)?;

        self.session = Some(DecodeSession {
            session,
            session_memory,
            session_params,
            coded_width: plan.coded_width,
            coded_height: plan.coded_height,
            picture_format: plan.picture_format,
            bit_depth: plan.bit_depth,
            pixel_format: plan.pixel_format,
            dpb_images,
            dpb_memories,
            dpb_views,
            dpb_slot_active: vec![false; plan.slot_count],
            use_layered_dpb: plan.use_layered_dpb,
            coincide: plan.coincide,
            spare_slots: plan.spare_slots,
            sampleable: plan.sampleable,
            output_image,
        });
        Ok(())
    }

    pub fn destroy_session(&mut self) {
        // Let every in-flight decode finish and its future resolve before the
        // session, DPB images and slots they refer to stop existing.
        self.pipeline.wait_all_free();
        let Some(session) = self.session.take() else {
            return;
        };
        // The slots these pins refer to are about to stop existing.
        if self.slot_pins.any_pinned() {
            tracing::warn!(
                "decode session torn down while decoded frames still hold DPB slots; \
                 those frames' images are now invalid"
            );
        }
        self.slot_pins.clear();
        unsafe {
            self.context.device().device_wait_idle().ok();
            self.video_queue_fn
                .destroy_video_session_parameters(session.session_params, None);
            for &view in &session.dpb_views {
                self.context.device().destroy_image_view(view, None);
            }
            for &image in &session.dpb_images {
                self.context.device().destroy_image(image, None);
            }
            for &memory in &session.dpb_memories {
                self.context.device().free_memory(memory, None);
            }
            if let Some((image, memory, view)) = session.output_image {
                self.context.device().destroy_image_view(view, None);
                self.context.device().destroy_image(image, None);
                self.context.device().free_memory(memory, None);
            }
            self.video_queue_fn
                .destroy_video_session(session.session, None);
            for &memory in &session.session_memory {
                self.context.device().free_memory(memory, None);
            }
        }
    }

    /// Copy a picture's slices into the staging buffer, each prefixed with a
    /// start code, and report the buffer range plus each slice's offset.
    pub fn stage_slices(
        &mut self,
        slices: &[&[u8]],
        profile_info: &vk::VideoProfileInfoKHR,
    ) -> Result<(u64, Vec<u32>)> {
        let total: usize = slices.iter().map(|s| s.len() + START_CODE.len()).sum();
        let aligned_total =
            crate::video::align_up(total as u32, self.bitstream_size_alignment as u32) as usize;
        self.pipeline
            .ensure_bitstream_capacity(&self.context, aligned_total, profile_info)?;
        let dst = self.pipeline.current().bitstream_ptr;

        let mut offsets = Vec::with_capacity(slices.len());
        let mut cursor = 0usize;
        for slice in slices {
            offsets.push(cursor as u32);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    START_CODE.as_ptr(),
                    dst.add(cursor),
                    START_CODE.len(),
                );
                cursor += START_CODE.len();
                std::ptr::copy_nonoverlapping(slice.as_ptr(), dst.add(cursor), slice.len());
                cursor += slice.len();
            }
        }
        // Zero the alignment padding so the driver never reads stale bytes.
        if aligned_total > cursor {
            unsafe {
                std::ptr::write_bytes(dst.add(cursor), 0, aligned_total - cursor);
            }
        }
        Ok((aligned_total as u64, offsets))
    }

    /// Transition the DPB slot this decode writes into, and the output image
    /// (if distinct), into the layouts `vkCmdDecodeVideo` requires.
    pub fn record_barriers(&self, slot: u8) {
        let session = self.session.as_ref().expect("active session");
        let device = self.context.device();
        let mut barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        let subresource = |layer: u32| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        };

        // The slot being written: UNDEFINED on first use, DPB afterwards.
        let (dst_image, dst_layer) = session.dpb_image_for_slot(slot);
        let old_layout = if session.dpb_slot_active[slot as usize] {
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        barriers.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image)
                .subresource_range(subresource(dst_layer)),
        );

        // A distinct output image must be in decode-DST layout.
        if let Some((image, _, _)) = session.output_image {
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::VIDEO_DECODE_DST_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(subresource(0)),
            );
        }

        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(self.decode_command_buffer(), &dependency) };
    }

    /// The decode command buffer of the picture currently being recorded.
    pub fn decode_command_buffer(&self) -> vk::CommandBuffer {
        self.pipeline.current().decode_command_buffer
    }

    /// The coded-data staging buffer of the picture currently being recorded.
    pub fn bitstream_buffer(&self) -> vk::Buffer {
        self.pipeline.current().bitstream_buffer
    }

    /// Wait until the next picture's slot is free, then begin recording into it.
    ///
    /// Waiting here is what makes the staging buffer and command buffers safe to
    /// overwrite: the slot is busy until its previous submission has completed.
    pub fn begin_decode_commands(&self) -> Result<()> {
        self.pipeline.wait_current_free();
        let device = self.context.device();
        let command_buffer = self.decode_command_buffer();
        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }
        Ok(())
    }

    /// Submit the recorded decode without waiting for it.
    ///
    /// The submission is chained onto the decode timeline, so the GPU still runs
    /// decodes in decode order despite the CPU running ahead.
    pub fn submit_decode(&mut self) -> Result<()> {
        let device = self.context.device().clone();
        let queue = self.decode_queue;
        self.pipeline.submit_decode(&device, queue)
    }

    /// Close off the current picture and hand `frames` to the completion thread.
    pub fn end_picture(&mut self, frames: Vec<DecodedFrame>) {
        self.pipeline.end_picture(frames);
    }

    /// Close off a `decode`/`flush` call, returning the future for its frames.
    pub fn finish_batch(&mut self, frames: Vec<DecodedFrame>) -> DecodeFuture {
        self.pipeline.finish_batch(frames)
    }
}

impl Drop for DecoderCommon {
    fn drop(&mut self) {
        // `destroy_session` drains the pipeline first, so by the time the
        // staging buffers and fences below are freed nothing is in flight.
        self.destroy_session();
        unsafe {
            self.context.device().device_wait_idle().ok();
            if let Some((buffer, memory, _)) = self.readback.take() {
                self.context.device().destroy_buffer(buffer, None);
                self.context.device().free_memory(memory, None);
            }
            let device = self.context.device().clone();
            self.pipeline.destroy(&device);
            self.context
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.context
                .device()
                .destroy_fence(self.transfer_fence, None);
            self.context
                .device()
                .destroy_command_pool(self.transfer_pool, None);
        }
    }
}

/// A command pool, one primary command buffer, and a fence for `family`.
///
/// The pipelined paths allocate their own per-slot buffers from the pool; this
/// covers the single-buffer synchronous paths (`download` and friends).
fn create_command_resources(
    context: &VideoContext,
    family: u32,
    label: &str,
) -> Result<(vk::CommandPool, vk::CommandBuffer, vk::Fence)> {
    let pool = create_command_pool(context, family, label)?;
    let buffer = allocate_command_buffers(context, pool, 1)?[0];
    let fence = create_fence(context, false)?;
    Ok((pool, buffer, fence))
}
