//! The H.264 Vulkan decode session: lazy stream-driven setup, per-picture
//! record/submit, and CPU readback.

use std::collections::HashMap;

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use crate::decoder::codec::{
    DecodedPicture, DecoderCommon, ReorderBuffer, SessionPlan, query_decode_caps,
};
use crate::decoder::h264::dpb::{DecodeDpb, PictureState};
use crate::decoder::h264::parser::{
    self, NalType, NalUnit, Pps, SliceHeader, SliceType, Sps, iter_nal_units,
};
use crate::decoder::pipeline::DecodeFuture;
use crate::decoder::{DecodeConfig, DecodedFrame, DecodedFrameData, OutputOrder};
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::{align_up, get_video_format, query_supported_video_formats};
use crate::vulkan::VideoContext;

/// H.264 decoder.
///
/// Owns only what is specific to H.264: the parameter sets seen so far, and the
/// POC / reference-marking state. Everything Vulkan lives in [`DecoderCommon`].
pub(crate) struct H264Decoder {
    common: DecoderCommon,

    /// How many DPB slots to reserve beyond the stream's needs, so the caller
    /// can hold decoded frames while decoding continues.
    output_depth: usize,

    /// Parameter sets seen so far, by id.
    sps_map: HashMap<u8, Sps>,
    pps_map: HashMap<u8, Pps>,

    /// The SPS/PPS the current session parameters were built from, so a change
    /// that matters can be detected.
    active_sps_id: u8,
    active_pps_id: u8,

    /// POC and reference state, recreated alongside the session.
    dpb: Option<DecodeDpb>,

    /// Display-order reordering (a pass-through in decode-order mode).
    reorder: ReorderBuffer,
    /// `max_num_reorder_frames` for the active SPS: how deep the reorder buffer
    /// may hold pictures before emitting.
    reorder_depth: usize,

    /// Whether decoding is waiting for a keyframe (IDR) before it can produce
    /// output. True until the first IDR is decoded: non-IDR pictures reference
    /// state that does not exist yet, so they are skipped and the caller is told
    /// to request an IDR. This is the normal state when joining a stream.
    awaiting_keyframe: bool,
}

/// The slices making up one picture, gathered from the input.
struct Picture<'a> {
    header: SliceHeader,
    nal_type: NalType,
    ref_idc: u8,
    is_intra: bool,
    slices: Vec<&'a [u8]>,
}

impl H264Decoder {
    pub(crate) fn create(context: VideoContext, config: &DecodeConfig) -> Result<Self> {
        Ok(Self {
            common: DecoderCommon::new(context)?,
            output_depth: config.output_depth,
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            active_sps_id: 0,
            active_pps_id: 0,
            dpb: None,
            reorder: ReorderBuffer::new(config.output_order == OutputOrder::Display),
            reorder_depth: 0,
            awaiting_keyframe: true,
        })
    }

    /// Copy a decoded picture back to the host. Entirely generic.
    pub(crate) fn download(&mut self, frame: &DecodedFrame) -> Result<DecodedFrameData> {
        self.common.download(frame)
    }

    /// Copy a decoded picture's planes into two caller-owned images. Generic.
    pub(crate) fn copy_frame_to_planes(
        &mut self,
        frame: &DecodedFrame,
        y_image: vk::Image,
        uv_image: vk::Image,
    ) -> Result<()> {
        self.common.copy_frame_to_planes(frame, y_image, uv_image)
    }

    /// Emit any pictures still held for reordering. Call at end of stream.
    ///
    /// These pictures were copied out by earlier calls, so this submits no new
    /// GPU work; the future still resolves in call order behind whatever is
    /// already in flight.
    pub(crate) fn flush(&mut self) -> Result<DecodeFuture> {
        let frames = self.reorder.flush();
        Ok(self.common.finish_batch(frames))
    }

    pub(crate) fn picture_format(&self) -> Option<vk::Format> {
        self.common.session.as_ref().map(|a| a.picture_format)
    }

    /// Build the H.264 decode profile info for a parsed SPS.
    fn profile_for(
        sps: &Sps,
    ) -> Result<(
        vk::VideoDecodeH264ProfileInfoKHR<'static>,
        PixelFormat,
        BitDepth,
    )> {
        let pixel_format = match sps.chroma_format_idc {
            1 => PixelFormat::Yuv420,
            3 => PixelFormat::Yuv444,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported chroma_format_idc {}",
                    other
                )));
            }
        };
        let bit_depth = match sps.bit_depth_luma_minus8 {
            0 => BitDepth::Eight,
            2 => BitDepth::Ten,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported luma bit depth {}",
                    other + 8
                )));
            }
        };
        if sps.seq_scaling_matrix_present_flag {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: streams with explicit scaling matrices are not supported"
                    .to_string(),
            ));
        }

        let std_profile = match sps.profile_idc {
            66 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
            77 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
            100 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
            244 => {
                ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
            }
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported profile_idc {}",
                    other
                )));
            }
        };

        let profile = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(std_profile)
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);

        Ok((profile, pixel_format, bit_depth))
    }

    /// Create (or recreate) the Vulkan session for the given parameter sets.
    fn activate(&mut self, sps: &Sps, pps: &Pps) -> Result<()> {
        // Reuse the session when nothing relevant changed.
        if let Some(active) = &self.common.session
            && self.active_sps_id == sps.sps_id
            && self.active_pps_id == pps.pps_id
            && active.coded_width == sps.coded_width()
            && active.coded_height == sps.coded_height()
        {
            return Ok(());
        }

        // `profile_for` rejects scaling matrices in the SPS; a PPS can carry
        // its own, and those are just as unsupported. Without this the session
        // would report pic_scaling_matrix_present_flag = 0 with null scaling
        // lists, and the driver would silently dequantise with a flat matrix.
        if pps.pic_scaling_matrix_present_flag {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: streams with explicit scaling matrices are not supported"
                    .to_string(),
            ));
        }

        let (mut profile, pixel_format, bit_depth) = Self::profile_for(sps)?;
        let chroma_subsampling = match pixel_format {
            PixelFormat::Yuv420 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            PixelFormat::Yuv444 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported pixel format {:?}",
                    other
                )));
            }
        };
        let depth_flag = match bit_depth {
            BitDepth::Eight => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            BitDepth::Ten => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        };
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(depth_flag)
            .chroma_bit_depth(depth_flag)
            .push(&mut profile);

        // Query capabilities for this exact profile. The chain borrows the
        // codec caps structs, so scope it and copy the results out.
        let mut h264_caps = vk::VideoDecodeH264CapabilitiesKHR::default();
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let (
            max_coded_extent,
            picture_access_granularity,
            max_dpb_slots,
            max_active_reference_pictures,
            cap_flags,
            std_header_version,
            min_bitstream_buffer_offset_alignment,
            min_bitstream_buffer_size_alignment,
        ) = {
            let mut caps = vk::VideoCapabilitiesKHR::default()
                .push(&mut h264_caps)
                .push(&mut decode_caps);
            query_decode_caps(&self.common.context, &profile_info, &mut caps)?;
            (
                caps.max_coded_extent,
                caps.picture_access_granularity,
                caps.max_dpb_slots,
                caps.max_active_reference_pictures,
                caps.flags,
                caps.std_header_version,
                caps.min_bitstream_buffer_offset_alignment,
                caps.min_bitstream_buffer_size_alignment,
            )
        };
        let decode_flags = decode_caps.flags;

        let coded_width = align_up(sps.coded_width(), picture_access_granularity.width.max(1));
        let coded_height = align_up(sps.coded_height(), picture_access_granularity.height.max(1));
        if coded_width > max_coded_extent.width || coded_height > max_coded_extent.height {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: stream is {}x{}, device maximum is {}x{}",
                coded_width, coded_height, max_coded_extent.width, max_coded_extent.height
            )));
        }

        self.common.bitstream_offset_alignment = min_bitstream_buffer_offset_alignment.max(1);
        self.common.bitstream_size_alignment = min_bitstream_buffer_size_alignment.max(1);

        let coincide =
            decode_flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);
        let use_layered_dpb =
            !cap_flags.contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);

        // The stream needs max_num_ref_frames references plus the current
        // picture. On top of that we reserve `output_depth` slots so decoded
        // frames can be handed out without a copy while decoding continues; the
        // device caps bound the total.
        let required = (sps.max_num_ref_frames as u32 + 1).max(2);
        let slot_limit = max_dpb_slots.min(crate::decoder::h264::dpb::MAX_DPB_SLOTS as u32);
        if required > slot_limit {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: stream needs {} DPB slots, device supports {}",
                required, max_dpb_slots
            )));
        }
        let slot_count = (required + self.output_depth as u32).min(slot_limit) as usize;
        let output_slots = slot_count - required as usize;
        if output_slots < self.output_depth {
            debug!(
                "H.264 decode: {} of {} requested output slots available \
                 (stream needs {} of the device's {} DPB slots); \
                 decode-order frames beyond that are copied",
                output_slots, self.output_depth, required, max_dpb_slots
            );
        }

        // Only the reference pictures are ever active at once: the current
        // picture and the output reservation are not references.
        let max_active_references = (required - 1).min(max_active_reference_pictures);
        if max_active_references < required - 1 {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: stream needs {} active reference pictures, device supports {}",
                required - 1,
                max_active_reference_pictures
            )));
        }

        // Pick a picture format the device actually supports for decode.
        let dpb_usage = if coincide {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
        };
        let supported =
            query_supported_video_formats(&self.common.context, &profile_info, dpb_usage)?;
        let preferred = get_video_format(pixel_format, bit_depth);
        let picture_format = if supported.contains(&preferred) {
            preferred
        } else {
            *supported.first().ok_or_else(|| {
                PixelForgeError::CodecNotSupported(
                    "H.264 decode: no supported picture format for this profile".to_string(),
                )
            })?
        };

        // Hand the shared layer everything it needs to build the session; it
        // owns the Vulkan objects, this module owns only the H.264 reasoning
        // that produced these numbers.
        let plan = SessionPlan {
            coded_width,
            coded_height,
            picture_format,
            bit_depth,
            pixel_format,
            slot_count,
            output_slots,
            max_active_references,
            coincide,
            use_layered_dpb,
            dpb_usage,
        };
        self.common.create_session(
            &plan,
            &profile_info,
            &std_header_version,
            |common, session| Self::build_session_params(common, session, sps, pps),
        )?;

        self.active_sps_id = sps.sps_id;
        self.active_pps_id = pps.pps_id;
        self.dpb = Some(DecodeDpb::new(slot_count, self.common.slot_pins.clone()));

        debug!(
            "H.264 decode session: {}x{} {:?}, {} DPB slots ({} for output), \
             layered={}, coincide={}, bitstream alignment {}/{} (offset/size)",
            coded_width,
            coded_height,
            picture_format,
            slot_count,
            output_slots,
            use_layered_dpb,
            coincide,
            self.common.bitstream_offset_alignment,
            self.common.bitstream_size_alignment
        );
        Ok(())
    }

    /// Translate the parsed SPS/PPS into StdVideo structs and create session
    /// parameters.
    fn build_session_params(
        common: &DecoderCommon,
        session: vk::VideoSessionKHR,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let mut sps_flags: ash::vk::native::StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
        sps_flags.set_constraint_set0_flag((sps.constraint_set_flags >> 7) as u32 & 1);
        sps_flags.set_constraint_set1_flag((sps.constraint_set_flags >> 6) as u32 & 1);
        sps_flags.set_constraint_set2_flag((sps.constraint_set_flags >> 5) as u32 & 1);
        sps_flags.set_constraint_set3_flag((sps.constraint_set_flags >> 4) as u32 & 1);
        sps_flags.set_constraint_set4_flag((sps.constraint_set_flags >> 3) as u32 & 1);
        sps_flags.set_constraint_set5_flag((sps.constraint_set_flags >> 2) as u32 & 1);
        sps_flags.set_direct_8x8_inference_flag(sps.direct_8x8_inference_flag as u32);
        sps_flags.set_mb_adaptive_frame_field_flag(sps.mb_adaptive_frame_field_flag as u32);
        sps_flags.set_frame_mbs_only_flag(sps.frame_mbs_only_flag as u32);
        sps_flags.set_delta_pic_order_always_zero_flag(sps.delta_pic_order_always_zero_flag as u32);
        sps_flags.set_separate_colour_plane_flag(sps.separate_colour_plane_flag as u32);
        sps_flags.set_gaps_in_frame_num_value_allowed_flag(
            sps.gaps_in_frame_num_value_allowed_flag as u32,
        );
        sps_flags.set_qpprime_y_zero_transform_bypass_flag(
            sps.qpprime_y_zero_transform_bypass_flag as u32,
        );
        sps_flags.set_frame_cropping_flag(sps.frame_cropping_flag as u32);
        sps_flags.set_seq_scaling_matrix_present_flag(0);
        sps_flags.set_vui_parameters_present_flag(0);

        let std_sps = ash::vk::native::StdVideoH264SequenceParameterSet {
            flags: sps_flags,
            profile_idc: std_profile_idc(sps.profile_idc)?,
            level_idc: std_level_idc(sps.level_idc),
            chroma_format_idc: std_chroma_format_idc(sps.chroma_format_idc)?,
            seq_parameter_set_id: sps.sps_id,
            bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
            log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
            pic_order_cnt_type: sps.pic_order_cnt_type as u32,
            offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
            offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
            log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
            num_ref_frames_in_pic_order_cnt_cycle: sps.offsets_for_ref_frame.len() as u8,
            max_num_ref_frames: sps.max_num_ref_frames,
            reserved1: 0,
            pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1,
            frame_crop_left_offset: sps.frame_crop_left_offset,
            frame_crop_right_offset: sps.frame_crop_right_offset,
            frame_crop_top_offset: sps.frame_crop_top_offset,
            frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
            reserved2: 0,
            pOffsetForRefFrame: if sps.offsets_for_ref_frame.is_empty() {
                std::ptr::null()
            } else {
                sps.offsets_for_ref_frame.as_ptr()
            },
            pScalingLists: std::ptr::null(),
            pSequenceParameterSetVui: std::ptr::null(),
        };

        let mut pps_flags: ash::vk::native::StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
        pps_flags.set_transform_8x8_mode_flag(pps.transform_8x8_mode_flag as u32);
        pps_flags.set_redundant_pic_cnt_present_flag(pps.redundant_pic_cnt_present_flag as u32);
        pps_flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag as u32);
        pps_flags.set_deblocking_filter_control_present_flag(
            pps.deblocking_filter_control_present_flag as u32,
        );
        pps_flags.set_weighted_pred_flag(pps.weighted_pred_flag as u32);
        pps_flags.set_bottom_field_pic_order_in_frame_present_flag(
            pps.bottom_field_pic_order_in_frame_present_flag as u32,
        );
        pps_flags.set_entropy_coding_mode_flag(pps.entropy_coding_mode_flag as u32);
        pps_flags.set_pic_scaling_matrix_present_flag(0);

        let std_pps = ash::vk::native::StdVideoH264PictureParameterSet {
            flags: pps_flags,
            seq_parameter_set_id: pps.sps_id,
            pic_parameter_set_id: pps.pps_id,
            num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
            weighted_bipred_idc: pps.weighted_bipred_idc as u32,
            pic_init_qp_minus26: pps.pic_init_qp_minus26,
            pic_init_qs_minus26: pps.pic_init_qs_minus26,
            chroma_qp_index_offset: pps.chroma_qp_index_offset,
            second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
            pScalingLists: std::ptr::null(),
        };

        let sps_array = [std_sps];
        let pps_array = [std_pps];
        let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let mut h264_create = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add_info);

        let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut h264_create);

        unsafe {
            common
                .video_queue_fn
                .create_video_session_parameters(&create_info, None)
        }
        .map_err(|e| {
            PixelForgeError::SessionParametersCreation(format!(
                "H.264 decode session parameters: {:?}",
                e
            ))
        })
    }

    /// Split `data` into pictures, absorbing any parameter sets encountered.
    /// Group NALs into pictures. Returns the pictures plus, if any slice was
    /// skipped for missing parameter sets, the reason a keyframe is needed.
    fn split_pictures<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<Picture<'a>>, Option<String>)> {
        // Parameter sets are absorbed into self; slice grouping is delegated to
        // the pure `group_slices` so it can be tested without a device.
        let mut pictures: Vec<Picture<'a>> = Vec::new();
        let mut awaiting: Option<String> = None;

        for nal in iter_nal_units(data) {
            match nal.nal_type {
                NalType::Sps => {
                    let sps = parser::parse_sps(nal.payload())?;
                    debug!(
                        "H.264 SPS id={} {}x{} profile={}",
                        sps.sps_id,
                        sps.coded_width(),
                        sps.coded_height(),
                        sps.profile_idc
                    );
                    self.sps_map.insert(sps.sps_id, sps);
                }
                NalType::Pps => {
                    let sps_map = &self.sps_map;
                    let pps = parser::parse_pps(nal.payload(), |id| {
                        sps_map.get(&id).map(|s| s.chroma_format_idc)
                    })?;
                    self.pps_map.insert(pps.pps_id, pps);
                }
                t if t.is_slice() => {
                    match group_slices(&mut pictures, &nal, &self.sps_map, &self.pps_map) {
                        Ok(()) => {}
                        // Missing parameter sets: skip the slice, remember why.
                        Err(PixelForgeError::NeedsKeyframe(reason)) => {
                            awaiting.get_or_insert(reason);
                        }
                        Err(e) => return Err(e),
                    }
                }
                // SEI, AUD, and everything else is not needed by the driver.
                _ => {}
            }
        }

        Ok((pictures, awaiting))
    }

    pub(crate) fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeFuture> {
        // What is missing when a picture cannot be decoded yet; set only while
        // no frame has been produced, so a keyframe later in the same buffer
        // still yields output. Seeded with slices skipped during splitting for
        // missing parameter sets.
        let (pictures, mut awaiting) = self.split_pictures(data)?;
        // Whether any picture actually reached the GPU, which decides between
        // returning a batch and asking for a keyframe.
        let mut decoded_any = false;

        for picture in pictures {
            let is_idr = picture.nal_type == NalType::IdrSlice;

            // A non-IDR picture before the first keyframe references reference
            // pictures and POC state that do not exist yet. Skip it and ask for
            // a keyframe rather than decoding garbage.
            if self.awaiting_keyframe && !is_idr {
                awaiting.get_or_insert_with(|| {
                    "no keyframe decoded yet; non-IDR picture cannot be decoded".to_string()
                });
                continue;
            }

            // Resolve the parameter sets this picture uses. Missing sets mean we
            // joined before they were sent — also a keyframe-recovery case.
            let pps = match self.pps_map.get(&picture.header.pps_id).cloned() {
                Some(pps) => pps,
                None => {
                    awaiting.get_or_insert_with(|| {
                        format!("PPS {} not yet received", picture.header.pps_id)
                    });
                    continue;
                }
            };
            let sps = match self.sps_map.get(&pps.sps_id).cloned() {
                Some(sps) => sps,
                None => {
                    awaiting.get_or_insert_with(|| format!("SPS {} not yet received", pps.sps_id));
                    continue;
                }
            };

            // (Re)build the session if the stream's geometry or ids changed.
            self.activate(&sps, &pps)?;
            // Reorder depth comes from the stream (VUI); without it, bound by
            // the reference count, which is never smaller than the reorder
            // depth in practice.
            self.reorder_depth = sps
                .max_num_reorder_frames
                .map(|v| v as usize)
                .unwrap_or(sps.max_num_ref_frames as usize);

            if let Some(frame) = self.decode_picture(&picture, &sps, pts)? {
                // A decoded IDR re-establishes a valid decode point.
                if is_idr {
                    self.awaiting_keyframe = false;
                }
                let depth_reorder = self.reorder_depth;
                let ready = self.reorder.push(&mut self.common, &frame, depth_reorder)?;
                decoded_any = true;
                // The picture's submissions are complete as far as recording
                // goes; hand its slot and frames to the completion thread and
                // move on to the next slot.
                self.common.end_picture(ready);
            }
        }

        // Only surface the keyframe request when nothing was decoded: if a
        // keyframe arrived later in the same buffer, its pictures take priority.
        if !decoded_any && let Some(reason) = awaiting {
            return Err(PixelForgeError::NeedsKeyframe(format!(
                "H.264 decode: {}",
                reason
            )));
        }

        Ok(self.common.finish_batch(Vec::new()))
    }

    /// Record and submit the decode of a single picture, and wait for it.
    fn decode_picture(
        &mut self,
        picture: &Picture,
        sps: &Sps,
        pts: u64,
    ) -> Result<Option<DecodedPicture>> {
        // --- Host-side bookkeeping: POC, slot, reference set ---
        let state = {
            let dpb = self.dpb.as_mut().expect("activated above");
            dpb.begin_picture(
                picture.nal_type,
                picture.ref_idc,
                &picture.header,
                sps,
                picture.is_intra,
            )?
        };
        // Every slot can be busy for two reasons: the stream keeps that many
        // references (a hard error), or a frame the caller still holds pins one
        // (wait for them to drop it, like the encoder waits for a free slot).
        let slot = loop {
            let dpb = self.dpb.as_mut().expect("active");
            if let Some(slot) = dpb.try_allocate_slot() {
                break slot;
            }
            if !dpb.has_pinned_slots() {
                return Err(PixelForgeError::InvalidInput(
                    "H.264 decode: no free DPB slot (stream exceeds negotiated DPB size)"
                        .to_string(),
                ));
            }
            self.common.slot_pins.wait_for_release();
        };

        // --- Pack the slice data into the staging buffer ---
        let (buffer_range, slice_offsets) = self.stage_slices(picture, sps)?;

        // --- Record and submit ---
        self.common.begin_decode_commands()?;
        self.common.record_barriers(slot);
        self.record_decode(picture, &state, slot, buffer_range, &slice_offsets)?;
        self.common.submit_decode()?;

        // --- Publish the frame, then retire references ---
        let active = self.common.session.as_mut().expect("active");
        active.dpb_slot_active[slot as usize] = true;

        let (image, image_view, layout, array_layer) = if active.coincide {
            let (image, layer) = active.dpb_image_for_slot(slot);
            (
                image,
                active.dpb_views[slot as usize],
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                layer,
            )
        } else {
            let (image, _, view) = active
                .output_image
                .expect("non-coincide implies output image");
            (image, view, vk::ImageLayout::VIDEO_DECODE_DST_KHR, 0)
        };

        let (width, height) = sps.display_dimensions();
        let frame = DecodedPicture {
            slot,
            image,
            image_view,
            layout,
            array_layer,
            pixel_format: active.pixel_format,
            width,
            height,
            coded_width: active.coded_width,
            coded_height: active.coded_height,
            pts,
            display_order: state.poc,
            is_keyframe: state.is_idr,
        };

        self.dpb
            .as_mut()
            .expect("active")
            .end_picture(slot, &state, sps);

        Ok(Some(frame))
    }

    /// Copy the picture's slice NALs into the staging buffer.
    ///
    /// Returns the aligned buffer range to hand to the driver and the offset of
    /// each slice within it.
    /// Stage this picture's slices for the driver.
    ///
    /// The copying is generic; only the profile (needed to size the buffer) is
    /// H.264's business.
    fn stage_slices(&mut self, picture: &Picture, sps: &Sps) -> Result<(u64, Vec<u32>)> {
        let (mut profile, pixel_format, bit_depth) = Self::profile_for(sps)?;
        let chroma_subsampling = match pixel_format {
            PixelFormat::Yuv420 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            _ => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        };
        let depth_flag = match bit_depth {
            BitDepth::Eight => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            BitDepth::Ten => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        };
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(depth_flag)
            .chroma_bit_depth(depth_flag)
            .push(&mut profile);

        self.common.stage_slices(&picture.slices, &profile_info)
    }

    /// Transition the DPB slots this decode touches into DPB layout, and the
    /// output image (if distinct) into decode-DST layout.
    /// Record `vkCmdBeginVideoCoding` / `vkCmdDecodeVideo` / `vkCmdEndVideoCoding`.
    fn record_decode(
        &self,
        picture: &Picture,
        state: &PictureState,
        slot: u8,
        buffer_range: u64,
        slice_offsets: &[u32],
    ) -> Result<()> {
        let active = self.common.session.as_ref().expect("active");

        let extent = vk::Extent2D {
            width: active.coded_width,
            height: active.coded_height,
        };

        // --- Reference slots ---
        // Every reference the driver may use, plus the slot we reconstruct into.
        let refs = self.dpb.as_ref().expect("active").references().to_vec();

        let mut ref_std_infos: Vec<ash::vk::native::StdVideoDecodeH264ReferenceInfo> = Vec::new();
        for r in &refs {
            let mut flags: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags =
                unsafe { std::mem::zeroed() };
            flags.set_top_field_flag(0);
            flags.set_bottom_field_flag(0);
            flags.set_used_for_long_term_reference(r.long_term as u32);
            flags.set_is_non_existing(0);
            // For a long-term reference this field carries LongTermFrameIdx
            // rather than frame_num; that is how the driver identifies it when
            // building the reference lists.
            let frame_num = if r.long_term {
                r.long_term_frame_idx as u16
            } else {
                r.frame_num
            };
            ref_std_infos.push(ash::vk::native::StdVideoDecodeH264ReferenceInfo {
                flags,
                FrameNum: frame_num,
                reserved: 0,
                PicOrderCnt: r.poc,
            });
        }

        // Setup (reconstruction) slot info for the current picture.
        let mut setup_flags: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags =
            unsafe { std::mem::zeroed() };
        setup_flags.set_top_field_flag(0);
        setup_flags.set_bottom_field_flag(0);
        setup_flags.set_used_for_long_term_reference(0);
        setup_flags.set_is_non_existing(0);
        let setup_std_info = ash::vk::native::StdVideoDecodeH264ReferenceInfo {
            flags: setup_flags,
            FrameNum: state.frame_num,
            reserved: 0,
            PicOrderCnt: [state.poc, state.poc],
        };

        // Picture resources. These must outlive the command recording.
        let dpb_resource = |slot: u8| -> vk::VideoPictureResourceInfoKHR<'_> {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(extent)
                .base_array_layer(0)
                .image_view_binding(active.dpb_views[slot as usize])
        };

        let ref_resources: Vec<vk::VideoPictureResourceInfoKHR> =
            refs.iter().map(|r| dpb_resource(r.slot)).collect();
        let setup_resource = dpb_resource(slot);

        let mut ref_h264_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> = ref_std_infos
            .iter()
            .map(|info| vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(info))
            .collect();
        let mut setup_h264_info =
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_std_info);

        // Slots passed to begin_video_coding: all active references, plus the
        // slot we reconstruct into. The latter is not active yet -- it only
        // becomes active via `setup_reference_slot` in the decode info -- and
        // a non-negative slot_index here must name an already-active slot
        // (VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239). So it is listed
        // with slot_index = -1, which still binds its picture resource.
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for ((r, resource), h264_info) in refs
            .iter()
            .zip(ref_resources.iter())
            .zip(ref_h264_infos.iter_mut())
        {
            begin_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(r.slot as i32)
                    .picture_resource(resource)
                    .push(h264_info),
            );
        }
        begin_slots.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource),
        );

        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(active.session)
            .video_session_parameters(active.session_params)
            .reference_slots(&begin_slots);

        let device = self.common.context.device();
        unsafe {
            self.common
                .video_queue_fn
                .cmd_begin_video_coding(self.common.decode_command_buffer(), &begin_info);
        }

        // On the first use of the session, all DPB slots must be reset.
        if !active.dpb_slot_active.iter().any(|&a| a) {
            let control = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            unsafe {
                self.common
                    .video_queue_fn
                    .cmd_control_video_coding(self.common.decode_command_buffer(), &control);
            }
        }

        // --- Picture info ---
        let mut pic_flags: ash::vk::native::StdVideoDecodeH264PictureInfoFlags =
            unsafe { std::mem::zeroed() };
        pic_flags.set_field_pic_flag(0);
        pic_flags.set_is_intra(state.is_intra as u32);
        pic_flags.set_IdrPicFlag(state.is_idr as u32);
        pic_flags.set_bottom_field_flag(0);
        pic_flags.set_is_reference(state.is_reference as u32);
        pic_flags.set_complementary_field_pair(0);

        let std_pic_info = ash::vk::native::StdVideoDecodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: self.active_sps_id,
            pic_parameter_set_id: picture.header.pps_id,
            reserved1: 0,
            reserved2: 0,
            frame_num: state.frame_num,
            idr_pic_id: state.idr_pic_id,
            PicOrderCnt: [state.poc, state.poc],
        };

        let mut h264_picture_info = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&std_pic_info)
            .slice_offsets(slice_offsets);

        // The decode destination: the DPB slot itself when the implementation
        // supports coincide, otherwise the separate output image.
        let output_resource = match active.output_image {
            None => setup_resource,
            Some((_, _, view)) => vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(extent)
                .base_array_layer(0)
                .image_view_binding(view),
        };

        // Only reference pictures need a setup slot; disposable pictures still
        // decode into a DPB resource but are not retained.
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(slot as i32)
            .picture_resource(&setup_resource)
            .push(&mut setup_h264_info);

        // Reference slots for the decode itself (the current slot is excluded).
        let mut decode_ref_h264_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> = ref_std_infos
            .iter()
            .map(|info| vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(info))
            .collect();
        let mut decode_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for ((r, resource), h264_info) in refs
            .iter()
            .zip(ref_resources.iter())
            .zip(decode_ref_h264_infos.iter_mut())
        {
            decode_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(r.slot as i32)
                    .picture_resource(resource)
                    .push(h264_info),
            );
        }

        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.common.bitstream_buffer())
            .src_buffer_offset(0)
            .src_buffer_range(buffer_range)
            .dst_picture_resource(output_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&decode_slots)
            .push(&mut h264_picture_info);

        unsafe {
            self.common
                .video_decode_fn
                .cmd_decode_video(self.common.decode_command_buffer(), &decode_info);
            self.common.video_queue_fn.cmd_end_video_coding(
                self.common.decode_command_buffer(),
                &vk::VideoEndCodingInfoKHR::default(),
            );
        }
        let _ = device;
        Ok(())
    }
}

impl Drop for H264Decoder {
    fn drop(&mut self) {
        // The reorder pool's images must be freed while `common` (and its
        // device) is still alive; `common`'s own Drop runs afterward.
        self.reorder.destroy(&self.common);
    }
}

/// Append a slice NAL to the picture in progress, or begin a new picture.
///
/// Free function (rather than a method) so the grouping logic — which is pure
/// and easy to get wrong for multi-slice streams — can be unit tested without a
/// Vulkan device.
fn group_slices<'a>(
    pictures: &mut Vec<Picture<'a>>,
    nal: &NalUnit<'a>,
    sps_map: &HashMap<u8, Sps>,
    pps_map: &HashMap<u8, Pps>,
) -> Result<()> {
    // Peek the header to find the picture this slice belongs to.
    // Peek just far enough to find which PPS (and hence SPS) applies.
    let mut probe = crate::decoder::bitreader::BitReader::new(nal.payload());
    let _first_mb_in_slice = probe.ue()?;
    let _slice_type = probe.ue()?;
    let pps_id = probe.ue()? as u8;

    // Missing parameter sets: we joined the stream before they were sent. Not a
    // fault — signal that a keyframe (which carries fresh sets) is needed.
    let pps = pps_map.get(&pps_id).ok_or_else(|| {
        PixelForgeError::NeedsKeyframe(format!("PPS {} not yet received", pps_id))
    })?;
    let sps = sps_map.get(&pps.sps_id).ok_or_else(|| {
        PixelForgeError::NeedsKeyframe(format!("SPS {} not yet received", pps.sps_id))
    })?;

    let header = parser::parse_slice_header(nal, sps, pps)?;
    let is_intra = matches!(header.slice_type, SliceType::I | SliceType::Si);

    // Picture boundary detection (clause 7.4.1.2.4, frame-picture subset).
    //
    // `first_mb_in_slice == 0` marks the first slice of a picture and is
    // sufficient for conforming streams; the syntax comparisons additionally
    // catch a picture boundary that a truncated or spliced stream would
    // otherwise hide (e.g. a dropped first slice).
    let starts_new = match pictures.last() {
        None => true,
        Some(prev) => {
            header.first_mb_in_slice == 0
                || prev.header.frame_num != header.frame_num
                || prev.header.pps_id != header.pps_id
                || prev.nal_type != nal.nal_type
                || prev.header.pic_order_cnt_lsb != header.pic_order_cnt_lsb
                || prev.header.idr_pic_id != header.idr_pic_id
        }
    };

    if starts_new {
        pictures.push(Picture {
            header,
            nal_type: nal.nal_type,
            ref_idc: nal.ref_idc,
            is_intra,
            slices: vec![nal.data],
        });
    } else {
        let picture = pictures.last_mut().expect("checked above");
        // A picture is intra only if *every* slice is intra.
        picture.is_intra &= is_intra;
        picture.slices.push(nal.data);
    }
    Ok(())
}

fn std_profile_idc(profile_idc: u8) -> Result<ash::vk::native::StdVideoH264ProfileIdc> {
    Ok(match profile_idc {
        66 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
        77 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
        100 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
        244 => {
            ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
        }
        other => {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: unsupported profile_idc {}",
                other
            )));
        }
    })
}

fn std_chroma_format_idc(idc: u8) -> Result<ash::vk::native::StdVideoH264ChromaFormatIdc> {
    Ok(match idc {
        0 => {
            ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME
        }
        1 => ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        2 => ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_422,
        3 => ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_444,
        other => {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: invalid chroma_format_idc {}",
                other
            )));
        }
    })
}

/// Map `level_idc` (10 * level) onto the StdVideo enum.
fn std_level_idc(level_idc: u8) -> ash::vk::native::StdVideoH264LevelIdc {
    use ash::vk::native::*;
    match level_idc {
        10 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
        11 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
        12 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2,
        13 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3,
        20 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
        21 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1,
        22 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_2,
        30 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
        31 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
        32 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_2,
        40 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
        41 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        42 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2,
        50 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
        51 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1,
        52 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2,
        60 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
        61 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1,
        _ => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Group the slices of a real multi-slice stream (4 slices per picture,
    /// generated by `x264 --slices 4`). Exercises `group_slices` without a GPU.
    #[test]
    fn groups_multislice_pictures() {
        let data: &[u8] = include_bytes!("../../../tests/data/multislice.264");

        let mut sps_map = HashMap::new();
        let mut pps_map = HashMap::new();
        let mut pictures: Vec<Picture> = Vec::new();

        for nal in parser::iter_nal_units(data) {
            match nal.nal_type {
                NalType::Sps => {
                    let sps = parser::parse_sps(nal.payload()).unwrap();
                    sps_map.insert(sps.sps_id, sps);
                }
                NalType::Pps => {
                    let pps = parser::parse_pps(nal.payload(), |id| {
                        sps_map.get(&id).map(|s: &Sps| s.chroma_format_idc)
                    })
                    .unwrap();
                    pps_map.insert(pps.pps_id, pps);
                }
                t if t.is_slice() => {
                    group_slices(&mut pictures, &nal, &sps_map, &pps_map).unwrap();
                }
                _ => {}
            }
        }

        // 120 slice NALs must group into 30 pictures of 4 slices each.
        assert_eq!(pictures.len(), 30, "expected 30 pictures");
        for (i, picture) in pictures.iter().enumerate() {
            assert_eq!(picture.slices.len(), 4, "picture {i} should have 4 slices");
        }

        // The first picture is the IDR and is intra-coded.
        assert_eq!(pictures[0].nal_type, NalType::IdrSlice);
        assert!(pictures[0].is_intra, "IDR must be intra");
        assert_ne!(pictures[0].ref_idc, 0, "IDR must be a reference");

        // frame_num must be constant within a picture and advance between them.
        assert_eq!(pictures[0].header.frame_num, 0);
        assert!(
            pictures[1].header.frame_num > 0,
            "frame_num must advance after the IDR"
        );
    }
}
