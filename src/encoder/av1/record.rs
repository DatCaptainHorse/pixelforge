//! AV1 per-frame encode recording: builds the StdVideo* graph for one frame and
//! submits it. AV1 uses single-reference prediction.

use super::Av1;

use crate::encoder::codec::{EncoderCommon, FramePlan, RateControlPlan};
use crate::encoder::pipeline::EncodeFuture;
use crate::encoder::resources::{
    prepare_encode_command_buffer, record_dpb_barriers, record_post_encode_dpb_barrier,
};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

impl Av1 {
    pub(super) fn record(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture> {
        let is_key_frame = plan.is_idr();
        // All frames need a setup reference slot (DPB write) per the Vulkan spec
        // when maxDpbSlots > 0.
        let is_reference = true;
        let current_dpb_slot = common.current_dpb_slot;

        let slot = common.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let bitstream_buffer = slot.bitstream_buffer;
        let bitstream_buffer_size = slot.bitstream_buffer_size;
        let input_image_view = slot.input_image_view;

        debug!(
            "av1 record: key={}, refs_len={}, dpb_slot={}",
            is_key_frame,
            self.references.len(),
            current_dpb_slot
        );

        let rc = RateControlPlan::new(&common.config, 128);
        let qp = rc.qp;

        unsafe {
            prepare_encode_command_buffer(common.device(), command_buffer, query_pool)?;
        }
        let ref_dpb_slots: Vec<u8> = self.references.iter().map(|r| r.dpb_slot).collect();
        unsafe {
            record_dpb_barriers(
                common.device(),
                command_buffer,
                &common.dpb_images,
                common.use_layered_dpb,
                current_dpb_slot,
                &ref_dpb_slots,
                common.dpb_slot_active[current_dpb_slot as usize],
            );
        }

        let frame_type = if is_key_frame {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        };

        // show_frame for all frames; error_resilient_mode on key frames (FFmpeg).
        let mut picture_info_flags = ash::vk::native::StdVideoEncodeAV1PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        picture_info_flags.set_show_frame(1);
        if is_key_frame {
            picture_info_flags.set_error_resilient_mode(1);
        } else {
            picture_info_flags.set_showable_frame(1);
        }

        // Without MOTION_VECTOR_SCALING all picture resource codedExtents must
        // match and equal the sequence header's max_frame dimensions.
        let frame_extent = vk::Extent2D {
            width: common.config.dimensions.width,
            height: common.config.dimensions.height,
        };

        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(common.dpb_image_views[current_dpb_slot as usize]);

        let reference_info_flags = ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(
                0, 0, 0,
            ),
        };
        let std_reference_info = ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
            flags: reference_info_flags,
            frame_type,
            RefFrameId: current_dpb_slot as u32,
            OrderHint: self.order_hint as u8,
            reserved1: [0; 3],
            pExtensionHeader: std::ptr::null(),
        };

        let mut setup_av1_dpb_info =
            vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);
        let mut setup_av1_dpb_info_ref0 = setup_av1_dpb_info;
        let setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource)
            .push(&mut setup_av1_dpb_info_ref0);

        // Reference frames for inter frames.
        let mut reference_slots = Vec::new();
        let mut av1_reference_infos = Vec::new();
        let mut ref_picture_resources = Vec::new();
        let mut ref_std_infos = Vec::new();

        if !is_key_frame && !self.references.is_empty() {
            let ref_info = &self.references[0];
            ref_std_infos.push(ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
                flags: reference_info_flags,
                frame_type: ref_info.frame_type,
                RefFrameId: ref_info.dpb_slot as u32,
                OrderHint: ref_info.order_hint as u8,
                reserved1: [0; 3],
                pExtensionHeader: std::ptr::null(),
            });
            av1_reference_infos.push(vk::VideoEncodeAV1DpbSlotInfoKHR::default());
            av1_reference_infos[0] = av1_reference_infos[0].std_reference_info(&ref_std_infos[0]);

            ref_picture_resources.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(frame_extent)
                    .base_array_layer(0)
                    .image_view_binding(common.dpb_image_views[ref_info.dpb_slot as usize]),
            );

            let ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(ref_info.dpb_slot as i32)
                .picture_resource(&ref_picture_resources[0]);
            reference_slots.push(ref_slot);
            reference_slots[0] = reference_slots[0].push(&mut av1_reference_infos[0]);
        }

        // Quantization (FFmpeg-style defaults).
        let quantization = ash::vk::native::StdVideoAV1Quantization {
            flags: ash::vk::native::StdVideoAV1QuantizationFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoAV1QuantizationFlags::new_bitfield_1(0, 0, 0),
            },
            base_q_idx: qp as u8,
            DeltaQYDc: 0,
            DeltaQUDc: 0,
            DeltaQUAc: 0,
            DeltaQVDc: 0,
            DeltaQVAc: 0,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        };

        let cdef = ash::vk::native::StdVideoAV1CDEF {
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            cdef_y_pri_strength: [0; 8],
            cdef_y_sec_strength: [0; 8],
            cdef_uv_pri_strength: [0; 8],
            cdef_uv_sec_strength: [0; 8],
        };

        let loop_filter = ash::vk::native::StdVideoAV1LoopFilter {
            flags: ash::vk::native::StdVideoAV1LoopFilterFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoAV1LoopFilterFlags::new_bitfield_1(0, 0, 0),
            },
            loop_filter_level: [0; 4],
            loop_filter_sharpness: 0,
            update_ref_delta: 0,
            loop_filter_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            update_mode_delta: 1,
            loop_filter_mode_deltas: [0; 2],
        };

        let (ref_frame_idx, ref_order_hint, primary_ref_frame, refresh_frame_flags) =
            self.calculate_reference_frame_mapping(is_key_frame, current_dpb_slot);

        let std_picture_info = ash::vk::native::StdVideoEncodeAV1PictureInfo {
            flags: picture_info_flags,
            frame_type,
            frame_presentation_time: self.frame_num,
            current_frame_id: current_dpb_slot as u32,
            order_hint: self.order_hint as u8,
            primary_ref_frame,
            refresh_frame_flags,
            coded_denom: 0,
            render_width_minus_1: (common.config.dimensions.width - 1) as u16,
            render_height_minus_1: (common.config.dimensions.height - 1) as u16,
            interpolation_filter: ash::vk::native::StdVideoAV1InterpolationFilter_STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
            TxMode: ash::vk::native::StdVideoAV1TxMode_STD_VIDEO_AV1_TX_MODE_SELECT,
            delta_q_res: 0,
            delta_lf_res: 0,
            ref_order_hint,
            ref_frame_idx,
            reserved1: [0; 3],
            delta_frame_id_minus_1: [0; 7],
            pTileInfo: std::ptr::null(),
            pQuantization: &quantization,
            pSegmentation: std::ptr::null(),
            pLoopFilter: &loop_filter,
            pCDEF: &cdef,
            pLoopRestoration: std::ptr::null(),
            pGlobalMotion: std::ptr::null(),
            pExtensionHeader: std::ptr::null(),
            pBufferRemovalTimes: std::ptr::null(),
        };

        // Map AV1 reference names to DPB slots (SINGLE_REFERENCE uses LAST_FRAME).
        let mut reference_name_slot_indices = [-1i32; 7];
        if !is_key_frame && !self.references.is_empty() {
            reference_name_slot_indices[0] = self.references[0].dpb_slot as i32;
        }

        let (prediction_mode, rate_control_group) = if is_key_frame {
            (
                vk::VideoEncodeAV1PredictionModeKHR::INTRA_ONLY,
                vk::VideoEncodeAV1RateControlGroupKHR::INTRA,
            )
        } else {
            (
                vk::VideoEncodeAV1PredictionModeKHR::SINGLE_REFERENCE,
                vk::VideoEncodeAV1RateControlGroupKHR::PREDICTIVE,
            )
        };

        let mut av1_picture_info = vk::VideoEncodeAV1PictureInfoKHR::default()
            .std_picture_info(&std_picture_info)
            .prediction_mode(prediction_mode)
            .rate_control_group(rate_control_group)
            .reference_name_slot_indices(reference_name_slot_indices);
        if rc.is_disabled() {
            av1_picture_info = av1_picture_info.constant_q_index(qp);
        }

        let mut av1_rc_layer_info = vk::VideoEncodeAV1RateControlLayerInfoKHR::default();
        if rc.is_disabled() {
            let q_index = vk::VideoEncodeAV1QIndexKHR {
                intra_q_index: qp,
                predictive_q_index: qp,
                bipredictive_q_index: qp,
            };
            av1_rc_layer_info = av1_rc_layer_info
                .use_min_q_index(true)
                .min_q_index(q_index)
                .use_max_q_index(true)
                .max_q_index(q_index);
        } else {
            av1_rc_layer_info = av1_rc_layer_info
                .use_min_q_index(false)
                .use_max_q_index(false);
        }

        let rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(rc.average_bitrate as u64)
            .max_bitrate(rc.max_bitrate as u64)
            .frame_rate_numerator(common.config.frame_rate_numerator)
            .frame_rate_denominator(common.config.frame_rate_denominator)
            .push(&mut av1_rc_layer_info);
        let rc_layers = [rc_layer_info];

        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc.mode);
        if !rc.is_disabled() {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(common.config.virtual_buffer_size_ms)
                .initial_virtual_buffer_size_in_ms(common.config.initial_virtual_buffer_size_ms);
        }

        // Begin coding: setup slot (slot_index -1) plus any active reference slots.
        let mut all_reference_slots = Vec::new();
        if is_reference {
            let setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_picture_resource)
                .push(&mut setup_av1_dpb_info);
            all_reference_slots.push(setup_slot_for_begin);
        }
        all_reference_slots.extend_from_slice(&reference_slots);

        let is_first_frame = plan.is_first_frame();
        let mut av1_rc_info = vk::VideoEncodeAV1RateControlInfoKHR::default()
            .gop_frame_count(common.config.gop_size)
            .key_frame_period(common.config.gop_size)
            .consecutive_bipredictive_frame_count(0)
            .temporal_layer_count(1);

        let begin_coding_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(common.session)
                .video_session_parameters(common.session_params)
                .reference_slots(&all_reference_slots)
                .push(&mut av1_rc_info)
        } else {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(common.session)
                .video_session_parameters(common.session_params)
                .reference_slots(&all_reference_slots)
                .push(&mut rc_info)
                .push(&mut av1_rc_info)
        };

        unsafe {
            common
                .video_queue_fn
                .cmd_begin_video_coding(command_buffer, &begin_coding_info);
        }

        if is_first_frame {
            let mut quality_level_info =
                vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0);
            let control_info = vk::VideoCodingControlInfoKHR::default()
                .flags(
                    vk::VideoCodingControlFlagsKHR::RESET
                        | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                        | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
                )
                .push(&mut rc_info)
                .push(&mut av1_rc_info)
                .push(&mut quality_level_info);
            unsafe {
                common
                    .video_queue_fn
                    .cmd_control_video_coding(command_buffer, &control_info);
            }
        }

        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(input_image_view);

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .src_picture_resource(src_picture_resource)
            .dst_buffer(bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(bitstream_buffer_size as u64);
        if is_reference {
            encode_info = encode_info.setup_reference_slot(&setup_reference_slot);
        }
        if !reference_slots.is_empty() {
            encode_info = encode_info.reference_slots(&reference_slots);
        }
        encode_info = encode_info.push(&mut av1_picture_info);

        unsafe {
            let device = common.device();
            device.cmd_begin_query(
                command_buffer,
                query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
            common
                .video_encode_fn
                .cmd_encode_video(command_buffer, &encode_info);
            device.cmd_end_query(command_buffer, query_pool, 0);

            record_post_encode_dpb_barrier(
                device,
                command_buffer,
                &common.dpb_images,
                common.use_layered_dpb,
                current_dpb_slot,
            );

            let end_coding_info = vk::VideoEndCodingInfoKHR::default();
            common
                .video_queue_fn
                .cmd_end_video_coding(command_buffer, &end_coding_info);

            device
                .end_command_buffer(command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        common.submit_frame()
    }

    /// Build `ref_frame_idx`, `ref_order_hint`, `primary_ref_frame` and
    /// `refresh_frame_flags` for the current frame.
    fn calculate_reference_frame_mapping(
        &self,
        is_key_frame: bool,
        current_dpb_slot: u8,
    ) -> ([i8; 7], [u8; 8], u8, u8) {
        if is_key_frame {
            // Key frame refreshes all 8 reference slots; all names point to slot 0.
            ([0i8; 7], [0u8; 8], 7u8, 0xFFu8)
        } else if let Some(last_ref) = self.references.first() {
            let mut ref_frame_idx = [0i8; 7];
            let mut ref_order_hint = [0u8; 8];
            ref_frame_idx[0] = last_ref.dpb_slot as i8;
            ref_order_hint[0] = last_ref.order_hint as u8;
            // Refresh only the current slot so this frame becomes the new LAST_FRAME.
            let refresh_flags = 1u8 << current_dpb_slot;
            (ref_frame_idx, ref_order_hint, 0u8, refresh_flags)
        } else {
            ([0i8; 7], [0u8; 8], 7u8, 0x00u8)
        }
    }
}
