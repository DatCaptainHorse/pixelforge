use super::H264Encoder;

use crate::encoder::{BitDepth, ColorDescription, PixelFormat};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use std::ptr;

impl H264Encoder {
    /// Build SPS/PPS and create Vulkan video session parameters.
    ///
    /// This is used both during initial encoder creation and by
    /// `set_color_description()` to rebuild session parameters with
    /// updated VUI color metadata. Keeping a single implementation
    /// ensures the parameter sets stay bit-for-bit consistent.
    pub(crate) fn create_session_params(
        &self,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let width = self.config.dimensions.width;
        let height = self.config.dimensions.height;

        let pic_width_in_mbs = self.aligned_width / 16;
        let pic_height_in_map_units = self.aligned_height / 16;

        // Cropping offsets are expressed in units that depend on chroma subsampling.
        // For progressive frames (frame_mbs_only_flag=1):
        // - 4:2:0 => crop_unit_x=2, crop_unit_y=2
        // - 4:4:4 => crop_unit_x=1, crop_unit_y=1
        let (crop_unit_x, crop_unit_y) = match self.config.pixel_format {
            PixelFormat::Yuv420 => (2u32, 2u32),
            PixelFormat::Yuv444 => (1u32, 1u32),
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format for H.264: {:?}",
                    self.config.pixel_format
                )));
            }
        };

        let coded_width = pic_width_in_mbs * 16;
        let coded_height = pic_height_in_map_units * 16;
        let frame_crop_right = coded_width.saturating_sub(width) / crop_unit_x;
        let frame_crop_bottom = coded_height.saturating_sub(height) / crop_unit_y;

        let mut sps_flags: ash::vk::native::StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
        sps_flags.set_constraint_set3_flag(0);
        sps_flags.set_direct_8x8_inference_flag(1);
        sps_flags.set_frame_mbs_only_flag(1);
        if frame_crop_right > 0 || frame_crop_bottom > 0 {
            sps_flags.set_frame_cropping_flag(1);
        }
        sps_flags.set_vui_parameters_present_flag(1);

        let chroma_format_idc = match self.config.pixel_format {
            PixelFormat::Yuv420 => {
                ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420
            }
            PixelFormat::Yuv444 => {
                ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_444
            }
            _ => unreachable!("Pixel format validated above"),
        };

        let (bit_depth_luma_minus8, bit_depth_chroma_minus8) = match self.config.bit_depth {
            BitDepth::Eight => (0u8, 0u8),
            BitDepth::Ten => (2u8, 2u8),
        };

        let max_active = self.active_reference_count as u8;

        let mut vui_flags: ash::vk::native::StdVideoH264SpsVuiFlags = unsafe { std::mem::zeroed() };
        vui_flags.set_aspect_ratio_info_present_flag(1);
        vui_flags.set_video_signal_type_present_flag(1);
        vui_flags.set_video_full_range_flag(if desc.full_range { 1 } else { 0 });
        vui_flags.set_color_description_present_flag(1);
        // Do not set HRD parameters when rate control is disabled/CQP.
        // HRD with zeroed bitrate values causes device loss on some drivers (AMD).
        vui_flags.set_nal_hrd_parameters_present_flag(0);
        vui_flags.set_bitstream_restriction_flag(1);

        let vui = ash::vk::native::StdVideoH264SequenceParameterSetVui {
            flags: vui_flags,
            aspect_ratio_idc:
                ash::vk::native::StdVideoH264AspectRatioIdc_STD_VIDEO_H264_ASPECT_RATIO_IDC_SQUARE,
            sar_width: 0,
            sar_height: 0,
            video_format: 5,
            colour_primaries: desc.color_primaries,
            transfer_characteristics: desc.transfer_characteristics,
            matrix_coefficients: desc.matrix_coefficients,
            num_units_in_tick: 0,
            time_scale: 0,
            max_num_reorder_frames: if self.config.b_frame_count > 0 { 1 } else { 0 },
            max_dec_frame_buffering: max_active + 1,
            chroma_sample_loc_type_top_field: 0,
            chroma_sample_loc_type_bottom_field: 0,
            reserved1: 0,
            pHrdParameters: ptr::null(),
        };

        let sps = ash::vk::native::StdVideoH264SequenceParameterSet {
            flags: sps_flags,
            profile_idc: self.profile_idc,
            level_idc: ash::vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
            chroma_format_idc,
            seq_parameter_set_id: 0,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            log2_max_frame_num_minus4: 4,
            pic_order_cnt_type: if self.config.b_frame_count > 0 {
                ash::vk::native::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0
            } else {
                ash::vk::native::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_2
            },
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            max_num_ref_frames: max_active,
            reserved1: 0,
            pic_width_in_mbs_minus1: pic_width_in_mbs - 1,
            pic_height_in_map_units_minus1: pic_height_in_map_units - 1,
            frame_crop_left_offset: 0,
            frame_crop_right_offset: frame_crop_right,
            frame_crop_top_offset: 0,
            frame_crop_bottom_offset: frame_crop_bottom,
            reserved2: 0,
            pOffsetForRefFrame: ptr::null(),
            pScalingLists: ptr::null(),
            pSequenceParameterSetVui: &vui,
        };

        let mut pps_flags: ash::vk::native::StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
        // Enable 8x8 transform for High profile and above (required by some
        // drivers for High 4:4:4 Predictive SPS/PPS generation).
        let transform_8x8 = self.profile_idc
            >= ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH;
        pps_flags.set_transform_8x8_mode_flag(transform_8x8 as u32);
        // Use the driver's preferred entropy coding mode from quality level properties.
        // Some drivers (e.g., NVIDIA for H.264 High 4:4:4 Predictive) require CAVLC.
        pps_flags.set_entropy_coding_mode_flag(self.preferred_entropy_cabac as u32);
        pps_flags.set_deblocking_filter_control_present_flag(1);

        // vk_video_samples sets chroma QP offsets to 6 for 4:4:4 unless lossless.
        // This improves driver compatibility for SPS/PPS generation.
        let (chroma_qp_index_offset, second_chroma_qp_index_offset) = match self.config.pixel_format
        {
            PixelFormat::Yuv444 => (6i8, 6i8),
            _ => (0i8, 0i8),
        };

        let pps = ash::vk::native::StdVideoH264PictureParameterSet {
            flags: pps_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            num_ref_idx_l0_default_active_minus1: (max_active as i8 - 1).max(0) as u8,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_bipred_idc:
                ash::vk::native::StdVideoH264WeightedBipredIdc_STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset,
            second_chroma_qp_index_offset,
            pScalingLists: ptr::null(),
        };

        let sps_array = [sps];
        let pps_array = [pps];

        let h264_add_info = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);

        let mut h264_params_create_info =
            vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
                .max_std_sps_count(1)
                .max_std_pps_count(1)
                .parameters_add_info(&h264_add_info);

        // Chain quality level info into session parameters creation.
        // This is required by AMD RADV and matches FFmpeg's approach.
        let mut quality_level_info = vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0);
        quality_level_info.p_next = (&mut h264_params_create_info
            as *mut vk::VideoEncodeH264SessionParametersCreateInfoKHR)
            .cast();

        let mut params_create_info =
            vk::VideoSessionParametersCreateInfoKHR::default().video_session(self.session);
        params_create_info.p_next =
            (&mut quality_level_info as *mut vk::VideoEncodeQualityLevelInfoKHR).cast();

        let mut session_params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (self.video_queue_fn.fp().create_video_session_parameters_khr)(
                self.context.device().handle(),
                &params_create_info,
                ptr::null(),
                &mut session_params,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::SessionParametersCreation(format!(
                "{:?}",
                result
            )));
        }

        Ok(session_params)
    }
}
