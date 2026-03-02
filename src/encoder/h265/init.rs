use super::H265Encoder;

use crate::encoder::dpb::{DecodedPictureBuffer, DecodedPictureBufferTrait, DpbConfig};
use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{
    align_up, allocate_session_memory, clear_input_image, create_bitstream_buffer,
    create_command_resources, create_dpb_images, create_image, get_video_format, lcm,
    make_codec_name, map_bitstream_buffer, query_supported_video_formats, ClearImageParams,
    MIN_BITSTREAM_BUFFER_SIZE,
};
use crate::encoder::{BitDepth, ColorDescription, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use std::ptr;
use tracing::{debug, info};

impl H265Encoder {
    /// Create a new H.265/HEVC encoder.
    pub fn new(context: VideoContext, config: crate::encoder::EncodeConfig) -> Result<Self> {
        // B-frames are not yet supported.
        if config.b_frame_count > 0 {
            panic!(
                "B-frame encoding is not yet supported. Set b_frame_count=0 in encoder config. \
                 Got b_frame_count={}",
                config.b_frame_count
            );
        }

        let width = config.dimensions.width;
        let height = config.dimensions.height;

        info!(
            "Creating H.265 encoder: {}x{}, pixel_format={:?}",
            width, height, config.pixel_format
        );

        // Load video queue extension functions.
        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_encode_fn =
            ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

        // Get chroma subsampling from pixel format via `From` impl
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();

        // Get bit depth flags from config
        let bit_depth_flags: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();
        let video_format = get_video_format(config.pixel_format, config.bit_depth);

        // Select profile based on pixel format and bit depth:
        // - Main for YUV420 8-bit
        // - Main 10 for YUV420 10-bit
        // - Main 4:4:4 for YUV444 8-bit
        // - Main 4:4:4 10 for YUV444 10-bit
        let profile_idc = match (config.pixel_format, config.bit_depth) {
            (PixelFormat::Yuv420, BitDepth::Eight) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
            }
            (PixelFormat::Yuv420, BitDepth::Ten) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
            }
            (PixelFormat::Yuv444, BitDepth::Eight) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
            }
            (PixelFormat::Yuv444, BitDepth::Ten) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format / bit depth combination for H.265: {:?} / {:?}",
                    config.pixel_format, config.bit_depth
                )));
            }
        };

        // Create H.265 encode profile
        let mut h265_profile_info =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);

        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags);
        profile_info.p_next =
            (&mut h265_profile_info as *mut vk::VideoEncodeH265ProfileInfoKHR).cast();

        // Query capabilities to determine limits.
        let video_queue_instance =
            ash::khr::video_queue::Instance::load(context.entry(), context.instance());
        let mut h265_capabilities = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR {
            p_next: (&mut h265_capabilities as *mut vk::VideoEncodeH265CapabilitiesKHR).cast(),
            ..Default::default()
        };
        let mut capabilities = vk::VideoCapabilitiesKHR {
            p_next: (&mut encode_capabilities as *mut vk::VideoEncodeCapabilitiesKHR).cast(),
            ..Default::default()
        };

        let result = unsafe {
            (video_queue_instance
                .fp()
                .get_physical_device_video_capabilities_khr)(
                context.physical_device(),
                &profile_info,
                &mut capabilities,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Failed to query H.265 capabilities: {:?}",
                result
            )));
        }

        // Compute aligned coded extent using capabilities (picture_access_granularity,
        // min/max_coded_extent) - required for AMD which reports granularity 64x16 and
        // min_coded_extent 130x128.
        let ctb_size = super::CTB_SIZE;
        let gran_w = capabilities.picture_access_granularity.width.max(1);
        let gran_h = capabilities.picture_access_granularity.height.max(1);
        let align_w = lcm(ctb_size, gran_w);
        let align_h = lcm(ctb_size, gran_h);

        let mut aligned_width = align_up(width, align_w);
        let mut aligned_height = align_up(height, align_h);

        aligned_width = align_up(
            aligned_width.max(capabilities.min_coded_extent.width),
            align_w,
        );
        aligned_height = align_up(
            aligned_height.max(capabilities.min_coded_extent.height),
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

        info!(
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

        // Query supported formats for SRC and DPB usage.
        let supported_src_formats = query_supported_video_formats(
            &context,
            &profile_info,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
        )?;
        let supported_dpb_formats = query_supported_video_formats(
            &context,
            &profile_info,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        )?;

        if supported_src_formats.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "No supported Vulkan Video SRC formats returned for this H.265 profile".to_string(),
            ));
        }
        info!("Supported SRC formats: {:?}", supported_src_formats);
        if supported_dpb_formats.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "No supported Vulkan Video DPB formats returned for this H.265 profile".to_string(),
            ));
        }
        info!("Supported DPB formats: {:?}", supported_dpb_formats);

        let picture_format = if supported_src_formats.contains(&video_format) {
            video_format
        } else {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Preferred input format {:?} is not supported for VIDEO_ENCODE_SRC_KHR. Supported: {:?}",
                video_format, supported_src_formats
            )));
        };

        let reference_picture_format = supported_dpb_formats
            .iter()
            .copied()
            .find(|f| *f == picture_format)
            .unwrap_or(supported_dpb_formats[0]);

        debug!(
            "Selected Vulkan Video formats: picture_format={:?}, reference_picture_format={:?}",
            picture_format, reference_picture_format
        );

        let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
        let max_active_reference_pictures_supported =
            capabilities.max_active_reference_pictures as usize;

        if max_dpb_slots_supported < 2 {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Device reports max_dpb_slots={} for this profile; need at least 2",
                max_dpb_slots_supported
            )));
        }

        // Target number of active reference pictures.
        let mut target_active_refs = (config.max_reference_frames as usize)
            .min(max_active_reference_pictures_supported)
            .min(15); // H.265 limit

        if target_active_refs < 1 && max_active_reference_pictures_supported >= 1 {
            target_active_refs = 1;
        }

        // Calculate needed DPB slots
        let needed_dpb_slots = if config.b_frame_count > 0 {
            target_active_refs + config.b_frame_count as usize + 2
        } else {
            target_active_refs + 1
        };

        let dpb_slot_count = needed_dpb_slots
            .min(max_dpb_slots_supported)
            .min(crate::encoder::dpb::MAX_DPB_SLOTS);

        // Final clamp
        let max_active_reference_pictures =
            target_active_refs.min(dpb_slot_count.saturating_sub(1));

        debug!(
            "Allocating {} DPB slots (req {}, max {}), active refs {} (req {}, max {})",
            dpb_slot_count,
            needed_dpb_slots,
            max_dpb_slots_supported,
            max_active_reference_pictures,
            target_active_refs,
            max_active_reference_pictures_supported
        );

        // Create video session.
        let std_header_version = vk::ExtensionProperties {
            extension_name: make_codec_name(b"VK_STD_vulkan_video_codec_h265_encode"),
            spec_version: vk::make_api_version(0, 1, 0, 0),
        };

        let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
        })?;

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(encode_queue_family)
            .flags(vk::VideoSessionCreateFlagsKHR::empty())
            .video_profile(&profile_info)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D {
                width: aligned_width,
                height: aligned_height,
            })
            .reference_picture_format(reference_picture_format)
            .max_dpb_slots(dpb_slot_count as u32)
            .max_active_reference_pictures(max_active_reference_pictures as u32)
            .std_header_version(&std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_khr)(
                context.device().handle(),
                &session_create_info,
                ptr::null(),
                &mut session,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::VideoSessionCreation(format!(
                "{:?}",
                result
            )));
        }

        // Query and allocate session memory.
        let session_memory = allocate_session_memory(&context, session, &video_queue_fn)?;

        // Create VPS, SPS and PPS
        // H.265 coding block sizes:
        // CTB size (cuSize) = 32x32 -> log2_ctb_size = 5 -> cuSize enum = 2
        // Min CB size (cuMinSize) = 16x16 -> log2_min_cb_size = 4 -> cuMinSize enum = 1
        let ctb_log2_size_y: u8 = 5; // 32x32 CTB
        let min_cb_log2_size_y: u8 = 4; // 16x16 min CB
        let log2_min_transform_block_size: u8 = 2; // 4x4 min TU
        let log2_max_transform_block_size: u8 = 5; // 32x32 max TU

        // Calculate SPS parameters.
        let pic_width_in_luma_samples = aligned_width;
        let pic_height_in_luma_samples = aligned_height;

        // Conformance window for cropping.
        // SubWidthC and SubHeightC depend on chroma format:
        // - YUV420: SubWidthC=2, SubHeightC=2
        // - YUV444: SubWidthC=1, SubHeightC=1
        let (sub_width_c, sub_height_c) = match config.pixel_format {
            PixelFormat::Yuv420 => (2u32, 2u32),
            PixelFormat::Yuv444 => (1u32, 1u32),
            _ => (2u32, 2u32), // Default to 4:2:0
        };
        let conf_win_right_offset = (aligned_width - width) / sub_width_c;
        let conf_win_bottom_offset = (aligned_height - height) / sub_height_c;
        let conformance_window_flag = conf_win_right_offset > 0 || conf_win_bottom_offset > 0;

        // Profile tier level - Main/Main10/Main 4:4:4 profile, level 5.1 (sufficient for 4K)
        let profile_tier_level = ash::vk::native::StdVideoH265ProfileTierLevel {
            flags: ash::vk::native::StdVideoH265ProfileTierLevelFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
                    0, // general_tier_flag (Main tier)
                    1, // general_progressive_source_flag
                    0, // general_interlaced_source_flag
                    0, // general_non_packed_constraint_flag
                    1, // general_frame_only_constraint_flag
                ),
                __bindgen_padding_0: [0; 3],
            },
            general_profile_idc: profile_idc,
            general_level_idc: ash::vk::native::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        };

        // Decoded Picture Buffer Manager
        let dec_pic_buf_mgr = ash::vk::native::StdVideoH265DecPicBufMgr {
            max_latency_increase_plus1: [0; 7],
            max_dec_pic_buffering_minus1: [(dpb_slot_count - 1) as u8, 0, 0, 0, 0, 0, 0],
            max_num_reorder_pics: [0; 7], // No B-frame reordering by default
        };

        // Short-term reference picture set (in SPS for RPS in SPS mode)
        // Set up a simple RPS with one reference picture
        let short_term_ref_pic_set = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
                    0, // inter_ref_pic_set_prediction_flag
                    0, // delta_rps_sign
                ),
                __bindgen_padding_0: [0; 3],
            },
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: 1, // First negative reference is used
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: 1, // One backward reference
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0; 16],
            delta_poc_s1_minus1: [0; 16],
        };

        let long_term_ref_pics_sps = ash::vk::native::StdVideoH265LongTermRefPicsSps {
            used_by_curr_pic_lt_sps_flag: 0,
            lt_ref_pic_poc_lsb_sps: [0; 32],
        };

        // SPS flags
        let sps_flags = ash::vk::native::StdVideoH265SpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265SpsFlags::new_bitfield_1(
                1,                                           // sps_temporal_id_nesting_flag
                0,                                           // separate_colour_plane_flag
                if conformance_window_flag { 1 } else { 0 }, // conformance_window_flag
                1, // sps_sub_layer_ordering_info_present_flag
                0, // scaling_list_enabled_flag
                0, // sps_scaling_list_data_present_flag
                1, // amp_enabled_flag (asymmetric motion partitions)
                1, // sample_adaptive_offset_enabled_flag
                0, // pcm_enabled_flag
                0, // pcm_loop_filter_disabled_flag
                0, // long_term_ref_pics_present_flag
                0, // sps_temporal_mvp_enabled_flag
                0, // strong_intra_smoothing_enabled_flag
                1, // vui_parameters_present_flag
                0, // sps_extension_present_flag
                0, // sps_range_extension_flag
                0, // transform_skip_rotation_enabled_flag
                0, // transform_skip_context_enabled_flag
                0, // implicit_rdpcm_enabled_flag
                0, // explicit_rdpcm_enabled_flag
                0, // extended_precision_processing_flag
                0, // intra_smoothing_disabled_flag
                0, // high_precision_offsets_enabled_flag
                0, // persistent_rice_adaptation_enabled_flag
                0, // cabac_bypass_alignment_enabled_flag
                0, // sps_scc_extension_flag
                0, // sps_curr_pic_ref_enabled_flag
                0, // palette_mode_enabled_flag
                0, // sps_palette_predictor_initializers_present_flag
                0, // intra_boundary_filtering_disabled_flag
            ),
        };

        // Build VUI structure. Defaults to BT.709 when no color description is set.
        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());
        let vui_flags = ash::vk::native::StdVideoH265SpsVuiFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265SpsVuiFlags::new_bitfield_1(
                1,                                         // aspect_ratio_info_present_flag
                0,                                         // overscan_info_present_flag
                0,                                         // overscan_appropriate_flag
                1,                                         // video_signal_type_present_flag
                if color_desc.full_range { 1 } else { 0 }, // video_full_range_flag
                1,                                         // colour_description_present_flag
                0,                                         // chroma_loc_info_present_flag
                0,                                         // neutral_chroma_indication_flag
                0,                                         // field_seq_flag
                0,                                         // frame_field_info_present_flag
                0,                                         // default_display_window_flag
                0,                                         // vui_timing_info_present_flag
                0,                                         // vui_poc_proportional_to_timing_flag
                0,                                         // vui_hrd_parameters_present_flag
                0,                                         // bitstream_restriction_flag
                0,                                         // tiles_fixed_structure_flag
                0, // motion_vectors_over_pic_boundaries_flag
                0, // restricted_ref_pic_lists_flag
            ),
            __bindgen_padding_0: 0,
        };

        let vui = ash::vk::native::StdVideoH265SequenceParameterSetVui {
            flags: vui_flags,
            aspect_ratio_idc:
                ash::vk::native::StdVideoH265AspectRatioIdc_STD_VIDEO_H265_ASPECT_RATIO_IDC_SQUARE,
            sar_width: 0,
            sar_height: 0,
            video_format: 5,
            colour_primaries: color_desc.color_primaries,
            transfer_characteristics: color_desc.transfer_characteristics,
            matrix_coeffs: color_desc.matrix_coefficients,
            chroma_sample_loc_type_top_field: 0,
            chroma_sample_loc_type_bottom_field: 0,
            reserved1: 0,
            reserved2: 0,
            def_disp_win_left_offset: 0,
            def_disp_win_right_offset: 0,
            def_disp_win_top_offset: 0,
            def_disp_win_bottom_offset: 0,
            vui_num_units_in_tick: 0,
            vui_time_scale: 0,
            vui_num_ticks_poc_diff_one_minus1: 0,
            min_spatial_segmentation_idc: 0,
            reserved3: 0,
            max_bytes_per_pic_denom: 0,
            max_bits_per_min_cu_denom: 0,
            log2_max_mv_length_horizontal: 0,
            log2_max_mv_length_vertical: 0,
            pHrdParameters: ptr::null(),
        };

        // Calculate bit depth minus 8 values for SPS (0 for 8-bit, 2 for 10-bit)
        let bit_depth_minus8: u8 = match config.bit_depth {
            BitDepth::Eight => 0,
            BitDepth::Ten => 2,
        };

        // Get chroma_format_idc based on pixel format.
        let chroma_format_idc = match config.pixel_format {
            PixelFormat::Yuv420 => {
                ash::vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420
            }
            PixelFormat::Yuv444 => {
                ash::vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_444
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format for H.265: {:?}",
                    config.pixel_format
                )));
            }
        };

        let sps = ash::vk::native::StdVideoH265SequenceParameterSet {
            flags: sps_flags,
            chroma_format_idc,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_seq_parameter_set_id: 0,
            bit_depth_luma_minus8: bit_depth_minus8,
            bit_depth_chroma_minus8: bit_depth_minus8,
            log2_max_pic_order_cnt_lsb_minus4: 4, // POC LSB range = 256 (wraps every ~4s at 60fps)
            log2_min_luma_coding_block_size_minus3: min_cb_log2_size_y - 3,
            log2_diff_max_min_luma_coding_block_size: ctb_log2_size_y - min_cb_log2_size_y,
            log2_min_luma_transform_block_size_minus2: log2_min_transform_block_size - 2,
            log2_diff_max_min_luma_transform_block_size: log2_max_transform_block_size
                - log2_min_transform_block_size,
            max_transform_hierarchy_depth_inter: (ctb_log2_size_y - log2_min_transform_block_size)
                .max(1),
            max_transform_hierarchy_depth_intra: 3,
            num_short_term_ref_pic_sets: 0,
            num_long_term_ref_pics_sps: 0,
            pcm_sample_bit_depth_luma_minus1: 7,
            pcm_sample_bit_depth_chroma_minus1: 7,
            log2_min_pcm_luma_coding_block_size_minus3: min_cb_log2_size_y - 3,
            log2_diff_max_min_pcm_luma_coding_block_size: ctb_log2_size_y - min_cb_log2_size_y,
            reserved1: 0,
            reserved2: 0,
            palette_max_size: 0,
            delta_palette_max_predictor_size: 0,
            motion_vector_resolution_control_idc: 0,
            sps_num_palette_predictor_initializers_minus1: 0,
            conf_win_left_offset: 0,
            conf_win_right_offset,
            conf_win_top_offset: 0,
            conf_win_bottom_offset,
            pProfileTierLevel: ptr::null(), // Will be set below
            pDecPicBufMgr: ptr::null(),
            pScalingLists: ptr::null(),
            pShortTermRefPicSet: ptr::null(),
            pLongTermRefPicsSps: ptr::null(),
            pSequenceParameterSetVui: &vui,
            pPredictorPaletteEntries: ptr::null(),
        };

        // VPS flags
        let vps_flags = ash::vk::native::StdVideoH265VpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265VpsFlags::new_bitfield_1(
                1, // vps_temporal_id_nesting_flag
                1, // vps_sub_layer_ordering_info_present_flag
                0, // vps_timing_info_present_flag
                0, // vps_poc_proportional_to_timing_flag
            ),
            __bindgen_padding_0: [0; 3],
        };

        let vps = ash::vk::native::StdVideoH265VideoParameterSet {
            flags: vps_flags,
            vps_video_parameter_set_id: 0,
            vps_max_sub_layers_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_num_ticks_poc_diff_one_minus1: 0,
            reserved3: 0,
            pDecPicBufMgr: ptr::null(),
            pHrdParameters: ptr::null(),
            pProfileTierLevel: ptr::null(),
        };

        // PPS flags
        let pps_flags = ash::vk::native::StdVideoH265PpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265PpsFlags::new_bitfield_1(
                0, // dependent_slice_segments_enabled_flag
                0, // output_flag_present_flag
                0, // sign_data_hiding_enabled_flag
                1, // cabac_init_present_flag
                0, // constrained_intra_pred_flag
                1, // transform_skip_enabled_flag
                1, // cu_qp_delta_enabled_flag
                0, // pps_slice_chroma_qp_offsets_present_flag
                0, // weighted_pred_flag
                0, // weighted_bipred_flag
                0, // transquant_bypass_enabled_flag
                0, // tiles_enabled_flag
                0, // entropy_coding_sync_enabled_flag
                0, // uniform_spacing_flag
                0, // loop_filter_across_tiles_enabled_flag
                1, // pps_loop_filter_across_slices_enabled_flag
                1, // deblocking_filter_control_present_flag
                0, // deblocking_filter_override_enabled_flag
                0, // pps_deblocking_filter_disabled_flag
                0, // pps_scaling_list_data_present_flag
                0, // lists_modification_present_flag
                0, // slice_segment_header_extension_present_flag
                0, // pps_extension_present_flag
                0, // cross_component_prediction_enabled_flag
                0, // chroma_qp_offset_list_enabled_flag
                0, // pps_curr_pic_ref_enabled_flag
                0, // residual_adaptive_colour_transform_enabled_flag
                0, // pps_slice_act_qp_offsets_present_flag
                0, // pps_palette_predictor_initializers_present_flag
                0, // monochrome_palette_flag
                0, // pps_range_extension_flag
            ),
        };

        let pps = ash::vk::native::StdVideoH265PictureParameterSet {
            flags: pps_flags,
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            num_extra_slice_header_bits: 0,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,
            log2_parallel_merge_level_minus2: 0,
            log2_max_transform_skip_block_size_minus2: 0,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list_len_minus1: 0,
            cb_qp_offset_list: [0; 6],
            cr_qp_offset_list: [0; 6],
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            pps_act_y_qp_offset_plus5: 0,
            pps_act_cb_qp_offset_plus5: 0,
            pps_act_cr_qp_offset_plus3: 0,
            pps_num_palette_predictor_initializers: 0,
            luma_bit_depth_entry_minus8: bit_depth_minus8,
            chroma_bit_depth_entry_minus8: bit_depth_minus8,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],
            reserved3: 0,
            pScalingLists: ptr::null(),
            pPredictorPaletteEntries: ptr::null(),
        };

        // Box the structures so they live long enough for session parameter creation
        let profile_tier_level_boxed = Box::new(profile_tier_level);
        let dec_pic_buf_mgr_boxed = Box::new(dec_pic_buf_mgr);
        let short_term_ref_pic_set_boxed = Box::new(short_term_ref_pic_set);
        let long_term_ref_pics_sps_boxed = Box::new(long_term_ref_pics_sps);

        // Create mutable copies with correct pointers
        let mut sps_with_ptrs = sps;
        sps_with_ptrs.pProfileTierLevel = profile_tier_level_boxed.as_ref();
        sps_with_ptrs.pDecPicBufMgr = dec_pic_buf_mgr_boxed.as_ref();
        sps_with_ptrs.pShortTermRefPicSet = short_term_ref_pic_set_boxed.as_ref();
        sps_with_ptrs.pLongTermRefPicsSps = long_term_ref_pics_sps_boxed.as_ref();

        let mut vps_with_ptrs = vps;
        vps_with_ptrs.pProfileTierLevel = profile_tier_level_boxed.as_ref();
        vps_with_ptrs.pDecPicBufMgr = dec_pic_buf_mgr_boxed.as_ref();

        let vps_array = [vps_with_ptrs];
        let sps_array = [sps_with_ptrs];
        let pps_array = [pps];

        let h265_add_info = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&vps_array)
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);

        let mut h265_params_create_info =
            vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
                .max_std_vps_count(1)
                .max_std_sps_count(1)
                .max_std_pps_count(1)
                .parameters_add_info(&h265_add_info);

        // Chain quality level info into session parameters creation.
        // This is required by AMD RADV and matches FFmpeg's approach.
        let mut quality_level_info = vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0); // Use quality level 0 (best quality).
        quality_level_info.p_next = (&mut h265_params_create_info
            as *mut vk::VideoEncodeH265SessionParametersCreateInfoKHR)
            .cast();

        let mut params_create_info =
            vk::VideoSessionParametersCreateInfoKHR::default().video_session(session);
        params_create_info.p_next =
            (&mut quality_level_info as *mut vk::VideoEncodeQualityLevelInfoKHR).cast();

        let mut session_params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_parameters_khr)(
                context.device().handle(),
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

        // Create profile info for images/buffers
        let mut h265_profile_for_resources =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);
        let mut profile_for_resources = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags);
        profile_for_resources.p_next =
            (&mut h265_profile_for_resources as *mut vk::VideoEncodeH265ProfileInfoKHR).cast();

        // Create input image
        let (input_image, input_image_memory, input_image_view) = create_image(
            &context,
            aligned_width,
            aligned_height,
            picture_format,
            false,
            &profile_for_resources,
        )?;

        // Determine DPB mode: use layered DPB when the driver does not advertise
        // support for separate reference images (required for AMD RADV).
        let supports_separate_dpb = capabilities
            .flags
            .contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);
        let use_layered_dpb = !supports_separate_dpb;
        if use_layered_dpb {
            info!("Using layered DPB (driver does not support separate reference images)");
        }

        // Create DPB images.
        let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
            &context,
            aligned_width,
            aligned_height,
            reference_picture_format,
            dpb_slot_count,
            &profile_for_resources,
            use_layered_dpb,
        )?;

        // Create bitstream buffer.
        let (bitstream_buffer, bitstream_buffer_memory) =
            create_bitstream_buffer(&context, MIN_BITSTREAM_BUFFER_SIZE, &profile_for_resources)?;

        // Persistently map the bitstream buffer to avoid per-frame map/unmap overhead.
        let bitstream_buffer_ptr =
            map_bitstream_buffer(&context, bitstream_buffer_memory, MIN_BITSTREAM_BUFFER_SIZE)?;

        // Create command pool, buffers, and fences.
        // Use the transfer queue family for upload commands when the encode queue
        // doesn't support transfer operations (AMD RADV).
        let upload_queue_family = context.transfer_queue_family();
        let cmd_resources =
            create_command_resources(&context, encode_queue_family, upload_queue_family)?;
        let command_pool = cmd_resources.command_pool;
        let upload_command_pool = cmd_resources.upload_command_pool;
        let upload_command_buffer = cmd_resources.upload_command_buffer;
        let encode_command_buffer = cmd_resources.encode_command_buffer;
        let upload_fence = cmd_resources.upload_fence;
        let encode_fence = cmd_resources.encode_fence;

        // Clear the input image so padding between user dimensions and the
        // aligned coded extent is zero-initialized.
        clear_input_image(
            &context,
            &ClearImageParams {
                command_buffer: upload_command_buffer,
                fence: upload_fence,
                queue: context.transfer_queue(),
                image: input_image,
                width: aligned_width,
                height: aligned_height,
                pixel_format: config.pixel_format,
                bit_depth: config.bit_depth,
            },
        )?;

        // Create query pool
        let mut h265_profile_info_query =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);

        let mut profile_info_query = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags);
        profile_info_query.p_next =
            (&mut h265_profile_info_query as *mut vk::VideoEncodeH265ProfileInfoKHR).cast();

        let mut encode_feedback_create = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );

        encode_feedback_create.p_next =
            (&mut profile_info_query as *mut vk::VideoProfileInfoKHR).cast();

        let mut query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1);
        query_pool_create_info.p_next = (&mut encode_feedback_create
            as *mut vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR)
            .cast();

        let query_pool = unsafe {
            context
                .device()
                .create_query_pool(&query_pool_create_info, None)
        }
        .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;

        // Create DPB and GOP structure
        let mut dpb = DecodedPictureBuffer::new();
        let dpb_config = DpbConfig {
            dpb_size: dpb_slot_count as u32,
            max_num_ref_frames: if config.b_frame_count > 0 { 2 } else { 1 },
            use_multiple_references: config.b_frame_count > 0,
            max_long_term_refs: 0,
            log2_max_frame_num_minus4: 0,         // Not used in H.265
            log2_max_pic_order_cnt_lsb_minus4: 4, // max_poc_lsb = 256
            num_temporal_layers: 1,
        };
        dpb.h265.sequence_start(dpb_config);

        let mut gop = if config.b_frame_count > 0 {
            GopStructure::new(config.gop_size, config.b_frame_count, config.gop_size)
        } else {
            GopStructure::new_ip_only(config.gop_size)
        };

        // Set GOP parameters to match SPS values
        // log2_max_pic_order_cnt_lsb_minus4 = 4, so max_poc_lsb = 2^8 = 256
        gop.set_max_frame_num(4); // Not used in H.265 but set for compatibility
        gop.set_max_poc_lsb(4);

        // Initialize DPB slot activation tracking
        let dpb_slot_active = vec![false; dpb_slot_count];

        info!("H.265 encoder created successfully");

        Ok(Self {
            context,
            config: config.clone(),
            dpb,
            gop,
            aligned_width,
            aligned_height,
            video_queue_fn,
            video_encode_fn,
            session,
            session_params,
            session_memory,
            input_frame_num: 0,
            encode_frame_num: 0,
            input_image,
            input_image_memory,
            input_image_view,
            input_image_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            dpb_images,
            dpb_image_memories,
            dpb_image_views,
            dpb_slot_count,
            use_layered_dpb,
            bitstream_buffer,
            bitstream_buffer_memory,
            bitstream_buffer_ptr,
            command_pool,
            upload_command_pool,
            upload_command_buffer,
            upload_fence,
            encode_command_buffer,
            encode_fence,
            query_pool,
            header_data: None,
            has_backward_reference: false,
            backward_reference_poc: 0,
            backward_reference_dpb_slot: 2,
            current_dpb_slot: 0,
            l0_references: Vec::new(),
            active_reference_count: max_active_reference_pictures as u32,
            dpb_slot_active,
        })
    }
}
