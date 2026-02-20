use super::{AV1Encoder, MIN_BITSTREAM_BUFFER_SIZE, SUPERBLOCK_SIZE};

use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{
    allocate_session_memory, create_bitstream_buffer, create_command_resources, create_dpb_images,
    create_image, get_video_format, make_codec_name, map_bitstream_buffer,
    query_supported_video_formats,
};
use crate::encoder::PixelFormat;
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use std::ptr;
use tracing::{debug, info};

impl AV1Encoder {
    /// Create a new AV1 encoder.
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
            "Creating AV1 encoder: requested {}x{}, pixel_format={:?}",
            width, height, config.pixel_format
        );

        // Load video queue extension functions.
        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_encode_fn =
            ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

        // Get chroma subsampling from pixel format.
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();
        let luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();
        let chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();

        // AV1 profile selection based on chroma subsampling (not bit depth).
        // Main profile: 8/10-bit, 4:2:0 only.
        // High profile: 8/10-bit, 4:2:0 and 4:4:4.
        let profile = match config.pixel_format {
            PixelFormat::Yuv420 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
            _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        };

        // Preferred input format based on pixel format and bit depth.
        let preferred_src_format = get_video_format(config.pixel_format, config.bit_depth);

        // Create AV1 encode profile.
        let mut av1_profile_info = vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(profile);

        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(luma_bit_depth)
            .chroma_bit_depth(chroma_bit_depth);
        profile_info.p_next =
            (&mut av1_profile_info as *mut vk::VideoEncodeAV1ProfileInfoKHR).cast();

        // Query encode capabilities.
        let video_queue_instance =
            ash::khr::video_queue::Instance::load(context.entry(), context.instance());
        let mut av1_capabilities = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR {
            p_next: (&mut av1_capabilities as *mut vk::VideoEncodeAV1CapabilitiesKHR).cast(),
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
                "Failed to query Vulkan Video encode capabilities for AV1: {:?}",
                result
            )));
        }

        // Helper functions for alignment calculations.
        let gcd = |mut a: u32, mut b: u32| {
            while b != 0 {
                let tmp = a % b;
                a = b;
                b = tmp;
            }
            a
        };
        let lcm = |a: u32, b: u32| {
            if a == 0 || b == 0 {
                0
            } else {
                a / gcd(a, b) * b
            }
        };
        let align_up = |value: u32, alignment: u32| {
            if alignment <= 1 {
                value
            } else {
                value.div_ceil(alignment) * alignment
            }
        };

        let gran_w = capabilities.picture_access_granularity.width.max(1);
        let gran_h = capabilities.picture_access_granularity.height.max(1);
        let align_w = lcm(SUPERBLOCK_SIZE, gran_w);
        let align_h = lcm(SUPERBLOCK_SIZE, gran_h);

        let mut aligned_width = align_up(width, align_w);
        let mut aligned_height = align_up(height, align_h);

        aligned_width = aligned_width.max(capabilities.min_coded_extent.width);
        aligned_height = aligned_height.max(capabilities.min_coded_extent.height);

        if aligned_width > capabilities.max_coded_extent.width
            || aligned_height > capabilities.max_coded_extent.height
        {
            return Err(PixelForgeError::InvalidInput(format!(
                "Requested coded extent {}x{} (aligned to {}x{}) exceeds device max {}x{}",
                width,
                height,
                aligned_width,
                aligned_height,
                capabilities.max_coded_extent.width,
                capabilities.max_coded_extent.height
            )));
        }

        info!(
            "Using coded extent {}x{} (granularity {}x{})",
            aligned_width, aligned_height, gran_w, gran_h
        );

        // Query supported formats.
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
                "No supported Vulkan Video SRC formats for AV1".to_string(),
            ));
        }
        if supported_dpb_formats.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "No supported Vulkan Video DPB formats for AV1".to_string(),
            ));
        }

        info!("Supported SRC formats: {:?}", supported_src_formats);
        info!("Supported DPB formats: {:?}", supported_dpb_formats);

        let picture_format = if supported_src_formats.contains(&preferred_src_format) {
            preferred_src_format
        } else {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Preferred input format {:?} not supported for AV1",
                preferred_src_format
            )));
        };

        let reference_picture_format = supported_dpb_formats
            .iter()
            .copied()
            .find(|f| *f == picture_format)
            .unwrap_or(supported_dpb_formats[0]);

        debug!(
            "Selected formats: picture={:?}, reference={:?}",
            picture_format, reference_picture_format
        );

        // Get encode queue family.
        let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
        })?;

        // Create video session.
        let std_header_version = vk::ExtensionProperties {
            extension_name: make_codec_name(b"VK_STD_vulkan_video_codec_av1_encode"),
            spec_version: vk::make_api_version(0, 1, 0, 0),
        };

        // Calculate DPB slots and active references.
        let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
        let max_active_reference_pictures_supported =
            capabilities.max_active_reference_pictures as usize;

        if max_dpb_slots_supported < 2 {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Device reports max_dpb_slots={}, need at least 2",
                max_dpb_slots_supported
            )));
        }

        let mut target_active_refs = (config.max_reference_frames as usize)
            .min(max_active_reference_pictures_supported)
            .min(8); // AV1 supports up to 8 reference frames

        if target_active_refs < 1 && max_active_reference_pictures_supported >= 1 {
            target_active_refs = 1;
        }

        // AV1 typically needs: active refs + 1 for current frame being setup
        let requested_dpb_slots = (target_active_refs + 1).min(max_dpb_slots_supported);

        info!(
            "DPB configuration: slots={}, active_refs={} (max_supported: slots={}, refs={})",
            requested_dpb_slots,
            target_active_refs,
            max_dpb_slots_supported,
            max_active_reference_pictures_supported
        );

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(encode_queue_family)
            .video_profile(&profile_info)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D {
                width: aligned_width,
                height: aligned_height,
            })
            .reference_picture_format(reference_picture_format)
            .max_dpb_slots(requested_dpb_slots as u32)
            .max_active_reference_pictures(target_active_refs as u32)
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

        // Allocate session memory.
        let session_memory = allocate_session_memory(&context, session, &video_queue_fn)?;

        // Create AV1 sequence header - similar to H.265 VPS/SPS/PPS but for AV1.
        // Calculate frame dimension representation bits.
        // Use actual display dimensions for sequence header (not coded extent).
        // The video session's max_coded_extent is the upper bound for alignment,
        // but the sequence header and per-frame coded extents use display dimensions.
        let frame_width_bits = 32 - (width - 1).leading_zeros();
        let frame_height_bits = 32 - (height - 1).leading_zeros();

        // AV1 color configuration.
        let color_config_flags = ash::vk::native::StdVideoAV1ColorConfigFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1ColorConfigFlags::new_bitfield_1(
                0, // mono_chrome
                1, // color_range (full range)
                0, // separate_uv_delta_q
                1, // color_description_present_flag (we provide color primaries/transfer/matrix)
                0, // reserved
            ),
        };

        // Bit depth: 8 for Eight, 10 for Ten
        let bit_depth = match config.bit_depth {
            crate::encoder::BitDepth::Eight => 8,
            crate::encoder::BitDepth::Ten => 10,
        };

        // Chroma subsampling based on pixel format.
        let (subsampling_x, subsampling_y) = match config.pixel_format {
            PixelFormat::Yuv420 => (1u8, 1u8), // 4:2:0
            PixelFormat::Yuv444 => (0u8, 0u8), // 4:4:4
            _ => (1u8, 1u8),                   // Default to 4:2:0
        };

        let color_config = ash::vk::native::StdVideoAV1ColorConfig {
            flags: color_config_flags,
            BitDepth: bit_depth,
            subsampling_x,
            subsampling_y,
            reserved1: 0,
            color_primaries: ash::vk::native::StdVideoAV1ColorPrimaries_STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709,
            transfer_characteristics: ash::vk::native::StdVideoAV1TransferCharacteristics_STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_709,
            matrix_coefficients: ash::vk::native::StdVideoAV1MatrixCoefficients_STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_709,
            chroma_sample_position: ash::vk::native::StdVideoAV1ChromaSamplePosition_STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN,
        };

        // AV1 sequence header flags - use minimal set to avoid driver issues.
        // Disable features we're not providing data for (restoration, most inter-frame features).
        let seq_flags = ash::vk::native::StdVideoAV1SequenceHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1SequenceHeaderFlags::new_bitfield_1(
                0, // still_picture
                0, // reduced_still_picture_header
                0, // use_128x128_superblock (use 64x64 superblocks)
                0, // enable_filter_intra - disable for simplicity
                0, // enable_intra_edge_filter - disable for simplicity
                0, // enable_interintra_compound
                0, // enable_masked_compound
                0, // enable_warped_motion - disable for simplicity
                0, // enable_dual_filter - disable for simplicity
                1, // enable_order_hint - keep for reference frames
                0, // enable_jnt_comp
                0, // enable_ref_frame_mvs
                0, // frame_id_numbers_present_flag
                0, // enable_superres
                1, // enable_cdef - keep enabled
                0, // enable_restoration - DISABLE (we don't provide restoration data)
                0, // film_grain_params_present
                0, // timing_info_present_flag
                0, // initial_display_delay_present_flag
                0, // reserved
            ),
        };

        let av1_sequence_header = ash::vk::native::StdVideoAV1SequenceHeader {
            flags: seq_flags,
            seq_profile: profile,
            frame_width_bits_minus_1: (frame_width_bits - 1) as u8,
            frame_height_bits_minus_1: (frame_height_bits - 1) as u8,
            max_frame_width_minus_1: (width - 1) as u16,
            max_frame_height_minus_1: (height - 1) as u16,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            order_hint_bits_minus_1: 7, // 8 bits for order hint
            seq_force_integer_mv: 0,
            seq_force_screen_content_tools: 0,
            reserved1: [0; 5],
            pColorConfig: &color_config,
            pTimingInfo: ptr::null(), // No timing info
        };

        // Create decoder model info (zero-initialized like FFmpeg).
        let decoder_model_info = ash::vk::native::StdVideoEncodeAV1DecoderModelInfo {
            buffer_delay_length_minus_1: 0,
            buffer_removal_time_length_minus_1: 0,
            frame_presentation_time_length_minus_1: 0,
            reserved1: 0,
            num_units_in_decoding_tick: 0,
        };

        // Create operating point info (single operating point like FFmpeg).
        let operating_point = ash::vk::native::StdVideoEncodeAV1OperatingPointInfo {
            flags: ash::vk::native::StdVideoEncodeAV1OperatingPointInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    ash::vk::native::StdVideoEncodeAV1OperatingPointInfoFlags::new_bitfield_1(
                        0, // decoder_model_present_for_this_op
                        0, // low_delay_mode_flag
                        0, // initial_display_delay_present_for_this_op
                        0, // reserved
                    ),
            },
            operating_point_idc: 0,
            seq_level_idx: 5, // Level 3.1 (encoded as: 2.0=0, 2.1=1, ... 3.0=4, 3.1=5)
            seq_tier: 0,
            initial_display_delay_minus_1: 0,
            decoder_buffer_delay: 0,
            encoder_buffer_delay: 0,
        };

        // Create session parameters with all required structures (matching FFmpeg).
        let mut av1_session_params_create_info =
            vk::VideoEncodeAV1SessionParametersCreateInfoKHR::default()
                .std_sequence_header(&av1_sequence_header)
                .std_decoder_model_info(&decoder_model_info)
                .std_operating_points(std::slice::from_ref(&operating_point));

        // Add quality level info to pNext chain (matching FFmpeg).
        // Chain: SessionParametersCreateInfo -> QualityLevelInfo -> AV1SessionParametersCreateInfo
        let mut quality_info = vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0); // Default quality level

        quality_info.p_next = (&mut av1_session_params_create_info
            as *mut vk::VideoEncodeAV1SessionParametersCreateInfoKHR)
            .cast();

        let session_params_create_info = vk::VideoSessionParametersCreateInfoKHR {
            video_session: session,
            p_next: (&mut quality_info as *mut vk::VideoEncodeQualityLevelInfoKHR).cast(),
            ..Default::default()
        };

        let session_params = unsafe {
            video_queue_fn
                .create_video_session_parameters(&session_params_create_info, None)
                .map_err(|e| {
                    PixelForgeError::SessionParametersCreation(format!(
                        "Failed to create AV1 session parameters: {:?}",
                        e
                    ))
                })?
        };

        // Create input image.
        let (input_image, input_image_memory, input_image_view) = create_image(
            &context,
            aligned_width,
            aligned_height,
            picture_format,
            false, // is_dpb
            &profile_info,
        )?;

        let input_image_layout = vk::ImageLayout::UNDEFINED;

        // Create DPB images.
        let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
            &context,
            aligned_width,
            aligned_height,
            reference_picture_format,
            requested_dpb_slots,
            &profile_info,
        )?;

        // Create bitstream buffer.
        let bitstream_buffer_size =
            MIN_BITSTREAM_BUFFER_SIZE.max(aligned_width as usize * aligned_height as usize);
        let (bitstream_buffer, bitstream_buffer_memory) =
            create_bitstream_buffer(&context, bitstream_buffer_size, &profile_info)?;

        // Map bitstream buffer persistently.
        let bitstream_buffer_ptr =
            map_bitstream_buffer(&context, bitstream_buffer_memory, bitstream_buffer_size)?;

        // Create command resources.
        let cmd_resources = create_command_resources(&context, encode_queue_family)?;
        let command_pool = cmd_resources.command_pool;
        let upload_command_buffer = cmd_resources.upload_command_buffer;
        let upload_fence = cmd_resources.upload_fence;
        let encode_command_buffer = cmd_resources.encode_command_buffer;
        let encode_fence = cmd_resources.encode_fence;

        // Create query pool for bitstream size queries.
        // Need 1 query to capture bitstream offset and size.
        // Need to provide profile info and feedback flags in pNext chain.
        let mut query_feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );
        query_feedback_info.p_next = (&profile_info as *const vk::VideoProfileInfoKHR).cast();

        let mut query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1);
        query_pool_create_info.p_next =
            (&query_feedback_info as *const vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR).cast();

        let query_pool = unsafe {
            context
                .device()
                .create_query_pool(&query_pool_create_info, None)
                .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?
        };

        // Initialize GOP structure.
        let gop = GopStructure::new(config.gop_size, config.b_frame_count, config.gop_size);

        Ok(Self {
            context,
            config,
            gop,
            video_queue_fn,
            video_encode_fn,
            session,
            session_params,
            session_memory,
            input_frame_num: 0,
            encode_frame_num: 0,
            frame_num: 0,
            order_hint: 0,
            input_image,
            input_image_memory,
            input_image_view,
            input_image_layout,
            dpb_images,
            dpb_image_memories,
            dpb_image_views,
            dpb_slot_count: requested_dpb_slots,
            bitstream_buffer,
            bitstream_buffer_memory,
            bitstream_buffer_ptr,
            command_pool,
            upload_command_buffer,
            upload_fence,
            encode_command_buffer,
            encode_fence,
            query_pool,
            header_data: None,
            current_dpb_slot: 0,
            references: Vec::new(),
            active_reference_count: target_active_refs as u32,
        })
    }
}
