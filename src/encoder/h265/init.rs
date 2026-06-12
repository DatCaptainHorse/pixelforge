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
use ash::vk::TaggedStructure;
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

        // Get encoder tuning mode, usage and content hints.
        let encode_usage_hint: vk::VideoEncodeUsageFlagsKHR = config.encode_usage_hint.into();
        let encode_content_hint: vk::VideoEncodeContentFlagsKHR = config.encode_content_hint.into();
        let encoder_tuning_mode: vk::VideoEncodeTuningModeKHR = config.encoder_tuning_mode.into();

        let mut video_encode_usage_info = vk::VideoEncodeUsageInfoKHR::default()
            .video_usage_hints(encode_usage_hint)
            .video_content_hints(encode_content_hint)
            .tuning_mode(encoder_tuning_mode);

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

        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags)
            .push(&mut h265_profile_info)
            .push(&mut video_encode_usage_info);

        // Query capabilities to determine limits.
        let video_queue_instance =
            ash::khr::video_queue::Instance::load(context.entry(), context.instance());
        let mut h265_capabilities = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut h265_capabilities)
            .push(&mut encode_capabilities);

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

        // Build VPS/SPS/PPS and session parameters via shared helper.
        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());

        // Create profile info for images/buffers
        let mut h265_profile_for_resources =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);
        let profile_for_resources = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags)
            .push(&mut h265_profile_for_resources);

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
            .chroma_bit_depth(bit_depth_flags)
            .push(&mut h265_profile_info_query);

        let mut encode_feedback_create = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );

        let query_pool_create_info = unsafe {
            vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
                .query_count(1)
                .extend(&mut profile_info_query)
                .push(&mut encode_feedback_create)
        };

        let query_pool = unsafe {
            context
                .device()
                .create_query_pool(&query_pool_create_info, None)
        }
        .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;

        // Create timestamping query pool and resources
        let timestamp_query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2); // start and end

        let timestamp_query_pool = unsafe {
            context
                .device()
                .create_query_pool(&timestamp_query_pool_create_info, None)
        }
        .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;

        let timestamp_period = unsafe {
            context
                .instance()
                .get_physical_device_properties(context.physical_device())
                .limits
                .timestamp_period
        };

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

        let mut encoder = Self {
            context,
            config: config.clone(),
            dpb,
            gop,
            aligned_width,
            aligned_height,
            video_queue_fn,
            video_encode_fn,
            session,
            session_params: vk::VideoSessionParametersKHR::null(),
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
            timestamp_query_pool,
            gpu_timestamps: [0; 2],
            timestamp_period,
            header_data: None,
            has_backward_reference: false,
            backward_reference_poc: 0,
            backward_reference_dpb_slot: 2,
            current_dpb_slot: 0,
            l0_references: Vec::new(),
            active_reference_count: max_active_reference_pictures as u32,
            profile_idc,
            dpb_slot_active,
        };

        encoder.session_params = encoder.create_session_params(&color_desc)?;
        Ok(encoder)
    }
}
