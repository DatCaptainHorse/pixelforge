use super::{Av1, MIN_BITSTREAM_BUFFER_SIZE, SUPERBLOCK_SIZE};

use crate::encoder::codec::{
    build_encoder_common, query_video_caps, CodecEncoder, CommonInitRequest,
};
use crate::encoder::{ColorDescription, EncodeConfig, PixelFormat};
use crate::error::Result;
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

impl Av1 {
    /// Create a new AV1 encoder.
    pub fn create(context: VideoContext, config: EncodeConfig) -> Result<CodecEncoder<Self>> {
        assert!(
            config.b_frame_count == 0,
            "B-frame encoding is not yet supported; set b_frame_count=0 (got {})",
            config.b_frame_count
        );

        info!(
            "Creating AV1 encoder: {}x{}, pixel_format={:?}",
            config.dimensions.width, config.dimensions.height, config.pixel_format
        );

        // AV1 profile from chroma subsampling: Main is 4:2:0, High adds 4:4:4.
        let profile = match config.pixel_format {
            PixelFormat::Yuv420 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
            _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        };

        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();
        let bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();

        let mut av1_profile_info = vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(profile);
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth)
            .chroma_bit_depth(bit_depth)
            .push(&mut av1_profile_info);

        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE
            .max(config.dimensions.width as usize * config.dimensions.height as usize);

        // Query device capabilities with the AV1-specific capability struct
        // chained in (required by the driver).
        let mut av1_caps = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut av1_caps);
        let caps = query_video_caps(&context, &profile_info, &mut capabilities)?;

        let init = build_encoder_common(&CommonInitRequest {
            context: &context,
            config: &config,
            profile_info: &profile_info,
            caps: &caps,
            align_unit: SUPERBLOCK_SIZE,
            max_active_refs_cap: 8,
            bitstream_buffer_size,
            // AV1 reference handling here does not use a layered DPB.
            allow_layered_dpb: false,
        })?;
        let common = init.common;

        let codec = Av1 {
            frame_num: 0,
            order_hint: 0,
            header_data: None,
            references: Vec::new(),
        };

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("AV1 encoder created successfully");
        Ok(encoder)
    }
}
