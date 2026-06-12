//! H.264 encoder implementation using Vulkan Video.
//!
//! This module implements H.264 video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;
mod session_params;

use ash::vk;
use tracing::debug;

use crate::encoder::resources::{
    destroy_encoder_resources, upload_image_to_input, EncoderResources, UploadParams,
};
use crate::error::Result;

use crate::encoder::dpb::DecodedPictureBuffer;
use crate::encoder::gop::GopStructure;
use crate::encoder::EncodeConfig;
use crate::vulkan::VideoContext;

/// H.264 macroblock size in pixels.
pub const MB_SIZE: u32 = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub frame_num: u32,
    pub poc: i32,
}

/// H.264 encoder.
pub struct H264Encoder {
    context: VideoContext,
    config: EncodeConfig,
    dpb: DecodedPictureBuffer,
    gop: GopStructure,

    /// Aligned width (macroblock + granularity aligned).
    aligned_width: u32,
    /// Aligned height (macroblock + granularity aligned).
    aligned_height: u32,

    // Video session.
    video_queue_fn: ash::khr::video_queue::Device,
    video_encode_fn: ash::khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // Frame counters.
    input_frame_num: u64,
    encode_frame_num: u64,
    frame_num_syntax: u32,
    idr_pic_id: u32,

    // Resources
    input_image: vk::Image,
    input_image_memory: vk::DeviceMemory,
    input_image_view: vk::ImageView,
    /// Current Vulkan image layout of `input_image` (tracked to avoid UB when transitioning).
    input_image_layout: vk::ImageLayout,
    /// DPB images (up to MAX_DPB_SLOTS for B-frame and long-term reference support).
    dpb_images: Vec<vk::Image>,
    dpb_image_memories: Vec<vk::DeviceMemory>,
    dpb_image_views: Vec<vk::ImageView>,
    /// Number of DPB slots allocated.
    dpb_slot_count: usize,
    /// Whether the DPB uses a single layered image (true) or separate images (false).
    use_layered_dpb: bool,
    /// Tracks which DPB slots have been activated (used at least once).
    dpb_slot_active: Vec<bool>,
    bitstream_buffer: vk::Buffer,
    bitstream_buffer_memory: vk::DeviceMemory,
    /// Persistently mapped pointer to the bitstream buffer (avoids per-frame map/unmap).
    bitstream_buffer_ptr: *mut u8,

    // Command resources.
    command_pool: vk::CommandPool,
    upload_command_pool: vk::CommandPool,
    upload_command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,
    encode_command_buffer: vk::CommandBuffer,
    encode_fence: vk::Fence,
    query_pool: vk::QueryPool,

    // Timestamping resources
    timestamp_query_pool: vk::QueryPool,
    gpu_timestamps: [u64; 2], // start and end timestamps
    timestamp_period: f32,    // nanoseconds per timestamp unit

    // SPS/PPS written flag.
    sps_written: bool,

    // Reference picture tracking.
    /// Whether we have a backward reference (for B-frames, L1).
    has_backward_reference: bool,
    /// Frame number of the L1 (backward) reference picture (for B-frames).
    backward_reference_frame_num: u32,
    /// POC of the L1 (backward) reference picture.
    backward_reference_poc: i32,
    /// DPB slot for L1 (backward) reference.
    backward_reference_dpb_slot: u8,
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active L0 reference pictures (for P-frames). Ordered from most recent to oldest.
    l0_references: Vec<ReferenceInfo>,
    /// Number of active reference frames (as configured/negotiated).
    active_reference_count: u32,
    /// H.264 profile IDC (cached from initialization for session parameter recreation).
    profile_idc: u32,
    /// Whether CABAC entropy coding is preferred (cached from quality level query).
    preferred_entropy_cabac: bool,
}

impl H264Encoder {
    /// Upload input frame from a GPU image.
    ///
    /// This copies from a source NV12 image directly to the encoder's input image,
    /// avoiding any CPU-side data copies. The source image must be in NV12 format
    /// with the same dimensions as the encoder configuration. The source image
    /// should be in GENERAL layout.
    fn upload_from_image(&mut self, src_image: vk::Image) -> Result<()> {
        if src_image == self.input_image {
            debug!("Source image is the encoder's input image, skipping upload copy");
            return Ok(());
        }

        let params = UploadParams {
            upload_command_buffer: self.upload_command_buffer,
            upload_fence: self.upload_fence,
            src_image,
            dst_image: self.input_image,
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
            pixel_format: self.config.pixel_format,
            input_image_layout: self.input_image_layout,
            upload_queue: self.context.transfer_queue(),
        };

        upload_image_to_input(&self.context, &params)?;

        // Update tracked layout.
        self.input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;

        Ok(())
    }
}

// SAFETY: The raw pointer bitstream_buffer_ptr is only used within the encoder's
// thread and is properly synchronized via Vulkan fences before access
unsafe impl Send for H264Encoder {}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        unsafe {
            // Wait on the queues used by the encoder rather than stalling
            // the entire device.
            let _ = self
                .context
                .device()
                .queue_wait_idle(self.context.transfer_queue());
            if let Some(q) = self.context.video_encode_queue() {
                let _ = self.context.device().queue_wait_idle(q);
            }
            // TODO(dathorse): Move into destroy_encoder_resources as EncoderResources field..
            self.context
                .device()
                .destroy_query_pool(self.timestamp_query_pool, None);
            destroy_encoder_resources(
                self.context.device(),
                &self.video_queue_fn,
                &EncoderResources {
                    query_pool: self.query_pool,
                    upload_fence: self.upload_fence,
                    encode_fence: self.encode_fence,
                    command_pool: self.command_pool,
                    upload_command_pool: self.upload_command_pool,
                    bitstream_buffer: self.bitstream_buffer,
                    bitstream_buffer_memory: self.bitstream_buffer_memory,
                    input_image: self.input_image,
                    input_image_memory: self.input_image_memory,
                    input_image_view: self.input_image_view,
                    dpb_images: &self.dpb_images,
                    dpb_image_memories: &self.dpb_image_memories,
                    dpb_image_views: &self.dpb_image_views,
                    session: self.session,
                    session_params: self.session_params,
                    session_memory: &self.session_memory,
                },
            );
        }
    }
}
