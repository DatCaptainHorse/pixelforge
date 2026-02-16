//! AV1 encoder implementation using Vulkan Video.
//!
//! This module implements AV1 video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;

use ash::vk;
use tracing::debug;

use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::error::Result;

use crate::encoder::gop::GopStructure;
use crate::encoder::EncodeConfig;
use crate::vulkan::VideoContext;

/// Minimum bitstream buffer size.
const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

/// AV1 superblock size in pixels (128x128 for most cases).
pub const SUPERBLOCK_SIZE: u32 = 128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub order_hint: u32,
}

/// AV1 encoder.
pub struct AV1Encoder {
    context: VideoContext,
    config: EncodeConfig,
    gop: GopStructure,

    // Video session.
    video_queue_fn: ash::khr::video_queue::Device,
    video_encode_fn: ash::khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // Frame counters.
    input_frame_num: u64,
    encode_frame_num: u64,
    frame_num: u32,
    order_hint: u32,

    // Resources
    input_image: vk::Image,
    input_image_memory: vk::DeviceMemory,
    input_image_view: vk::ImageView,
    /// Current Vulkan image layout of `input_image` (tracked to avoid UB when transitioning).
    input_image_layout: vk::ImageLayout,
    /// DPB images for reference frames.
    dpb_images: Vec<vk::Image>,
    dpb_image_memories: Vec<vk::DeviceMemory>,
    dpb_image_views: Vec<vk::ImageView>,
    /// Number of DPB slots allocated.
    dpb_slot_count: usize,
    bitstream_buffer: vk::Buffer,
    bitstream_buffer_memory: vk::DeviceMemory,
    /// Persistently mapped pointer to the bitstream buffer (avoids per-frame map/unmap).
    bitstream_buffer_ptr: *mut u8,

    // Command resources.
    command_pool: vk::CommandPool,
    upload_command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,
    encode_command_buffer: vk::CommandBuffer,
    encode_fence: vk::Fence,
    query_pool: vk::QueryPool,

    // Cached AV1 sequence header OBU (retrieved from session parameters).
    header_data: Option<Vec<u8>>,

    // Reference picture tracking.
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active reference pictures. Ordered from most recent to oldest.
    references: Vec<ReferenceInfo>,
    /// Number of active reference frames (as configured/negotiated).
    active_reference_count: u32,
}

impl AV1Encoder {
    /// Upload input frame from a GPU image.
    ///
    /// This copies from a source image directly to the encoder's input image,
    /// avoiding any CPU-side data copies. The source image must match the
    /// encoder's configured pixel format and dimensions, and should be in
    /// GENERAL layout.
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
        };

        upload_image_to_input(&self.context, &params)?;

        // Update tracked layout.
        self.input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;

        Ok(())
    }
}

// SAFETY: The raw pointer bitstream_buffer_ptr is only used within the encoder's
// thread and is properly synchronized via Vulkan fences before access
unsafe impl Send for AV1Encoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superblock_size() {
        assert_eq!(SUPERBLOCK_SIZE, 128);
    }

    #[test]
    fn test_superblock_alignment() {
        // Dimensions should be aligned up to superblock boundaries.
        let align = |v: u32| (v + SUPERBLOCK_SIZE - 1) & !(SUPERBLOCK_SIZE - 1);

        assert_eq!(align(1920), 1920); // Already aligned
        assert_eq!(align(1080), 1152); // 1080 rounds up to 1152
        assert_eq!(align(2560), 2560); // Already aligned
        assert_eq!(align(1440), 1536); // 1440 rounds up to 1536
        assert_eq!(align(1), 128); // Minimum is one superblock
    }

    #[test]
    fn test_reference_info() {
        let ref_info = ReferenceInfo {
            dpb_slot: 2,
            order_hint: 42,
        };

        assert_eq!(ref_info.dpb_slot, 2);
        assert_eq!(ref_info.order_hint, 42);

        // Should be Copy + Clone.
        let copied = ref_info;
        assert_eq!(copied.dpb_slot, ref_info.dpb_slot);
        assert_eq!(copied.order_hint, ref_info.order_hint);
    }

    #[test]
    fn test_order_hint_wrapping() {
        // AV1 order hints are 8-bit, wrapping at 256.
        let mut order_hint: u32 = 254;
        for _ in 0..4 {
            order_hint = (order_hint + 1) & 0xFF;
        }
        // 254 -> 255 -> 0 -> 1 -> 2
        assert_eq!(order_hint, 2);
    }

    #[test]
    fn test_reference_tracking() {
        // Simulate building up a reference list like the encoder does.
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let max_refs = 4usize;

        for i in 0..6u8 {
            let ref_info = ReferenceInfo {
                dpb_slot: i % max_refs as u8,
                order_hint: i as u32,
            };
            references.insert(0, ref_info);
            while references.len() > max_refs {
                references.pop();
            }
        }

        // Should have exactly max_refs entries.
        assert_eq!(references.len(), max_refs);
        // Most recent should be first.
        assert_eq!(references[0].order_hint, 5);
        assert_eq!(references[max_refs - 1].order_hint, 2);
    }

    #[test]
    fn test_key_frame_clears_references() {
        // Simulate the encoder's key frame behavior: references.clear() on IDR.
        let mut references: Vec<ReferenceInfo> = Vec::new();

        // Build up some references (like a sequence of P-frames).
        for i in 0..3u8 {
            references.insert(
                0,
                ReferenceInfo {
                    dpb_slot: i,
                    order_hint: i as u32,
                },
            );
        }
        assert_eq!(references.len(), 3);

        // Key frame resets everything.
        references.clear();
        assert!(references.is_empty());

        // First frame after key should start fresh at slot 0.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
            },
        );
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].dpb_slot, 0);
        assert_eq!(references[0].order_hint, 0);
    }

    #[test]
    fn test_dpb_slot_reuse() {
        // Simulate the encoder's DPB slot allocation: after encoding a reference
        // frame, find the first slot not used by any active reference.
        let max_refs = 2usize;
        let dpb_slot_count = 3u8; // active_refs + 1
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let mut current_dpb_slot: u8 = 0;

        // Helper: find next free slot (mirrors api.rs logic).
        let find_free_slot = |refs: &[ReferenceInfo], slot_count: u8| -> u8 {
            let used: Vec<u8> = refs.iter().map(|r| r.dpb_slot).collect();
            for i in 0..slot_count {
                if !used.contains(&i) {
                    return i;
                }
            }
            0 // fallback (shouldn't happen with correct slot_count)
        };

        // Frame 0 (IDR): uses slot 0.
        assert_eq!(current_dpb_slot, 0);
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1); // slot 0 is used, next free is 1

        // Frame 1 (P): uses slot 1.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 1,
                order_hint: 1,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 2); // slots 0,1 used, next free is 2

        // Frame 2 (P): uses slot 2. Now all 3 slots have been touched,
        // but max_refs=2 means the oldest reference (slot 0) gets evicted.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 2,
                order_hint: 2,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        // references = [{slot:2, hint:2}, {slot:1, hint:1}] - slot 0 evicted
        assert_eq!(references.len(), 2);
        assert_eq!(references[0].dpb_slot, 2);
        assert_eq!(references[1].dpb_slot, 1);
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 0); // slot 0 is now free again (reuse!)

        // Frame 3 (P): uses recycled slot 0.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 3,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        // references = [{slot:0, hint:3}, {slot:2, hint:2}] - slot 1 evicted
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1); // slot 1 recycled
    }

    #[test]
    fn test_idr_p_p_idr_cycle() {
        // Full GOP cycle: IDR -> P -> P -> IDR, verifying DPB slot allocation
        // and reference list state at each step.
        let max_refs = 2usize;
        let dpb_slot_count = 3u8;
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let mut current_dpb_slot: u8 = 0;

        let find_free_slot = |refs: &[ReferenceInfo], slot_count: u8| -> u8 {
            let used: Vec<u8> = refs.iter().map(|r| r.dpb_slot).collect();
            for i in 0..slot_count {
                if !used.contains(&i) {
                    return i;
                }
            }
            0
        };

        // IDR frame: clears refs, writes to slot 0.
        references.clear();
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: current_dpb_slot,
                order_hint: 0,
            },
        );
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);

        assert_eq!(references.len(), 1);
        assert_eq!(references[0].order_hint, 0);
        assert_eq!(current_dpb_slot, 1);

        // P frame 1: writes to slot 1.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: current_dpb_slot,
                order_hint: 1,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);

        assert_eq!(references.len(), 2);
        assert_eq!(references[0].order_hint, 1);
        assert_eq!(current_dpb_slot, 2);

        // P frame 2: writes to slot 2, evicts oldest ref (slot 0).
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: current_dpb_slot,
                order_hint: 2,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);

        assert_eq!(references.len(), 2);
        assert_eq!(references[0].order_hint, 2);
        assert_eq!(current_dpb_slot, 0); // slot 0 recycled

        // Second IDR: everything resets.
        references.clear();
        assert!(references.is_empty());
    }

    #[test]
    fn test_single_reference_slot() {
        // Edge case: only 1 active reference with 2 DPB slots (minimum viable).
        let max_refs = 1usize;
        let dpb_slot_count = 2u8;
        let mut references: Vec<ReferenceInfo> = Vec::new();

        let find_free_slot = |refs: &[ReferenceInfo], slot_count: u8| -> u8 {
            let used: Vec<u8> = refs.iter().map(|r| r.dpb_slot).collect();
            for i in 0..slot_count {
                if !used.contains(&i) {
                    return i;
                }
            }
            0
        };

        // Frame 0: slot 0.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        let mut current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1);

        // Frame 1: slot 1. Old ref (slot 0) evicted since max_refs=1.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 1,
                order_hint: 1,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].dpb_slot, 1);
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 0); // ping-pong between 0 and 1

        // Frame 2: slot 0 again.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 2,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1); // ping-pong back
    }
}

impl Drop for AV1Encoder {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.device().device_wait_idle();
            self.context
                .device()
                .destroy_query_pool(self.query_pool, None);
            self.context.device().destroy_fence(self.upload_fence, None);
            self.context.device().destroy_fence(self.encode_fence, None);
            self.context
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.context
                .device()
                .destroy_buffer(self.bitstream_buffer, None);
            // Unmap the persistently mapped bitstream buffer before freeing memory.
            self.context
                .device()
                .unmap_memory(self.bitstream_buffer_memory);
            self.context
                .device()
                .free_memory(self.bitstream_buffer_memory, None);
            self.context
                .device()
                .destroy_image_view(self.input_image_view, None);
            self.context.device().destroy_image(self.input_image, None);
            self.context
                .device()
                .free_memory(self.input_image_memory, None);

            for i in 0..self.dpb_images.len() {
                self.context
                    .device()
                    .destroy_image_view(self.dpb_image_views[i], None);
                self.context
                    .device()
                    .destroy_image(self.dpb_images[i], None);
                self.context
                    .device()
                    .free_memory(self.dpb_image_memories[i], None);
            }

            self.video_queue_fn
                .destroy_video_session_parameters(self.session_params, None);
            self.video_queue_fn
                .destroy_video_session(self.session, None);
            for mem in &self.session_memory {
                self.context.device().free_memory(*mem, None);
            }
        }
    }
}
