//! Getting decoded pictures out of the decoder: host readback, the reorder-pool
//! copy, and plane copies into caller-owned images.
//!
//! These all run on the transfer queue rather than the decode queue, because a
//! dedicated video decode queue family need not advertise `TRANSFER_BIT` (it
//! does not on RADV), so a copy recorded there would be invalid.

use ash::vk;

use crate::decoder::common::DecoderCommon;
use crate::decoder::frames::DecodedPicture;
use crate::decoder::{DecodedFrame, DecodedFrameData};
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::find_memory_type;

impl DecoderCommon {
    /// Ensure the readback buffer holds at least `size` bytes.
    ///
    /// The stored capacity is the *allocated* size (which the driver may round
    /// up), so that `download` can map the whole allocation rather than a
    /// sub-range that might violate `nonCoherentAtomSize`.
    fn ensure_readback_capacity(&mut self, size: usize) -> Result<()> {
        if let Some((_, _, capacity)) = self.readback
            && capacity >= size
        {
            return Ok(());
        }
        if let Some((buffer, memory, _)) = self.readback.take() {
            unsafe {
                self.context.device().device_wait_idle().ok();
                self.context.device().destroy_buffer(buffer, None);
                self.context.device().free_memory(memory, None);
            }
        }

        let create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.context.device().create_buffer(&create_info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("readback buffer: {}", e)))?;
        let reqs = unsafe { self.context.device().get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            self.context.memory_properties(),
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| {
            PixelForgeError::MemoryAllocation("No host-visible memory for readback".to_string())
        })?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { self.context.device().allocate_memory(&alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;
        unsafe { self.context.device().bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        // Record the allocated size, not the requested one.
        self.readback = Some((buffer, memory, reqs.size as usize));
        Ok(())
    }

    /// Wait for in-flight decodes when a caller-driven copy would otherwise race
    /// them.
    ///
    /// Reading a frame means transitioning its image out of its current layout
    /// and back. That is harmless for a pool image, which nothing else touches,
    /// but a zero-copy frame's image is a live DPB slot: later pictures may
    /// still be reading it as a reference, and moving its layout underneath them
    /// corrupts their output. These copies submit on the transfer queue with no
    /// dependency on the decode queue, so the only correct answer is to let the
    /// decode queue drain first.
    fn wait_before_reading(&self, frame: &DecodedFrame) {
        if frame
            .pin
            .as_ref()
            .is_some_and(|pin| pin.borrows_dpb_image())
        {
            self.pipeline.wait_all_free();
        }
    }

    /// Copy a decoded picture back to the host, cropped to its visible region.
    ///
    /// Entirely codec-independent: it copies planes out of an image the decode
    /// queue has already finished writing.
    ///
    /// The copy runs on the transfer queue. The video decode queue family is a
    /// dedicated engine that need not advertise `TRANSFER_BIT` (it does not on
    /// RADV), so recording a copy there is invalid.
    pub fn download(&mut self, frame: &DecodedFrame) -> Result<DecodedFrameData> {
        self.wait_before_reading(frame);
        let (bit_depth, pixel_format) = {
            let session = self.session()?;
            (session.bit_depth, session.pixel_format)
        };

        let geom = PlaneGeometry::new(
            frame.coded_width,
            frame.coded_height,
            frame.width,
            frame.height,
            bit_depth,
            pixel_format,
        );
        let PlaneGeometry {
            chroma_div,
            y_stride,
            y_size,
            uv_stride,
            total,
            ..
        } = geom;

        self.ensure_readback_capacity(total)?;
        let (buffer, memory, allocated) = self.readback.expect("ensured above");

        let device = self.context.device().clone();
        unsafe {
            device
                .reset_command_buffer(
                    self.transfer_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    self.transfer_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let base_layer = frame.array_layer;

        // A multi-planar image must be transitioned with the plane aspects that
        // the copy will use; a COLOR barrier does not cover PLANE_0/PLANE_1 and
        // leaves the planes in an undefined layout.
        let copy_aspects = vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1;

        // Recorded on the transfer queue, so no VIDEO_DECODE_* stage or access
        // may appear here. The decode is already fence-waited, so the writes are
        // visible and NONE is a correct source scope.
        let to_src = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(frame.layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: copy_aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: base_layer,
                layer_count: 1,
            });
        let barriers = [to_src];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dependency) };

        let regions = [
            vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(frame.coded_width)
                .buffer_image_height(frame.coded_height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: base_layer,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: frame.coded_width,
                    height: frame.coded_height,
                    depth: 1,
                }),
            vk::BufferImageCopy2::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(frame.coded_width / chroma_div as u32)
                .buffer_image_height(frame.coded_height / chroma_div as u32)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: base_layer,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: frame.coded_width / chroma_div as u32,
                    height: frame.coded_height / chroma_div as u32,
                    depth: 1,
                }),
        ];
        let copy_info = vk::CopyImageToBufferInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_buffer(buffer)
            .regions(&regions);
        unsafe { device.cmd_copy_image_to_buffer2(self.transfer_command_buffer, &copy_info) };

        // Restore the layout so a DPB image stays usable as a reference. (For a
        // separate output image this is redundant but harmless: the next decode
        // re-transitions it from UNDEFINED.)
        let restore = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(frame.layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: copy_aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: base_layer,
                layer_count: 1,
            });
        let barriers = [restore];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dependency) };

        unsafe {
            device
                .end_command_buffer(self.transfer_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let command_buffers = [self.transfer_command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .reset_fences(&[self.transfer_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .queue_submit(self.transfer_queue, &[submit], self.transfer_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .wait_for_fences(&[self.transfer_fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }

        // Read out, cropping to the visible region.
        // Map the whole allocation: a sub-range map must respect
        // nonCoherentAtomSize, and WHOLE_SIZE sidesteps that entirely.
        let ptr =
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?
                as *const u8;
        debug_assert!(
            allocated >= geom.max_read_offset(),
            "readback buffer too small: host reads up to {} but only {} bytes are mapped",
            geom.max_read_offset(),
            allocated
        );

        let visible_y_stride = geom.visible_y_stride;
        let visible_uv_stride = geom.visible_uv_stride;
        let mut y = Vec::with_capacity(visible_y_stride * geom.visible_y_rows);
        let mut uv = Vec::with_capacity(visible_uv_stride * geom.visible_uv_rows);
        unsafe {
            for row in 0..geom.visible_y_rows {
                let src = ptr.add(row * y_stride);
                y.extend_from_slice(std::slice::from_raw_parts(src, visible_y_stride));
            }
            for row in 0..geom.visible_uv_rows {
                let src = ptr.add(y_size + row * uv_stride);
                uv.extend_from_slice(std::slice::from_raw_parts(src, visible_uv_stride));
            }
            device.unmap_memory(memory);
        }

        Ok(DecodedFrameData {
            y,
            uv,
            y_stride: visible_y_stride,
            uv_stride: visible_uv_stride,
            width: frame.width,
            height: frame.height,
            bit_depth,
            pixel_format,
        })
    }

    /// Copy a freshly decoded picture into a pool image, so it survives past the
    /// next decode submission. This is what lets a [`DecodedFrame`] outlive the
    /// DPB slot the picture was decoded into.
    ///
    /// Records only; [`Self::submit_copy`] submits it. The copy runs on the
    /// transfer queue, because a dedicated video decode queue need not support
    /// transfer operations, and waits on the decode's timeline value rather than
    /// a fence. A semaphore wait makes the decode's writes available and
    /// visible, so `NONE` remains a correct source scope, and the DPB image is
    /// created with concurrent sharing between the two families so no ownership
    /// transfer is needed.
    ///
    /// The source's layout is restored afterward so a DPB image stays usable as
    /// a reference; the destination is left in `TRANSFER_DST_OPTIMAL` (reported
    /// as the pooled frame's layout, which `download` then transitions from).
    pub fn record_picture_copy(&self, frame: &DecodedPicture, dst_image: vk::Image) -> Result<()> {
        let command_buffer = self.pipeline.current().transfer_command_buffer;
        let base_layer = frame.array_layer;
        let device = self.context.device().clone();
        let aspects = vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1;
        let range = |image: vk::Image, layer: u32| vk::ImageMemoryBarrier2 {
            image,
            ..vk::ImageMemoryBarrier2::default()
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspects,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        };

        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let to_transfer = [
            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::COPY,
                dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
                old_layout: frame.layout,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ..range(frame.image, base_layer)
            },
            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::COPY,
                dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ..range(dst_image, 0)
            },
        ];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep) };

        let plane = |aspect: vk::ImageAspectFlags| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let (cdiv_hor, cdiv_vert) = frame.pixel_format.chroma_div();
        let regions = [
            vk::ImageCopy2::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    base_array_layer: base_layer,
                    ..plane(vk::ImageAspectFlags::PLANE_0)
                })
                .dst_subresource(plane(vk::ImageAspectFlags::PLANE_0))
                .extent(vk::Extent3D {
                    width: frame.coded_width,
                    height: frame.coded_height,
                    depth: 1,
                }),
            vk::ImageCopy2::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    base_array_layer: base_layer,
                    ..plane(vk::ImageAspectFlags::PLANE_1)
                })
                .dst_subresource(plane(vk::ImageAspectFlags::PLANE_1))
                .extent(vk::Extent3D {
                    width: frame.coded_width / cdiv_hor,
                    height: frame.coded_height / cdiv_vert,
                    depth: 1,
                }),
        ];
        let copy = vk::CopyImageInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(dst_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&regions);
        unsafe { device.cmd_copy_image2(command_buffer, &copy) };

        // Restore the source layout so a DPB image stays a valid reference.
        let restore = [vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COPY,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
            dst_access_mask: vk::AccessFlags2::NONE,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: frame.layout,
            ..range(frame.image, base_layer)
        }];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&restore);
        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep) };

        Ok(())
    }

    /// Submit the recorded reorder copy without waiting for it.
    pub fn submit_copy(&mut self) -> Result<()> {
        let device = self.context.device().clone();
        let queue = self.transfer_queue;
        self.pipeline.submit_copy(&device, queue)
    }

    /// Copy a decoded picture's two planes into two single-plane caller images:
    /// luma into `y_image` (an R8/R16 image) and interleaved chroma into
    /// `uv_image` (an R8G8/R16G16 image), each in the plane's own resolution.
    ///
    /// This suits consumers that sample Y and UV as separate textures (the
    /// common YUV→RGB shader layout) without a sampler-YCbCr conversion. Like
    /// [`Self::copy_frame_to_image`] it runs on the transfer queue, restores the
    /// source layout, and leaves both destinations in `TRANSFER_DST_OPTIMAL`.
    ///
    /// `y_image` and `uv_image` must live on this context's device and be sized
    /// to the frame's coded dimensions (chroma at half size for 4:2:0).
    pub fn copy_frame_to_planes(
        &self,
        frame: &DecodedFrame,
        y_image: vk::Image,
        uv_image: vk::Image,
    ) -> Result<()> {
        self.wait_before_reading(frame);
        let base_layer = frame.array_layer;
        let device = self.context.device().clone();

        let color = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let src_planes = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: base_layer,
            layer_count: 1,
        };

        unsafe {
            device
                .reset_command_buffer(
                    self.transfer_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    self.transfer_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let qfi = vk::QUEUE_FAMILY_IGNORED;
        let to_transfer = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(frame.layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(qfi)
                .dst_queue_family_index(qfi)
                .image(frame.image)
                .subresource_range(src_planes),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(qfi)
                .dst_queue_family_index(qfi)
                .image(y_image)
                .subresource_range(color),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(qfi)
                .dst_queue_family_index(qfi)
                .image(uv_image)
                .subresource_range(color),
        ];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dep) };

        let color_layers = |aspect: vk::ImageAspectFlags, layer: u32| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: layer,
            layer_count: 1,
        };
        // Luma: source plane 0 -> y_image, full resolution.
        let y_region = vk::ImageCopy2::default()
            .src_subresource(color_layers(vk::ImageAspectFlags::PLANE_0, base_layer))
            .dst_subresource(color_layers(vk::ImageAspectFlags::COLOR, 0))
            .extent(vk::Extent3D {
                width: frame.coded_width,
                height: frame.coded_height,
                depth: 1,
            });
        let y_copy = vk::CopyImageInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(y_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(std::slice::from_ref(&y_region));
        // Chroma: source plane 1 -> uv_image, at chroma resolution.
        let (cdiv_hor, cdiv_vert) = frame.pixel_format.chroma_div();
        let uv_region = vk::ImageCopy2::default()
            .src_subresource(color_layers(vk::ImageAspectFlags::PLANE_1, base_layer))
            .dst_subresource(color_layers(vk::ImageAspectFlags::COLOR, 0))
            .extent(vk::Extent3D {
                width: frame.coded_width / cdiv_hor,
                height: frame.coded_height / cdiv_vert,
                depth: 1,
            });
        let uv_copy = vk::CopyImageInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(uv_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(std::slice::from_ref(&uv_region));
        unsafe {
            device.cmd_copy_image2(self.transfer_command_buffer, &y_copy);
            device.cmd_copy_image2(self.transfer_command_buffer, &uv_copy);
        }

        let restore = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(frame.layout)
            .src_queue_family_index(qfi)
            .dst_queue_family_index(qfi)
            .image(frame.image)
            .subresource_range(src_planes)];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&restore);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dep) };

        unsafe {
            device
                .end_command_buffer(self.transfer_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let command_buffers = [self.transfer_command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .reset_fences(&[self.transfer_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .queue_submit(self.transfer_queue, &[submit], self.transfer_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .wait_for_fences(&[self.transfer_fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }
        Ok(())
    }
}

/// Byte layout of a decoded picture in the readback buffer, and of the visible
/// region the host actually copies out.
///
/// The decoded image is `coded_width` x `coded_height` (macroblock-aligned),
/// but only `width` x `height` is displayed, so the host reads a cropped
/// sub-rectangle out of a larger buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneGeometry {
    /// 1 for 4:4:4, 2 for 4:2:0.
    pub chroma_div: usize,
    pub y_stride: usize,
    pub y_size: usize,
    pub uv_stride: usize,
    pub total: usize,
    pub visible_y_stride: usize,
    pub visible_uv_stride: usize,
    pub visible_y_rows: usize,
    pub visible_uv_rows: usize,
}

impl PlaneGeometry {
    pub fn new(
        coded_width: u32,
        coded_height: u32,
        width: u32,
        height: u32,
        bit_depth: BitDepth,
        pixel_format: PixelFormat,
    ) -> Self {
        let bytes_per_sample = match bit_depth {
            BitDepth::Eight => 1,
            BitDepth::Ten => 2,
        };
        let chroma_div = match pixel_format {
            PixelFormat::Yuv444 => 1,
            _ => 2,
        };
        let y_stride = coded_width as usize * bytes_per_sample;
        let y_size = y_stride * coded_height as usize;
        let uv_stride = (coded_width as usize / chroma_div) * 2 * bytes_per_sample;
        let uv_size = uv_stride * (coded_height as usize / chroma_div);
        Self {
            chroma_div,
            y_stride,
            y_size,
            uv_stride,
            total: y_size + uv_size,
            visible_y_stride: width as usize * bytes_per_sample,
            visible_uv_stride: (width as usize / chroma_div) * 2 * bytes_per_sample,
            visible_y_rows: height as usize,
            visible_uv_rows: height as usize / chroma_div,
        }
    }

    /// Byte offset one past the last byte the host reads. Must not exceed
    /// [`Self::total`], or `download` would read out of bounds.
    pub fn max_read_offset(&self) -> usize {
        let last_y = if self.visible_y_rows == 0 {
            0
        } else {
            (self.visible_y_rows - 1) * self.y_stride + self.visible_y_stride
        };
        let last_uv = if self.visible_uv_rows == 0 {
            0
        } else {
            self.y_size + (self.visible_uv_rows - 1) * self.uv_stride + self.visible_uv_stride
        };
        last_y.max(last_uv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_geometry_reads_stay_in_bounds() {
        // (coded_width, coded_height, width, height)
        let cases = [
            (1920, 1088, 1920, 1080),
            (320, 240, 320, 240),
            (1280, 720, 1280, 720),
            (3840, 2160, 3840, 2160),
            (640, 480, 636, 476),
        ];
        for (cw, ch, w, h) in cases {
            for bd in [BitDepth::Eight, BitDepth::Ten] {
                for pf in [PixelFormat::Yuv420, PixelFormat::Yuv444] {
                    let g = PlaneGeometry::new(cw, ch, w, h, bd, pf);
                    assert!(
                        g.max_read_offset() <= g.total,
                        "{cw}x{ch} -> {w}x{h} {bd:?} {pf:?}: reads {} bytes but buffer is {}",
                        g.max_read_offset(),
                        g.total
                    );
                }
            }
        }
    }

    #[test]
    fn plane_geometry_nv12_1080p() {
        let g = PlaneGeometry::new(1920, 1088, 1920, 1080, BitDepth::Eight, PixelFormat::Yuv420);
        assert_eq!(g.y_stride, 1920);
        assert_eq!(g.y_size, 1920 * 1088);
        assert_eq!(g.uv_stride, 1920);
        assert_eq!(g.total, 1920 * 1088 + 1920 * 544);
        assert_eq!(g.visible_y_rows, 1080);
        assert_eq!(g.visible_uv_rows, 540);
    }

    #[test]
    fn plane_geometry_p010_is_two_bytes_per_sample() {
        let g = PlaneGeometry::new(1920, 1088, 1920, 1080, BitDepth::Ten, PixelFormat::Yuv420);
        assert_eq!(g.y_stride, 1920 * 2);
        assert_eq!(g.visible_y_stride, 1920 * 2);
    }

    #[test]
    fn plane_geometry_yuv444_chroma_is_full_size() {
        let g = PlaneGeometry::new(320, 240, 320, 240, BitDepth::Eight, PixelFormat::Yuv444);
        assert_eq!(g.chroma_div, 1);
        assert_eq!(g.uv_stride, 320 * 2);
        assert_eq!(g.visible_uv_rows, 240);
    }
}
