use super::{AV1Encoder, MIN_BITSTREAM_BUFFER_SIZE};

use crate::encoder::gop::GopPosition;
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl AV1Encoder {
    pub(super) fn encode_frame_internal(
        &mut self,
        gop_position: &GopPosition,
        is_key_frame: bool,
    ) -> Result<Vec<u8>> {
        let is_reference = gop_position.is_reference;

        debug!(
            "encode_frame_internal: key={}, ref={}, refs_len={}, dpb_slot={}",
            is_key_frame,
            is_reference,
            self.references.len(),
            self.current_dpb_slot
        );

        // Rate control setup.
        let (rc_mode, average_bitrate, max_bitrate, _qp) = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                100_000_000, // 100 Mbps
                100_000_000,
                self.config.quality_level as i32,
            ),
            crate::encoder::RateControlMode::Cbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                self.config.target_bitrate,
                self.config.target_bitrate,
                128, // Default QP for AV1 to adjust from
            ),
            crate::encoder::RateControlMode::Vbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                self.config.target_bitrate,
                self.config.max_bitrate,
                128, // Default QP for AV1 to adjust from
            ),
        };

        // Reset command buffer.
        unsafe {
            self.context.device().reset_command_buffer(
                self.encode_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Begin command buffer.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device()
                .begin_command_buffer(self.encode_command_buffer, &begin_info)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Reset query pool (1 query for bitstream feedback).
        unsafe {
            self.context.device().cmd_reset_query_pool(
                self.encode_command_buffer,
                self.query_pool,
                0,
                1,
            );
        }

        // Transition DPB image to video encode DPB layout.
        let dpb_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.dpb_images[self.current_dpb_slot as usize])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::empty());

        unsafe {
            self.context.device().cmd_pipeline_barrier(
                self.encode_command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[dpb_barrier],
            );
        }

        // AV1 frame type.
        let frame_type = if is_key_frame {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        };

        // Build picture info flags using ash's accessor methods.
        // show_frame must be set for all frames; error_resilient_mode for key frames (match FFmpeg).
        let mut picture_info_flags = ash::vk::native::StdVideoEncodeAV1PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        picture_info_flags.set_show_frame(1);
        if is_key_frame {
            picture_info_flags.set_error_resilient_mode(1);
        }

        // Frame extent uses display dimensions (not superblock-aligned coded extent).
        // The video session's max_coded_extent is the upper bound; individual frames
        // use actual dimensions so the decoder doesn't show a green border.
        let frame_extent = vk::Extent2D {
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
        };

        // Setup reconstructed picture (DPB slot for output).
        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(self.dpb_image_views[self.current_dpb_slot as usize]);
        // AV1 reference info for the setup slot.
        let reference_info_flags = ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(
                0, 0, 0,
            ),
        };

        let std_reference_info = ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
            flags: reference_info_flags,
            frame_type: if is_key_frame {
                ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
            } else {
                ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
            },
            RefFrameId: self.current_dpb_slot as u32,
            OrderHint: self.order_hint as u8,
            reserved1: [0; 3],
            pExtensionHeader: std::ptr::null(),
        };

        // AV1 DPB slot info for the setup reference slot (the slot being written).
        let setup_av1_dpb_info =
            vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource);

        // Attach AV1 DPB info to setup slot's pNext chain.
        setup_reference_slot.p_next =
            (&setup_av1_dpb_info as *const vk::VideoEncodeAV1DpbSlotInfoKHR).cast();

        // Reference frames for inter frames.
        let mut reference_slots = Vec::new();
        let mut av1_reference_infos = Vec::new();
        let mut ref_picture_resources = Vec::new();
        let mut ref_std_infos = Vec::new(); // Store std info to keep it alive

        if !is_key_frame && !self.references.is_empty() {
            // Use the most recent reference frame.
            let ref_info = &self.references[0];

            // Create StdVideoEncodeAV1ReferenceInfo for the reference slot.
            let ref_std_info = ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
                flags: reference_info_flags,
                frame_type: ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER,
                RefFrameId: ref_info.dpb_slot as u32,
                OrderHint: ref_info.order_hint as u8,
                reserved1: [0; 3],
                pExtensionHeader: std::ptr::null(),
            };
            ref_std_infos.push(ref_std_info);

            // Create AV1 DPB slot info for the reference (without pointer first).
            let av1_ref_info = vk::VideoEncodeAV1DpbSlotInfoKHR::default();
            av1_reference_infos.push(av1_ref_info);
            // Now set the pointer after it's in the vector at its final location.
            av1_reference_infos[0] = av1_reference_infos[0].std_reference_info(&ref_std_infos[0]);

            let ref_picture_resource = vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(frame_extent)
                .base_array_layer(0)
                .image_view_binding(self.dpb_image_views[ref_info.dpb_slot as usize]);
            ref_picture_resources.push(ref_picture_resource);

            // Create reference slot (without pNext first).
            let ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(ref_info.dpb_slot as i32)
                .picture_resource(&ref_picture_resources[0]);
            reference_slots.push(ref_slot);
            // Now set pNext after it's in the vector at its final location.
            reference_slots[0].p_next =
                (&av1_reference_infos[0] as *const vk::VideoEncodeAV1DpbSlotInfoKHR).cast();
        }

        // AV1 quantization parameters - required structure.
        // Start with a moderate QP that the rate controller can adjust.
        let quantization_flags = ash::vk::native::StdVideoAV1QuantizationFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1QuantizationFlags::new_bitfield_1(
                0, // using_qmatrix
                0, // diff_uv_delta
                0, // reserved
            ),
        };

        let quantization = ash::vk::native::StdVideoAV1Quantization {
            flags: quantization_flags,
            base_q_idx: 128, // Moderate QP (0-255 range)
            DeltaQYDc: 0,
            DeltaQUDc: 0,
            DeltaQUAc: 0,
            DeltaQVDc: 0,
            DeltaQVAc: 0,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        };

        // CDEF (Constrained Directional Enhancement Filter) - required since we enabled it in sequence header.
        // Match FFmpeg's default initialization (all zeros).
        let cdef = ash::vk::native::StdVideoAV1CDEF {
            cdef_damping_minus_3: 0,                       // Match FFmpeg: damping = 3
            cdef_bits: 0,                                  // 1 CDEF strength combination (2^0)
            cdef_y_pri_strength: [0, 0, 0, 0, 0, 0, 0, 0], // Match FFmpeg: all zeros
            cdef_y_sec_strength: [0, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_pri_strength: [0, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_sec_strength: [0, 0, 0, 0, 0, 0, 0, 0],
        };

        // Loop filter - deblocking filter parameters.
        // Match FFmpeg's default initialization.
        let loop_filter_flags = ash::vk::native::StdVideoAV1LoopFilterFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1LoopFilterFlags::new_bitfield_1(
                0, // loop_filter_delta_enabled
                0, // loop_filter_delta_update
                0, // reserved
            ),
        };

        let loop_filter = ash::vk::native::StdVideoAV1LoopFilter {
            flags: loop_filter_flags,
            loop_filter_level: [0, 0, 0, 0], // Match FFmpeg: disable filter initially
            loop_filter_sharpness: 0,
            update_ref_delta: 0,
            // Match FFmpeg's default_loop_filter_ref_deltas: { 1, 0, 0, 0, -1, 0, -1, -1 }
            loop_filter_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            update_mode_delta: 1, // Match FFmpeg: set to 1
            loop_filter_mode_deltas: [0; 2],
        };

        // Tile info - FFmpeg has this commented out with "TODO FIX" at line 340.
        // Match FFmpeg: don't provide tile info (set to null).
        // Note: If this causes issues, we may need to re-enable it with proper values.

        // Build ref_frame_idx, ref_order_hint, and primary_ref_frame based on frame type.
        // Match FFmpeg: for P frames all 7 ref_frame_idx entries point to same reference slot.
        let (ref_frame_idx, ref_order_hint, primary_ref_frame) =
            if !is_key_frame && !self.references.is_empty() {
                let ref_info = &self.references[0];
                let ref_idx = [ref_info.dpb_slot as i8; 7];
                let mut order_hints = [0u8; 8];
                order_hints[ref_info.dpb_slot as usize] = ref_info.order_hint as u8;
                // primary_ref_frame is a reference NAME index (0=LAST_FRAME), not a DPB slot.
                // We use SINGLE_REFERENCE with LAST_FRAME, so always 0.
                (ref_idx, order_hints, 0u8)
            } else {
                ([0i8; 7], [0u8; 8], 7u8) // 7 = PRIMARY_REF_NONE for key frames
            };

        // refresh_frame_flags: key frames refresh ALL slots, P frames refresh current slot.
        let refresh_frame_flags = if is_key_frame {
            0xFF // Key frames refresh all 8 reference slots (match FFmpeg)
        } else if is_reference {
            1u8 << self.current_dpb_slot // P frames refresh current slot
        } else {
            0x00 // Non-reference frames don't refresh
        };

        // AV1 encode picture info.
        let std_picture_info = ash::vk::native::StdVideoEncodeAV1PictureInfo {
            flags: picture_info_flags,
            frame_type,
            frame_presentation_time: self.frame_num,
            current_frame_id: self.current_dpb_slot as u32, // Match FFmpeg: slot index
            order_hint: self.order_hint as u8,
            primary_ref_frame,
            refresh_frame_flags,
            coded_denom: 0,
            render_width_minus_1: (self.config.dimensions.width - 1) as u16,
            render_height_minus_1: (self.config.dimensions.height - 1) as u16,
            interpolation_filter: ash::vk::native::StdVideoAV1InterpolationFilter_STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
            TxMode: ash::vk::native::StdVideoAV1TxMode_STD_VIDEO_AV1_TX_MODE_SELECT,
            delta_q_res: 0,
            delta_lf_res: 0,
            ref_order_hint,
            ref_frame_idx,
            reserved1: [0; 3],
            delta_frame_id_minus_1: [0; 7],
            pTileInfo: std::ptr::null(), // Match FFmpeg: null (not implemented)
            pQuantization: &quantization,
            pSegmentation: std::ptr::null(),
            pLoopFilter: &loop_filter,
            pCDEF: &cdef,
            pLoopRestoration: std::ptr::null(),
            pGlobalMotion: std::ptr::null(),
            pExtensionHeader: std::ptr::null(),
            pBufferRemovalTimes: std::ptr::null(),
        };

        // Reference name slot indices - maps AV1 reference frame names to DPB slot indices.
        // For SINGLE_REFERENCE mode, we use LAST_FRAME (index 0).
        let mut reference_name_slot_indices = [-1i32; 7];

        // For inter frames, map LAST_FRAME to our reference DPB slot.
        if !is_key_frame && !self.references.is_empty() {
            let ref_info = &self.references[0];
            reference_name_slot_indices[0] = ref_info.dpb_slot as i32;
        }

        // Set prediction mode and rate control group based on frame type.
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

        let av1_picture_info = vk::VideoEncodeAV1PictureInfoKHR::default()
            .std_picture_info(&std_picture_info)
            .prediction_mode(prediction_mode)
            .rate_control_group(rate_control_group)
            .reference_name_slot_indices(reference_name_slot_indices);

        // Rate control layer info.
        let rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(average_bitrate as u64)
            .max_bitrate(max_bitrate as u64)
            .frame_rate_numerator(self.config.frame_rate_numerator)
            .frame_rate_denominator(self.config.frame_rate_denominator);

        // Rate control info.
        let rc_info = vk::VideoEncodeRateControlInfoKHR::default()
            .rate_control_mode(rc_mode)
            .virtual_buffer_size_in_ms(1000) // 1 second VBV buffer
            .layers(std::slice::from_ref(&rc_layer_info));

        // Video begin coding info.
        // FFmpeg trick: Include the setup slot in reference_slots but with slotIndex = -1.
        // This binds the picture resource without requiring the slot to be active yet.
        let mut all_reference_slots = Vec::new();

        if is_reference {
            // Create a copy of setup_reference_slot with slotIndex = -1
            let mut setup_slot_for_binding = setup_reference_slot;
            setup_slot_for_binding.slot_index = -1; // Magic value to bind resource without activation
            all_reference_slots.push(setup_slot_for_binding);
        }

        // Add reference slots (already active slots we're reading from)
        all_reference_slots.extend_from_slice(&reference_slots);

        // Begin video coding with rate control info on first frame only (matching HEVC pattern).
        let is_first_frame = self.encode_frame_num == 0;
        let mut begin_coding_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
        } else {
            let mut info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params);
            info.p_next = (&rc_info as *const vk::VideoEncodeRateControlInfoKHR).cast();
            info
        };

        if !all_reference_slots.is_empty() {
            begin_coding_info = begin_coding_info.reference_slots(&all_reference_slots);
        }

        unsafe {
            self.video_queue_fn
                .cmd_begin_video_coding(self.encode_command_buffer, &begin_coding_info);
        }

        // Initialize video session on first frame.
        if is_first_frame {
            let reset_control_info = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            unsafe {
                self.video_queue_fn
                    .cmd_control_video_coding(self.encode_command_buffer, &reset_control_info);
            }

            // Set rate control after reset (matching HEVC pattern).
            let mut rate_control = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL);
            rate_control.p_next = (&rc_info as *const vk::VideoEncodeRateControlInfoKHR).cast();
            unsafe {
                self.video_queue_fn
                    .cmd_control_video_coding(self.encode_command_buffer, &rate_control);
            }
        }

        // Encode info.
        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(self.input_image_view);

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .src_picture_resource(src_picture_resource)
            .dst_buffer(self.bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(MIN_BITSTREAM_BUFFER_SIZE as u64);

        if is_reference {
            encode_info = encode_info.setup_reference_slot(&setup_reference_slot);
        }

        if !reference_slots.is_empty() {
            encode_info = encode_info.reference_slots(&reference_slots);
        }

        encode_info.p_next = (&av1_picture_info as *const vk::VideoEncodeAV1PictureInfoKHR).cast();

        // Begin query to capture encode feedback (bitstream size, status).
        unsafe {
            self.context.device().cmd_begin_query(
                self.encode_command_buffer,
                self.query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
        }

        unsafe {
            self.video_encode_fn
                .cmd_encode_video(self.encode_command_buffer, &encode_info);
        }

        // End query.
        unsafe {
            self.context
                .device()
                .cmd_end_query(self.encode_command_buffer, self.query_pool, 0);
        }

        // End video coding.
        let end_coding_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            self.video_queue_fn
                .cmd_end_video_coding(self.encode_command_buffer, &end_coding_info);
        }

        // End command buffer.
        unsafe {
            self.context
                .device()
                .end_command_buffer(self.encode_command_buffer)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Submit command buffer.
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.encode_command_buffer));

        unsafe { self.context.device().reset_fences(&[self.encode_fence]) }
            .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;

        unsafe {
            self.context.device().queue_submit(
                self.context
                    .video_encode_queue()
                    .expect("Video encode queue should be available"),
                &[submit_info],
                self.encode_fence,
            )
        }
        .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;

        // Wait for encode to complete.
        unsafe {
            self.context
                .device()
                .wait_for_fences(&[self.encode_fence], true, u64::MAX)
        }
        .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;

        // Get query results for bitstream size.
        // Need to use raw vkGetQueryPoolResults because ash's wrapper infers query count from buffer size.
        // We want 1 query returning 12 bytes (3 u32s): offset, bytes_written, status.
        let mut query_results = [0u32; 3];
        let result = unsafe {
            (self.context.device().fp_v1_0().get_query_pool_results)(
                self.context.device().handle(),
                self.query_pool,
                0,  // firstQuery
                1,  // queryCount (explicit: we want 1 query, not 3!)
                12, // dataSize in bytes (3 u32s: offset, bytes_written, status)
                query_results.as_mut_ptr() as *mut std::ffi::c_void,
                12, // stride (12 bytes per query result)
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::QueryPool(format!(
                "Failed to get query results: {:?}",
                result
            )));
        }

        // Query result layout (WITH_STATUS at the END per Vulkan spec):
        //   [0] = bitstream buffer offset (from BITSTREAM_BUFFER_OFFSET feedback flag)
        //   [1] = bitstream bytes written (from BITSTREAM_BYTES_WRITTEN feedback flag)
        //   [2] = status (VkQueryResultStatusKHR: 1 = COMPLETE, 0 = NOT_READY, -1 = ERROR)
        let bitstream_offset = query_results[0];
        let bitstream_size = query_results[1] as usize;
        let status = query_results[2] as i32;

        debug!(
            "AV1 query results: status={}, offset={}, size={}",
            status, bitstream_offset, bitstream_size
        );

        if status != 1 {
            return Err(PixelForgeError::CommandBuffer(format!(
                "Encode query status indicates failure: {} (1=COMPLETE, 0=NOT_READY, -1=ERROR)",
                status
            )));
        }

        if bitstream_size == 0 || bitstream_size > MIN_BITSTREAM_BUFFER_SIZE {
            return Err(PixelForgeError::CommandBuffer(format!(
                "Invalid bitstream size: {}",
                bitstream_size
            )));
        }

        // Copy encoded data from mapped buffer at the reported offset.
        let mut encoded_data = vec![0u8; bitstream_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.bitstream_buffer_ptr.add(bitstream_offset as usize),
                encoded_data.as_mut_ptr(),
                bitstream_size,
            );
        }

        debug!("Encoded frame: {} bytes", bitstream_size);

        Ok(encoded_data)
    }
}
