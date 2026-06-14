use super::H265Encoder;

use crate::encoder::dpb::{DecodedPictureBufferTrait, DpbConfig, PictureStartInfo, PictureType};
use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::encoder::pipeline::SlotPacketMetadata;
use crate::encoder::{ColorDescription, EncodedPacket};
use crate::error::Result;
use crate::PixelForgeError;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

impl H265Encoder {
    /// Get the internal input image.
    ///
    /// This image can be used as a target for `ColorConverter::convert` to avoid
    /// an intermediate copy.
    pub fn input_image(&self) -> vk::Image {
        self.pipeline.input_image()
    }

    /// Try to receive a finished encoded packet without blocking.
    ///
    /// Encoding is asynchronous (see [`Self::encode`]); packets are produced by
    /// a background readback thread and surfaced here. Returns `None` when no
    /// packet is currently ready.
    pub fn poll_packet(&self) -> Result<Option<EncodedPacket>> {
        self.pipeline.poll_packet().transpose()
    }

    /// Take ownership of the encoded-packet receiver for out-of-band consumption.
    pub fn take_packet_receiver(&mut self) -> Option<crate::encoder::EncodedPacketReceiver> {
        self.pipeline.take_packet_receiver()
    }

    /// Encode a frame from a GPU image.
    ///
    /// This accepts a source NV12 image on the GPU and encodes it directly without.
    /// any CPU-side data copies. The source image must be in NV12 format with the
    /// same dimensions as the encoder configuration, and should be in GENERAL layout.
    ///
    /// Encoding is asynchronous and pipelined (depth 2): this submits the given
    /// frame to the encode queue without blocking and returns immediately. The
    /// encoded packet is produced by a background readback thread and retrieved
    /// via [`Self::poll_packet`] (or [`Self::flush`] at end-of-stream). This call
    /// only blocks if all pipeline slots are still in flight (backpressure).
    ///
    /// # Panics
    ///
    /// The encoder will panic at creation time if B-frames are enabled (b_frame_count > 0),
    /// as B-frame encoding is not yet supported.
    pub fn encode(&mut self, src_image: vk::Image) -> Result<()> {
        // Ensure the slot we are about to record over and reuse has been read
        // back by the completion thread.
        self.pipeline.wait_current_free();

        let gop_position = self.gop.get_next_frame();
        let display_order = self.input_frame_num;
        self.input_frame_num += 1;

        debug!(
            "Encoding frame {} from GPU image: type={:?}, poc={}",
            display_order, gop_position.frame_type, gop_position.pic_order_cnt
        );

        // Upload from GPU image.
        self.upload_from_image(src_image)?;

        self.encode_current_frame(&gop_position, display_order)?;

        self.pipeline.advance();

        Ok(())
    }

    /// Internal method to encode the current frame already uploaded to input_image.
    fn encode_current_frame(
        &mut self,
        gop_position: &GopPosition,
        display_order: u64,
    ) -> Result<()> {
        let is_idr = gop_position.frame_type.is_idr();
        let is_reference = gop_position.is_reference;
        let is_b_frame = gop_position.frame_type == GopFrameType::B;

        debug!(
            "Encoding frame: display_order={}, type={:?}, idr={}, ref={}, poc={}, l0_refs={:?}",
            display_order,
            gop_position.frame_type,
            is_idr,
            is_reference,
            gop_position.pic_order_cnt,
            self.l0_references
                .iter()
                .map(|r| (r.dpb_slot, r.poc))
                .collect::<Vec<_>>()
        );

        // Determine frame_type.
        let frame_type = if is_idr {
            crate::encoder::FrameType::I
        } else {
            match gop_position.frame_type {
                GopFrameType::Idr | GopFrameType::I => crate::encoder::FrameType::I,
                GopFrameType::P => crate::encoder::FrameType::P,
                GopFrameType::B => crate::encoder::FrameType::B,
            }
        };

        if is_idr {
            // Reset DPB by calling sequence_start with new config for IDR.
            let dpb_config = DpbConfig {
                dpb_size: self.dpb_slot_count as u32,
                max_num_ref_frames: self.config.max_reference_frames,
                use_multiple_references: self.config.b_frame_count > 0,
                log2_max_frame_num_minus4: 0,
                log2_max_pic_order_cnt_lsb_minus4: 4,
                ..Default::default()
            };
            self.dpb.h265.sequence_start(dpb_config);
            self.l0_references.clear();
            self.has_backward_reference = false;
            // Reset DPB slot activation tracking on IDR - all slots become inactive.
            for active in &mut self.dpb_slot_active {
                *active = false;
            }
        }

        let pic_order_cnt = gop_position.pic_order_cnt;

        // For IDR frames, prepend VPS/SPS/PPS header.
        let header = if is_idr {
            if self.header_data.is_none() {
                let header = self.get_h265_header()?;
                // Debug: print first few bytes of header.
                debug!(
                    "H.265 header ({} bytes): {:02X?}",
                    header.len(),
                    &header[..std::cmp::min(32, header.len())]
                );
                self.header_data = Some(header);
            }
            self.header_data.clone()
        } else {
            None
        };

        // Record packet metadata before submitting: `submit_current` hands it to
        // the readback thread together with the bitstream. `dts` reads the
        // pre-increment encode counter (also used by `encode_frame_internal`).
        let dts = self.encode_frame_num;
        self.pipeline.set_pending_metadata(SlotPacketMetadata {
            frame_type,
            is_key_frame: is_idr,
            pts: display_order,
            dts,
            header,
        });

        self.encode_frame_internal(gop_position, pic_order_cnt, is_idr)?;

        self.encode_frame_num += 1;

        if is_reference {
            let dpb_pic_type = if is_idr {
                PictureType::Idr
            } else if is_b_frame {
                PictureType::B
            } else {
                PictureType::P
            };
            let pic_info = PictureStartInfo {
                frame_id: display_order,
                pic_order_cnt,
                frame_num: 0,
                pic_type: dpb_pic_type,
                is_reference,
                ..Default::default()
            };
            self.dpb.h265.picture_start(pic_info);
            self.dpb.h265.picture_end(is_reference);

            // Update reference tracking for the next P-frame.
            // The current frame becomes the reference for subsequent frames.
            let ref_info = super::ReferenceInfo {
                dpb_slot: self.current_dpb_slot,
                poc: pic_order_cnt,
            };
            self.l0_references.insert(0, ref_info);

            // Limit to active_reference_count
            while self.l0_references.len() > self.active_reference_count as usize {
                self.l0_references.pop();
            }

            if !is_b_frame {
                // Find a free DPB slot for the next frame.
                // Avoid slots currently in l0_references.
                // B-frames don't consume a slot permanently (in this simple implementation they are not reference),
                // but if we support B-frame references we need more complex logic.
                // Assuming B-frames are not reference for now (is_reference=false for B in this impl usually).

                let used_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
                for i in 0..self.dpb_slot_count as u8 {
                    if !used_slots.contains(&i) {
                        self.current_dpb_slot = i;
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Flush the encoder and get any remaining packets.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        self.pipeline.flush()
    }

    /// Request that the next frame be an IDR frame.
    pub fn request_idr(&mut self) {
        self.gop.request_idr();
    }

    /// Retrieve encoded VPS/SPS/PPS header data from video session parameters.
    /// This uses vkGetEncodedVideoSessionParametersKHR to get the NAL units.
    fn get_h265_header(&self) -> Result<Vec<u8>> {
        // H.265-specific get info requesting VPS, SPS and PPS.
        let mut h265_get_info = vk::VideoEncodeH265SessionParametersGetInfoKHR::default()
            .write_std_vps(true)
            .write_std_sps(true)
            .write_std_pps(true)
            .std_vps_id(0)
            .std_sps_id(0)
            .std_pps_id(0);

        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
            .video_session_parameters(self.session_params)
            .push(&mut h265_get_info);

        // Some implementations misbehave for a size-only query (pData = NULL). Use a
        // preallocated buffer and retry on INCOMPLETE (vk_video_samples-style).
        let mut data = vec![0u8; 4096];
        let mut data_size: usize = data.len();
        let mut h265_feedback = vk::VideoEncodeH265SessionParametersFeedbackInfoKHR::default();
        let mut feedback =
            vk::VideoEncodeSessionParametersFeedbackInfoKHR::default().push(&mut h265_feedback);

        let mut attempts = 0;
        loop {
            attempts += 1;
            let result = unsafe {
                (self
                    .video_encode_fn
                    .fp()
                    .get_encoded_video_session_parameters_khr)(
                    self.context.device().handle(),
                    &get_info,
                    &mut feedback,
                    &mut data_size,
                    data.as_mut_ptr() as *mut std::ffi::c_void,
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    if data_size == 0 {
                        return Err(PixelForgeError::SessionParametersCreation(
                            "Encoded parameters size is 0".to_string(),
                        ));
                    }
                    data.truncate(data_size);
                    return Ok(data);
                }
                vk::Result::INCOMPLETE if attempts < 3 => {
                    let new_size = data_size.max(data.len() * 2).max(1);
                    data.resize(new_size, 0);
                    data_size = data.len();
                }
                err => {
                    return Err(PixelForgeError::SessionParametersCreation(format!(
                        "Failed to get encoded parameters: {:?}",
                        err
                    )));
                }
            }
        }
    }

    /// Update the color description (VUI parameters) in the encoded stream.
    ///
    /// This recreates the video session parameters with a new SPS containing the
    /// updated VUI color primaries, transfer characteristics, and matrix coefficients.
    /// The next encoded frame will be an IDR with the new VPS/SPS/PPS prepended.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        // Wait for all in-flight encodes to be read back before modifying the
        // shared session parameters.
        self.pipeline.wait_all_free();

        // Save old handle so we can destroy it after successful creation.
        let old_session_params = self.session_params;

        let new_session_params = self.create_session_params(&desc)?;

        // Destroy old session parameters now that the new ones are created.
        unsafe {
            (self
                .video_queue_fn
                .fp()
                .destroy_video_session_parameters_khr)(
                self.context.device().handle(),
                old_session_params,
                std::ptr::null(),
            );
        }

        self.session_params = new_session_params;
        self.config.color_description = Some(desc);
        self.header_data = None; // Invalidate cached VPS/SPS/PPS header
        self.gop.request_idr();

        debug!(
            "H.265 color description updated: primaries={}, transfer={}, matrix={}, full_range={}",
            desc.color_primaries,
            desc.transfer_characteristics,
            desc.matrix_coefficients,
            desc.full_range
        );

        Ok(())
    }
}
