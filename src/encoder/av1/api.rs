use super::AV1Encoder;

use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::encoder::EncodedPacket;
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl AV1Encoder {
    /// Get the internal input image.
    ///
    /// This image can be used as a target for `ColorConverter::convert` to avoid
    /// an intermediate copy.
    pub fn input_image(&self) -> vk::Image {
        self.input_image
    }

    /// Encode a frame from a GPU image.
    ///
    /// This accepts a source image on the GPU and encodes it directly without
    /// any CPU-side data copies. The source image must be in the correct format
    /// with the same dimensions as the encoder configuration, and should be in GENERAL layout.
    ///
    /// # Panics
    ///
    /// The encoder will panic at creation time if B-frames are enabled (b_frame_count > 0),
    /// as B-frame encoding is not yet supported.
    pub fn encode(&mut self, src_image: vk::Image) -> Result<Vec<EncodedPacket>> {
        let gop_position = self.gop.get_next_frame();
        let display_order = self.input_frame_num;
        self.input_frame_num += 1;

        debug!(
            "AV1 encode: frame {} from GPU image, type={:?}",
            display_order, gop_position.frame_type
        );

        // Upload from GPU image.
        self.upload_from_image(src_image)?;

        // Encode immediately.
        let packet = self.encode_current_frame(&gop_position, display_order)?;

        Ok(vec![packet])
    }

    /// Internal method to encode the current frame already uploaded to input_image.
    fn encode_current_frame(
        &mut self,
        gop_position: &GopPosition,
        display_order: u64,
    ) -> Result<EncodedPacket> {
        let is_key_frame =
            gop_position.frame_type.is_idr() || gop_position.frame_type == GopFrameType::I;
        let is_reference = gop_position.is_reference;
        let frame_type = match gop_position.frame_type {
            GopFrameType::Idr | GopFrameType::I => crate::encoder::FrameType::I,
            GopFrameType::P => crate::encoder::FrameType::P,
            GopFrameType::B => crate::encoder::FrameType::B,
        };

        debug!(
            "Encoding frame: display_order={}, type={:?}, key={}, ref={}",
            display_order, frame_type, is_key_frame, is_reference
        );

        if is_key_frame {
            self.frame_num = 0;
            self.order_hint = 0;
            // Reset references for key frames.
            self.references.clear();
        }

        let mut encoded_data = Vec::new();

        // AV1 Temporal Delimiter OBU: type=2, has_size=1, size=0.
        // Required as the first OBU in each temporal unit for conformant bitstreams.
        // This enables ffmpeg's AV1 demuxer to detect frame boundaries in raw OBU streams.
        encoded_data.extend_from_slice(&[0x12, 0x00]);

        // For key frames, prepend the AV1 Sequence Header OBU.
        // This is required for AV1 decoders to initialize (equivalent to H.265 VPS/SPS/PPS).
        if is_key_frame {
            if self.header_data.is_none() {
                let header = self.get_av1_sequence_header()?;
                debug!(
                    "AV1 sequence header ({} bytes): {:02X?}",
                    header.len(),
                    &header[..std::cmp::min(32, header.len())]
                );
                self.header_data = Some(header);
            }
            if let Some(ref header) = self.header_data {
                encoded_data.extend_from_slice(header);
            }
        }

        encoded_data.extend_from_slice(&self.encode_frame_internal(gop_position, is_key_frame)?);

        // Save the order_hint used during encoding BEFORE incrementing.
        let encoded_order_hint = self.order_hint;
        self.encode_frame_num += 1;
        self.frame_num += 1;
        self.order_hint = (self.order_hint + 1) & 0xFF; // 8-bit order hint

        if is_reference {
            // Update reference tracking for the next frame.
            // Use the order_hint value that was active during encoding, not the incremented value.
            let ref_info = super::ReferenceInfo {
                dpb_slot: self.current_dpb_slot,
                order_hint: encoded_order_hint,
            };
            self.references.insert(0, ref_info);
            // Limit to negotiated max active references
            while self.references.len() > self.active_reference_count as usize {
                self.references.pop();
            }

            // Find a free DPB slot for the next frame.
            let used_slots: Vec<u8> = self.references.iter().map(|r| r.dpb_slot).collect();
            for i in 0..self.dpb_slot_count as u8 {
                if !used_slots.contains(&i) {
                    self.current_dpb_slot = i;
                    break;
                }
            }
        }

        Ok(EncodedPacket {
            data: encoded_data,
            frame_type,
            is_key_frame,
            pts: display_order,
            dts: self.encode_frame_num - 1,
        })
    }

    /// Flush the encoder and get any remaining packets.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        // No buffered frames in the current implementation.
        Ok(Vec::new())
    }

    /// Request that the next frame be an IDR/key frame.
    pub fn request_idr(&mut self) {
        self.gop.request_idr();
    }

    /// Retrieve encoded AV1 Sequence Header OBU from video session parameters.
    ///
    /// Uses vkGetEncodedVideoSessionParametersKHR to get the driver-generated OBU.
    /// The driver's sequence header must be used because the frame OBUs it produces
    /// reference values from its internal sequence header (not ours).
    fn get_av1_sequence_header(&self) -> Result<Vec<u8>> {
        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR {
            video_session_parameters: self.session_params,
            ..Default::default()
        };

        let mut data = vec![0u8; 4096];
        let mut data_size: usize = data.len();
        let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default();

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
                            "AV1 sequence header size is 0".to_string(),
                        ));
                    }
                    data.truncate(data_size);
                    debug!("Retrieved AV1 sequence header: {} bytes", data.len());
                    return Ok(data);
                }
                vk::Result::INCOMPLETE if attempts < 3 => {
                    let new_size = data_size.max(data.len() * 2).max(1);
                    data.resize(new_size, 0);
                    data_size = data.len();
                }
                err => {
                    return Err(PixelForgeError::SessionParametersCreation(format!(
                        "Failed to get AV1 sequence header: {:?}",
                        err
                    )));
                }
            }
        }
    }
}
