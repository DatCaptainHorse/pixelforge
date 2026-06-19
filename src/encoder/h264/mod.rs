//! H.264/AVC codec: the differences from the generic encoder.
//!
//! The shared session/DPB/pipeline machinery lives in [`crate::encoder::codec`];
//! this folder holds only what is specific to H.264: its reference-picture
//! tracking and syntax counters ([`H264`]), the per-frame StdVideo* graph
//! (`record`), and the SPS/PPS generation (`session_params`).

mod init;
mod record;
mod session_params;

use crate::encoder::codec::{EncoderCommon, FramePlan, PictureSetup, VideoCodec};
use crate::encoder::dpb::{
    DecodedPictureBuffer, DecodedPictureBufferTrait, DpbConfig, PictureStartInfo, PictureType,
};
use crate::encoder::pipeline::EncodeFuture;
use crate::encoder::ColorDescription;
use crate::error::Result;
use ash::vk;

/// H.264 macroblock size in pixels.
pub const MB_SIZE: u32 = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub frame_num: u32,
    pub poc: i32,
}

/// H.264-specific encoder state (everything the generic encoder doesn't own).
pub(crate) struct H264 {
    /// Decoded-picture-buffer bookkeeping (reference marking).
    dpb: DecodedPictureBuffer,
    /// `frame_num` syntax element (mod `max_frame_num`).
    frame_num_syntax: u32,
    /// IDR picture id, toggled each IDR.
    idr_pic_id: u32,
    /// Whether an L1 (backward) reference is available, for B-frames.
    has_backward_reference: bool,
    backward_reference_frame_num: u32,
    backward_reference_poc: i32,
    backward_reference_dpb_slot: u8,
    /// Active L0 references, most-recent first.
    l0_references: Vec<ReferenceInfo>,
    /// Negotiated number of active references.
    active_reference_count: u32,
    /// Profile IDC (cached for parameter-set recreation).
    profile_idc: u32,
    /// Whether CABAC entropy coding is preferred (from quality-level query).
    preferred_entropy_cabac: bool,
}

impl VideoCodec for H264 {
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        if plan.is_idr() {
            self.frame_num_syntax = 0;
            self.idr_pic_id = (self.idr_pic_id + 1) & 1;
            // Reset the DPB for the new coded video sequence.
            let dpb_config = DpbConfig {
                dpb_size: common.dpb_slot_count as u32,
                max_num_ref_frames: common.config.max_reference_frames,
                use_multiple_references: common.config.b_frame_count > 0,
                log2_max_frame_num_minus4: 4,
                log2_max_pic_order_cnt_lsb_minus4: 4,
                ..Default::default()
            };
            self.dpb.h264.sequence_start(dpb_config);
            self.l0_references.clear();
            self.has_backward_reference = false;
        }

        let header = if plan.is_idr() {
            Some(self.build_header(common)?)
        } else {
            None
        };

        Ok(PictureSetup {
            frame_type: plan.frame_type(),
            header,
        })
    }

    fn record_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture> {
        self.record(common, plan)
    }

    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) {
        // `frame_num` carried by this frame (pre-increment), reused below for the
        // reference entry even after the syntax counter advances.
        let frame_num = self.frame_num_syntax;
        let pic_order_cnt = plan.pic_order_cnt();

        if plan.is_reference() && !plan.is_b_frame() {
            self.frame_num_syntax = (frame_num + 1) % 256;
        }

        if plan.is_reference() {
            let pic_type = if plan.is_idr() {
                PictureType::Idr
            } else if plan.is_b_frame() {
                PictureType::B
            } else {
                PictureType::P
            };
            let pic_info = PictureStartInfo {
                frame_id: plan.display_order,
                pic_order_cnt,
                frame_num,
                pic_type,
                is_reference: true,
                ..Default::default()
            };
            self.dpb.h264.picture_start(pic_info);
            self.dpb.h264.picture_end(true);

            // The current frame becomes a reference for subsequent frames.
            self.l0_references.insert(
                0,
                ReferenceInfo {
                    dpb_slot: common.current_dpb_slot,
                    frame_num,
                    poc: pic_order_cnt,
                },
            );
            while self.l0_references.len() > self.active_reference_count as usize {
                self.l0_references.pop();
            }

            // Pick the next free DPB slot for the reconstructed picture.
            let used_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
            for i in 0..common.dpb_slot_count as u8 {
                if !used_slots.contains(&i) {
                    common.current_dpb_slot = i;
                    break;
                }
            }
        }
    }

    fn create_session_params(
        &self,
        common: &EncoderCommon,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR> {
        self.build_session_params(common, desc)
    }

    fn invalidate_header_cache(&mut self) {
        // H.264 regenerates SPS/PPS on every IDR, so there is nothing cached.
    }
}
