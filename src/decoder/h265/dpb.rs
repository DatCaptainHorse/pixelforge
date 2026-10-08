//! Decode-side H.265 picture order count and reference picture management.
//!
//! The encoder's DPB decides which pictures to reference; a decoder rebuilds
//! that from the stream. Unlike H.264's running marking process, H.265 states
//! the whole set of pictures to keep in every slice header (the reference
//! picture set), so retirement is a matter of resolving that set against what
//! is held and dropping everything else.

use std::sync::Arc;

use crate::decoder::frames::SlotPins;
use crate::decoder::h265::parser::{self, SliceHeader, Sps};
use crate::error::{PixelForgeError, Result};

/// The most DPB slots H.265 can address (`STD_VIDEO_H265_MAX_DPB_SIZE`).
pub(crate) const MAX_DPB_SLOTS: usize = 16;

/// How many entries one of the three "current" lists can hold in Vulkan's
/// picture info.
const MAX_CURRENT_REFS: usize = 8;

/// A picture held in the DPB as a reference.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RefPicture {
    pub slot: u8,
    pub poc: i32,
    pub long_term: bool,
}

/// Per-picture state derived from the bitstream.
#[derive(Debug, Clone)]
pub(crate) struct PictureState {
    pub poc: i32,
    pub is_irap: bool,
    pub is_idr: bool,
    /// An IRAP picture that starts a new coded video sequence: display order
    /// restarts here.
    pub starts_sequence: bool,
    /// Whether the picture is output at all (`pic_output_flag`, and not a RASL
    /// picture whose references are missing).
    pub output: bool,
    /// Not a sub-layer non-reference picture.
    pub is_reference: bool,
    /// DPB slots of the pictures the current one predicts from, in the three
    /// lists Vulkan's picture info names.
    pub st_curr_before: Vec<u8>,
    pub st_curr_after: Vec<u8>,
    pub lt_curr: Vec<u8>,
}

/// What [`DecodeDpb::begin_picture`] decided about a picture.
pub(crate) enum Begin {
    Decode(PictureState),
    /// A RASL picture whose IRAP started the sequence: its references precede
    /// the IRAP and do not exist, and the spec says not to decode it.
    SkipRasl,
    /// A reference the picture needs is not in the DPB: a picture was lost, or
    /// decoding began somewhere it should not have.
    MissingReference(String),
}

pub(crate) struct DecodeDpb {
    pins: Arc<SlotPins>,
    refs: Vec<RefPicture>,
    slot_used: [bool; MAX_DPB_SLOTS],
    slot_count: usize,
    /// POC of `prevTid0Pic`, the anchor POC msb is derived from.
    prev_tid0_poc: i32,
    /// The next IRAP starts a sequence even if it is a CRA: nothing has been
    /// decoded yet, or an end of sequence NAL unit has just gone past.
    at_sequence_start: bool,
    /// The IRAP most recently seen started a sequence, so its RASL pictures
    /// are skipped.
    skip_rasl: bool,
}

impl DecodeDpb {
    pub fn new(slot_count: usize, pins: Arc<SlotPins>) -> Self {
        Self {
            pins,
            refs: Vec::new(),
            slot_used: [false; MAX_DPB_SLOTS],
            slot_count: slot_count.min(MAX_DPB_SLOTS),
            prev_tid0_poc: 0,
            at_sequence_start: true,
            skip_rasl: false,
        }
    }

    pub fn references(&self) -> &[RefPicture] {
        &self.refs
    }

    /// An end of sequence NAL unit went past.
    pub fn end_of_sequence(&mut self) {
        self.at_sequence_start = true;
    }

    /// Work out the POC and reference picture set of the picture about to be
    /// decoded, retire everything the set does not keep, and report which DPB
    /// slots it reads from. Call once per picture, before
    /// [`Self::try_allocate_slot`].
    pub fn begin_picture(
        &mut self,
        nal_type: u8,
        temporal_id: u8,
        header: &SliceHeader,
        sps: &Sps,
    ) -> Result<Begin> {
        let is_irap = parser::is_irap(nal_type);
        let is_idr = parser::is_idr(nal_type);
        let mut starts_sequence = false;
        if is_irap {
            starts_sequence = is_idr || parser::is_bla(nal_type) || self.at_sequence_start;
            self.at_sequence_start = false;
            self.skip_rasl = starts_sequence;
        }
        if parser::is_rasl(nal_type) && self.skip_rasl {
            return Ok(Begin::SkipRasl);
        }

        // Picture order count: the low bits are in the slice header, and the
        // high bits are worked out from the previous picture so the count keeps
        // growing past the point where the low bits wrap.
        let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb();
        let lsb = header.pic_order_cnt_lsb as i32;
        let msb = if is_irap && starts_sequence {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        if temporal_id == 0
            && !parser::is_rasl(nal_type)
            && !parser::is_radl(nal_type)
            && !parser::is_sub_layer_non_reference(nal_type)
        {
            self.prev_tid0_poc = poc;
        }

        // An IRAP that starts a sequence invalidates everything before it.
        if is_irap && starts_sequence {
            self.refs.clear();
            self.slot_used = [false; MAX_DPB_SLOTS];
        }

        // The reference picture set: which pictures this one predicts from, and
        // which others to keep for pictures after it.
        let mut st_curr_before = Vec::new();
        let mut st_curr_after = Vec::new();
        let mut st_foll = Vec::new();
        let mut lt_curr: Vec<(i32, bool)> = Vec::new();
        let mut lt_foll: Vec<(i32, bool)> = Vec::new();
        if !is_idr {
            let rps = &header.short_term_rps;
            for i in 0..rps.num_negative_pics as usize {
                let p = poc + rps.delta_poc_s0[i];
                if rps.used_by_curr_pic_s0[i] {
                    st_curr_before.push(p);
                } else {
                    st_foll.push(p);
                }
            }
            for i in 0..rps.num_positive_pics as usize {
                let p = poc + rps.delta_poc_s1[i];
                if rps.used_by_curr_pic_s1[i] {
                    st_curr_after.push(p);
                } else {
                    st_foll.push(p);
                }
            }
            for lt in &header.long_term_refs {
                let mut p = lt.poc_lsb as i32;
                if lt.delta_poc_msb_present_flag {
                    p += poc - lt.delta_poc_msb_cycle as i32 * max_lsb - (poc & (max_lsb - 1));
                }
                if lt.used_by_curr_pic {
                    lt_curr.push((p, lt.delta_poc_msb_present_flag));
                } else {
                    lt_foll.push((p, lt.delta_poc_msb_present_flag));
                }
            }
        }
        // An IRAP predicts from nothing. Its set exists for the pictures that
        // follow it, and what it names is not there to be used by the IRAP.
        if is_irap {
            st_foll.append(&mut st_curr_before);
            st_foll.append(&mut st_curr_after);
            lt_foll.append(&mut lt_curr);
        }
        if st_curr_before.len() > MAX_CURRENT_REFS
            || st_curr_after.len() > MAX_CURRENT_REFS
            || lt_curr.len() > MAX_CURRENT_REFS
        {
            return Err(PixelForgeError::InvalidInput(
                "H.265 decode: a picture predicts from more than 8 pictures in one list"
                    .to_string(),
            ));
        }

        // Long-term pictures are matched first, and a picture matched as one
        // is no longer a candidate for a short-term entry.
        let find_lt = |refs: &[RefPicture], p: i32, msb_present: bool| {
            refs.iter().position(|r| {
                if msb_present {
                    r.poc == p
                } else {
                    (r.poc & (max_lsb - 1)) == p
                }
            })
        };
        let mut keep = vec![false; self.refs.len()];
        let mut to_long_term = Vec::new();
        let mut lt_curr_slots = Vec::new();
        let mut missing = None;
        for &(p, msb) in &lt_curr {
            match find_lt(&self.refs, p, msb) {
                Some(i) => {
                    keep[i] = true;
                    to_long_term.push(i);
                    lt_curr_slots.push(self.refs[i].slot);
                }
                None => missing = Some(format!("long-term reference (POC {p}) is not held")),
            }
        }
        for &(p, msb) in &lt_foll {
            if let Some(i) = find_lt(&self.refs, p, msb) {
                keep[i] = true;
                to_long_term.push(i);
            }
        }
        for &i in &to_long_term {
            self.refs[i].long_term = true;
        }

        let find_st =
            |refs: &[RefPicture], p: i32| refs.iter().position(|r| !r.long_term && r.poc == p);
        let mut resolve = |list: &[i32], keep: &mut Vec<bool>, refs: &[RefPicture]| {
            let mut slots = Vec::new();
            for &p in list {
                match find_st(refs, p) {
                    Some(i) => {
                        keep[i] = true;
                        slots.push(refs[i].slot);
                    }
                    None => missing = Some(format!("short-term reference (POC {p}) is not held")),
                }
            }
            slots
        };
        let before_slots = resolve(&st_curr_before, &mut keep, &self.refs);
        let after_slots = resolve(&st_curr_after, &mut keep, &self.refs);
        for &p in &st_foll {
            if let Some(i) = find_st(&self.refs, p) {
                keep[i] = true;
            }
        }
        if let Some(reason) = missing {
            return Ok(Begin::MissingReference(reason));
        }

        // Everything the set does not name is no longer a reference.
        let mut i = 0;
        self.refs.retain(|r| {
            let kept = keep[i];
            i += 1;
            if !kept {
                self.slot_used[r.slot as usize] = false;
            }
            kept
        });

        Ok(Begin::Decode(PictureState {
            poc,
            is_irap,
            is_idr,
            starts_sequence,
            output: header.pic_output_flag,
            is_reference: !parser::is_sub_layer_non_reference(nal_type),
            st_curr_before: before_slots,
            st_curr_after: after_slots,
            lt_curr: lt_curr_slots,
        }))
    }

    /// Reserve a slot for the picture about to be decoded: one that is neither
    /// a reference nor held by a frame the caller still has.
    pub fn try_allocate_slot(&mut self) -> Option<u8> {
        for slot in 0..self.slot_count {
            if !self.slot_used[slot] && !self.pins.is_pinned(slot as u8) {
                self.slot_used[slot] = true;
                return Some(slot as u8);
            }
        }
        None
    }

    pub fn has_pinned_slots(&self) -> bool {
        self.pins.any_pinned()
    }

    /// The picture in `slot` has been decoded. It stays a (short-term)
    /// reference until a later picture's set stops naming it.
    pub fn end_picture(&mut self, slot: u8, state: &PictureState) {
        self.refs.push(RefPicture {
            slot,
            poc: state.poc,
            long_term: false,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::h265::parser::{ShortTermRps, SliceType};

    fn sps() -> Sps {
        Sps {
            log2_max_pic_order_cnt_lsb_minus4: 4,
            ..Default::default()
        }
    }

    fn rps(before: &[(i32, bool)], after: &[(i32, bool)]) -> ShortTermRps {
        let mut rps = ShortTermRps::default();
        for (i, &(d, used)) in before.iter().enumerate() {
            rps.delta_poc_s0[i] = d;
            rps.used_by_curr_pic_s0[i] = used;
        }
        for (i, &(d, used)) in after.iter().enumerate() {
            rps.delta_poc_s1[i] = d;
            rps.used_by_curr_pic_s1[i] = used;
        }
        rps.num_negative_pics = before.len() as u8;
        rps.num_positive_pics = after.len() as u8;
        rps
    }

    fn header(lsb: u32, short_term_rps: ShortTermRps) -> SliceHeader {
        SliceHeader {
            slice_type: SliceType::P,
            pic_output_flag: true,
            pic_order_cnt_lsb: lsb,
            short_term_ref_pic_set_sps_flag: false,
            short_term_rps,
            short_term_rps_bits: 0,
            long_term_refs: Vec::new(),
        }
    }

    fn decode(dpb: &mut DecodeDpb, nal_type: u8, h: &SliceHeader) -> (u8, PictureState) {
        let Ok(Begin::Decode(state)) = dpb.begin_picture(nal_type, 0, h, &sps()) else {
            panic!("expected the picture to decode");
        };
        let slot = dpb.try_allocate_slot().expect("a free slot");
        dpb.end_picture(slot, &state);
        (slot, state)
    }

    const IDR: u8 = parser::nal::IDR_W_RADL;
    const TRAIL_R: u8 = 1;

    #[test]
    fn idr_resets_everything() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        decode(&mut dpb, TRAIL_R, &header(1, rps(&[(-1, true)], &[])));
        assert_eq!(dpb.references().len(), 2);
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        assert_eq!(dpb.references().len(), 1);
    }

    #[test]
    fn a_picture_names_the_slots_it_predicts_from() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        let (idr_slot, _) = decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        let (p_slot, p) = decode(&mut dpb, TRAIL_R, &header(2, rps(&[(-2, true)], &[])));
        assert_eq!(p.st_curr_before, vec![idr_slot]);
        // A B picture between them looks both ways.
        let (_, b) = decode(
            &mut dpb,
            TRAIL_R,
            &header(1, rps(&[(-1, true)], &[(1, true)])),
        );
        assert_eq!(b.poc, 1);
        assert_eq!(b.st_curr_before, vec![idr_slot]);
        assert_eq!(b.st_curr_after, vec![p_slot]);
    }

    #[test]
    fn pictures_the_set_leaves_out_are_dropped() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        decode(&mut dpb, TRAIL_R, &header(1, rps(&[(-1, true)], &[])));
        // Only the previous picture is kept; the IDR goes.
        decode(&mut dpb, TRAIL_R, &header(2, rps(&[(-1, true)], &[])));
        let pocs: Vec<i32> = dpb.references().iter().map(|r| r.poc).collect();
        assert_eq!(pocs, vec![1, 2]);
    }

    #[test]
    fn foll_entries_are_kept_but_not_predicted_from() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        decode(&mut dpb, TRAIL_R, &header(1, rps(&[(-1, true)], &[])));
        let (_, state) = decode(
            &mut dpb,
            TRAIL_R,
            &header(2, rps(&[(-1, true), (-2, false)], &[])),
        );
        assert_eq!(state.st_curr_before.len(), 1);
        assert_eq!(dpb.references().len(), 3);
    }

    #[test]
    fn poc_msb_wraps() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        // 8 bits of lsb: POC wraps at 256.
        let mut s = sps();
        s.log2_max_pic_order_cnt_lsb_minus4 = 4;
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        for (lsb, expected) in [(100u32, 100), (200, 200), (44, 300), (150, 406)] {
            let Ok(Begin::Decode(state)) =
                dpb.begin_picture(TRAIL_R, 0, &header(lsb, rps(&[], &[])), &s)
            else {
                panic!("expected decode");
            };
            assert_eq!(state.poc, expected);
            let slot = dpb.try_allocate_slot().unwrap();
            dpb.end_picture(slot, &state);
        }
    }

    #[test]
    fn a_missing_reference_is_reported() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        let h = header(5, rps(&[(-3, true)], &[]));
        assert!(matches!(
            dpb.begin_picture(TRAIL_R, 0, &h, &sps()),
            Ok(Begin::MissingReference(_))
        ));
    }

    #[test]
    fn rasl_after_a_sequence_start_is_skipped() {
        let mut dpb = DecodeDpb::new(4, Arc::new(SlotPins::default()));
        // A CRA that opens the stream starts a sequence, so its RASL pictures
        // (which reference what came before it) are not decoded.
        decode(
            &mut dpb,
            parser::nal::CRA,
            &header(8, rps(&[(-4, false)], &[])),
        );
        let h = header(6, rps(&[(-4, true)], &[(2, true)]));
        assert!(matches!(
            dpb.begin_picture(parser::nal::RASL_R, 0, &h, &sps()),
            Ok(Begin::SkipRasl)
        ));
        // After an end of sequence the next CRA does the same again.
        dpb.end_of_sequence();
        decode(
            &mut dpb,
            parser::nal::CRA,
            &header(8, rps(&[(-4, false)], &[])),
        );
        assert!(matches!(
            dpb.begin_picture(parser::nal::RASL_N, 0, &h, &sps()),
            Ok(Begin::SkipRasl)
        ));
    }

    #[test]
    fn a_mid_stream_cra_keeps_its_rasl_pictures() {
        let mut dpb = DecodeDpb::new(5, Arc::new(SlotPins::default()));
        decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        decode(&mut dpb, TRAIL_R, &header(4, rps(&[(-4, true)], &[])));
        // A CRA in the middle starts nothing; the pictures before it are still
        // there for its RASL pictures to use.
        decode(
            &mut dpb,
            parser::nal::CRA,
            &header(12, rps(&[(-8, false), (-12, false)], &[])),
        );
        let h = header(10, rps(&[(-6, true)], &[(2, true)]));
        assert!(matches!(
            dpb.begin_picture(parser::nal::RASL_R, 0, &h, &sps()),
            Ok(Begin::Decode(_))
        ));
    }

    #[test]
    fn pinned_slots_are_not_reallocated() {
        let pins = Arc::new(SlotPins::default());
        let mut dpb = DecodeDpb::new(2, pins.clone());
        let (slot, _) = decode(&mut dpb, IDR, &header(0, ShortTermRps::default()));
        pins.pin(slot);
        // A new IDR drops the reference, but the caller still holds the frame.
        let Ok(Begin::Decode(_)) =
            dpb.begin_picture(IDR, 0, &header(0, ShortTermRps::default()), &sps())
        else {
            panic!("expected decode");
        };
        let next = dpb.try_allocate_slot().unwrap();
        assert_ne!(next, slot);
    }
}
