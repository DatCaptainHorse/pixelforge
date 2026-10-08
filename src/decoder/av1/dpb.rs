//! Decode-side AV1 reference management.
//!
//! AV1 names eight reference slots, and a frame may refresh any number of them
//! with itself, so several slots can hold one picture. Vulkan's DPB slots are a
//! different thing: one per stored picture. This maps the first onto the
//! second, and keeps what the driver is told about each stored picture.

use std::sync::Arc;

use crate::decoder::av1::frame_header::NUM_REF_FRAMES;
use crate::decoder::frames::SlotPins;

/// What Vulkan wants to know about a picture held in a DPB slot, besides its
/// pixels (`StdVideoDecodeAV1ReferenceInfo`), and what is needed to show it
/// again.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotMeta {
    pub frame_type: u8,
    pub order_hint: u8,
    /// `OrderHints` the picture was decoded with, by reference name.
    pub saved_order_hints: [u8; 8],
    /// Bit `i` is `RefFrameSignBias[i]` for the picture when it was decoded.
    pub ref_frame_sign_bias: u8,
    pub disable_frame_end_update_cdf: bool,
    pub segmentation_enabled: bool,
    /// The size of the picture, as it was coded.
    pub width: u32,
    pub height: u32,
}

pub(crate) struct DecodeDpb {
    pins: Arc<SlotPins>,
    /// DPB slot held by each of the eight reference slots.
    ref_map: [Option<u8>; NUM_REF_FRAMES],
    meta: Vec<Option<SlotMeta>>,
    slot_count: usize,
}

impl DecodeDpb {
    pub fn new(slot_count: usize, pins: Arc<SlotPins>) -> Self {
        Self {
            pins,
            ref_map: [None; NUM_REF_FRAMES],
            meta: vec![None; slot_count],
            slot_count,
        }
    }

    /// The DPB slot behind reference slot `idx`.
    pub fn slot_of(&self, idx: usize) -> Option<u8> {
        self.ref_map[idx]
    }

    pub fn meta(&self, slot: u8) -> Option<&SlotMeta> {
        self.meta[slot as usize].as_ref()
    }

    /// Whether any reference slot still holds `slot`.
    fn is_referenced(&self, slot: u8) -> bool {
        self.ref_map.contains(&Some(slot))
    }

    /// A slot to decode the next picture into: not held by any reference slot
    /// and not by a frame the caller still has.
    pub fn try_allocate_slot(&self) -> Option<u8> {
        (0..self.slot_count as u8)
            .find(|&slot| !self.is_referenced(slot) && !self.pins.is_pinned(slot))
    }

    pub fn has_pinned_slots(&self) -> bool {
        self.pins.any_pinned()
    }

    /// A picture has been decoded into `slot`: record it, and give it to every
    /// reference slot `refresh_frame_flags` names.
    pub fn store(&mut self, slot: u8, meta: SlotMeta, refresh_frame_flags: u8) {
        self.meta[slot as usize] = Some(meta);
        self.refresh(slot, refresh_frame_flags);
    }

    pub fn refresh(&mut self, slot: u8, refresh_frame_flags: u8) {
        for i in 0..NUM_REF_FRAMES {
            if (refresh_frame_flags >> i) & 1 != 0 {
                self.ref_map[i] = Some(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> SlotMeta {
        SlotMeta {
            frame_type: 0,
            order_hint: 0,
            saved_order_hints: [0; 8],
            ref_frame_sign_bias: 0,
            disable_frame_end_update_cdf: false,
            segmentation_enabled: false,
            width: 64,
            height: 64,
        }
    }

    #[test]
    fn a_slot_held_by_any_reference_is_not_reallocated() {
        let mut dpb = DecodeDpb::new(3, Arc::new(SlotPins::default()));
        let a = dpb.try_allocate_slot().unwrap();
        // Refreshes references 0 and 3 with the one picture.
        dpb.store(a, meta(), 0b1001);
        assert_eq!(dpb.slot_of(0), Some(a));
        assert_eq!(dpb.slot_of(3), Some(a));
        assert_ne!(dpb.try_allocate_slot().unwrap(), a);

        // Replacing one reference leaves the other holding it.
        let b = dpb.try_allocate_slot().unwrap();
        dpb.store(b, meta(), 0b0001);
        assert_ne!(
            dpb.try_allocate_slot(),
            Some(a),
            "reference 3 still holds it"
        );

        // Replacing the last one frees it.
        dpb.store(b, meta(), 0b1000);
        assert!(!dpb.is_referenced(a));
    }

    #[test]
    fn a_frame_nothing_references_frees_its_slot() {
        let mut dpb = DecodeDpb::new(2, Arc::new(SlotPins::default()));
        let a = dpb.try_allocate_slot().unwrap();
        dpb.store(a, meta(), 0);
        assert_eq!(dpb.try_allocate_slot(), Some(a));
    }

    #[test]
    fn pinned_slots_are_not_reallocated() {
        let pins = Arc::new(SlotPins::default());
        let mut dpb = DecodeDpb::new(2, pins.clone());
        let a = dpb.try_allocate_slot().unwrap();
        dpb.store(a, meta(), 0);
        pins.pin(a);
        assert_ne!(dpb.try_allocate_slot(), Some(a));
        pins.release(a);
        assert_eq!(dpb.try_allocate_slot(), Some(a));
    }
}
