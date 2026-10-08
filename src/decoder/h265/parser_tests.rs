//! Parser tests against real x265-generated streams.
//!
//! The fixtures in `tests/data` are 320x240, 30-frame streams produced by
//! x265. They cover B-frames with a closed and an open GOP, several slices per
//! picture, 10-bit, and scaling lists.
//!
//! These tests need no GPU, so they run everywhere.

use super::parser::{
    SliceType, iter_nal_units, nal, parse_pps, parse_short_term_rps, parse_slice_header, parse_sps,
    parse_vps,
};

const BFRAMES: &[u8] = include_bytes!("../../../tests/data/h265_bframes.265");
const OPEN_GOP: &[u8] = include_bytes!("../../../tests/data/h265_opengop.265");
const MULTISLICE: &[u8] = include_bytes!("../../../tests/data/h265_multislice.265");
const MAIN10: &[u8] = include_bytes!("../../../tests/data/h265_main10.265");
const SCALING: &[u8] = include_bytes!("../../../tests/data/h265_scaling.265");

struct Bits(Vec<bool>);
impl Bits {
    fn put(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            self.0.push((v >> i) & 1 != 0);
        }
    }
    fn ue(&mut self, v: u32) {
        let x = v + 1;
        let len = 32 - x.leading_zeros();
        self.put(0, len - 1);
        self.put(x, len);
    }
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }
    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.0.len().div_ceil(8)];
        for (i, &b) in self.0.iter().enumerate() {
            if b {
                out[i / 8] |= 0x80 >> (i % 8);
            }
        }
        out
    }
}

fn first_of(data: &[u8], nal_type: u8) -> Vec<u8> {
    iter_nal_units(data)
        .find(|n| n.nal_type == nal_type)
        .expect("stream has the NAL unit")
        .payload()
        .to_vec()
}

#[test]
fn parses_parameter_sets_of_every_fixture() {
    for (name, data) in [
        ("bframes", BFRAMES),
        ("opengop", OPEN_GOP),
        ("multislice", MULTISLICE),
        ("main10", MAIN10),
        ("scaling", SCALING),
    ] {
        let vps = parse_vps(&first_of(data, nal::VPS)).expect("VPS parses");
        let sps = parse_sps(&first_of(data, nal::SPS)).expect("SPS parses");
        let pps = parse_pps(&first_of(data, nal::PPS)).expect("PPS parses");

        assert_eq!(sps.vps_id, vps.vps_id, "{name}: SPS names the VPS");
        assert_eq!(pps.sps_id, sps.sps_id, "{name}: PPS names the SPS");
        assert_eq!(sps.pic_width_in_luma_samples, 320, "{name}: width");
        // x265 pads to a multiple of the minimum coding block (8), and crops.
        assert_eq!(sps.display_dimensions(), (320, 240), "{name}: visible size");
        assert_eq!(sps.chroma_format_idc, 1, "{name}: 4:2:0");
    }
}

#[test]
fn reads_profile_and_bit_depth() {
    let sps = parse_sps(&first_of(BFRAMES, nal::SPS)).unwrap();
    assert_eq!(sps.ptl.profile_idc, 1, "Main");
    assert_eq!(sps.bit_depth_luma_minus8, 0);

    let sps = parse_sps(&first_of(MAIN10, nal::SPS)).unwrap();
    assert_eq!(sps.ptl.profile_idc, 2, "Main 10");
    assert_eq!(sps.bit_depth_luma_minus8, 2);
    assert_eq!(sps.bit_depth_chroma_minus8, 2);
}

/// B-frames mean pictures wait to be output, and the SPS has to say how many.
#[test]
fn b_frame_streams_declare_a_reorder_depth() {
    let sps = parse_sps(&first_of(BFRAMES, nal::SPS)).unwrap();
    let (dpb_size, reorder) = sps.highest_dec_pic_buf();
    assert!(reorder > 0, "x265 -bf 3 reorders");
    assert!(dpb_size > reorder, "the DPB holds more than it reorders");
}

#[test]
fn scaling_lists_are_read() {
    let sps = parse_sps(&first_of(SCALING, nal::SPS)).unwrap();
    assert!(sps.scaling_list_enabled_flag);
    // `scaling-list=default` enables the feature without sending any lists.
    assert!(!sps.sps_scaling_list_data_present_flag);
    let lists = sps.scaling_lists.expect("the default lists are filled in");
    assert_eq!(lists.list_8x8[0][0], 16);
    assert_eq!(lists.list_8x8[0][63], 115, "the last intra coefficient");
    assert_eq!(lists.list_8x8[3][63], 91, "the last inter coefficient");
}

/// The first slice of every picture is where a picture begins; later ones are
/// not.
#[test]
fn groups_slice_segments_into_pictures() {
    let pictures = |data: &[u8]| {
        iter_nal_units(data)
            .filter(|n| n.is_slice() && n.payload()[0] & 0x80 != 0)
            .count()
    };
    assert_eq!(pictures(BFRAMES), 30);
    assert_eq!(pictures(MULTISLICE), 30);
    let slices = iter_nal_units(MULTISLICE).filter(|n| n.is_slice()).count();
    assert_eq!(slices, 120, "four slices a picture");
}

#[test]
fn slice_headers_give_poc_and_reference_sets() {
    let sps = parse_sps(&first_of(BFRAMES, nal::SPS)).unwrap();
    let pps = parse_pps(&first_of(BFRAMES, nal::PPS)).unwrap();

    let mut saw_b = false;
    let mut first = true;
    for nal in iter_nal_units(BFRAMES).filter(|n| n.is_slice()) {
        let header = parse_slice_header(&nal, &sps, &pps).expect("slice header parses");
        if first {
            first = false;
            assert!(super::parser::is_idr(nal.nal_type), "opens with an IDR");
            assert_eq!(header.pic_order_cnt_lsb, 0);
            assert_eq!(header.short_term_rps.num_delta_pocs(), 0);
            continue;
        }
        if header.slice_type == SliceType::B {
            saw_b = true;
            // A B picture in a pyramid predicts from before and after it.
            let rps = &header.short_term_rps;
            assert!(rps.num_negative_pics > 0);
        }
        // Every non-IDR picture names at least one picture to keep.
        assert!(header.short_term_rps.num_delta_pocs() > 0);
    }
    assert!(saw_b, "the fixture has B slices");
}

#[test]
fn open_gop_stream_has_a_cra_with_leading_pictures() {
    let types: Vec<u8> = iter_nal_units(OPEN_GOP)
        .filter(|n| n.is_slice())
        .map(|n| n.nal_type)
        .collect();
    assert!(
        types.contains(&nal::CRA),
        "a CRA picture starts the second GOP"
    );
    assert!(
        types.iter().any(|&t| super::parser::is_rasl(t)),
        "and it has RASL pictures"
    );
}

/// An inter-predicted set must resolve to the same pictures as one coded
/// directly. The fixtures use both forms, and the derived deltas have to be
/// strictly ordered whichever way they were reached.
#[test]
fn derived_short_term_sets_are_ordered() {
    for data in [BFRAMES, OPEN_GOP, MULTISLICE] {
        let sps = parse_sps(&first_of(data, nal::SPS)).unwrap();
        for rps in &sps.short_term_rps {
            let neg = rps.num_negative_pics as usize;
            let pos = rps.num_positive_pics as usize;
            for w in rps.delta_poc_s0[..neg].windows(2) {
                assert!(w[0] > w[1], "negative deltas must move away from zero");
            }
            for w in rps.delta_poc_s1[..pos].windows(2) {
                assert!(w[0] < w[1], "positive deltas must move away from zero");
            }
            assert!(rps.delta_poc_s0[..neg].iter().all(|&d| d < 0));
            assert!(rps.delta_poc_s1[..pos].iter().all(|&d| d > 0));
        }
    }
}

#[test]
fn explicit_scaling_lists_round_trip_through_the_syntax() {
    // Built by hand: a 4x4 list coded in prediction mode 1 (explicit), flat 16,
    // then every other list copied from the previous one or defaulted. The
    // writer below is the inverse of the parser's reading.
    let mut w = Bits(Vec::new());
    for size_id in 0..4usize {
        let step = if size_id == 3 { 3 } else { 1 };
        for matrix_id in (0..6usize).step_by(step) {
            if size_id == 0 && matrix_id == 0 {
                // Explicit: 16 deltas leaving the running value at 20.
                w.put(1, 1);
                w.se(12);
                for _ in 1..16 {
                    w.se(0);
                }
            } else if matrix_id == 0 {
                // Default list.
                w.put(0, 1);
                w.ue(0);
            } else {
                // Copy the previous matrix of this size.
                w.put(0, 1);
                w.ue(1);
            }
        }
    }
    // A PPS carrying just these lists, through the public parse entry: wrap in
    // an SPS-less harness by parsing a PPS with the scaling syntax present.
    // The PPS syntax before the lists is fixed and short, so build it.
    let mut pps = Bits(Vec::new());
    pps.ue(0); // pps id
    pps.ue(0); // sps id
    pps.put(0, 1); // dependent slice segments
    pps.put(0, 1); // output flag present
    pps.put(0, 3); // extra slice header bits
    pps.put(0, 1); // sign data hiding
    pps.put(0, 1); // cabac init present
    pps.ue(0);
    pps.ue(0); // ref idx defaults
    pps.se(0); // init qp
    pps.put(0, 1); // constrained intra
    pps.put(0, 1); // transform skip
    pps.put(0, 1); // cu qp delta
    pps.se(0);
    pps.se(0); // cb, cr offsets
    pps.put(0, 5); // slice chroma offsets, weighted pred, bipred, transquant bypass, tiles
    pps.put(0, 1); // entropy coding sync
    pps.put(0, 1); // loop filter across slices
    pps.put(0, 1); // deblocking control
    pps.put(1, 1); // pps scaling list data present
    pps.0.extend(w.0);
    pps.put(0, 1); // lists modification
    pps.ue(0); // parallel merge level
    pps.put(0, 1); // slice header extension
    pps.put(0, 1); // pps extension
    pps.put(1, 1); // stop bit
    let parsed = parse_pps(&pps.bytes()).expect("PPS with scaling lists parses");
    let lists = parsed.scaling_lists.expect("lists present");
    assert_eq!(lists.list_4x4[0], [20; 16], "explicit list: 8 + 12");
    assert_eq!(
        lists.list_4x4[1], lists.list_4x4[0],
        "copied from the one before"
    );
    assert_eq!(lists.list_8x8[0][0], 16, "default list");
    assert_eq!(lists.list_8x8[1], lists.list_8x8[0], "copied");
    assert_eq!(lists.dc_16x16[0], 16, "a defaulted list has the default DC");
}

/// An inter-predicted set is built from another by shifting its pictures, and
/// and the order the shifted pictures come out in matters.
#[test]
fn inter_predicted_sets_are_derived_from_their_reference() {
    // Set 0 is coded directly: two pictures before the current one, at -1 and
    // -3. Set 1 predicts from it with a shift of -1.
    let mut w = Bits(Vec::new());
    // st_ref_pic_set(0): no inter_ref_pic_set_prediction_flag at index 0.
    w.ue(2);
    w.ue(0);
    w.ue(0); // -1
    w.put(1, 1);
    w.ue(1); // -3
    w.put(1, 1);
    // st_ref_pic_set(1), inter predicted, in the SPS (no delta_idx).
    w.put(1, 1);
    w.put(1, 1); // delta_rps_sign
    w.ue(0); // abs_delta_rps_minus1: deltaRps = -1
    for _ in 0..3 {
        w.put(1, 1); // used_by_curr_pic_flag for the reference's three entries
    }
    w.put(1, 1); // stop bit
    let bytes = w.bytes();
    let mut r = crate::decoder::bitreader::BitReader::new(&bytes);
    let first = parse_short_term_rps(&mut r, 0, 2, &[]).unwrap();
    assert_eq!(first.delta_poc_s0[..2], [-1, -3]);
    let second = parse_short_term_rps(&mut r, 1, 2, &[first]).unwrap();
    // The reference's -1 and -3 move to -2 and -4, and the reference picture
    // itself, now one step back, appears at -1.
    assert_eq!(second.num_negative_pics, 3);
    assert_eq!(second.delta_poc_s0[..3], [-1, -2, -4]);
    assert_eq!(second.num_positive_pics, 0);
    assert_eq!(second.num_delta_pocs_of_ref_rps, 2);
    assert!(second.inter_ref_pic_set_prediction_flag);
}
