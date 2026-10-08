//! Parser tests against real libaom and SVT-AV1 streams.
//!
//! The fixtures in `tests/data` are 320x240, 30-frame streams. Between them
//! they cover hidden frames shown again later, a prediction pyramid with skip
//! mode, 2x2 tiles, and 10-bit.
//!
//! These tests need no GPU, so they run everywhere.

use std::sync::Arc;

use super::frame_header::{FrameHeader, HeaderParser, KEY_FRAME};
use super::sequence::{SequenceHeader, obu_type, parse_sequence_header, split_obus};
use super::temporal_unit_starts;

const HIDDEN_FRAMES: &[u8] = include_bytes!("../../../tests/data/av1_hidden_frames.obu");
const TILES: &[u8] = include_bytes!("../../../tests/data/av1_tiles.obu");
const MAIN10: &[u8] = include_bytes!("../../../tests/data/av1_main10.obu");
const PYRAMID: &[u8] = include_bytes!("../../../tests/data/av1_pyramid.obu");

/// Every frame header in `data`, in stream order, parsed against the
/// reference state the earlier frames leave behind.
fn parse_all(data: &[u8]) -> (Arc<SequenceHeader>, Vec<FrameHeader>) {
    let mut parser = HeaderParser::new();
    let mut headers = Vec::new();
    for obu in split_obus(data).expect("stream splits into OBUs") {
        match obu.obu_type {
            obu_type::SEQUENCE_HEADER => {
                let seq = parse_sequence_header(obu.payload).expect("sequence header parses");
                if parser.seq.as_deref() != Some(&seq) {
                    parser.seq = Some(Arc::new(seq));
                    parser.reset_references();
                }
            }
            obu_type::FRAME_HEADER | obu_type::FRAME => {
                headers.push(
                    parser
                        .parse(&obu, obu.payload)
                        .expect("frame header parses"),
                );
            }
            _ => {}
        }
    }
    (parser.seq.expect("a sequence header"), headers)
}

/// How many frames the stream outputs: shown directly, or shown again.
fn output_count(headers: &[FrameHeader]) -> usize {
    headers
        .iter()
        .filter(|h| h.show_frame || h.show_existing_frame)
        .count()
}

#[test]
fn a_stream_opens_with_a_delimiter_and_a_sequence_header() {
    let obus = split_obus(HIDDEN_FRAMES).unwrap();
    assert_eq!(obus[0].obu_type, obu_type::TEMPORAL_DELIMITER);
    assert_eq!(obus[1].obu_type, obu_type::SEQUENCE_HEADER);
    assert_eq!(obus[2].obu_type, obu_type::FRAME);
}

#[test]
fn reads_the_sequence_geometry_and_format() {
    for (name, data, bit_depth) in [
        ("hidden", HIDDEN_FRAMES, 8),
        ("tiles", TILES, 8),
        ("main10", MAIN10, 10),
        ("pyramid", PYRAMID, 8),
    ] {
        let (seq, _) = parse_all(data);
        assert_eq!(seq.max_frame_width_minus_1 + 1, 320, "{name}: width");
        assert_eq!(seq.max_frame_height_minus_1 + 1, 240, "{name}: height");
        assert_eq!(seq.color.bit_depth, bit_depth, "{name}: bit depth");
        assert_eq!(seq.seq_profile, 0, "{name}: Main profile");
        assert!(
            seq.color.subsampling_x && seq.color.subsampling_y,
            "{name}: 4:2:0"
        );
        assert!(seq.enable_order_hint, "{name}: order hints");
    }
}

#[test]
fn every_stream_outputs_thirty_frames() {
    for (name, data) in [
        ("hidden", HIDDEN_FRAMES),
        ("tiles", TILES),
        ("main10", MAIN10),
        ("pyramid", PYRAMID),
    ] {
        let (_, headers) = parse_all(data);
        assert_eq!(output_count(&headers), 30, "{name}");
        assert_eq!(
            headers[0].frame_type, KEY_FRAME,
            "{name}: opens with a key frame"
        );
        assert!(headers[0].show_frame, "{name}: shown");
    }
}

/// Hidden frames are decoded and kept, then shown by a later header that
/// carries no picture of its own.
#[test]
fn hidden_frames_are_shown_again_by_reference() {
    for data in [HIDDEN_FRAMES, PYRAMID] {
        let (_, headers) = parse_all(data);
        let hidden = headers
            .iter()
            .filter(|h| !h.show_existing_frame && !h.show_frame)
            .count();
        let shown_again = headers.iter().filter(|h| h.show_existing_frame).count();
        assert!(hidden > 0, "the stream has hidden frames");
        assert_eq!(hidden, shown_again, "each hidden frame is shown once");
    }
}

#[test]
fn tile_layout_matches_what_the_encoder_was_asked_for() {
    let (_, headers) = parse_all(TILES);
    let tiles = &headers[0].tile_info;
    assert_eq!((tiles.tile_cols, tiles.tile_rows), (2, 2));
    assert!(tiles.uniform_tile_spacing_flag);
    // The starts cover the frame and each tile is as wide as it says.
    assert_eq!(tiles.mi_col_starts[0], 0);
    assert_eq!(tiles.mi_col_starts[2], headers[0].mi_cols);
    assert_eq!(tiles.mi_row_starts[2], headers[0].mi_rows);

    // The tile group of the first frame holds four tiles, each with data.
    let parser_seq = parse_all(TILES).0;
    let mut parser = HeaderParser::new();
    parser.seq = Some(parser_seq);
    let obu = split_obus(TILES)
        .unwrap()
        .into_iter()
        .find(|o| o.obu_type == obu_type::FRAME)
        .unwrap();
    let header = parser.parse(&obu, obu.payload).unwrap();
    let group = parser
        .parse_tile_group(&header, &obu.payload[header.header_bytes..])
        .unwrap();
    assert_eq!(group.tiles.len(), 4);
    assert_eq!(group.tg_end, 3);
    let rest = obu.payload.len() - header.header_bytes;
    for tile in &group.tiles {
        assert!(tile.size > 0 && tile.offset + tile.size <= rest);
    }
    // Tiles are laid out one after another with their size prefixes between.
    for pair in group.tiles.windows(2) {
        assert!(pair[1].offset > pair[0].offset + pair[0].size - 1);
    }
}

/// SVT-AV1's prediction pyramid exercises the parts of the header that depend
/// on earlier frames: order hints, skip mode, which references are valid.
#[test]
fn a_pyramid_exercises_skip_mode_and_distant_references() {
    let (_, headers) = parse_all(PYRAMID);
    let coded: Vec<_> = headers.iter().filter(|h| !h.show_existing_frame).collect();
    assert!(coded.iter().any(|h| h.skip_mode_present));
    assert!(coded.iter().any(|h| h.reference_select));
    assert!(coded.iter().any(|h| h.use_ref_frame_mvs));
    // The first inter frame is far ahead of the key frame in display order.
    let first_inter = coded.iter().find(|h| !h.frame_is_intra()).unwrap();
    assert!(first_inter.order_hint > 8);
    // Skip mode names two distinct references.
    for h in coded.iter().filter(|h| h.skip_mode_present) {
        assert!(h.skip_mode_frame[0] < h.skip_mode_frame[1]);
        assert!((1..=7).contains(&h.skip_mode_frame[0]));
        assert!((1..=7).contains(&h.skip_mode_frame[1]));
    }
}

/// The secondary strengths are the coded values: the driver widens a 3 to 4
/// itself, so a header handing one over already widened decodes wrongly.
#[test]
fn cdef_secondary_strengths_stay_as_coded() {
    for data in [HIDDEN_FRAMES, PYRAMID, TILES] {
        let (_, headers) = parse_all(data);
        for h in headers.iter().filter(|h| !h.show_existing_frame) {
            for i in 0..8 {
                assert!(h.cdef.y_sec_strength[i] <= 3);
                assert!(h.cdef.uv_sec_strength[i] <= 3);
            }
        }
    }
}

#[test]
fn only_a_shown_key_frame_lets_decoding_start() {
    let mut parser = HeaderParser::new();
    let (seq, _) = parse_all(PYRAMID);
    parser.seq = Some(seq);
    let frames: Vec<_> = split_obus(PYRAMID)
        .unwrap()
        .into_iter()
        .filter(|o| o.obu_type == obu_type::FRAME || o.obu_type == obu_type::FRAME_HEADER)
        .collect();
    assert!(parser.starts_with_shown_key_frame(frames[0].payload));
    assert!(
        frames[1..]
            .iter()
            .all(|f| !parser.starts_with_shown_key_frame(f.payload))
    );
}

/// Without a sequence header the first frame cannot be read, and that is
/// reported as missing state rather than as a broken stream.
#[test]
fn a_frame_before_any_sequence_header_is_missing_state() {
    use super::frame_header::HeaderError;
    let mut parser = HeaderParser::new();
    let obu = split_obus(HIDDEN_FRAMES)
        .unwrap()
        .into_iter()
        .find(|o| o.obu_type == obu_type::FRAME)
        .unwrap();
    assert!(matches!(
        parser.parse(&obu, obu.payload),
        Err(HeaderError::MissingState(_))
    ));
}

#[test]
fn finds_temporal_units_in_a_raw_stream() {
    for data in [HIDDEN_FRAMES, TILES, MAIN10, PYRAMID] {
        assert_eq!(temporal_unit_starts(data).len(), 30);
    }
}

/// A byte-stream feed can cut anywhere. Everything before the last delimiter
/// is whole temporal units, and a cut inside an OBU is not mistaken for a
/// boundary.
#[test]
fn a_stream_cut_mid_obu_still_frames_correctly() {
    let starts = temporal_unit_starts(PYRAMID);
    // A delimiter is two bytes; one cut inside it is not a boundary yet.
    for cut in [starts[3] + 2, starts[3] + 7, starts[7] - 1, starts[7] + 2] {
        let prefix = super::complete_prefix(&PYRAMID[..cut]);
        let expected = *starts.iter().rev().find(|&&s| s < cut).unwrap();
        assert_eq!(prefix, expected, "cut at {cut}");
    }
    // Nothing before the first delimiter is complete.
    assert_eq!(super::complete_prefix(&PYRAMID[..starts[1] - 1]), 0);
}

#[test]
fn an_obu_that_runs_past_the_data_is_an_error() {
    let truncated = &HIDDEN_FRAMES[..HIDDEN_FRAMES.len() - 1];
    assert!(split_obus(truncated).is_err());
}
