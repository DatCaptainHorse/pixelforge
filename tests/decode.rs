//! Decode the streams in `tests/data` on a real device and compare the result
//! with ffmpeg's software decoder, byte for byte, for every codec.
//!
//! Hardware decoding of a conforming stream is exact, so there is no
//! tolerance: any difference is a bug in what is handed to the driver, or in
//! the driver. The fixtures are chosen to stay clear of the corners where a
//! driver is known to differ (see the notes in `src/decoder/mod.rs`), so a
//! failure here is worth looking at.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test --test decode -- --ignored`, and with `PIXELFORGE_VALIDATION=1`
//! to have the validation layer's findings shown. On Intel under Mesa, set
//! `ANV_DEBUG=video-decode,video-encode` first.

use std::process::Command;

#[allow(dead_code)]
mod common;
use common::{Readback, decode_stream};

use pixelforge::decoder::{DecodeConfig, DecodeStatus, Decoder};
use pixelforge::encoder::{BitDepth, Codec};
use pixelforge::vulkan::{VideoContext, VideoContextBuilder};

struct Fixture {
    name: &'static str,
    codec: Codec,
    path: &'static str,
    ten_bit: bool,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "h264 baseline",
        codec: Codec::H264,
        path: "tests/data/base.264",
        ten_bit: false,
    },
    Fixture {
        name: "h264 b-frames",
        codec: Codec::H264,
        path: "tests/data/bframes.264",
        ten_bit: false,
    },
    Fixture {
        name: "h264 multislice",
        codec: Codec::H264,
        path: "tests/data/multislice.264",
        ten_bit: false,
    },
    Fixture {
        name: "h265 b-frames",
        codec: Codec::H265,
        path: "tests/data/h265_bframes.265",
        ten_bit: false,
    },
    Fixture {
        name: "h265 open gop",
        codec: Codec::H265,
        path: "tests/data/h265_opengop.265",
        ten_bit: false,
    },
    Fixture {
        name: "h265 multislice",
        codec: Codec::H265,
        path: "tests/data/h265_multislice.265",
        ten_bit: false,
    },
    Fixture {
        name: "h265 scaling lists",
        codec: Codec::H265,
        path: "tests/data/h265_scaling.265",
        ten_bit: false,
    },
    Fixture {
        name: "h265 main10",
        codec: Codec::H265,
        path: "tests/data/h265_main10.265",
        ten_bit: true,
    },
    Fixture {
        name: "av1 hidden frames",
        codec: Codec::AV1,
        path: "tests/data/av1_hidden_frames.obu",
        ten_bit: false,
    },
    Fixture {
        name: "av1 tiles",
        codec: Codec::AV1,
        path: "tests/data/av1_tiles.obu",
        ten_bit: false,
    },
    Fixture {
        name: "av1 pyramid",
        codec: Codec::AV1,
        path: "tests/data/av1_pyramid.obu",
        ten_bit: false,
    },
    Fixture {
        name: "av1 main10",
        codec: Codec::AV1,
        path: "tests/data/av1_main10.obu",
        ten_bit: true,
    },
];

fn config_for(codec: Codec) -> DecodeConfig {
    match codec {
        Codec::H264 => DecodeConfig::h264(),
        Codec::H265 => DecodeConfig::h265(),
        Codec::AV1 => DecodeConfig::av1(),
    }
    .with_byte_stream()
    .with_output_depth(8)
}

fn context_for(codec: Codec) -> Option<VideoContext> {
    common::init_logging();
    match VideoContextBuilder::new()
        .app_name("pixelforge-decode-test")
        .require_decode(codec)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()
    {
        Ok(context) => Some(context),
        Err(e) => {
            eprintln!("skipping {codec:?}: {e}");
            None
        }
    }
}

/// ffmpeg's software decode of `path`, as raw NV12 or P010.
fn reference(path: &str, ten_bit: bool) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i", path, "-pix_fmt"])
        .arg(if ten_bit { "p010le" } else { "nv12" })
        .args(["-f", "rawvideo", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(out.status.success(), "ffmpeg failed on {path}");
    out.stdout
}

/// What pixelforge decodes `path` to, in the same layout.
fn decode_to_bytes(context: &VideoContext, codec: Codec, path: &str) -> Vec<u8> {
    let mut readback = Readback::new(context).expect("readback");
    let mut decoder = Decoder::new(context.clone(), config_for(codec)).expect("decoder");
    let stream = std::fs::read(path).expect("read fixture");
    let mut bytes = Vec::new();
    decode_stream(&mut decoder, &stream, |frame| {
        assert_eq!(
            frame.bit_depth,
            if path.contains("main10") {
                BitDepth::Ten
            } else {
                BitDepth::Eight
            }
        );
        let data = readback.read(&frame)?;
        bytes.extend_from_slice(&data.y);
        bytes.extend_from_slice(&data.uv);
        Ok(())
    })
    .expect("decode");
    bytes
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn every_fixture_decodes_like_ffmpeg() {
    let mut failures = Vec::new();
    let mut ran = 0;
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        let Some(context) = context_for(codec) else {
            continue;
        };
        for fixture in FIXTURES.iter().filter(|f| f.codec == codec) {
            ran += 1;
            let expected = reference(fixture.path, fixture.ten_bit);
            let actual = decode_to_bytes(&context, codec, fixture.path);
            if expected != actual {
                let first = expected
                    .iter()
                    .zip(&actual)
                    .position(|(a, b)| a != b)
                    .unwrap_or(expected.len().min(actual.len()));
                failures.push(format!(
                    "{}: {} bytes expected, {} decoded, first difference at byte {first}",
                    fixture.name,
                    expected.len(),
                    actual.len()
                ));
            } else {
                eprintln!(
                    "{}: identical to ffmpeg ({} bytes)",
                    fixture.name,
                    actual.len()
                );
            }
        }
    }
    assert!(failures.is_empty(), "differences:\n{}", failures.join("\n"));
    if ran == 0 {
        eprintln!("no codec could be decoded on this machine; nothing was checked");
    }
}

/// Byte offsets at which each picture of an H.26x stream starts, found the
/// plain way: a start code, then a slice NAL unit whose first slice flag is
/// set. Enough for these fixtures, which carry nothing exotic.
fn annexb_picture_starts(data: &[u8], h265: bool) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 4 < data.len() {
        if data[i..i + 3] == [0, 0, 1] {
            let (is_slice, first) = if h265 {
                let t = (data[i + 3] >> 1) & 0x3F;
                (t < 32, data.get(i + 5).is_some_and(|b| b & 0x80 != 0))
            } else {
                let t = data[i + 3] & 0x1F;
                (
                    (t == 1 || t == 5),
                    data.get(i + 4).is_some_and(|b| b & 0x80 != 0),
                )
            };
            if is_slice && first {
                // Include a 4-byte start code's leading zero.
                starts.push(if i > 0 && data[i - 1] == 0 { i - 1 } else { i });
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    starts
}

/// Offsets of the temporal delimiters in a raw AV1 OBU stream.
fn av1_temporal_units(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let header = data[pos];
        let mut cursor = pos + 1 + ((header >> 2) & 1) as usize;
        let mut size = 0usize;
        for i in 0..8 {
            let b = data[cursor];
            cursor += 1;
            size |= ((b & 0x7F) as usize) << (7 * i);
            if b & 0x80 == 0 {
                break;
            }
        }
        if (header >> 3) & 0xF == 2 {
            starts.push(pos);
        }
        pos = cursor + size;
    }
    starts
}

/// A client that joins a stream partway sends pictures that cannot be decoded
/// yet, and has to be told to ask for a keyframe, not handed an error or a
/// garbage picture.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn joining_mid_stream_asks_for_a_keyframe() {
    for (codec, path) in [
        (Codec::H264, "tests/data/bframes.264"),
        (Codec::H265, "tests/data/h265_bframes.265"),
        (Codec::AV1, "tests/data/av1_pyramid.obu"),
    ] {
        let Some(context) = context_for(codec) else {
            continue;
        };
        let data = std::fs::read(path).unwrap();
        let starts = match codec {
            Codec::H264 => annexb_picture_starts(&data, false),
            Codec::H265 => annexb_picture_starts(&data, true),
            Codec::AV1 => av1_temporal_units(&data),
        };
        // The second picture, whole.
        let (second, third) = (starts[1], starts[2]);
        // Whole coded frames, as a packet-based client would send them.
        let config = match codec {
            Codec::H264 => DecodeConfig::h264(),
            Codec::H265 => DecodeConfig::h265(),
            Codec::AV1 => DecodeConfig::av1(),
        };
        let mut decoder = Decoder::new(context, config).unwrap();
        let status = decoder.decode(&data[second..third], 0).unwrap();
        assert_eq!(status, DecodeStatus::NeedsKeyframe, "{path}");
        decoder.finish().unwrap();
        assert!(
            pollster::block_on(decoder.next_frame()).unwrap().is_none(),
            "{path}: nothing may come out of a stream that never reached a key frame"
        );
    }
}
