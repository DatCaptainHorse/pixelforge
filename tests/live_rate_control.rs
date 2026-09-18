//! Retuning a bitrate-controlled encode while it runs.
//!
//! Encodes at a high CBR target, calls [`Encoder::set_target_bitrate`] partway
//! through to drop it by an order of magnitude, and keeps encoding the same
//! frames. Two things have to be true afterwards, and they are the two halves of
//! the feature:
//!
//!  - **The frames after the retune are substantially smaller.** Otherwise the
//!    call did nothing, which is what a hardcoded QP floor used to guarantee:
//!    H.264 pinned `min_qp = 18` under CBR, so a target below what that quality
//!    costs was unreachable and the encoder quietly emitted ten times what was
//!    asked for.
//!  - **No keyframe is emitted to do it.** A caller adapting to a congested path
//!    retunes repeatedly, and a keyframe per adjustment is the largest frame
//!    there is landing on the path least able to carry it. An implementation
//!    that rebuilt the encoder would pass the first check and fail this one.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test --test live_rate_control -- --ignored`.
//!
//! On Intel Alchemist under Mesa the video queues are hidden unless
//! `ANV_DEBUG=video-encode,video-decode` is set, and the test will skip rather
//! than fail without it.

use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};
use std::collections::VecDeque;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAMES: u64 = 120;
/// Frame at which the target drops. Late enough that the opening IDR and the
/// encoder's initial settling are behind us.
const RETUNE_AT: u64 = 60;

const HIGH_BPS: u32 = 8_000_000;
const LOW_BPS: u32 = 1_000_000;

/// Raw frames of real gameplay, 1280x720 YUV420 planar.
///
/// Real content, not a synthetic pattern, and the difference is not cosmetic.
/// Noise is incompressible: 720p noise cannot be encoded below about 2.8 Mbps at
/// any QP, so a low target is unreachable for reasons that have nothing to do
/// with rate control and the test cannot tell a working knob from a stuck one.
/// A static clip fails the other way -- its P-frames cost a couple of hundred
/// bytes whatever the target is, so no target ever binds. Gameplay has enough
/// detail and motion that the target is the thing deciding the frame size.
///
/// Override with `PIXELFORGE_TEST_CLIP`. Generate one with:
/// `ffmpeg -i <clip>.mkv -frames:v 180 -pix_fmt yuv420p -f rawvideo <out>.yuv`
const DEFAULT_CLIP: &str = "testdata/cp2077_1280x720_yuv420p.yuv";

/// How much smaller the low-bitrate half must be to count as a real change.
///
/// Deliberately far below the 10x ratio between the two targets. A hardware
/// rate controller is not obliged to hit a number exactly and this test is not
/// a conformance check -- it distinguishes "the knob moved" from "the knob was
/// wired to nothing", and the failure it exists for produced no change at all.
const MIN_SHRINK: f64 = 4.0;

#[test]
#[ignore = "requires a Vulkan Video device"]
fn retuning_bitrate_takes_effect_without_a_keyframe() -> Result<(), Box<dyn std::error::Error>> {
    let Some((clip, frame_size)) = load_clip() else {
        println!(
            "skipped: no raw clip at {DEFAULT_CLIP} (set PIXELFORGE_TEST_CLIP). \
             See DEFAULT_CLIP for how to make one."
        );
        return Ok(());
    };

    let context = VideoContextBuilder::new()
        .app_name("Live Rate Control")
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    let mut ran = 0;
    let mut failures = Vec::new();
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            println!("{codec:?}: skipped (encode not supported)");
            continue;
        }
        ran += 1;
        match run_codec(&context, codec, &clip, frame_size) {
            Ok(report) => println!("{codec:?}: ok -- {report}"),
            Err(e) => {
                println!("{codec:?}: FAIL: {e}");
                failures.push(format!("{codec:?}: {e}"));
            }
        }
    }

    if ran == 0 {
        println!(
            "no codec offered an encode queue; on Intel Alchemist try \
             ANV_DEBUG=video-encode,video-decode"
        );
        return Ok(());
    }
    if !failures.is_empty() {
        return Err(format!("live rate control failed: {}", failures.join("; ")).into());
    }
    Ok(())
}

/// Periodic IDR interval for the GOP test, in frames.
const GOP_FRAMES: u32 = 15;

/// The clip as raw frames, or `None` when it is not on this machine.
fn load_clip() -> Option<(Vec<u8>, usize)> {
    let path = std::env::var("PIXELFORGE_TEST_CLIP").unwrap_or_else(|_| DEFAULT_CLIP.to_string());
    let raw = std::fs::read(&path).ok()?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    if raw.len() < frame_size {
        return None;
    }
    Some((raw, frame_size))
}

fn run_codec(
    context: &pixelforge::VideoContext,
    codec: Codec,
    clip: &[u8],
    frame_size: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let config = match codec {
        Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
        Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
        Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
    }
    .with_rate_control(RateControlMode::Cbr)
    .with_target_bitrate(HIGH_BPS)
    .with_frame_rate(60, 1)
    .with_pixel_format(PixelFormat::Yuv420)
    .with_bit_depth(EncodeBitDepth::Eight)
    // Infinite GOP: the only keyframe that may appear is the opening one, so a
    // keyframe anywhere after the retune can only have come from the retune.
    .with_gop_size(0)
    .with_b_frames(0);

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = InputImage::new(
        context.clone(),
        codec,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;

    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    // (pts, bytes, is_key_frame) for every packet, so the two halves can be
    // compared after the fact rather than accumulated into a number that hides
    // which frame did what.
    let mut packets: Vec<(u64, usize, bool)> = Vec::new();

    let drain_one = |pending: &mut VecDeque<pixelforge::EncodeFuture>,
                     packets: &mut Vec<(u64, usize, bool)>|
     -> Result<(), Box<dyn std::error::Error>> {
        let packet = pollster::block_on(pending.pop_front().unwrap())?;
        packets.push((packet.pts, packet.data.len(), packet.is_key_frame));
        Ok(())
    };

    for i in 0..FRAMES {
        if i == RETUNE_AT {
            encoder.set_target_bitrate(LOW_BPS)?;
        }

        // The clip loops if it is shorter than the run, so both halves encode
        // the same pictures and a size difference between them cannot be
        // content.
        let frames_in_clip = clip.len() / frame_size;
        let start = (i as usize % frames_in_clip) * frame_size;
        let encoder_image = encoder.input_image();
        input_image.upload_yuv420_to(encoder_image, &clip[start..start + frame_size])?;
        pending.push_back(encoder.encode(encoder_image)?);
        while pending.len() > 2 {
            drain_one(&mut pending, &mut packets)?;
        }
    }

    encoder.flush()?;
    while !pending.is_empty() {
        drain_one(&mut pending, &mut packets)?;
    }

    // Skip the frames either side of the boundary: the encoder's rate
    // controller needs a moment to converge, and measuring during it would make
    // this test about how fast it converges rather than whether it moved.
    const SETTLE: u64 = 8;
    let mean = |lo: u64, hi: u64| -> f64 {
        let taken: Vec<usize> = packets
            .iter()
            .filter(|(pts, _, is_key)| *pts >= lo && *pts < hi && !*is_key)
            .map(|(_, len, _)| *len)
            .collect();
        if taken.is_empty() {
            return 0.0;
        }
        taken.iter().sum::<usize>() as f64 / taken.len() as f64
    };

    let before = mean(SETTLE, RETUNE_AT);
    let after = mean(RETUNE_AT + SETTLE, FRAMES);
    if before == 0.0 || after == 0.0 {
        return Err("one half of the run produced no delta frames to compare".into());
    }

    // The negative first, because it is the one a plausible-looking
    // implementation gets wrong.
    if let Some((pts, _, _)) = packets
        .iter()
        .find(|(pts, _, is_key)| *is_key && *pts >= RETUNE_AT)
    {
        return Err(format!(
            "retuning emitted a keyframe at pts {pts}; a live retune must not \
             reset the session or rebuild the encoder"
        )
        .into());
    }

    let shrink = before / after;
    if shrink < MIN_SHRINK {
        return Err(format!(
            "target went {HIGH_BPS} -> {LOW_BPS} bps ({:.1}x) but mean delta frame \
             went {before:.0} -> {after:.0} bytes ({shrink:.2}x); the encoder did not \
             follow the new target",
            HIGH_BPS as f64 / LOW_BPS as f64,
        )
        .into());
    }

    Ok(format!(
        "mean delta {before:.0} -> {after:.0} bytes ({shrink:.1}x smaller), no keyframe"
    ))
}

/// Turning periodic key frames off, live.
///
/// The controller's second lever: when keyframes are what a path cannot carry,
/// fewer of them beats smaller ones. `None` has to actually stop them, and an
/// explicit request has to keep working afterwards -- a stream that can never
/// produce a key frame again cannot recover a client that has lost sync.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn turning_off_periodic_keyframes_takes_effect() -> Result<(), Box<dyn std::error::Error>> {
    let Some((clip, frame_size)) = load_clip() else {
        println!("skipped: no raw clip at {DEFAULT_CLIP} (set PIXELFORGE_TEST_CLIP)");
        return Ok(());
    };
    let context = VideoContextBuilder::new()
        .app_name("Live GOP")
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    let mut failures = Vec::new();
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            continue;
        }
        match run_gop_codec(&context, codec, &clip, frame_size) {
            Ok(report) => println!("{codec:?}: ok -- {report}"),
            Err(e) => {
                println!("{codec:?}: FAIL: {e}");
                failures.push(format!("{codec:?}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        return Err(format!("live gop failed: {}", failures.join("; ")).into());
    }
    Ok(())
}

fn run_gop_codec(
    context: &pixelforge::VideoContext,
    codec: Codec,
    clip: &[u8],
    frame_size: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let config = match codec {
        Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
        Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
        Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
    }
    .with_rate_control(RateControlMode::Cbr)
    .with_target_bitrate(HIGH_BPS)
    .with_frame_rate(60, 1)
    .with_pixel_format(PixelFormat::Yuv420)
    .with_bit_depth(EncodeBitDepth::Eight)
    .with_gop_size(GOP_FRAMES)
    .with_b_frames(0);

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = InputImage::new(
        context.clone(),
        codec,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;

    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    let mut keys: Vec<u64> = Vec::new();
    let frames_in_clip = clip.len() / frame_size;

    let mut drain = |pending: &mut VecDeque<pixelforge::EncodeFuture>,
                     keys: &mut Vec<u64>|
     -> Result<(), Box<dyn std::error::Error>> {
        let p = pollster::block_on(pending.pop_front().unwrap())?;
        if p.is_key_frame {
            keys.push(p.pts);
        }
        Ok(())
    };

    for i in 0..FRAMES {
        if i == RETUNE_AT {
            encoder.set_gop_size(None);
        }
        let start = (i as usize % frames_in_clip) * frame_size;
        let image = encoder.input_image();
        input_image.upload_yuv420_to(image, &clip[start..start + frame_size])?;
        pending.push_back(encoder.encode(image)?);
        while pending.len() > 2 {
            drain(&mut pending, &mut keys)?;
        }
    }
    encoder.flush()?;
    while !pending.is_empty() {
        drain(&mut pending, &mut keys)?;
    }

    let before: Vec<u64> = keys.iter().copied().filter(|p| *p < RETUNE_AT).collect();
    let after: Vec<u64> = keys.iter().copied().filter(|p| *p >= RETUNE_AT).collect();

    if before.len() < 2 {
        return Err(format!(
            "expected periodic keyframes every {GOP_FRAMES} frames before the change, saw {before:?}"
        )
        .into());
    }
    if !after.is_empty() {
        return Err(
            format!("periodic keyframes continued after set_gop_size(None): {after:?}").into(),
        );
    }
    Ok(format!("{} keyframes before, none after", before.len()))
}
