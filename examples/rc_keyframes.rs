//! Example: Mid-Stream Key Frames Under Rate Control
//!
//! Encodes AV1 with CBR and a short GOP so key frames land mid-stream, then
//! verifies a full-stream ffmpeg decode plus PSNR against the source. NVIDIA
//! emitted undecodable mid-stream key frames unless the coding state is reset
//! on every key frame; the PSNR floor also catches the decodes-but-corrupt case.

use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};
use std::process::Command;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: u64 = 30;
/// Short GOP so key frames land mid-stream (frames 0, 10 and 20).
const GOP_SIZE: u32 = 10;
/// PSNR below this means a key frame corrupted the stream without a hard error.
const MIN_PSNR: f64 = 30.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("PixelForge mid-stream key frame (rate control) verification\n");

    let input_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv420p.yuv");
    ensure_test_data("yuv420p", &input_path)?;

    let context = VideoContextBuilder::new()
        .app_name("RC key frame verification")
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    if !context.supports_encode(Codec::AV1) {
        println!("AV1: skipped (encode not supported)");
        return Ok(());
    }

    let output_filename = "output_rc_keyframes_AV1.obu";
    let decoded_filename = "decoded_rc_keyframes_AV1.yuv";

    let config = EncodeConfig::av1(WIDTH, HEIGHT)
        .with_rate_control(RateControlMode::Cbr)
        .with_pixel_format(PixelFormat::Yuv420)
        .with_bit_depth(EncodeBitDepth::Eight)
        .with_gop_size(GOP_SIZE)
        .with_b_frames(0);

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = InputImage::new(
        context.clone(),
        Codec::AV1,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;

    let mut yuv_data = Vec::new();
    File::open(&input_path)?.read_to_end(&mut yuv_data)?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;

    let mut output_file = File::create(output_filename)?;
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    let mut key_frames = 0u32;

    let drain_one = |pending: &mut VecDeque<pixelforge::EncodeFuture>,
                     output_file: &mut File,
                     key_frames: &mut u32|
     -> Result<(), Box<dyn std::error::Error>> {
        let packet = pollster::block_on(pending.pop_front().unwrap())?;
        if packet.is_key_frame {
            *key_frames += 1;
        }
        output_file.write_all(&packet.data)?;
        Ok(())
    };

    for i in 0..FRAMES {
        let start = (i as usize) * frame_size;
        let end = start + frame_size;
        if end > yuv_data.len() {
            break;
        }

        let encoder_image = encoder.input_image();
        input_image.upload_yuv420_to(encoder_image, &yuv_data[start..end])?;
        pending.push_back(encoder.encode(encoder_image)?);
        while pending.len() > 2 {
            drain_one(&mut pending, &mut output_file, &mut key_frames)?;
        }
    }

    encoder.flush()?;
    while !pending.is_empty() {
        drain_one(&mut pending, &mut output_file, &mut key_frames)?;
    }
    drop(output_file);

    if key_frames < 2 {
        return Err(format!(
            "only {key_frames} key frame(s) produced — the mid-stream key frame scenario never engaged"
        )
        .into());
    }

    let psnr = decode_and_psnr(output_filename, decoded_filename, &input_path)?;
    std::fs::remove_file(output_filename).ok();
    std::fs::remove_file(decoded_filename).ok();

    if psnr < MIN_PSNR {
        return Err(format!(
            "full-stream PSNR {psnr:.2} dB below {MIN_PSNR} dB (a mid-stream key frame corrupted the stream)"
        )
        .into());
    }

    println!("AV1: PASS — {key_frames} key frames under CBR, full-stream PSNR {psnr:.2} dB");
    Ok(())
}

/// Decode the bitstream to raw YUV and return its PSNR against the source.
fn decode_and_psnr(
    bitstream: &str,
    decoded: &str,
    source: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            bitstream,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            decoded,
        ])
        .output()?;
    if !status.status.success() {
        return Err(format!(
            "ffmpeg decode failed: {}",
            String::from_utf8_lossy(&status.stderr)
        )
        .into());
    }

    let size = format!("{WIDTH}x{HEIGHT}");
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "info",
            "-s",
            &size,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-i",
            source,
            "-s",
            &size,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-i",
            decoded,
            "-lavfi",
            "psnr",
            "-f",
            "null",
            "-",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let pos = stderr
        .find("average:")
        .ok_or_else(|| format!("could not parse PSNR: {stderr}"))?;
    let rest = &stderr[pos + 8..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Ok(rest[..end].parse()?)
}

fn ensure_test_data(pix_fmt: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if std::path::Path::new(path).exists() {
        return Ok(());
    }
    println!("Generating {path}...");
    let status = Command::new("ffmpeg")
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=duration=1:size={WIDTH}x{HEIGHT}:rate=30"),
            "-pix_fmt",
            pix_fmt,
            "-f",
            "rawvideo",
            "-y",
            path,
        ])
        .output()?;
    if !status.status.success() {
        return Err(format!("failed to generate test data: {status:?}").into());
    }
    Ok(())
}
