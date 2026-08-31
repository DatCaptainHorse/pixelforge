//! Encode with pixelforge, then decode the result with pixelforge.
//!
//! The decoder must handle anything this project's encoder can produce. That is
//! not covered by decoding third-party streams: the encoder makes its own
//! choices (notably explicit reference marking for B-frames), so it needs its
//! own round trip.
//!
//! Usage:
//!   roundtrip_h264 <input.yuv> <width> <height> <b_frames> <gop> <out.h264> [out.yuv]
//!
//! The input is planar YUV420 (I420). The decoded output is NV12 in display
//! order, so it can be compared against ffmpeg decoding the same bitstream:
//!
//!   ffmpeg -i out.h264 -pix_fmt nv12 reference.yuv
//!   cmp reference.yuv out.yuv

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};

#[allow(dead_code)]
mod common;
use common::Readback;

use pixelforge::decoder::{DecodeConfig, Decoder, FramePoll};
use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};

/// Bytes handed to the decoder per call, standing in for a network read.
const CHUNK_SIZE: usize = 64 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        eprintln!(
            "usage: roundtrip_h264 <input.yuv> <width> <height> <b_frames> <gop> <out.h264> [out.yuv]"
        );
        std::process::exit(1);
    }
    let input_path = &args[0];
    let width: u32 = args[1].parse()?;
    let height: u32 = args[2].parse()?;
    let b_frames: u32 = args[3].parse()?;
    let gop: u32 = args[4].parse()?;
    let bitstream_path = &args[5];
    let yuv_path = args.get(6);

    let context = VideoContextBuilder::new()
        .app_name("pixelforge-roundtrip")
        .require_encode(Codec::H264)
        .require_decode(Codec::H264)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;

    // --- Encode ---
    let mut yuv = Vec::new();
    File::open(input_path)?.read_to_end(&mut yuv)?;
    let frame_size = (width * height * 3 / 2) as usize;
    let frame_count = yuv.len() / frame_size;

    let config = EncodeConfig::h264(width, height)
        .with_rate_control(RateControlMode::Cqp)
        .with_quality_level(26)
        .with_frame_rate(30, 1)
        .with_gop_size(gop)
        .with_b_frames(b_frames);

    let mut input_image = InputImage::new(
        context.clone(),
        Codec::H264,
        width,
        height,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;
    let mut encoder = Encoder::new(context.clone(), config)?;

    let mut bitstream = Vec::new();
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    for i in 0..frame_count {
        input_image.upload_yuv420(&yuv[i * frame_size..(i + 1) * frame_size])?;
        pending.push_back(encoder.encode(input_image.image())?);
        while pending.len() > 2 {
            let packet = pollster::block_on(pending.pop_front().expect("non-empty"))?;
            bitstream.extend_from_slice(&packet.data);
        }
    }
    encoder.flush()?;
    while let Some(future) = pending.pop_front() {
        let packet = pollster::block_on(future)?;
        bitstream.extend_from_slice(&packet.data);
    }
    File::create(bitstream_path)?.write_all(&bitstream)?;
    println!(
        "Encoded {} frames -> {} bytes (b_frames={}, gop={})",
        frame_count,
        bitstream.len(),
        b_frames,
        gop
    );

    // --- Decode it back ---
    // Display order by default, so frames come out ready to write; `flush`
    // drains whatever the reorder buffer still holds at end of stream.
    let mut out = yuv_path.map(File::create).transpose()?;
    let mut readback = out.is_some().then(|| Readback::new(&context)).transpose()?;
    let mut decoder = Decoder::new(context, DecodeConfig::h264().with_byte_stream())?;
    let mut decoded_count = 0usize;

    let write = |frame: &pixelforge::decoder::DecodedFrame,
                 readback: &mut Option<Readback>,
                 out: &mut Option<File>,
                 count: &mut usize|
     -> Result<(), Box<dyn std::error::Error>> {
        if let (Some(file), Some(readback)) = (out.as_mut(), readback.as_mut()) {
            let data = readback.read(frame)?;
            file.write_all(&data.y)?;
            file.write_all(&data.uv)?;
        }
        *count += 1;
        Ok(())
    };

    // Feed a chunk, then take whatever has become ready; `Pending` means the
    // GPU is still working, so those frames are collected on a later pass.
    for (i, chunk) in bitstream.chunks(CHUNK_SIZE).enumerate() {
        let _ = decoder.decode(chunk, i as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            write(&frame, &mut readback, &mut out, &mut decoded_count)?;
        }
    }
    // End of stream: decodes the trailing picture, emits what reordering held
    // back, and closes the source.
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        write(&frame, &mut readback, &mut out, &mut decoded_count)?;
    }
    println!("Decoded {} frames", decoded_count);

    if decoded_count != frame_count {
        return Err(format!(
            "round trip lost frames: encoded {} but decoded {}",
            frame_count, decoded_count
        )
        .into());
    }
    if let Some(path) = yuv_path {
        println!("Wrote decoded frames to {}", path);
    }

    Ok(())
}
