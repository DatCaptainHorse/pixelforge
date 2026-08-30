//! Decode an H.264 Annex B stream to raw YUV.
//!
//! Usage:
//!   cargo run --example decode_h264 -- input.264 [output.yuv]
//!
//! Produces the same layout as `ffmpeg -i input.264 -pix_fmt nv12 out.yuv`,
//! so the output can be compared directly:
//!
//!   ffmpeg -i input.264 -pix_fmt nv12 reference.yuv
//!   cmp reference.yuv output.yuv

use std::fs::File;
use std::io::Write;

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::encoder::Codec;
use pixelforge::vulkan::VideoContextBuilder;

/// Bytes handed to the decoder per call, standing in for a network read.
const CHUNK_SIZE: usize = 64 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let input_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: decode_h264 <input.264> [output.yuv]");
        std::process::exit(1);
    });
    let output_path = args.next();

    // PIXELFORGE_VALIDATION=1 enables the Vulkan validation layers, which
    // report the exact offending call for any API misuse.
    let validation = std::env::var("PIXELFORGE_VALIDATION").is_ok();
    // Set PIXELFORGE_NO_UNIFIED_LAYOUTS=1 to exercise the copying fallback on
    // hardware that would otherwise take the zero-copy path.
    let mut builder = VideoContextBuilder::new()
        .app_name("pixelforge-decode")
        .require_decode(Codec::H264)
        .enable_validation(validation);
    if std::env::var("PIXELFORGE_NO_UNIFIED_LAYOUTS").is_ok() {
        builder = builder.without_unified_image_layouts();
    }
    let context = builder.build()?;

    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(context.device_properties().device_name.as_ptr()) }
            .to_string_lossy()
    );

    // Frames arrive in presentation order, ready to show without sorting, and
    // without ever being copied: a picture waiting its turn stays pinned in the
    // DPB slot it was decoded into.
    // 64 KB of byte stream can hold several coded frames, and every frame a
    // `decode` call emits is outstanding at once. The default budget of 4 is
    // sized for one frame per call, so raise it: past the budget the decoder
    // copies pictures out instead of handing over its own images, which still
    // decodes correctly but costs throughput.
    let config = DecodeConfig::h264().with_byte_stream().with_output_depth(8);
    let mut decoder = Decoder::new(context, config)?;
    let stream = std::fs::read(&input_path)?;
    let mut output = output_path.as_ref().map(File::create).transpose()?;

    let start = std::time::Instant::now();
    let mut frame_count = 0usize;
    let mut last_info = None;

    // Write out one resolved batch. Dropping each frame at the end of the loop
    // hands its storage back to the decoder.
    let write_batch = |decoder: &mut Decoder,
                       frames: Vec<pixelforge::decoder::DecodedFrame>,
                       output: &mut Option<File>,
                       frame_count: &mut usize,
                       last_info: &mut Option<(u32, u32, i32, bool)>|
     -> Result<(), Box<dyn std::error::Error>> {
        for frame in frames {
            if let Some(file) = output.as_mut() {
                let data = decoder.download(&frame)?;
                file.write_all(&data.y)?;
                file.write_all(&data.uv)?;
            }
            *last_info = Some((
                frame.width,
                frame.height,
                frame.display_order,
                frame.is_keyframe,
            ));
            *frame_count += 1;
        }
        Ok(())
    };

    // Each `decode()` returns a future for the frames that call produces. Keep
    // a couple in flight so parsing and submission overlap the GPU decode, and
    // drain the oldest once the pipeline is full, which preserves output order.
    //
    // The depth is bounded by `DecodeConfig::output_depth` (2 by default):
    // every un-dropped frame holds a DPB slot, so holding more batches than
    // there are output slots would make `decode` wait for a frame that only
    // this loop can release.
    let mut pending: std::collections::VecDeque<pixelforge::decoder::DecodeFuture> =
        std::collections::VecDeque::new();

    // A file is a raw byte stream that can cut anywhere, so let the decoder do
    // the framing and feed it in fixed-size chunks, the way a socket delivers
    // one. `flush` ends the final picture, which nothing follows.
    for (i, chunk) in stream.chunks(CHUNK_SIZE).enumerate() {
        match decoder.decode(chunk, i as u64) {
            Ok(batch) => pending.push_back(batch),
            // Joining mid-stream (or after loss): skip until a keyframe. A live
            // client would ask the sender for an IDR here.
            Err(pixelforge::error::PixelForgeError::NeedsKeyframe(_)) => continue,
            Err(e) => return Err(e.into()),
        }
        while pending.len() >= 2 {
            let frames = pollster::block_on(pending.pop_front().unwrap())?;
            write_batch(
                &mut decoder,
                frames,
                &mut output,
                &mut frame_count,
                &mut last_info,
            )?;
        }
    }

    // `flush` emits whatever the reorder buffer still holds; its future resolves
    // behind everything already in flight.
    pending.push_back(decoder.flush()?);
    while let Some(batch) = pending.pop_front() {
        let frames = pollster::block_on(batch)?;
        write_batch(
            &mut decoder,
            frames,
            &mut output,
            &mut frame_count,
            &mut last_info,
        )?;
    }
    let decode_time = start.elapsed();

    println!(
        "Decoded {} frames in {:.1?} ({:.1} fps)",
        frame_count,
        decode_time,
        frame_count as f64 / decode_time.as_secs_f64()
    );
    if let Some((w, h, order, key)) = last_info {
        println!(
            "Last frame: {}x{} display_order={} keyframe={}",
            w, h, order, key
        );
    }
    if let Some(path) = output_path {
        println!("Wrote raw frames to {}", path);
    }

    Ok(())
}
