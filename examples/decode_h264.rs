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
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-decode")
        .require_decode(Codec::H264)
        .enable_validation(validation)
        .build()?;

    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(context.device_properties().device_name.as_ptr()) }
            .to_string_lossy()
    );

    // Display order by default: frames arrive ready to present, no sorting.
    // Set PIXELFORGE_DECODE_ORDER=1 for the low-latency decode-order path.
    let config = if std::env::var("PIXELFORGE_DECODE_ORDER").is_ok() {
        DecodeConfig::h264().with_decode_order()
    } else {
        DecodeConfig::h264()
    };
    let mut decoder = Decoder::new(context, config)?;
    let stream = std::fs::read(&input_path)?;
    let mut output = output_path.as_ref().map(File::create).transpose()?;

    let start = std::time::Instant::now();
    let mut frame_count = 0usize;
    let mut last_info = None;

    // Write each frame as it comes out. `flush` at the end drains the frames
    // the reorder buffer was still holding.
    let write = |decoder: &mut Decoder,
                 frame: &pixelforge::decoder::DecodedFrame,
                 output: &mut Option<File>|
     -> Result<(), Box<dyn std::error::Error>> {
        if let Some(file) = output.as_mut() {
            let data = decoder.download(frame)?;
            file.write_all(&data.y)?;
            file.write_all(&data.uv)?;
        }
        Ok(())
    };

    for au in decoder.split(&stream) {
        let decoded = match decoder.decode(au, frame_count as u64) {
            Ok(frames) => frames,
            // Joining mid-stream (or after loss): skip until a keyframe. A live
            // client would ask the sender for an IDR here.
            Err(pixelforge::error::PixelForgeError::NeedsKeyframe(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        for frame in decoded {
            write(&mut decoder, &frame, &mut output)?;
            last_info = Some((
                frame.width,
                frame.height,
                frame.display_order,
                frame.is_keyframe,
            ));
            frame_count += 1;
        }
    }
    for frame in decoder.flush()? {
        write(&mut decoder, &frame, &mut output)?;
        last_info = Some((
            frame.width,
            frame.height,
            frame.display_order,
            frame.is_keyframe,
        ));
        frame_count += 1;
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
