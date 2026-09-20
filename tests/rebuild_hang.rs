//! Replacing an encoder while its frames are still in flight.
//!
//! nescapture rebuilds the encoder when a game changes resolution, and holds
//! the futures it has already produced in a bounded channel that another
//! thread drains. So at the moment of the rebuild there are submissions
//! outstanding that nobody has polled yet.
//!
//! `EncodePipeline::flush` is documented as the thing to call "before
//! mutating shared session state", and the rebuild path does not call it. This
//! reproduces that shape: submit, keep the futures, build a second encoder at
//! a different size, and only then resolve them.
//!
//! Ignored by default: requires a Vulkan Video device.

use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};

fn config(w: u32, h: u32) -> EncodeConfig {
    EncodeConfig::h264(w, h)
        .with_rate_control(RateControlMode::Cbr)
        .with_target_bitrate(4_000_000)
        .with_frame_rate(60, 1)
        .with_pixel_format(PixelFormat::Yuv420)
        .with_bit_depth(EncodeBitDepth::Eight)
        .with_gop_size(30)
        .with_b_frames(0)
}

fn encode_some(
    context: &pixelforge::VideoContext,
    w: u32,
    h: u32,
    count: usize,
) -> Result<(Encoder, Vec<pixelforge::EncodeFuture>), Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(context.clone(), config(w, h))?;
    let mut input = InputImage::new(
        context.clone(),
        Codec::H264,
        w,
        h,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;
    let frame = vec![0x40u8; (w as usize * h as usize * 3) / 2];
    let mut pending = Vec::new();
    for _ in 0..count {
        let image = encoder.input_image();
        input.upload_yuv420_to(image, &frame)?;
        pending.push(encoder.encode(image)?);
    }
    Ok((encoder, pending))
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn an_encoder_can_be_replaced_while_its_frames_are_still_in_flight()
-> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .app_name("Rebuild")
        .enable_validation(false)
        .build()?;
    if !context.supports_encode(Codec::H264) {
        println!("skipped: no H.264 encode");
        return Ok(());
    }

    // Two frames outstanding and unpolled, matching the depth of the channel
    // nescapture hands them to.
    let (old, pending) = encode_some(&context, 1280, 720, 2)?;

    // The game changed resolution. A new encoder is built while the old one
    // still exists, and the old one is then dropped.
    let (new, new_pending) = encode_some(&context, 1920, 1080, 1)?;
    drop(old);

    // Whatever was outstanding must still resolve. If this hangs, a rebuild
    // strands the frames that were in flight across it -- and in nescapture
    // that stalls the thread draining them, which fills the channel, which
    // stops the encoder thread, with nothing logged anywhere.
    for f in pending {
        let _ = pollster::block_on(f)?;
    }
    for f in new_pending {
        let _ = pollster::block_on(f)?;
    }
    drop(new);
    Ok(())
}
