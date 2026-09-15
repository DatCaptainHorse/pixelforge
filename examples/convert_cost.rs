//! Measure what a colour conversion costs the GPU.
//!
//! `ColorConverter::convert` submits a compute dispatch and waits for it, so the
//! time it takes to return is submit plus wait plus execution. On an idle GPU
//! those are nearly the same number and on a busy one they are not: the wait
//! grows with whatever else the device is doing, while the execution time is the
//! share of the GPU the conversion is actually taking away from it.
//!
//! A caller that shares one GPU between a renderer and this library wants the
//! second number, and until `last_gpu_time_ns` there was no way to read it. This
//! example prints both, so the difference is visible.
//!
//! Usage:
//!   cargo run --release --example convert_cost -- [width] [height] [frames]
//!
//! Defaults to 1920x1080 and 120 frames.

use ash::vk;
use pixelforge::{
    Codec, ColorConverter, ColorConverterConfig, EncodeConfig, Encoder, InputFormat, OutputFormat,
    VideoContextBuilder,
};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            ),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let width: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1920);
    let height: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1080);
    let frames: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(120);

    let context = VideoContextBuilder::new()
        .app_name("convert_cost")
        .require_encode(Codec::H264)
        .build()?;

    // The conversion writes into an encoder's input image, which is how every
    // caller uses it; measuring against anything else would measure a different
    // memory layout.
    let encoder = Encoder::new(context.clone(), EncodeConfig::h264(width, height))?;

    let mut converter = ColorConverter::new(
        context.clone(),
        ColorConverterConfig::new(width, height, InputFormat::BGRA, OutputFormat::NV12),
    )?;

    let (src_image, src_memory) = create_source_image(&context, width, height)?;

    println!("{width}x{height} BGRA -> NV12, {frames} conversions\n");

    let mut wall_us = Vec::with_capacity(frames);
    let mut gpu_us = Vec::with_capacity(frames);

    for i in 0..frames {
        // UNDEFINED on the first pass so the image is transitioned from its
        // freshly created state; GENERAL thereafter, which is the layout the
        // converter leaves it in — and the layout a reused DMA-BUF arrives in.
        let layout = if i == 0 {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::GENERAL
        };

        let started = std::time::Instant::now();
        converter.convert(src_image, layout, encoder.input_image())?;
        wall_us.push(started.elapsed().as_micros() as u64);

        if let Some(ns) = converter.last_gpu_time_ns() {
            gpu_us.push(ns / 1000);
        }
    }

    report("wall", &mut wall_us);
    if gpu_us.is_empty() {
        println!("gpu   unavailable — this compute queue family cannot timestamp");
    } else {
        report("gpu ", &mut gpu_us);
    }

    unsafe {
        context.device().destroy_image(src_image, None);
        context.device().free_memory(src_memory, None);
    }

    Ok(())
}

fn report(label: &str, samples: &mut [u64]) {
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p95 = samples[samples.len() * 95 / 100];
    let mean = samples.iter().sum::<u64>() / samples.len() as u64;
    println!(
        "{label}  median {:.3}ms  mean {:.3}ms  p95 {:.3}ms  max {:.3}ms",
        median as f64 / 1000.0,
        mean as f64 / 1000.0,
        p95 as f64 / 1000.0,
        samples[samples.len() - 1] as f64 / 1000.0,
    );
}

/// A device-local BGRA image for the converter to read.
///
/// Its contents are never initialised. The conversion reads every texel
/// regardless of what is in them, so the cost is the same and the output is not
/// being checked here — `examples/encode.rs` is where correctness is looked at.
fn create_source_image(
    context: &pixelforge::VideoContext,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory), Box<dyn std::error::Error>> {
    let device = context.device();

    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&info, None) }?;
    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let memory_properties = unsafe {
        context
            .instance()
            .get_physical_device_memory_properties(context.physical_device())
    };
    let memory_type = (0..memory_properties.memory_type_count)
        .find(|&i| {
            requirements.memory_type_bits & (1 << i) != 0
                && memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or("no device-local memory type for the source image")?;

    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&allocate, None) }?;
    unsafe { device.bind_image_memory(image, memory, 0) }?;

    Ok((image, memory))
}
