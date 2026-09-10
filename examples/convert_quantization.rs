//! Example: Measure the color converter's quantization error
//!
//! The conversion shader turns floating-point YUV into integer code values.
//! How it does that rounding is invisible from outside the library, and getting
//! it wrong is a silent, uniform darkening rather than an obvious failure, so
//! this measures it directly.
//!
//! A synthetic frame goes through [`ColorConverter`], the luma plane is read
//! back off the GPU, and each sample is scored against the same BT.709 math
//! computed on the CPU. Two hypotheses are tested per sample: that the shader
//! rounds, and that it truncates. A correct shader matches the rounding
//! prediction and shows a mean signed error near zero; a truncating one matches
//! the other prediction and carries a bias of about half a code value.
//!
//! Luma only. It is one sample per pixel, whereas NV12 and P010 chroma is a 2x2
//! average, which would fold averaging error into the measurement.
//!
//! Non-grey colours on purpose: grey converts to exact code values under
//! BT.709, so a grey ramp cannot tell rounding from truncation at all.
//!
//! ```text
//! cargo run --example convert_quantization
//! ```

use ash::vk;
use pixelforge::{
    Codec, ColorConverter, ColorConverterConfig, ColorRange, EncodeBitDepth, EncodeConfig, Encoder,
    InputFormat, OutputFormat, RateControlMode, VideoContext, VideoContextBuilder,
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 64;

/// Deterministic colours that are never grey, so every channel differs and the
/// luma lands between code values often enough to be worth measuring.
fn colour_at(x: u32, y: u32) -> [u8; 3] {
    [
        (x ^ y.wrapping_mul(37)) as u8,
        x.wrapping_mul(7).wrapping_add(y.wrapping_mul(11)) as u8,
        x.wrapping_mul(13).wrapping_add(y.wrapping_mul(3)) as u8,
    ]
}

fn make_frame_bgra() -> Vec<u8> {
    let mut data = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let [r, g, b] = colour_at(x, y);
            data.extend_from_slice(&[b, g, r, 255]);
        }
    }
    data
}

/// The shader's BT.709 luma, in f32, in the same operation order.
fn luma(rgb: [u8; 3]) -> f32 {
    let r = rgb[0] as f32 / 255.0;
    let g = rgb[1] as f32 / 255.0;
    let b = rgb[2] as f32 / 255.0;
    (0.2126f32 * r + 0.7152f32 * g + 0.0722f32 * b).clamp(0.0, 1.0)
}

/// The unquantized code value the shader is aiming for, matching the scale and
/// offset of whichever quantizer branch applies.
fn ideal_code(y: f32, ten_bit: bool, full_range: bool) -> f32 {
    match (ten_bit, full_range) {
        (false, true) => (y * 255.0).clamp(0.0, 255.0),
        (false, false) => (y * 219.0 + 16.0).clamp(0.0, 255.0),
        (true, true) => (y * 1023.0).clamp(0.0, 1023.0),
        (true, false) => (y * 876.0 + 64.0).clamp(0.0, 1023.0),
    }
}

struct SrcImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

struct Score {
    if_truncating: usize,
    if_rounding: usize,
    mean_signed_error: f64,
    mean_abs_error: f64,
}

unsafe fn one_shot<F: FnOnce(vk::CommandBuffer)>(
    context: &VideoContext,
    record: F,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = context.device();
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(context.transfer_queue_family())
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { device.create_command_pool(&pool_info, None) }?;
    let cb_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe { device.allocate_command_buffers(&cb_info) }?[0];
    unsafe {
        device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        record(cb);
        device.end_command_buffer(cb)?;
        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        device.queue_submit(context.transfer_queue(), &[submit], vk::Fence::null())?;
        device.queue_wait_idle(context.transfer_queue())?;
        device.destroy_command_pool(pool, None);
    }
    Ok(())
}

unsafe fn host_buffer(
    context: &VideoContext,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
    let device = context.device();
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type = context
        .find_memory_type(
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("no host visible memory")?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
    Ok((buffer, memory))
}

unsafe fn create_src_image(context: &VideoContext) -> Result<SrcImage, Box<dyn std::error::Error>> {
    let device = context.device();
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .extent(vk::Extent3D {
            width: WIDTH,
            height: HEIGHT,
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
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = context
        .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device local memory")?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }?;
    unsafe { device.bind_image_memory(image, memory, 0) }?;

    let pixels = make_frame_bgra();
    let (staging, staging_mem) = unsafe {
        host_buffer(
            context,
            pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?
    };
    unsafe {
        let ptr = device.map_memory(
            staging_mem,
            0,
            pixels.len() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr as *mut u8, pixels.len());
        device.unmap_memory(staging_mem);
    }

    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    unsafe {
        one_shot(context, |cb| {
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: WIDTH,
                    height: HEIGHT,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cb,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        })?;
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
    }
    Ok(SrcImage { image, memory })
}

/// Convert one frame and read the luma plane back as code values.
fn convert_and_read_luma(
    context: &VideoContext,
    src: &SrcImage,
    encoder: &Encoder,
    output_format: OutputFormat,
    full_range: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let config = ColorConverterConfig::new(WIDTH, HEIGHT, InputFormat::BGRA, output_format)
        .with_range(if full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        });
    let mut converter = ColorConverter::new(context.clone(), config)?;
    converter.convert(
        src.image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        encoder.input_image(),
    )?;

    let sample_bytes = output_format.bytes_per_sample() as u64;
    let plane_bytes = (WIDTH * HEIGHT) as u64 * sample_bytes;
    let device = context.device();
    let (readback, readback_mem) =
        unsafe { host_buffer(context, plane_bytes, vk::BufferUsageFlags::TRANSFER_DST)? };
    unsafe {
        one_shot(context, |cb| {
            let copy = vk::BufferCopy::default().size(plane_bytes);
            device.cmd_copy_buffer(cb, converter.output_buffer(), readback, &[copy]);
        })?;
    }

    let mut bytes = vec![0u8; plane_bytes as usize];
    unsafe {
        let ptr = device.map_memory(readback_mem, 0, plane_bytes, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, bytes.as_mut_ptr(), bytes.len());
        device.unmap_memory(readback_mem);
        device.destroy_buffer(readback, None);
        device.free_memory(readback_mem, None);
    }

    // 10-bit output uses the P010 layout: the value sits in the upper 10 bits
    // of a little-endian 16-bit word.
    let codes = if sample_bytes == 2 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| (u16::from_le_bytes(*w) >> 6) as u32)
            .collect()
    } else {
        bytes.iter().map(|&b| b as u32).collect()
    };
    Ok(codes)
}

fn score(codes: &[u32], ten_bit: bool, full_range: bool) -> Score {
    let mut if_truncating = 0;
    let mut if_rounding = 0;
    let mut signed_error = 0f64;
    let mut abs_error = 0f64;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let got = codes[(y * WIDTH + x) as usize];
            let ideal = ideal_code(luma(colour_at(x, y)), ten_bit, full_range);
            if got != ideal as u32 {
                if_truncating += 1;
            }
            if got != ideal.round() as u32 {
                if_rounding += 1;
            }
            signed_error += got as f64 - ideal as f64;
            abs_error += (got as f64 - ideal as f64).abs();
        }
    }
    let n = (WIDTH * HEIGHT) as f64;
    Score {
        if_truncating,
        if_rounding,
        mean_signed_error: signed_error / n,
        mean_abs_error: abs_error / n,
    }
}

fn report(label: &str, s: &Score) -> bool {
    let verdict = if s.if_rounding <= s.if_truncating {
        "rounds"
    } else {
        "TRUNCATES"
    };
    println!(
        "{label:<22} mismatches: {:>5} if truncating, {:>5} if rounding | \
mean signed error {:+.4}, mean |error| {:.4} | {verdict}",
        s.if_truncating, s.if_rounding, s.mean_signed_error, s.mean_abs_error
    );
    s.if_rounding <= s.if_truncating
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let context = VideoContextBuilder::new()
        .app_name("convert_quantization")
        .require_encode(Codec::H264)
        .build()?;
    let src = unsafe { create_src_image(&context)? };

    println!(
        "{} samples per case, BT.709 luma, {WIDTH}x{HEIGHT}\n",
        WIDTH * HEIGHT
    );

    let mut all_round = true;

    let encoder_8bit = Encoder::new(
        context.clone(),
        EncodeConfig::h264(WIDTH, HEIGHT)
            .with_rate_control(RateControlMode::Cqp)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )?;
    for full_range in [false, true] {
        let codes = convert_and_read_luma(
            &context,
            &src,
            &encoder_8bit,
            OutputFormat::NV12,
            full_range,
        )?;
        let label = format!("NV12 {}", if full_range { "full" } else { "limited" });
        all_round &= report(&label, &score(&codes, false, full_range));
    }

    // 10-bit needs a Main10-capable encoder for the conversion target. Not
    // every driver exposes one, and the 8-bit cases are worth having on their
    // own, so a missing one is reported rather than fatal.
    let encoder_10bit = Encoder::new(
        context.clone(),
        EncodeConfig::h265(WIDTH, HEIGHT)
            .with_bit_depth(EncodeBitDepth::Ten)
            .with_rate_control(RateControlMode::Cqp)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    );
    match encoder_10bit {
        Ok(encoder) => {
            for full_range in [false, true] {
                let codes = convert_and_read_luma(
                    &context,
                    &src,
                    &encoder,
                    OutputFormat::P010,
                    full_range,
                )?;
                let label = format!("P010 {}", if full_range { "full" } else { "limited" });
                all_round &= report(&label, &score(&codes, true, full_range));
            }
        }
        Err(e) => println!("P010 cases skipped, no 10-bit H.265 encoder here: {e}"),
    }

    unsafe {
        let device = context.device();
        device.destroy_image(src.image, None);
        device.free_memory(src.memory, None);
    }

    if all_round {
        println!("\nAll cases round. A mean signed error near zero is the point:");
        println!("truncation shows up as a bias of about half a code value.");
        Ok(())
    } else {
        Err("at least one quantizer truncates, see the cases marked TRUNCATES above".into())
    }
}
