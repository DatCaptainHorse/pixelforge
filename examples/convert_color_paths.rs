//! Example: Check every conversion path against a model of itself
//!
//! [`ColorConverterConfig`] takes two colour decisions: a [`ColorSpec`] for
//! what the input pixels already are, and a [`ColorSpec`] for what the
//! encoded stream should be. Between them they select which of the shader's
//! stages run: an sRGB decode, a BT.709 to BT.2020 gamut hop, a PQ encode, or
//! none of them for a passthrough.
//!
//! Getting one of those stages wrong does not fail loudly. It produces a
//! plausible picture with the wrong colour, which is why this reimplements the
//! same pipeline on the CPU, stage for stage, and compares. Every supported
//! pair is checked at 8-bit and 10-bit, in both cases against a model built
//! from the source and target rather than from the shader's branches, so a
//! branch taken wrongly shows up as a large disagreement.
//!
//! It also asserts the refusals. Only [`ColorSpec::Srgb`] can reach
//! [`ColorSpec::Srgb`]; a linear or PQ source would need a forward gamma
//! encode or tone mapping, and the converter rejects those pairs rather than
//! passing the samples through and mislabelling them.
//!
//! Luma only, for the same reason as `convert_quantization`: chroma is a 2x2
//! average in NV12 and P010, which would mix averaging error into the result.
//!
//! ```text
//! cargo run --example convert_color_paths
//! ```

use ash::vk;
use pixelforge::{
    Codec, ColorConverter, ColorConverterConfig, ColorRange, ColorSpec, EncodeBitDepth,
    EncodeConfig, Encoder, InputFormat, OutputFormat, RateControlMode, VideoContext,
    VideoContextBuilder,
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

// --- CPU model of each shader stage, same order, same constants. ---

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn bt709_to_bt2020(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    [
        0.6274 * r + 0.3293 * g + 0.0433 * b,
        0.0691 * r + 0.9195 * g + 0.0114 * b,
        0.0164 * r + 0.0880 * g + 0.8956 * b,
    ]
}

fn linear_to_pq(l: f32) -> f32 {
    // Computed in f64. This is the reference; the shader's own f32 rounding is
    // what the one-code-value tolerance covers.
    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;
    let lm1 = (l.max(0.0) as f64).powf(M1);
    (((C1 + C2 * lm1) / (1.0 + C3 * lm1)).powf(M2)) as f32
}

/// The signal the shader should arrive at for this source and target, then its
/// luma, then its code value. Mirrors `read_rgb` and `rgb_to_yuv`.
fn expected_code(
    rgb: [u8; 3],
    source: ColorSpec,
    target: ColorSpec,
    ten_bit: bool,
    full: bool,
) -> f32 {
    let mut v = [
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    ];
    if target == ColorSpec::Bt2020Pq && source != ColorSpec::Bt2020Pq {
        if matches!(source, ColorSpec::Srgb) {
            v = [
                srgb_to_linear(v[0]),
                srgb_to_linear(v[1]),
                srgb_to_linear(v[2]),
            ];
        }
        if !matches!(source, ColorSpec::Bt2020Linear) {
            v = bt709_to_bt2020(v);
        }
        let nits = source.reference_white_nits().unwrap();
        v = [
            linear_to_pq(v[0] * (nits / 10000.0)),
            linear_to_pq(v[1] * (nits / 10000.0)),
            linear_to_pq(v[2] * (nits / 10000.0)),
        ];
    }
    let y = if target == ColorSpec::Bt2020Pq {
        0.2627f32 * v[0] + 0.6780f32 * v[1] + 0.0593f32 * v[2]
    } else {
        0.2126f32 * v[0] + 0.7152f32 * v[1] + 0.0722f32 * v[2]
    }
    .clamp(0.0, 1.0);
    match (ten_bit, full) {
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

/// Read the converter's luma plane back as code values.
fn read_luma(
    context: &VideoContext,
    converter: &ColorConverter,
    output_format: OutputFormat,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
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
    Ok(if sample_bytes == 2 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| (u16::from_le_bytes(*w) >> 6) as u32)
            .collect()
    } else {
        bytes.iter().map(|&b| b as u32).collect()
    })
}

fn check(
    context: &VideoContext,
    src: &SrcImage,
    encoder: &Encoder,
    source: ColorSpec,
    target: ColorSpec,
    output_format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    let ten_bit = output_format.bytes_per_sample() == 2;
    let config = ColorConverterConfig::new(
        WIDTH,
        HEIGHT,
        InputFormat::BGRA,
        output_format,
        source,
        target,
        ColorRange::Full,
    );
    let mut converter = ColorConverter::new(context.clone(), config)?;
    let description = converter.color_description();
    converter.convert(
        src.image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        encoder.input_image(),
    )?;
    let codes = read_luma(context, &converter, output_format)?;

    let mut worst = 0f64;
    let mut sum = 0f64;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let got = codes[(y * WIDTH + x) as usize] as f64;
            let want = expected_code(colour_at(x, y), source, target, ten_bit, true).round() as f64;
            let d = (got - want).abs();
            worst = worst.max(d);
            sum += d;
        }
    }
    let mean = sum / (WIDTH * HEIGHT) as f64;
    // One code value of slack: the GPU is free to order and fuse these f32
    // operations differently, which moves samples that land on a tie.
    let ok = worst <= 1.0;
    println!(
        "{:<34} -> {:<10} {:<6} worst {worst:>5.1}, mean {mean:.4}  VUI {}/{}/{} {}  {}",
        format!("{source:?}"),
        format!("{target:?}"),
        if ten_bit { "P010" } else { "NV12" },
        description.color_primaries,
        description.transfer_characteristics,
        description.matrix_coefficients,
        if description.full_range { "pc" } else { "tv" },
        if ok { "ok" } else { "MISMATCH" }
    );
    Ok(ok)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .app_name("path probe")
        .require_encode(Codec::H264)
        .build()?;
    let src = unsafe { create_src_image(&context)? };
    let encoder_8 = Encoder::new(
        context.clone(),
        EncodeConfig::h264(WIDTH, HEIGHT)
            .with_rate_control(RateControlMode::Cqp)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )?;
    let encoder_10 = Encoder::new(
        context.clone(),
        EncodeConfig::h265(WIDTH, HEIGHT)
            .with_bit_depth(EncodeBitDepth::Ten)
            .with_rate_control(RateControlMode::Cqp)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )?;

    let supported = [
        (ColorSpec::Srgb, ColorSpec::Srgb),
        (ColorSpec::Srgb, ColorSpec::Bt2020Pq),
        (ColorSpec::Bt709Linear, ColorSpec::Bt2020Pq),
        (ColorSpec::Bt2020Linear, ColorSpec::Bt2020Pq),
        (ColorSpec::Bt2020Pq, ColorSpec::Bt2020Pq),
    ];
    let mut all_ok = true;
    for (source, target) in supported {
        all_ok &= check(
            &context,
            &src,
            &encoder_8,
            source,
            target,
            OutputFormat::NV12,
        )?;
        all_ok &= check(
            &context,
            &src,
            &encoder_10,
            source,
            target,
            OutputFormat::P010,
        )?;
    }

    println!("\n--- these must be refused ---");
    let refused = [
        // An SDR target from something not already SDR-encoded: would need a
        // forward gamma encode, or tone mapping from PQ.
        (ColorSpec::Bt709Linear, ColorSpec::Srgb),
        (ColorSpec::Bt2020Linear, ColorSpec::Srgb),
        (ColorSpec::Bt2020Pq, ColorSpec::Srgb),
        // A target no decoder can be told about: linear light has no VUI code
        // points, so these specs are source-only.
        (ColorSpec::Srgb, ColorSpec::Bt709Linear),
        (ColorSpec::Srgb, ColorSpec::Bt2020Linear),
    ];
    for (source, target) in refused {
        // The description has to be absent for exactly the unencodable ones.
        let config = ColorConverterConfig::new(
            WIDTH,
            HEIGHT,
            InputFormat::BGRA,
            OutputFormat::NV12,
            source,
            target,
            ColorRange::Full,
        );
        let described = config.color_description().is_some();
        if described != target.is_encodable() {
            println!("{target:?}: color_description() and is_encodable() disagree");
            all_ok = false;
        }
        match ColorConverter::new(context.clone(), config) {
            Ok(_) => {
                println!("{source:?} -> {target:?}: ACCEPTED, should not be");
                all_ok = false;
            }
            Err(e) => println!("{source:?} -> {target:?}: refused ({e})"),
        }
    }

    unsafe {
        let device = context.device();
        device.destroy_image(src.image, None);
        device.free_memory(src.memory, None);
    }
    if all_ok {
        Ok(())
    } else {
        Err("at least one path disagrees with the model".into())
    }
}
