//! Decode on a caller-created device.
//!
//! Mirrors what an application with its own Vulkan device does: create the
//! instance and device itself, then hand them to pixelforge via
//! `build_from_existing_decode` so decoded images live on the app's device.
//!
//! Usage:
//!   decode_adopted <input.264> [output.yuv]
//!
//! Output is NV12 in display order, comparable to:
//!   ffmpeg -i input.264 -pix_fmt nv12 reference.yuv

use std::fs::File;
use std::io::Write;

use ash::vk;
use ash::vk::TaggedStructure;
mod common;
use common::Readback;

use pixelforge::decoder::{DecodeConfig, Decoder, FramePoll};
use pixelforge::encoder::Codec;
use pixelforge::vulkan::VideoContextBuilder;

/// Bytes handed to the decoder per call, standing in for a network read.
const CHUNK_SIZE: usize = 64 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .expect("usage: decode_adopted <input.264> [output.yuv]");
    let output_path = args.next();

    // --- The application creates its own instance and device ---
    let entry = unsafe { ash::Entry::load()? };
    let app_name = c"decode-adopted";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    // Ask pixelforge what a decode device needs, and pick a physical device that
    // satisfies it — exactly the negotiation an app would do.
    let builder = VideoContextBuilder::new().require_decode(Codec::H264);
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let (physical_device, reqs) = physical_devices
        .iter()
        .find_map(|&pd| {
            builder
                .decode_device_requirements(&entry, &instance, pd)
                .ok()
                .map(|r| (pd, r))
        })
        .expect("no physical device can decode H.264");

    let name = unsafe {
        std::ffi::CStr::from_ptr(
            instance
                .get_physical_device_properties(physical_device)
                .device_name
                .as_ptr(),
        )
    };
    println!("Device: {}", name.to_string_lossy());
    println!(
        "pixelforge needs queue families {:?} and {} extensions",
        reqs.queue_families,
        reqs.extensions.len()
    );

    // The app merges pixelforge's requirements with its own. Here there is
    // nothing else, so we use them directly.
    let priorities = [1.0f32];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo> = reqs
        .queue_families
        .iter()
        .map(|&f| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(f)
                .queue_priorities(&priorities)
        })
        .collect();
    let ext_ptrs: Vec<*const std::os::raw::c_char> =
        reqs.extensions.iter().map(|e| e.as_ptr()).collect();

    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    // The decoder orders its pipelined submissions with timeline semaphores.
    let mut timeline =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    // Unified image layouts are what let decoded frames be used in place, with
    // no copy and no layout transition. `reqs` says whether this device can,
    // and puts the extension in `reqs.extensions` when it can; both feature
    // bits have to be enabled here, and pixelforge told with
    // `declare_unified_image_layouts` below, because Vulkan cannot be asked
    // afterwards which features a device was created with.
    let mut unified = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default()
        .unified_image_layouts(true)
        .unified_image_layouts_video(true);
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut sync2)
        .push(&mut timeline);
    if reqs.unified_image_layouts {
        device_info = device_info.push(&mut unified);
    }
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

    // --- Hand the app's device to pixelforge ---
    let builder = if reqs.unified_image_layouts {
        builder.declare_unified_image_layouts()
    } else {
        println!("device has no unified image layouts; frames will be copied out");
        builder
    };
    let context = builder.build_from_existing_decode(
        entry.clone(),
        instance.clone(),
        physical_device,
        device.clone(),
    )?;

    let stream = std::fs::read(&input_path)?;
    let mut output = output_path.as_ref().map(File::create).transpose()?;
    let mut readback = output
        .is_some()
        .then(|| Readback::new(&context))
        .transpose()?;
    let mut decoder = Decoder::new(context, DecodeConfig::h264().with_byte_stream())?;
    let mut count = 0usize;

    let write = |frame: &pixelforge::decoder::DecodedFrame,
                 readback: &mut Option<Readback>,
                 output: &mut Option<File>,
                 count: &mut usize|
     -> Result<(), Box<dyn std::error::Error>> {
        if let (Some(file), Some(readback)) = (output.as_mut(), readback.as_mut()) {
            let data = readback.read(frame)?;
            file.write_all(&data.y)?;
            file.write_all(&data.uv)?;
        }
        *count += 1;
        Ok(())
    };

    // Feed a chunk, then take whatever has become ready; `Pending` means the
    // GPU is still working, so those frames are collected on a later pass.
    for (i, chunk) in stream.chunks(CHUNK_SIZE).enumerate() {
        let _ = decoder.decode(chunk, i as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            write(&frame, &mut readback, &mut output, &mut count)?;
        }
    }
    // End of stream: decodes the trailing picture, emits what reordering held
    // back, and closes the source.
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        write(&frame, &mut readback, &mut output, &mut count)?;
    }
    println!("Decoded {} frames on the adopted device", count);

    // The context borrowed the device, so everything holding Vulkan objects has
    // to go before the device does. That includes the readback helper, which
    // owns a command pool, a fence and a staging buffer of its own.
    drop(decoder);
    drop(readback);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}
