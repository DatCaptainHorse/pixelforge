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
use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::encoder::Codec;
use pixelforge::vulkan::VideoContextBuilder;

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
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut sync2)
        .push(&mut timeline);
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

    // --- Hand the app's device to pixelforge ---
    let context = builder.build_from_existing_decode(
        entry.clone(),
        instance.clone(),
        physical_device,
        device.clone(),
    )?;

    let mut decoder = Decoder::new(context, DecodeConfig::h264())?;
    let stream = std::fs::read(&input_path)?;
    let mut output = output_path.as_ref().map(File::create).transpose()?;
    let mut count = 0usize;

    let write = |decoder: &mut Decoder,
                 frame: &pixelforge::decoder::DecodedFrame,
                 output: &mut Option<File>,
                 count: &mut usize|
     -> Result<(), Box<dyn std::error::Error>> {
        if let Some(file) = output.as_mut() {
            let data = decoder.download(frame)?;
            file.write_all(&data.y)?;
            file.write_all(&data.uv)?;
        }
        *count += 1;
        Ok(())
    };

    for (i, au) in decoder.split(&stream).enumerate() {
        for frame in pollster::block_on(decoder.decode(au, i as u64)?)? {
            write(&mut decoder, &frame, &mut output, &mut count)?;
        }
    }
    for frame in pollster::block_on(decoder.flush()?)? {
        write(&mut decoder, &frame, &mut output, &mut count)?;
    }
    println!("Decoded {} frames on the adopted device", count);

    // The context borrowed the device; drop it before we destroy the device.
    drop(decoder);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}
