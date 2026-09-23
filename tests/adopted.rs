//! Pixelforge on a device the caller created, from an instance that asked for
//! Vulkan 1.1.
//!
//! An application that shares its device with pixelforge chose that device's
//! API version for its own reasons, and plenty choose less than 1.3. On such a
//! device the core 1.2 and 1.3 entry points are null, so anything pixelforge
//! reaches that way crashes. These tests build exactly that device, hand
//! pixelforge a spare queue where the hardware has one, and check the output
//! against a context pixelforge created itself on the same GPU.

use ash::vk;
use ash::vk::TaggedStructure;
#[allow(dead_code)]
mod common;
use common::{Readback, decode_stream};

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::encoder::Codec;
use pixelforge::vulkan::{DeviceQueue, DeviceRequirements, VideoContext, VideoContextBuilder};

/// An instance and device the test owns, destroyed on drop after everything
/// built on them.
struct AppDevice {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
}

impl Drop for AppDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn validation_requested() -> bool {
    std::env::var("PIXELFORGE_VALIDATION").is_ok()
}

/// Create a Vulkan 1.1 instance, with validation when asked for.
fn instance_1_1() -> Result<(ash::Entry, ash::Instance), Box<dyn std::error::Error>> {
    let entry = unsafe { ash::Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"pixelforge-adopted-test")
        .api_version(vk::API_VERSION_1_1);
    let validation = c"VK_LAYER_KHRONOS_validation";
    let has_validation = unsafe { entry.enumerate_instance_layer_properties()? }
        .iter()
        .any(|l| unsafe { std::ffi::CStr::from_ptr(l.layer_name.as_ptr()) } == validation);
    let layers = if validation_requested() && has_validation {
        vec![validation.as_ptr()]
    } else {
        Vec::new()
    };
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers);
    let instance = unsafe { entry.create_instance(&info, None)? };
    Ok((entry, instance))
}

/// Create the application's device from `reqs`, keeping queue 0 of every
/// family for itself wherever the family has a second queue. Returns the
/// queue pixelforge should use for each of `roles`.
fn create_app_device(
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    reqs: &DeviceRequirements,
) -> Result<AppDevice, Box<dyn std::error::Error>> {
    let family_props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let priorities = [1.0f32; 2];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo> = reqs
        .queue_families
        .iter()
        .map(|&f| {
            let count = family_props[f as usize].queue_count.min(2) as usize;
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(f)
                .queue_priorities(&priorities[..count])
        })
        .collect();
    let ext_ptrs: Vec<_> = reqs.extensions.iter().map(|e| e.as_ptr()).collect();

    // The KHR feature structs, since this device is not 1.3: the promoted
    // struct types are the same types under their extension names.
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let mut timeline =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    let mut unified = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default()
        .unified_image_layouts(true)
        .unified_image_layouts_video(true);
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut sync2)
        .push(&mut timeline);
    if reqs.unified_image_layouts {
        info = info.push(&mut unified);
    }
    let device = unsafe { instance.create_device(physical_device, &info, None)? };
    Ok(AppDevice {
        entry,
        instance,
        physical_device,
        device,
    })
}

/// The last queue this test created in `family`: a spare where there are two.
fn spare(app: &AppDevice, family: u32) -> DeviceQueue {
    let props = unsafe {
        app.instance
            .get_physical_device_queue_family_properties(app.physical_device)
    };
    DeviceQueue::new(family, props[family as usize].queue_count.min(2) - 1)
}

/// Decode `stream` on `context` and read every frame back.
fn decode_all(
    context: &VideoContext,
    stream: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut readback = Readback::new(context)?;
    let mut decoder = Decoder::new(context.clone(), DecodeConfig::h264().with_byte_stream())?;
    let mut frames = Vec::new();
    decode_stream(&mut decoder, stream, |frame| {
        let data = readback.read(&frame)?;
        frames.push([data.y, data.uv].concat());
        Ok(())
    })?;
    drop(decoder);
    drop(readback);
    Ok(frames)
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn decode_on_a_vulkan_1_1_device() -> Result<(), Box<dyn std::error::Error>> {
    let stream = std::fs::read("tests/data/bframes.264")?;

    let own = VideoContextBuilder::new()
        .enable_validation(validation_requested())
        .require_decode(Codec::H264)
        .build()?;
    let expected = decode_all(&own, &stream)?;
    let own_name = own.device_properties().device_name;
    drop(own);

    let (entry, instance) = instance_1_1()?;
    let builder = VideoContextBuilder::new().require_decode(Codec::H264);
    // The same GPU the own context picked, so the frames are comparable.
    let physical_device = unsafe { instance.enumerate_physical_devices()? }
        .into_iter()
        .find(|&pd| unsafe { instance.get_physical_device_properties(pd) }.device_name == own_name)
        .expect("the GPU the own context used");
    let reqs = builder.decode_device_requirements(&entry, &instance, physical_device)?;
    let app = create_app_device(entry, instance, physical_device, &reqs)?;

    let mut builder = builder
        .with_decode_queue(spare(&app, reqs.queues.decode.unwrap()))
        .with_transfer_queue(spare(&app, reqs.queues.transfer))
        .with_compute_queue(spare(&app, reqs.queues.compute));
    if reqs.unified_image_layouts {
        builder = builder.declare_unified_image_layouts();
    }
    let context = builder.build_from_existing_decode(
        app.entry.clone(),
        app.instance.clone(),
        app.physical_device,
        app.device.clone(),
    )?;
    let actual = decode_all(&context, &stream)?;
    drop(context);

    assert_eq!(actual.len(), expected.len(), "frame count");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(a == e, "frame {i} differs from the own-context decode");
    }
    Ok(())
}
