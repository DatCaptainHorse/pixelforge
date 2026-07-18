//! Codec- and direction-agnostic Vulkan Video helpers.
//!
//! Everything here is shared between the encoder and the decoder: format and
//! capability queries, video-profile-tagged image and buffer creation, video
//! session memory binding, and small arithmetic helpers. Direction-specific
//! resource management (encode slot pipelines, decode DPB pools) lives in
//! `encoder::resources` and `decoder` respectively, built on these primitives.

use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use std::ptr;

/// Compute greatest common divisor of two values.
pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

/// Compute least common multiple of two values.
pub(crate) fn lcm(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)).saturating_mul(b)
    }
}

/// Align a value up to the next multiple of the given alignment.
pub(crate) fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

pub(crate) fn query_supported_video_formats(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    image_usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::Format>> {
    let video_queue_fn = ash::khr::video_queue::Instance::load(context.entry(), context.instance());

    // Vulkan expects a profile list in the pNext chain.
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(image_usage)
        .push(&mut profile_list);

    let physical_device = context.physical_device();
    let mut count = 0u32;
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            ptr::null_mut(),
        )
    };

    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to query video format properties for usage {:?}: {:?}",
            image_usage, result
        )));
    }

    if count == 0 {
        return Ok(Vec::new());
    }

    let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };

    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to enumerate video format properties for usage {:?}: {:?}",
            image_usage, result
        )));
    }

    props.truncate(count as usize);
    Ok(props.into_iter().map(|p| p.format).collect())
}

/// Get the Vulkan format for a given pixel format and bit depth.
///
/// Supports YUV420 and YUV444 in 8-bit and 10-bit.
/// For YUV444, uses 2-plane (semi-planar) formats from VK_EXT_ycbcr_2plane_444_formats
/// which are supported by NVIDIA hardware for video encoding.
pub(crate) fn get_video_format(pixel_format: PixelFormat, bit_depth: BitDepth) -> vk::Format {
    match (pixel_format, bit_depth) {
        (PixelFormat::Yuv420, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        (PixelFormat::Yuv420, BitDepth::Ten) => {
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        }
        // Use 2-plane semi-planar formats for YUV444 (supported by NVIDIA for video encoding).
        (PixelFormat::Yuv444, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_444_UNORM,
        (PixelFormat::Yuv444, BitDepth::Ten) => {
            vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
        }
        // TODO: Add support for YUV422 formats.
        _ => unimplemented!(
            "Unsupported pixel format / bit depth combination: {:?} / {:?}",
            pixel_format,
            bit_depth
        ),
    }
}

/// Create a buffer that requires device addresses (SHADER_DEVICE_ADDRESS usage).
///
/// This allocates memory with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` so that
/// `get_buffer_device_address` returns a valid address.
pub(crate) fn create_buffer_with_device_address(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("buffer creation: {}", e)))?;

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index = find_memory_type(
        memory_properties,
        mem_requirements.memory_type_bits,
        properties,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation(format!(
            "No suitable memory type for buffer with properties {:?}",
            properties
        ))
    })?;

    let mut alloc_flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index)
        .push(&mut alloc_flags_info);

    let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(m) => m,
        Err(e) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(PixelForgeError::MemoryAllocation(e.to_string()));
        }
    };

    match unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        Ok(()) => Ok((buffer, memory)),
        Err(e) => {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            Err(PixelForgeError::MemoryAllocation(e.to_string()))
        }
    }
}

pub(crate) fn find_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory_props.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && memory_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
    })
}

pub(crate) fn create_bitstream_buffer(
    context: &VideoContext,
    size: usize,
    usage: vk::BufferUsageFlags,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let create_info = vk::BufferCreateInfo::default()
        .size(size as vk::DeviceSize)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push(&mut profile_list);

    let buffer = unsafe { context.device().create_buffer(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("buffer creation: {}", e)))?;

    let mem_requirements = unsafe { context.device().get_buffer_memory_requirements(buffer) };

    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for buffer".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { context.device().bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    Ok((buffer, memory))
}

pub(crate) fn create_timeline_semaphore(context: &VideoContext) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);

    unsafe { context.device().create_semaphore(&create_info, None) }
        .map_err(|e| PixelForgeError::Synchronization(e.to_string()))
}

/// Create an image for video encoding (input or DPB).
///
/// This creates a VkImage suitable for use with a video encoder.
/// For DPB images, the usage is VIDEO_ENCODE_DPB_KHR.
/// For input images, the usage is VIDEO_ENCODE_SRC_KHR | TRANSFER_DST.
///
/// # Arguments
/// * `context` - The Vulkan video context
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `format` - The Vulkan format to use for the image
/// * `is_dpb` - If true, create a DPB image; if false, create an input image
/// * `profile_info` - Video profile info for the encoder session
pub(crate) fn create_video_image(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    sharing_families: &[u32],
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    // Use CONCURRENT sharing mode when multiple queue families need access
    // (e.g. video queue + transfer queue for upload/readback, compute for
    // color conversion). Callers pass the deduplicated family list.
    let mut queue_families: Vec<u32> = Vec::new();
    for &family in sharing_families {
        if !queue_families.contains(&family) {
            queue_families.push(family);
        }
    }
    let sharing_mode = if queue_families.len() > 1 {
        vk::SharingMode::CONCURRENT
    } else {
        queue_families.clear();
        vk::SharingMode::EXCLUSIVE
    };

    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(sharing_mode)
        .queue_family_indices(&queue_families)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push(&mut profile_list);

    let image = unsafe { context.device().create_image(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("image creation: {}", e)))?;

    let mem_requirements = unsafe { context.device().get_image_memory_requirements(image) };

    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for image".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { context.device().bind_image_memory(image, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    let view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = unsafe { context.device().create_image_view(&view_create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("image view creation: {}", e)))?;

    Ok((image, memory, view))
}

/// Allocate and bind memory for a video session.
///
/// Returns the allocated device memory handles.
pub(crate) fn allocate_session_memory(
    context: &VideoContext,
    session: vk::VideoSessionKHR,
    video_queue_fn: &ash::khr::video_queue::Device,
) -> Result<Vec<vk::DeviceMemory>> {
    // Query memory requirements count.
    let mut memory_requirements_count = 0u32;
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_video_session_memory_requirements_khr)(
            context.device().handle(),
            session,
            &mut memory_requirements_count,
            ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    // Query actual requirements.
    let mut memory_requirements =
        vec![vk::VideoSessionMemoryRequirementsKHR::default(); memory_requirements_count as usize];
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_video_session_memory_requirements_khr)(
            context.device().handle(),
            session,
            &mut memory_requirements_count,
            memory_requirements.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    // Allocate and bind memory for each requirement.
    let mut session_memory = Vec::new();
    let mut bind_infos = Vec::new();

    for req in &memory_requirements {
        let memory_type_index = find_memory_type(
            context.memory_properties(),
            req.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type(
                context.memory_properties(),
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or_else(|| {
            PixelForgeError::MemoryAllocation(format!(
                "No suitable memory type for video session (type_bits: 0x{:x})",
                req.memory_requirements.memory_type_bits
            ))
        })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        bind_infos.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(req.memory_bind_index)
                .memory(memory)
                .memory_offset(0)
                .memory_size(req.memory_requirements.size),
        );

        session_memory.push(memory);
    }

    // Bind all memory to the session.
    let result = unsafe {
        (video_queue_fn.fp().bind_video_session_memory_khr)(
            context.device().handle(),
            session,
            bind_infos.len() as u32,
            bind_infos.as_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    Ok(session_memory)
}

/// Map a bitstream buffer for persistent access.
pub(crate) fn map_bitstream_buffer(
    context: &VideoContext,
    memory: vk::DeviceMemory,
    size: usize,
) -> Result<*mut u8> {
    let ptr = unsafe {
        context.device().map_memory(
            memory,
            0,
            size as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(|e| {
        PixelForgeError::MemoryAllocation(format!("Failed to map bitstream buffer: {}", e))
    })? as *mut u8;

    Ok(ptr)
}

/// Create the decoded picture buffer images for a video session.
///
/// Two layouts are possible, selected by `use_layered`:
/// - **layered**: a single image with `count` array layers (required when the
///   device does not report `SEPARATE_REFERENCE_IMAGES`), with one view per layer;
/// - **separate**: `count` independent single-layer images, one view each.
///
/// Either way the returned view vector is indexed by DPB slot, so callers need
/// not care which layout was used. `usage` and `sharing_families` differ
/// between encode and decode (decode DPB images may also serve as the decode
/// output and be copied from), so they are supplied by the caller.
pub(crate) fn create_dpb_images(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    count: usize,
    usage: vk::ImageUsageFlags,
    sharing_families: &[u32],
    profile_info: &vk::VideoProfileInfoKHR,
    use_layered: bool,
) -> Result<(Vec<vk::Image>, Vec<vk::DeviceMemory>, Vec<vk::ImageView>)> {
    if !use_layered {
        let mut images = Vec::with_capacity(count);
        let mut memories = Vec::with_capacity(count);
        let mut views = Vec::with_capacity(count);
        for _ in 0..count {
            let (image, memory, view) = create_video_image(
                context,
                width,
                height,
                format,
                usage,
                sharing_families,
                profile_info,
            )?;
            images.push(image);
            memories.push(memory);
            views.push(view);
        }
        return Ok((images, memories, views));
    }

    // Layered: one image, `count` array layers, one view per layer.
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let mut queue_families: Vec<u32> = Vec::new();
    for &family in sharing_families {
        if !queue_families.contains(&family) {
            queue_families.push(family);
        }
    }
    let sharing_mode = if queue_families.len() > 1 {
        vk::SharingMode::CONCURRENT
    } else {
        queue_families.clear();
        vk::SharingMode::EXCLUSIVE
    };

    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(count as u32)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(sharing_mode)
        .queue_family_indices(&queue_families)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push(&mut profile_list);

    let image = unsafe { context.device().create_image(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("layered DPB image: {}", e)))?;

    let mem_requirements = unsafe { context.device().get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation(
            "No suitable memory type for layered DPB image".to_string(),
        )
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;
    unsafe { context.device().bind_image_memory(image, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    let mut views = Vec::with_capacity(count);
    for layer in 0..count as u32 {
        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            });
        let view = unsafe { context.device().create_image_view(&view_create_info, None) }.map_err(
            |e| {
                PixelForgeError::ResourceCreation(format!(
                    "layered DPB view layer {}: {}",
                    layer, e
                ))
            },
        )?;
        views.push(view);
    }

    Ok((vec![image], vec![memory], views))
}
