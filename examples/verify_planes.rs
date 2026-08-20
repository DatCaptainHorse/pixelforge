//! Verify `Decoder::copy_frame_to_planes`: copy each decoded frame into caller
//! Y (R8) and UV (R8G8) images, read them back, reconstruct NV12, and compare
//! to ffmpeg. Uses validation layers to catch any Vulkan misuse in the copy.
//!
//! Usage: verify_planes <input.264> <out.yuv>

use ash::vk;
use ash::vk::TaggedStructure;
use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::vulkan::VideoContextBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = std::env::args()
        .nth(1)
        .expect("usage: verify_planes <in.264> <out.yuv>");
    let out_path = std::env::args()
        .nth(2)
        .expect("usage: verify_planes <in.264> <out.yuv>");

    let entry = unsafe { ash::Entry::load()? };
    let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers),
            None,
        )?
    };

    let builder = VideoContextBuilder::new().require_decode(pixelforge::Codec::H264);
    let (pdevice, reqs) = unsafe { instance.enumerate_physical_devices()? }
        .into_iter()
        .find_map(|pd| {
            builder
                .decode_device_requirements(&entry, &instance, pd)
                .ok()
                .map(|r| (pd, r))
        })
        .expect("no decode-capable device");

    let priorities = [1.0f32];
    let qinfos: Vec<_> = reqs
        .queue_families
        .iter()
        .map(|&f| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(f)
                .queue_priorities(&priorities)
        })
        .collect();
    let exts: Vec<_> = reqs.extensions.iter().map(|e| e.as_ptr()).collect();
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    // The decoder orders its pipelined submissions with timeline semaphores.
    let mut timeline =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    let device = unsafe {
        instance.create_device(
            pdevice,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&qinfos)
                .enabled_extension_names(&exts)
                .push(&mut sync2)
                .push(&mut timeline),
            None,
        )?
    };

    // A queue + pool for the readback copies (use the transfer family).
    let transfer_family = reqs.queue_families[1];
    let queue = unsafe { device.get_device_queue(transfer_family, 0) };
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(transfer_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };

    let context = builder.build_from_existing_decode(
        entry.clone(),
        instance.clone(),
        pdevice,
        device.clone(),
    )?;
    let mut decoder = Decoder::new(context, DecodeConfig::h264())?;

    let stream = std::fs::read(&input)?;
    let mut out = std::fs::File::create(&out_path)?;
    let mut y_res: Option<PlaneImg> = None;
    let mut uv_res: Option<PlaneImg> = None;
    let mut count = 0;

    let mut handle = |decoder: &mut Decoder,
                      f: &pixelforge::decoder::DecodedFrame,
                      out: &mut std::fs::File|
     -> Result<(), Box<dyn std::error::Error>> {
        let (cw, ch) = (f.coded_width, f.coded_height);
        let y = y_res.get_or_insert_with(|| {
            PlaneImg::new(&device, &mem_props, cw, ch, vk::Format::R8_UNORM)
        });
        let uv = uv_res.get_or_insert_with(|| {
            PlaneImg::new(&device, &mem_props, cw / 2, ch / 2, vk::Format::R8G8_UNORM)
        });
        decoder.copy_frame_to_planes(f, y.image, uv.image)?;

        // Read back the visible region of each plane and emit NV12.
        let y_bytes = y.read_back(&device, &mem_props, pool, queue, f.width, f.height, 1);
        let uv_bytes = uv.read_back(
            &device,
            &mem_props,
            pool,
            queue,
            f.width / 2,
            f.height / 2,
            2,
        );
        use std::io::Write;
        out.write_all(&y_bytes)?;
        out.write_all(&uv_bytes)?;
        Ok(())
    };

    for (i, au) in decoder.split(&stream).enumerate() {
        for f in pollster::block_on(decoder.decode(au, i as u64)?)? {
            handle(&mut decoder, &f, &mut out)?;
            count += 1;
        }
    }
    for f in pollster::block_on(decoder.flush()?)? {
        handle(&mut decoder, &f, &mut out)?;
        count += 1;
    }
    println!("Copied {count} frames through Y/UV planes");

    drop(decoder);
    unsafe {
        if let Some(p) = y_res {
            p.destroy(&device);
        }
        if let Some(p) = uv_res {
            p.destroy(&device);
        }
        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}

struct PlaneImg {
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
}

impl PlaneImg {
    fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Self {
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
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
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .unwrap();
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                reqs.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .unwrap();
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .unwrap();
        unsafe { device.bind_image_memory(image, memory, 0).unwrap() };
        Self {
            image,
            memory,
            width,
            height,
        }
    }

    /// Copy the top-left `w`x`h` region (texel size `bpp`) to the host.
    fn read_back(
        &self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        pool: vk::CommandPool,
        queue: vk::Queue,
        w: u32,
        h: u32,
        bpp: u32,
    ) -> Vec<u8> {
        let size = (self.width * self.height * bpp) as u64;
        // Host-visible staging buffer.
        let buf = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }
        .unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                reqs.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .expect("host-visible memory");
        let mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .unwrap();
        unsafe { device.bind_buffer_memory(buf, mem, 0).unwrap() };

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
        }
        .unwrap()[0];
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            let to_src = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_src)),
            );
            let region = vk::BufferImageCopy2::default()
                .buffer_row_length(self.width)
                .buffer_image_height(self.height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer2(
                cmd,
                &vk::CopyImageToBufferInfo2::default()
                    .src_image(self.image)
                    .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .dst_buffer(buf)
                    .regions(std::slice::from_ref(&region)),
            );
            device.end_command_buffer(cmd).unwrap();
            let cbs = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            device
                .queue_submit(queue, &[submit], vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(queue).unwrap();

            let ptr = device
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())
                .unwrap() as *const u8;
            // Rows are buffer_row_length (coded) wide; take the visible w*bpp per row.
            let stride = (self.width * bpp) as usize;
            let mut out = Vec::with_capacity((w * h * bpp) as usize);
            for row in 0..h as usize {
                let s = ptr.add(row * stride);
                out.extend_from_slice(std::slice::from_raw_parts(s, (w * bpp) as usize));
            }
            device.unmap_memory(mem);
            device.free_command_buffers(pool, &[cmd]);
            device.destroy_buffer(buf, None);
            device.free_memory(mem, None);
            out
        }
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
