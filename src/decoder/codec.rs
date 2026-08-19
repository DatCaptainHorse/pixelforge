//! The codec-generic decoder.
//!
//! Everything that is identical across H.264, H.265 and AV1 lives here: the
//! shared per-decoder state ([`DecoderCommon`]), the Vulkan video session and
//! its DPB images ([`DecodeSession`]), coded-data staging, and the readback
//! path that copies a decoded picture back to the host.
//!
//! A codec supplies only its differences: parsing its bitstream into pictures,
//! reconstructing the reference state the encoder implied, and building the
//! codec-specific `StdVideo*` graph for a picture. Reading a codec's folder
//! shows those differences; the scaffolding is here.
//!
//! There is deliberately no `VideoDecodeCodec` trait yet. The per-picture hook
//! shape is only knowable once a second codec exists to constrain it; H.265 is
//! the forcing function. Until then a codec owns a `DecoderCommon` directly.
//!
//! This mirrors the encoder's [`crate::encoder::codec`] split deliberately, so
//! the two directions read the same way.

use ash::vk;

use crate::decoder::{DecodedFrame, DecodedFrameData};
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::{
    VideoImageParams, allocate_session_memory, create_bitstream_buffer, create_dpb_images,
    create_video_image, find_memory_type, map_bitstream_buffer,
};
use crate::vulkan::VideoContext;

/// Initial size of the coded-data staging buffer; grows as needed.
const INITIAL_BITSTREAM_BUFFER_SIZE: usize = 1024 * 1024;

/// Annex B start code prefixed to each slice handed to the driver.
pub(crate) const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Query decode capabilities for a profile.
///
/// `caps` is supplied by the caller already chained with the codec-specific
/// capability struct, since only the codec knows which one applies.
pub(crate) fn query_decode_caps(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    caps: &mut vk::VideoCapabilitiesKHR,
) -> Result<()> {
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let result = unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            caps,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::CodecNotSupported(format!(
            "decode profile not supported: {:?}",
            result
        )));
    }
    Ok(())
}

/// What a codec decided the session should look like.
///
/// The codec derives these from its own parameter sets and the device caps; the
/// generic layer turns them into Vulkan objects.
pub(crate) struct SessionPlan {
    /// Coded dimensions, already aligned to the device's picture granularity.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub pixel_format: PixelFormat,
    /// DPB slots: references plus the picture being decoded.
    pub slot_count: usize,
    /// Driver decodes straight into the DPB image (`DPB_AND_OUTPUT_COINCIDE`).
    pub coincide: bool,
    /// One layered DPB image rather than one image per slot.
    pub use_layered_dpb: bool,
    pub dpb_usage: vk::ImageUsageFlags,
}

/// The Vulkan video session and the images it decodes into.
///
/// Recreated whenever the stream's parameter sets change in a way that matters
/// (a new resolution, most often).
pub(crate) struct DecodeSession {
    pub session: vk::VideoSessionKHR,
    pub session_memory: Vec<vk::DeviceMemory>,
    pub session_params: vk::VideoSessionParametersKHR,

    /// Coded dimensions the session was created for.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub pixel_format: PixelFormat,

    /// DPB images. One image per slot, or a single layered image.
    pub dpb_images: Vec<vk::Image>,
    pub dpb_memories: Vec<vk::DeviceMemory>,
    /// One view per slot (indexes slots even when layered).
    pub dpb_views: Vec<vk::ImageView>,
    /// Whether each slot has been written at least once (governs the
    /// UNDEFINED-vs-DPB old layout in the pre-decode barrier).
    pub dpb_slot_active: Vec<bool>,
    pub use_layered_dpb: bool,

    /// True when the driver decodes into the DPB image directly; false when a
    /// distinct output image is required.
    pub coincide: bool,
    /// Distinct decode output image, only when `!coincide`.
    pub output_image: Option<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
}
impl DecodeSession {
    /// The image and array layer holding DPB slot `slot`.
    /// For layered DPBs all slots share `dpb_images[0]` and are distinguished
    /// by layer; for separate images each slot has its own image at layer 0.
    pub fn dpb_image_for_slot(&self, slot: u8) -> (vk::Image, u32) {
        if self.use_layered_dpb {
            (self.dpb_images[0], slot as u32)
        } else {
            (self.dpb_images[slot as usize], 0)
        }
    }
}

/// Per-decoder state shared by every codec.
///
/// Holds the Vulkan video session, the DPB images, the coded-data staging
/// buffer and the readback path — none of which differ between codecs.
/// Codec-specific state (parameter sets, POC, reference marking) lives in the
/// codec instead.
pub(crate) struct DecoderCommon {
    pub context: VideoContext,
    pub video_queue_fn: ash::khr::video_queue::Device,
    pub video_decode_fn: ash::khr::video_decode_queue::Device,
    pub decode_queue: vk::Queue,
    pub decode_queue_family: u32,

    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,

    /// Readback resources. The video decode queue family is a dedicated engine
    /// that does not advertise `TRANSFER_BIT` (true on RADV, among others), so
    /// copies must be recorded and submitted on the transfer queue instead.
    pub transfer_queue: vk::Queue,
    pub transfer_pool: vk::CommandPool,
    pub transfer_command_buffer: vk::CommandBuffer,
    pub transfer_fence: vk::Fence,

    /// Staging buffer for coded data handed to the decode queue.
    pub bitstream_buffer: vk::Buffer,
    pub bitstream_memory: vk::DeviceMemory,
    pub bitstream_ptr: *mut u8,
    pub bitstream_size: usize,
    /// Required alignment of bitstream buffer offsets/ranges.
    pub bitstream_offset_alignment: u64,
    pub bitstream_size_alignment: u64,

    /// Readback buffer for `download`, allocated on first use.
    pub readback: Option<(vk::Buffer, vk::DeviceMemory, usize)>,

    /// The active session, created lazily from the stream's parameter sets.
    pub session: Option<DecodeSession>,
}

// `bitstream_ptr` is a persistently mapped host-visible allocation owned by
// this struct; it is never aliased or shared across threads.
unsafe impl Send for DecoderCommon {}

impl DecoderCommon {
    pub fn new(context: VideoContext) -> Result<Self> {
        let decode_queue_family = context.video_decode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("Device has no video decode queue family".to_string())
        })?;
        let decode_queue = context.video_decode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("Device has no video decode queue".to_string())
        })?;

        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_decode_fn =
            ash::khr::video_decode_queue::Device::load(context.instance(), context.device());

        let (command_pool, command_buffer, fence) =
            create_command_resources(&context, decode_queue_family, "decode")?;
        let (transfer_pool, transfer_command_buffer, transfer_fence) =
            create_command_resources(&context, context.transfer_queue_family(), "decode transfer")?;

        Ok(Self {
            transfer_queue: context.transfer_queue(),
            context,
            video_queue_fn,
            video_decode_fn,
            decode_queue,
            decode_queue_family,
            command_pool,
            command_buffer,
            fence,
            transfer_pool,
            transfer_command_buffer,
            transfer_fence,
            bitstream_buffer: vk::Buffer::null(),
            bitstream_memory: vk::DeviceMemory::null(),
            bitstream_ptr: std::ptr::null_mut(),
            bitstream_size: 0,
            bitstream_offset_alignment: 1,
            bitstream_size_alignment: 1,
            readback: None,
            session: None,
        })
    }

    /// The active session, or an error if no parameter sets have been seen yet.
    pub fn session(&self) -> Result<&DecodeSession> {
        self.session
            .as_ref()
            .ok_or_else(|| PixelForgeError::InvalidInput("decode: no active session".to_string()))
    }

    /// Create the session, its DPB images and (when needed) a distinct output
    /// image, replacing any existing one.
    ///
    /// `make_params` builds the codec's session parameters once the session
    /// exists, since parameter sets are entirely codec-specific.
    pub fn create_session(
        &mut self,
        plan: &SessionPlan,
        profile_info: &vk::VideoProfileInfoKHR,
        std_header_version: &vk::ExtensionProperties,
        make_params: impl FnOnce(&Self, vk::VideoSessionKHR) -> Result<vk::VideoSessionParametersKHR>,
    ) -> Result<()> {
        self.destroy_session();

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.decode_queue_family)
            .flags(vk::VideoSessionCreateFlagsKHR::empty())
            .video_profile(profile_info)
            .picture_format(plan.picture_format)
            .max_coded_extent(vk::Extent2D {
                width: plan.coded_width,
                height: plan.coded_height,
            })
            .reference_picture_format(plan.picture_format)
            .max_dpb_slots(plan.slot_count as u32)
            .max_active_reference_pictures(plan.slot_count as u32 - 1)
            .std_header_version(std_header_version);

        let session = unsafe {
            self.video_queue_fn
                .create_video_session(&session_create_info, None)
        }
        .map_err(|e| PixelForgeError::VideoSessionCreation(format!("{:?}", e)))?;

        let session_memory = allocate_session_memory(&self.context, session, &self.video_queue_fn)?;

        // When the DPB doubles as the decode output it may also be copied from,
        // so the transfer queue needs access too.
        let families = [
            self.decode_queue_family,
            self.context.transfer_queue_family(),
        ];

        let dpb_params = VideoImageParams {
            width: plan.coded_width,
            height: plan.coded_height,
            format: plan.picture_format,
            usage: plan.dpb_usage,
            sharing_families: &families,
        };
        let (dpb_images, dpb_memories, dpb_views) = create_dpb_images(
            &self.context,
            &dpb_params,
            profile_info,
            plan.slot_count,
            plan.use_layered_dpb,
        )?;

        let output_image = if plan.coincide {
            None
        } else {
            let output_params = VideoImageParams {
                usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                ..dpb_params
            };
            Some(create_video_image(
                &self.context,
                &output_params,
                profile_info,
            )?)
        };

        let session_params = make_params(self, session)?;

        self.session = Some(DecodeSession {
            session,
            session_memory,
            session_params,
            coded_width: plan.coded_width,
            coded_height: plan.coded_height,
            picture_format: plan.picture_format,
            bit_depth: plan.bit_depth,
            pixel_format: plan.pixel_format,
            dpb_images,
            dpb_memories,
            dpb_views,
            dpb_slot_active: vec![false; plan.slot_count],
            use_layered_dpb: plan.use_layered_dpb,
            coincide: plan.coincide,
            output_image,
        });
        Ok(())
    }

    pub fn destroy_session(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        unsafe {
            self.context.device().device_wait_idle().ok();
            self.video_queue_fn
                .destroy_video_session_parameters(session.session_params, None);
            for &view in &session.dpb_views {
                self.context.device().destroy_image_view(view, None);
            }
            for &image in &session.dpb_images {
                self.context.device().destroy_image(image, None);
            }
            for &memory in &session.dpb_memories {
                self.context.device().free_memory(memory, None);
            }
            if let Some((image, memory, view)) = session.output_image {
                self.context.device().destroy_image_view(view, None);
                self.context.device().destroy_image(image, None);
                self.context.device().free_memory(memory, None);
            }
            self.video_queue_fn
                .destroy_video_session(session.session, None);
            for &memory in &session.session_memory {
                self.context.device().free_memory(memory, None);
            }
        }
    }

    /// Ensure the coded-data staging buffer holds at least `size` bytes.
    pub fn ensure_bitstream_capacity(
        &mut self,
        size: usize,
        profile_info: &vk::VideoProfileInfoKHR,
    ) -> Result<()> {
        if self.bitstream_size >= size && self.bitstream_buffer != vk::Buffer::null() {
            return Ok(());
        }
        let new_size = size.max(INITIAL_BITSTREAM_BUFFER_SIZE).next_power_of_two();

        if self.bitstream_buffer != vk::Buffer::null() {
            unsafe {
                self.context.device().device_wait_idle().ok();
                self.context.device().unmap_memory(self.bitstream_memory);
                self.context
                    .device()
                    .destroy_buffer(self.bitstream_buffer, None);
                self.context
                    .device()
                    .free_memory(self.bitstream_memory, None);
            }
        }

        let (buffer, memory) = create_bitstream_buffer(
            &self.context,
            new_size,
            vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR,
            profile_info,
        )?;
        self.bitstream_ptr = map_bitstream_buffer(&self.context, memory, new_size)?;
        self.bitstream_buffer = buffer;
        self.bitstream_memory = memory;
        self.bitstream_size = new_size;
        Ok(())
    }

    /// Copy a picture's slices into the staging buffer, each prefixed with a
    /// start code, and report the buffer range plus each slice's offset.
    pub fn stage_slices(
        &mut self,
        slices: &[&[u8]],
        profile_info: &vk::VideoProfileInfoKHR,
    ) -> Result<(u64, Vec<u32>)> {
        let total: usize = slices.iter().map(|s| s.len() + START_CODE.len()).sum();
        let aligned_total =
            crate::video::align_up(total as u32, self.bitstream_size_alignment as u32) as usize;
        self.ensure_bitstream_capacity(aligned_total, profile_info)?;

        let mut offsets = Vec::with_capacity(slices.len());
        let mut cursor = 0usize;
        for slice in slices {
            offsets.push(cursor as u32);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    START_CODE.as_ptr(),
                    self.bitstream_ptr.add(cursor),
                    START_CODE.len(),
                );
                cursor += START_CODE.len();
                std::ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    self.bitstream_ptr.add(cursor),
                    slice.len(),
                );
                cursor += slice.len();
            }
        }
        // Zero the alignment padding so the driver never reads stale bytes.
        if aligned_total > cursor {
            unsafe {
                std::ptr::write_bytes(self.bitstream_ptr.add(cursor), 0, aligned_total - cursor);
            }
        }
        Ok((aligned_total as u64, offsets))
    }

    /// Transition the DPB slot this decode writes into, and the output image
    /// (if distinct), into the layouts `vkCmdDecodeVideo` requires.
    pub fn record_barriers(&self, slot: u8) {
        let session = self.session.as_ref().expect("active session");
        let device = self.context.device();
        let mut barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        let subresource = |layer: u32| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        };

        // The slot being written: UNDEFINED on first use, DPB afterwards.
        let (dst_image, dst_layer) = session.dpb_image_for_slot(slot);
        let old_layout = if session.dpb_slot_active[slot as usize] {
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        barriers.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image)
                .subresource_range(subresource(dst_layer)),
        );

        // A distinct output image must be in decode-DST layout.
        if let Some((image, _, _)) = session.output_image {
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::VIDEO_DECODE_DST_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(subresource(0)),
            );
        }

        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(self.command_buffer, &dependency) };
    }

    /// Begin recording this picture's decode command buffer.
    pub fn begin_decode_commands(&self) -> Result<()> {
        let device = self.context.device();
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }
        Ok(())
    }

    /// Ensure the readback buffer holds at least `size` bytes.
    ///
    /// The stored capacity is the *allocated* size (which the driver may round
    /// up), so that `download` can map the whole allocation rather than a
    /// sub-range that might violate `nonCoherentAtomSize`.
    fn ensure_readback_capacity(&mut self, size: usize) -> Result<()> {
        if let Some((_, _, capacity)) = self.readback
            && capacity >= size
        {
            return Ok(());
        }
        if let Some((buffer, memory, _)) = self.readback.take() {
            unsafe {
                self.context.device().device_wait_idle().ok();
                self.context.device().destroy_buffer(buffer, None);
                self.context.device().free_memory(memory, None);
            }
        }

        let create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.context.device().create_buffer(&create_info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("readback buffer: {}", e)))?;
        let reqs = unsafe { self.context.device().get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            self.context.memory_properties(),
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| {
            PixelForgeError::MemoryAllocation("No host-visible memory for readback".to_string())
        })?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { self.context.device().allocate_memory(&alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;
        unsafe { self.context.device().bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        // Record the allocated size, not the requested one.
        self.readback = Some((buffer, memory, reqs.size as usize));
        Ok(())
    }

    /// Copy a decoded picture back to the host, cropped to its visible region.
    ///
    /// Entirely codec-independent: it copies planes out of an image the decode
    /// queue has already finished writing.
    ///
    /// The copy runs on the transfer queue. The video decode queue family is a
    /// dedicated engine that need not advertise `TRANSFER_BIT` (it does not on
    /// RADV), so recording a copy there is invalid.
    pub fn download(&mut self, frame: &DecodedFrame) -> Result<DecodedFrameData> {
        let (bit_depth, pixel_format) = {
            let session = self.session()?;
            (session.bit_depth, session.pixel_format)
        };

        let geom = PlaneGeometry::new(
            frame.coded_width,
            frame.coded_height,
            frame.width,
            frame.height,
            bit_depth,
            pixel_format,
        );
        let PlaneGeometry {
            chroma_div,
            y_stride,
            y_size,
            uv_stride,
            total,
            ..
        } = geom;

        self.ensure_readback_capacity(total)?;
        let (buffer, memory, allocated) = self.readback.expect("ensured above");

        let device = self.context.device().clone();
        unsafe {
            device
                .reset_command_buffer(
                    self.transfer_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    self.transfer_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let base_layer = frame.array_layer;

        // A multi-planar image must be transitioned with the plane aspects that
        // the copy will use; a COLOR barrier does not cover PLANE_0/PLANE_1 and
        // leaves the planes in an undefined layout.
        let copy_aspects = vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1;

        // Recorded on the transfer queue, so no VIDEO_DECODE_* stage or access
        // may appear here. The decode is already fence-waited, so the writes are
        // visible and NONE is a correct source scope.
        let to_src = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(frame.layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: copy_aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: base_layer,
                layer_count: 1,
            });
        let barriers = [to_src];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dependency) };

        let regions = [
            vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(frame.coded_width)
                .buffer_image_height(frame.coded_height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: base_layer,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: frame.coded_width,
                    height: frame.coded_height,
                    depth: 1,
                }),
            vk::BufferImageCopy2::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(frame.coded_width / chroma_div as u32)
                .buffer_image_height(frame.coded_height / chroma_div as u32)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: base_layer,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: frame.coded_width / chroma_div as u32,
                    height: frame.coded_height / chroma_div as u32,
                    depth: 1,
                }),
        ];
        let copy_info = vk::CopyImageToBufferInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_buffer(buffer)
            .regions(&regions);
        unsafe { device.cmd_copy_image_to_buffer2(self.transfer_command_buffer, &copy_info) };

        // Restore the layout so a DPB image stays usable as a reference. (For a
        // separate output image this is redundant but harmless: the next decode
        // re-transitions it from UNDEFINED.)
        let restore = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(frame.layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: copy_aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: base_layer,
                layer_count: 1,
            });
        let barriers = [restore];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dependency) };

        unsafe {
            device
                .end_command_buffer(self.transfer_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let command_buffers = [self.transfer_command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .reset_fences(&[self.transfer_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .queue_submit(self.transfer_queue, &[submit], self.transfer_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .wait_for_fences(&[self.transfer_fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }

        // Read out, cropping to the visible region.
        // Map the whole allocation: a sub-range map must respect
        // nonCoherentAtomSize, and WHOLE_SIZE sidesteps that entirely.
        let ptr =
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?
                as *const u8;
        debug_assert!(
            allocated >= geom.max_read_offset(),
            "readback buffer too small: host reads up to {} but only {} bytes are mapped",
            geom.max_read_offset(),
            allocated
        );

        let visible_y_stride = geom.visible_y_stride;
        let visible_uv_stride = geom.visible_uv_stride;
        let mut y = Vec::with_capacity(visible_y_stride * geom.visible_y_rows);
        let mut uv = Vec::with_capacity(visible_uv_stride * geom.visible_uv_rows);
        unsafe {
            for row in 0..geom.visible_y_rows {
                let src = ptr.add(row * y_stride);
                y.extend_from_slice(std::slice::from_raw_parts(src, visible_y_stride));
            }
            for row in 0..geom.visible_uv_rows {
                let src = ptr.add(y_size + row * uv_stride);
                uv.extend_from_slice(std::slice::from_raw_parts(src, visible_uv_stride));
            }
            device.unmap_memory(memory);
        }

        Ok(DecodedFrameData {
            y,
            uv,
            y_stride: visible_y_stride,
            uv_stride: visible_uv_stride,
            width: frame.width,
            height: frame.height,
            bit_depth,
            pixel_format,
        })
    }

    /// Submit the recorded decode and wait for it to complete.
    pub fn submit_decode(&self) -> Result<()> {
        let device = self.context.device();
        unsafe {
            device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let command_buffers = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .reset_fences(&[self.fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .queue_submit(self.decode_queue, &[submit], self.fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }
        Ok(())
    }

    /// Copy a decoded picture into a caller-owned image, so it survives past the
    /// next `decode` call. Used by the reorder buffer to retain frames while
    /// later pictures are decoded ahead of them in display order.
    ///
    /// Runs on the transfer queue, mirroring `download`: the decode is already
    /// fence-waited, so `NONE` is a correct source scope. The source's layout is
    /// restored afterward so a DPB image stays usable as a reference; the
    /// destination is left in `TRANSFER_DST_OPTIMAL` (reported as the pooled
    /// frame's layout, which `download` then transitions from).
    pub fn copy_frame_to_image(&self, frame: &DecodedFrame, dst_image: vk::Image) -> Result<()> {
        let base_layer = frame.array_layer;
        let device = self.context.device().clone();
        let aspects = vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1;
        let range = |image: vk::Image, layer: u32| vk::ImageMemoryBarrier2 {
            image,
            ..vk::ImageMemoryBarrier2::default()
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspects,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        };

        unsafe {
            device
                .reset_command_buffer(
                    self.transfer_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    self.transfer_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let to_transfer = [
            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::COPY,
                dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
                old_layout: frame.layout,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ..range(frame.image, base_layer)
            },
            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::COPY,
                dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ..range(dst_image, 0)
            },
        ];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dep) };

        let plane = |aspect: vk::ImageAspectFlags| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let (cdiv_hor, cdiv_vert) = frame.pixel_format.chroma_div();
        let regions = [
            vk::ImageCopy2::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    base_array_layer: base_layer,
                    ..plane(vk::ImageAspectFlags::PLANE_0)
                })
                .dst_subresource(plane(vk::ImageAspectFlags::PLANE_0))
                .extent(vk::Extent3D {
                    width: frame.coded_width,
                    height: frame.coded_height,
                    depth: 1,
                }),
            vk::ImageCopy2::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    base_array_layer: base_layer,
                    ..plane(vk::ImageAspectFlags::PLANE_1)
                })
                .dst_subresource(plane(vk::ImageAspectFlags::PLANE_1))
                .extent(vk::Extent3D {
                    width: frame.coded_width / cdiv_hor,
                    height: frame.coded_height / cdiv_vert,
                    depth: 1,
                }),
        ];
        let copy = vk::CopyImageInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(dst_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&regions);
        unsafe { device.cmd_copy_image2(self.transfer_command_buffer, &copy) };

        // Restore the source layout so a DPB image stays a valid reference.
        let restore = [vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COPY,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
            dst_access_mask: vk::AccessFlags2::NONE,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: frame.layout,
            ..range(frame.image, base_layer)
        }];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&restore);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dep) };

        unsafe {
            device
                .end_command_buffer(self.transfer_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let command_buffers = [self.transfer_command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .reset_fences(&[self.transfer_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .queue_submit(self.transfer_queue, &[submit], self.transfer_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .wait_for_fences(&[self.transfer_fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }
        Ok(())
    }

    /// Copy a decoded picture's two planes into two single-plane caller images:
    /// luma into `y_image` (an R8/R16 image) and interleaved chroma into
    /// `uv_image` (an R8G8/R16G16 image), each in the plane's own resolution.
    ///
    /// This suits consumers that sample Y and UV as separate textures (the
    /// common YUV→RGB shader layout) without a sampler-YCbCr conversion. Like
    /// [`Self::copy_frame_to_image`] it runs on the transfer queue, restores the
    /// source layout, and leaves both destinations in `TRANSFER_DST_OPTIMAL`.
    ///
    /// `y_image` and `uv_image` must live on this context's device and be sized
    /// to the frame's coded dimensions (chroma at half size for 4:2:0).
    pub fn copy_frame_to_planes(
        &self,
        frame: &DecodedFrame,
        y_image: vk::Image,
        uv_image: vk::Image,
    ) -> Result<()> {
        let base_layer = frame.array_layer;
        let device = self.context.device().clone();

        let color = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let src_planes = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: base_layer,
            layer_count: 1,
        };

        unsafe {
            device
                .reset_command_buffer(
                    self.transfer_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    self.transfer_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let qfi = vk::QUEUE_FAMILY_IGNORED;
        let to_transfer = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(frame.layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(qfi)
                .dst_queue_family_index(qfi)
                .image(frame.image)
                .subresource_range(src_planes),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(qfi)
                .dst_queue_family_index(qfi)
                .image(y_image)
                .subresource_range(color),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(qfi)
                .dst_queue_family_index(qfi)
                .image(uv_image)
                .subresource_range(color),
        ];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dep) };

        let color_layers = |aspect: vk::ImageAspectFlags, layer: u32| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: layer,
            layer_count: 1,
        };
        // Luma: source plane 0 -> y_image, full resolution.
        let y_region = vk::ImageCopy2::default()
            .src_subresource(color_layers(vk::ImageAspectFlags::PLANE_0, base_layer))
            .dst_subresource(color_layers(vk::ImageAspectFlags::COLOR, 0))
            .extent(vk::Extent3D {
                width: frame.coded_width,
                height: frame.coded_height,
                depth: 1,
            });
        let y_copy = vk::CopyImageInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(y_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(std::slice::from_ref(&y_region));
        // Chroma: source plane 1 -> uv_image, at chroma resolution.
        let (cdiv_hor, cdiv_vert) = frame.pixel_format.chroma_div();
        let uv_region = vk::ImageCopy2::default()
            .src_subresource(color_layers(vk::ImageAspectFlags::PLANE_1, base_layer))
            .dst_subresource(color_layers(vk::ImageAspectFlags::COLOR, 0))
            .extent(vk::Extent3D {
                width: frame.coded_width / cdiv_hor,
                height: frame.coded_height / cdiv_vert,
                depth: 1,
            });
        let uv_copy = vk::CopyImageInfo2::default()
            .src_image(frame.image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(uv_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(std::slice::from_ref(&uv_region));
        unsafe {
            device.cmd_copy_image2(self.transfer_command_buffer, &y_copy);
            device.cmd_copy_image2(self.transfer_command_buffer, &uv_copy);
        }

        let restore = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(frame.layout)
            .src_queue_family_index(qfi)
            .dst_queue_family_index(qfi)
            .image(frame.image)
            .subresource_range(src_planes)];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&restore);
        unsafe { device.cmd_pipeline_barrier2(self.transfer_command_buffer, &dep) };

        unsafe {
            device
                .end_command_buffer(self.transfer_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let command_buffers = [self.transfer_command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .reset_fences(&[self.transfer_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .queue_submit(self.transfer_queue, &[submit], self.transfer_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            device
                .wait_for_fences(&[self.transfer_fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for DecoderCommon {
    fn drop(&mut self) {
        self.destroy_session();
        unsafe {
            self.context.device().device_wait_idle().ok();
            if let Some((buffer, memory, _)) = self.readback.take() {
                self.context.device().destroy_buffer(buffer, None);
                self.context.device().free_memory(memory, None);
            }
            if self.bitstream_buffer != vk::Buffer::null() {
                self.context.device().unmap_memory(self.bitstream_memory);
                self.context
                    .device()
                    .destroy_buffer(self.bitstream_buffer, None);
                self.context
                    .device()
                    .free_memory(self.bitstream_memory, None);
            }
            self.context.device().destroy_fence(self.fence, None);
            self.context
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.context
                .device()
                .destroy_fence(self.transfer_fence, None);
            self.context
                .device()
                .destroy_command_pool(self.transfer_pool, None);
        }
    }
}

/// State of one image in the reorder buffer's pool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolState {
    /// Reusable.
    Free,
    /// Holds a decoded picture awaiting its turn in display order.
    Buffered,
    /// Returned to the caller; kept alive until the next `decode`/`flush`.
    HandedOut,
}

/// One pooled image, sized to the picture it currently holds.
///
/// No image view: the pool image is only ever a copy target and a `download`
/// source, neither of which needs one, and a valid multi-planar view would
/// require video-decode usage that some drivers do not allow here.
struct PoolImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    coded_width: u32,
    coded_height: u32,
    format: vk::Format,
    state: PoolState,
}

/// A buffered picture, referencing its pool image by index.
struct ReorderEntry {
    pool_index: usize,
    poc: i32,
    pts: u64,
    is_idr: bool,
    width: u32,
    height: u32,
    coded_width: u32,
    coded_height: u32,
    array_layer: u32,
    pixel_format: PixelFormat,
}

/// Reorders decoded pictures from decode order into display (POC) order.
///
/// Decoded pictures are returned in the order the hardware produces them, which
/// for streams with B-frames is not display order. This buffer copies each
/// decoded picture into a pool image — so it survives while later pictures are
/// decoded ahead of it — and emits them in POC order.
///
/// Emission follows the DPB bumping model: a picture is held until at most
/// `max_num_reorder_frames` pictures precede it in the buffer, an IDR drains the
/// previous coded video sequence (POC restarts at each IDR), and [`Self::flush`]
/// drains the rest at end of stream.
///
/// When disabled (decode-order mode) it is a pass-through: no copy, no latency,
/// and the returned frame points straight at the decoder's DPB image.
pub(crate) struct ReorderBuffer {
    enabled: bool,
    pool: Vec<PoolImage>,
    buffered: Vec<ReorderEntry>,
}

impl ReorderBuffer {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pool: Vec::new(),
            buffered: Vec::new(),
        }
    }

    /// Reclaim the images returned by the previous call. Must run before a
    /// `decode`/`flush` produces new frames: it is what makes the "valid until
    /// the next decode/flush" contract hold.
    fn begin_batch(&mut self) {
        for img in &mut self.pool {
            if img.state == PoolState::HandedOut {
                img.state = PoolState::Free;
            }
        }
    }

    /// Add a freshly decoded picture and return whatever is now ready to output.
    ///
    /// `reorder_depth` is the stream's `max_num_reorder_frames`.
    pub fn push(
        &mut self,
        common: &DecoderCommon,
        frame: &DecodedFrame,
        reorder_depth: usize,
    ) -> Result<Vec<DecodedFrame>> {
        if !self.enabled {
            return Ok(vec![frame.clone()]);
        }
        self.begin_batch();

        let mut out = Vec::new();
        // POC restarts at an IDR, so the previous sequence must be fully drained
        // before this picture (which belongs to the new one) is buffered.
        if frame.is_idr {
            out.extend(self.drain_all());
        }

        let pool_index = self.acquire_slot(common, frame)?;
        common.copy_frame_to_image(frame, self.pool[pool_index].image)?;
        self.pool[pool_index].state = PoolState::Buffered;
        self.buffered.push(ReorderEntry {
            pool_index,
            poc: frame.poc,
            pts: frame.pts,
            is_idr: frame.is_idr,
            width: frame.width,
            height: frame.height,
            coded_width: frame.coded_width,
            coded_height: frame.coded_height,
            array_layer: frame.array_layer,
            pixel_format: frame.pixel_format,
        });

        while self.buffered.len() > reorder_depth {
            out.push(self.pop_min_poc());
        }
        Ok(out)
    }

    /// Emit every buffered picture in display order. Call at end of stream.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        if !self.enabled {
            return Vec::new();
        }
        self.begin_batch();
        self.drain_all()
    }

    /// Drain the whole buffer in ascending POC order.
    fn drain_all(&mut self) -> Vec<DecodedFrame> {
        let mut out = Vec::with_capacity(self.buffered.len());
        while !self.buffered.is_empty() {
            out.push(self.pop_min_poc());
        }
        out
    }

    /// Remove and return the buffered picture with the smallest POC.
    fn pop_min_poc(&mut self) -> DecodedFrame {
        let i = self
            .buffered
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.poc)
            .map(|(i, _)| i)
            .expect("buffer is non-empty");
        let entry = self.buffered.remove(i);
        let img = &mut self.pool[entry.pool_index];
        img.state = PoolState::HandedOut;
        DecodedFrame {
            image: img.image,
            // Pool images carry no view (see PoolImage); a caller needing one
            // for GPU work creates it over `image` with the usage it wants.
            image_view: vk::ImageView::null(),
            // copy_frame_to_image leaves the pool image in TRANSFER_DST layout.
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            array_layer: entry.array_layer,
            pixel_format: entry.pixel_format,
            width: entry.width,
            height: entry.height,
            coded_width: entry.coded_width,
            coded_height: entry.coded_height,
            pts: entry.pts,
            poc: entry.poc,
            is_idr: entry.is_idr,
        }
    }

    /// A free pool image matching the picture, creating or resizing one as
    /// needed. Resolution changes are handled by recreating a mismatched slot.
    fn acquire_slot(&mut self, common: &DecoderCommon, frame: &DecodedFrame) -> Result<usize> {
        let format = common
            .session
            .as_ref()
            .map(|s| s.picture_format)
            .expect("session active while decoding");

        let matching = self.pool.iter().position(|p| {
            p.state == PoolState::Free
                && p.coded_width == frame.coded_width
                && p.coded_height == frame.coded_height
                && p.format == format
        });
        if let Some(i) = matching {
            return Ok(i);
        }

        let (image, memory) = create_pool_image(
            &common.context,
            frame.coded_width,
            frame.coded_height,
            format,
        )?;
        let slot = PoolImage {
            image,
            memory,
            coded_width: frame.coded_width,
            coded_height: frame.coded_height,
            format,
            state: PoolState::Free,
        };

        // Reuse a free-but-mismatched slot if one exists, else grow the pool.
        if let Some(i) = self.pool.iter().position(|p| p.state == PoolState::Free) {
            self.destroy_pool_image(common, i);
            self.pool[i] = slot;
            Ok(i)
        } else {
            self.pool.push(slot);
            Ok(self.pool.len() - 1)
        }
    }

    fn destroy_pool_image(&mut self, common: &DecoderCommon, i: usize) {
        let p = &self.pool[i];
        if p.image == vk::Image::null() {
            return;
        }
        unsafe {
            common.context.device().device_wait_idle().ok();
            common.context.device().destroy_image(p.image, None);
            common.context.device().free_memory(p.memory, None);
        }
    }

    /// Free every pool image. The caller must be done with handed-out frames.
    pub fn destroy(&mut self, common: &DecoderCommon) {
        for i in 0..self.pool.len() {
            self.destroy_pool_image(common, i);
            self.pool[i].image = vk::Image::null();
        }
        self.pool.clear();
        self.buffered.clear();
    }
}

/// A plain device-local image for the reorder pool: a copy target and readback
/// source, with no view and no video profile (so it works on every driver).
fn create_pool_image(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory)> {
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
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { context.device().create_image(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("reorder pool image: {}", e)))?;

    let reqs = unsafe { context.device().get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type(
        context.memory_properties(),
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No device-local memory for reorder pool".to_string())
    })?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;
    unsafe { context.device().bind_image_memory(image, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    Ok((image, memory))
}

/// A command pool, one primary command buffer, and a fence for `family`.
fn create_command_resources(
    context: &VideoContext,
    family: u32,
    label: &str,
) -> Result<(vk::CommandPool, vk::CommandBuffer, vk::Fence)> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = unsafe { context.device().create_command_pool(&pool_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("{} command pool: {}", label, e)))?;

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let buffer = unsafe { context.device().allocate_command_buffers(&alloc_info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?[0];

    let fence = unsafe {
        context
            .device()
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;

    Ok((pool, buffer, fence))
}

/// Byte layout of a decoded picture in the readback buffer, and of the visible
/// region the host actually copies out.
///
/// The decoded image is `coded_width` x `coded_height` (macroblock-aligned),
/// but only `width` x `height` is displayed, so the host reads a cropped
/// sub-rectangle out of a larger buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneGeometry {
    /// 1 for 4:4:4, 2 for 4:2:0.
    pub chroma_div: usize,
    pub y_stride: usize,
    pub y_size: usize,
    pub uv_stride: usize,
    pub total: usize,
    pub visible_y_stride: usize,
    pub visible_uv_stride: usize,
    pub visible_y_rows: usize,
    pub visible_uv_rows: usize,
}

impl PlaneGeometry {
    pub fn new(
        coded_width: u32,
        coded_height: u32,
        width: u32,
        height: u32,
        bit_depth: BitDepth,
        pixel_format: PixelFormat,
    ) -> Self {
        let bytes_per_sample = match bit_depth {
            BitDepth::Eight => 1,
            BitDepth::Ten => 2,
        };
        let chroma_div = match pixel_format {
            PixelFormat::Yuv444 => 1,
            _ => 2,
        };
        let y_stride = coded_width as usize * bytes_per_sample;
        let y_size = y_stride * coded_height as usize;
        let uv_stride = (coded_width as usize / chroma_div) * 2 * bytes_per_sample;
        let uv_size = uv_stride * (coded_height as usize / chroma_div);
        Self {
            chroma_div,
            y_stride,
            y_size,
            uv_stride,
            total: y_size + uv_size,
            visible_y_stride: width as usize * bytes_per_sample,
            visible_uv_stride: (width as usize / chroma_div) * 2 * bytes_per_sample,
            visible_y_rows: height as usize,
            visible_uv_rows: height as usize / chroma_div,
        }
    }

    /// Byte offset one past the last byte the host reads. Must not exceed
    /// [`Self::total`], or `download` would read out of bounds.
    pub fn max_read_offset(&self) -> usize {
        let last_y = if self.visible_y_rows == 0 {
            0
        } else {
            (self.visible_y_rows - 1) * self.y_stride + self.visible_y_stride
        };
        let last_uv = if self.visible_uv_rows == 0 {
            0
        } else {
            self.y_size + (self.visible_uv_rows - 1) * self.uv_stride + self.visible_uv_stride
        };
        last_y.max(last_uv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_geometry_reads_stay_in_bounds() {
        // (coded_width, coded_height, width, height)
        let cases = [
            (1920, 1088, 1920, 1080),
            (320, 240, 320, 240),
            (1280, 720, 1280, 720),
            (3840, 2160, 3840, 2160),
            (640, 480, 636, 476),
        ];
        for (cw, ch, w, h) in cases {
            for bd in [BitDepth::Eight, BitDepth::Ten] {
                for pf in [PixelFormat::Yuv420, PixelFormat::Yuv444] {
                    let g = PlaneGeometry::new(cw, ch, w, h, bd, pf);
                    assert!(
                        g.max_read_offset() <= g.total,
                        "{cw}x{ch} -> {w}x{h} {bd:?} {pf:?}: reads {} bytes but buffer is {}",
                        g.max_read_offset(),
                        g.total
                    );
                }
            }
        }
    }

    #[test]
    fn plane_geometry_nv12_1080p() {
        let g = PlaneGeometry::new(1920, 1088, 1920, 1080, BitDepth::Eight, PixelFormat::Yuv420);
        assert_eq!(g.y_stride, 1920);
        assert_eq!(g.y_size, 1920 * 1088);
        assert_eq!(g.uv_stride, 1920);
        assert_eq!(g.total, 1920 * 1088 + 1920 * 544);
        assert_eq!(g.visible_y_rows, 1080);
        assert_eq!(g.visible_uv_rows, 540);
    }

    #[test]
    fn plane_geometry_p010_is_two_bytes_per_sample() {
        let g = PlaneGeometry::new(1920, 1088, 1920, 1080, BitDepth::Ten, PixelFormat::Yuv420);
        assert_eq!(g.y_stride, 1920 * 2);
        assert_eq!(g.visible_y_stride, 1920 * 2);
    }

    #[test]
    fn plane_geometry_yuv444_chroma_is_full_size() {
        let g = PlaneGeometry::new(320, 240, 320, 240, BitDepth::Eight, PixelFormat::Yuv444);
        assert_eq!(g.chroma_div, 1);
        assert_eq!(g.uv_stride, 320 * 2);
        assert_eq!(g.visible_uv_rows, 240);
    }
}
