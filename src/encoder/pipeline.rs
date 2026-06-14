//! Asynchronous (push) encode pipelining, shared by all codecs.
//!
//! Each in-flight frame owns an [`EncodeSlot`] (its own input image, bitstream
//! buffer, encode command buffer, fence and query pool). [`EncodePipeline`]
//! rotates through the slots so that the CPU can record and submit frame N+1
//! while the GPU is still encoding frame N, instead of blocking on a fence after
//! every frame.
//!
//! Bitstream readback is performed off the calling thread: a single background
//! *completion thread* waits on each submission's fence, copies the bitstream
//! out, and pushes the finished [`EncodedPacket`] onto a channel the moment the
//! GPU signals — rather than deferring readback until the slot is reused. This
//! delivers each packet at roughly the GPU encode time instead of one or two
//! `encode()` calls later.
//!
//! The DPB images and video session are shared across slots, so encode
//! submissions must still run in DPB order on the GPU. That ordering is enforced
//! with a single timeline semaphore (each submit waits on the previous submit's
//! value and signals its own). Only the calling thread ever touches the queue or
//! the timeline; the completion thread only waits on fences and reads bitstream
//! buffers, so the two never race on the same Vulkan object.
//!
//! Slot reuse is coordinated with a per-slot "busy" flag ([`SlotSync`]): a slot
//! is busy from submit until the completion thread has finished reading its
//! bitstream. The calling thread waits for the current slot to be free before
//! converting into / recording over it, which also covers the write-after-read
//! hazard on the shared input image.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use ash::vk;

use crate::encoder::resources::{
    clear_input_image, create_bitstream_buffer, create_encode_feedback_query_pool, create_image,
    create_timeline_semaphore, map_bitstream_buffer, submit_encode_only, wait_and_read_bitstream,
    ClearImageParams,
};
use crate::encoder::{BitDepth, EncodedPacket, EncodedPacketReceiver, FrameType, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// Number of encode submissions allowed to be in flight at once.
///
/// See the module docs and `docs` discussion for why 2 is the sweet spot: it
/// fully overlaps capture/convert/upload of the next frame with the GPU encode
/// of the current one, while keeping the GPU-serialized DPB chain from growing.
pub(crate) const ENCODE_PIPELINE_DEPTH: usize = 2;

/// Per-frame packet info captured at submit time and attached to the bitstream
/// when the completion thread reads the slot back.
pub(crate) struct SlotPacketMetadata {
    pub frame_type: FrameType,
    pub is_key_frame: bool,
    pub pts: u64,
    pub dts: u64,
    /// Codec header (SPS/PPS, VPS/SPS/PPS, or AV1 sequence header) to prepend.
    /// `Some` only for frames that carry one (e.g. the IDR/key frame).
    pub header: Option<Vec<u8>>,
}

/// A raw pointer wrapper asserting `Send` so the persistently-mapped bitstream
/// pointer can be handed to the completion thread. The memory is only read by
/// that thread, and only after the encode fence has signalled.
struct SendPtr(*const u8);
// SAFETY: the pointed-to bitstream buffer is host-coherent, persistently mapped
// for the lifetime of the slot, and read exclusively by the completion thread
// after the encode fence signals. The calling thread does not touch the buffer
// until the slot is marked free again (after this read completes).
unsafe impl Send for SendPtr {}

/// A submission handed from the calling thread to the completion thread.
struct WorkItem {
    slot_index: usize,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    bitstream_ptr: SendPtr,
    metadata: SlotPacketMetadata,
}

/// Cross-thread per-slot readiness. A slot is "busy" from the moment its encode
/// is submitted until the completion thread has finished reading its bitstream.
struct SlotSync {
    busy: Mutex<Vec<bool>>,
    cv: Condvar,
}

impl SlotSync {
    fn new(slot_count: usize) -> Self {
        Self {
            busy: Mutex::new(vec![false; slot_count]),
            cv: Condvar::new(),
        }
    }

    /// Block until slot `index` is free (its previous encode has been read back).
    fn wait_free(&self, index: usize) {
        let mut busy = self.busy.lock().unwrap();
        while busy[index] {
            busy = self.cv.wait(busy).unwrap();
        }
    }

    /// Block until every slot is free (no submissions in flight).
    fn wait_all_free(&self) {
        let mut busy = self.busy.lock().unwrap();
        while busy.iter().any(|b| *b) {
            busy = self.cv.wait(busy).unwrap();
        }
    }

    /// Mark a slot busy at submit time. No notify: nobody waits to *enter* busy.
    fn set_busy(&self, index: usize) {
        self.busy.lock().unwrap()[index] = true;
    }

    /// Mark a slot free once its bitstream has been read; wake any waiters.
    fn set_free(&self, index: usize) {
        self.busy.lock().unwrap()[index] = false;
        self.cv.notify_all();
    }
}

/// All per-frame resources that must be private to a single in-flight encode.
pub(crate) struct EncodeSlot {
    pub input_image: vk::Image,
    pub input_image_memory: vk::DeviceMemory,
    pub input_image_view: vk::ImageView,
    /// Tracked layout of `input_image` (to avoid UB when transitioning).
    pub input_image_layout: vk::ImageLayout,

    pub bitstream_buffer: vk::Buffer,
    pub bitstream_buffer_memory: vk::DeviceMemory,
    pub bitstream_buffer_size: usize,
    /// Persistently mapped pointer to the bitstream buffer.
    pub bitstream_buffer_ptr: *mut u8,

    pub encode_command_buffer: vk::CommandBuffer,
    pub encode_fence: vk::Fence,
    pub query_pool: vk::QueryPool,

    /// Packet metadata recorded before submit, moved to the completion thread
    /// with the work item.
    pub pending_metadata: Option<SlotPacketMetadata>,
}

/// Configuration for building an [`EncodePipeline`].
pub(crate) struct PipelineConfig<'a> {
    pub context: &'a VideoContext,
    pub aligned_width: u32,
    pub aligned_height: u32,
    pub picture_format: vk::Format,
    pub pixel_format: PixelFormat,
    pub bit_depth: BitDepth,
    pub bitstream_buffer_size: usize,
    /// Codec profile (with the codec-specific profile chained in) used for the
    /// input images, bitstream buffers and feedback query pools.
    pub profile_info: &'a vk::VideoProfileInfoKHR<'a>,
    pub command_pool: vk::CommandPool,
    /// Transfer command buffer/fence reused to zero-initialize each input image.
    pub upload_command_buffer: vk::CommandBuffer,
    pub upload_fence: vk::Fence,
}

/// Rotating set of [`EncodeSlot`]s plus the timeline semaphore that orders their
/// encode submissions and the completion thread that reads bitstreams back.
pub(crate) struct EncodePipeline {
    slots: Vec<EncodeSlot>,
    current_slot: usize,
    /// Orders encode submissions that share DPB state.
    timeline: vk::Semaphore,
    /// Value the next submit will signal.
    next_value: u64,
    /// Value the most recent submit signaled (0 = none yet).
    last_value: u64,

    /// Per-slot busy flags shared with the completion thread.
    slot_sync: Arc<SlotSync>,
    /// Sends submitted work to the completion thread. Dropped on shutdown to end
    /// the thread.
    work_tx: Option<Sender<WorkItem>>,
    /// Receives finished packets from the completion thread. `None` once the
    /// caller has taken ownership via [`EncodePipeline::take_packet_receiver`].
    packet_rx: Option<EncodedPacketReceiver>,
    /// The completion thread handle, joined on shutdown.
    completion_thread: Option<JoinHandle<()>>,
}

impl EncodePipeline {
    /// Allocate the timeline semaphore, `ENCODE_PIPELINE_DEPTH` slots and spawn
    /// the bitstream-readback completion thread.
    pub(crate) fn new(config: &PipelineConfig) -> Result<Self> {
        let context = config.context;
        let device = context.device();

        let timeline = create_timeline_semaphore(context)?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(config.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(ENCODE_PIPELINE_DEPTH as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        let mut slots = Vec::with_capacity(ENCODE_PIPELINE_DEPTH);
        for &encode_command_buffer in &command_buffers {
            let (input_image, input_image_memory, input_image_view) = create_image(
                context,
                config.aligned_width,
                config.aligned_height,
                config.picture_format,
                false,
                config.profile_info,
            )?;

            let (bitstream_buffer, bitstream_buffer_memory) =
                create_bitstream_buffer(context, config.bitstream_buffer_size, config.profile_info)?;
            let bitstream_buffer_ptr =
                map_bitstream_buffer(context, bitstream_buffer_memory, config.bitstream_buffer_size)?;

            // Zero the padding between the user dimensions and the aligned coded
            // extent so the first frame has no undefined samples.
            clear_input_image(
                context,
                &ClearImageParams {
                    command_buffer: config.upload_command_buffer,
                    fence: config.upload_fence,
                    queue: context.transfer_queue(),
                    image: input_image,
                    width: config.aligned_width,
                    height: config.aligned_height,
                    pixel_format: config.pixel_format,
                    bit_depth: config.bit_depth,
                },
            )?;

            // Created signaled so it is safe to wait on before the first encode;
            // `submit_encode_only` resets it before each submit.
            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let encode_fence = unsafe { device.create_fence(&fence_create_info, None) }
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            let mut profile = *config.profile_info;
            let query_pool = create_encode_feedback_query_pool(context, &mut profile)?;

            slots.push(EncodeSlot {
                input_image,
                input_image_memory,
                input_image_view,
                input_image_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                bitstream_buffer,
                bitstream_buffer_memory,
                bitstream_buffer_size: config.bitstream_buffer_size,
                bitstream_buffer_ptr,
                encode_command_buffer,
                encode_fence,
                query_pool,
                pending_metadata: None,
            });
        }

        let slot_sync = Arc::new(SlotSync::new(slots.len()));
        let (work_tx, work_rx) = std::sync::mpsc::channel::<WorkItem>();
        let (packet_tx, packet_rx) = std::sync::mpsc::channel::<Result<EncodedPacket>>();

        // The completion thread only needs a handle to the device; the Vulkan
        // device handle is internally shared and safe to use from this thread
        // for fence waits, query reads and host-coherent buffer reads.
        let thread_device = device.clone();
        let thread_sync = slot_sync.clone();
        let completion_thread = std::thread::Builder::new()
            .name("pixelforge-encode-readback".to_string())
            .spawn(move || {
                run_completion_thread(thread_device, work_rx, packet_tx, thread_sync);
            })
            .map_err(|e| PixelForgeError::CommandBuffer(format!("spawn readback thread: {e}")))?;

        Ok(Self {
            slots,
            current_slot: 0,
            timeline,
            next_value: 1,
            last_value: 0,
            slot_sync,
            work_tx: Some(work_tx),
            packet_rx: Some(packet_rx),
            completion_thread: Some(completion_thread),
        })
    }

    /// The slot the next frame will be encoded into.
    pub(crate) fn current(&self) -> &EncodeSlot {
        &self.slots[self.current_slot]
    }

    pub(crate) fn current_mut(&mut self) -> &mut EncodeSlot {
        &mut self.slots[self.current_slot]
    }

    /// Return the current slot's input image, first waiting until the slot is
    /// free so it is safe to use as a convert/upload target (write-after-read on
    /// the shared input image).
    pub(crate) fn input_image(&self) -> vk::Image {
        self.slot_sync.wait_free(self.current_slot);
        self.slots[self.current_slot].input_image
    }

    /// Wait until the current slot is free to record over and submit.
    pub(crate) fn wait_current_free(&self) {
        self.slot_sync.wait_free(self.current_slot);
    }

    /// Wait until every in-flight submission has been read back. Used before
    /// mutating shared session state and at teardown.
    pub(crate) fn wait_all_free(&self) {
        self.slot_sync.wait_all_free();
    }

    /// Record the metadata for the packet that the current slot will produce.
    /// Must be called before [`EncodePipeline::submit_current`].
    pub(crate) fn set_pending_metadata(&mut self, metadata: SlotPacketMetadata) {
        self.slots[self.current_slot].pending_metadata = Some(metadata);
    }

    /// Submit the current slot's recorded command buffer without waiting, and
    /// hand the slot to the completion thread for bitstream readback.
    ///
    /// Chains onto the timeline semaphore so the GPU keeps encodes in DPB order,
    /// and marks the slot busy until the completion thread reads it back.
    pub(crate) fn submit_current(
        &mut self,
        device: &ash::Device,
        encode_queue: vk::Queue,
    ) -> Result<()> {
        let wait = (self.last_value > 0).then_some((self.timeline, self.last_value));
        let signal_value = self.next_value;
        let slot_index = self.current_slot;

        // Capture the Copy handles + metadata, releasing the slot borrow before
        // touching the cross-thread channels.
        let (command_buffer, fence, query_pool, bitstream_ptr, metadata) = {
            let slot = &mut self.slots[slot_index];
            let metadata = slot.pending_metadata.take().ok_or_else(|| {
                PixelForgeError::CommandBuffer(
                    "submit_current called without pending packet metadata".to_string(),
                )
            })?;
            (
                slot.encode_command_buffer,
                slot.encode_fence,
                slot.query_pool,
                slot.bitstream_buffer_ptr as *const u8,
                metadata,
            )
        };

        unsafe {
            submit_encode_only(
                device,
                command_buffer,
                fence,
                encode_queue,
                wait,
                Some((self.timeline, signal_value)),
            )?;
        }

        self.last_value = signal_value;
        self.next_value = signal_value + 1;

        // Mark busy *before* handing the work off, so the completion thread can
        // never clear the flag before it is set.
        self.slot_sync.set_busy(slot_index);

        let work = WorkItem {
            slot_index,
            fence,
            query_pool,
            bitstream_ptr: SendPtr(bitstream_ptr),
            metadata,
        };
        if let Some(tx) = &self.work_tx {
            // The receiver only disconnects during shutdown, after the queue is
            // idle; a failed send there is benign.
            let _ = tx.send(work);
        }

        Ok(())
    }

    /// Advance to the next slot after a frame has been submitted.
    pub(crate) fn advance(&mut self) {
        self.current_slot = (self.current_slot + 1) % self.slots.len();
    }

    /// Try to receive a finished packet without blocking.
    ///
    /// Returns `None` if no packet is ready, or if the caller has taken
    /// ownership of the receiver via [`EncodePipeline::take_packet_receiver`].
    pub(crate) fn poll_packet(&self) -> Option<Result<EncodedPacket>> {
        self.packet_rx.as_ref().and_then(|rx| rx.try_recv().ok())
    }

    /// Take ownership of the packet receiver for out-of-band consumption (e.g.
    /// a dedicated sender thread). After this, [`EncodePipeline::poll_packet`]
    /// and [`EncodePipeline::flush`] no longer return packets.
    pub(crate) fn take_packet_receiver(&mut self) -> Option<EncodedPacketReceiver> {
        self.packet_rx.take()
    }

    /// Wait for all in-flight frames to be read back, returning any packets not
    /// yet polled. Used to drain at end of stream.
    ///
    /// If the receiver has been taken, this only acts as a completion barrier
    /// and returns an empty `Vec` (packets are delivered to the taken receiver).
    pub(crate) fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        // Once every slot is free, the completion thread has already sent every
        // packet (it sends before marking a slot free), so draining now is
        // complete.
        self.slot_sync.wait_all_free();

        let mut packets = Vec::new();
        if let Some(rx) = &self.packet_rx {
            while let Ok(result) = rx.try_recv() {
                packets.push(result?);
            }
        }
        Ok(packets)
    }

    /// Stop the completion thread and wait for it to finish any in-flight
    /// readback. Safe to call more than once.
    fn shutdown(&mut self) {
        // Dropping the sender ends the thread's `for work in rx` loop once it
        // has drained outstanding items.
        self.work_tx.take();
        if let Some(handle) = self.completion_thread.take() {
            let _ = handle.join();
        }
    }

    /// Destroy all slot resources and the timeline semaphore.
    ///
    /// # Safety
    ///
    /// All queues that may reference these resources must be idle.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        // Join the readback thread before freeing the fences/buffers it reads.
        self.shutdown();

        for slot in &mut self.slots {
            if !slot.bitstream_buffer_ptr.is_null() {
                device.unmap_memory(slot.bitstream_buffer_memory);
                slot.bitstream_buffer_ptr = std::ptr::null_mut();
            }
            device.destroy_query_pool(slot.query_pool, None);
            device.destroy_fence(slot.encode_fence, None);
            device.destroy_buffer(slot.bitstream_buffer, None);
            device.free_memory(slot.bitstream_buffer_memory, None);
            device.destroy_image_view(slot.input_image_view, None);
            device.destroy_image(slot.input_image, None);
            device.free_memory(slot.input_image_memory, None);
        }
        device.destroy_semaphore(self.timeline, None);
    }
}

/// Completion-thread body: wait on each submission's fence, copy its bitstream
/// out, deliver the packet, then mark the slot free.
fn run_completion_thread(
    device: ash::Device,
    work_rx: Receiver<WorkItem>,
    packet_tx: Sender<Result<EncodedPacket>>,
    slot_sync: Arc<SlotSync>,
) {
    for work in work_rx {
        let result = unsafe {
            wait_and_read_bitstream(&device, work.fence, work.query_pool, work.bitstream_ptr.0)
        };

        let packet = result.map(|bitstream| {
            let mut data = work.metadata.header.unwrap_or_default();
            data.extend_from_slice(&bitstream);
            EncodedPacket {
                data,
                frame_type: work.metadata.frame_type,
                is_key_frame: work.metadata.is_key_frame,
                pts: work.metadata.pts,
                dts: work.metadata.dts,
            }
        });

        // Deliver the packet *before* freeing the slot. This ordering means that
        // once all slots are observed free, every packet has already been sent,
        // which `flush` relies on for completeness.
        let _ = packet_tx.send(packet);
        slot_sync.set_free(work.slot_index);
    }
}
