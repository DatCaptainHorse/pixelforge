//! Asynchronous decode pipelining, shared by all codecs.
//!
//! The mirror image of [`crate::encoder::pipeline`]. Each in-flight picture owns
//! a [`DecodeSlot`]: its own coded-data staging buffer, decode command buffer
//! and fence, plus a transfer command buffer and fence for the copy that gives
//! a display-order frame storage of its own. [`DecodePipeline`] rotates through
//! the slots, so the CPU can parse and submit picture N+1 while the GPU is still
//! decoding picture N instead of blocking on a fence after every picture.
//!
//! # Ordering
//!
//! Decode submissions share the video session and the DPB, so they must execute
//! in decode order on the GPU. A timeline semaphore enforces that: each decode
//! waits on the previous decode's value and signals its own.
//!
//! The reorder copy runs on the *transfer* queue, because a dedicated video
//! decode queue need not support transfer operations (it does not on RADV), so
//! it cannot be recorded into the decode command buffer. It gets a timeline of
//! its own and waits on two things: the decode that produced the picture, and
//! the previous copy. Chaining copies to each other is what makes "the last
//! submission finished" imply "every earlier one finished", which is the whole
//! basis for one fence covering a batch of frames.
//!
//! # Completion
//!
//! A single background *completion thread* waits on each batch's fence and
//! resolves that batch's [`DecodeFuture`]. Only the calling thread ever touches
//! a queue or a timeline; the completion thread only waits on fences and hands
//! frames onward, so the two never race on the same Vulkan object.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::task::{Context, Poll};
use std::thread::JoinHandle;

use ash::vk::{self, Handle};
use futures_channel::oneshot;

use crate::decoder::DecodedFrame;
use crate::error::{PixelForgeError, Result};
use crate::video::{
    SlotSync, create_bitstream_buffer, create_timeline_semaphore, map_bitstream_buffer,
};
use crate::vulkan::VideoContext;

/// Number of decode submissions allowed to be in flight at once.
///
/// Two for the same reason the encoder uses two: it fully overlaps parsing and
/// staging the next picture with the GPU decode of the current one, without
/// letting the GPU-serialized DPB chain grow.
pub(crate) const DECODE_PIPELINE_DEPTH: usize = 2;

/// Initial size of a slot's coded-data staging buffer; grown on demand.
const INITIAL_BITSTREAM_BUFFER_SIZE: usize = 1024 * 1024;

/// A handle to the frames one batch of decode submissions will produce.
///
/// Returned by `Decoder::decode` and `Decoder::flush`, one per call. Awaiting it
/// yields the frames that call emitted, once the GPU work it submitted has
/// completed. Futures resolve in call order, so awaiting them in the order they
/// were returned yields frames in the decoder's output order.
///
/// Dropping the future before it resolves is harmless, but it also drops the
/// frames it was carrying, which releases their DPB slots and pool images.
pub struct DecodeFuture {
    rx: oneshot::Receiver<Result<Vec<DecodedFrame>>>,
}

impl Future for DecodeFuture {
    type Output = Result<Vec<DecodedFrame>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            // The sender is only dropped without sending during teardown.
            Poll::Ready(Err(oneshot::Canceled)) => {
                Poll::Ready(Err(PixelForgeError::Synchronization(
                    "decode cancelled: decoder shut down before the frames were ready".to_string(),
                )))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// All per-picture resources that must be private to a single in-flight decode.
pub(crate) struct DecodeSlot {
    /// Coded data handed to the decode queue.
    pub bitstream_buffer: vk::Buffer,
    pub bitstream_memory: vk::DeviceMemory,
    pub bitstream_size: usize,
    /// Persistently mapped pointer to `bitstream_buffer`.
    pub bitstream_ptr: *mut u8,

    pub decode_command_buffer: vk::CommandBuffer,
    pub decode_fence: vk::Fence,

    /// Reorder copy: transfer queue, so it needs its own command buffer.
    pub transfer_command_buffer: vk::CommandBuffer,
    pub transfer_fence: vk::Fence,
}

/// One unit of work handed from the calling thread to the completion thread.
///
/// A `decode` call submits one item per picture, then a final item carrying the
/// sender. The completion thread accumulates frames across the items and
/// resolves the future when it reaches the one holding the sender, so a call
/// that decoded several pictures still yields a single batch of frames in
/// order.
struct WorkItem {
    /// Slot to release once this item's work is finished. `None` for an item
    /// that submitted nothing (the closing item of a batch).
    slot_index: Option<usize>,
    /// Fence signalling this item's GPU work, if it submitted any.
    fence: Option<vk::Fence>,
    /// Frames this item contributes to the batch.
    frames: Vec<DecodedFrame>,
    /// Present on the last item of a batch: resolves that call's future.
    result_tx: Option<oneshot::Sender<Result<Vec<DecodedFrame>>>>,
}

/// Rotating set of [`DecodeSlot`]s, the timelines that order their submissions,
/// and the completion thread that resolves futures.
pub(crate) struct DecodePipeline {
    slots: Vec<DecodeSlot>,
    current_slot: usize,

    /// Orders decode submissions, which share DPB state.
    decode_timeline: vk::Semaphore,
    next_decode_value: u64,
    last_decode_value: u64,

    /// Orders reorder copies against each other.
    copy_timeline: vk::Semaphore,
    next_copy_value: u64,
    last_copy_value: u64,

    /// The fence of the most recent submission, whichever queue it went to.
    last_fence: Option<vk::Fence>,

    slot_sync: Arc<SlotSync>,
    /// Sends submitted batches to the completion thread. Dropped on shutdown.
    work_tx: Option<Sender<WorkItem>>,
    completion_thread: Option<JoinHandle<()>>,
}

impl DecodePipeline {
    /// Allocate the timelines, `DECODE_PIPELINE_DEPTH` slots and the completion
    /// thread. Staging buffers start empty and are sized on first use.
    pub(crate) fn new(
        context: &VideoContext,
        decode_pool: vk::CommandPool,
        transfer_pool: vk::CommandPool,
    ) -> Result<Self> {
        let device = context.device();

        let alloc = |pool: vk::CommandPool| -> Result<Vec<vk::CommandBuffer>> {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(DECODE_PIPELINE_DEPTH as u32);
            unsafe { device.allocate_command_buffers(&info) }
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))
        };
        let decode_buffers = alloc(decode_pool)?;
        let transfer_buffers = alloc(transfer_pool)?;

        let mut slots = Vec::with_capacity(DECODE_PIPELINE_DEPTH);
        for i in 0..DECODE_PIPELINE_DEPTH {
            // Created signaled: nothing has been submitted yet, so a wait must
            // return immediately.
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let decode_fence = unsafe { device.create_fence(&fence_info, None) }
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            let transfer_fence = unsafe { device.create_fence(&fence_info, None) }
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            slots.push(DecodeSlot {
                bitstream_buffer: vk::Buffer::null(),
                bitstream_memory: vk::DeviceMemory::null(),
                bitstream_size: 0,
                bitstream_ptr: std::ptr::null_mut(),
                decode_command_buffer: decode_buffers[i],
                decode_fence,
                transfer_command_buffer: transfer_buffers[i],
                transfer_fence,
            });
        }

        let slot_sync = Arc::new(SlotSync::new(slots.len()));
        let (work_tx, work_rx) = std::sync::mpsc::channel::<WorkItem>();
        let thread_device = device.clone();
        let thread_sync = slot_sync.clone();
        let completion_thread = std::thread::Builder::new()
            .name("pixelforge-decode-completion".to_string())
            .spawn(move || run_completion_thread(thread_device, work_rx, thread_sync))
            .map_err(|e| PixelForgeError::CommandBuffer(format!("spawn completion thread: {e}")))?;

        Ok(Self {
            slots,
            current_slot: 0,
            decode_timeline: create_timeline_semaphore(context)?,
            next_decode_value: 1,
            last_decode_value: 0,
            copy_timeline: create_timeline_semaphore(context)?,
            next_copy_value: 1,
            last_copy_value: 0,
            last_fence: None,
            slot_sync,
            work_tx: Some(work_tx),
            completion_thread: Some(completion_thread),
        })
    }

    pub(crate) fn current(&self) -> &DecodeSlot {
        &self.slots[self.current_slot]
    }

    /// Wait until the current slot's previous submission is finished, so its
    /// command buffers and staging buffer can be recorded over.
    pub(crate) fn wait_current_free(&self) {
        self.slot_sync.wait_free(self.current_slot);
    }

    /// Wait until nothing is in flight. Used before mutating shared session
    /// state and at teardown.
    pub(crate) fn wait_all_free(&self) {
        self.slot_sync.wait_all_free();
    }

    /// Ensure the current slot's staging buffer holds at least `size` bytes.
    ///
    /// Growing destroys the old buffer, so the slot must already be free (the
    /// caller waits before recording) and nothing else may reference it.
    pub(crate) fn ensure_bitstream_capacity(
        &mut self,
        context: &VideoContext,
        size: usize,
        profile_info: &vk::VideoProfileInfoKHR,
    ) -> Result<()> {
        let slot = &mut self.slots[self.current_slot];
        if slot.bitstream_size >= size && slot.bitstream_buffer != vk::Buffer::null() {
            return Ok(());
        }
        let new_size = size.max(INITIAL_BITSTREAM_BUFFER_SIZE).next_power_of_two();

        if slot.bitstream_buffer != vk::Buffer::null() {
            unsafe {
                context.device().unmap_memory(slot.bitstream_memory);
                context.device().destroy_buffer(slot.bitstream_buffer, None);
                context.device().free_memory(slot.bitstream_memory, None);
            }
        }

        let (buffer, memory) = create_bitstream_buffer(
            context,
            new_size,
            vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR,
            profile_info,
        )?;
        slot.bitstream_ptr = map_bitstream_buffer(context, memory, new_size)?;
        slot.bitstream_buffer = buffer;
        slot.bitstream_memory = memory;
        slot.bitstream_size = new_size;
        Ok(())
    }

    /// Submit the current slot's recorded decode without waiting for it.
    ///
    /// Chains onto the decode timeline so the GPU keeps decodes in decode order.
    pub(crate) fn submit_decode(
        &mut self,
        device: &ash::Device,
        decode_queue: vk::Queue,
    ) -> Result<()> {
        let slot = &self.slots[self.current_slot];
        let signal_value = self.next_decode_value;

        unsafe {
            device
                .end_command_buffer(slot.decode_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .reset_fences(&[slot.decode_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }

        // A timeline wait on value 0 is satisfied immediately, so the first
        // submission needs no special case.
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.decode_timeline)
            .value(self.last_decode_value)
            .stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.decode_timeline)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)];
        let command_buffers =
            [vk::CommandBufferSubmitInfo::default().command_buffer(slot.decode_command_buffer)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signals);

        unsafe {
            device
                .queue_submit2(decode_queue, &[submit], slot.decode_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }

        self.last_decode_value = signal_value;
        self.next_decode_value = signal_value + 1;
        self.last_fence = Some(slot.decode_fence);
        Ok(())
    }

    /// Submit the current slot's recorded reorder copy without waiting for it.
    ///
    /// Waits on the decode that produced the picture and on the previous copy,
    /// so copies complete in submission order.
    pub(crate) fn submit_copy(
        &mut self,
        device: &ash::Device,
        transfer_queue: vk::Queue,
    ) -> Result<()> {
        let slot = &self.slots[self.current_slot];
        let signal_value = self.next_copy_value;

        unsafe {
            device
                .end_command_buffer(slot.transfer_command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .reset_fences(&[slot.transfer_fence])
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }

        let waits = [
            // The decode that produced the picture we are copying.
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.decode_timeline)
                .value(self.last_decode_value)
                .stage_mask(vk::PipelineStageFlags2::COPY),
            // The previous copy, so copies complete in submission order.
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.copy_timeline)
                .value(self.last_copy_value)
                .stage_mask(vk::PipelineStageFlags2::COPY),
        ];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.copy_timeline)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::COPY)];
        let command_buffers =
            [vk::CommandBufferSubmitInfo::default().command_buffer(slot.transfer_command_buffer)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signals);

        unsafe {
            device
                .queue_submit2(transfer_queue, &[submit], slot.transfer_fence)
                .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
        }

        self.last_copy_value = signal_value;
        self.next_copy_value = signal_value + 1;
        self.last_fence = Some(slot.transfer_fence);
        Ok(())
    }

    /// Close off the current picture: hand its slot and frames to the
    /// completion thread and move to the next slot.
    ///
    /// The picture's work is tied to its most recent submission's fence. Both
    /// timelines order the copy after its own decode, so waiting on the later
    /// fence covers the earlier submission too.
    pub(crate) fn end_picture(&mut self, frames: Vec<DecodedFrame>) {
        let slot_index = self.current_slot;
        let fence = self.last_fence.take();
        // Busy before the hand-off, so the completion thread cannot clear the
        // flag before it is set.
        self.slot_sync.set_busy(slot_index);
        self.send(WorkItem {
            slot_index: Some(slot_index),
            fence,
            frames,
            result_tx: None,
        });
        self.current_slot = (self.current_slot + 1) % self.slots.len();
    }

    /// Close off a `decode`/`flush` call and take the future for its frames.
    ///
    /// `frames` are the ones emitted without any new submission of their own,
    /// which is everything a flush produces. Frames already handed over by
    /// [`Self::end_picture`] are accumulated by the completion thread and
    /// delivered through the same future.
    pub(crate) fn finish_batch(&mut self, frames: Vec<DecodedFrame>) -> DecodeFuture {
        let (result_tx, rx) = oneshot::channel();
        self.send(WorkItem {
            slot_index: None,
            fence: None,
            frames,
            result_tx: Some(result_tx),
        });
        DecodeFuture { rx }
    }

    fn send(&self, work: WorkItem) {
        if let Some(tx) = &self.work_tx {
            // The receiver only disconnects during shutdown, after the queues
            // are idle; a failed send there is benign.
            let _ = tx.send(work);
        }
    }

    /// Stop the completion thread once it has drained its queue.
    fn shutdown(&mut self) {
        self.work_tx.take();
        if let Some(handle) = self.completion_thread.take() {
            let _ = handle.join();
        }
    }

    /// Destroy every slot's resources and both timelines.
    ///
    /// # Safety
    ///
    /// All queues that may reference these resources must be idle.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        // Join the completion thread before freeing the fences it waits on.
        self.shutdown();

        for slot in &mut self.slots {
            unsafe {
                if !slot.bitstream_ptr.is_null() {
                    device.unmap_memory(slot.bitstream_memory);
                    slot.bitstream_ptr = std::ptr::null_mut();
                }
                if slot.bitstream_buffer != vk::Buffer::null() {
                    device.destroy_buffer(slot.bitstream_buffer, None);
                    device.free_memory(slot.bitstream_memory, None);
                }
                device.destroy_fence(slot.decode_fence, None);
                device.destroy_fence(slot.transfer_fence, None);
            }
        }
        unsafe {
            device.destroy_semaphore(self.decode_timeline, None);
            device.destroy_semaphore(self.copy_timeline, None);
        }
    }
}

/// Completion-thread body: wait for each item's fence, accumulate its frames,
/// and resolve a call's future when its closing item arrives.
fn run_completion_thread(
    device: ash::Device,
    work_rx: Receiver<WorkItem>,
    slot_sync: Arc<SlotSync>,
) {
    let mut batch: Vec<DecodedFrame> = Vec::new();
    let mut failure: Option<PixelForgeError> = None;

    for work in work_rx {
        if let Some(fence) = work.fence
            && !fence.is_null()
        {
            let wait = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) };
            if let Err(e) = wait {
                failure.get_or_insert(PixelForgeError::Synchronization(e.to_string()));
            }
        }
        batch.extend(work.frames);

        // Resolve *before* freeing the slot, so that once every slot is free
        // every future has already been resolved. A dropped receiver makes the
        // send a no-op, which also drops the frames and releases their storage.
        if let Some(tx) = work.result_tx {
            let frames = std::mem::take(&mut batch);
            let result = match failure.take() {
                Some(e) => Err(e),
                None => Ok(frames),
            };
            let _ = tx.send(result);
        }
        if let Some(index) = work.slot_index {
            slot_sync.set_free(index);
        }
    }
}
