//! Frame ownership: what a decoded picture is copied into, what keeps it alive,
//! and how display order is restored.
//!
//! The decoder hands out [`DecodedFrame`]s whose storage stays reserved until
//! the caller drops them. Two kinds of storage exist: a pinned DPB slot (no
//! copy, decode-order output) and a pool image (a copy, which is what lets a
//! picture outlive the slot it was decoded into, and so what makes display-order
//! reordering possible).

use std::sync::{Arc, Condvar, Mutex};

use ash::vk;

use crate::decoder::DecodedFrame;
use crate::decoder::common::DecoderCommon;
use crate::encoder::PixelFormat;
use crate::error::{PixelForgeError, Result};
use crate::video::find_memory_type;
use crate::vulkan::VideoContext;

/// A freshly decoded picture, before it is handed to the caller.
///
/// The pixels live in a DPB slot (or the session's decode output image), so this
/// descriptor is only valid until the next decode submission. [`ReorderBuffer`]
/// turns it into a [`DecodedFrame`], which owns its storage.
pub(crate) struct DecodedPicture {
    /// DPB slot the picture was decoded into.
    pub slot: u8,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub array_layer: u32,
    pub pixel_format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub coded_width: u32,
    pub coded_height: u32,
    pub pts: u64,
    pub display_order: i32,
    pub is_keyframe: bool,
}

/// An image in the frame pool, by index.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolIndex(usize);

/// DPB slots reserved by frames the caller is still holding.
///
/// A pinned slot is one the driver decoded into and the caller is now reading,
/// so the codec must not decode over it. Releases come from [`FramePin::drop`]
/// on whatever thread the frame died on, which is why the decoder waits on the
/// condition variable rather than polling.
#[derive(Debug, Default)]
pub(crate) struct SlotPins {
    /// One bit per DPB slot.
    pinned: Mutex<u32>,
    released: Condvar,
}

impl SlotPins {
    pub fn is_pinned(&self, slot: u8) -> bool {
        *self.pinned.lock().unwrap() & (1 << slot) != 0
    }

    pub fn any_pinned(&self) -> bool {
        *self.pinned.lock().unwrap() != 0
    }

    pub fn pin(&self, slot: u8) {
        *self.pinned.lock().unwrap() |= 1 << slot;
    }

    pub fn release(&self, slot: u8) {
        *self.pinned.lock().unwrap() &= !(1 << slot);
        self.released.notify_all();
    }

    /// Block until at least one pinned slot is released. Returns immediately if
    /// nothing is pinned, since then no release can ever come.
    pub fn wait_for_release(&self) {
        let mut pinned = self.pinned.lock().unwrap();
        let before = *pinned;
        if before == 0 {
            return;
        }
        while *pinned == before {
            pinned = self.released.wait(pinned).unwrap();
        }
    }

    /// Forget every pin. Called when the session is torn down and the slots it
    /// refers to no longer exist.
    pub fn clear(&self) {
        *self.pinned.lock().unwrap() = 0;
        self.released.notify_all();
    }
}

/// Pins released by frames the caller has dropped, waiting to be reclaimed.
///
/// A [`DecodedFrame`] can be dropped on any thread, including while the decoder
/// is submitting the next picture, so releases land here and the decoder folds
/// them in the next time it needs storage.
#[derive(Debug, Default)]
pub(crate) struct ReleaseQueue {
    released: Mutex<Vec<PoolIndex>>,
}

impl ReleaseQueue {
    fn push(&self, index: PoolIndex) {
        self.released.lock().unwrap().push(index);
    }

    fn take(&self) -> Vec<PoolIndex> {
        std::mem::take(&mut *self.released.lock().unwrap())
    }
}

/// Keeps a [`DecodedFrame`]'s storage reserved for as long as the frame lives.
///
/// Dropping the frame drops this, which returns the storage to the decoder.
#[derive(Debug)]
pub(crate) enum FramePin {
    /// A copy in the frame pool. Released lazily: the decoder never waits on
    /// pool images, it just allocates another one.
    Pool {
        index: PoolIndex,
        releases: Arc<ReleaseQueue>,
    },
    /// The DPB slot the picture was decoded into, handed out without a copy.
    /// Released eagerly, because a decode may be blocked waiting for it.
    DpbSlot { slot: u8, pins: Arc<SlotPins> },
}

impl FramePin {
    /// Whether this frame's image is a DPB slot the decoder is still using,
    /// rather than a private copy.
    pub(crate) fn borrows_dpb_image(&self) -> bool {
        matches!(self, FramePin::DpbSlot { .. })
    }
}

impl Drop for FramePin {
    fn drop(&mut self) {
        match self {
            FramePin::Pool { index, releases } => releases.push(*index),
            FramePin::DpbSlot { slot, pins } => pins.release(*slot),
        }
    }
}

/// State of one image in the frame pool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolState {
    /// Reusable.
    Free,
    /// Holds a decoded picture awaiting its turn in display order.
    Buffered,
    /// Handed to the caller; reserved until their [`DecodedFrame`] is dropped.
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

/// Images that decoded pictures are copied into so they outlive their DPB slot.
///
/// The pool grows on demand and shrinks never: an image is reused as soon as the
/// frame holding it is dropped, so steady-state size is "what the caller holds
/// at once, plus the reorder depth". Nothing here blocks, because pool images
/// are ordinary allocations rather than a hardware-bounded resource.
struct FramePool {
    images: Vec<PoolImage>,
    releases: Arc<ReleaseQueue>,
}

impl FramePool {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            releases: Arc::new(ReleaseQueue::default()),
        }
    }

    /// Fold in the images released by dropped frames.
    fn reclaim(&mut self) {
        for PoolIndex(i) in self.releases.take() {
            self.images[i].state = PoolState::Free;
        }
    }

    /// A free image matching `picture`'s geometry, creating or resizing one as
    /// needed. Resolution changes recreate a mismatched image.
    fn acquire(&mut self, common: &DecoderCommon, picture: &DecodedPicture) -> Result<usize> {
        self.reclaim();

        let format = common
            .session
            .as_ref()
            .map(|s| s.picture_format)
            .expect("session active while decoding");

        let matching = self.images.iter().position(|p| {
            p.state == PoolState::Free
                && p.coded_width == picture.coded_width
                && p.coded_height == picture.coded_height
                && p.format == format
        });
        if let Some(i) = matching {
            return Ok(i);
        }

        let (image, memory) = create_pool_image(
            &common.context,
            picture.coded_width,
            picture.coded_height,
            format,
        )?;
        let slot = PoolImage {
            image,
            memory,
            coded_width: picture.coded_width,
            coded_height: picture.coded_height,
            format,
            state: PoolState::Free,
        };

        // Reuse a free-but-mismatched image if one exists, else grow the pool.
        if let Some(i) = self.images.iter().position(|p| p.state == PoolState::Free) {
            self.destroy_image(common, i);
            self.images[i] = slot;
            Ok(i)
        } else {
            self.images.push(slot);
            Ok(self.images.len() - 1)
        }
    }

    /// Mark `index` as handed to the caller and mint its pin.
    fn hand_out(&mut self, index: usize) -> FramePin {
        self.images[index].state = PoolState::HandedOut;
        FramePin::Pool {
            index: PoolIndex(index),
            releases: self.releases.clone(),
        }
    }

    fn destroy_image(&mut self, common: &DecoderCommon, i: usize) {
        let p = &self.images[i];
        if p.image == vk::Image::null() {
            return;
        }
        unsafe {
            common.context.device().device_wait_idle().ok();
            common.context.device().destroy_image(p.image, None);
            common.context.device().free_memory(p.memory, None);
        }
    }

    /// Whether any frame handed to the caller is still alive.
    fn has_live_frames(&mut self) -> bool {
        self.reclaim();
        self.images.iter().any(|p| p.state == PoolState::HandedOut)
    }

    /// Free every pool image. The caller must be done with handed-out frames.
    fn destroy(&mut self, common: &DecoderCommon) {
        for i in 0..self.images.len() {
            self.destroy_image(common, i);
            self.images[i].image = vk::Image::null();
        }
        self.images.clear();
    }
}

/// A buffered picture, referencing its pool image by index.
struct ReorderEntry {
    pool_index: usize,
    display_order: i32,
    pts: u64,
    is_keyframe: bool,
    width: u32,
    height: u32,
    coded_width: u32,
    coded_height: u32,
    array_layer: u32,
    pixel_format: PixelFormat,
}

/// Reorders decoded pictures from decode order into display order.
///
/// Decoded pictures are returned in the order the hardware produces them, which
/// for streams with B-frames is not display order. This buffer copies each
/// decoded picture into a pool image, so it survives while later pictures are
/// decoded ahead of it, and emits them in display order.
///
/// Emission follows the DPB bumping model: a picture is held until at most
/// `reorder_depth` pictures precede it in the buffer, a keyframe drains the
/// previous coded video sequence (display order restarts there), and
/// [`Self::flush`] drains the rest at end of stream.
///
/// When disabled (decode-order mode) it is a pass-through: no copy, no latency,
/// and the returned frame points straight at the decoder's DPB image.
pub(crate) struct ReorderBuffer {
    enabled: bool,
    pool: FramePool,
    buffered: Vec<ReorderEntry>,
}

impl ReorderBuffer {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pool: FramePool::new(),
            buffered: Vec::new(),
        }
    }

    /// Add a freshly decoded picture and return whatever is now ready to output.
    ///
    /// `reorder_depth` is how many pictures may precede a held one before it
    /// has to be emitted.
    pub fn push(
        &mut self,
        common: &mut DecoderCommon,
        picture: &DecodedPicture,
        reorder_depth: usize,
    ) -> Result<Vec<DecodedFrame>> {
        if !self.enabled {
            return Ok(vec![self.emit_immediately(common, picture)?]);
        }

        let mut out = Vec::new();
        // Display order restarts at a keyframe, so the previous sequence must be
        // fully drained before this picture, which belongs to the new one.
        if picture.is_keyframe {
            out.extend(self.drain_all());
        }

        let pool_index = self.pool.acquire(common, picture)?;
        common.record_picture_copy(picture, self.pool.images[pool_index].image)?;
        common.submit_copy()?;
        self.pool.images[pool_index].state = PoolState::Buffered;
        self.buffered.push(ReorderEntry {
            pool_index,
            display_order: picture.display_order,
            pts: picture.pts,
            is_keyframe: picture.is_keyframe,
            width: picture.width,
            height: picture.height,
            coded_width: picture.coded_width,
            coded_height: picture.coded_height,
            array_layer: picture.array_layer,
            pixel_format: picture.pixel_format,
        });

        while self.buffered.len() > reorder_depth {
            out.push(self.pop_min_display_order());
        }
        Ok(out)
    }

    /// Hand a picture straight to the caller (decode-order mode).
    ///
    /// Zero-copy when the driver decoded into the DPB image and the session
    /// reserved a spare slot for it: the slot is pinned, so the codec will not
    /// decode over it until the frame is dropped. Otherwise the picture lives
    /// somewhere that the next decode overwrites (the session's single output
    /// image, or a DPB slot the stream needs back), so it is copied into a pool
    /// image instead.
    fn emit_immediately(
        &mut self,
        common: &mut DecoderCommon,
        picture: &DecodedPicture,
    ) -> Result<DecodedFrame> {
        let session = common.session()?;
        if session.coincide && session.output_slots > 0 {
            common.slot_pins.pin(picture.slot);
            return Ok(DecodedFrame {
                image: picture.image,
                image_view: picture.image_view,
                layout: picture.layout,
                array_layer: picture.array_layer,
                pixel_format: picture.pixel_format,
                width: picture.width,
                height: picture.height,
                coded_width: picture.coded_width,
                coded_height: picture.coded_height,
                pts: picture.pts,
                display_order: picture.display_order,
                is_keyframe: picture.is_keyframe,
                pin: Some(FramePin::DpbSlot {
                    slot: picture.slot,
                    pins: common.slot_pins.clone(),
                }),
            });
        }
        self.emit_copy(common, picture)
    }

    /// Copy a picture into a pool image and hand out the copy.
    fn emit_copy(
        &mut self,
        common: &mut DecoderCommon,
        picture: &DecodedPicture,
    ) -> Result<DecodedFrame> {
        let index = self.pool.acquire(common, picture)?;
        common.record_picture_copy(picture, self.pool.images[index].image)?;
        common.submit_copy()?;
        let pin = self.pool.hand_out(index);
        Ok(DecodedFrame {
            image: self.pool.images[index].image,
            // Pool images carry no view (see PoolImage); a caller needing one
            // for GPU work creates it over `image` with the usage it wants.
            image_view: vk::ImageView::null(),
            // copy_picture_to_image leaves the pool image in TRANSFER_DST layout.
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            array_layer: 0,
            pixel_format: picture.pixel_format,
            width: picture.width,
            height: picture.height,
            coded_width: picture.coded_width,
            coded_height: picture.coded_height,
            pts: picture.pts,
            display_order: picture.display_order,
            is_keyframe: picture.is_keyframe,
            pin: Some(pin),
        })
    }

    /// Emit every buffered picture in display order. Call at end of stream.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        if !self.enabled {
            return Vec::new();
        }
        self.drain_all()
    }

    /// Drain the whole buffer in ascending display order.
    fn drain_all(&mut self) -> Vec<DecodedFrame> {
        let mut out = Vec::with_capacity(self.buffered.len());
        while !self.buffered.is_empty() {
            out.push(self.pop_min_display_order());
        }
        out
    }

    /// Remove and return the buffered picture that comes first in display order.
    fn pop_min_display_order(&mut self) -> DecodedFrame {
        let i = self
            .buffered
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.display_order)
            .map(|(i, _)| i)
            .expect("buffer is non-empty");
        let entry = self.buffered.remove(i);
        let pin = self.pool.hand_out(entry.pool_index);
        DecodedFrame {
            image: self.pool.images[entry.pool_index].image,
            // Pool images carry no view (see PoolImage); a caller needing one
            // for GPU work creates it over `image` with the usage it wants.
            image_view: vk::ImageView::null(),
            // copy_picture_to_image leaves the pool image in TRANSFER_DST layout.
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            array_layer: entry.array_layer,
            pixel_format: entry.pixel_format,
            width: entry.width,
            height: entry.height,
            coded_width: entry.coded_width,
            coded_height: entry.coded_height,
            pts: entry.pts,
            display_order: entry.display_order,
            is_keyframe: entry.is_keyframe,
            pin: Some(pin),
        }
    }

    /// Free every pool image.
    ///
    /// Frames the caller still holds point at these images, so this warns rather
    /// than silently leaving dangling handles behind.
    pub fn destroy(&mut self, common: &DecoderCommon) {
        if self.pool.has_live_frames() {
            tracing::warn!(
                "decoder dropped while decoded frames are still alive; their images \
                 are now invalid. Drop every DecodedFrame before the Decoder."
            );
        }
        self.pool.destroy(common);
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
