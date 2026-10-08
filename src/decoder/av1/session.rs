//! The AV1 Vulkan decode session: lazy sequence-driven setup, OBU walking and
//! per-frame record/submit.

use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use crate::decoder::av1::dpb::{DecodeDpb, SlotMeta};
use crate::decoder::av1::frame_header::{
    FrameHeader, HeaderError, HeaderParser, KEY_FRAME, TileGroup,
};
use crate::decoder::av1::params::{self, PictureStd};
use crate::decoder::av1::sequence::{
    Obu, SequenceHeader, obu_type, parse_sequence_header, split_obus,
};
use crate::decoder::common::{DecodeCaps, DecoderCommon, StreamNeeds, query_decode_caps};
use crate::decoder::frames::{DecodedPicture, ReorderBuffer};
use crate::decoder::{DecodeConfig, DecodeStatus, Framing};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// Reference slots AV1 names, and so the pictures it may keep at once.
const NUM_REF_FRAMES: u32 = 8;
/// References one frame predicts from.
const REFS_PER_FRAME: u32 = 7;

/// AV1 decoder.
///
/// Owns what is specific to AV1: the header parser with its reference state,
/// and the mapping of reference slots to DPB slots. Everything Vulkan lives in
/// [`DecoderCommon`].
pub(crate) struct Av1Decoder {
    common: DecoderCommon,
    output_depth: usize,
    parser: HeaderParser,
    /// The sequence the session was built for.
    active: Option<Arc<SequenceHeader>>,
    dpb: Option<DecodeDpb>,
    reorder: ReorderBuffer,
    framing: Framing,
    carry: Vec<u8>,
    carry_pts: u64,
    /// Decoding waits for a shown key frame: nothing else can be decoded
    /// without the references one sets up.
    awaiting_keyframe: bool,
    /// The next frame's `display_order`, restarting at each key frame.
    display_counter: i32,
}

/// A frame whose header has been parsed and whose tiles are being collected.
struct Pending<'a> {
    header: FrameHeader,
    tiles: Vec<&'a [u8]>,
}

impl Av1Decoder {
    pub(crate) fn create(context: VideoContext, config: &DecodeConfig) -> Result<Self> {
        Ok(Self {
            common: DecoderCommon::new(context, config.consumer_queue_family)?,
            output_depth: config.output_depth,
            parser: HeaderParser::new(),
            active: None,
            dpb: None,
            reorder: ReorderBuffer::new(),
            framing: config.framing,
            carry: Vec::new(),
            carry_pts: 0,
            awaiting_keyframe: true,
            display_counter: 0,
        })
    }

    pub(crate) fn finish(&mut self) -> Result<()> {
        if !self.carry.is_empty() {
            let tail = std::mem::take(&mut self.carry);
            let pts = self.carry_pts;
            self.decode_frames(&tail, pts)?;
        }
        let frames = self.reorder.flush();
        self.common.finish_stream(frames);
        Ok(())
    }

    pub(crate) fn take_frame_receiver(&mut self) -> Option<crate::decoder::FrameReceiver> {
        self.common.frames_rx.take()
    }

    pub(crate) fn picture_format(&self) -> Option<vk::Format> {
        self.common.session.as_ref().map(|a| a.picture_format)
    }

    /// Create the Vulkan session for `seq`, unless it is the one already
    /// there. AV1's session parameters are the sequence header and nothing
    /// else, so a new sequence is a new session.
    fn activate(&mut self, seq: &Arc<SequenceHeader>) -> Result<()> {
        if self.common.session.is_some() && self.active.as_ref() == Some(seq) {
            return Ok(());
        }
        let mut profile = params::decode_profile(seq)?;
        let (pixel_format, bit_depth) = (profile.pixel_format, profile.bit_depth);
        let profile_info = profile.info();

        let mut av1_caps = vk::VideoDecodeAV1CapabilitiesKHR::default();
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let caps = {
            let mut caps = vk::VideoCapabilitiesKHR::default()
                .push(&mut av1_caps)
                .push(&mut decode_caps);
            query_decode_caps(&self.common.context, &profile_info, &mut caps)?;
            DecodeCaps::new(&caps)
        }
        .with_decode_flags(decode_caps.flags);

        // A frame can be shown again after later ones have been decoded, which
        // needs it to still be readable in its DPB slot.
        if !caps
            .decode_flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE)
        {
            return Err(PixelForgeError::CodecNotSupported(
                "AV1 decode needs a device that decodes into the DPB image".to_string(),
            ));
        }

        let plan = self.common.plan_session(
            &caps,
            &StreamNeeds {
                codec: "AV1",
                profile_info: &profile_info,
                pixel_format,
                bit_depth,
                width: seq.max_frame_width_minus_1 as u32 + 1,
                height: seq.max_frame_height_minus_1 as u32 + 1,
                references: NUM_REF_FRAMES,
                active_references: REFS_PER_FRAME,
                slot_cap: NUM_REF_FRAMES + 1 + 16,
                // Frames come out in the order they are decoded and shown.
                reorder_depth: 0,
                output_depth: self.output_depth,
            },
        )?;
        let slot_count = plan.slot_count;
        let spare_slots = plan.spare_slots;
        let (coded_width, coded_height, picture_format) =
            (plan.coded_width, plan.coded_height, plan.picture_format);
        let (use_layered_dpb, pinnable, sampleable, plane_views) = (
            plan.use_layered_dpb,
            plan.pinnable,
            plan.sampleable,
            plan.plane_views,
        );
        self.common.create_session(
            &plan,
            &profile_info,
            &caps.std_header_version,
            |common, session| params::build_session_params(common, session, seq),
        )?;

        self.active = Some(seq.clone());
        let pins = self.common.session()?.slot_pins.clone();
        self.dpb = Some(DecodeDpb::new(slot_count, pins));
        self.display_counter = 0;

        debug!(
            "AV1 decode session: {}x{} {:?}, {} DPB slots ({} spare), \
             layered={}, pinnable={}, sampleable={}, plane_views={}, \
             bitstream alignment {}/{} (offset/size)",
            coded_width,
            coded_height,
            picture_format,
            slot_count,
            spare_slots,
            use_layered_dpb,
            pinnable,
            sampleable,
            plane_views,
            self.common.bitstream_offset_alignment,
            self.common.bitstream_size_alignment
        );
        Ok(())
    }

    /// A sequence header went past. One that differs from the last starts a new
    /// coded video sequence, which has to begin with a key frame.
    fn on_sequence_header(&mut self, payload: &[u8]) -> Result<()> {
        let seq = parse_sequence_header(payload)?;
        if self.parser.seq.as_deref() == Some(&seq) {
            return Ok(());
        }
        self.parser.seq = Some(Arc::new(seq));
        self.parser.reset_references();
        self.awaiting_keyframe = true;
        Ok(())
    }

    /// Decode a chunk of input, doing the framing the configuration calls for.
    pub(crate) fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        match self.framing {
            Framing::FrameAligned => self.decode_frames(data, pts),
            Framing::ByteStream => self.decode_byte_stream(data, pts),
        }
    }

    /// Framing for input that may cut anywhere: everything before the last
    /// temporal delimiter is whole temporal units.
    fn decode_byte_stream(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        let carried = self.carry.len();
        let mut buffer = std::mem::take(&mut self.carry);
        buffer.extend_from_slice(data);

        let split = super::complete_prefix(&buffer);
        let result = if split > 0 {
            self.decode_frames(&buffer[..split], self.carry_pts)
        } else {
            Ok(DecodeStatus::Buffered)
        };
        if split >= carried {
            self.carry_pts = pts;
        }
        buffer.drain(..split);
        self.carry = buffer;
        result
    }

    /// Whether `obu` belongs to a layer outside the operating point decoded.
    fn dropped(&self, obu: &Obu) -> bool {
        let Some(seq) = &self.parser.seq else {
            return false;
        };
        let idc = seq.operating_point_idc();
        if idc == 0
            || !obu.has_extension
            || obu.obu_type == obu_type::SEQUENCE_HEADER
            || obu.obu_type == obu_type::TEMPORAL_DELIMITER
        {
            return false;
        }
        let in_temporal = (idc >> obu.temporal_id) & 1 != 0;
        let in_spatial = (idc >> (obu.spatial_id + 8)) & 1 != 0;
        !(in_temporal && in_spatial)
    }

    /// Decode input already known to end on a frame boundary.
    fn decode_frames(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        let obus = split_obus(data)?;
        let mut decoded_any = false;
        let mut awaiting: Option<String> = None;
        let mut pending: Option<Pending> = None;

        for obu in obus {
            if self.dropped(&obu) {
                continue;
            }
            match obu.obu_type {
                obu_type::TEMPORAL_DELIMITER => pending = None,
                obu_type::SEQUENCE_HEADER => self.on_sequence_header(obu.payload)?,
                obu_type::FRAME_HEADER | obu_type::FRAME => {
                    // A frame header OBU repeated inside a frame carries
                    // nothing new.
                    if pending.is_some() && obu.obu_type == obu_type::FRAME_HEADER {
                        continue;
                    }
                    pending = None;
                    if self.awaiting_keyframe
                        && !self.parser.starts_with_shown_key_frame(obu.payload)
                    {
                        awaiting.get_or_insert_with(|| {
                            "no key frame decoded yet; this frame cannot be decoded".to_string()
                        });
                        continue;
                    }
                    let header = match self.parser.parse(&obu, obu.payload) {
                        Ok(header) => header,
                        Err(HeaderError::MissingState(reason)) => {
                            self.awaiting_keyframe = true;
                            awaiting.get_or_insert(reason);
                            continue;
                        }
                        Err(HeaderError::Invalid(e)) => return Err(e),
                    };
                    if header.show_existing_frame {
                        if self.show_existing(&header, pts)? {
                            decoded_any = true;
                        } else {
                            awaiting.get_or_insert_with(|| {
                                "a frame to show again is not held".to_string()
                            });
                        }
                        continue;
                    }
                    let mut frame = Pending {
                        header,
                        tiles: Vec::new(),
                    };
                    if obu.obu_type == obu_type::FRAME {
                        let rest = &obu.payload[frame.header.header_bytes.min(obu.payload.len())..];
                        if self.collect_tiles(&mut frame, rest)? {
                            self.finish_frame(frame, pts, &mut decoded_any, &mut awaiting)?;
                            continue;
                        }
                    }
                    pending = Some(frame);
                }
                obu_type::TILE_GROUP => {
                    let Some(mut frame) = pending.take() else {
                        // The frame header never arrived.
                        continue;
                    };
                    if self.collect_tiles(&mut frame, obu.payload)? {
                        self.finish_frame(frame, pts, &mut decoded_any, &mut awaiting)?;
                    } else {
                        pending = Some(frame);
                    }
                }
                _ => {}
            }
        }

        if decoded_any {
            return Ok(DecodeStatus::Decoded);
        }
        if let Some(reason) = awaiting {
            debug!("AV1 decode: awaiting keyframe: {}", reason);
            return Ok(DecodeStatus::NeedsKeyframe);
        }
        Ok(DecodeStatus::Buffered)
    }

    /// Add a tile group's tiles to `frame`. Returns whether that completes it.
    fn collect_tiles<'a>(&self, frame: &mut Pending<'a>, payload: &'a [u8]) -> Result<bool> {
        let group: TileGroup = self.parser.parse_tile_group(&frame.header, payload)?;
        for tile in &group.tiles {
            frame
                .tiles
                .push(&payload[tile.offset..tile.offset + tile.size]);
        }
        let tile_info = &frame.header.tile_info;
        Ok(group.tg_end + 1 == tile_info.tile_rows * tile_info.tile_cols)
    }

    fn finish_frame(
        &mut self,
        frame: Pending,
        pts: u64,
        decoded_any: &mut bool,
        awaiting: &mut Option<String>,
    ) -> Result<()> {
        match self.decode_frame(&frame.header, &frame.tiles, pts)? {
            None => *decoded_any = true,
            Some(reason) => {
                self.awaiting_keyframe = true;
                awaiting.get_or_insert(reason);
            }
        }
        Ok(())
    }

    /// Output the frame a `show_existing_frame` header names. False if the
    /// slot holds nothing, which is what a lost frame looks like.
    fn show_existing(&mut self, header: &FrameHeader, pts: u64) -> Result<bool> {
        let Some(dpb) = self.dpb.as_mut() else {
            return Ok(false);
        };
        let Some(slot) = dpb.slot_of(header.frame_to_show_map_idx as usize) else {
            return Ok(false);
        };
        let Some(meta) = dpb.meta(slot).copied() else {
            return Ok(false);
        };
        tracing::trace!(
            "AV1 show existing: slot {} (DPB {}), order_hint {}",
            header.frame_to_show_map_idx,
            slot,
            meta.order_hint
        );
        let is_key = header.frame_type == KEY_FRAME;
        if is_key {
            // Showing a key frame refreshes every reference with it.
            dpb.refresh(slot, header.refresh_frame_flags);
            self.display_counter = 0;
        }
        self.common.pipeline.wait_current_free();
        let picture = self.picture_descriptor(slot, &meta, pts, is_key);
        let ready = self.reorder.push(&mut self.common, &picture, 0)?;
        self.common.end_picture(ready);
        Ok(true)
    }

    /// Describe the picture in `slot` for the reorder buffer, which hands it to
    /// the caller.
    fn picture_descriptor(
        &mut self,
        slot: u8,
        meta: &SlotMeta,
        pts: u64,
        is_key: bool,
    ) -> DecodedPicture {
        let generation = self.common.generation;
        let layout = self.common.picture_layout();
        let session = self.common.session.as_ref().expect("active");
        let (image, array_layer) = session.dpb_image_for_slot(slot);
        let display_order = self.display_counter;
        self.display_counter += 1;
        DecodedPicture {
            slot,
            image,
            image_view: session.dpb_entry(slot).2,
            layout,
            array_layer,
            pixel_format: session.pixel_format,
            bit_depth: session.bit_depth,
            generation,
            width: meta.width,
            height: meta.height,
            coded_width: session.coded_width,
            coded_height: session.coded_height,
            pts,
            display_order,
            is_keyframe: is_key,
            starts_sequence: is_key,
        }
    }

    /// Record and submit the decode of one frame. Returns why it could not be
    /// decoded, if it could not.
    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        tiles: &[&[u8]],
        pts: u64,
    ) -> Result<Option<String>> {
        let seq =
            self.parser.seq.clone().ok_or_else(|| {
                PixelForgeError::InvalidInput("AV1 decode: no sequence".to_string())
            })?;
        self.activate(&seq)?;

        // The DPB slots the frame predicts from, by reference name.
        let mut name_slots = [-1i32; 7];
        let mut ref_slots: Vec<u8> = Vec::new();
        if !header.frame_is_intra() {
            let dpb = self.dpb.as_ref().expect("activated above");
            for (i, name_slot) in name_slots.iter_mut().enumerate() {
                let idx = header.ref_frame_idx[i] as usize;
                let Some(slot) = dpb.slot_of(idx) else {
                    return Ok(Some(format!("reference slot {idx} holds no picture")));
                };
                *name_slot = slot as i32;
                if !ref_slots.contains(&slot) {
                    ref_slots.push(slot);
                }
            }
        }

        let slot = loop {
            let dpb = self.dpb.as_ref().expect("active");
            if let Some(slot) = dpb.try_allocate_slot() {
                break slot;
            }
            if !dpb.has_pinned_slots() {
                return Err(PixelForgeError::InvalidInput(
                    "AV1 decode: no free DPB slot (stream exceeds negotiated DPB size)".to_string(),
                ));
            }
            self.common.session()?.slot_pins.wait_for_release();
        };

        tracing::trace!(
            "AV1 frame: type={} show={} order_hint={} refresh={:#04x} refs={:?} \
             primary_ref={} mvs={} skip_mode={} ref_select={} tiles={} q={}",
            header.frame_type,
            header.show_frame,
            header.order_hint,
            header.refresh_frame_flags,
            header.ref_frame_idx,
            header.primary_ref_frame,
            header.use_ref_frame_mvs,
            header.skip_mode_present,
            header.reference_select,
            tiles.len(),
            header.quant.base_q_idx,
        );
        self.common.begin_decode_commands()?;
        let mut profile = params::decode_profile(&seq)?;
        let (buffer_range, tile_offsets) = self.common.stage_chunks(tiles, &profile.info())?;
        let tile_sizes: Vec<u32> = tiles.iter().map(|t| t.len() as u32).collect();
        self.common.record_barriers(slot);

        let mut ref_metas = Vec::with_capacity(ref_slots.len());
        for &s in &ref_slots {
            let meta = *self
                .dpb
                .as_ref()
                .and_then(|d| d.meta(s))
                .expect("a referenced slot has metadata");
            ref_metas.push(meta);
        }
        let mut order_hints = [0u8; 8];
        for (dst, &src) in order_hints.iter_mut().zip(&header.order_hints) {
            *dst = src as u8;
        }
        let mut sign_bias = 0u8;
        for (i, &bias) in header.ref_frame_sign_bias.iter().enumerate() {
            sign_bias |= (bias as u8) << i;
        }
        let new_meta = SlotMeta {
            frame_type: header.frame_type,
            order_hint: header.order_hint as u8,
            saved_order_hints: order_hints,
            ref_frame_sign_bias: sign_bias,
            disable_frame_end_update_cdf: header.disable_frame_end_update_cdf,
            segmentation_enabled: header.segmentation.enabled,
            width: header.upscaled_width,
            height: header.frame_height,
        };
        self.record_decode(
            header,
            slot,
            &ref_slots,
            &ref_metas,
            &new_meta,
            &name_slots,
            buffer_range,
            &tile_offsets,
            &tile_sizes,
        )?;
        self.common.submit_decode()?;

        let active = self.common.session.as_mut().expect("active");
        active.dpb_slot_active[slot as usize] = true;

        self.dpb
            .as_mut()
            .expect("active")
            .store(slot, new_meta, header.refresh_frame_flags);

        let is_key = header.frame_type == KEY_FRAME;
        let ready = if header.show_frame {
            if is_key {
                self.display_counter = 0;
            }
            let picture = self.picture_descriptor(slot, &new_meta, pts, is_key);
            self.reorder.push(&mut self.common, &picture, 0)?
        } else {
            Vec::new()
        };
        if is_key && header.show_frame {
            self.awaiting_keyframe = false;
        }
        self.common.end_picture(ready);
        Ok(None)
    }

    /// Record `vkCmdBeginVideoCoding` / `vkCmdDecodeVideo` / `vkCmdEndVideoCoding`.
    #[allow(clippy::too_many_arguments)]
    fn record_decode(
        &self,
        header: &FrameHeader,
        slot: u8,
        ref_slots: &[u8],
        ref_metas: &[SlotMeta],
        new_meta: &SlotMeta,
        name_slots: &[i32; 7],
        buffer_range: u64,
        tile_offsets: &[u32],
        tile_sizes: &[u32],
    ) -> Result<()> {
        let active = self.common.session.as_ref().expect("active");
        // The frame's own size, not rounded up to anything: AV1 takes its frame
        // geometry from the extent, and a driver that is given a larger one
        // decodes the picture edge as though the frame were bigger.
        let extent_of = |meta: &SlotMeta| vk::Extent2D {
            width: meta.width,
            height: meta.height,
        };

        let ref_std_infos: Vec<_> = ref_metas.iter().map(params::reference_info).collect();
        let setup_std_info = params::reference_info(new_meta);

        let resource = |slot: u8, extent: vk::Extent2D| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(extent)
                .base_array_layer(0)
                .image_view_binding(active.dpb_entry(slot).2)
        };
        let ref_resources: Vec<_> = ref_slots
            .iter()
            .zip(ref_metas)
            .map(|(&s, meta)| resource(s, extent_of(meta)))
            .collect();
        let setup_resource = resource(slot, extent_of(new_meta));

        let mut begin_infos: Vec<_> = ref_std_infos
            .iter()
            .map(|i| vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(i))
            .collect();
        let mut setup_info =
            vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(&setup_std_info);

        // As for the other codecs: the slot reconstructed into is bound with
        // index -1 until the decode itself activates it.
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for ((&s, resource), info) in ref_slots
            .iter()
            .zip(ref_resources.iter())
            .zip(begin_infos.iter_mut())
        {
            begin_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(s as i32)
                    .picture_resource(resource)
                    .push(info),
            );
        }
        begin_slots.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource),
        );
        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(active.session)
            .video_session_parameters(active.session_params)
            .reference_slots(&begin_slots);
        unsafe {
            self.common
                .video_queue_fn
                .cmd_begin_video_coding(self.common.decode_command_buffer(), &begin_info);
        }

        if !active.dpb_slot_active.iter().any(|&a| a) {
            let control = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            unsafe {
                self.common
                    .video_queue_fn
                    .cmd_control_video_coding(self.common.decode_command_buffer(), &control);
            }
        }

        let std = PictureStd::new(header);
        let mut av1_picture_info = vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(&std.info)
            .reference_name_slot_indices(*name_slots)
            .frame_header_offset(0)
            .tile_offsets(tile_offsets)
            .tile_sizes(tile_sizes);

        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(slot as i32)
            .picture_resource(&setup_resource)
            .push(&mut setup_info);

        let mut decode_infos: Vec<_> = ref_std_infos
            .iter()
            .map(|i| vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(i))
            .collect();
        let mut decode_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for ((&s, resource), info) in ref_slots
            .iter()
            .zip(ref_resources.iter())
            .zip(decode_infos.iter_mut())
        {
            decode_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(s as i32)
                    .picture_resource(resource)
                    .push(info),
            );
        }

        // Coincident DPB and output is required (see `activate`), so the frame
        // is reconstructed straight into its slot.
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.common.bitstream_buffer())
            .src_buffer_offset(0)
            .src_buffer_range(buffer_range)
            .dst_picture_resource(setup_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&decode_slots)
            .push(&mut av1_picture_info);

        unsafe {
            self.common
                .video_decode_fn
                .cmd_decode_video(self.common.decode_command_buffer(), &decode_info);
            self.common.video_queue_fn.cmd_end_video_coding(
                self.common.decode_command_buffer(),
                &vk::VideoEndCodingInfoKHR::default(),
            );
        }
        Ok(())
    }
}

impl Drop for Av1Decoder {
    fn drop(&mut self) {
        self.reorder.destroy(&self.common);
    }
}
