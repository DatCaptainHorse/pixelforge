//! The H.265 Vulkan decode session: lazy stream-driven setup, per-picture
//! record/submit.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use crate::decoder::common::{DecodeCaps, DecoderCommon, StreamNeeds, query_decode_caps};
use crate::decoder::frames::{DecodedPicture, ReorderBuffer};
use crate::decoder::h265::dpb::{Begin, DecodeDpb, MAX_DPB_SLOTS, PictureState};
use crate::decoder::h265::params::{self, ParameterSets};
use crate::decoder::h265::parser::{
    self, NalUnit, Pps, SliceHeader, Sps, Vps, iter_nal_units, nal,
};
use crate::decoder::{DecodeConfig, DecodeStatus, Framing};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// H.265 decoder.
///
/// Owns only what is specific to H.265: the parameter sets seen so far, and the
/// POC / reference picture state. Everything Vulkan lives in [`DecoderCommon`].
pub(crate) struct H265Decoder {
    common: DecoderCommon,
    output_depth: usize,

    vps_map: HashMap<u8, Arc<Vps>>,
    sps_map: HashMap<u8, Arc<Sps>>,
    pps_map: HashMap<u8, Arc<Pps>>,

    /// The parameter sets the current session and its parameters were built
    /// from, kept whole so that a set reusing an id for different content is
    /// seen as the change it is.
    active: Option<ParameterSets>,

    dpb: Option<DecodeDpb>,
    reorder: ReorderBuffer,
    reorder_depth: usize,

    framing: Framing,
    /// Bytes of a picture not yet seen to end. Only in [`Framing::ByteStream`].
    carry: Vec<u8>,
    carry_pts: u64,

    /// Decoding waits for an IRAP picture: anything before one references
    /// state that does not exist. True until the first is decoded, and again
    /// after a reference goes missing.
    awaiting_keyframe: bool,
    /// The session was created with the plainer profile; see `activate`.
    relaxed_profile: bool,
}

/// What `split_units` found in the input, in stream order.
enum Unit<'a> {
    Picture(Box<Picture<'a>>),
    /// An end of sequence NAL unit: the next IRAP starts afresh.
    EndOfSequence,
}

/// The slice segments making up one picture.
struct Picture<'a> {
    header: SliceHeader,
    nal_type: u8,
    temporal_id: u8,
    slices: Vec<&'a [u8]>,
    /// The parameter sets in effect when the picture was grouped; see the
    /// equivalent field in the H.264 decoder for why they are captured here.
    sets: ParameterSets,
}

impl H265Decoder {
    pub(crate) fn create(context: VideoContext, config: &DecodeConfig) -> Result<Self> {
        Ok(Self {
            common: DecoderCommon::new(context, config.consumer_queue_family)?,
            output_depth: config.output_depth,
            vps_map: HashMap::new(),
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            active: None,
            dpb: None,
            reorder: ReorderBuffer::new(),
            reorder_depth: 0,
            framing: config.framing,
            carry: Vec::new(),
            carry_pts: 0,
            awaiting_keyframe: true,
            relaxed_profile: false,
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

    /// Whether the device can decode `sps`'s profile as chosen.
    fn profile_supported(&self, sps: &Sps, relaxed: bool) -> bool {
        let Ok(mut profile) = params::decode_profile(sps, relaxed) else {
            return false;
        };
        let profile_info = profile.info();
        let mut h265_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push(&mut h265_caps)
            .push(&mut decode_caps);
        query_decode_caps(&self.common.context, &profile_info, &mut caps).is_ok()
    }

    /// Create the Vulkan session for the given parameter sets, or bring an
    /// existing one up to date with them.
    ///
    /// A new SPS or VPS means a new session. A new PPS alone does not: it only
    /// changes how pictures are coded, so the session, and with it the DPB
    /// images, stays and just its parameters are replaced.
    fn activate(&mut self, sets: &ParameterSets) -> Result<()> {
        let session_compatible = self.common.session.is_some()
            && self
                .active
                .as_ref()
                .is_some_and(|a| a.sps == sets.sps && a.vps == sets.vps);
        if session_compatible {
            if self.active.as_ref() == Some(sets) {
                return Ok(());
            }
            self.common
                .replace_session_params(|common, session| params::build(common, session, sets))?;
            self.active = Some(sets.clone());
            return Ok(());
        }

        let sps = &sets.sps;
        // Use the profile the stream names, unless the device does not list it
        // and a plainer one decodes the same stream.
        self.relaxed_profile =
            params::can_relax_profile(sps) && !self.profile_supported(sps, false);
        let mut profile = params::decode_profile(sps, self.relaxed_profile)?;
        let (pixel_format, bit_depth) = (profile.pixel_format, profile.bit_depth);
        let profile_info = profile.info();

        let mut h265_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let caps = {
            let mut caps = vk::VideoCapabilitiesKHR::default()
                .push(&mut h265_caps)
                .push(&mut decode_caps);
            query_decode_caps(&self.common.context, &profile_info, &mut caps)?;
            DecodeCaps::new(&caps)
        }
        .with_decode_flags(decode_caps.flags);

        let (dpb_size, reorder_depth) = sps.highest_dec_pic_buf();
        let plan = self.common.plan_session(
            &caps,
            &StreamNeeds {
                codec: "H.265",
                profile_info: &profile_info,
                pixel_format,
                bit_depth,
                width: sps.pic_width_in_luma_samples,
                height: sps.pic_height_in_luma_samples,
                // The DPB size counts the current picture too.
                references: dpb_size - 1,
                active_references: dpb_size - 1,
                slot_cap: MAX_DPB_SLOTS as u32,
                reorder_depth: reorder_depth as usize,
                output_depth: self.output_depth,
            },
        )?;
        let slot_count = plan.slot_count;
        let spare_slots = plan.spare_slots;
        let (coded_width, coded_height, picture_format) =
            (plan.coded_width, plan.coded_height, plan.picture_format);
        let (use_layered_dpb, coincide, pinnable, sampleable, plane_views) = (
            plan.use_layered_dpb,
            plan.coincide,
            plan.pinnable,
            plan.sampleable,
            plan.plane_views,
        );
        self.common.create_session(
            &plan,
            &profile_info,
            &caps.std_header_version,
            |common, session| params::build(common, session, sets),
        )?;

        self.active = Some(sets.clone());
        self.reorder_depth = reorder_depth as usize;
        let pins = self.common.session()?.slot_pins.clone();
        self.dpb = Some(DecodeDpb::new(slot_count, pins));

        debug!(
            "H.265 decode session: {}x{} {:?}, {} DPB slots ({} spare), \
             layered={}, coincide={}, pinnable={}, sampleable={}, \
             plane_views={}, bitstream alignment {}/{} (offset/size)",
            coded_width,
            coded_height,
            picture_format,
            slot_count,
            spare_slots,
            use_layered_dpb,
            coincide,
            pinnable,
            sampleable,
            plane_views,
            self.common.bitstream_offset_alignment,
            self.common.bitstream_size_alignment
        );
        Ok(())
    }

    /// Group the input into pictures, absorbing parameter sets on the way.
    /// Also returns why a keyframe is needed if slices had to be skipped.
    fn split_units<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<Unit<'a>>, Option<String>)> {
        let mut units: Vec<Unit<'a>> = Vec::new();
        let mut awaiting: Option<String> = None;

        for nal in iter_nal_units(data) {
            match nal.nal_type {
                nal::VPS => {
                    let vps = parser::parse_vps(nal.payload())?;
                    self.vps_map.insert(vps.vps_id, Arc::new(vps));
                }
                nal::SPS => {
                    let sps = parser::parse_sps(nal.payload())?;
                    debug!(
                        "H.265 SPS id={} {}x{} profile={}",
                        sps.sps_id,
                        sps.pic_width_in_luma_samples,
                        sps.pic_height_in_luma_samples,
                        sps.ptl.profile_idc
                    );
                    self.sps_map.insert(sps.sps_id, Arc::new(sps));
                }
                nal::PPS => {
                    let pps = parser::parse_pps(nal.payload())?;
                    self.pps_map.insert(pps.pps_id, Arc::new(pps));
                }
                nal::EOS | nal::EOB => units.push(Unit::EndOfSequence),
                _ if nal.is_decodable_slice() => {
                    if let Some(reason) = self.group_slice(&mut units, &nal)? {
                        awaiting.get_or_insert(reason);
                    }
                }
                _ => {}
            }
        }
        Ok((units, awaiting))
    }

    /// Attach a slice segment to the picture in progress, or begin a picture.
    /// Returns the reason when the slice has to be skipped.
    fn group_slice<'a>(
        &self,
        units: &mut Vec<Unit<'a>>,
        nal: &NalUnit<'a>,
    ) -> Result<Option<String>> {
        let first_slice = nal.payload().first().is_some_and(|b| b & 0x80 != 0);
        if !first_slice {
            // A later segment of the picture in progress. Without one, its
            // first segment never arrived and there is nothing to attach to.
            if let Some(Unit::Picture(picture)) = units.last_mut() {
                picture.slices.push(nal.data);
            }
            return Ok(None);
        }

        let pps_id = peek_pps_id(nal)?;
        let Some(pps) = self.pps_map.get(&pps_id) else {
            return Ok(Some(format!("PPS {pps_id} not yet received")));
        };
        let Some(sps) = self.sps_map.get(&pps.sps_id) else {
            return Ok(Some(format!("SPS {} not yet received", pps.sps_id)));
        };
        let Some(vps) = self.vps_map.get(&sps.vps_id) else {
            return Ok(Some(format!("VPS {} not yet received", sps.vps_id)));
        };
        let header = parser::parse_slice_header(nal, sps, pps)?;
        units.push(Unit::Picture(Box::new(Picture {
            header,
            nal_type: nal.nal_type,
            temporal_id: nal.temporal_id,
            slices: vec![nal.data],
            sets: ParameterSets {
                vps: Arc::clone(vps),
                sps: Arc::clone(sps),
                pps: Arc::clone(pps),
            },
        })));
        Ok(None)
    }

    /// Decode a chunk of input, doing the framing the configuration calls for.
    pub(crate) fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        match self.framing {
            Framing::FrameAligned => self.decode_frames(data, pts),
            Framing::ByteStream => self.decode_byte_stream(data, pts),
        }
    }

    /// Framing for input that may cut anywhere; see the H.264 decoder.
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

    /// Decode input already known to end on a picture boundary.
    fn decode_frames(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        let (units, mut awaiting) = self.split_units(data)?;
        let mut decoded_any = false;

        for unit in units {
            let picture = match unit {
                Unit::EndOfSequence => {
                    if let Some(dpb) = self.dpb.as_mut() {
                        dpb.end_of_sequence();
                    }
                    // What follows an end of sequence has to be an IRAP.
                    self.awaiting_keyframe = true;
                    continue;
                }
                Unit::Picture(picture) => picture,
            };
            let is_irap = parser::is_irap(picture.nal_type);

            // Before the first IRAP picture there is no reference state to
            // predict from. Skip and ask for a keyframe rather than decode
            // garbage.
            if self.awaiting_keyframe && !is_irap {
                awaiting.get_or_insert_with(|| {
                    "no keyframe decoded yet; non-IRAP picture cannot be decoded".to_string()
                });
                continue;
            }

            self.activate(&picture.sets)?;

            match self.decode_picture(&picture, pts)? {
                Decoded::Picture(frame) => {
                    self.awaiting_keyframe = false;
                    decoded_any = true;
                    let ready = match frame {
                        Some(frame) => {
                            let depth = self.reorder_depth;
                            self.reorder.push(&mut self.common, &frame, depth)?
                        }
                        None => Vec::new(),
                    };
                    self.common.end_picture(ready);
                }
                Decoded::Skipped => {}
                Decoded::MissingReference(reason) => {
                    self.awaiting_keyframe = true;
                    awaiting.get_or_insert(reason);
                }
            }
        }

        if decoded_any {
            return Ok(DecodeStatus::Decoded);
        }
        if let Some(reason) = awaiting {
            debug!("H.265 decode: awaiting keyframe: {}", reason);
            return Ok(DecodeStatus::NeedsKeyframe);
        }
        Ok(DecodeStatus::Buffered)
    }

    /// Record and submit the decode of a single picture.
    fn decode_picture(&mut self, picture: &Picture, pts: u64) -> Result<Decoded> {
        let state = {
            let dpb = self.dpb.as_mut().expect("activated above");
            match dpb.begin_picture(
                picture.nal_type,
                picture.temporal_id,
                &picture.header,
                &picture.sets.sps,
            )? {
                Begin::Decode(state) => state,
                Begin::SkipRasl => return Ok(Decoded::Skipped),
                Begin::MissingReference(reason) => return Ok(Decoded::MissingReference(reason)),
            }
        };
        let slot = loop {
            let dpb = self.dpb.as_mut().expect("active");
            if let Some(slot) = dpb.try_allocate_slot() {
                break slot;
            }
            if !dpb.has_pinned_slots() {
                return Err(PixelForgeError::InvalidInput(
                    "H.265 decode: no free DPB slot (stream exceeds negotiated DPB size)"
                        .to_string(),
                ));
            }
            self.common.session()?.slot_pins.wait_for_release();
        };

        self.common.begin_decode_commands()?;
        let mut profile = params::decode_profile(&picture.sets.sps, self.relaxed_profile)?;
        let (buffer_range, slice_offsets) =
            self.common.stage_slices(&picture.slices, &profile.info())?;
        self.common.record_barriers(slot);
        self.record_decode(picture, &state, slot, buffer_range, &slice_offsets)?;
        self.common.submit_decode()?;

        let active = self.common.session.as_mut().expect("active");
        active.dpb_slot_active[slot as usize] = true;

        let picture_layout = self.common.picture_layout();
        let output_layout = self.common.output_layout();
        let generation = self.common.generation;
        let active = self.common.session.as_ref().expect("active");
        let (image, image_view, layout, array_layer) = if active.coincide {
            let (image, layer) = active.dpb_image_for_slot(slot);
            (image, active.dpb_entry(slot).2, picture_layout, layer)
        } else {
            let (image, _, view) = active
                .output_image
                .expect("non-coincide implies output image");
            (image, view, output_layout, 0)
        };

        let (width, height) = picture.sets.sps.display_dimensions();
        let frame = DecodedPicture {
            slot,
            image,
            image_view,
            layout,
            array_layer,
            pixel_format: active.pixel_format,
            bit_depth: active.bit_depth,
            generation,
            width,
            height,
            coded_width: active.coded_width,
            coded_height: active.coded_height,
            pts,
            display_order: state.poc,
            is_keyframe: state.is_irap,
            starts_sequence: state.starts_sequence,
        };
        let output = state.output;

        self.dpb.as_mut().expect("active").end_picture(slot, &state);

        Ok(Decoded::Picture(output.then_some(frame)))
    }

    /// Record `vkCmdBeginVideoCoding` / `vkCmdDecodeVideo` / `vkCmdEndVideoCoding`.
    fn record_decode(
        &self,
        picture: &Picture,
        state: &PictureState,
        slot: u8,
        buffer_range: u64,
        slice_offsets: &[u32],
    ) -> Result<()> {
        let active = self.common.session.as_ref().expect("active");
        let extent = vk::Extent2D {
            width: active.coded_width,
            height: active.coded_height,
        };

        // Every picture still held as a reference goes to the driver, both for
        // the scope and for the decode, including the ones this picture does
        // not predict from. Some drivers (RADV) decode the pictures that follow
        // a mid-stream CRA wrongly when the decode names only what it reads.
        let refs = self.dpb.as_ref().expect("active").references().to_vec();

        let ref_std_infos: Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo> = refs
            .iter()
            .map(|r| {
                let mut flags: ash::vk::native::StdVideoDecodeH265ReferenceInfoFlags =
                    unsafe { std::mem::zeroed() };
                flags.set_used_for_long_term_reference(r.long_term as u32);
                ash::vk::native::StdVideoDecodeH265ReferenceInfo {
                    flags,
                    PicOrderCntVal: r.poc,
                }
            })
            .collect();
        let setup_std_info = ash::vk::native::StdVideoDecodeH265ReferenceInfo {
            flags: unsafe { std::mem::zeroed() },
            PicOrderCntVal: state.poc,
        };

        let dpb_resource = |slot: u8| -> vk::VideoPictureResourceInfoKHR<'_> {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(extent)
                .base_array_layer(0)
                .image_view_binding(active.dpb_entry(slot).2)
        };
        let ref_resources: Vec<vk::VideoPictureResourceInfoKHR> =
            refs.iter().map(|r| dpb_resource(r.slot)).collect();
        let setup_resource = dpb_resource(slot);

        let mut begin_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> = ref_std_infos
            .iter()
            .map(|info| vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(info))
            .collect();
        let mut setup_info =
            vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&setup_std_info);

        // As for H.264: the slot being reconstructed into is bound with index
        // -1 until the decode itself activates it.
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for ((r, resource), info) in refs
            .iter()
            .zip(ref_resources.iter())
            .zip(begin_infos.iter_mut())
        {
            begin_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(r.slot as i32)
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

        // --- Picture info ---
        let mut pic_flags: ash::vk::native::StdVideoDecodeH265PictureInfoFlags =
            unsafe { std::mem::zeroed() };
        pic_flags.set_IrapPicFlag(state.is_irap as u32);
        pic_flags.set_IdrPicFlag(state.is_idr as u32);
        pic_flags.set_IsReference(state.is_reference as u32);
        pic_flags.set_short_term_ref_pic_set_sps_flag(
            picture.header.short_term_ref_pic_set_sps_flag as u32,
        );
        // 0xFF marks an unused entry.
        let to_array = |slots: &[u8]| {
            let mut out = [0xFFu8; 8];
            out[..slots.len()].copy_from_slice(slots);
            out
        };
        let std_pic_info = ash::vk::native::StdVideoDecodeH265PictureInfo {
            flags: pic_flags,
            sps_video_parameter_set_id: picture.sets.sps.vps_id,
            pps_seq_parameter_set_id: picture.sets.pps.sps_id,
            pps_pic_parameter_set_id: picture.sets.pps.pps_id,
            NumDeltaPocsOfRefRpsIdx: picture.header.short_term_rps.num_delta_pocs_of_ref_rps,
            PicOrderCntVal: state.poc,
            NumBitsForSTRefPicSetInSlice: picture.header.short_term_rps_bits as u16,
            reserved: 0,
            RefPicSetStCurrBefore: to_array(&state.st_curr_before),
            RefPicSetStCurrAfter: to_array(&state.st_curr_after),
            RefPicSetLtCurr: to_array(&state.lt_curr),
        };
        let mut h265_picture_info = vk::VideoDecodeH265PictureInfoKHR::default()
            .std_picture_info(&std_pic_info)
            .slice_segment_offsets(slice_offsets);

        let output_resource = match active.output_image {
            None => setup_resource,
            Some((_, _, view)) => vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(extent)
                .base_array_layer(0)
                .image_view_binding(view),
        };
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(slot as i32)
            .picture_resource(&setup_resource)
            .push(&mut setup_info);

        let decode_refs: Vec<usize> = refs.iter().enumerate().map(|(i, _)| i).collect();
        let mut decode_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> = decode_refs
            .iter()
            .map(|&i| {
                vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&ref_std_infos[i])
            })
            .collect();
        let mut decode_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for (&i, info) in decode_refs.iter().zip(decode_infos.iter_mut()) {
            decode_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(refs[i].slot as i32)
                    .picture_resource(&ref_resources[i])
                    .push(info),
            );
        }

        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.common.bitstream_buffer())
            .src_buffer_offset(0)
            .src_buffer_range(buffer_range)
            .dst_picture_resource(output_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&decode_slots)
            .push(&mut h265_picture_info);

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

/// What became of a picture handed to `decode_picture`.
enum Decoded {
    /// Submitted to the GPU. Holds the descriptor to output, or `None` for a
    /// picture that is decoded as a reference but not shown.
    Picture(Option<DecodedPicture>),
    /// Not decoded, by the rules of the standard (a RASL picture after a
    /// random access point).
    Skipped,
    /// The picture needs a reference that is not there.
    MissingReference(String),
}

impl Drop for H265Decoder {
    fn drop(&mut self) {
        self.reorder.destroy(&self.common);
    }
}

/// The PPS id a slice segment names, which is what finds its other parameter
/// sets before the header can be parsed in full.
fn peek_pps_id(nal: &NalUnit) -> Result<u8> {
    let mut r = crate::decoder::bitreader::BitReader::new(nal.payload());
    r.skip(1)?; // first_slice_segment_in_pic_flag
    if nal.is_irap() {
        r.skip(1)?; // no_output_of_prior_pics_flag
    }
    Ok(r.ue()?.min(255) as u8)
}
