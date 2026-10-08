//! Translating parsed AV1 headers into the `StdVideo*` structs Vulkan takes,
//! and the decode profile a sequence implies.

use ash::vk;
use ash::vk::TaggedStructure;
use ash::vk::native::*;

use crate::decoder::av1::dpb::SlotMeta;
use crate::decoder::av1::frame_header::FrameHeader;
use crate::decoder::av1::sequence::SequenceHeader;
use crate::decoder::common::DecoderCommon;
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};

fn unsupported(what: String) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("AV1 decode: {what}"))
}

/// The decode profile a sequence asks for.
pub(crate) struct DecodeProfile {
    av1: vk::VideoDecodeAV1ProfileInfoKHR<'static>,
    chroma: vk::VideoChromaSubsamplingFlagsKHR,
    depth: vk::VideoComponentBitDepthFlagsKHR,
    pub pixel_format: PixelFormat,
    pub bit_depth: BitDepth,
}

impl DecodeProfile {
    /// The profile as Vulkan takes it. Borrows `self`, which owns the codec
    /// part the struct chains to.
    pub fn info(&mut self) -> vk::VideoProfileInfoKHR<'_> {
        vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
            .chroma_subsampling(self.chroma)
            .luma_bit_depth(self.depth)
            .chroma_bit_depth(self.depth)
            .push(&mut self.av1)
    }
}

pub(crate) fn decode_profile(seq: &SequenceHeader) -> Result<DecodeProfile> {
    let c = &seq.color;
    if c.mono_chrome {
        return Err(unsupported(
            "monochrome streams are not supported".to_string(),
        ));
    }
    let (pixel_format, chroma) = match (c.subsampling_x, c.subsampling_y) {
        (true, true) => (
            PixelFormat::Yuv420,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        ),
        (true, false) => (
            PixelFormat::Yuv422,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
        ),
        (false, false) => (
            PixelFormat::Yuv444,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        ),
        _ => return Err(unsupported("unsupported chroma subsampling".to_string())),
    };
    let (bit_depth, depth) = match c.bit_depth {
        8 => (BitDepth::Eight, vk::VideoComponentBitDepthFlagsKHR::TYPE_8),
        10 => (BitDepth::Ten, vk::VideoComponentBitDepthFlagsKHR::TYPE_10),
        other => return Err(unsupported(format!("unsupported bit depth {other}"))),
    };
    let spatial_layers = (seq.operating_point_idc() >> 8).count_ones();
    if spatial_layers > 1 {
        return Err(unsupported(
            "spatial scalability is not supported".to_string(),
        ));
    }
    let std_profile = match seq.seq_profile {
        0 => StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
        1 => StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        _ => StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_PROFESSIONAL,
    };
    Ok(DecodeProfile {
        // Film grain is not applied, so the profile does not ask for it.
        av1: vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(std_profile)
            .film_grain_support(false),
        chroma,
        depth,
        pixel_format,
        bit_depth,
    })
}

/// Create the Vulkan session parameters holding `seq`.
pub(crate) fn build_session_params(
    common: &DecoderCommon,
    session: vk::VideoSessionKHR,
    seq: &SequenceHeader,
) -> Result<vk::VideoSessionParametersKHR> {
    let mut color_flags: StdVideoAV1ColorConfigFlags = unsafe { std::mem::zeroed() };
    color_flags.set_mono_chrome(seq.color.mono_chrome as u32);
    color_flags.set_color_range(seq.color.color_range as u32);
    color_flags.set_separate_uv_delta_q(seq.color.separate_uv_delta_q as u32);
    color_flags.set_color_description_present_flag(seq.color.color_description_present_flag as u32);
    let color_config = StdVideoAV1ColorConfig {
        flags: color_flags,
        BitDepth: seq.color.bit_depth,
        subsampling_x: seq.color.subsampling_x as u8,
        subsampling_y: seq.color.subsampling_y as u8,
        reserved1: 0,
        color_primaries: seq.color.color_primaries as StdVideoAV1ColorPrimaries,
        transfer_characteristics: seq.color.transfer_characteristics
            as StdVideoAV1TransferCharacteristics,
        matrix_coefficients: seq.color.matrix_coefficients as StdVideoAV1MatrixCoefficients,
        chroma_sample_position: seq.color.chroma_sample_position as StdVideoAV1ChromaSamplePosition,
    };

    let timing = seq.timing_info.map(|t| {
        let mut flags: StdVideoAV1TimingInfoFlags = unsafe { std::mem::zeroed() };
        flags.set_equal_picture_interval(t.equal_picture_interval as u32);
        StdVideoAV1TimingInfo {
            flags,
            num_units_in_display_tick: t.num_units_in_display_tick,
            time_scale: t.time_scale,
            num_ticks_per_picture_minus_1: t.num_ticks_per_picture_minus_1,
        }
    });

    let mut flags: StdVideoAV1SequenceHeaderFlags = unsafe { std::mem::zeroed() };
    flags.set_still_picture(seq.still_picture as u32);
    flags.set_reduced_still_picture_header(seq.reduced_still_picture_header as u32);
    flags.set_use_128x128_superblock(seq.use_128x128_superblock as u32);
    flags.set_enable_filter_intra(seq.enable_filter_intra as u32);
    flags.set_enable_intra_edge_filter(seq.enable_intra_edge_filter as u32);
    flags.set_enable_interintra_compound(seq.enable_interintra_compound as u32);
    flags.set_enable_masked_compound(seq.enable_masked_compound as u32);
    flags.set_enable_warped_motion(seq.enable_warped_motion as u32);
    flags.set_enable_dual_filter(seq.enable_dual_filter as u32);
    flags.set_enable_order_hint(seq.enable_order_hint as u32);
    flags.set_enable_jnt_comp(seq.enable_jnt_comp as u32);
    flags.set_enable_ref_frame_mvs(seq.enable_ref_frame_mvs as u32);
    flags.set_frame_id_numbers_present_flag(seq.frame_id_numbers_present_flag as u32);
    flags.set_enable_superres(seq.enable_superres as u32);
    flags.set_enable_cdef(seq.enable_cdef as u32);
    flags.set_enable_restoration(seq.enable_restoration as u32);
    flags.set_film_grain_params_present(seq.film_grain_params_present as u32);
    flags.set_timing_info_present_flag(seq.timing_info.is_some() as u32);
    flags.set_initial_display_delay_present_flag(seq.initial_display_delay_present_flag as u32);

    let std_seq = StdVideoAV1SequenceHeader {
        flags,
        seq_profile: seq.seq_profile as StdVideoAV1Profile,
        frame_width_bits_minus_1: seq.frame_width_bits_minus_1,
        frame_height_bits_minus_1: seq.frame_height_bits_minus_1,
        max_frame_width_minus_1: seq.max_frame_width_minus_1,
        max_frame_height_minus_1: seq.max_frame_height_minus_1,
        delta_frame_id_length_minus_2: seq.delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1: seq.additional_frame_id_length_minus_1,
        order_hint_bits_minus_1: seq.order_hint_bits.saturating_sub(1) as u8,
        seq_force_integer_mv: seq.seq_force_integer_mv,
        seq_force_screen_content_tools: seq.seq_force_screen_content_tools,
        reserved1: [0; 5],
        pColorConfig: &color_config,
        pTimingInfo: timing.as_ref().map_or(std::ptr::null(), |t| t as *const _),
    };

    let mut av1_create =
        vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default().std_sequence_header(&std_seq);
    let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
        .video_session(session)
        .push(&mut av1_create);
    unsafe {
        common
            .video_queue_fn
            .create_video_session_parameters(&create_info, None)
    }
    .map_err(|e| {
        PixelForgeError::SessionParametersCreation(format!("AV1 decode session parameters: {e:?}"))
    })
}

/// The `StdVideo*` description of one frame, which the picture info points
/// into. Boxed so the pointers stay put.
pub(crate) struct PictureStd {
    pub info: StdVideoDecodeAV1PictureInfo,
    tile_info: StdVideoAV1TileInfo,
    quantization: StdVideoAV1Quantization,
    segmentation: StdVideoAV1Segmentation,
    loop_filter: StdVideoAV1LoopFilter,
    cdef: StdVideoAV1CDEF,
    loop_restoration: StdVideoAV1LoopRestoration,
    global_motion: StdVideoAV1GlobalMotion,
    mi_col_starts: [u16; 64],
    mi_row_starts: [u16; 64],
    width_in_sbs_minus_1: [u16; 64],
    height_in_sbs_minus_1: [u16; 64],
}

impl PictureStd {
    pub fn new(fh: &FrameHeader) -> Box<Self> {
        let t = &fh.tile_info;
        let mut mi_col_starts = [0u16; 64];
        let mut mi_row_starts = [0u16; 64];
        let mut width_in_sbs_minus_1 = [0u16; 64];
        let mut height_in_sbs_minus_1 = [0u16; 64];
        for i in 0..64 {
            mi_col_starts[i] = t.mi_col_starts[i] as u16;
            mi_row_starts[i] = t.mi_row_starts[i] as u16;
            width_in_sbs_minus_1[i] = t.width_in_sbs_minus_1[i] as u16;
            height_in_sbs_minus_1[i] = t.height_in_sbs_minus_1[i] as u16;
        }

        let mut tile_flags: StdVideoAV1TileInfoFlags = unsafe { std::mem::zeroed() };
        tile_flags.set_uniform_tile_spacing_flag(t.uniform_tile_spacing_flag as u32);
        let tile_info = StdVideoAV1TileInfo {
            flags: tile_flags,
            TileCols: t.tile_cols as u8,
            TileRows: t.tile_rows as u8,
            context_update_tile_id: t.context_update_tile_id as u16,
            tile_size_bytes_minus_1: t.tile_size_bytes.saturating_sub(1) as u8,
            reserved1: [0; 7],
            pMiColStarts: std::ptr::null(),
            pMiRowStarts: std::ptr::null(),
            pWidthInSbsMinus1: std::ptr::null(),
            pHeightInSbsMinus1: std::ptr::null(),
        };

        let q = &fh.quant;
        let mut q_flags: StdVideoAV1QuantizationFlags = unsafe { std::mem::zeroed() };
        q_flags.set_using_qmatrix(q.using_qmatrix as u32);
        q_flags.set_diff_uv_delta(q.diff_uv_delta as u32);
        let quantization = StdVideoAV1Quantization {
            flags: q_flags,
            base_q_idx: q.base_q_idx,
            DeltaQYDc: q.delta_q_y_dc,
            DeltaQUDc: q.delta_q_u_dc,
            DeltaQUAc: q.delta_q_u_ac,
            DeltaQVDc: q.delta_q_v_dc,
            DeltaQVAc: q.delta_q_v_ac,
            qm_y: q.qm_y,
            qm_u: q.qm_u,
            qm_v: q.qm_v,
        };

        let mut segmentation = StdVideoAV1Segmentation {
            FeatureEnabled: [0; 8],
            FeatureData: [[0; 8]; 8],
        };
        for i in 0..8 {
            for j in 0..8 {
                segmentation.FeatureEnabled[i] |=
                    (fh.segmentation.feature_enabled[i][j] as u8) << j;
                segmentation.FeatureData[i][j] = fh.segmentation.feature_data[i][j];
            }
        }

        let lf = &fh.loop_filter;
        let mut lf_flags: StdVideoAV1LoopFilterFlags = unsafe { std::mem::zeroed() };
        lf_flags.set_loop_filter_delta_enabled(lf.delta_enabled as u32);
        lf_flags.set_loop_filter_delta_update(lf.delta_update as u32);
        let loop_filter = StdVideoAV1LoopFilter {
            flags: lf_flags,
            loop_filter_level: lf.level,
            loop_filter_sharpness: lf.sharpness,
            update_ref_delta: 0,
            loop_filter_ref_deltas: lf.ref_deltas,
            update_mode_delta: 0,
            loop_filter_mode_deltas: lf.mode_deltas,
        };

        let cdef = StdVideoAV1CDEF {
            cdef_damping_minus_3: fh.cdef.damping_minus_3,
            cdef_bits: fh.cdef.bits,
            cdef_y_pri_strength: fh.cdef.y_pri_strength,
            cdef_y_sec_strength: fh.cdef.y_sec_strength,
            cdef_uv_pri_strength: fh.cdef.uv_pri_strength,
            cdef_uv_sec_strength: fh.cdef.uv_sec_strength,
        };

        use crate::decoder::av1::frame_header::{
            RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
        };
        let lr_type = |t: u8| match t {
            RESTORE_WIENER => {
                StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER
            }
            RESTORE_SGRPROJ => {
                StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ
            }
            RESTORE_SWITCHABLE => {
                StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE
            }
            RESTORE_NONE => {
                StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE
            }
            _ => StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
        };
        let lr = &fh.loop_restoration;
        let loop_restoration = StdVideoAV1LoopRestoration {
            FrameRestorationType: [
                lr_type(lr.frame_restoration_type[0]),
                lr_type(lr.frame_restoration_type[1]),
                lr_type(lr.frame_restoration_type[2]),
            ],
            LoopRestorationSize: [
                lr.size_log2_minus_5[0] as u16,
                lr.size_log2_minus_5[1] as u16,
                lr.size_log2_minus_5[2] as u16,
            ],
        };

        let global_motion = StdVideoAV1GlobalMotion {
            GmType: fh.global_motion.gm_type,
            gm_params: fh.global_motion.gm_params,
        };

        let mut f: StdVideoDecodeAV1PictureInfoFlags = unsafe { std::mem::zeroed() };
        f.set_error_resilient_mode(fh.error_resilient_mode as u32);
        f.set_disable_cdf_update(fh.disable_cdf_update as u32);
        f.set_use_superres(fh.use_superres as u32);
        f.set_render_and_frame_size_different(fh.render_and_frame_size_different as u32);
        f.set_allow_screen_content_tools(fh.allow_screen_content_tools as u32);
        f.set_is_filter_switchable(fh.is_filter_switchable as u32);
        f.set_force_integer_mv(fh.force_integer_mv as u32);
        f.set_frame_size_override_flag(fh.frame_size_override_flag as u32);
        f.set_buffer_removal_time_present_flag(fh.buffer_removal_time_present_flag as u32);
        f.set_allow_intrabc(fh.allow_intrabc as u32);
        f.set_frame_refs_short_signaling(fh.frame_refs_short_signaling as u32);
        f.set_allow_high_precision_mv(fh.allow_high_precision_mv as u32);
        f.set_is_motion_mode_switchable(fh.is_motion_mode_switchable as u32);
        f.set_use_ref_frame_mvs(fh.use_ref_frame_mvs as u32);
        f.set_disable_frame_end_update_cdf(fh.disable_frame_end_update_cdf as u32);
        f.set_allow_warped_motion(fh.allow_warped_motion as u32);
        f.set_reduced_tx_set(fh.reduced_tx_set as u32);
        f.set_reference_select(fh.reference_select as u32);
        f.set_skip_mode_present(fh.skip_mode_present as u32);
        f.set_delta_q_present(fh.delta_q_present as u32);
        f.set_delta_lf_present(fh.delta_lf_present as u32);
        f.set_delta_lf_multi(fh.delta_lf_multi as u32);
        f.set_segmentation_enabled(fh.segmentation.enabled as u32);
        f.set_segmentation_update_map(fh.segmentation.update_map as u32);
        f.set_segmentation_temporal_update(fh.segmentation.temporal_update as u32);
        f.set_segmentation_update_data(fh.segmentation.update_data as u32);
        f.set_UsesLr(lr.uses_lr as u32);
        f.set_usesChromaLr(lr.uses_chroma_lr as u32);
        // Film grain is not applied: the picture is decoded without it.
        f.set_apply_grain(0);

        let mut order_hints = [0u8; 8];
        for (dst, &src) in order_hints.iter_mut().zip(&fh.order_hints) {
            *dst = src as u8;
        }
        let info = StdVideoDecodeAV1PictureInfo {
            flags: f,
            frame_type: fh.frame_type as StdVideoAV1FrameType,
            current_frame_id: fh.current_frame_id,
            OrderHint: fh.order_hint as u8,
            primary_ref_frame: fh.primary_ref_frame,
            refresh_frame_flags: fh.refresh_frame_flags,
            reserved1: 0,
            interpolation_filter: fh.interpolation_filter as StdVideoAV1InterpolationFilter,
            TxMode: fh.tx_mode as StdVideoAV1TxMode,
            delta_q_res: fh.delta_q_res,
            delta_lf_res: fh.delta_lf_res,
            SkipModeFrame: fh.skip_mode_frame,
            coded_denom: fh.coded_denom,
            reserved2: [0; 3],
            OrderHints: order_hints,
            expectedFrameId: [0; 8],
            pTileInfo: std::ptr::null(),
            pQuantization: std::ptr::null(),
            pSegmentation: std::ptr::null(),
            pLoopFilter: std::ptr::null(),
            pCDEF: std::ptr::null(),
            pLoopRestoration: std::ptr::null(),
            pGlobalMotion: std::ptr::null(),
            pFilmGrain: std::ptr::null(),
        };

        let mut boxed = Box::new(Self {
            info,
            tile_info,
            quantization,
            segmentation,
            loop_filter,
            cdef,
            loop_restoration,
            global_motion,
            mi_col_starts,
            mi_row_starts,
            width_in_sbs_minus_1,
            height_in_sbs_minus_1,
        });
        // Now that nothing moves any more, point the structs at each other.
        boxed.tile_info.pMiColStarts = boxed.mi_col_starts.as_ptr();
        boxed.tile_info.pMiRowStarts = boxed.mi_row_starts.as_ptr();
        boxed.tile_info.pWidthInSbsMinus1 = boxed.width_in_sbs_minus_1.as_ptr();
        boxed.tile_info.pHeightInSbsMinus1 = boxed.height_in_sbs_minus_1.as_ptr();
        boxed.info.pTileInfo = &boxed.tile_info;
        boxed.info.pQuantization = &boxed.quantization;
        boxed.info.pSegmentation = &boxed.segmentation;
        boxed.info.pLoopFilter = &boxed.loop_filter;
        boxed.info.pCDEF = &boxed.cdef;
        boxed.info.pLoopRestoration = &boxed.loop_restoration;
        boxed.info.pGlobalMotion = &boxed.global_motion;
        boxed
    }
}

/// The reference info Vulkan wants for a picture held in a DPB slot.
pub(crate) fn reference_info(meta: &SlotMeta) -> StdVideoDecodeAV1ReferenceInfo {
    let mut flags: StdVideoDecodeAV1ReferenceInfoFlags = unsafe { std::mem::zeroed() };
    flags.set_disable_frame_end_update_cdf(meta.disable_frame_end_update_cdf as u32);
    flags.set_segmentation_enabled(meta.segmentation_enabled as u32);
    StdVideoDecodeAV1ReferenceInfo {
        flags,
        frame_type: meta.frame_type,
        RefFrameSignBias: meta.ref_frame_sign_bias,
        OrderHint: meta.order_hint,
        SavedOrderHints: meta.saved_order_hints,
    }
}
