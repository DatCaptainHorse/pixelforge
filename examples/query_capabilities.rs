//! Example: Query Video Capabilities
//!
//! Asks the device everything it will say about video encoding and prints it.
//!
//! Most of this is per *profile* rather than per device -- a profile being a
//! codec plus a chroma subsampling plus a bit depth -- and the difference
//! matters more than it looks. Intra refresh in particular is advertised as a
//! device extension but its usable modes come back per profile, so "this GPU
//! supports intra refresh" can be true while the codec you are encoding with
//! offers no mode at all. The per-profile sections below are the ones that
//! answer whether a given encode can actually do a thing; the device section
//! only answers whether the driver knows the word.
//!
//! Run it with a resolution to change the derived figures:
//!
//! ```text
//! cargo run --example query_capabilities -- 2560x1440 144
//! ```

use ash::vk;
use ash::vk::TaggedStructure;
use pixelforge::{Codec, VideoContextBuilder};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

/// The picture the derived figures are worked out for. Defaults to 1080p60;
/// override from the command line.
#[derive(Clone, Copy)]
struct Reference {
    width: u32,
    height: u32,
    fps: u32,
}

impl Reference {
    fn from_args() -> Self {
        let mut r = Self {
            width: 1920,
            height: 1080,
            fps: 60,
        };
        let args: Vec<String> = std::env::args().skip(1).collect();
        for arg in &args {
            if let Some((w, h)) = arg.split_once(['x', 'X'])
                && let (Ok(w), Ok(h)) = (w.trim().parse(), h.trim().parse())
            {
                r.width = w;
                r.height = h;
            } else if let Ok(fps) = arg.trim().parse::<u32>() {
                r.fps = fps.max(1);
            }
        }
        r
    }

    /// How long `pictures` frames last at this rate, as a printable string.
    fn seconds(&self, pictures: u32) -> String {
        format!("{:.2} s", pictures as f32 / self.fps as f32)
    }
}

/// One line of the closing summary: what a single profile offers for intra
/// refresh, so the profiles can be compared side by side without scrolling
/// back through the detail.
struct IntraRefreshRow {
    codec: Codec,
    format: &'static str,
    modes: Option<vk::VideoEncodeIntraRefreshModeFlagsKHR>,
    max_cycle: u32,
    max_active_refs: u32,
    /// Whether the codec lets `constrained_intra_pred_flag` be set. Without it
    /// a refreshed region may predict from an unrefreshed one, so the sweep
    /// never actually converges. `None` where the codec has no such flag.
    constrained_intra_pred: Option<bool>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            ),
        )
        .init();

    let reference = Reference::from_args();

    println!("PixelForge Video Capability Report");
    println!("==================================\n");

    let context = VideoContextBuilder::new()
        .app_name("Capabilities Example")
        .app_version(1, 0, 0)
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    let video_queue_fn = ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let video_encode_fn =
        ash::khr::video_encode_queue::Instance::load(context.entry(), context.instance());

    print_device_section(&context, reference);

    let mut summary: Vec<IntraRefreshRow> = Vec::new();

    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        let heading = format!("{codec:?} encode");
        println!("\n{heading}");
        println!("{}", "-".repeat(heading.len()));

        if !context.supports_encode(codec) {
            println!("  not supported on this device");
            continue;
        }

        report_codec(
            &context,
            codec,
            &video_queue_fn,
            &video_encode_fn,
            reference,
            &mut summary,
        );
    }

    print_decode_image_flags(&context)?;
    print_intra_refresh_summary(&summary, reference);

    Ok(())
}

// ── Device ────────────────────────────────────────────────────────────────────

/// What the device says about itself, and the two things intra refresh needs
/// before any profile is asked anything.
fn print_device_section(context: &pixelforge::VideoContext, reference: Reference) {
    let props = context.device_properties();
    println!("Device");
    println!("------");
    println!("  Name:          {}", c_name(&props.device_name));
    println!("  Type:          {:?}", props.device_type);
    println!(
        "  Vulkan API:    {}.{}.{}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version),
    );
    println!(
        "  Vendor/device: {:#06x}/{:#06x}",
        props.vendor_id, props.device_id
    );

    let families = unsafe {
        context
            .instance()
            .get_physical_device_queue_family_properties(context.physical_device())
    };
    println!("  Queue families:");
    for (i, f) in families.iter().enumerate() {
        let video = f.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR)
            || f.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR);
        println!(
            "    {i}: {:?} ({} queue{}){}",
            f.queue_flags,
            f.queue_count,
            if f.queue_count == 1 { "" } else { "s" },
            if video { "  <- video" } else { "" },
        );
    }

    // The extension and the feature bit are reported apart because they fail
    // apart. An extension present with its feature off means the driver parses
    // the intra refresh structs and ignores them -- worse than absent, because
    // the encode looks configured and the stream silently never refreshes.
    let (ext_present, feature_on) = intra_refresh_availability(context);
    println!("  VK_KHR_video_encode_intra_refresh:");
    println!("    extension advertised: {}", yes_no(ext_present));
    println!("    feature enabled:      {}", yes_no(feature_on));
    println!(
        "    usable by pixelforge: {}",
        yes_no(context.has_video_encode_intra_refresh())
    );
    if ext_present && !feature_on {
        println!("    ^ the driver knows the structs and will ignore them");
    }

    println!(
        "\n  Derived figures below assume {}x{} at {} fps.",
        reference.width, reference.height, reference.fps
    );
}

/// Whether the extension is advertised, and whether its feature bit is set.
fn intra_refresh_availability(context: &pixelforge::VideoContext) -> (bool, bool) {
    let exts = unsafe {
        context
            .instance()
            .enumerate_device_extension_properties(context.physical_device())
    };
    let Ok(exts) = exts else {
        return (false, false);
    };
    let present = exts.iter().any(|e| {
        c_name(&e.extension_name) == ash::khr::video_encode_intra_refresh::NAME.to_string_lossy()
    });
    if !present {
        return (false, false);
    }
    let mut feature = vk::PhysicalDeviceVideoEncodeIntraRefreshFeaturesKHR::default();
    let mut query = vk::PhysicalDeviceFeatures2::default().push(&mut feature);
    unsafe {
        context
            .instance()
            .get_physical_device_features2(context.physical_device(), &mut query);
    }
    (true, feature.video_encode_intra_refresh != 0)
}

// ── Per codec ─────────────────────────────────────────────────────────────────

fn report_codec(
    context: &pixelforge::VideoContext,
    codec: Codec,
    video_queue_fn: &ash::khr::video_queue::Instance,
    video_encode_fn: &ash::khr::video_encode_queue::Instance,
    reference: Reference,
    summary: &mut Vec<IntraRefreshRow>,
) {
    let physical_device = context.physical_device();

    let combinations = [
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            "4:2:0 8-bit",
        ),
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            "4:4:4 8-bit",
        ),
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            "4:2:0 10-bit",
        ),
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            "4:4:4 10-bit",
        ),
    ];

    for (subsampling, bit_depth, desc) in combinations {
        println!("\n  {desc}");

        let (mut profile_info, mut h264_profile, mut h265_profile, mut av1_profile) =
            build_profile(codec, subsampling, bit_depth);
        if let Some(h264) = &mut h264_profile {
            profile_info = profile_info.push(h264);
        }
        if let Some(h265) = &mut h265_profile {
            profile_info = profile_info.push(h265);
        }
        if let Some(av1) = &mut av1_profile {
            profile_info = profile_info.push(av1);
        }

        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut intra_caps = vk::VideoEncodeIntraRefreshCapabilitiesKHR::default();
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut av1_caps = vk::VideoEncodeAV1CapabilitiesKHR::default();

        let mut caps = vk::VideoCapabilitiesKHR::default().push(&mut encode_caps);
        // Only chained where the device can fill it. A struct the driver does
        // not know is ignored and left at its defaults, and zeroes that came
        // from nobody read exactly like zeroes that came from the device.
        let ask_intra = context.has_video_encode_intra_refresh();
        if ask_intra {
            caps = caps.push(&mut intra_caps);
        }
        match codec {
            Codec::H264 => caps = caps.push(&mut h264_caps),
            Codec::H265 => caps = caps.push(&mut h265_caps),
            Codec::AV1 => caps = caps.push(&mut av1_caps),
        }

        let result = unsafe {
            (video_queue_fn
                .fp()
                .get_physical_device_video_capabilities_khr)(
                physical_device,
                &profile_info,
                &mut caps,
            )
        };
        if result != vk::Result::SUCCESS {
            println!("    unsupported ({result:?})");
            continue;
        }

        print_video_caps(&caps);
        print_encode_caps(&encode_caps);

        let modes = if ask_intra {
            print_intra_refresh(&intra_caps, &encode_caps, reference);
            Some(intra_caps.intra_refresh_modes)
        } else {
            println!("\n    Intra refresh");
            println!("      unavailable on this device — not queried");
            None
        };

        let constrained = match codec {
            Codec::H264 => {
                print_h264_caps(&h264_caps);
                Some(
                    h264_caps
                        .std_syntax_flags
                        .contains(vk::VideoEncodeH264StdFlagsKHR::CONSTRAINED_INTRA_PRED_FLAG_SET),
                )
            }
            Codec::H265 => {
                print_h265_caps(&h265_caps);
                Some(
                    h265_caps
                        .std_syntax_flags
                        .contains(vk::VideoEncodeH265StdFlagsKHR::CONSTRAINED_INTRA_PRED_FLAG_SET),
                )
            }
            Codec::AV1 => {
                print_av1_caps(&av1_caps);
                None
            }
        };

        print_quality_levels(
            video_encode_fn,
            physical_device,
            &profile_info,
            &encode_caps,
            codec,
        );
        print_formats(video_queue_fn, physical_device, &profile_info);

        summary.push(IntraRefreshRow {
            codec,
            format: desc,
            modes,
            max_cycle: intra_caps.max_intra_refresh_cycle_duration,
            max_active_refs: intra_caps.max_intra_refresh_active_reference_pictures,
            constrained_intra_pred: constrained,
        });
    }
}

#[allow(clippy::type_complexity)]
fn build_profile(
    codec: Codec,
    subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> (
    vk::VideoProfileInfoKHR<'static>,
    Option<vk::VideoEncodeH264ProfileInfoKHR<'static>>,
    Option<vk::VideoEncodeH265ProfileInfoKHR<'static>>,
    Option<vk::VideoEncodeAV1ProfileInfoKHR<'static>>,
) {
    let base = |op| {
        vk::VideoProfileInfoKHR::default()
            .video_codec_operation(op)
            .chroma_subsampling(subsampling)
            .luma_bit_depth(bit_depth)
            .chroma_bit_depth(bit_depth)
    };
    let is_444 = subsampling == vk::VideoChromaSubsamplingFlagsKHR::TYPE_444;
    let is_10bit = bit_depth == vk::VideoComponentBitDepthFlagsKHR::TYPE_10;

    match codec {
        Codec::H264 => {
            let idc = if is_444 {
                vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
            } else {
                vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH
            };
            (
                base(vk::VideoCodecOperationFlagsKHR::ENCODE_H264),
                Some(vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(idc)),
                None,
                None,
            )
        }
        Codec::H265 => {
            let idc = if is_444 {
                vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
            } else if is_10bit {
                vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
            } else {
                vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
            };
            (
                base(vk::VideoCodecOperationFlagsKHR::ENCODE_H265),
                None,
                Some(vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(idc)),
                None,
            )
        }
        Codec::AV1 => {
            let profile = if is_444 {
                vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH
            } else {
                vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN
            };
            (
                base(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1),
                None,
                None,
                Some(vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(profile)),
            )
        }
    }
}

// ── Capability sections ───────────────────────────────────────────────────────

fn print_video_caps(caps: &vk::VideoCapabilitiesKHR) {
    println!("    Session");
    println!(
        "      Coded extent:        {}x{} .. {}x{}",
        caps.min_coded_extent.width,
        caps.min_coded_extent.height,
        caps.max_coded_extent.width,
        caps.max_coded_extent.height,
    );
    println!(
        "      Picture granularity: {}x{}",
        caps.picture_access_granularity.width, caps.picture_access_granularity.height,
    );
    println!("      DPB slots:           {}", caps.max_dpb_slots);
    println!(
        "      Active references:   {}",
        caps.max_active_reference_pictures
    );
    println!("      Flags:               {}", flags_or_none(caps.flags));
    println!(
        "      Bitstream alignment: offset {}, size {}",
        caps.min_bitstream_buffer_offset_alignment, caps.min_bitstream_buffer_size_alignment,
    );
    let v = caps.std_header_version.spec_version;
    println!(
        "      Std header:          {} v{}.{}.{}",
        c_name(&caps.std_header_version.extension_name),
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v),
    );
}

fn print_encode_caps(caps: &vk::VideoEncodeCapabilitiesKHR) {
    println!("    Encode");
    println!("      Flags:               {}", flags_or_none(caps.flags));
    println!(
        "      Rate control modes:  {}",
        flags_or_none(caps.rate_control_modes)
    );
    println!(
        "      Rate control layers: {}",
        caps.max_rate_control_layers
    );
    println!(
        "      Max bitrate:         {} bps ({:.1} Mbps)",
        caps.max_bitrate,
        caps.max_bitrate as f64 / 1_000_000.0,
    );
    println!("      Quality levels:      {}", caps.max_quality_levels);
    println!(
        "      Input granularity:   {}x{}",
        caps.encode_input_picture_granularity.width, caps.encode_input_picture_granularity.height,
    );
    println!(
        "      Feedback flags:      {}",
        flags_or_none(caps.supported_encode_feedback_flags)
    );
}

/// Intra refresh, plus the arithmetic that turns the reported numbers into the
/// thing anyone actually wants to know: how coarse a sweep this device can do.
///
/// A refresh region is built out of the encoder's input picture granularity
/// blocks, so a picture is a fixed number of blocks tall and wide. The finest
/// possible sweep refreshes one block row per picture, which takes as many
/// pictures as the picture is blocks tall. When the device's maximum cycle
/// duration is shorter than that, the finest sweep is not available: each
/// picture must refresh more than one row, the refreshed band is that much
/// wider, and a wider band is a more visible one.
fn print_intra_refresh(
    caps: &vk::VideoEncodeIntraRefreshCapabilitiesKHR,
    encode: &vk::VideoEncodeCapabilitiesKHR,
    reference: Reference,
) {
    println!("    Intra refresh");
    println!(
        "      Modes:               {}",
        flags_or_none(caps.intra_refresh_modes)
    );
    if caps.intra_refresh_modes.is_empty() {
        println!("      ^ this profile offers no refresh mode; key frames are the only option");
        return;
    }
    println!(
        "      Max cycle duration:  {} pictures ({})",
        caps.max_intra_refresh_cycle_duration,
        reference.seconds(caps.max_intra_refresh_cycle_duration),
    );
    println!(
        "      Max active refs:     {}",
        caps.max_intra_refresh_active_reference_pictures
    );
    println!(
        "      Partition independent regions: {}",
        yes_no(caps.partition_independent_intra_refresh_regions != 0)
    );
    println!(
        "      Non-rectangular regions:       {}",
        yes_no(caps.non_rectangular_intra_refresh_regions != 0)
    );

    let bw = encode.encode_input_picture_granularity.width.max(1);
    let bh = encode.encode_input_picture_granularity.height.max(1);
    let blocks_wide = reference.width.div_ceil(bw);
    let blocks_tall = reference.height.div_ceil(bh);
    println!(
        "      At {}x{} with {bw}x{bh} blocks: {blocks_wide} wide x {blocks_tall} tall",
        reference.width, reference.height,
    );

    for (label, needed) in [
        ("one block row per picture", blocks_tall),
        ("one block column per picture", blocks_wide),
    ] {
        let cap = caps.max_intra_refresh_cycle_duration;
        if cap >= needed {
            println!(
                "        {label}: {needed} pictures per sweep ({}) — allowed",
                reference.seconds(needed),
            );
        } else {
            // Rounded up: the rows have to be covered, so a cycle that divides
            // unevenly makes some pictures carry an extra row rather than
            // leaving one unrefreshed.
            let rows_per_picture = needed.div_ceil(cap.max(1));
            println!(
                "        {label}: would need {needed} pictures, device allows {cap} — \
                 at least {rows_per_picture} per picture",
            );
        }
    }
}

fn print_h264_caps(caps: &vk::VideoEncodeH264CapabilitiesKHR) {
    println!("    H.264");
    println!("      Flags:               {}", flags_or_none(caps.flags));
    println!(
        "      QP range:            {}..{}",
        caps.min_qp, caps.max_qp
    );
    println!("      Max level idc:       {:?}", caps.max_level_idc);
    println!("      Max slices:          {}", caps.max_slice_count);
    println!(
        "      References:          P L0 {}, B L0 {}, L1 {}",
        caps.max_p_picture_l0_reference_count,
        caps.max_b_picture_l0_reference_count,
        caps.max_l1_reference_count,
    );
    println!(
        "      Temporal layers:     {} (dyadic expected: {})",
        caps.max_temporal_layer_count,
        yes_no(caps.expect_dyadic_temporal_layer_pattern != 0),
    );
    println!(
        "      GOP remaining frames: prefers {}, requires {}",
        yes_no(caps.prefers_gop_remaining_frames != 0),
        yes_no(caps.requires_gop_remaining_frames != 0),
    );
    println!(
        "      Std syntax flags:    {}",
        flags_or_none(caps.std_syntax_flags)
    );
    print_constrained_intra_pred(
        caps.std_syntax_flags
            .contains(vk::VideoEncodeH264StdFlagsKHR::CONSTRAINED_INTRA_PRED_FLAG_SET),
    );
}

fn print_h265_caps(caps: &vk::VideoEncodeH265CapabilitiesKHR) {
    println!("    H.265");
    println!("      Flags:               {}", flags_or_none(caps.flags));
    println!(
        "      QP range:            {}..{}",
        caps.min_qp, caps.max_qp
    );
    println!("      Max level idc:       {:?}", caps.max_level_idc);
    println!(
        "      Max slice segments:  {}",
        caps.max_slice_segment_count
    );
    println!(
        "      Max tiles:           {}x{}",
        caps.max_tiles.width, caps.max_tiles.height
    );
    println!(
        "      CTB sizes:           {}",
        flags_or_none(caps.ctb_sizes)
    );
    println!(
        "      Transform blocks:    {}",
        flags_or_none(caps.transform_block_sizes)
    );
    println!(
        "      References:          P L0 {}, B L0 {}, L1 {}",
        caps.max_p_picture_l0_reference_count,
        caps.max_b_picture_l0_reference_count,
        caps.max_l1_reference_count,
    );
    println!(
        "      Sub-layers:          {} (dyadic expected: {})",
        caps.max_sub_layer_count,
        yes_no(caps.expect_dyadic_temporal_sub_layer_pattern != 0),
    );
    println!(
        "      GOP remaining frames: prefers {}, requires {}",
        yes_no(caps.prefers_gop_remaining_frames != 0),
        yes_no(caps.requires_gop_remaining_frames != 0),
    );
    println!(
        "      Std syntax flags:    {}",
        flags_or_none(caps.std_syntax_flags)
    );
    print_constrained_intra_pred(
        caps.std_syntax_flags
            .contains(vk::VideoEncodeH265StdFlagsKHR::CONSTRAINED_INTRA_PRED_FLAG_SET),
    );
}

fn print_av1_caps(caps: &vk::VideoEncodeAV1CapabilitiesKHR) {
    println!("    AV1");
    println!("      Flags:               {}", flags_or_none(caps.flags));
    println!(
        "      q-index range:       {}..{}",
        caps.min_q_index, caps.max_q_index
    );
    println!("      Max level:           {:?}", caps.max_level);
    println!(
        "      Picture alignment:   {}x{}",
        caps.coded_picture_alignment.width, caps.coded_picture_alignment.height,
    );
    println!(
        "      Tiles:               up to {}x{}, size {}x{} .. {}x{}",
        caps.max_tiles.width,
        caps.max_tiles.height,
        caps.min_tile_size.width,
        caps.min_tile_size.height,
        caps.max_tile_size.width,
        caps.max_tile_size.height,
    );
    println!(
        "      Superblock sizes:    {}",
        flags_or_none(caps.superblock_sizes)
    );
    println!(
        "      Single references:   {} (name mask {:#x})",
        caps.max_single_reference_count, caps.single_reference_name_mask,
    );
    println!(
        "      Layers:              {} temporal, {} spatial, {} operating points",
        caps.max_temporal_layer_count, caps.max_spatial_layer_count, caps.max_operating_points,
    );
    println!(
        "      GOP remaining frames: prefers {}, requires {}",
        yes_no(caps.prefers_gop_remaining_frames != 0),
        yes_no(caps.requires_gop_remaining_frames != 0),
    );
    println!(
        "      Std syntax flags:    {}",
        flags_or_none(caps.std_syntax_flags)
    );
}

/// Called out separately because intra refresh depends on it and nothing else
/// in the flag list says so. A refreshed region that is allowed to predict
/// from an unrefreshed one carries the stale content forward, so the sweep
/// passes over the picture without ever making it correct.
fn print_constrained_intra_pred(settable: bool) {
    println!(
        "      constrained_intra_pred settable: {}{}",
        yes_no(settable),
        if settable {
            ""
        } else {
            "  <- intra refresh cannot converge without it"
        },
    );
}

/// The driver's own preferences, per quality level.
///
/// Worth asking because it is the only place the implementation says what it
/// would rather be given: a rate control mode, a constant QP, a GOP length. A
/// configuration that disagrees with all of them is not wrong, but it is
/// running against the grain of whatever the driver tuned for.
fn print_quality_levels(
    video_encode_fn: &ash::khr::video_encode_queue::Instance,
    physical_device: vk::PhysicalDevice,
    profile: &vk::VideoProfileInfoKHR<'_>,
    encode_caps: &vk::VideoEncodeCapabilitiesKHR,
    codec: Codec,
) {
    println!("    Quality levels");
    if encode_caps.max_quality_levels == 0 {
        println!("      none reported");
        return;
    }

    for level in 0..encode_caps.max_quality_levels {
        let mut h264 = vk::VideoEncodeH264QualityLevelPropertiesKHR::default();
        let mut h265 = vk::VideoEncodeH265QualityLevelPropertiesKHR::default();
        let mut av1 = vk::VideoEncodeAV1QualityLevelPropertiesKHR::default();
        let mut props = vk::VideoEncodeQualityLevelPropertiesKHR::default();
        props = match codec {
            Codec::H264 => props.push(&mut h264),
            Codec::H265 => props.push(&mut h265),
            Codec::AV1 => props.push(&mut av1),
        };

        let info = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR::default()
            .video_profile(profile)
            .quality_level(level);

        let result = unsafe {
            video_encode_fn.get_physical_device_video_encode_quality_level_properties(
                physical_device,
                &info,
                &mut props,
            )
        };
        if let Err(e) = result {
            println!("      {level}: query failed ({e:?})");
            continue;
        }

        println!(
            "      {level}: prefers {} with {} layer(s)",
            flags_or_none(props.preferred_rate_control_mode),
            props.preferred_rate_control_layer_count,
        );
        match codec {
            Codec::H264 => println!(
                "         GOP {}, IDR period {}, {} B-frame(s), QP I/P/B {}/{}/{}, \
                 entropy coding {}, rc flags {}",
                h264.preferred_gop_frame_count,
                h264.preferred_idr_period,
                h264.preferred_consecutive_b_frame_count,
                h264.preferred_constant_qp.qp_i,
                h264.preferred_constant_qp.qp_p,
                h264.preferred_constant_qp.qp_b,
                if h264.preferred_std_entropy_coding_mode_flag != 0 {
                    "CABAC"
                } else {
                    "CAVLC"
                },
                flags_or_none(h264.preferred_rate_control_flags),
            ),
            Codec::H265 => println!(
                "         GOP {}, IDR period {}, {} B-frame(s), QP I/P/B {}/{}/{}, rc flags {}",
                h265.preferred_gop_frame_count,
                h265.preferred_idr_period,
                h265.preferred_consecutive_b_frame_count,
                h265.preferred_constant_qp.qp_i,
                h265.preferred_constant_qp.qp_p,
                h265.preferred_constant_qp.qp_b,
                flags_or_none(h265.preferred_rate_control_flags),
            ),
            Codec::AV1 => println!(
                "         GOP {}, key frame period {}, q-index I/P/B {}/{}/{}, rc flags {}",
                av1.preferred_gop_frame_count,
                av1.preferred_key_frame_period,
                av1.preferred_constant_q_index.intra_q_index,
                av1.preferred_constant_q_index.predictive_q_index,
                av1.preferred_constant_q_index.bipredictive_q_index,
                flags_or_none(av1.preferred_rate_control_flags),
            ),
        }
    }
}

fn print_formats(
    video_queue_fn: &ash::khr::video_queue::Instance,
    physical_device: vk::PhysicalDevice,
    profile: &vk::VideoProfileInfoKHR<'_>,
) {
    println!("    Formats");
    for (label, usage) in [
        ("Input (SRC)", vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR),
        ("DPB", vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR),
    ] {
        let profiles = [*profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(usage)
            .push(&mut list);

        let mut count = 0u32;
        let result = unsafe {
            (video_queue_fn
                .fp()
                .get_physical_device_video_format_properties_khr)(
                physical_device,
                &info,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if result != vk::Result::SUCCESS || count == 0 {
            println!("      {label}: none");
            continue;
        }
        let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
        unsafe {
            let _ = (video_queue_fn
                .fp()
                .get_physical_device_video_format_properties_khr)(
                physical_device,
                &info,
                &mut count,
                props.as_mut_ptr(),
            );
        }
        println!("      {label}:");
        for prop in props.iter().take(count as usize) {
            let planes = prop
                .image_create_flags
                .contains(vk::ImageCreateFlags::MUTABLE_FORMAT);
            println!(
                "        {:?}  flags: {}{}",
                prop.format,
                flags_or_none(prop.image_create_flags),
                if planes { "  (per-plane views)" } else { "" },
            );
        }
    }
}

// ── Summary ───────────────────────────────────────────────────────────────────

/// Every profile's intra refresh support on adjacent lines.
///
/// The per-profile sections above have the detail; this is the comparison,
/// which is what the detail is usually being read for. Modes are advertised
/// per profile, so two rows here routinely disagree on the same device.
fn print_intra_refresh_summary(rows: &[IntraRefreshRow], reference: Reference) {
    println!("\nIntra Refresh Summary");
    println!("---------------------");
    if rows.is_empty() {
        println!("  no encode profile on this device was queryable");
        return;
    }
    println!(
        "  {:<6} {:<14} {:<10} {:<8} {:<7} modes",
        "codec", "format", "cycle max", "refs", "c.i.p."
    );
    for row in rows {
        let modes = match row.modes {
            None => "not queried".to_string(),
            Some(m) if m.is_empty() => "none".to_string(),
            Some(m) => format!("{m:?}"),
        };
        let cycle = match row.modes {
            Some(m) if !m.is_empty() => format!("{}", row.max_cycle),
            _ => "-".to_string(),
        };
        let refs = match row.modes {
            Some(m) if !m.is_empty() => format!("{}", row.max_active_refs),
            _ => "-".to_string(),
        };
        let cip = match row.constrained_intra_pred {
            Some(true) => "yes",
            Some(false) => "NO",
            None => "n/a",
        };
        println!(
            "  {:<6} {:<14} {:<10} {:<8} {:<7} {}",
            format!("{:?}", row.codec),
            row.format,
            cycle,
            refs,
            cip,
            modes,
        );
    }
    println!(
        "\n  cycle max is in pictures — {} at {} fps for the largest value shown.",
        reference.seconds(rows.iter().map(|r| r.max_cycle).max().unwrap_or(0)),
        reference.fps,
    );
    println!(
        "  c.i.p. is constrained_intra_pred: without it a refreshed region may predict from an"
    );
    println!("  unrefreshed one, so the sweep never makes the picture correct.");
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Report the image creation flags H.264 decode pictures allow.
///
/// Worth its own section because `imageCreateFlags` varies by *usage*, not just
/// by format, and the answer a consumer cares about is the one for the usage
/// pixelforge actually creates pictures with. `MUTABLE_FORMAT` there means
/// decoded frames can have their luma and chroma planes viewed separately, so a
/// renderer can read them as two ordinary textures instead of needing a
/// sampler-YCbCr conversion. Asking about the DPB usage alone gives a different,
/// and for this purpose wrong, answer: both RADV and ANV report no flags at all
/// for that.
fn print_decode_image_flags(
    context: &pixelforge::VideoContext,
) -> Result<(), Box<dyn std::error::Error>> {
    if !context.supports_decode(Codec::H264) {
        return Ok(());
    }
    println!("\nH.264 Decode Picture Image Flags");
    println!("---------------------------------");

    let mut h264 = vk::VideoDecodeH264ProfileInfoKHR::default()
        .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH)
        .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
    let profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push(&mut h264);

    let dpb = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
    let coincide = dpb
        | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::SAMPLED;

    let video_queue_fn = ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    for (label, usage) in [
        (
            "pictures pixelforge creates (DPB|DST|SRC|SAMPLED)",
            coincide,
        ),
        ("reference-only DPB", dpb),
    ] {
        let profiles = [profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(usage)
            .push(&mut list);
        let mut count = 0u32;
        let result = unsafe {
            (video_queue_fn
                .fp()
                .get_physical_device_video_format_properties_khr)(
                context.physical_device(),
                &info,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if result != vk::Result::SUCCESS || count == 0 {
            println!("  {label}: unsupported");
            continue;
        }
        let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
        unsafe {
            let _ = (video_queue_fn
                .fp()
                .get_physical_device_video_format_properties_khr)(
                context.physical_device(),
                &info,
                &mut count,
                props.as_mut_ptr(),
            );
        }
        println!("  {label}:");
        for prop in props.iter().take(count as usize) {
            let planes = prop
                .image_create_flags
                .contains(vk::ImageCreateFlags::MUTABLE_FORMAT);
            println!(
                "    {:?}  flags: {:?}  per-plane views: {}",
                prop.format, prop.image_create_flags, planes
            );
        }
    }
    Ok(())
}

// ── Formatting ────────────────────────────────────────────────────────────────

fn yes_no(v: bool) -> &'static str {
    if v { "yes" } else { "no" }
}

/// `{:?}` on an empty bitflags prints `0` or an empty set depending on the
/// flag type, and neither reads as "the device offers nothing here".
fn flags_or_none<T: std::fmt::Debug + Copy>(flags: T) -> String {
    let s = format!("{flags:?}");
    if s.is_empty() || s == "0" || s == "(empty)" {
        "none".to_string()
    } else {
        s
    }
}

/// A Vulkan fixed-size name array as a `String`.
fn c_name(name: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
