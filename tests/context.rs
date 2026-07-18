//! Verifies device selection fails cleanly (not by panic) when the device has
//! no video decode support. Runs against lavapipe, which has no video queues.
use pixelforge::encoder::Codec;

#[test]
fn require_decode_fails_cleanly_without_video_support() {
    let result = pixelforge::vulkan::VideoContextBuilder::new()
        .app_name("pixelforge-test")
        .require_decode(Codec::H264)
        .build();
    match result {
        Ok(ctx) => {
            // If a real video-capable GPU is present, the contract must hold.
            assert!(ctx.supports_decode(Codec::H264));
        }
        Err(e) => {
            // Expected on CPU/drivers: a typed error, not a panic or hang.
            let msg = e.to_string();
            assert!(
                msg.contains("No device with required video support") || msg.contains("suitable"),
                "expected a NoSuitableDevice error, got: {msg}"
            );
        }
    }
}

#[test]
fn context_creation_reaches_device_enumeration() {
    // Guards against the test above passing for the wrong reason (e.g. the
    // Vulkan loader itself failing). The instance must be created and physical
    // devices enumerated before any device-support verdict is possible.
    let err = pixelforge::vulkan::VideoContextBuilder::new()
        .app_name("pixelforge-test")
        .require_decode(pixelforge::encoder::Codec::H264)
        .build()
        .err();
    if let Some(e) = err {
        let msg = e.to_string();
        assert!(
            !msg.contains("Failed to create Vulkan instance"),
            "Vulkan loader unavailable; the decode-support test would be vacuous: {msg}"
        );
    }
}
