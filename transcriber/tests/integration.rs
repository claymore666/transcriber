use std::path::PathBuf;

use transcriber::{AudioProcessing, Model, TranscribeOptions};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Test that the audio pipeline decodes, processes, and produces valid samples
/// across all supported fixture formats.
#[tokio::test]
async fn audio_pipeline_decodes_all_formats() {
    let files = [
        ("sine_440hz_2s.wav", 30_000, 34_000),
        ("sine_440hz_1s.mp3", 14_000, 18_000),
        ("sine_440hz_1s.opus", 14_000, 18_000),
        ("sine_48khz_1s.flac", 14_000, 18_000),
        ("stereo_2s.wav", 30_000, 34_000),
    ];

    for (file, min_samples, max_samples) in files {
        let path = fixtures_dir().join(file);
        assert!(path.exists(), "fixture missing: {file}");

        // Use spawn_blocking since load_audio is sync
        let samples = tokio::task::spawn_blocking(move || {
            transcriber::__test_load_audio(&path, &AudioProcessing::default())
        })
        .await
        .unwrap()
        .unwrap_or_else(|e| panic!("{file}: {e}"));

        assert!(
            samples.len() > min_samples && samples.len() < max_samples,
            "{file}: expected {min_samples}-{max_samples} samples, got {}",
            samples.len()
        );

        // All samples should be in [-1.0, 1.0]
        for &s in &samples {
            assert!(
                (-1.0..=1.0).contains(&s),
                "{file}: sample {s} out of range"
            );
        }
    }
}

/// Test that audio processing options (DC offset, normalization, trimming)
/// all work together without errors.
#[tokio::test]
async fn audio_processing_all_options() {
    let path = fixtures_dir().join("sine_440hz_2s.wav");
    let processing = AudioProcessing::all();

    let samples = tokio::task::spawn_blocking(move || {
        transcriber::__test_load_audio(&path, &processing)
    })
    .await
    .unwrap()
    .unwrap();

    assert!(!samples.is_empty());

    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    assert!(
        (peak - 1.0).abs() < 0.01,
        "should be peak-normalized, got peak={peak}"
    );
}

/// Test that output formatters produce valid output for a constructed transcript.
#[test]
fn output_formats_roundtrip() {
    let transcript = transcriber::Transcript {
        segments: vec![
            transcriber::Segment {
                start: 0.0,
                end: 2.5,
                text: " Hello world.".into(),
                speaker_turn: false,
                no_speech_probability: 0.1,
                words: None,
            },
            transcriber::Segment {
                start: 3.0,
                end: 5.0,
                text: " Second segment.".into(),
                speaker_turn: true,
                no_speech_probability: 0.05,
                words: None,
            },
        ],
        language: "en".into(),
        duration: 5.0,
        model: "tiny".into(),
        source_url: None,
        source_title: None,
    };

    // SRT
    let srt = transcript.to_srt();
    assert!(srt.contains("1\n"));
    assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
    assert!(srt.contains("Hello world."));
    assert!(srt.contains("2\n"));

    // VTT
    let vtt = transcript.to_vtt();
    assert!(vtt.starts_with("WEBVTT\n\n"));
    assert!(vtt.contains("00:00:00.000 --> 00:00:02.500"));

    // JSON roundtrip
    let json = transcript.to_json().unwrap();
    let deserialized: transcriber::Transcript = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.segments.len(), 2);
    assert_eq!(deserialized.language, "en");

    // Text
    assert_eq!(transcript.text(), "Hello world. Second segment.");
}

/// Full end-to-end transcription test. Requires the tiny model to be cached.
/// Run with: cargo test --test integration -- --ignored full_transcription_pipeline
#[tokio::test]
#[ignore = "requires whisper tiny model (run with --download-model tiny first)"]
async fn full_transcription_pipeline() {
    let path = fixtures_dir().join("sine_440hz_2s.wav");

    let opts = TranscribeOptions::new()
        .model(Model::Tiny)
        .gpu(false);

    let transcript = transcriber::transcribe_file_with_options(&path, &opts)
        .await
        .expect("transcription should succeed with tiny model");

    assert!(transcript.duration > 1.0, "should detect audio duration");
    assert_eq!(transcript.model, "tiny");
    // The tiny model may or may not produce segments for a sine wave,
    // but it should not error out.
}
