//! Video/audio transcription library — URL or file in, transcript with timestamps out.
//!
//! **transcriber** handles the full pipeline: downloading (via yt-dlp), audio decoding
//! (via symphonia), resampling to 16 kHz mono, and transcription (via whisper.cpp).
//! Output as plain text, SRT, WebVTT, or JSON.
//!
//! # Quick start
//!
//! ```rust,no_run
//! # #[tokio::main]
//! # async fn main() -> transcriber::Result<()> {
//! // Transcribe a local file
//! let transcript = transcriber::transcribe_file("meeting.mp3").await?;
//! println!("{}", transcript.text());
//!
//! // Or from a URL (requires the "download" feature, enabled by default)
//! let transcript = transcriber::transcribe("https://example.com/video").await?;
//! println!("{}", transcript.to_srt());
//! # Ok(())
//! # }
//! ```
//!
//! See the [README](https://github.com/claymore666/transcriber) for full documentation,
//! feature flags, and CLI usage.

pub(crate) mod audio;
pub mod config;
#[cfg(feature = "download")]
pub(crate) mod download;
pub mod error;
pub mod model;
#[cfg(feature = "speaker-id")]
pub mod speaker;
pub(crate) mod transcribe;
pub mod types;

pub use config::{AudioProcessing, Language, Model, TranscribeOptions};
pub use error::{Error, Result};
pub use types::{Segment, Transcript, Word};

/// Test-only access to audio loading (not part of the public API).
#[doc(hidden)]
pub fn __test_load_audio(
    path: &std::path::Path,
    processing: &AudioProcessing,
) -> Result<Vec<f32>> {
    audio::load_audio(path, processing)
}

use std::path::Path;

/// Transcribe a local audio/video file with default options.
pub async fn transcribe_file(path: impl AsRef<Path>) -> Result<Transcript> {
    transcribe_file_with_options(path, &TranscribeOptions::default()).await
}

/// Transcribe a local audio/video file with custom options.
pub async fn transcribe_file_with_options(
    path: impl AsRef<Path>,
    options: &TranscribeOptions,
) -> Result<Transcript> {
    let path = path.as_ref().to_path_buf();

    // Ensure model is available
    let cache_dir = options.resolve_cache_dir();
    let model_path = model::ensure_model(&options.model, &cache_dir).await?;

    // Load and process audio (blocking ffmpeg subprocess)
    let processing = options.audio_processing.clone();
    let samples = tokio::task::spawn_blocking({
        let path = path.clone();
        move || audio::load_audio(&path, &processing)
    })
    .await
    .map_err(|e| Error::Transcription(format!("audio loading task failed: {e}")))??;

    // Transcribe (blocking CPU-intensive whisper inference)
    let options_clone = options.clone();
    let samples_clone = samples.clone();
    #[allow(unused_mut)]
    let mut transcript = tokio::task::spawn_blocking(move || {
        transcribe::transcribe_samples(&samples_clone, &model_path, &options_clone)
    })
    .await
    .map_err(|e| Error::Transcription(format!("transcription task failed: {e}")))??;

    // Speaker identification pass (if enabled)
    #[cfg(feature = "speaker-id")]
    if options.speaker_identification {
        run_speaker_identification(&mut transcript, &samples, options).await?;
    }

    Ok(transcript)
}

/// Transcribe from a URL (downloads audio first, then transcribes).
#[cfg(feature = "download")]
pub async fn transcribe(url: &str) -> Result<Transcript> {
    transcribe_with_options(url, &TranscribeOptions::default()).await
}

/// Transcribe from a URL with custom options.
#[cfg(feature = "download")]
pub async fn transcribe_with_options(
    url: &str,
    options: &TranscribeOptions,
) -> Result<Transcript> {
    // Create a unique temp directory per invocation so concurrent runs
    // (even within the same process) don't collide.
    let tmp_dir = std::env::temp_dir().join(format!(
        "transcriber-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let _cleanup = TempDirGuard(&tmp_dir);

    let download_result = download::download_audio(url, &tmp_dir).await?;

    // Ensure model is available
    let cache_dir = options.resolve_cache_dir();
    let model_path = model::ensure_model(&options.model, &cache_dir).await?;

    // Load and process audio (blocking ffmpeg subprocess)
    let processing = options.audio_processing.clone();
    let audio_path = download_result.audio_path.clone();
    let samples = tokio::task::spawn_blocking(move || {
        audio::load_audio(&audio_path, &processing)
    })
    .await
    .map_err(|e| Error::Transcription(format!("audio loading task failed: {e}")))??;

    // Transcribe (blocking CPU-intensive whisper inference)
    let options_clone = options.clone();
    let samples_clone = samples.clone();
    #[allow(unused_mut)]
    let mut transcript = tokio::task::spawn_blocking(move || {
        transcribe::transcribe_samples(&samples_clone, &model_path, &options_clone)
    })
    .await
    .map_err(|e| Error::Transcription(format!("transcription task failed: {e}")))??;

    // Speaker identification pass (if enabled)
    #[cfg(feature = "speaker-id")]
    if options.speaker_identification {
        run_speaker_identification(&mut transcript, &samples, options).await?;
    }

    // Attach source metadata
    transcript.source_url = Some(url.to_string());
    transcript.source_title = download_result.title;

    Ok(transcript)
}

/// Run speaker identification on a completed transcript.
#[cfg(feature = "speaker-id")]
async fn run_speaker_identification(
    transcript: &mut Transcript,
    samples: &[f32],
    options: &TranscribeOptions,
) -> Result<()> {
    let cache_dir = options.resolve_cache_dir();
    let model_path = match &options.speaker_model_path {
        Some(p) => p.clone(),
        None => speaker::ensure_speaker_model(&cache_dir).await?,
    };
    let profiles_path = options
        .speaker_profiles_path
        .clone()
        .unwrap_or_else(speaker::default_profiles_path);

    let threshold = options.speaker_threshold;
    let samples = samples.to_vec();
    let mut segments = std::mem::take(&mut transcript.segments);

    // Speaker ID is CPU/GPU bound — run in blocking context
    let segments = tokio::task::spawn_blocking(move || -> Result<Vec<types::Segment>> {
        let mut identifier = speaker::SpeakerIdentifier::new(
            &model_path,
            &profiles_path,
            threshold,
            &[speaker::ExecutionProvider::Cpu],
        )?;
        identifier.identify_segments(&mut segments, &samples)?;
        Ok(segments)
    })
    .await
    .map_err(|e| Error::Transcription(format!("speaker identification task failed: {e}")))??;

    transcript.segments = segments;
    Ok(())
}

/// RAII guard that removes an entire temp directory when dropped.
#[cfg(feature = "download")]
struct TempDirGuard<'a>(&'a std::path::Path);

#[cfg(feature = "download")]
impl Drop for TempDirGuard<'_> {
    fn drop(&mut self) {
        if self.0.exists() {
            if let Err(e) = std::fs::remove_dir_all(self.0) {
                tracing::warn!(path = %self.0.display(), error = %e, "failed to clean up temp dir");
            }
        }
    }
}
