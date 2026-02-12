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
pub(crate) mod transcribe;
pub mod types;

pub use config::{AudioProcessing, Language, Model, TranscribeOptions};
pub use error::{Error, Result};
pub use types::{Segment, Transcript, Word};

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
    let path = path.as_ref();

    // Ensure model is available
    let cache_dir = options.resolve_cache_dir();
    let model_path = model::ensure_model(&options.model, &cache_dir).await?;

    // Load and process audio
    let samples = audio::load_audio(path, &options.audio_processing)?;

    // Transcribe
    let transcript = transcribe::transcribe_samples(&samples, &model_path, options)?;

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

    // Load and process audio
    let samples = audio::load_audio(&download_result.audio_path, &options.audio_processing)?;

    // Transcribe
    let mut transcript = transcribe::transcribe_samples(&samples, &model_path, options)?;

    // Attach source metadata
    transcript.source_url = Some(url.to_string());
    transcript.source_title = download_result.title;

    Ok(transcript)
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
