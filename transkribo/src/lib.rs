pub mod audio;
pub mod config;
#[cfg(feature = "download")]
pub mod download;
pub mod error;
pub mod model;
pub mod transcribe;
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
    let tmp_dir = std::env::temp_dir().join("transkribo");
    let download_result = download::download_audio(url, &tmp_dir).await?;

    // Clean up downloaded file on all exit paths (success or error)
    let _cleanup = CleanupGuard(&download_result.audio_path);

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

/// RAII guard that removes a file when dropped.
#[cfg(feature = "download")]
struct CleanupGuard<'a>(&'a std::path::Path);

#[cfg(feature = "download")]
impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_file(self.0) {
            tracing::warn!(path = %self.0.display(), error = %e, "failed to clean up temp file");
        }
    }
}
