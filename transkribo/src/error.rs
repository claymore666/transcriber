use std::path::PathBuf;

/// All errors that can occur in transkribo.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model error: {0}")]
    Model(String),

    #[error("model not found: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("model download failed: {0}")]
    ModelDownload(String),

    #[error("audio decoding error: {0}")]
    AudioDecode(String),

    #[error("audio file not found: {path}")]
    AudioNotFound { path: PathBuf },

    #[error("unsupported language: \"{0}\" — use Language::supported() to list valid codes")]
    UnsupportedLanguage(String),

    #[error("transcription error: {0}")]
    Transcription(String),

    #[error("whisper error: {0}")]
    Whisper(#[from] whisper_rs::WhisperError),

    #[cfg(feature = "download")]
    #[error("download error: {0}")]
    Download(String),

    #[cfg(feature = "download")]
    #[error("yt-dlp not found — install with: pip install yt-dlp")]
    YtDlpNotFound,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
