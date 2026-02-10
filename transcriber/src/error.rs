use std::path::PathBuf;

/// All errors that can occur in transcriber.
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

    #[error("invalid option: {0}")]
    InvalidOption(String),

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_model() {
        let e = Error::Model("bad model".into());
        assert_eq!(e.to_string(), "model error: bad model");
    }

    #[test]
    fn test_error_display_model_not_found() {
        let e = Error::ModelNotFound {
            path: PathBuf::from("/tmp/model.bin"),
        };
        assert!(e.to_string().contains("/tmp/model.bin"));
    }

    #[test]
    fn test_error_display_audio_not_found() {
        let e = Error::AudioNotFound {
            path: PathBuf::from("/tmp/audio.wav"),
        };
        assert!(e.to_string().contains("/tmp/audio.wav"));
    }

    #[test]
    fn test_error_display_unsupported_language() {
        let e = Error::UnsupportedLanguage("klingon".into());
        let msg = e.to_string();
        assert!(msg.contains("klingon"));
        assert!(msg.contains("Language::supported()"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let e: Error = io_err.into();
        assert!(matches!(e, Error::Io(_)));
        assert!(e.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_from_json() {
        let json_err = serde_json::from_str::<String>("invalid json").unwrap_err();
        let e: Error = json_err.into();
        assert!(matches!(e, Error::Json(_)));
    }

    #[test]
    fn test_error_debug_impl() {
        let e = Error::AudioDecode("test error".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("AudioDecode"));
    }
}
