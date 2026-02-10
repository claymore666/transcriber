use std::fmt;
use std::path::PathBuf;

use crate::error::Error;

/// A validated language for whisper transcription.
///
/// Wraps a language code that has been verified against whisper.cpp's
/// supported language list (100 languages). Accepts both short codes ("en", "de")
/// and full names ("english", "german").
///
/// Use `Language::Auto` for automatic detection, or `Language::new("en")` for
/// a specific language.
#[derive(Debug, Clone)]
pub enum Language {
    /// Auto-detect language from audio.
    Auto,
    /// A validated language code (e.g. "en", "de", "ja").
    Code {
        /// Short code as whisper expects it.
        code: String,
        /// Whisper internal language ID.
        id: i32,
    },
}

impl Language {
    /// Create a language from a code or full name, validating against whisper.cpp.
    ///
    /// Accepts short codes ("en", "de", "fr") or full names ("english", "german", "french").
    /// Returns an error if the language is not supported.
    pub fn new(lang: &str) -> Result<Self, Error> {
        let lower = lang.to_lowercase();
        if lower == "auto" {
            return Ok(Language::Auto);
        }

        match whisper_rs::get_lang_id(&lower) {
            Some(id) => {
                // Normalize to short code
                let code = whisper_rs::get_lang_str(id)
                    .unwrap_or(&lower)
                    .to_string();
                Ok(Language::Code { code, id })
            }
            None => Err(Error::UnsupportedLanguage(lang.to_string())),
        }
    }

    /// Get the short language code (e.g. "en"), or None for Auto.
    pub fn code(&self) -> Option<&str> {
        match self {
            Language::Auto => None,
            Language::Code { code, .. } => Some(code),
        }
    }

    /// Whether this is auto-detection mode.
    pub fn is_auto(&self) -> bool {
        matches!(self, Language::Auto)
    }

    /// List all supported languages as (code, full_name) pairs.
    pub fn supported() -> Vec<(&'static str, &'static str)> {
        let max = whisper_rs::get_lang_max_id();
        (0..=max)
            .filter_map(|id| {
                let code = whisper_rs::get_lang_str(id)?;
                let name = whisper_rs::get_lang_str_full(id)?;
                Some((code, name))
            })
            .collect()
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::Auto => write!(f, "auto"),
            Language::Code { code, .. } => write!(f, "{code}"),
        }
    }
}

impl Default for Language {
    fn default() -> Self {
        Language::Auto
    }
}

/// Whisper model sizes.
#[derive(Debug, Clone)]
pub enum Model {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    /// User-provided .ggml file path.
    Custom(PathBuf),
}

impl Model {
    /// Model filename as used by HuggingFace / whisper.cpp.
    pub fn filename(&self) -> String {
        match self {
            Model::Tiny => "ggml-tiny.bin".into(),
            Model::TinyEn => "ggml-tiny.en.bin".into(),
            Model::Base => "ggml-base.bin".into(),
            Model::BaseEn => "ggml-base.en.bin".into(),
            Model::Small => "ggml-small.bin".into(),
            Model::SmallEn => "ggml-small.en.bin".into(),
            Model::Medium => "ggml-medium.bin".into(),
            Model::MediumEn => "ggml-medium.en.bin".into(),
            Model::LargeV2 => "ggml-large-v2.bin".into(),
            Model::LargeV3 => "ggml-large-v3.bin".into(),
            Model::LargeV3Turbo => "ggml-large-v3-turbo.bin".into(),
            Model::Custom(path) => path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| "custom-model".into()),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            Model::Tiny => "tiny",
            Model::TinyEn => "tiny.en",
            Model::Base => "base",
            Model::BaseEn => "base.en",
            Model::Small => "small",
            Model::SmallEn => "small.en",
            Model::Medium => "medium",
            Model::MediumEn => "medium.en",
            Model::LargeV2 => "large-v2",
            Model::LargeV3 => "large-v3",
            Model::LargeV3Turbo => "large-v3-turbo",
            Model::Custom(_) => "custom",
        }
    }

    /// Parse from string (e.g. CLI argument).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "tiny" => Some(Model::Tiny),
            "tiny.en" => Some(Model::TinyEn),
            "base" => Some(Model::Base),
            "base.en" => Some(Model::BaseEn),
            "small" => Some(Model::Small),
            "small.en" => Some(Model::SmallEn),
            "medium" => Some(Model::Medium),
            "medium.en" => Some(Model::MediumEn),
            "large-v2" => Some(Model::LargeV2),
            "large-v3" => Some(Model::LargeV3),
            "large-v3-turbo" => Some(Model::LargeV3Turbo),
            _ => None,
        }
    }
}

/// Audio processing options.
///
/// By default all processing steps are **off** — the raw decoded/resampled PCM
/// is passed straight to whisper, which is what the proven brewery pipeline does.
/// Enable individual steps only when you know the source material needs it
/// (e.g. recordings with DC bias, wildly varying levels, or long silence padding).
pub struct AudioProcessing {
    /// Remove DC offset by subtracting the sample mean.
    pub dc_offset_removal: bool,
    /// Peak-normalize samples to [-1.0, 1.0].
    pub normalize: bool,
    /// Trim leading/trailing silence.
    pub trim_silence: bool,
    /// RMS threshold in dB for silence detection (default -40 dB).
    /// Only used when `trim_silence` is true.
    pub silence_threshold_db: f32,
    /// Padding in milliseconds to keep around detected speech boundaries.
    /// Prevents clipping speech onset/offset. Only used when `trim_silence` is true.
    pub silence_pad_ms: u32,
}

impl Default for AudioProcessing {
    fn default() -> Self {
        Self {
            dc_offset_removal: false,
            normalize: false,
            trim_silence: false,
            silence_threshold_db: -40.0,
            silence_pad_ms: 50,
        }
    }
}

impl AudioProcessing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dc_offset_removal(mut self, enabled: bool) -> Self {
        self.dc_offset_removal = enabled;
        self
    }

    pub fn normalize(mut self, enabled: bool) -> Self {
        self.normalize = enabled;
        self
    }

    pub fn trim_silence(mut self, enabled: bool) -> Self {
        self.trim_silence = enabled;
        self
    }

    pub fn silence_threshold_db(mut self, db: f32) -> Self {
        self.silence_threshold_db = db;
        self
    }

    pub fn silence_pad_ms(mut self, ms: u32) -> Self {
        self.silence_pad_ms = ms;
        self
    }

    /// Enable all processing steps (DC offset removal, normalization, silence trimming).
    pub fn all() -> Self {
        Self {
            dc_offset_removal: true,
            normalize: true,
            trim_silence: true,
            ..Self::default()
        }
    }
}

/// Builder for transcription options.
pub struct TranscribeOptions {
    pub model: Model,
    pub language: Language,
    pub translate: bool,
    pub word_timestamps: bool,
    pub diarize: bool,
    pub n_threads: Option<u32>,
    pub gpu: bool,
    pub gpu_device: u32,
    pub vad: bool,
    pub temperature: f32,
    pub beam_size: Option<u32>,
    pub cache_dir: Option<PathBuf>,
    pub audio_processing: AudioProcessing,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            model: Model::LargeV3,
            language: Language::Auto,
            translate: false,
            word_timestamps: false,
            diarize: false,
            n_threads: None,
            gpu: true,
            gpu_device: 0,
            vad: true,
            temperature: 0.0,
            beam_size: None,
            cache_dir: None,
            audio_processing: AudioProcessing::default(),
        }
    }
}

impl TranscribeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    /// Set the language. Validates against whisper's supported languages.
    /// Accepts codes ("en", "de") or full names ("english", "german").
    pub fn language(mut self, lang: &str) -> Result<Self, Error> {
        self.language = Language::new(lang)?;
        Ok(self)
    }

    pub fn translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    pub fn word_timestamps(mut self, enabled: bool) -> Self {
        self.word_timestamps = enabled;
        self
    }

    pub fn diarize(mut self, enabled: bool) -> Self {
        self.diarize = enabled;
        self
    }

    pub fn n_threads(mut self, n: u32) -> Self {
        self.n_threads = Some(n);
        self
    }

    pub fn gpu(mut self, enabled: bool) -> Self {
        self.gpu = enabled;
        self
    }

    pub fn gpu_device(mut self, device: u32) -> Self {
        self.gpu_device = device;
        self
    }

    pub fn vad(mut self, enabled: bool) -> Self {
        self.vad = enabled;
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn beam_size(mut self, size: u32) -> Self {
        self.beam_size = Some(size);
        self
    }

    pub fn cache_dir(mut self, dir: PathBuf) -> Self {
        self.cache_dir = Some(dir);
        self
    }

    pub fn audio_processing(mut self, ap: AudioProcessing) -> Self {
        self.audio_processing = ap;
        self
    }

    /// Resolve the cache directory, defaulting to ~/.cache/transkribo/models.
    pub fn resolve_cache_dir(&self) -> PathBuf {
        self.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("transkribo")
                .join("models")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Language tests ---

    #[test]
    fn test_language_auto() {
        let lang = Language::Auto;
        assert!(lang.is_auto());
        assert_eq!(lang.code(), None);
        assert_eq!(lang.to_string(), "auto");
    }

    #[test]
    fn test_language_from_code() {
        let lang = Language::new("en").unwrap();
        assert!(!lang.is_auto());
        assert_eq!(lang.code(), Some("en"));
        assert_eq!(lang.to_string(), "en");
    }

    #[test]
    fn test_language_from_full_name() {
        let lang = Language::new("german").unwrap();
        assert_eq!(lang.code(), Some("de"));
    }

    #[test]
    fn test_language_from_full_name_english() {
        let lang = Language::new("english").unwrap();
        assert_eq!(lang.code(), Some("en"));
    }

    #[test]
    fn test_language_auto_string() {
        let lang = Language::new("auto").unwrap();
        assert!(lang.is_auto());
    }

    #[test]
    fn test_language_case_insensitive() {
        let lang = Language::new("EN").unwrap();
        assert_eq!(lang.code(), Some("en"));

        let lang = Language::new("German").unwrap();
        assert_eq!(lang.code(), Some("de"));
    }

    #[test]
    fn test_language_invalid() {
        let result = Language::new("klingon");
        assert!(result.is_err());
    }

    #[test]
    fn test_language_invalid_empty() {
        let result = Language::new("");
        assert!(result.is_err());
    }

    #[test]
    fn test_language_supported_list() {
        let supported = Language::supported();
        assert!(supported.len() >= 50); // whisper supports ~100 languages
        assert!(supported.iter().any(|(code, _)| *code == "en"));
        assert!(supported.iter().any(|(code, _)| *code == "de"));
        assert!(supported.iter().any(|(code, _)| *code == "fr"));
        assert!(supported.iter().any(|(code, _)| *code == "ja"));
        assert!(supported.iter().any(|(code, _)| *code == "zh"));
    }

    #[test]
    fn test_language_supported_has_names() {
        let supported = Language::supported();
        let en = supported.iter().find(|(code, _)| *code == "en").unwrap();
        assert_eq!(en.1, "english");
    }

    #[test]
    fn test_language_all_codes_roundtrip() {
        // Every code from supported() should be valid
        for (code, _) in Language::supported() {
            let lang = Language::new(code)
                .unwrap_or_else(|_| panic!("supported code '{}' should be valid", code));
            assert_eq!(lang.code(), Some(code));
        }
    }

    #[test]
    fn test_language_default_is_auto() {
        let lang = Language::default();
        assert!(lang.is_auto());
    }

    // --- Model tests ---

    #[test]
    fn test_model_from_str() {
        assert!(matches!(Model::from_str("tiny"), Some(Model::Tiny)));
        assert!(matches!(Model::from_str("large-v3"), Some(Model::LargeV3)));
        assert!(matches!(Model::from_str("large-v3-turbo"), Some(Model::LargeV3Turbo)));
        assert!(Model::from_str("nonexistent").is_none());
    }

    #[test]
    fn test_model_filename() {
        assert_eq!(Model::Tiny.filename(), "ggml-tiny.bin");
        assert_eq!(Model::LargeV3.filename(), "ggml-large-v3.bin");
        assert_eq!(Model::BaseEn.filename(), "ggml-base.en.bin");
    }

    #[test]
    fn test_model_name() {
        assert_eq!(Model::Tiny.name(), "tiny");
        assert_eq!(Model::LargeV3.name(), "large-v3");
        assert_eq!(Model::Custom(PathBuf::from("/tmp/model.bin")).name(), "custom");
    }

    #[test]
    fn test_model_custom_filename() {
        let model = Model::Custom(PathBuf::from("/path/to/my-model.ggml"));
        assert_eq!(model.filename(), "my-model.ggml");
    }

    #[test]
    fn test_all_models_roundtrip() {
        let names = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo",
        ];
        for name in names {
            let model = Model::from_str(name)
                .unwrap_or_else(|| panic!("model '{}' should parse", name));
            assert_eq!(model.name(), name);
        }
    }

    // --- AudioProcessing tests ---

    #[test]
    fn test_audio_processing_default_all_off() {
        let ap = AudioProcessing::default();
        assert!(!ap.dc_offset_removal);
        assert!(!ap.normalize);
        assert!(!ap.trim_silence);
        assert_eq!(ap.silence_threshold_db, -40.0);
        assert_eq!(ap.silence_pad_ms, 50);
    }

    #[test]
    fn test_audio_processing_all() {
        let ap = AudioProcessing::all();
        assert!(ap.dc_offset_removal);
        assert!(ap.normalize);
        assert!(ap.trim_silence);
    }

    #[test]
    fn test_audio_processing_builder() {
        let ap = AudioProcessing::new()
            .dc_offset_removal(true)
            .silence_threshold_db(-30.0)
            .silence_pad_ms(100);
        assert!(ap.dc_offset_removal);
        assert!(!ap.normalize);
        assert!(!ap.trim_silence);
        assert_eq!(ap.silence_threshold_db, -30.0);
        assert_eq!(ap.silence_pad_ms, 100);
    }

    // --- TranscribeOptions tests ---

    #[test]
    fn test_options_defaults() {
        let opts = TranscribeOptions::default();
        assert!(opts.language.is_auto());
        assert!(opts.gpu);
        assert!(opts.vad);
        assert_eq!(opts.temperature, 0.0);
        assert!(!opts.translate);
        assert!(!opts.word_timestamps);
        assert!(opts.beam_size.is_none());
        assert!(opts.n_threads.is_none());
    }

    #[test]
    fn test_options_builder_chain() {
        let opts = TranscribeOptions::new()
            .model(Model::Tiny)
            .translate(true)
            .word_timestamps(true)
            .gpu(false)
            .vad(false)
            .temperature(0.5)
            .beam_size(5)
            .n_threads(4);

        assert!(matches!(opts.model, Model::Tiny));
        assert!(opts.translate);
        assert!(opts.word_timestamps);
        assert!(!opts.gpu);
        assert!(!opts.vad);
        assert_eq!(opts.temperature, 0.5);
        assert_eq!(opts.beam_size, Some(5));
        assert_eq!(opts.n_threads, Some(4));
    }

    #[test]
    fn test_options_language_validation() {
        let opts = TranscribeOptions::new().language("en");
        assert!(opts.is_ok());

        let opts = TranscribeOptions::new().language("gibberish");
        assert!(opts.is_err());
    }

    #[test]
    fn test_options_resolve_cache_dir_default() {
        let opts = TranscribeOptions::default();
        let cache = opts.resolve_cache_dir();
        assert!(cache.ends_with("transkribo/models"));
    }

    #[test]
    fn test_options_resolve_cache_dir_custom() {
        let opts = TranscribeOptions::new().cache_dir(PathBuf::from("/tmp/my-models"));
        assert_eq!(opts.resolve_cache_dir(), PathBuf::from("/tmp/my-models"));
    }
}
