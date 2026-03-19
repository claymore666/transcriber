//! Speaker identification via voice embeddings.
//!
//! Uses a wespeaker ONNX model to extract speaker embeddings from audio segments,
//! then matches them against enrolled speaker profiles using cosine similarity.

pub mod audio_slice;
pub mod embedding;
pub mod profile;

use std::path::{Path, PathBuf};

use ort::session::Session;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};
use crate::types::Segment;

pub use profile::{cosine_similarity, ProfileStore, SpeakerProfile};

/// GPU backend for speaker embedding inference.
#[derive(Debug, Clone)]
pub enum ExecutionProvider {
    Cpu,
    Cuda { device_id: u32 },
}

/// Result of matching an embedding against speaker profiles.
#[derive(Debug, Clone)]
pub struct SpeakerMatch {
    pub name: String,
    pub confidence: f32,
    pub is_known: bool,
}

/// Speaker identification engine.
///
/// Holds a loaded ONNX model session and enrolled speaker profiles.
/// Use [`SpeakerIdentifier::new`] to load from disk, then
/// [`SpeakerIdentifier::identify_segments`] to annotate transcript segments.
pub struct SpeakerIdentifier {
    session: Session,
    profiles: ProfileStore,
    threshold: f32,
}

impl SpeakerIdentifier {
    /// Load the speaker embedding model and speaker profiles.
    pub fn new(
        model_path: &Path,
        profiles_path: &Path,
        threshold: f32,
        execution_providers: &[ExecutionProvider],
    ) -> Result<Self> {
        let session = embedding::create_session(model_path, execution_providers)?;
        let profiles = ProfileStore::load(profiles_path)?;

        info!(
            threshold,
            profiles = profiles.profiles.len(),
            "speaker identifier ready"
        );

        Ok(Self {
            session,
            profiles,
            threshold,
        })
    }

    /// Extract a speaker embedding from raw audio samples (16kHz mono f32).
    pub fn embed(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        embedding::extract_embedding(&mut self.session, audio)
    }

    /// Match an embedding against all enrolled profiles.
    pub fn identify(&self, embedding_vec: &[f32]) -> SpeakerMatch {
        let mut best_name = "Unknown".to_string();
        let mut best_score = 0.0f32;

        for profile in &self.profiles.profiles {
            let score = cosine_similarity(embedding_vec, &profile.centroid);
            if score > best_score {
                best_score = score;
                best_name = profile.short_name.clone();
            }
        }

        SpeakerMatch {
            name: if best_score >= self.threshold {
                best_name
            } else {
                "Unknown".into()
            },
            confidence: best_score,
            is_known: best_score >= self.threshold,
        }
    }

    /// Enroll a new speaker from audio samples.
    /// The embedding is computed and added to the profile store.
    pub fn enroll(&mut self, name: &str, audio: &[f32]) -> Result<()> {
        let embedding_vec = self.embed(audio)?;
        self.profiles.enroll(name, embedding_vec);
        Ok(())
    }

    /// Save the current profiles to disk.
    pub fn save_profiles(&self, path: &Path) -> Result<()> {
        self.profiles.save(path)
    }

    /// Get a reference to the profile store.
    pub fn profiles(&self) -> &ProfileStore {
        &self.profiles
    }

    /// Process all segments: assign speaker_id and speaker_confidence to each.
    ///
    /// Segments shorter than 1 second or with high no_speech_probability are skipped.
    /// The full audio buffer (16kHz mono f32) must be the same buffer that was
    /// used for transcription.
    pub fn identify_segments(
        &mut self,
        segments: &mut [Segment],
        full_audio: &[f32],
    ) -> Result<()> {
        if self.profiles.profiles.is_empty() {
            warn!("no speaker profiles enrolled — skipping identification");
            return Ok(());
        }

        let mut identified = 0u32;
        let mut unknown = 0u32;
        let mut skipped = 0u32;

        for seg in segments.iter_mut() {
            // Skip segments with high no-speech probability
            if seg.no_speech_probability > 0.6 {
                skipped += 1;
                continue;
            }

            // Extract audio slice for this segment
            let slice = match audio_slice::extract_slice(full_audio, seg.start, seg.end) {
                Some(s) => s,
                None => {
                    skipped += 1;
                    continue;
                }
            };

            // Compute embedding and match
            match self.embed(slice) {
                Ok(emb) => {
                    let m = self.identify(&emb);
                    seg.speaker_id = Some(m.name);
                    seg.speaker_confidence = Some(m.confidence);
                    if m.is_known {
                        identified += 1;
                    } else {
                        unknown += 1;
                    }
                }
                Err(e) => {
                    debug!(
                        start = seg.start,
                        end = seg.end,
                        error = %e,
                        "failed to embed segment, skipping"
                    );
                    skipped += 1;
                }
            }
        }

        info!(identified, unknown, skipped, "speaker identification complete");

        // Apply temporal smoothing
        smooth_speaker_labels(segments, self.threshold);

        Ok(())
    }
}

/// Temporal smoothing: if segment N is "Unknown" but N-1 and N+1 are the same
/// known speaker, assign N to that speaker too. Reduces flickering.
fn smooth_speaker_labels(segments: &mut [Segment], _threshold: f32) {
    if segments.len() < 3 {
        return;
    }

    // Collect smoothing decisions first to avoid borrow issues
    let mut updates: Vec<(usize, String, f32)> = Vec::new();

    for i in 1..segments.len() - 1 {
        let prev_id = segments[i - 1].speaker_id.as_deref();
        let curr_id = segments[i].speaker_id.as_deref();
        let next_id = segments[i + 1].speaker_id.as_deref();

        // If current is Unknown or None, and neighbors agree on a known speaker
        let is_unknown = curr_id.is_none() || curr_id == Some("Unknown");
        if is_unknown {
            if let (Some(prev), Some(next)) = (prev_id, next_id) {
                if prev == next && prev != "Unknown" {
                    // Inherit neighbor's confidence (take the lower one)
                    let conf = match (
                        segments[i - 1].speaker_confidence,
                        segments[i + 1].speaker_confidence,
                    ) {
                        (Some(a), Some(b)) => a.min(b),
                        (Some(a), None) | (None, Some(a)) => a,
                        (None, None) => 0.0,
                    };
                    updates.push((i, prev.to_string(), conf));
                }
            }
        }
    }

    let smoothed = updates.len();
    for (i, name, conf) in updates {
        segments[i].speaker_id = Some(name);
        segments[i].speaker_confidence = Some(conf);
    }

    if smoothed > 0 {
        debug!(smoothed, "applied temporal smoothing");
    }
}

/// Default path for the wespeaker ONNX model.
pub fn default_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("transcriber")
        .join("models")
        .join("wespeaker_en_voxceleb_CAM++.onnx")
}

/// Default path for speaker profiles.
pub fn default_profiles_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("transcriber")
        .join("speakers.json")
}

/// Download the wespeaker model if not already cached.
pub async fn ensure_speaker_model(cache_dir: &Path) -> Result<PathBuf> {
    let model_path = cache_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    if model_path.exists() {
        info!(path = %model_path.display(), "speaker model already cached");
        return Ok(model_path);
    }

    let url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx";
    info!(%url, "downloading speaker embedding model");

    std::fs::create_dir_all(cache_dir)?;

    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await?
        .error_for_status()
        .map_err(|e| Error::SpeakerId(format!("failed to download speaker model: {e}")))?;

    let bytes = response.bytes().await?;

    if bytes.len() < 1_000_000 {
        return Err(Error::SpeakerId(format!(
            "downloaded speaker model too small ({} bytes) — likely an error page",
            bytes.len()
        )));
    }

    std::fs::write(&model_path, &bytes)?;
    info!(
        path = %model_path.display(),
        size = bytes.len(),
        "speaker model saved"
    );

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(start: f64, end: f64, speaker: Option<&str>, conf: Option<f32>) -> Segment {
        Segment {
            start,
            end,
            text: "test".into(),
            speaker_turn: false,
            no_speech_probability: 0.0,
            words: None,
            speaker_id: speaker.map(|s| s.to_string()),
            speaker_confidence: conf,
        }
    }

    #[test]
    fn test_smooth_fills_unknown_gap() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, Some("Unknown"), Some(0.3)),
            make_segment(4.0, 6.0, Some("Alice"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_smooth_no_change_different_neighbors() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, Some("Unknown"), Some(0.3)),
            make_segment(4.0, 6.0, Some("Bob"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Unknown"));
    }

    #[test]
    fn test_smooth_no_change_known_speaker() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, Some("Bob"), Some(0.8)),
            make_segment(4.0, 6.0, Some("Alice"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        // Bob should not be overwritten — he's a known speaker, not Unknown
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_smooth_short_input() {
        let mut segments = vec![make_segment(0.0, 2.0, Some("Alice"), Some(0.9))];
        smooth_speaker_labels(&mut segments, 0.6); // should not panic
        assert_eq!(segments[0].speaker_id.as_deref(), Some("Alice"));
    }
}
