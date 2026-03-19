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
    /// After identification, short segments are merged with neighbors and temporal
    /// smoothing is applied to reduce speaker flickering.
    ///
    /// Returns a [`SpeakerSummary`] with identification statistics.
    pub fn identify_segments(
        &mut self,
        segments: &mut [Segment],
        full_audio: &[f32],
    ) -> Result<SpeakerSummary> {
        if self.profiles.profiles.is_empty() {
            warn!("no speaker profiles enrolled — skipping identification");
            return Ok(SpeakerSummary::default());
        }

        let mut identified = 0u32;
        let mut unknown = 0u32;
        let mut skipped = 0u32;

        // Collect embeddings for re-enrollment suggestions
        let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(segments.len());

        for seg in segments.iter_mut() {
            // Skip segments with high no-speech probability
            if seg.no_speech_probability > 0.6 {
                skipped += 1;
                embeddings.push(None);
                continue;
            }

            // Extract audio slice for this segment
            let slice = match audio_slice::extract_slice(full_audio, seg.start, seg.end) {
                Some(s) => s,
                None => {
                    skipped += 1;
                    embeddings.push(None);
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
                    embeddings.push(Some(emb));
                }
                Err(e) => {
                    debug!(
                        start = seg.start,
                        end = seg.end,
                        error = %e,
                        "failed to embed segment, skipping"
                    );
                    skipped += 1;
                    embeddings.push(None);
                }
            }
        }

        info!(identified, unknown, skipped, "speaker identification complete");

        // Post-processing: merge short segments, then smooth
        let merged = merge_short_segments(segments);
        let smoothed = smooth_speaker_labels(segments, self.threshold);

        // Cluster unknown segments for re-enrollment suggestions
        let unknown_clusters = cluster_unknowns(segments, &embeddings);

        Ok(SpeakerSummary {
            identified,
            unknown,
            skipped,
            merged,
            smoothed,
            unknown_clusters,
        })
    }
}

/// Statistics from speaker identification, including post-processing counts.
#[derive(Debug, Clone, Default)]
pub struct SpeakerSummary {
    /// Segments matched to a known speaker.
    pub identified: u32,
    /// Segments below threshold (attributed as "Unknown").
    pub unknown: u32,
    /// Segments skipped (too short, no-speech, or embedding failure).
    pub skipped: u32,
    /// Segments merged from short unidentified neighbors.
    pub merged: u32,
    /// Segments reassigned by temporal smoothing.
    pub smoothed: u32,
    /// Clusters of unknown segments that may be the same unregistered speaker.
    pub unknown_clusters: Vec<UnknownCluster>,
}

/// A cluster of unknown segments that appear to be the same unregistered speaker.
#[derive(Debug, Clone)]
pub struct UnknownCluster {
    /// Number of segments in this cluster.
    pub segment_count: usize,
    /// Total speaking duration in seconds.
    pub total_duration: f64,
    /// Start time of the most representative segment (closest to centroid).
    pub representative_start: f64,
    /// End time of the most representative segment.
    pub representative_end: f64,
}

/// Merge short segments that were skipped during identification.
///
/// Segments with `speaker_id == None` and duration < MIN_SEGMENT_DURATION that
/// don't have high no-speech probability inherit their speaker from the nearest
/// neighbor. Prefers the previous segment (speakers tend to continue).
///
/// Returns the number of segments merged.
fn merge_short_segments(segments: &mut [Segment]) -> u32 {
    let mut merged = 0u32;

    for i in 0..segments.len() {
        // Only target segments that have no speaker_id (skipped due to short duration)
        if segments[i].speaker_id.is_some() {
            continue;
        }

        let duration = segments[i].end - segments[i].start;
        if duration >= audio_slice::MIN_SEGMENT_DURATION {
            continue;
        }

        // Don't merge no-speech segments
        if segments[i].no_speech_probability > 0.6 {
            continue;
        }

        // Try previous segment first, then next
        let donor = if i > 0 && segments[i - 1].speaker_id.is_some() {
            Some(i - 1)
        } else if i + 1 < segments.len() && segments[i + 1].speaker_id.is_some() {
            Some(i + 1)
        } else {
            None
        };

        if let Some(d) = donor {
            let name = segments[d].speaker_id.clone().unwrap();
            let conf = segments[d].speaker_confidence;
            segments[i].speaker_id = Some(name);
            segments[i].speaker_confidence = conf;
            merged += 1;
        }
    }

    if merged > 0 {
        debug!(merged, "merged short segments with neighbors");
    }
    merged
}

/// Maximum duration in seconds for a segment to be eligible for smoothing
/// when it has a low-confidence attribution that differs from its neighbors.
const SMOOTHING_MAX_DURATION: f64 = 2.0;

/// Temporal smoothing: reduces speaker flickering.
///
/// Two smoothing rules applied over multiple passes:
/// 1. If segment N is Unknown/None and N-1 and N+1 agree on a known speaker,
///    assign N to that speaker.
/// 2. If segment N is short (< 2s), has low confidence, and both neighbors
///    agree on a different known speaker, reassign N to the neighbor's speaker.
///
/// Returns the total number of segments smoothed.
fn smooth_speaker_labels(segments: &mut [Segment], threshold: f32) -> u32 {
    if segments.len() < 3 {
        return 0;
    }

    let weak_confidence = threshold + 0.1;
    let mut total_smoothed = 0u32;

    // Run up to 3 passes to propagate corrections
    for pass in 0..3 {
        let mut updates: Vec<(usize, String, f32)> = Vec::new();

        for i in 1..segments.len() - 1 {
            let prev_id = segments[i - 1].speaker_id.as_deref();
            let curr_id = segments[i].speaker_id.as_deref();
            let next_id = segments[i + 1].speaker_id.as_deref();

            let (Some(prev), Some(next)) = (prev_id, next_id) else {
                continue;
            };

            // Both neighbors must agree and be known speakers
            if prev != next || prev == "Unknown" {
                continue;
            }

            let neighbor_conf = match (
                segments[i - 1].speaker_confidence,
                segments[i + 1].speaker_confidence,
            ) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => 0.0,
            };

            let is_unknown = curr_id.is_none() || curr_id == Some("Unknown");

            if is_unknown {
                // Rule 1: Unknown between two same known speakers
                updates.push((i, prev.to_string(), neighbor_conf));
            } else if curr_id != Some(prev) {
                // Rule 2: Short, low-confidence segment attributed to wrong speaker
                let duration = segments[i].end - segments[i].start;
                let conf = segments[i].speaker_confidence.unwrap_or(0.0);
                if duration < SMOOTHING_MAX_DURATION && conf < weak_confidence {
                    updates.push((i, prev.to_string(), neighbor_conf));
                }
            }
        }

        if updates.is_empty() {
            break;
        }

        let pass_count = updates.len() as u32;
        for (i, name, conf) in updates {
            segments[i].speaker_id = Some(name);
            segments[i].speaker_confidence = Some(conf);
        }

        total_smoothed += pass_count;
        debug!(pass = pass + 1, smoothed = pass_count, "smoothing pass");
    }

    if total_smoothed > 0 {
        debug!(total_smoothed, "applied temporal smoothing");
    }
    total_smoothed
}

/// Cluster unknown segments by embedding similarity to detect potential
/// unregistered speakers.
///
/// Uses simple single-linkage agglomerative clustering with cosine similarity
/// threshold of 0.7. Only reports clusters with >= 5 segments and >= 30s total.
fn cluster_unknowns(
    segments: &[Segment],
    embeddings: &[Option<Vec<f32>>],
) -> Vec<UnknownCluster> {
    // Collect unknown segments with their embeddings
    let mut unknowns: Vec<(usize, &Vec<f32>)> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if seg.speaker_id.as_deref() == Some("Unknown") {
            if let Some(Some(emb)) = embeddings.get(i) {
                unknowns.push((i, emb));
            }
        }
    }

    if unknowns.len() < 5 {
        return Vec::new();
    }

    // Simple greedy clustering: assign each unknown to the first cluster
    // where cosine similarity to centroid >= 0.7, or create a new cluster.
    const CLUSTER_THRESHOLD: f32 = 0.7;

    struct Cluster {
        members: Vec<usize>,      // indices into unknowns
        centroid: Vec<f32>,        // running average (unnormalized sum / count)
    }

    let mut clusters: Vec<Cluster> = Vec::new();

    for (ui, (_seg_idx, emb)) in unknowns.iter().enumerate() {
        let mut best_cluster = None;
        let mut best_sim = 0.0f32;

        for (ci, cluster) in clusters.iter().enumerate() {
            let sim = cosine_similarity(emb, &cluster.centroid);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = Some(ci);
            }
        }

        if best_sim >= CLUSTER_THRESHOLD {
            let ci = best_cluster.unwrap();
            let n = clusters[ci].members.len() as f32;
            // Update centroid as running average
            for (j, &v) in emb.iter().enumerate() {
                if j < clusters[ci].centroid.len() {
                    clusters[ci].centroid[j] = (clusters[ci].centroid[j] * n + v) / (n + 1.0);
                }
            }
            clusters[ci].members.push(ui);
        } else {
            clusters.push(Cluster {
                members: vec![ui],
                centroid: emb.to_vec(),
            });
        }
    }

    // Convert to UnknownCluster, filtering by minimum size
    const MIN_SEGMENTS: usize = 5;
    const MIN_DURATION: f64 = 30.0;

    let mut result = Vec::new();
    for cluster in &clusters {
        if cluster.members.len() < MIN_SEGMENTS {
            continue;
        }

        let mut total_duration = 0.0;
        let mut best_idx = 0usize;
        let mut best_sim = 0.0f32;

        for &ui in &cluster.members {
            let seg_idx = unknowns[ui].0;
            total_duration += segments[seg_idx].end - segments[seg_idx].start;

            // Find most representative segment (closest to centroid)
            let sim = cosine_similarity(unknowns[ui].1, &cluster.centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = unknowns[ui].0;
            }
        }

        if total_duration < MIN_DURATION {
            continue;
        }

        result.push(UnknownCluster {
            segment_count: cluster.members.len(),
            total_duration,
            representative_start: segments[best_idx].start,
            representative_end: segments[best_idx].end,
        });
    }

    // Sort by segment count descending
    result.sort_by(|a, b| b.segment_count.cmp(&a.segment_count));

    if !result.is_empty() {
        info!(
            clusters = result.len(),
            "found unknown speaker clusters for potential enrollment"
        );
    }

    result
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

    // --- Temporal smoothing tests ---

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
    fn test_smooth_no_change_known_speaker_high_conf() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, Some("Bob"), Some(0.8)),
            make_segment(4.0, 6.0, Some("Alice"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        // Bob has high confidence — should not be overwritten
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_smooth_short_input() {
        let mut segments = vec![make_segment(0.0, 2.0, Some("Alice"), Some(0.9))];
        smooth_speaker_labels(&mut segments, 0.6); // should not panic
        assert_eq!(segments[0].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_smooth_short_low_conf_reassigned() {
        // Short segment (1.5s) with low confidence attributed to Bob,
        // sandwiched between Alice segments — should be reassigned
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 3.5, Some("Bob"), Some(0.55)), // short, low conf
            make_segment(3.5, 6.0, Some("Alice"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_smooth_long_segment_not_reassigned() {
        // Long segment (3s) — even with low confidence, not smoothed
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 5.0, Some("Bob"), Some(0.55)), // long, low conf
            make_segment(5.0, 7.0, Some("Alice"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_smooth_multi_pass() {
        // Multi-pass: pass 1 resolves single Unknown gap,
        // pass 2 resolves the newly-created single gap
        // [Alice, Unknown, Alice(low-conf->Bob), Unknown, Alice]
        // Pass 1: seg 1 (Unknown between Alice, Alice) -> Alice
        //         seg 3 (Unknown between Bob, Alice) -> no change
        // But let's use alternating single gaps:
        // [Alice, Unknown, Alice, Unknown, Alice]
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, Some("Unknown"), Some(0.3)),
            make_segment(4.0, 6.0, Some("Alice"), Some(0.85)),
            make_segment(6.0, 8.0, Some("Unknown"), Some(0.3)),
            make_segment(8.0, 10.0, Some("Alice"), Some(0.9)),
        ];

        let count = smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
        assert_eq!(segments[3].speaker_id.as_deref(), Some("Alice"));
        assert_eq!(count, 2);
    }

    // --- Short segment merging tests ---

    #[test]
    fn test_merge_short_inherits_from_previous() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 2.5, None, None), // 0.5s, no speaker
            make_segment(2.5, 5.0, Some("Alice"), Some(0.85)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 1);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_merge_short_inherits_from_next_at_start() {
        let mut segments = vec![
            make_segment(0.0, 0.5, None, None), // 0.5s, start of recording
            make_segment(0.5, 3.0, Some("Bob"), Some(0.8)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 1);
        assert_eq!(segments[0].speaker_id.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_merge_short_skips_no_speech() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            {
                let mut seg = make_segment(2.0, 2.5, None, None);
                seg.no_speech_probability = 0.8; // high no-speech
                seg
            },
            make_segment(2.5, 5.0, Some("Alice"), Some(0.85)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 0);
        assert!(segments[1].speaker_id.is_none());
    }

    #[test]
    fn test_merge_does_not_touch_identified_segments() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 2.5, Some("Bob"), Some(0.7)), // short but already identified
            make_segment(2.5, 5.0, Some("Alice"), Some(0.85)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 0);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Bob"));
    }

    // --- Unknown clustering tests ---

    #[test]
    fn test_cluster_unknowns_no_unknowns() {
        let segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
        ];
        let embeddings = vec![Some(vec![1.0, 0.0, 0.0])];
        let clusters = cluster_unknowns(&segments, &embeddings);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_unknowns_too_few() {
        // 3 unknowns, below the minimum of 5
        let segments: Vec<Segment> = (0..3)
            .map(|i| make_segment(i as f64 * 2.0, i as f64 * 2.0 + 2.0, Some("Unknown"), Some(0.3)))
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..3)
            .map(|_| Some(vec![1.0, 0.0, 0.0]))
            .collect();
        let clusters = cluster_unknowns(&segments, &embeddings);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_unknowns_groups_similar() {
        // 6 similar unknowns with enough total duration
        let segments: Vec<Segment> = (0..6)
            .map(|i| make_segment(i as f64 * 10.0, i as f64 * 10.0 + 8.0, Some("Unknown"), Some(0.3)))
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..6)
            .map(|_| Some(vec![0.5, 0.5, 0.5]))
            .collect();
        let clusters = cluster_unknowns(&segments, &embeddings);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].segment_count, 6);
    }

    // --- Additional smoothing tests ---

    #[test]
    fn test_smooth_high_conf_short_segment_preserved() {
        // Short segment with HIGH confidence — genuine interjection, must not be reassigned
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 3.5, Some("Bob"), Some(0.85)), // short but high conf
            make_segment(3.5, 6.0, Some("Alice"), Some(0.9)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_smooth_above_weak_confidence_preserved() {
        // Confidence clearly above threshold + 0.1 — should NOT be reassigned
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 3.5, Some("Bob"), Some(0.75)), // above weak boundary
            make_segment(3.5, 6.0, Some("Alice"), Some(0.9)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_smooth_below_weak_confidence_reassigned() {
        // Confidence below threshold + 0.1 — should be reassigned
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 3.5, Some("Bob"), Some(0.65)), // below weak boundary
            make_segment(3.5, 6.0, Some("Alice"), Some(0.9)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_smooth_none_speaker_between_same() {
        // Segment with speaker_id = None (not Unknown, just unset) between same speakers
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, None, None),
            make_segment(4.0, 6.0, Some("Alice"), Some(0.85)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_smooth_returns_zero_for_empty() {
        let mut segments: Vec<Segment> = vec![];
        assert_eq!(smooth_speaker_labels(&mut segments, 0.6), 0);
    }

    #[test]
    fn test_smooth_returns_zero_for_two_segments() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, Some("Unknown"), Some(0.3)),
        ];
        assert_eq!(smooth_speaker_labels(&mut segments, 0.6), 0);
    }

    #[test]
    fn test_smooth_does_not_override_with_unknown_neighbors() {
        // Both neighbors are Unknown — should not smooth
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Unknown"), Some(0.3)),
            make_segment(2.0, 4.0, Some("Alice"), Some(0.55)),
            make_segment(4.0, 6.0, Some("Unknown"), Some(0.3)),
        ];

        smooth_speaker_labels(&mut segments, 0.6);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    // --- Additional short segment merging tests ---

    #[test]
    fn test_merge_multiple_consecutive_short_segments() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 2.3, None, None), // 0.3s
            make_segment(2.3, 2.6, None, None), // 0.3s
            make_segment(2.6, 2.9, None, None), // 0.3s
            make_segment(2.9, 5.0, Some("Bob"), Some(0.8)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 3);
        // First short inherits from previous (Alice)
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
        // Second short inherits from previous (now Alice)
        assert_eq!(segments[2].speaker_id.as_deref(), Some("Alice"));
        // Third short inherits from previous (now Alice)
        assert_eq!(segments[3].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_merge_short_at_end_of_recording() {
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 2.5, None, None), // short, at end
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 1);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_merge_short_between_different_speakers() {
        // Short segment between different speakers — inherits from previous
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 2.5, None, None),
            make_segment(2.5, 5.0, Some("Bob"), Some(0.8)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 1);
        assert_eq!(segments[1].speaker_id.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_merge_short_no_neighbors_with_speaker() {
        // All segments have no speaker — nothing to inherit from
        let mut segments = vec![
            make_segment(0.0, 0.5, None, None),
            make_segment(0.5, 0.8, None, None),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 0);
    }

    #[test]
    fn test_merge_does_not_touch_long_unidentified() {
        // Long segment (2s) with no speaker_id — should not be merged
        let mut segments = vec![
            make_segment(0.0, 2.0, Some("Alice"), Some(0.9)),
            make_segment(2.0, 4.0, None, None), // 2s, above MIN_SEGMENT_DURATION
            make_segment(4.0, 6.0, Some("Alice"), Some(0.85)),
        ];

        let merged = merge_short_segments(&mut segments);
        assert_eq!(merged, 0);
        assert!(segments[1].speaker_id.is_none());
    }

    // --- Additional unknown clustering tests ---

    #[test]
    fn test_cluster_unknowns_two_distinct_clusters() {
        // Two groups of unknowns with very different embeddings
        let mut segments = Vec::new();
        let mut embeddings: Vec<Option<Vec<f32>>> = Vec::new();

        // Cluster A: 6 segments with embedding near [1, 0, 0]
        for i in 0..6 {
            segments.push(make_segment(i as f64 * 10.0, i as f64 * 10.0 + 8.0, Some("Unknown"), Some(0.3)));
            embeddings.push(Some(vec![0.95 + (i as f32 * 0.005), 0.05, 0.05]));
        }

        // Cluster B: 6 segments with embedding near [0, 1, 0]
        for i in 0..6 {
            segments.push(make_segment(60.0 + i as f64 * 10.0, 68.0 + i as f64 * 10.0, Some("Unknown"), Some(0.3)));
            embeddings.push(Some(vec![0.05, 0.95 + (i as f32 * 0.005), 0.05]));
        }

        let clusters = cluster_unknowns(&segments, &embeddings);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].segment_count, 6);
        assert_eq!(clusters[1].segment_count, 6);
    }

    #[test]
    fn test_cluster_unknowns_below_min_duration() {
        // 6 very short unknowns — above min segment count but below min total duration (30s)
        let segments: Vec<Segment> = (0..6)
            .map(|i| make_segment(i as f64 * 2.0, i as f64 * 2.0 + 1.5, Some("Unknown"), Some(0.3)))
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..6)
            .map(|_| Some(vec![0.5, 0.5, 0.5]))
            .collect();
        // Total duration = 6 * 1.5 = 9s < 30s minimum
        let clusters = cluster_unknowns(&segments, &embeddings);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_unknowns_ignores_known_speakers() {
        // Mix of known and unknown — only unknowns should be clustered
        let mut segments = Vec::new();
        let mut embeddings: Vec<Option<Vec<f32>>> = Vec::new();

        // 3 known speakers (should be ignored)
        for i in 0..3 {
            segments.push(make_segment(i as f64 * 10.0, i as f64 * 10.0 + 8.0, Some("Alice"), Some(0.9)));
            embeddings.push(Some(vec![0.5, 0.5, 0.5]));
        }

        // 6 unknowns with enough duration
        for i in 0..6 {
            segments.push(make_segment(30.0 + i as f64 * 10.0, 38.0 + i as f64 * 10.0, Some("Unknown"), Some(0.3)));
            embeddings.push(Some(vec![0.5, 0.5, 0.5]));
        }

        let clusters = cluster_unknowns(&segments, &embeddings);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].segment_count, 6);
    }

    #[test]
    fn test_cluster_unknowns_missing_embeddings() {
        // Unknowns without embeddings should be skipped
        let segments: Vec<Segment> = (0..6)
            .map(|i| make_segment(i as f64 * 10.0, i as f64 * 10.0 + 8.0, Some("Unknown"), Some(0.3)))
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..6).map(|_| None).collect();
        let clusters = cluster_unknowns(&segments, &embeddings);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_unknowns_representative_segment() {
        // Verify that representative segment has valid time range
        let segments: Vec<Segment> = (0..6)
            .map(|i| make_segment(i as f64 * 10.0, i as f64 * 10.0 + 8.0, Some("Unknown"), Some(0.3)))
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..6)
            .map(|_| Some(vec![0.5, 0.5, 0.5]))
            .collect();
        let clusters = cluster_unknowns(&segments, &embeddings);
        assert_eq!(clusters.len(), 1);
        // Representative should be one of the segments
        assert!(clusters[0].representative_start >= 0.0);
        assert!(clusters[0].representative_end > clusters[0].representative_start);
        assert!(clusters[0].total_duration > 30.0);
    }
}
