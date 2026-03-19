use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::Result;

/// Versioned container for all speaker profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileStore {
    pub version: u32,
    pub profiles: Vec<SpeakerProfile>,
}

/// A single speaker's voice profile with enrollment embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub name: String,
    pub short_name: String,
    pub enrolled_at: String,
    pub embeddings: Vec<Vec<f32>>,
    pub centroid: Vec<f32>,
}

impl ProfileStore {
    /// Create a new empty profile store.
    pub fn new() -> Self {
        Self {
            version: 1,
            profiles: Vec::new(),
        }
    }

    /// Load profiles from a JSON file. Returns an empty store if the file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            debug!(path = %path.display(), "no profiles file, starting empty");
            return Ok(Self::new());
        }

        let data = std::fs::read_to_string(path)?;
        let store: Self = serde_json::from_str(&data)?;
        info!(
            path = %path.display(),
            profiles = store.profiles.len(),
            "loaded speaker profiles"
        );
        Ok(store)
    }

    /// Save profiles to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        info!(path = %path.display(), profiles = self.profiles.len(), "saved speaker profiles");
        Ok(())
    }

    /// Find a profile by name (case-insensitive).
    pub fn find(&self, name: &str) -> Option<&SpeakerProfile> {
        let lower = name.to_lowercase();
        self.profiles
            .iter()
            .find(|p| p.name.to_lowercase() == lower || p.short_name.to_lowercase() == lower)
    }

    /// Find a mutable profile by name (case-insensitive).
    pub fn find_mut(&mut self, name: &str) -> Option<&mut SpeakerProfile> {
        let lower = name.to_lowercase();
        self.profiles
            .iter_mut()
            .find(|p| p.name.to_lowercase() == lower || p.short_name.to_lowercase() == lower)
    }

    /// Add or update a speaker profile with a new embedding.
    /// If the speaker already exists, the embedding is appended and the centroid is recomputed.
    pub fn enroll(&mut self, name: &str, embedding: Vec<f32>) {
        if let Some(profile) = self.find_mut(name) {
            profile.embeddings.push(embedding);
            profile.centroid = compute_centroid(&profile.embeddings);
            info!(
                name,
                samples = profile.embeddings.len(),
                "updated speaker profile"
            );
        } else {
            let centroid = embedding.clone();
            let now = chrono_now();
            let short_name = name
                .split_whitespace()
                .next()
                .unwrap_or(name)
                .to_string();
            let profile = SpeakerProfile {
                name: name.to_string(),
                short_name,
                enrolled_at: now,
                embeddings: vec![embedding],
                centroid,
            };
            info!(name, "enrolled new speaker");
            self.profiles.push(profile);
        }
    }

    /// Remove a speaker profile by name. Returns true if removed.
    pub fn remove(&mut self, name: &str) -> bool {
        let lower = name.to_lowercase();
        let before = self.profiles.len();
        self.profiles.retain(|p| {
            p.name.to_lowercase() != lower && p.short_name.to_lowercase() != lower
        });
        before != self.profiles.len()
    }
}

impl Default for ProfileStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the centroid (element-wise mean) of multiple embeddings.
fn compute_centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    if embeddings.len() == 1 {
        return embeddings[0].clone();
    }

    let dim = embeddings[0].len();
    let mut centroid = vec![0.0f32; dim];
    let n = embeddings.len() as f32;

    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            if i < dim {
                centroid[i] += v;
            }
        }
    }

    for v in &mut centroid {
        *v /= n;
    }

    // L2 normalize the centroid
    let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for v in &mut centroid {
            *v /= norm;
        }
    }

    centroid
}

/// Simple ISO-8601 timestamp without pulling in chrono.
fn chrono_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Approximate: good enough for a human-readable enrolled_at field
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    format!(
        "{years:04}-{months:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z"
    )
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_compute_centroid_single() {
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let centroid = compute_centroid(&embeddings);
        assert_eq!(centroid.len(), 3);
        assert!((centroid[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_centroid_multiple() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let centroid = compute_centroid(&embeddings);
        assert_eq!(centroid.len(), 2);
        // Mean is [0.5, 0.5], L2 normalized = [0.707, 0.707]
        let expected = 0.5 / (0.5f32.powi(2) + 0.5f32.powi(2)).sqrt();
        assert!((centroid[0] - expected).abs() < 1e-5);
        assert!((centroid[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_profile_store_roundtrip() {
        let tmp = std::env::temp_dir().join("transcriber_test_profiles.json");
        let _ = fs::remove_file(&tmp);

        let mut store = ProfileStore::new();
        store.enroll("Alice Tester", vec![1.0, 0.0, 0.0]);
        store.enroll("Bob Example", vec![0.0, 1.0, 0.0]);
        store.save(&tmp).unwrap();

        let loaded = ProfileStore::load(&tmp).unwrap();
        assert_eq!(loaded.profiles.len(), 2);
        assert_eq!(loaded.profiles[0].name, "Alice Tester");
        assert_eq!(loaded.profiles[0].short_name, "Alice");
        assert_eq!(loaded.profiles[1].name, "Bob Example");

        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_profile_store_load_nonexistent() {
        let store = ProfileStore::load(Path::new("/nonexistent/profiles.json")).unwrap();
        assert!(store.profiles.is_empty());
    }

    #[test]
    fn test_profile_enroll_append() {
        let mut store = ProfileStore::new();
        store.enroll("Alice", vec![1.0, 0.0, 0.0]);
        store.enroll("Alice", vec![0.0, 1.0, 0.0]);
        assert_eq!(store.profiles.len(), 1);
        assert_eq!(store.profiles[0].embeddings.len(), 2);
    }

    #[test]
    fn test_profile_remove() {
        let mut store = ProfileStore::new();
        store.enroll("Alice", vec![1.0, 0.0]);
        store.enroll("Bob", vec![0.0, 1.0]);
        assert!(store.remove("Alice"));
        assert_eq!(store.profiles.len(), 1);
        assert_eq!(store.profiles[0].name, "Bob");
    }

    #[test]
    fn test_profile_remove_nonexistent() {
        let mut store = ProfileStore::new();
        assert!(!store.remove("Nobody"));
    }

    #[test]
    fn test_profile_find_case_insensitive() {
        let mut store = ProfileStore::new();
        store.enroll("Alice Tester", vec![1.0, 0.0]);
        assert!(store.find("alice tester").is_some());
        assert!(store.find("Alice").is_some()); // short_name match
        assert!(store.find("ALICE").is_some());
        assert!(store.find("Bob").is_none());
    }
}
