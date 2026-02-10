use std::path::Path;
use std::process::Command;

use tracing::{debug, info};

use crate::config::AudioProcessing;
use crate::error::{Error, Result};

/// Target sample rate for whisper.cpp.
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Maximum audio duration in seconds (8 hours).
/// Prevents unbounded memory allocation from very long audio files.
/// 8 hours at 16kHz mono f32 = ~1.8 GB.
const MAX_AUDIO_DURATION_SECS: f64 = 8.0 * 3600.0;

/// Minimum RMS level — below this we consider the audio silent/empty.
const MIN_RMS: f32 = 1e-6;

/// Load an audio file, decode it, and return 16kHz mono f32 samples ready for whisper.
///
/// Uses ffmpeg to decode any audio format, downnmix to mono, and resample to 16kHz —
/// exactly like the proven brewery/whisperx pipeline. Supports every format ffmpeg does
/// (mp3, wav, ogg, opus, webm, aac, flac, m4a, wma, aiff, ...).
///
/// Optional processing (controlled by `AudioProcessing`):
/// - Remove DC offset
/// - Peak normalize
/// - Trim leading/trailing silence
pub fn load_audio(path: &Path, processing: &AudioProcessing) -> Result<Vec<f32>> {
    info!(path = %path.display(), "loading audio");

    if !path.exists() {
        return Err(Error::AudioNotFound {
            path: path.to_path_buf(),
        });
    }

    let mut samples = decode_with_ffmpeg(path)?;

    let duration_raw = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    debug!(
        samples = samples.len(),
        duration_secs = format!("{duration_raw:.1}"),
        "decoded audio"
    );

    if duration_raw > MAX_AUDIO_DURATION_SECS {
        return Err(Error::AudioDecode(format!(
            "audio too long ({:.0}s) — maximum supported duration is {:.0}s",
            duration_raw, MAX_AUDIO_DURATION_SECS
        )));
    }

    // Optional processing steps
    if processing.dc_offset_removal {
        remove_dc_offset(&mut samples);
    }

    if processing.normalize {
        normalize_peak(&mut samples);
    }

    if processing.trim_silence {
        samples = trim_silence(
            &samples,
            processing.silence_threshold_db,
            processing.silence_pad_ms,
        );
    }

    let duration = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    info!(duration_secs = format!("{duration:.1}"), "audio ready");

    Ok(samples)
}

/// Decode any audio file to 16kHz mono f32 via ffmpeg subprocess.
///
/// ffmpeg handles decoding, resampling, and channel mixing in one shot.
/// Output format is raw PCM signed 16-bit little-endian, which we convert to f32.
fn decode_with_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-threads",
            "0",
            "-i",
        ])
        .arg(path)
        .args([
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            &WHISPER_SAMPLE_RATE.to_string(),
            "-",
        ])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::AudioDecode(
                    "ffmpeg not found — install with: apt install ffmpeg".into(),
                )
            } else {
                Error::AudioDecode(format!("failed to run ffmpeg: {e}"))
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::AudioDecode(format!("ffmpeg failed: {stderr}")));
    }

    if output.stdout.is_empty() {
        return Err(Error::AudioDecode("ffmpeg produced no output".into()));
    }

    // Convert s16le bytes to f32 samples, normalized to [-1.0, 1.0]
    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}

/// Remove DC offset by subtracting the mean value.
fn remove_dc_offset(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    if mean.abs() > MIN_RMS {
        debug!(dc_offset = mean, "removing DC offset");
        for s in samples.iter_mut() {
            *s -= mean;
        }
    }
}

/// Normalize peak amplitude to 1.0.
fn normalize_peak(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    let peak = samples
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);

    if peak < MIN_RMS {
        debug!("audio is silent (peak below threshold)");
        return;
    }

    if (peak - 1.0).abs() > 0.01 {
        debug!(peak, "normalizing peak amplitude");
        let scale = 1.0 / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }
}

/// Trim leading and trailing silence using a window-based RMS approach.
fn trim_silence(samples: &[f32], threshold_db: f32, pad_ms: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let threshold = db_to_linear(threshold_db);

    // 10ms windows
    let window_size = (WHISPER_SAMPLE_RATE as usize / 100).max(1);

    let start = find_first_active(samples, window_size, threshold).unwrap_or(0);
    let end = find_last_active(samples, window_size, threshold).unwrap_or(samples.len());

    if start >= end {
        return samples.to_vec();
    }

    let pad_samples = (WHISPER_SAMPLE_RATE as usize * pad_ms as usize) / 1000;
    let start = start.saturating_sub(pad_samples);
    let end = (end + pad_samples).min(samples.len());

    if start == 0 && end == samples.len() {
        return samples.to_vec();
    }

    debug!(
        trimmed_start_ms = (start as f64 / WHISPER_SAMPLE_RATE as f64 * 1000.0) as u64,
        trimmed_end_ms =
            ((samples.len() - end) as f64 / WHISPER_SAMPLE_RATE as f64 * 1000.0) as u64,
        "trimmed silence"
    );

    samples[start..end].to_vec()
}

fn find_first_active(samples: &[f32], window_size: usize, threshold: f32) -> Option<usize> {
    for (i, window) in samples.windows(window_size).enumerate() {
        if rms(window) > threshold {
            return Some(i);
        }
    }
    None
}

fn find_last_active(samples: &[f32], window_size: usize, threshold: f32) -> Option<usize> {
    let total_windows = samples.len().saturating_sub(window_size - 1);
    for i in (0..total_windows).rev() {
        if rms(&samples[i..i + window_size]) > threshold {
            return Some(i + window_size);
        }
    }
    None
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::config::AudioProcessing;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    // --- ffmpeg decoding tests ---

    #[test]
    fn test_load_wav() {
        let path = fixtures_dir().join("sine_440hz_2s.wav");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        // 2 seconds at 16kHz = 32000 samples (roughly)
        assert!(samples.len() > 30_000);
        assert!(samples.len() < 34_000);
    }

    #[test]
    fn test_load_mp3() {
        let path = fixtures_dir().join("sine_440hz_1s.mp3");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        // 1 second at 16kHz = ~16000 samples
        assert!(samples.len() > 14_000);
        assert!(samples.len() < 18_000);
    }

    #[test]
    fn test_load_opus() {
        let path = fixtures_dir().join("sine_440hz_1s.opus");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        assert!(samples.len() > 14_000);
        assert!(samples.len() < 18_000);
    }

    #[test]
    fn test_load_flac() {
        let path = fixtures_dir().join("sine_48khz_1s.flac");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        // Resampled from 48kHz to 16kHz, 1 second
        assert!(samples.len() > 14_000);
        assert!(samples.len() < 18_000);
    }

    #[test]
    fn test_load_stereo_downmix() {
        let path = fixtures_dir().join("stereo_2s.wav");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        // Should be mono after ffmpeg -ac 1
        assert!(samples.len() > 30_000);
        assert!(samples.len() < 34_000);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let path = fixtures_dir().join("does_not_exist.wav");
        let result = load_audio(&path, &AudioProcessing::default());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::AudioNotFound { .. }));
    }

    #[test]
    fn test_samples_in_valid_range() {
        let path = fixtures_dir().join("sine_440hz_2s.wav");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        for &s in &samples {
            assert!(s >= -1.0 && s <= 1.0, "sample {s} out of range");
        }
    }

    // --- DC offset removal tests ---

    #[test]
    fn test_dc_offset_removal() {
        let mut samples = vec![0.5, 0.6, 0.4, 0.5, 0.7, 0.3];
        let mean_before: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean_before.abs() > 0.1);

        remove_dc_offset(&mut samples);

        let mean_after: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean_after.abs() < 1e-5);
    }

    #[test]
    fn test_dc_offset_removal_empty() {
        let mut samples: Vec<f32> = vec![];
        remove_dc_offset(&mut samples); // should not panic
    }

    #[test]
    fn test_dc_offset_removal_zero_mean() {
        let mut samples = vec![-0.5, 0.5, -0.5, 0.5];
        let original = samples.clone();
        remove_dc_offset(&mut samples);
        // Mean is already ~0, samples should be unchanged
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // --- Peak normalization tests ---

    #[test]
    fn test_normalize_peak() {
        let mut samples = vec![0.1, -0.2, 0.3, -0.15, 0.25];
        normalize_peak(&mut samples);

        let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut samples = vec![0.5, -1.0, 0.7, -0.3];
        let original = samples.clone();
        normalize_peak(&mut samples);
        // Peak is already 1.0, should be unchanged
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_normalize_silent() {
        let mut samples = vec![0.0, 0.0, 0.0];
        normalize_peak(&mut samples); // should not panic or divide by zero
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_normalize_empty() {
        let mut samples: Vec<f32> = vec![];
        normalize_peak(&mut samples); // should not panic
    }

    // --- Silence trimming tests ---

    #[test]
    fn test_trim_silence_leading() {
        // 1 second of silence + 1 second of signal
        let mut samples = vec![0.0; 16_000];
        samples.extend(vec![0.5; 16_000]);

        let trimmed = trim_silence(&samples, -40.0, 50);
        assert!(trimmed.len() < samples.len());
        // Should have trimmed most of the leading silence
        assert!(trimmed.len() < 18_000);
    }

    #[test]
    fn test_trim_silence_trailing() {
        // 1 second of signal + 1 second of silence
        let mut samples = vec![0.5; 16_000];
        samples.extend(vec![0.0; 16_000]);

        let trimmed = trim_silence(&samples, -40.0, 50);
        assert!(trimmed.len() < samples.len());
        assert!(trimmed.len() < 18_000);
    }

    #[test]
    fn test_trim_silence_both_sides() {
        // silence + signal + silence
        let mut samples = vec![0.0; 16_000];
        samples.extend(vec![0.5; 16_000]);
        samples.extend(vec![0.0; 16_000]);

        let trimmed = trim_silence(&samples, -40.0, 50);
        assert!(trimmed.len() < samples.len());
        // Should be roughly 1 second of signal + padding
        assert!(trimmed.len() < 20_000);
    }

    #[test]
    fn test_trim_silence_no_silence() {
        let samples = vec![0.5; 16_000];
        let trimmed = trim_silence(&samples, -40.0, 50);
        assert_eq!(trimmed.len(), samples.len());
    }

    #[test]
    fn test_trim_silence_all_silent() {
        let samples = vec![0.0; 16_000];
        let trimmed = trim_silence(&samples, -40.0, 50);
        // Should return original since there's no active audio
        assert_eq!(trimmed.len(), samples.len());
    }

    #[test]
    fn test_trim_silence_empty() {
        let trimmed = trim_silence(&[], -40.0, 50);
        assert!(trimmed.is_empty());
    }

    // --- Audio processing integration ---

    #[test]
    fn test_load_with_all_processing() {
        let path = fixtures_dir().join("sine_440hz_2s.wav");
        let samples = load_audio(&path, &AudioProcessing::all()).unwrap();
        assert!(!samples.is_empty());

        let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.01, "should be normalized, peak={peak}");
    }

    #[test]
    fn test_load_with_no_processing() {
        let path = fixtures_dir().join("sine_440hz_2s.wav");
        let samples = load_audio(&path, &AudioProcessing::default()).unwrap();
        assert!(!samples.is_empty());
    }

    // --- Helper function tests ---

    #[test]
    fn test_rms() {
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((rms(&[0.0, 0.0, 0.0])).abs() < 1e-6);

        // RMS of [1, -1, 1, -1] = 1.0
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-6);
        assert!((db_to_linear(-40.0) - 0.01).abs() < 1e-6);
        assert!((db_to_linear(-60.0) - 0.001).abs() < 1e-5);
    }

    // --- Security tests ---

    #[test]
    fn test_load_rejects_nonexistent() {
        let result = load_audio(
            &PathBuf::from("/nonexistent/../../etc/passwd"),
            &AudioProcessing::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_rejects_non_audio_file() {
        // Try to load a text file — ffmpeg should fail
        let tmp = std::env::temp_dir().join("transcriber_test_not_audio.txt");
        std::fs::write(&tmp, "this is not audio").unwrap();
        let result = load_audio(&tmp, &AudioProcessing::default());
        assert!(result.is_err());
        std::fs::remove_file(&tmp).ok();
    }
}
