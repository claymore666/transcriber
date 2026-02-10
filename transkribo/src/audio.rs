use std::path::Path;
use std::process::Command;

use tracing::{debug, info};

use crate::config::AudioProcessing;
use crate::error::{Error, Result};

/// Target sample rate for whisper.cpp.
const WHISPER_SAMPLE_RATE: u32 = 16_000;

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
