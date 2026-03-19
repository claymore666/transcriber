use crate::audio::WHISPER_SAMPLE_RATE;

/// Minimum segment duration in seconds for reliable speaker embedding.
/// Segments shorter than this are skipped during identification.
pub const MIN_SEGMENT_DURATION: f64 = 1.0;

/// Extract an audio slice from the full recording by timestamp.
///
/// Returns a slice of the audio buffer corresponding to `[start, end)` in seconds.
/// Returns `None` if the resulting slice is too short for reliable embedding.
pub fn extract_slice(audio: &[f32], start: f64, end: f64) -> Option<&[f32]> {
    let sample_rate = WHISPER_SAMPLE_RATE as f64;
    let start_sample = (start * sample_rate) as usize;
    let end_sample = (end * sample_rate) as usize;

    let start_sample = start_sample.min(audio.len());
    let end_sample = end_sample.min(audio.len());

    if end_sample <= start_sample {
        return None;
    }

    let duration = (end_sample - start_sample) as f64 / sample_rate;
    if duration < MIN_SEGMENT_DURATION {
        return None;
    }

    Some(&audio[start_sample..end_sample])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_valid_slice() {
        // 3 seconds of audio at 16kHz
        let audio = vec![0.0f32; 48_000];
        let slice = extract_slice(&audio, 0.5, 2.0);
        assert!(slice.is_some());
        let s = slice.unwrap();
        // 1.5 seconds = 24000 samples
        assert_eq!(s.len(), 24_000);
    }

    #[test]
    fn test_extract_too_short() {
        let audio = vec![0.0f32; 48_000];
        // 0.5 seconds < MIN_SEGMENT_DURATION
        let slice = extract_slice(&audio, 1.0, 1.5);
        assert!(slice.is_none());
    }

    #[test]
    fn test_extract_clamps_to_bounds() {
        let audio = vec![0.0f32; 16_000]; // 1 second
        // Request beyond audio length
        let slice = extract_slice(&audio, 0.0, 5.0);
        assert!(slice.is_some());
        assert_eq!(slice.unwrap().len(), 16_000);
    }

    #[test]
    fn test_extract_reversed_range() {
        let audio = vec![0.0f32; 48_000];
        let slice = extract_slice(&audio, 2.0, 1.0);
        assert!(slice.is_none());
    }

    #[test]
    fn test_extract_empty_audio() {
        let audio: Vec<f32> = vec![];
        let slice = extract_slice(&audio, 0.0, 1.0);
        assert!(slice.is_none());
    }
}
