use std::collections::VecDeque;
use std::path::Path;

use tracing::{debug, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::{Language, TranscribeOptions};
use crate::error::{Error, Result};
use crate::types::{Segment, Transcript, Word};

/// Window size for detecting hallucination loops via rolling text history.
const HALLUCINATION_WINDOW: usize = 6;

/// If a single phrase appears this many times in the rolling window, it's a loop.
const HALLUCINATION_THRESHOLD: usize = 3;

/// Transcribe audio samples using whisper.cpp.
/// Samples must be 16kHz mono f32.
pub fn transcribe_samples(
    samples: &[f32],
    model_path: &Path,
    options: &TranscribeOptions,
) -> Result<Transcript> {
    info!(model = %model_path.display(), "loading whisper model");

    let mut ctx_params = WhisperContextParameters::new();
    ctx_params.use_gpu(options.gpu);
    ctx_params.gpu_device(
        i32::try_from(options.gpu_device)
            .map_err(|_| Error::Transcription(format!("gpu_device {} exceeds i32 range", options.gpu_device)))?
    );

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or_else(|| {
            Error::Model("model path contains invalid UTF-8".into())
        })?,
        ctx_params,
    )?;

    let mut state = ctx.create_state()?;

    let mut params = match options.beam_size {
        Some(beam_size) => FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: i32::try_from(beam_size)
                .map_err(|_| Error::Transcription(format!("beam_size {} exceeds i32 range", beam_size)))?,
            patience: -1.0,
        }),
        None => FullParams::new(SamplingStrategy::Greedy { best_of: 5 }),
    };

    // Language
    match &options.language {
        Language::Auto => params.set_detect_language(true),
        Language::Code { code, .. } => params.set_language(Some(code)),
    }

    params.set_translate(options.translate);
    params.set_token_timestamps(options.word_timestamps);
    params.set_temperature(options.temperature);

    // Anti-hallucination decoder settings:
    // - entropy_thold: segments with high entropy (repetitive/low-info) get retried
    //   at higher temperature. Default 2.4, we use 2.4 (whisper's own default).
    // - logprob_thold: segments with very low confidence get retried. Default -1.0.
    // - temperature_inc: how much to bump temperature on retry. Default 0.2.
    // - suppress_nst: suppress non-speech tokens to reduce hallucinated filler.
    // - no_speech_thold: threshold for no-speech probability. Default 0.6.
    // - n_max_text_ctx: limit past text used as decoder prompt. Default 16384.
    //   Setting to 0 prevents hallucination loops from poisoning subsequent chunks —
    //   each 30s window starts with a clean decoder slate.
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    params.set_temperature_inc(0.2);
    params.set_suppress_nst(true);
    params.set_no_speech_thold(0.6);
    params.set_n_max_text_ctx(0);

    #[cfg(feature = "diarize")]
    params.set_tdrz_enable(options.diarize);

    // Threading
    if let Some(n) = options.n_threads {
        params.set_n_threads(
            i32::try_from(n)
                .map_err(|_| Error::Transcription(format!("n_threads {} exceeds i32 range", n)))?
        );
    }

    // VAD — requires a separate Silero VAD model file.
    // Only enable if the user explicitly turned it on; we don't ship a default model.
    if options.vad {
        params.set_vad_model_path(options.vad_model_path.as_deref());
        params.enable_vad(true);
    }

    // Disable stderr printing from whisper.cpp
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    info!(samples = samples.len(), "running transcription");
    state.full(params, samples)?;

    let num_segments = state.full_n_segments();
    debug!(num_segments, "transcription complete");

    let mut segments = Vec::with_capacity(num_segments as usize);

    for i in 0..num_segments {
        let segment = state
            .get_segment(i)
            .ok_or_else(|| Error::Transcription(format!("segment {i} not found")))?;

        let start_ts = segment.start_timestamp();
        let end_ts = segment.end_timestamp();
        let text = segment.to_str_lossy()
            .map_err(|e| Error::Transcription(format!("segment text error: {e}")))?
            .into_owned();
        let speaker_turn = segment.next_segment_speaker_turn();
        let no_speech_prob = segment.no_speech_probability();

        // Word-level timestamps
        let words = if options.word_timestamps {
            let n_tokens = segment.n_tokens();
            let mut word_list = Vec::new();

            for t in 0..n_tokens {
                let token = match segment.get_token(t) {
                    Some(tok) => tok,
                    None => continue,
                };

                let token_text = match token.to_str_lossy() {
                    Ok(s) => s.into_owned(),
                    Err(_) => continue,
                };

                // Skip special tokens (they start with '[' or '<')
                let trimmed = token_text.trim();
                if trimmed.is_empty()
                    || trimmed.starts_with('[')
                    || trimmed.starts_with('<')
                {
                    continue;
                }

                let token_data = token.token_data();

                word_list.push(Word {
                    text: token_text,
                    start: token_data.t0 as f64 / 100.0,
                    end: token_data.t1 as f64 / 100.0,
                    probability: token_data.p,
                });
            }

            Some(word_list)
        } else {
            None
        };

        segments.push(Segment {
            start: start_ts as f64 / 100.0,
            end: end_ts as f64 / 100.0,
            text,
            speaker_turn,
            no_speech_probability: no_speech_prob,
            words,
        });
    }

    // Post-processing: detect and remove hallucination loops
    let before = segments.len();
    let segments = suppress_hallucinations(segments);
    let removed = before - segments.len();
    if removed > 0 {
        warn!(removed, "suppressed hallucinated segments");
    }

    let duration = samples.len() as f64 / crate::audio::WHISPER_SAMPLE_RATE as f64;

    // Get detected language from whisper state
    let detected_lang_id = state.full_lang_id_from_state();
    let language = whisper_rs::get_lang_str(detected_lang_id)
        .unwrap_or("unknown")
        .to_string();

    Ok(Transcript {
        segments,
        language,
        duration,
        model: options.model.name().to_string(),
        source_url: None,
        source_title: None,
    })
}

/// Detect and remove hallucination loops from segments.
///
/// Uses a rolling window to catch:
/// - Exact consecutive repeats (A, A, A, A...)
/// - Alternating patterns (A, B, A, B, A, B...)
/// - Short cycle loops (A, B, C, A, B, C...)
///
/// A segment is considered hallucinated if the same normalized text appears
/// >= HALLUCINATION_THRESHOLD times within the last HALLUCINATION_WINDOW segments.
fn suppress_hallucinations(segments: Vec<Segment>) -> Vec<Segment> {
    if segments.len() < HALLUCINATION_THRESHOLD {
        return segments;
    }

    let mut result: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut window: VecDeque<String> = VecDeque::with_capacity(HALLUCINATION_WINDOW);

    for seg in segments {
        let normalized = seg.text.trim().to_lowercase();

        // Count how many times this text appears in the rolling window
        let count = window.iter().filter(|t| **t == normalized).count();

        if count >= HALLUCINATION_THRESHOLD {
            debug!(
                text = seg.text.trim(),
                window_count = count + 1,
                "suppressed hallucination"
            );
            // Don't add to window either — prevents the window from being
            // entirely hallucinated text which would mask new hallucinations
            continue;
        }

        // Maintain rolling window
        if window.len() >= HALLUCINATION_WINDOW {
            window.pop_front();
        }
        window.push_back(normalized);
        result.push(seg);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start: f64, end: f64) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker_turn: false,
            no_speech_probability: 0.0,
            words: None,
        }
    }

    #[test]
    fn test_suppress_exact_repeats() {
        let segments = vec![
            seg("Hello", 0.0, 1.0),
            seg("Hello", 1.0, 2.0),
            seg("Hello", 2.0, 3.0),
            seg("Hello", 3.0, 4.0),
            seg("World", 4.0, 5.0),
        ];
        let result = suppress_hallucinations(segments);
        assert_eq!(result.len(), 4); // 3x Hello (threshold) + World
    }

    #[test]
    fn test_suppress_alternating_pattern() {
        let segments = vec![
            seg("Ja, ja.", 0.0, 1.0),
            seg("Das stimmt.", 1.0, 2.0),
            seg("Ja, ja.", 2.0, 3.0),
            seg("Das stimmt.", 3.0, 4.0),
            seg("Ja, ja.", 4.0, 5.0),
            seg("Das stimmt.", 5.0, 6.0),
            seg("Ja, ja.", 6.0, 7.0),
            seg("Real content", 7.0, 8.0),
        ];
        let result = suppress_hallucinations(segments);
        // First few get through, then suppression kicks in
        let texts: Vec<&str> = result.iter().map(|s| s.text.trim()).collect();
        assert!(texts.contains(&"Real content"));
        // The alternating phrases should be limited
        let ja_count = texts.iter().filter(|t| **t == "Ja, ja.").count();
        assert!(ja_count <= 3, "too many repeats: {ja_count}");
    }

    #[test]
    fn test_no_false_positives_on_short_words() {
        let segments = vec![
            seg("Ja.", 0.0, 1.0),
            seg("Okay.", 1.0, 2.0),
            seg("Ja.", 2.0, 3.0),
            seg("Nein.", 3.0, 4.0),
        ];
        let result = suppress_hallucinations(segments);
        assert_eq!(result.len(), 4); // only 2 "Ja." — below threshold
    }

    #[test]
    fn test_case_insensitive() {
        let segments = vec![
            seg("Hello", 0.0, 1.0),
            seg("hello", 1.0, 2.0),
            seg("HELLO", 2.0, 3.0),
            seg("Hello", 3.0, 4.0),
        ];
        let result = suppress_hallucinations(segments);
        assert!(result.len() < 4);
    }

    #[test]
    fn test_empty_and_short() {
        assert_eq!(suppress_hallucinations(vec![]).len(), 0);
        assert_eq!(suppress_hallucinations(vec![seg("A", 0.0, 1.0)]).len(), 1);
    }
}
