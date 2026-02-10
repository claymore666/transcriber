use std::path::Path;

use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::{Language, TranscribeOptions};
use crate::error::{Error, Result};
use crate::types::{Segment, Transcript, Word};

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
    ctx_params.gpu_device(options.gpu_device as i32);

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or_else(|| {
            Error::Model("model path contains invalid UTF-8".into())
        })?,
        ctx_params,
    )?;

    let mut state = ctx.create_state()?;

    let mut params = match options.beam_size {
        Some(beam_size) => FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: beam_size as i32,
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

    #[cfg(feature = "diarize")]
    params.set_tdrz_enable(options.diarize);

    // Threading
    if let Some(n) = options.n_threads {
        params.set_n_threads(n as i32);
    }

    // VAD
    if options.vad {
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

    let duration = samples.len() as f64 / 16_000.0;

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
