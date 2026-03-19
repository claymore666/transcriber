use std::path::Path;

use ndarray::{Array2, Axis};
use nnnoiseless::DenoiseState;
use ort::session::Session;
use ort::value::Tensor;
use tracing::{debug, info};

use crate::error::{Error, Result};
use crate::speaker::ExecutionProvider;

/// ONNX input tensor name for wespeaker models.
const INPUT_NAME: &str = "feats";

/// ONNX output tensor name for wespeaker models.
const OUTPUT_NAME: &str = "embs";

/// Fbank configuration matching wespeaker's training config.
fn create_fbank_options() -> kaldi_native_fbank::FbankOptions {
    let mut opts = kaldi_native_fbank::FbankOptions::default();

    // Frame extraction
    opts.frame_opts.samp_freq = 16000.0;
    opts.frame_opts.frame_shift_ms = 10.0;
    opts.frame_opts.frame_length_ms = 25.0;
    opts.frame_opts.dither = 0.0; // No dither at inference time
    opts.frame_opts.preemph_coeff = 0.97;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "povey".to_string();
    opts.frame_opts.round_to_power_of_two = true;
    opts.frame_opts.snip_edges = true;

    // Mel filterbank
    opts.mel_opts.num_bins = 80;

    // Fbank
    opts.use_energy = false;
    opts.use_log_fbank = true;
    opts.use_power = true;

    opts
}

/// Compute Kaldi-compatible fbank features from audio samples.
///
/// Input: f32 samples at 16kHz mono.
/// Output: [num_frames, 80] feature matrix with CMN applied.
pub fn compute_fbank(samples: &[f32]) -> Result<Array2<f32>> {
    let opts = create_fbank_options();
    let computer = kaldi_native_fbank::online::FeatureComputer::Fbank(
        kaldi_native_fbank::FbankComputer::new(opts)
            .map_err(|e| Error::SpeakerId(format!("failed to create fbank computer: {e}")))?,
    );

    let mut online = kaldi_native_fbank::OnlineFeature::new(computer);
    online.accept_waveform(16000.0, samples);
    online.input_finished();

    let num_frames = online.num_frames_ready();
    if num_frames == 0 {
        return Err(Error::SpeakerId(
            "audio too short — no fbank frames extracted".into(),
        ));
    }

    let num_bins = 80;
    let mut features = Array2::zeros((num_frames, num_bins));
    for i in 0..num_frames {
        let frame = online
            .get_frame(i)
            .ok_or_else(|| Error::SpeakerId(format!("failed to get fbank frame {i}")))?;
        features
            .row_mut(i)
            .assign(&ndarray::ArrayView1::from(frame));
    }

    // Apply CMN: subtract per-bin mean across time axis
    let mean = features.mean_axis(Axis(0)).ok_or_else(|| {
        Error::SpeakerId("failed to compute fbank mean for CMN".into())
    })?;
    features -= &mean;

    debug!(num_frames, num_bins, "computed fbank features");
    Ok(features)
}

/// Create an ONNX runtime session for the speaker embedding model.
pub fn create_session(
    model_path: &Path,
    execution_providers: &[ExecutionProvider],
) -> Result<Session> {
    info!(path = %model_path.display(), "loading speaker embedding model");

    let mut builder = Session::builder().map_err(|e| {
        Error::SpeakerId(format!("failed to create ort session builder: {e}"))
    })?;

    // Register execution providers in order (ort falls back to next if one fails)
    let eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = execution_providers
        .iter()
        .map(|ep| match ep {
            ExecutionProvider::Cpu => {
                ort::execution_providers::CPUExecutionProvider::default().build()
            }
            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda { device_id } => {
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(*device_id as i32)
                    .build()
            }
            #[cfg(not(feature = "cuda"))]
            ExecutionProvider::Cuda { .. } => {
                tracing::warn!("CUDA requested but cuda feature not enabled, falling back to CPU");
                ort::execution_providers::CPUExecutionProvider::default().build()
            }
        })
        .collect();

    if !eps.is_empty() {
        builder = builder.with_execution_providers(eps).map_err(|e| {
            Error::SpeakerId(format!("failed to set execution providers: {e}"))
        })?;
    }

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| Error::SpeakerId(format!("failed to load model {}: {e}", model_path.display())))?;

    // Log model input/output info
    for input in session.inputs() {
        debug!(name = %input.name(), "model input");
    }
    for output in session.outputs() {
        debug!(name = %output.name(), "model output");
    }

    Ok(session)
}

/// Apply RNNoise denoising to audio samples.
///
/// Cleans background noise (HVAC, paper rustling, cross-talk) from the audio
/// before fbank extraction. This significantly improves speaker embedding quality
/// for room mic recordings where noise features would otherwise dominate the
/// mel filterbank.
///
/// Input/output: f32 samples at 16kHz mono.
fn denoise(audio: &[f32]) -> Vec<f32> {
    let mut state = DenoiseState::new();
    let mut output = Vec::with_capacity(audio.len());

    // RNNoise processes 480-sample frames (30ms at 16kHz)
    const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE; // 480

    for chunk in audio.chunks(FRAME_SIZE) {
        let mut input_frame = [0.0f32; FRAME_SIZE];
        let mut output_frame = [0.0f32; FRAME_SIZE];
        input_frame[..chunk.len()].copy_from_slice(chunk);
        state.process_frame(&mut output_frame, &input_frame);
        output.extend_from_slice(&output_frame[..chunk.len()]);
    }

    output
}

/// Extract a speaker embedding from audio samples.
///
/// Pipeline: audio -> RNNoise denoise -> fbank features -> CMN -> ONNX inference -> L2 normalize.
pub fn extract_embedding(session: &mut Session, samples: &[f32]) -> Result<Vec<f32>> {
    // 1. Denoise audio to remove background noise before feature extraction
    let denoised = denoise(samples);

    // 2. Compute fbank features: [num_frames, 80]
    let features = compute_fbank(&denoised)?;

    // 3. Add batch dimension: [1, num_frames, 80]
    let shape = features.shape().to_vec();
    let features_3d = features
        .into_shape_with_order((1, shape[0], shape[1]))
        .map_err(|e| Error::SpeakerId(format!("failed to reshape features: {e}")))?;

    // 4. Run ONNX inference (Tensor::from_array takes ownership)
    let input_tensor = Tensor::from_array(features_3d)
        .map_err(|e| Error::SpeakerId(format!("failed to create input tensor: {e}")))?;
    let outputs = session
        .run(ort::inputs![INPUT_NAME => input_tensor])
        .map_err(|e| Error::SpeakerId(format!("onnx inference failed: {e}")))?;

    let output = outputs.get(OUTPUT_NAME).ok_or_else(|| {
        Error::SpeakerId(format!(
            "model has no output tensor named '{OUTPUT_NAME}'"
        ))
    })?;
    let (_shape, data) = output.try_extract_tensor::<f32>().map_err(|e| {
        Error::SpeakerId(format!("failed to extract embedding tensor: {e}"))
    })?;

    // 5. L2 normalize
    let embedding: Vec<f32> = data.to_vec();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return Err(Error::SpeakerId("embedding has near-zero norm".into()));
    }

    Ok(embedding.iter().map(|x| x / norm).collect())
}

/// Get the embedding dimension from a loaded session by inspecting the output tensor.
pub fn embedding_dim(session: &Session) -> Result<usize> {
    for output in session.outputs() {
        if output.name() == OUTPUT_NAME {
            if let ort::value::ValueType::Tensor { shape, .. } = output.dtype() {
                // shape is e.g. [-1, 256] where -1 is batch dim
                if let Some(&dim) = shape.last() {
                    if dim > 0 {
                        return Ok(dim as usize);
                    }
                }
            }
        }
    }

    Err(Error::SpeakerId(
        "could not determine embedding dimension from model metadata".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fbank_sine() {
        // 2 seconds of 440Hz sine wave at 16kHz
        let samples: Vec<f32> = (0..32_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let features = compute_fbank(&samples).unwrap();
        assert_eq!(features.shape()[1], 80); // 80 mel bins
        assert!(features.shape()[0] > 100); // ~200 frames for 2s
        assert!(features.shape()[0] < 250);
    }

    #[test]
    fn test_compute_fbank_empty() {
        let result = compute_fbank(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_fbank_very_short() {
        // 10ms of audio — might be too short for even one frame
        let samples = vec![0.0f32; 160];
        // Just verify it doesn't panic
        let _ = compute_fbank(&samples);
    }

    #[test]
    fn test_denoise_preserves_length() {
        // 2 seconds of audio at 16kHz
        let samples: Vec<f32> = (0..32_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let denoised = denoise(&samples);
        assert_eq!(denoised.len(), samples.len());
    }

    #[test]
    fn test_denoise_non_frame_aligned_length() {
        // Length not divisible by 480 (RNNoise frame size)
        let samples = vec![0.1f32; 1000];
        let denoised = denoise(&samples);
        assert_eq!(denoised.len(), 1000);
    }

    #[test]
    fn test_denoise_empty_input() {
        let denoised = denoise(&[]);
        assert!(denoised.is_empty());
    }

    #[test]
    fn test_denoise_single_frame() {
        // Exactly one RNNoise frame (480 samples)
        let samples = vec![0.1f32; 480];
        let denoised = denoise(&samples);
        assert_eq!(denoised.len(), 480);
    }

    #[test]
    fn test_denoise_reduces_noise_on_silence() {
        // Pure silence should remain near-silent after denoising
        let samples = vec![0.0f32; 16_000];
        let denoised = denoise(&samples);
        let rms: f32 = (denoised.iter().map(|x| x * x).sum::<f32>() / denoised.len() as f32).sqrt();
        assert!(rms < 0.01, "denoised silence should have near-zero RMS, got {rms}");
    }

    #[test]
    fn test_denoise_does_not_destroy_speech_signal() {
        // A 440Hz sine (speech-like frequency) should retain significant energy after denoising
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let input_rms: f32 = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        let denoised = denoise(&samples);
        let output_rms: f32 = (denoised.iter().map(|x| x * x).sum::<f32>() / denoised.len() as f32).sqrt();

        // RNNoise may attenuate pure tones somewhat, but shouldn't zero them out
        assert!(output_rms > input_rms * 0.01,
            "denoised signal lost too much energy: input_rms={input_rms}, output_rms={output_rms}");
    }

    #[test]
    fn test_denoise_fbank_produces_valid_features() {
        // Verify that fbank extraction still works on denoised audio
        let samples: Vec<f32> = (0..32_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let denoised = denoise(&samples);
        let features = compute_fbank(&denoised).unwrap();
        assert_eq!(features.shape()[1], 80);
        assert!(features.shape()[0] > 50, "should produce frames from denoised audio");
    }

    #[test]
    fn test_compute_fbank_cmn_applied() {
        // After CMN, the per-bin mean should be approximately zero
        let samples: Vec<f32> = (0..32_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let features = compute_fbank(&samples).unwrap();
        let mean = features.mean_axis(Axis(0)).unwrap();
        for &m in mean.iter() {
            assert!(m.abs() < 1e-5, "CMN not applied correctly, mean={m}");
        }
    }
}
