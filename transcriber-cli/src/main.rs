use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use transcriber::{Language, Model, TranscribeOptions};

#[derive(Parser)]
#[command(name = "transcriber", about = "Transcribe audio/video from URL or file")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// URL or local file path to transcribe.
    input: Option<String>,

    /// Output format.
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Write output to file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Whisper model to use.
    #[arg(short, long, default_value = "large-v3")]
    model: String,

    /// Language code (e.g. "en", "de") or "auto" for detection.
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Translate to English.
    #[arg(long)]
    translate: bool,

    /// Enable word-level timestamps.
    #[arg(long)]
    word_timestamps: bool,

    /// Disable GPU acceleration.
    #[arg(long)]
    no_gpu: bool,

    /// GPU device ID.
    #[arg(long, default_value = "0")]
    gpu_device: u32,

    /// Number of threads (default: auto).
    #[arg(long)]
    threads: Option<u32>,

    /// Enable voice activity detection (requires --vad-model-path).
    #[arg(long)]
    vad: bool,

    /// Path to Silero VAD model file.
    #[arg(long)]
    vad_model_path: Option<String>,

    /// Sampling temperature.
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Beam search size (default: greedy).
    #[arg(long)]
    beam_size: Option<u32>,

    /// Model cache directory.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Enable DC offset removal.
    #[arg(long)]
    dc_offset: bool,

    /// Enable peak normalization.
    #[arg(long)]
    normalize: bool,

    /// Enable silence trimming.
    #[arg(long)]
    trim_silence: bool,

    /// List available models.
    #[arg(long)]
    list_models: bool,

    /// Download a model without transcribing.
    #[arg(long)]
    download_model: Option<String>,

    /// List supported languages.
    #[arg(long)]
    list_languages: bool,

    // --- Speaker identification flags ---

    /// Enable speaker identification.
    #[arg(long)]
    speaker_id: bool,

    /// Path to speaker profiles JSON file.
    #[arg(long)]
    speaker_profiles: Option<PathBuf>,

    /// Path to wespeaker ONNX model.
    #[arg(long)]
    speaker_model: Option<PathBuf>,

    /// Speaker identification confidence threshold (0.0-1.0).
    #[arg(long, default_value = "0.6")]
    speaker_threshold: f32,

    /// Download the speaker embedding model.
    #[arg(long)]
    download_speaker_model: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Enroll a speaker from an audio file.
    Enroll {
        /// Speaker name.
        #[arg(long)]
        name: String,

        /// Audio file to enroll from.
        #[arg(long)]
        audio: PathBuf,

        /// Start time (e.g. "01:23" or "83.5").
        #[arg(long)]
        start: Option<String>,

        /// End time (e.g. "02:45" or "165.0").
        #[arg(long)]
        end: Option<String>,

        /// Path to speaker profiles.
        #[arg(long)]
        profiles: Option<PathBuf>,

        /// Path to speaker embedding model.
        #[arg(long)]
        speaker_model: Option<PathBuf>,
    },

    /// Manage speaker profiles.
    Speakers {
        #[command(subcommand)]
        action: SpeakersAction,
    },
}

#[derive(Subcommand)]
enum SpeakersAction {
    /// List enrolled speakers.
    List {
        /// Path to speaker profiles.
        #[arg(long)]
        profiles: Option<PathBuf>,
    },
    /// Remove a speaker profile.
    Remove {
        /// Speaker name to remove.
        #[arg(long)]
        name: String,
        /// Path to speaker profiles.
        #[arg(long)]
        profiles: Option<PathBuf>,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Srt,
    Vtt,
    Json,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("transcriber=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();

    // Handle subcommands first
    if let Some(command) = cli.command {
        match command {
            Command::Enroll {
                name,
                audio,
                start,
                end,
                profiles,
                speaker_model,
            } => {
                cmd_enroll(name, audio, start, end, profiles, speaker_model).await;
            }
            Command::Speakers { action } => match action {
                SpeakersAction::List { profiles } => cmd_speakers_list(profiles),
                SpeakersAction::Remove { name, profiles } => cmd_speakers_remove(name, profiles),
            },
        }
        return;
    }

    if cli.list_languages {
        println!("{:<6} LANGUAGE", "CODE");
        println!("{:<6} --------", "----");
        for (code, name) in Language::supported() {
            println!("{code:<6} {name}");
        }
        return;
    }

    if cli.list_models {
        let models = [
            ("tiny", "75 MB"),
            ("tiny.en", "75 MB"),
            ("base", "142 MB"),
            ("base.en", "142 MB"),
            ("small", "466 MB"),
            ("small.en", "466 MB"),
            ("medium", "1.5 GB"),
            ("medium.en", "1.5 GB"),
            ("large-v2", "2.9 GB"),
            ("large-v3", "2.9 GB"),
            ("large-v3-turbo", "~1.6 GB"),
        ];
        println!("{:<16} SIZE", "MODEL");
        println!("{:<16} ----", "-----");
        for (name, size) in models {
            println!("{name:<16} {size}");
        }

        let opts = TranscribeOptions::default();
        let cache_dir = opts.resolve_cache_dir();
        let cached = transcriber::model::list_cached_models(&cache_dir);
        if !cached.is_empty() {
            println!("\nCached models in {}:", cache_dir.display());
            for path in cached {
                let size = std::fs::metadata(&path)
                    .map(|m| format_bytes(m.len()))
                    .unwrap_or_default();
                println!(
                    "  {} ({})",
                    path.file_name()
                        .map(|f| f.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    size
                );
            }
        }
        return;
    }

    if let Some(model_name) = &cli.download_model {
        let model = match Model::parse_name(model_name) {
            Some(m) => m,
            None => {
                eprintln!("Unknown model: {model_name}");
                eprintln!("Use --list-models to see available models");
                std::process::exit(1);
            }
        };
        let opts = TranscribeOptions::default();
        let cache_dir = cli.cache_dir.unwrap_or_else(|| opts.resolve_cache_dir());
        match transcriber::model::ensure_model(&model, &cache_dir).await {
            Ok(path) => println!("Model ready: {}", path.display()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if cli.download_speaker_model {
        let opts = TranscribeOptions::default();
        let cache_dir = cli.cache_dir.unwrap_or_else(|| opts.resolve_cache_dir());
        match transcriber::speaker::ensure_speaker_model(&cache_dir).await {
            Ok(path) => println!("Speaker model ready: {}", path.display()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let input = match cli.input {
        Some(i) => i,
        None => {
            eprintln!("Error: no input specified. Provide a URL or file path.");
            eprintln!("Usage: transcriber-cli <INPUT> [OPTIONS]");
            eprintln!("       transcriber-cli enroll --name <NAME> --audio <AUDIO>");
            eprintln!("       transcriber-cli speakers list");
            std::process::exit(1);
        }
    };

    // Build options
    let model = match Model::parse_name(&cli.model) {
        Some(m) => m,
        None => {
            let path = PathBuf::from(&cli.model);
            if path.exists() {
                Model::Custom(path)
            } else {
                eprintln!("Unknown model: {}", cli.model);
                eprintln!("Use --list-models to see available models, or provide a path to a .ggml file");
                std::process::exit(1);
            }
        }
    };

    let language = match Language::new(&cli.language) {
        Ok(lang) => lang,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("Use --list-languages to see supported languages");
            std::process::exit(1);
        }
    };

    let mut opts = match TranscribeOptions::new()
        .model(model)
        .translate(cli.translate)
        .word_timestamps(cli.word_timestamps)
        .gpu(!cli.no_gpu)
        .gpu_device(cli.gpu_device)
        .vad(cli.vad)
        .temperature(cli.temperature)
    {
        Ok(o) => o.audio_processing(
            transcriber::AudioProcessing::new()
                .dc_offset_removal(cli.dc_offset)
                .normalize(cli.normalize)
                .trim_silence(cli.trim_silence),
        ),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    opts.language = language;

    if let Some(n) = cli.threads {
        opts = match opts.n_threads(n) {
            Ok(o) => o,
            Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
        };
    }
    if let Some(size) = cli.beam_size {
        opts = match opts.beam_size(size) {
            Ok(o) => o,
            Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
        };
    }
    if let Some(path) = cli.vad_model_path {
        opts = opts.vad_model_path(path);
    }
    if let Some(dir) = cli.cache_dir {
        opts = opts.cache_dir(dir);
    }

    // Speaker identification options
    if cli.speaker_id {
        opts = opts.speaker_identification(true)
            .speaker_threshold(cli.speaker_threshold);
        if let Some(path) = cli.speaker_profiles {
            opts = opts.speaker_profiles_path(path);
        }
        if let Some(path) = cli.speaker_model {
            opts = opts.speaker_model_path(path);
        }
    }

    // Determine if input is a URL or file
    let is_url = input.starts_with("http://") || input.starts_with("https://");

    let result = if is_url {
        transcriber::transcribe_with_options(&input, &opts).await
    } else {
        transcriber::transcribe_file_with_options(&input, &opts).await
    };

    let transcript = match result {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Transcription complete: {:.1}s of audio, {} segments, language: {}",
        transcript.duration,
        transcript.segments.len(),
        transcript.language,
    );

    // Print speaker summary if identification was used
    if cli.speaker_id {
        print_speaker_summary(&transcript);
    }

    let output_text = match cli.format {
        OutputFormat::Text => transcript.text(),
        OutputFormat::Srt => transcript.to_srt(),
        OutputFormat::Vtt => transcript.to_vtt(),
        OutputFormat::Json => match transcript.to_json_pretty() {
            Ok(j) => j,
            Err(e) => {
                eprintln!("JSON error: {e}");
                std::process::exit(1);
            }
        },
    };

    match cli.output {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &output_text) {
                eprintln!("Error writing to {}: {e}", path.display());
                std::process::exit(1);
            }
            eprintln!("Written to {}", path.display());
        }
        None => print!("{output_text}"),
    }
}

/// Enroll a speaker from an audio file.
async fn cmd_enroll(
    name: String,
    audio: PathBuf,
    start: Option<String>,
    end: Option<String>,
    profiles: Option<PathBuf>,
    speaker_model: Option<PathBuf>,
) {
    let profiles_path = profiles.unwrap_or_else(transcriber::speaker::default_profiles_path);

    // Ensure speaker model is available
    let opts = TranscribeOptions::default();
    let cache_dir = opts.resolve_cache_dir();
    let model_path = match speaker_model {
        Some(p) => p,
        None => match transcriber::speaker::ensure_speaker_model(&cache_dir).await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error loading speaker model: {e}");
                std::process::exit(1);
            }
        },
    };

    // Load audio
    let processing = transcriber::AudioProcessing::default();
    let audio_clone = audio.clone();
    let samples = match tokio::task::spawn_blocking(move || {
        transcriber::__test_load_audio(&audio_clone, &processing)
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            eprintln!("Error loading audio: {e}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    // Extract time range if specified
    let samples = if start.is_some() || end.is_some() {
        let total_duration = samples.len() as f64 / 16000.0;
        let start_secs = start.map(|s| parse_time(&s)).unwrap_or(0.0);
        let end_secs = end.map(|s| parse_time(&s)).unwrap_or(total_duration);

        let start_sample = (start_secs * 16000.0) as usize;
        let end_sample = (end_secs * 16000.0) as usize;
        let start_sample = start_sample.min(samples.len());
        let end_sample = end_sample.min(samples.len());

        if end_sample <= start_sample {
            eprintln!("Error: invalid time range ({start_secs:.1}s - {end_secs:.1}s)");
            std::process::exit(1);
        }

        eprintln!(
            "Using time range {start_secs:.1}s - {end_secs:.1}s ({:.1}s)",
            (end_sample - start_sample) as f64 / 16000.0
        );
        samples[start_sample..end_sample].to_vec()
    } else {
        samples
    };

    // Create identifier and enroll
    let result = tokio::task::spawn_blocking(move || -> Result<(), transcriber::Error> {
        let mut identifier = transcriber::speaker::SpeakerIdentifier::new(
            &model_path,
            &profiles_path,
            0.6,
            &[transcriber::speaker::ExecutionProvider::Cpu],
        )?;
        identifier.enroll(&name, &samples)?;
        identifier.save_profiles(&profiles_path)?;
        let profile = identifier.profiles().find(&name).unwrap();
        eprintln!(
            "Enrolled '{}' ({} sample(s))",
            profile.name,
            profile.embeddings.len()
        );
        Ok(())
    })
    .await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// List enrolled speakers.
fn cmd_speakers_list(profiles: Option<PathBuf>) {
    let profiles_path = profiles.unwrap_or_else(transcriber::speaker::default_profiles_path);
    let store = match transcriber::speaker::ProfileStore::load(&profiles_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading profiles: {e}");
            std::process::exit(1);
        }
    };

    if store.profiles.is_empty() {
        println!("No speakers enrolled.");
        println!("Profiles path: {}", profiles_path.display());
        return;
    }

    println!("{:<20} {:<8} ENROLLED", "NAME", "SAMPLES");
    println!("{:<20} {:<8} --------", "----", "-------");
    for profile in &store.profiles {
        println!(
            "{:<20} {:<8} {}",
            profile.name,
            profile.embeddings.len(),
            profile.enrolled_at,
        );
    }
    println!("\nProfiles path: {}", profiles_path.display());
}

/// Remove a speaker profile.
fn cmd_speakers_remove(name: String, profiles: Option<PathBuf>) {
    let profiles_path = profiles.unwrap_or_else(transcriber::speaker::default_profiles_path);
    let mut store = match transcriber::speaker::ProfileStore::load(&profiles_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading profiles: {e}");
            std::process::exit(1);
        }
    };

    if store.remove(&name) {
        if let Err(e) = store.save(&profiles_path) {
            eprintln!("Error saving profiles: {e}");
            std::process::exit(1);
        }
        println!("Removed '{name}'");
    } else {
        eprintln!("Speaker '{name}' not found");
        std::process::exit(1);
    }
}

/// Print a summary of speaker identification results.
fn print_speaker_summary(transcript: &transcriber::Transcript) {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, (u32, f32)> = HashMap::new();

    for seg in &transcript.segments {
        if let Some(speaker) = &seg.speaker_id {
            let entry = counts.entry(speaker.as_str()).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += seg.speaker_confidence.unwrap_or(0.0);
        }
    }

    if counts.is_empty() {
        return;
    }

    eprintln!("\nSpeaker summary:");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (name, (count, total_conf)) in sorted {
        let avg_conf = total_conf / *count as f32;
        eprintln!("  {name}: {count} segments (avg confidence {avg_conf:.2})");
    }
}

/// Parse a time string like "01:23" or "83.5" into seconds.
fn parse_time(s: &str) -> f64 {
    if s.contains(':') {
        let parts: Vec<&str> = s.split(':').collect();
        match parts.len() {
            2 => {
                let mins: f64 = parts[0].parse().unwrap_or(0.0);
                let secs: f64 = parts[1].parse().unwrap_or(0.0);
                mins * 60.0 + secs
            }
            3 => {
                let hours: f64 = parts[0].parse().unwrap_or(0.0);
                let mins: f64 = parts[1].parse().unwrap_or(0.0);
                let secs: f64 = parts[2].parse().unwrap_or(0.0);
                hours * 3600.0 + mins * 60.0 + secs
            }
            _ => s.parse().unwrap_or(0.0),
        }
    } else {
        s.parse().unwrap_or(0.0)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    }
}
