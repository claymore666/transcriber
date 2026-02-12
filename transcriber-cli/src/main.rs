use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use transcriber::{Language, Model, TranscribeOptions};

#[derive(Parser)]
#[command(name = "transcriber", about = "Transcribe audio/video from URL or file")]
struct Cli {
    /// URL or local file path to transcribe.
    #[arg(required_unless_present_any = ["list_models", "download_model", "list_languages"])]
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

    /// Disable voice activity detection.
    #[arg(long)]
    no_vad: bool,

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

    if cli.list_languages {
        println!("{:<6} {}", "CODE", "LANGUAGE");
        println!("{:<6} {}", "----", "--------");
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
        println!("{:<16} {}", "MODEL", "SIZE");
        println!("{:<16} {}", "-----", "----");
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

    let input = cli.input.unwrap();

    // Build options
    let model = match Model::parse_name(&cli.model) {
        Some(m) => m,
        None => {
            // Try as custom model path
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
        .vad(!cli.no_vad)
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
    if let Some(dir) = cli.cache_dir {
        opts = opts.cache_dir(dir);
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

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    }
}
