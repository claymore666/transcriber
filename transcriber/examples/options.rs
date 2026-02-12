//! Transcribe with custom model, language, and word timestamps.
//!
//! Usage: cargo run --example options -- path/to/audio.mp3

use transcriber::{Model, TranscribeOptions};

#[tokio::main]
async fn main() -> transcriber::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: options <audio-file>");

    let opts = TranscribeOptions::new()
        .model(Model::Small)
        .language("en")?
        .word_timestamps(true)
        .beam_size(5)?;

    let transcript = transcriber::transcribe_file_with_options(&path, &opts).await?;

    for segment in &transcript.segments {
        println!("[{:.1}s - {:.1}s] {}", segment.start, segment.end, segment.text.trim());

        if let Some(words) = &segment.words {
            for word in words {
                println!("    {:.2}s  {} (p={:.2})", word.start, word.text.trim(), word.probability);
            }
        }
    }

    Ok(())
}
