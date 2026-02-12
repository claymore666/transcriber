//! Output a transcript as SRT, WebVTT, and JSON.
//!
//! Usage: cargo run --example formats -- path/to/audio.mp3

#[tokio::main]
async fn main() -> transcriber::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: formats <audio-file>");

    let transcript = transcriber::transcribe_file(&path).await?;

    println!("=== SRT ===\n{}", transcript.to_srt());
    println!("=== WebVTT ===\n{}", transcript.to_vtt());
    println!("=== JSON ===\n{}", transcript.to_json_pretty()?);

    Ok(())
}
