//! Transcribe a local audio file and print the text.
//!
//! Usage: cargo run --example basic -- path/to/audio.mp3

#[tokio::main]
async fn main() -> transcriber::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: basic <audio-file>");

    let transcript = transcriber::transcribe_file(&path).await?;

    println!("{}", transcript.text());

    Ok(())
}
