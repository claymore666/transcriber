use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::{debug, info};

use crate::error::{Error, Result};

/// Result of downloading audio from a URL.
pub struct DownloadResult {
    pub audio_path: PathBuf,
    pub title: Option<String>,
    pub duration: Option<f64>,
}

#[derive(Deserialize)]
struct YtDlpInfo {
    title: Option<String>,
    duration: Option<f64>,
}

/// Download audio from a URL using yt-dlp.
/// Returns the path to the downloaded audio file.
pub async fn download_audio(url: &str, output_dir: &Path) -> Result<DownloadResult> {
    info!(%url, "downloading audio");

    // Check yt-dlp is installed
    let check = tokio::process::Command::new("yt-dlp")
        .arg("--version")
        .output()
        .await;

    if check.is_err() {
        return Err(Error::YtDlpNotFound);
    }

    std::fs::create_dir_all(output_dir)?;

    let output_template = output_dir.join("%(id)s.%(ext)s");

    // First, get metadata
    let info_output = tokio::process::Command::new("yt-dlp")
        .args([
            "--dump-json",
            "--no-download",
            url,
        ])
        .output()
        .await?;

    let info: Option<YtDlpInfo> = if info_output.status.success() {
        serde_json::from_slice(&info_output.stdout).ok()
    } else {
        None
    };

    // Download best audio, extract as WAV for maximum compatibility
    let output = tokio::process::Command::new("yt-dlp")
        .args([
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--no-playlist",
            "--output", output_template.to_str().unwrap_or("%(id)s.%(ext)s"),
            "--print", "after_move:filepath",
            url,
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Download(format!("yt-dlp failed: {stderr}")));
    }

    let audio_path_str = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_string();

    // yt-dlp --print after_move:filepath gives us the final path
    let audio_path = if audio_path_str.is_empty() {
        // Fallback: find the file in output_dir
        find_audio_file(output_dir)?
    } else {
        PathBuf::from(&audio_path_str)
    };

    if !audio_path.exists() {
        return Err(Error::Download(format!(
            "downloaded file not found at {}",
            audio_path.display()
        )));
    }

    debug!(path = %audio_path.display(), "audio downloaded");

    Ok(DownloadResult {
        audio_path,
        title: info.as_ref().and_then(|i| i.title.clone()),
        duration: info.as_ref().and_then(|i| i.duration),
    })
}

/// Find the most recently modified audio file in a directory.
fn find_audio_file(dir: &Path) -> Result<PathBuf> {
    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "wav" | "mp3" | "ogg" | "m4a" | "opus" | "flac") {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        if best.as_ref().is_none_or(|(_, t)| modified > *t) {
                            best = Some((path, modified));
                        }
                    }
                }
            }
        }
    }

    best.map(|(p, _)| p)
        .ok_or_else(|| Error::Download("no audio file found after download".into()))
}
