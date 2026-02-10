use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::{debug, info, warn};

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

/// Validate that a string looks like a URL.
/// Rejects anything that isn't http:// or https://.
fn validate_url(url: &str) -> Result<()> {
    let trimmed = url.trim();
    if trimmed.starts_with("https://") || trimmed.starts_with("http://") {
        Ok(())
    } else {
        Err(Error::Download(format!(
            "invalid URL (must start with http:// or https://): {trimmed}"
        )))
    }
}

/// Download audio from a URL using yt-dlp.
/// Returns the path to the downloaded audio file.
///
/// # Security
/// - URL is validated to start with http:// or https://
/// - Arguments are passed to yt-dlp via `.arg()` (no shell expansion)
/// - `--no-exec` prevents yt-dlp from running post-processing commands
/// - Downloaded file path is validated to be inside output_dir
pub async fn download_audio(url: &str, output_dir: &Path) -> Result<DownloadResult> {
    validate_url(url)?;

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

    let output_template = output_dir
        .join("%(id)s.%(ext)s")
        .to_str()
        .ok_or_else(|| {
            Error::Download("output directory path contains invalid UTF-8".into())
        })?
        .to_string();

    // First, get metadata
    let info_output = tokio::process::Command::new("yt-dlp")
        .args(["--dump-json", "--no-download", "--no-exec"])
        .arg(url)
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
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--no-playlist",
            "--no-exec",
            "--output",
            &output_template,
            "--print",
            "after_move:filepath",
        ])
        .arg(url)
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Limit error message length to avoid dumping huge stderr
        let stderr_truncated: String = stderr.chars().take(1000).collect();
        return Err(Error::Download(format!("yt-dlp failed: {stderr_truncated}")));
    }

    let audio_path_str = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_string();

    // yt-dlp --print after_move:filepath gives us the final path
    let audio_path = if audio_path_str.is_empty() {
        // Fallback: find the file in output_dir
        find_audio_file(output_dir)?
    } else {
        let candidate = PathBuf::from(&audio_path_str);
        // Validate the returned path is inside output_dir
        validate_path_in_dir(&candidate, output_dir)?;
        candidate
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

/// Normalize a path by resolving `.` and `..` components without touching the filesystem.
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                parts.pop();
            }
            Component::CurDir => {}
            other => parts.push(other),
        }
    }
    parts.iter().collect()
}

/// Validate that a path is inside the expected directory (prevents path traversal).
fn validate_path_in_dir(path: &Path, expected_dir: &Path) -> Result<()> {
    // Try filesystem canonicalization first (most reliable when paths exist)
    let canonical_dir = expected_dir
        .canonicalize()
        .unwrap_or_else(|_| normalize_path(expected_dir));
    let canonical_path = path
        .canonicalize()
        .unwrap_or_else(|_| normalize_path(path));

    if canonical_path.starts_with(&canonical_dir) {
        Ok(())
    } else {
        warn!(
            path = %path.display(),
            expected_dir = %expected_dir.display(),
            "downloaded file path outside expected directory"
        );
        Err(Error::Download(
            "downloaded file path is outside the expected output directory".into(),
        ))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_url_https() {
        assert!(validate_url("https://youtube.com/watch?v=abc").is_ok());
    }

    #[test]
    fn test_validate_url_http() {
        assert!(validate_url("http://example.com/audio.mp3").is_ok());
    }

    #[test]
    fn test_validate_url_rejects_no_scheme() {
        assert!(validate_url("youtube.com/watch?v=abc").is_err());
    }

    #[test]
    fn test_validate_url_rejects_file_scheme() {
        assert!(validate_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_validate_url_rejects_empty() {
        assert!(validate_url("").is_err());
    }

    #[test]
    fn test_validate_url_rejects_command() {
        assert!(validate_url("$(whoami)").is_err());
    }

    #[test]
    fn test_validate_url_rejects_pipe() {
        assert!(validate_url("| cat /etc/passwd").is_err());
    }

    #[test]
    fn test_validate_path_in_dir_valid() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_file.wav");
        assert!(validate_path_in_dir(&path, &dir).is_ok());
    }

    #[test]
    fn test_validate_path_in_dir_traversal() {
        let dir = std::env::temp_dir().join("transkribo_test");
        let path = PathBuf::from("/etc/passwd");
        assert!(validate_path_in_dir(&path, &dir).is_err());
    }

    #[test]
    fn test_validate_path_in_dir_parent_traversal() {
        let dir = std::env::temp_dir().join("transkribo_test");
        let path = dir.join("..").join("..").join("etc").join("passwd");
        assert!(validate_path_in_dir(&path, &dir).is_err());
    }
}
