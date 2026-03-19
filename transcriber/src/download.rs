use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::error::{Error, Result};

/// Maximum bytes to read from yt-dlp stdout/stderr (10 MB).
const MAX_SUBPROCESS_OUTPUT: usize = 10_000_000;

/// Timeout for yt-dlp subprocess (10 minutes).
const YTDLP_TIMEOUT: Duration = Duration::from_secs(600);

/// Result of downloading audio from a URL.
pub struct DownloadResult {
    pub audio_path: PathBuf,
    pub title: Option<String>,
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

    // Check yt-dlp is installed and working
    let check = tokio::process::Command::new("yt-dlp")
        .arg("--version")
        .output()
        .await;

    match check {
        Err(_) => return Err(Error::YtDlpNotFound),
        Ok(output) if !output.status.success() => return Err(Error::YtDlpNotFound),
        _ => {}
    }

    std::fs::create_dir_all(output_dir)?;

    let output_template = output_dir
        .join("%(id)s.%(ext)s")
        .to_str()
        .ok_or_else(|| {
            Error::Download("output directory path contains invalid UTF-8".into())
        })?
        .to_string();

    // Single yt-dlp call: download audio and print title + filepath
    let output = tokio::time::timeout(
        YTDLP_TIMEOUT,
        tokio::process::Command::new("yt-dlp")
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
                // Print title and final filepath, separated by newline
                "--print",
                "title",
                "--print",
                "after_move:filepath",
            ])
            .arg(url)
            .output(),
    )
    .await
    .map_err(|_| Error::Download(format!(
        "yt-dlp timed out after {} seconds", YTDLP_TIMEOUT.as_secs()
    )))?
    .map_err(Error::Io)?;

    if !output.status.success() {
        let stderr_bytes = &output.stderr[..output.stderr.len().min(4000)];
        let stderr = String::from_utf8_lossy(stderr_bytes);
        return Err(Error::Download(format!("yt-dlp failed: {stderr}")));
    }

    if output.stdout.len() > MAX_SUBPROCESS_OUTPUT {
        return Err(Error::Download("yt-dlp produced unexpectedly large output".into()));
    }

    // Parse --print output: first line is title, second is filepath
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.trim().lines();
    let title = lines.next().map(|s| s.to_string());
    let filepath_line = lines.next().unwrap_or("").to_string();

    let audio_path = if filepath_line.is_empty() {
        // Fallback: find the file in output_dir
        find_audio_file(output_dir)?
    } else {
        let candidate = PathBuf::from(&filepath_line);
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
        title,
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
        let dir = std::env::temp_dir().join("transcriber_test");
        let path = PathBuf::from("/etc/passwd");
        assert!(validate_path_in_dir(&path, &dir).is_err());
    }

    #[test]
    fn test_validate_path_in_dir_parent_traversal() {
        let dir = std::env::temp_dir().join("transcriber_test");
        let path = dir.join("..").join("..").join("etc").join("passwd");
        assert!(validate_path_in_dir(&path, &dir).is_err());
    }
}
