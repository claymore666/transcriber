use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use crate::config::Model;
use crate::error::{Error, Result};

const HUGGINGFACE_BASE: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Maximum model file size (5 GB). The largest whisper model (large-v3) is ~2.9 GB.
const MAX_MODEL_BYTES: u64 = 5_000_000_000;

/// Ensure a model is available locally, downloading if necessary.
/// Returns the path to the model file.
pub async fn ensure_model(model: &Model, cache_dir: &Path) -> Result<PathBuf> {
    match model {
        Model::Custom(path) => {
            if path.exists() {
                Ok(path.clone())
            } else {
                Err(Error::ModelNotFound { path: path.clone() })
            }
        }
        _ => {
            let filename = model.filename();
            let model_path = cache_dir.join(&filename);

            if model_path.exists() {
                info!(path = %model_path.display(), "model already cached");
                return Ok(model_path);
            }

            std::fs::create_dir_all(cache_dir).map_err(|e| {
                Error::Model(format!("failed to create cache dir {}: {e}", cache_dir.display()))
            })?;

            let url = format!("{HUGGINGFACE_BASE}/{filename}");
            info!(%url, "downloading model");
            download_model(&url, &model_path).await?;

            Ok(model_path)
        }
    }
}

async fn download_model(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await?
        .error_for_status()
        .map_err(|e| Error::ModelDownload(format!("HTTP error: {e}")))?;

    let total_size = response.content_length().unwrap_or(0);

    // Reject obviously wrong Content-Length before downloading
    if total_size > MAX_MODEL_BYTES {
        return Err(Error::ModelDownload(format!(
            "model file too large ({total_size} bytes, max {MAX_MODEL_BYTES})"
        )));
    }

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .expect("valid template")
            .progress_chars("#>-"),
    );
    pb.set_message(format!(
        "Downloading {}",
        dest.file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));

    // Write to a unique temp file to avoid concurrent download corruption.
    // PartFileGuard ensures the .part file is cleaned up on any error.
    let tmp_path = dest.with_extension(format!("bin.part.{}", std::process::id()));
    let mut _part_guard = PartFileGuard { path: &tmp_path, armed: true };
    let mut file = std::fs::File::create(&tmp_path)?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    use std::io::Write;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        downloaded += chunk.len() as u64;
        if downloaded > MAX_MODEL_BYTES {
            return Err(Error::ModelDownload(format!(
                "download exceeded max size ({MAX_MODEL_BYTES} bytes)"
            )));
        }
        file.write_all(&chunk)?;
        pb.set_position(downloaded);
    }

    file.flush()?;
    drop(file);

    // Verify the download before moving into cache
    let file_size = std::fs::metadata(&tmp_path)?.len();
    if file_size < 1_000_000 {
        return Err(Error::ModelDownload(format!(
            "downloaded file too small ({file_size} bytes) — likely an error page"
        )));
    }

    if total_size > 0 && file_size != total_size {
        return Err(Error::ModelDownload(format!(
            "file size mismatch (expected {total_size} bytes, got {file_size}) — download may be corrupt"
        )));
    }

    // All checks passed — move into cache (disarm the cleanup guard)
    std::fs::rename(&tmp_path, dest)?;
    _part_guard.disarm();
    pb.finish_with_message("Download complete");

    info!(path = %dest.display(), size = file_size, "model saved");
    Ok(())
}

/// RAII guard that removes a .part file on drop unless disarmed.
struct PartFileGuard<'a> {
    path: &'a Path,
    armed: bool,
}

impl<'a> PartFileGuard<'a> {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PartFileGuard<'_> {
    fn drop(&mut self) {
        if self.armed && self.path.exists() {
            std::fs::remove_file(self.path).ok();
        }
    }
}

/// List all cached models.
pub fn list_cached_models(cache_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return Vec::new();
    };

    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "bin")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Model;
    use std::fs;

    #[test]
    fn test_list_cached_models_empty_dir() {
        let tmp = std::env::temp_dir().join("transcriber_test_empty_cache");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let models = list_cached_models(&tmp);
        assert!(models.is_empty());

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_list_cached_models_nonexistent_dir() {
        let models = list_cached_models(Path::new("/nonexistent/path"));
        assert!(models.is_empty());
    }

    #[test]
    fn test_list_cached_models_finds_bin_files() {
        let tmp = std::env::temp_dir().join("transcriber_test_list_cache");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        // Create fake model files
        fs::write(tmp.join("ggml-tiny.bin"), b"fake model").unwrap();
        fs::write(tmp.join("ggml-base.bin"), b"fake model").unwrap();
        fs::write(tmp.join("ggml-tiny.bin.part"), b"partial").unwrap(); // should be excluded
        fs::write(tmp.join("readme.txt"), b"not a model").unwrap(); // should be excluded

        let models = list_cached_models(&tmp);
        assert_eq!(models.len(), 2);
        assert!(models.iter().all(|p| p.extension().unwrap() == "bin"));

        fs::remove_dir_all(&tmp).ok();
    }

    #[tokio::test]
    async fn test_ensure_model_custom_exists() {
        let tmp = std::env::temp_dir().join("transcriber_test_custom_model.bin");
        fs::write(&tmp, b"fake model data").unwrap();

        let model = Model::Custom(tmp.clone());
        let result = ensure_model(&model, Path::new("/unused")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), tmp);

        fs::remove_file(&tmp).ok();
    }

    #[tokio::test]
    async fn test_ensure_model_custom_not_found() {
        let model = Model::Custom(PathBuf::from("/nonexistent/model.bin"));
        let result = ensure_model(&model, Path::new("/unused")).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn test_ensure_model_uses_cache() {
        let tmp = std::env::temp_dir().join("transcriber_test_model_cache");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        // Pre-populate cache with a fake model
        let model_path = tmp.join("ggml-tiny.bin");
        fs::write(&model_path, b"fake cached model").unwrap();

        let model = Model::Tiny;
        let result = ensure_model(&model, &tmp).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);

        fs::remove_dir_all(&tmp).ok();
    }
}
