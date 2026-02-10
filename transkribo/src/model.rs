use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, warn};

use crate::config::Model;
use crate::error::{Error, Result};

const HUGGINGFACE_BASE: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

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

    // Write to a temp file first, then rename (atomic-ish)
    let tmp_path = dest.with_extension("bin.part");
    let mut file = std::fs::File::create(&tmp_path)?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    use std::io::Write;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush()?;
    drop(file);

    // Verify we got something reasonable
    let file_size = std::fs::metadata(&tmp_path)?.len();
    if file_size < 1_000_000 {
        std::fs::remove_file(&tmp_path).ok();
        return Err(Error::ModelDownload(format!(
            "downloaded file too small ({file_size} bytes) — likely an error page"
        )));
    }

    std::fs::rename(&tmp_path, dest)?;
    pb.finish_with_message("Download complete");

    if total_size > 0 && file_size != total_size {
        warn!(
            expected = total_size,
            actual = file_size,
            "file size mismatch — model may be corrupt"
        );
    }

    info!(path = %dest.display(), size = file_size, "model saved");
    Ok(())
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
