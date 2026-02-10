use serde::{Deserialize, Serialize};

/// A single word with timing and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

/// A transcript segment (sentence/phrase).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub speaker_turn: bool,
    pub no_speech_probability: f32,
    pub words: Option<Vec<Word>>,
}

/// Complete transcription result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub segments: Vec<Segment>,
    pub language: String,
    pub duration: f64,
    pub model: String,
    pub source_url: Option<String>,
    pub source_title: Option<String>,
}

impl Transcript {
    /// Full text (all segments concatenated).
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Format as SRT subtitles.
    pub fn to_srt(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.segments.iter().enumerate() {
            out.push_str(&format!("{}\n", i + 1));
            out.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(seg.start),
                format_srt_time(seg.end)
            ));
            out.push_str(seg.text.trim());
            out.push_str("\n\n");
        }
        out
    }

    /// Format as WebVTT subtitles.
    pub fn to_vtt(&self) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for seg in &self.segments {
            out.push_str(&format!(
                "{} --> {}\n",
                format_vtt_time(seg.start),
                format_vtt_time(seg.end)
            ));
            out.push_str(seg.text.trim());
            out.push_str("\n\n");
        }
        out
    }

    /// Format as JSON.
    pub fn to_json(&self) -> crate::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Format as pretty-printed JSON.
    pub fn to_json_pretty(&self) -> crate::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Format seconds as SRT timestamp: HH:MM:SS,mmm
fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1_000;
    let ms = total_ms % 1_000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Format seconds as VTT timestamp: HH:MM:SS.mmm
fn format_vtt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1_000;
    let ms = total_ms % 1_000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}
