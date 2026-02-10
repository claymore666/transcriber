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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_transcript() -> Transcript {
        Transcript {
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 2.5,
                    text: " Hello world.".into(),
                    speaker_turn: false,
                    no_speech_probability: 0.1,
                    words: Some(vec![
                        Word { text: " Hello".into(), start: 0.0, end: 1.0, probability: 0.95 },
                        Word { text: " world.".into(), start: 1.0, end: 2.5, probability: 0.90 },
                    ]),
                },
                Segment {
                    start: 3.0,
                    end: 5.5,
                    text: " How are you?".into(),
                    speaker_turn: true,
                    no_speech_probability: 0.05,
                    words: None,
                },
            ],
            language: "en".into(),
            duration: 5.5,
            model: "large-v3".into(),
            source_url: Some("https://example.com/video".into()),
            source_title: Some("Test Video".into()),
        }
    }

    #[test]
    fn test_text_output() {
        let t = sample_transcript();
        assert_eq!(t.text(), "Hello world. How are you?");
    }

    #[test]
    fn test_text_empty_transcript() {
        let t = Transcript {
            segments: vec![],
            language: "en".into(),
            duration: 0.0,
            model: "tiny".into(),
            source_url: None,
            source_title: None,
        };
        assert_eq!(t.text(), "");
    }

    #[test]
    fn test_text_single_segment() {
        let t = Transcript {
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: " Just one segment.".into(),
                speaker_turn: false,
                no_speech_probability: 0.0,
                words: None,
            }],
            language: "en".into(),
            duration: 1.0,
            model: "tiny".into(),
            source_url: None,
            source_title: None,
        };
        assert_eq!(t.text(), "Just one segment.");
    }

    #[test]
    fn test_srt_format() {
        let t = sample_transcript();
        let srt = t.to_srt();

        assert!(srt.starts_with("1\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
        assert!(srt.contains("Hello world."));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("00:00:03,000 --> 00:00:05,500"));
        assert!(srt.contains("How are you?"));
    }

    #[test]
    fn test_srt_empty() {
        let t = Transcript {
            segments: vec![],
            language: "en".into(),
            duration: 0.0,
            model: "tiny".into(),
            source_url: None,
            source_title: None,
        };
        assert_eq!(t.to_srt(), "");
    }

    #[test]
    fn test_vtt_format() {
        let t = sample_transcript();
        let vtt = t.to_vtt();

        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.500"));
        assert!(vtt.contains("Hello world."));
        assert!(vtt.contains("00:00:03.000 --> 00:00:05.500"));
        assert!(vtt.contains("How are you?"));
    }

    #[test]
    fn test_vtt_header() {
        let t = Transcript {
            segments: vec![],
            language: "en".into(),
            duration: 0.0,
            model: "tiny".into(),
            source_url: None,
            source_title: None,
        };
        assert_eq!(t.to_vtt(), "WEBVTT\n\n");
    }

    #[test]
    fn test_json_roundtrip() {
        let t = sample_transcript();
        let json = t.to_json().unwrap();
        let deserialized: Transcript = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.segments.len(), 2);
        assert_eq!(deserialized.language, "en");
        assert_eq!(deserialized.duration, 5.5);
        assert_eq!(deserialized.model, "large-v3");
        assert_eq!(deserialized.source_url.as_deref(), Some("https://example.com/video"));
        assert_eq!(deserialized.segments[0].text, " Hello world.");
        assert_eq!(deserialized.segments[1].speaker_turn, true);
    }

    #[test]
    fn test_json_pretty() {
        let t = sample_transcript();
        let json = t.to_json_pretty().unwrap();
        assert!(json.contains('\n'));
        assert!(json.contains("  ")); // indentation
    }

    #[test]
    fn test_json_words_present() {
        let t = sample_transcript();
        let json = t.to_json().unwrap();
        assert!(json.contains("\"probability\""));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_srt_time_formatting() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(61.123), "00:01:01,123");
        assert_eq!(format_srt_time(3661.999), "01:01:01,999");
    }

    #[test]
    fn test_vtt_time_formatting() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(1.5), "00:00:01.500");
        assert_eq!(format_vtt_time(61.123), "00:01:01.123");
        assert_eq!(format_vtt_time(3661.999), "01:01:01.999");
    }

    #[test]
    fn test_srt_vs_vtt_separator() {
        // SRT uses comma, VTT uses period
        assert!(format_srt_time(1.5).contains(','));
        assert!(format_vtt_time(1.5).contains('.'));
        assert!(!format_srt_time(1.5).contains('.'));
        // Note: SRT format uses : which also appears in VTT
    }

    #[test]
    fn test_segment_numbering_srt() {
        let t = Transcript {
            segments: (0..5)
                .map(|i| Segment {
                    start: i as f64,
                    end: (i + 1) as f64,
                    text: format!(" Segment {i}"),
                    speaker_turn: false,
                    no_speech_probability: 0.0,
                    words: None,
                })
                .collect(),
            language: "en".into(),
            duration: 5.0,
            model: "tiny".into(),
            source_url: None,
            source_title: None,
        };
        let srt = t.to_srt();
        for i in 1..=5 {
            assert!(srt.contains(&format!("{i}\n")));
        }
    }
}
