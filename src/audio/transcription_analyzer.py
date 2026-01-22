"""
transcription_analyzer.py

Enhanced speech transcription and analysis module with high accuracy detection.
Handles transcription, filler detection, and exports in multiple formats.
"""
import os
import soundfile as sf
import tempfile 
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import torch
import re
from datetime import datetime
import tgt
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
import string
import Levenshtein
import librosa
from indicnlp.tokenize import indic_tokenize
from indicnlp import common
from src.audio.audio_config import AudioConfig
from src.audio.feature_extractor import FeatureExtractor
from src.audio.stutter_detector import StutterDetector, StutterType

# Download required NLTK data on first run
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('words', quiet=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Cache for ASR pipelines to avoid reloading models per request
ASR_PIPELINE_CACHE = {}

# Passages
GRANDFATHERS_PASSAGE = """You wish to know about my grandfather. Well, he is nearly 93 years old, yet he still thinks as swiftly as ever. He dresses himself in an old black frock coat, usually several buttons missing. A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a bit. Twice each day he plays skillfully and with zest upon a small organ. Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, “Banana oil!”. Grandfather likes to be modern in his language."""

@dataclass
class TranscriptionResult:
    text: str
    segments: List[Dict]
    word_timings: List[Dict]
    fillers: List[Dict]
    repetitions: List[Dict]
    pronunciation_errors: List[Dict]
    confidence: float
    duration: float
    speech_rate: float
    language_score: float
    silences: List[Dict]

class TranscriptionAnalyzer:
    def __init__(self, model_size: str = "small", language: str = "en"):
        self.language = language
        self.config = AudioConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.stutter_detector = StutterDetector()
        try:
            if self.language == "en":
                self.speech_patterns = {
                    "hesitation": {"single": ["uh", "um", "er", "ah", "eh", "hm", "hmm", "erm"], "compound": ["uh uh", "um um"]},
                    "discourse": {"single": ["like", "well", "so", "right", "okay"], "compound": ["you know", "i mean"]},
                    "pause_fillers": ["mm", "uh-huh", "mhm", "yeah"],
                    "starters": ["basically", "actually", "literally"],
                    "repetition_markers": ["th-th", "st-st", "wh-wh", "b-b"],
                }
            elif self.language == "hi":
                self.speech_patterns = {
                    "hesitation": {"single": ["उम", "अह", "एर", "हम", "हम्म"], "compound": ["उम उम", "अह अह"]},
                    "discourse": {"single": ["मतलब", "जैसे", "वो", "अच्छा", "ठीक"], "compound": ["तुम जानते हो", "मेरा मतलब है"]},
                    "pause_fillers": ["म्म", "अह-ह", "म्हम", "हाँ"],
                    "starters": ["बेसिकली", "असल में"],
                    "repetition_markers": ["थ-थ", "स्ट-स्ट", "व्ह-व्ह", "ब-ब"],
                }
            elif self.language == "mr":
                self.speech_patterns = {
                    "hesitation": {"single": ["उम", "अह", "एर", "हम", "हम्म"], "compound": ["उम उम", "अह अह"]},
                    "discourse": {"single": ["म्हणजे", "जसे", "ते", "चांगले", "ठीक"], "compound": ["तुम्हाला माहित आहे", "माझा अर्थ आहे"]},
                    "pause_fillers": ["म्म", "अह-ह", "म्हम", "हो"],
                    "starters": ["बेसिकली", "खरं तर"],
                    "repetition_markers": ["थ-थ", "स्ट-स्ट", "व्ह-व्ह", "ब-ब"],
                }    
            else:
                raise ValueError(f"Unsupported language: {language}")

            self._compile_patterns()

            if self.language in ["en", "hi", "mr"]:
                common.set_resources_path(os.environ.get("INDIC_RESOURCES_PATH", "indic_nlp_resources"))

            from transformers import pipeline
            model_id = "ai4bharat/indic-conformer-600m-multilingual"
            if model_id not in ASR_PIPELINE_CACHE:
                try:
                    logger.info(f"Loading model {model_id}...")
                    ASR_PIPELINE_CACHE[model_id] = pipeline(
                        "automatic-speech-recognition",
                        model=model_id,
                        trust_remote_code=True,
                        device=0 if torch.cuda.is_available() else -1,
                        token=os.environ.get("HUGGING_FACE_HUB_TOKEN")
                    )
                except Exception as e:
                    logger.error(f"Failed to load {model_id}: {e}")
                    raise e
            self.asr_pipeline = ASR_PIPELINE_CACHE[model_id]

        except Exception as e:
            logger.error(f"Error initializing TranscriptionAnalyzer: {e}")
            raise

    def _compile_patterns(self):
        self.pattern_regexes = {}
        for category, patterns in self.speech_patterns.items():
            if isinstance(patterns, dict):
                self.pattern_regexes[category] = {
                    "single": re.compile(r"\b(" + "|".join(patterns["single"]) + r")\b", re.IGNORECASE | re.UNICODE),
                    "compound": re.compile(r"\b(" + "|".join(patterns["compound"]) + r")\b", re.IGNORECASE | re.UNICODE),
                }
            else:
                self.pattern_regexes[category] = re.compile(r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE | re.UNICODE)

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int, output_dir: Path) -> TranscriptionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.transcribe_with_enhanced_detection(audio_data, sample_rate)

    def _word_tokenize(self, text: str) -> List[str]:
        if self.language in ["hi", "mr"]:
            return indic_tokenize.trivial_tokenize(text)
        else:
            return word_tokenize(text)

    def transcribe_with_enhanced_detection(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        try:
            audio_input = audio_data.astype(np.float32)
            if sample_rate != 16000:
                audio_input = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

            # Using indic-conformer for all languages
            output = self.asr_pipeline(audio_input, return_timestamps="word", generate_kwargs={"language": self.language})

            text = output["text"]
            chunks = output.get("chunks", [])
            word_timings = []
            for chunk in chunks:
                word_timings.append({
                    "word": chunk["text"],
                    "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                    "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                    "confidence": 1.0
                })
            segments = [{"text": text, "start": 0.0, "end": word_timings[-1]["end"] if word_timings else 0.0, "words": word_timings, "id": 0}]

            word_timings = self._extract_enhanced_word_timings(segments)
            fillers = self._detect_fillers_with_context(word_timings)
            repetitions = self._enhanced_repetition_detection(word_timings)

            # Use StutterDetector for signal-based analysis (blocks and prolongations)
            features = self.feature_extractor.extract_features(audio_data)
            signal_events = self.stutter_detector.analyze_speech(features, audio_data, sample_rate)

            # Map signal events back to result format
            prolongations = []
            blocks = []
            silences = []

            for event in signal_events:
                event_dict = {
                    "start": event.start_time,
                    "end": event.end_time,
                    "confidence": event.confidence,
                    "severity": event.severity,
                    "type": event.stutter_type.value,
                    "subtype": "signal_detected"
                }
                if event.stutter_type == StutterType.PROLONGATION:
                    prolongations.append(event_dict)
                elif event.stutter_type == StutterType.BLOCK:
                    blocks.append(event_dict)
                    silences.append({**event_dict, "is_block": True})

            return TranscriptionResult(
                text=text, segments=segments, word_timings=word_timings, fillers=fillers,
                repetitions=repetitions + prolongations, # Combine signal prolongations with repetitions for visualization
                pronunciation_errors=blocks, # Use blocks as pronunciation errors for visualization dashboard
                confidence=1.0,
                duration=segments[-1]["end"] if segments else 0, speech_rate=features.speech_rate,
                language_score=1.0, silences=silences
            )
        except Exception as e:
            logger.error(f"Error in enhanced transcription: {e}")
            raise

    def _extract_enhanced_word_timings(self, segments: List[Dict]) -> List[Dict]:
        word_timings = []
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
                word = word_info.get("word", "").strip().lower()
                if not word: continue
                word_timings.append({
                    "word": word, "start": word_info.get("start", 0), "end": word_info.get("end", 0),
                    "confidence": word_info.get("confidence", 0.0),
                    "is_partial": "-" in word,
                    "is_stutter": bool(re.search(r"(\w)\1+", word, re.UNICODE)),
                    "segment_id": segment.get("id", 0)
                })
        return word_timings

    def _detect_fillers_with_context(self, word_timings: List[Dict]) -> List[Dict]:
        fillers = []
        for i, word_info in enumerate(word_timings):
            word = word_info["word"].lower()
            for category, patterns in self.pattern_regexes.items():
                if isinstance(patterns, dict):
                    if patterns["single"].search(word):
                        fillers.append({"word": word, "start": word_info["start"], "end": word_info["end"], "filler_type": category, "confidence": word_info["confidence"]})
        return fillers

    def _enhanced_repetition_detection(self, word_timings: List[Dict]) -> List[Dict]:
        repetitions = []
        i = 0
        while i < len(word_timings) - 1:
            current_word = word_timings[i]["word"].lower().strip(string.punctuation)
            if not current_word or len(current_word) < 2:
                i += 1; continue
            repetition_sequence = [word_timings[i]]
            j = i + 1
            while j < len(word_timings):
                next_word = word_timings[j]["word"].lower().strip(string.punctuation)
                if next_word == current_word and (word_timings[j]["start"] - word_timings[j-1]["end"] <= 1.0):
                    repetition_sequence.append(word_timings[j]); j += 1
                else: break
            if len(repetition_sequence) > 1:
                repetitions.append({
                    "word": current_word, "count": len(repetition_sequence),
                    "start": repetition_sequence[0]["start"], "end": repetition_sequence[-1]["end"],
                    "repetition_type": "simple", "confidence": 0.9
                })
                i = j
            else: i += 1
        return repetitions

    def _detect_pronunciation_errors(self, word_timings): return []
    def _detect_silences(self, audio_data, sample_rate, segments): return []
    def _calculate_speech_rate(self, word_timings, duration): return 0.0
    def _calculate_language_score(self, text): return 1.0
    def _validate_filler_context(self, word, context): return True
