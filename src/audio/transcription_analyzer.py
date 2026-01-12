"""
transcription_analyzer.py

Enhanced speech transcription and analysis module with high accuracy detection.
Handles transcription, filler detection, and exports in multiple formats.

Features:
   - High-accuracy speech-to-text conversion
   - Advanced stutter detection (repetitions, prolongations, blocks)
   - Multiple export formats (TXT, VTT, TextGrid, JSON)
   - Comprehensive analysis reports
   - Reference text comparison
"""
import os
import replicate
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

# Download required NLTK data on first run (safe & idempotent)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('words', quiet=True)  # already there, but good to have

'''
# Download required NLTK data
try:
    nltk.download("punkt")
    nltk.download("words")
except Exception as e:
    logging.warning(f"Failed to download NLTK data: {e}")
'''

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Grandfather's passage for reference (English-only; for Indic, use a different reference if needed)
GRANDFATHERS_PASSAGE = """You wish to know about my grandfather. Well, he is nearly 93 years old, yet he still thinks as swiftly as ever. He dresses himself in an old black frock coat, usually several buttons missing. A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a bit. Twice each day he plays skillfully and with zest upon a small organ. Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, “Banana oil!”. Grandfather likes to be modern in his language."""

@dataclass
class TranscriptionResult:
    """Container for transcription analysis results."""

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
        """
        Initialize with high-accuracy speech recognition.

        Args:
            model_size: Whisper model size (recommended: "large" for best accuracy)
            language: Language code ("en", "hi", or "mr")
        """
        self.language = language
        try:
            # Language-specific speech patterns
            if self.language == "en":
                self.speech_patterns = {
                    "hesitation": {
                        "single": ["uh", "um", "er", "ah", "eh", "hm", "hmm", "erm"],
                        "compound": ["uh uh", "um um", "er er", "ah ah"],
                    },
                    "discourse": {
                        "single": ["like", "well", "so", "right", "okay", "see"],
                        "compound": ["you know", "i mean", "kind of", "sort of", "you see"],
                    },
                    "pause_fillers": ["mm", "uh-huh", "mhm", "yeah"],
                    "starters": [
                        "basically",
                        "actually",
                        "literally",
                        "obviously",
                        "frankly",
                    ],
                    "repetition_markers": ["th-th", "st-st", "wh-wh", "b-b"],
                }
            elif self.language == "hi":
                self.speech_patterns = {
                    "hesitation": {
                        "single": ["उम", "अह", "एर", "हम", "हम्म"],  # um, ah, er, hm, hmm
                        "compound": ["उम उम", "अह अह"],
                    },
                    "discourse": {
                        "single": ["मतलब", "जैसे", "वो", "अच्छा", "ठीक"],
                        "compound": ["तुम जानते हो", "मेरा मतलब है", "जैसे कि"],
                    },
                    "pause_fillers": ["म्म", "अह-हuh", "म्हम", "हाँ"],
                    "starters": [
                        "बेसिकली",  # borrowed
                        "असल में",
                        "लिटरली",  # borrowed
                        "ओब्वियसली",  # borrowed
                        "फ्रैंकली",  # borrowed
                    ],
                    "repetition_markers": ["थ-थ", "स्ट-स्ट", "व्ह-व्ह", "ब-ब"],
                }
            elif self.language == "mr":
                self.speech_patterns = {
                    "hesitation": {
                        "single": ["उम", "अह", "एर", "हम", "हम्म"],  # similar to Hindi
                        "compound": ["उम उम", "अह अह"],
                    },
                    "discourse": {
                        "single": ["म्हणजे", "जसे", "ते", "चांगले", "ठीक"],
                        "compound": ["तुम्हाला माहित आहे", "माझा अर्थ आहे", "जसे की"],
                    },
                    "pause_fillers": ["म्म", "अह-हuh", "म्हम", "हो"],
                    "starters": [
                        "बेसिकली", 
                        "खरं तर",
                        "लिटरली", 
                        "ओब्वियसली", 
                        "फ्रँकली",
                    ],
                    "repetition_markers": ["थ-थ", "स्ट-स्ट", "व्ह-व्ह", "ब-ब"],
                }    
            else:
                raise ValueError(f"Unsupported language: {language}")

            # Compile regex patterns
            self._compile_patterns()

            # Prepare reference text (English-only; skip for Indic or provide Indic reference)
            if self.language == "en":
                self.reference_words = word_tokenize(GRANDFATHERS_PASSAGE.lower())
            else:
                self.reference_words = []

            # Load Indic model if needed
            '''
            if self.language in ["hi", "mr"]:
                from transformers import pipeline
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="ai4bharat/indic-conformer-600m-multilingual",
                    device=0 if torch.cuda.is_available() else -1,
                )
            '''
        except Exception as e:
            logger.error(f"Error initializing TranscriptionAnalyzer: {e}")
            raise

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.pattern_regexes = {}
        for category, patterns in self.speech_patterns.items():
            if isinstance(patterns, dict):
                self.pattern_regexes[category] = {
                    "single": re.compile(
                        r"\b(" + "|".join(patterns["single"]) + r")\b", re.IGNORECASE | re.UNICODE
                    ),
                    "compound": re.compile(
                        r"\b(" + "|".join(patterns["compound"]) + r")\b", re.IGNORECASE | re.UNICODE
                    ),
                }
            else:
                self.pattern_regexes[category] = re.compile(
                    r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE | re.UNICODE
                )

    def analyze_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_dir: Path
    ) -> TranscriptionResult:
        """
        Perform complete audio analysis with enhanced accuracy.

        Args:
            audio_data: Audio signal
            sample_rate: Audio sample rate
            output_dir: Directory for output files

        Returns:
            TranscriptionResult containing detailed analysis
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Perform enhanced transcription
            result = self.transcribe_with_enhanced_detection(audio_data, sample_rate)

            return result

        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            raise

    def transcribe_with_enhanced_detection(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe audio using language-specific model and perform enhanced speech analysis."""
        try:
            if self.language == "en":
                # Original Replicate Whisper code
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    sf.write(tmp_wav.name, audio_data, sample_rate)
                    audio_path = tmp_wav.name

                output = replicate.run(
                    "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
                    input={
                        "audio": open(audio_path, "rb"),
                        "language": "en",
                        "task": "transcribe",
                        "temperature": 0.0,
                        "initial_prompt": "Transcribe the audio exactly as spoken. Do not correct grammar, fill in gaps, or interpret unclear words. Include all hesitations, partial words, repetitions, and fillers as they are heard."
                    }
                )

                text = output.get("transcription", "")
                segments = output.get("segments", [])

                print("Replicate output: ",output)
                print("Extracted text: ",text)
                print("SEgments count: ",segments)
                
                if not segments:
                    segments = [{"text": text, "start": 0.0, "end": len(audio_data) / sample_rate}]

                os.remove(audio_path)

            else:  # hi or mr
                # Indic Conformer via transformers pipeline
                audio_input = audio_data.astype(np.float32)
                if sample_rate != 16000:
                    audio_input = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

                output = self.asr_pipeline(
                    audio_input,
                    return_timestamps="word",
                    chunk_length_s=30,
                    batch_size=8,
                    generate_kwargs={"language": self.language}
                )

                text = output["text"]
                chunks = output.get("chunks", [])
                word_timings = []
                for chunk in chunks:
                    word = chunk["text"]
                    start, end = chunk["timestamp"]
                    word_timings.append({
                        "word": word,
                        "start": start if start is not None else 0.0,
                        "end": end if end is not None else 0.0,
                        "confidence": 1.0  # Placeholder, as model doesn't provide
                    })

                segments = [{"text": text, "start": 0.0, "end": word_timings[-1]["end"] if word_timings else 0.0, "words": word_timings}]

            # Common post-processing (works for both)
            word_timings = self._extract_enhanced_word_timings(segments)
            fillers = self._detect_fillers_with_context(word_timings)
            repetitions = self._enhanced_repetition_detection(word_timings)
            pronunciation_errors = self._detect_pronunciation_errors(word_timings)
            silences = self._detect_silences(audio_data, sample_rate, segments)

            confidence = 1.0  # Placeholder
            duration = segments[-1]["end"] if segments else 0
            speech_rate = self._calculate_speech_rate(word_timings, duration)
            language_score = self._calculate_language_score(text)

            return TranscriptionResult(
                text=text,
                segments=segments,
                word_timings=word_timings,
                fillers=fillers,
                repetitions=repetitions,
                pronunciation_errors=pronunciation_errors,
                confidence=confidence,
                duration=duration,
                speech_rate=speech_rate,
                language_score=language_score,
                silences=silences,
            )

        except Exception as e:
            logger.error(f"Error in enhanced transcription: {e}")
            raise

    # The rest of the methods remain the same as in your original code (e.g., _extract_enhanced_word_timings, _detect_fillers_with_context, etc.).
    # For Indic languages, if NLTK tokenization fails on Devanagari, consider adding a simple Unicode-aware tokenizer:
    # def word_tokenize(text): return re.findall(r'\w+', text, re.UNICODE)
    # But for now, it's kept as is. If you need the full original methods, pull from your repo and merge.

    def _extract_enhanced_word_timings(self, segments: List[Dict]) -> List[Dict]:
        """Extract detailed word timings with confidence scores."""
        word_timings = []

        for segment in segments:
            words = segment.get("words", [])
            for i, word_info in enumerate(words):
                if not isinstance(word_info, dict):
                    continue

                word = word_info.get("word", "").strip().lower()
                if not word:
                    continue

                # Check for partial words and stutters
                is_partial = bool(re.search(r"-", word))
                is_stutter = bool(re.search(r"([a-z])\1+", word))

                timing = {
                    "word": word,
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "confidence": word_info.get("confidence", 0.0),
                    "is_partial": is_partial,
                    "is_stutter": is_stutter,
                    "segment_id": segment["id"],
                }

                word_timings.append(timing)

        return word_timings

    def _detect_fillers_with_context(self, word_timings: List[Dict]) -> List[Dict]:
        """Detect filler words with context analysis."""
        fillers = []
        window_size = 3

        for i, word_info in enumerate(word_timings):
            word = word_info["word"].lower()

            # Get context window
            start_idx = max(0, i - window_size)
            end_idx = min(len(word_timings), i + window_size + 1)
            context = word_timings[start_idx:end_idx]

            # Check for single-word fillers
            for category, patterns in self.pattern_regexes.items():
                if isinstance(patterns, dict):
                    if patterns["single"].search(word):
                        if self._validate_filler_context(word, context):
                            fillers.append(
                                self._create_filler_entry(word_info, category, context)
                            )

                    # Check for compound fillers
                    if i < len(word_timings) - 1:
                        compound = f"{word} {word_timings[i+1]['word']}".lower()
                        if patterns["compound"].search(compound):
                            fillers.append(
                                self._create_filler_entry(
                                    word_info,
                                    category,
                                    context,
                                    end_time=word_timings[i + 1]["end"],
                                    compound=True,
                                )
                            )

        return fillers

    def _create_filler_entry(
        self,
        word_info: Dict,
        category: str,
        context: List[Dict],
        end_time: float = None,
        compound: bool = False,
    ) -> Dict:
        """Create standardized filler entry."""
        return {
            "word": word_info["word"],
            "start": word_info["start"],
            "end": end_time or word_info["end"],
            "event_type": "filler",  # Standardized event type
            "filler_type": category,  # Specific filler category
            "compound": compound,
            "confidence": word_info["confidence"],
            "context": " ".join(w["word"] for w in context),
        }

    def _enhanced_repetition_detection(self, word_timings: List[Dict]) -> List[Dict]:
        """Improved repetition detection focusing on short interval repetitions."""
        repetitions = []
        max_time_between_repetitions = 1.0  # Maximum 1 second between repetitions

        # Process words in sequence
        i = 0
        while i < len(word_timings) - 1:
            current_word = word_timings[i]["word"].lower().strip(string.punctuation)
            if (
                not current_word or len(current_word) < 2
            ):  # Skip empty or very short words
                i += 1
                continue

            # Look for repetitions in a short time window
            repetition_sequence = [
                {
                    "word": current_word,
                    "start": word_timings[i]["start"],
                    "end": word_timings[i]["end"],
                }
            ]
            j = i + 1

            while j < len(word_timings):
                next_word = word_timings[j]["word"].lower().strip(string.punctuation)
                if not next_word:
                    j += 1
                    continue

                time_gap = word_timings[j]["start"] - word_timings[j - 1]["end"]

                # Check if this is a repetition within the time threshold
                if (
                    self._is_repetition(current_word, next_word)
                    and time_gap <= max_time_between_repetitions
                ):
                    repetition_sequence.append(
                        {
                            "word": next_word,
                            "start": word_timings[j]["start"],
                            "end": word_timings[j]["end"],
                        }
                    )
                    j += 1
                else:
                    break

            # If we found repetitions (more than 1 occurrence)
            if len(repetition_sequence) > 1:
                repetitions.append(
                    {
                        "word": current_word,
                        "pattern": repetition_sequence,
                        "count": len(repetition_sequence),
                        "start": repetition_sequence[0]["start"],
                        "end": repetition_sequence[-1]["end"],
                        "event_type": "repetition",
                        "repetition_type": (
                            "continuous" if len(repetition_sequence) > 2 else "simple"
                        ),
                        "confidence": 0.9,  # High confidence for detected repetitions
                    }
                )
                i = j  # Skip the words we've already processed
            else:
                i += 1

        return repetitions

    def _is_repetition(self, word1: str, word2: str) -> bool:
        """Check if two words represent a repetition."""
        # Exact match
        if word1 == word2:
            return True

        # Check for partial word repetitions (e.g., "st-st")
        if '-' in word1 or '-' in word2:
            if word1.split('-')[0] == word2.split('-')[0]:
                return True

        # Levenshtein similarity for close matches
        if Levenshtein.ratio(word1, word2) > 0.8:
            return True

        return False

    # Add stubs for other methods if not retrieved (e.g., _detect_pronunciation_errors, _detect_silences, _calculate_speech_rate, _calculate_language_score, _validate_filler_context)
    # These can be copied from your original code. For example:
    def _detect_pronunciation_errors(self, word_timings: List[Dict]) -> List[Dict]:
        # Placeholder - adapt for Indic if needed
        return []

    def _detect_silences(self, audio_data: np.ndarray, sample_rate: int, segments: List[Dict]) -> List[Dict]:
        # Placeholder - use librosa to detect silences
        return []

    def _calculate_speech_rate(self, word_timings: List[Dict], duration: float) -> float:
        if duration == 0:
            return 0.0
        return len(word_timings) / duration * 60  # words per minute

    def _calculate_language_score(self, text: str) -> float:
        # Placeholder - for English; for Indic, use language-specific metrics
        return 1.0

    def _validate_filler_context(self, word: str, context: List[Dict]) -> bool:
        # Placeholder - always true for now
        return True
