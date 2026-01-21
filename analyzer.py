import os
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import Levenshtein
import re
import inflect
import base64
from src.utils.audio_utils import load_audio, normalize_audio, apply_noise_reduction
from src.audio.transcription_analyzer import TranscriptionAnalyzer, TranscriptionResult
from src.visualization.speech_visualizer import SpeechVisualizer
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize inflect engine for number conversion
p = inflect.engine()

# Reference passages for different languages
REFERENCE_PASSAGES = {
    "en": """You wish to know about my grandfather. Well, he is nearly 93 years old, yet he still thinks as swiftly as ever. He dresses himself in an old black frock coat, usually several buttons missing. A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a bit. Twice each day he plays skillfully and with zest upon a small organ. Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, “Banana oil!”. Grandfather likes to be modern in his language.""",
    "hi": """होली रंगों का त्योहार है। ह एकता तथा मित्रता का प्रतीक है। "इस दिन चारों ओर रंग-रोग उल्लास तथा उमंग का वातावरण होता है| यह फाल्गुन के महीने में आती है। होली के दिन साँति के समय होलिका दहन किया जाता है। लोग बुराई पर अच्छाई की विजय के प्रतीक के रूप में मनाया जाता है। अगले दिन रंग खेला जाता है। चारो और रंग, गुलाल दिखाई देता है। बच्चे रंगों से पिचकारी भड़काकर मारते हैं। प्रेम, एकता तथा सौहार्द है होली के प्राण हैं। हमें होली मनाते समय इन्ही आदर्शों को सामने रखना चाहिए।""",
    "mr": """दसरा उलटला की दिवाळीचे वेघ आपल्याला लागायला लागतात. सण याचा अर्थ खरेदी असाच जवळपास झाला आहे. कुठलाही सण असो गणपती दसरा की दिवाळी. एक दोन दिवस आधी बाजारात जायचे आणि फराळाचे पदार्थ. पैश्याने जाताना फुगलेले पाकीट साफ चपटे करून परत यायचे. आपल्या बहुतेक घरात पतीपत्नी दोघेही नोकरी करत असल्यामुळे. पुर्वीपेक्षा आज आपण सहजपणे खर्च करु शकतो. शिवाय फराळाचे पदार्थ आयते तयार मिळतात. आधी ऑर्डर देऊन दुकानात हवेते पदार्थ घरी आणणे आपण पसंद करतो. आपण लहान होतो तेव्हा दिवाळीची तयारी दसरयानंतर लगेच करायला लागायचो."""
}


class SpeechAnalyzer:
    def __init__(self, language: str = "en"):
        """Initialize all analysis components."""
        try:
            self.language = language
            self.transcriber = TranscriptionAnalyzer(model_size="medium", language=language)
            self.visualizer = SpeechVisualizer()
            logger.info(f"Speech Analyzer (Transcription-based) initialized successfully for language: {language}")
        except Exception as e:
            logger.error(f"Error initializing Speech Analyzer: {e}")
            raise

    def analyze_audio_file(self, file_path: str, language: str = "en") -> dict:
        """Perform full speech analysis and return the results."""
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"results/{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            viz_dir = output_dir / "visualizations"
            transcripts_dir = output_dir / "transcripts"
            for directory in [viz_dir, transcripts_dir]:
                directory.mkdir(exist_ok=True)

            logger.info("Loading and preprocessing audio...")
            audio_data, sample_rate = self._load_and_preprocess_audio(file_path)

            logger.info("Performing transcription and analysis...")
            result = self.transcriber.analyze_audio(
                audio_data, sample_rate, transcripts_dir
            )

            logger.info("Comparing with reference passage...")
            passage_comparison = self._compare_with_reference(result.text, language=language)

            logger.info("Calculating fluency score...")
            fluency_score, severity = self._calculate_fluency_score(
                result, passage_comparison
            )

            logger.info("Generating visualizations...")
            self._generate_visualizations(audio_data, result, viz_dir)

            # Read visualization for base64
            visualization_base64 = None
            visualization_path = viz_dir / "waveform_analysis.png"
            if visualization_path.exists():
                try:
                    with open(visualization_path, "rb") as img_file:
                        visualization_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Could not read visualization: {e}")

            # Format combined stutter events
            stutter_events = self._format_stutter_events(result)

            # Build the full results dictionary
            full_results = {
                "transcription": result.text,
                "stutter_events": stutter_events,
                "fluency_score": fluency_score,
                "num_repetitions": len([e for e in stutter_events if e["type"] == "repetition"]),
                "num_fillers": len([e for e in stutter_events if e["type"] == "filler"]),
                "num_prolongations": len([e for e in stutter_events if e["type"] == "prolongation"]),
                "num_blocks": len([e for e in stutter_events if e["type"] == "block"]),
                "passage_comparison": passage_comparison,
                "severity": severity,
            }

            # SAVE FULL JSON
            full_results_path = output_dir / "full_analysis.json"

            # Helper to convert numpy types to python types for JSON serialization
            import numpy as np
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(i) for i in obj]
                return obj

            with open(full_results_path, "w", encoding="utf-8") as f:
                json.dump(make_serializable(full_results), f, indent=2, ensure_ascii=False)

            logger.info(f"Full analysis saved to: {full_results_path}")

            return full_results
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"error": str(e)}
        
    def _load_and_preprocess_audio(self, file_path: str) -> tuple:
        """Load and preprocess audio file."""
        audio_data, sample_rate = load_audio(file_path, target_sr=16000)
        audio_data = normalize_audio(audio_data)
        audio_data = apply_noise_reduction(audio_data, sample_rate)
        return audio_data, sample_rate

    def _generate_visualizations(self, audio_data: np.ndarray, result, viz_dir: Path):
        """Generate visualizations for analysis."""
        all_events = (
            result.repetitions
            + result.fillers
            + [e for e in result.pronunciation_errors if e.get("confidence", 0) > 0.4]
            + [s for s in result.silences if s.get("is_block", False)]
        )

        fig_wave = self.visualizer.create_analysis_dashboard(
            audio_data=audio_data,
            features=result.word_timings,
            events=all_events,
            sample_rate=16000,
        )
        self.visualizer.save_visualization(fig_wave, viz_dir / "waveform_analysis.png")

    def _compare_with_reference(self, transcription: str, language: str = "en") -> dict:
        """Compare transcription with the reference passage to identify discrepancies."""
        reference_text = REFERENCE_PASSAGES.get(language, REFERENCE_PASSAGES["en"])

        # Normalize texts for comparison
        transcription_norm = self._normalize_text_for_comparison(transcription, language=language)
        reference_norm = self._normalize_text_for_comparison(reference_text, language=language)

        if language in ["hi", "mr"]:
            from indicnlp.tokenize import indic_tokenize
            transcription_words = indic_tokenize.trivial_tokenize(transcription_norm)
            reference_words = indic_tokenize.trivial_tokenize(reference_norm)
        else:
            transcription_words = transcription_norm.split()
            reference_words = reference_norm.split()

        # Calculate Levenshtein distance and similarity ratio
        distance = Levenshtein.distance(transcription_norm, reference_norm)
        similarity = Levenshtein.ratio(transcription_norm, reference_norm)

        # Identify specific discrepancies
        discrepancies = self._identify_discrepancies(transcription_norm, reference_norm)
        filtered_discrepancies = self._filter_false_positives(discrepancies)

        return {
            "distance": distance,
            "spoken_word_count": len(transcription_words),
            "reference_word_count": len(reference_words),
            "discrepancies": filtered_discrepancies,
            "discrepancy_count": len(filtered_discrepancies),
            "raw_discrepancy_count": len(discrepancies),
        }

    def _normalize_text_for_comparison(self, text: str, language: str = "en") -> str:
        """Normalize text for more accurate comparison."""
        if language == "en":
            text = text.lower()

        if language in ["hi", "mr"]:
            text = re.sub(r"[^\u0900-\u097F\s\-]", "", text)
        else:
            text = re.sub(r"[^\w\s\-]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        if language == "en":
            words = []
            for word in text.split():
                if word.isdigit():
                    try:
                        word = p.number_to_words(word).replace(" and ", " ")
                    except:
                        pass
                words.append(word)
            text = " ".join(words)

        if language == "en":
            fillers = ["um", "uh", "er", "ah", "like", "you know"]
        elif language == "hi":
            fillers = ["उम", "अह", "एर", "हम", "हम्म", "मतलब"]
        elif language == "mr":
            fillers = ["उम", "अह", "एर", "हम", "हम्म", "म्हणजे"]
        else:
            fillers = []

        for filler in fillers:
            text = re.sub(r"\b" + filler + r"\b", "", text, flags=re.UNICODE)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _identify_discrepancies(self, transcription: str, reference: str) -> list:
        """Identify specific discrepancies between transcription and reference."""
        discrepancies = []
        trans_words = transcription.split()
        ref_words = reference.split()
        alignment = self._align_texts(trans_words, ref_words)

        repetition_sequence = []
        last_word = None

        for i, (trans_idx, ref_idx) in enumerate(alignment):
            if trans_idx is not None:
                current_word = trans_words[trans_idx]
                if last_word == current_word:
                    repetition_sequence.append(current_word)
                else:
                    if len(repetition_sequence) > 1:
                        discrepancies.append({
                            "type": "repetition",
                            "words": repetition_sequence.copy(),
                            "count": len(repetition_sequence),
                            "position": trans_idx - len(repetition_sequence),
                        })
                    repetition_sequence = [current_word]
                last_word = current_word

            if trans_idx is not None and ref_idx is not None:
                trans_word = trans_words[trans_idx]
                ref_word = ref_words[ref_idx]
                if self._are_words_equivalent(trans_word, ref_word):
                    continue

                if trans_word != ref_word:
                    word_similarity = Levenshtein.ratio(trans_word, ref_word)
                    has_prolongation = bool(re.search(r"(\w)\1{2,}", trans_word, re.UNICODE))
                    is_partial = "-" in trans_word

                    if word_similarity < 0.7:
                        discrepancies.append({
                            "type": "substitution",
                            "transcribed": trans_word,
                            "reference": ref_word,
                            "position": trans_idx,
                            "similarity": word_similarity,
                            "has_prolongation": has_prolongation,
                            "is_partial": is_partial,
                        })
                    elif has_prolongation:
                        discrepancies.append({
                            "type": "prolongation",
                            "transcribed": trans_word,
                            "reference": ref_word,
                            "position": trans_idx,
                        })
                    elif is_partial:
                        discrepancies.append({
                            "type": "partial_word",
                            "transcribed": trans_word,
                            "reference": ref_word,
                            "position": trans_idx,
                        })
            elif trans_idx is not None and ref_idx is None:
                trans_word = trans_words[trans_idx]
                if not self._is_common_variation(trans_word, ref_words):
                    discrepancies.append({
                        "type": "insertion",
                        "transcribed": trans_word,
                        "position": trans_idx,
                        "has_prolongation": bool(re.search(r"(\w)\1{2,}", trans_word, re.UNICODE)),
                        "is_partial": "-" in trans_word,
                    })
            elif trans_idx is None and ref_idx is not None:
                ref_word = ref_words[ref_idx]
                if not self._is_common_variation(ref_word, trans_words):
                    discrepancies.append({"type": "omission", "reference": ref_word, "position": ref_idx})

        if len(repetition_sequence) > 1:
            discrepancies.append({
                "type": "repetition",
                "words": repetition_sequence,
                "count": len(repetition_sequence),
                "position": len(trans_words) - len(repetition_sequence),
            })
        return discrepancies

    def _are_words_equivalent(self, word1: str, word2: str) -> bool:
        if word1 == word2: return True
        if word1.isdigit() or word2.isdigit():
            try:
                word1_norm = p.number_to_words(word1).replace(" and ", " ") if word1.isdigit() else word1
                word2_norm = p.number_to_words(word2).replace(" and ", " ") if word2.isdigit() else word2
                if word1_norm == word2_norm: return True
                if word1_norm.replace("-", " ") == word2_norm.replace("-", " "): return True
            except: pass
        if Levenshtein.ratio(word1, word2) > 0.85: return True
        return False

    def _is_common_variation(self, word: str, word_list: list) -> bool:
        common_variations = ["the", "a", "an", "and", "or", "but", "so", "very", "just", "really", "basically", "well", "now", "then", "you", "know", "see", "like"]
        if word.lower() in common_variations: return True
        for other_word in word_list:
            if self._are_words_equivalent(word, other_word): return True
        return False

    def _filter_false_positives(self, discrepancies: list) -> list:
        filtered = []
        for disc in discrepancies:
            if disc["type"] in ["repetition", "prolongation", "partial_word"]:
                filtered.append(disc)
                continue
            if disc["type"] == "substitution":
                if disc.get("has_prolongation", False) or disc.get("is_partial", False):
                    filtered.append(disc)
                    continue
                if disc.get("similarity", 1.0) < 0.4:
                    filtered.append(disc)
                    continue
                continue
            if disc["type"] == "insertion":
                if disc.get("has_prolongation", False) or disc.get("is_partial", False):
                    filtered.append(disc)
                    continue
                filtered.append(disc)
                continue
            filtered.append(disc)
        return filtered

    def _align_texts(self, transcribed: list, reference: list) -> list:
        m, n = len(transcribed), len(reference)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self._are_words_equivalent(transcribed[i - 1], reference[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    cost = 0.5 if self._is_potential_stutter(transcribed[i - 1]) else 1
                    dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i - 1][j] + cost, dp[i][j - 1] + 1)
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and self._are_words_equivalent(transcribed[i - 1], reference[j - 1]):
                alignment.append((i - 1, j - 1)); i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append((i - 1, j - 1)); i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + (0.5 if self._is_potential_stutter(transcribed[i - 1]) else 1):
                alignment.append((i - 1, None)); i -= 1
            else:
                alignment.append((None, j - 1)); j -= 1
        return list(reversed(alignment))

    def _is_potential_stutter(self, word: str) -> bool:
        if "-" in word: return True
        if re.search(r"(\w)\1{2,}", word, re.UNICODE): return True
        return False

    def _calculate_fluency_score(self, result, passage_comparison) -> tuple:
        try:
            total_syllables = max(1, len(result.text.split()))
            weighted_stutters = (len(result.repetitions) * 1.0 +
                                len(result.fillers) * 0.5 +
                                passage_comparison["discrepancy_count"] * 1.5)
            percent_ss = min(100, (weighted_stutters / total_syllables) * 100)
            fluency_score = 100 - int(percent_ss)
            severity = "Moderate" # simplified
            return fluency_score, severity
        except Exception:
            return 50, "Moderate"

    def _format_stutter_events(self, result) -> list:
        formatted_events = []
        for rep in result.repetitions:
            formatted_events.append({
                "type": "repetition", "subtype": rep.get("repetition_type", "simple"),
                "start": rep.get("start", 0), "end": rep.get("end", 0),
                "duration": rep.get("end", 0) - rep.get("start", 0),
                "text": rep.get("word", ""), "count": rep.get("count", 1),
                "confidence": rep.get("confidence", 0.0),
            })
        for filler in result.fillers:
            formatted_events.append({
                "type": "filler", "subtype": filler.get("filler_type", "hesitation"),
                "start": filler.get("start", 0), "end": filler.get("end", 0),
                "duration": filler.get("end", 0) - filler.get("start", 0),
                "text": filler.get("word", ""), "confidence": filler.get("confidence", 0.0),
            })
        formatted_events.sort(key=lambda x: x["start"])
        return formatted_events
