"""
audio_recorder.py

This module provides audio recording functionality for the stutter detection system.
It handles real-time audio capture, buffer management, and basic audio preprocessing.
"""

import numpy as np
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None
from typing import Optional, Tuple, List
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path

from src.audio.audio_config import AudioConfig, AudioDeviceConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioRecordingError(Exception):
    """Custom exception for audio recording errors."""
    pass


class AudioRecorder:
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device_config: Optional[AudioDeviceConfig] = None,
    ):
        self.config = config or AudioConfig()
        self.device_config = device_config or AudioDeviceConfig()

        # Validate configurations
        if not self.config.validate():
            raise AudioRecordingError("Invalid audio configuration")
        if sd and not self.device_config.setup_devices():
            raise AudioRecordingError("Failed to setup audio device")

        # Initialize state
        self.is_recording = False
        self.is_paused = False
        self._audio_buffer = queue.Queue()
        self._recording_thread = None

        # Audio monitoring
        self._audio_levels: List[float] = []
        self._silence_counter = 0

        logger.info("AudioRecorder initialized (PortAudio available: %s)", sd is not None)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status
    ) -> None:
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self.is_recording and not self.is_paused:
            audio_level = np.max(np.abs(indata))
            self._audio_levels.append(audio_level)
            if audio_level < self.config.silence_threshold:
                self._silence_counter += 1
            else:
                self._silence_counter = 0
            self._audio_buffer.put(indata.copy())

    def start_recording(self) -> None:
        if not sd:
            raise AudioRecordingError("sounddevice/PortAudio not available")
        if self.is_recording:
            raise AudioRecordingError("Recording is already in progress")

        try:
            self._audio_buffer = queue.Queue()
            self._audio_levels = []
            self._silence_counter = 0
            self.is_paused = False

            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype.value,
                blocksize=self.config.block_size,
                callback=self._audio_callback,
                device=self.device_config.device_id,
            )

            self.stream.start()
            self.is_recording = True
            logger.info("Recording started")

        except Exception as e:
            raise AudioRecordingError(f"Failed to start recording: {e}")

    def stop_recording(self) -> Tuple[np.ndarray, float]:
        if not self.is_recording:
            raise AudioRecordingError("No recording in progress")

        try:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()

            audio_chunks = []
            while not self._audio_buffer.empty():
                audio_chunks.append(self._audio_buffer.get())

            audio_data = np.concatenate(audio_chunks, axis=0)
            duration = len(audio_data) / self.config.sample_rate

            logger.info(f"Recording stopped. Duration: {duration:.2f} seconds")
            return audio_data, duration

        except Exception as e:
            raise AudioRecordingError(f"Failed to stop recording: {e}")

    def pause_recording(self) -> None:
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            logger.info("Recording paused")

    def resume_recording(self) -> None:
        if self.is_recording and self.is_paused:
            self.is_paused = False
            logger.info("Recording resumed")

    def get_audio_level(self) -> float:
        if not self._audio_levels:
            return 0.0
        return float(np.mean(self._audio_levels[-10:]))

    def save_recording(
        self, audio_data: np.ndarray, filepath: Optional[str] = None
    ) -> str:
        try:
            from scipy.io import wavfile
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"recording_{timestamp}.wav"
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(filepath, self.config.sample_rate, audio_data)
            logger.info(f"Recording saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            raise AudioRecordingError(f"Failed to save recording: {e}")
