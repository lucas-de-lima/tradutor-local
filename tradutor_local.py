from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
import soundcard as sc
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from misaki import espeak
from misaki.espeak import EspeakG2P

BASE_DIR = Path(__file__).resolve().parent
INPUT_RATE = 48_000
WHISPER_RATE = 16_000
CAPTURE_BLOCK_SECONDS = 0.08
RMS_THRESHOLD = 0.004
SILENCE_TO_SPLIT_SECONDS = 0.40
MAX_CHUNK_SECONDS = 2.0
MIN_CHUNK_SECONDS = 0.55
MAX_ITEM_AGE_SECONDS = 5.5
MAX_SOURCE_AGE_SECONDS = 7.0
PAUSE_DROP_AFTER_SECONDS = 3.0
# When a window ends on a mid-sentence pause (no terminal punctuation), hold the
# assembled utterance a little while for a possible continuation. If the speaker
# resumes, the next chunk merges and no sentence is split; only a real stop is
# committed after the grace period.
GRACE_PAUSE_SECONDS = 2.2
# Utterance is flushed when it outgrows these context bounds, preferably at the
# last real sentence boundary inside the text.
CONTEXT_DURATION_SECONDS = 3.8
CONTEXT_WORDS = 22
# Above this many words a translation stream with no terminal punctuation is
# split, but only at a strong clause boundary (;, :, —, ,), never mid-sentence.
MAX_STREAM_UNIT_WORDS = 24
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "translategemma:4b"
KOKORO_MODEL = BASE_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = BASE_DIR / "voices-v1.0.bin"
TTS_SAMPLE_RATE = 24_000
# Pauses at sentence / clause marks inside a synthesized part.
TTS_SENTENCE_PAUSE = 0.32
TTS_CLAUSE_PAUSE = 0.15
# Pauses inserted at the end of a part so the next part does not start abruptly.
TTS_PAUSE_AFTER_SENTENCE = 0.30
TTS_PAUSE_AFTER_CLAUSE = 0.16
TTS_PAUSE_AFTER_FRAGMENT = 0.10
OUTPUT_SAMPLE_RATE_CANDIDATES = (48_000, 44_100, 24_000, 22_050)
OUTPUT_BLOCK_FRAMES = 1024
OUTPUT_QUEUE_BLOCKS = 32  # <= about 0.68 sec at 48 kHz


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    captured_at: float
    duration: float
    epoch: int
    overlap: bool = False
    speech_boundary: bool = False


@dataclass(frozen=True)
class Transcript:
    seq: int
    text: str
    language: str
    # captured_at is the end/ready time of the source utterance, used for stale-work control.
    captured_at: float
    duration: float
    epoch: int
    source_started_at: float | None = None


@dataclass(frozen=True)
class SpeechEvent:
    seq: int
    part: int
    text: str
    final: bool
    captured_at: float
    epoch: int
    error: str | None = None
    source_started_at: float | None = None


@dataclass(frozen=True)
class ASRResult:
    chunk_order: int
    chunk: AudioChunk
    text: str
    language: str
    elapsed: float
    error: str | None = None


@dataclass
class AudioPart:
    event: SpeechEvent
    audio_chunks: queue.Queue
    done: threading.Event
    cancel: threading.Event
    error: str | None = None


class StreamingAudioPlayer:
    """Continuous PortAudio output; producers enqueue small PCM blocks, callback never blocks."""

    def __init__(self, device_name: str, stop_event: threading.Event):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Falta sounddevice. Instale com: python -m pip install sounddevice") from exc

        self.sd = sd
        self.stop_event = stop_event
        devices = sd.query_devices()
        matches: list[tuple[int, dict]] = []
        fragments = [device_name]
        # Windows may localize the prefix but keep the name in parentheses.
        match = re.search(r"\(([^()]*)\)\s*$", device_name)
        if match:
            fragments.insert(0, match.group(1))
        for fragment in fragments:
            if not fragment:
                continue
            matches = [
                (idx, dict(dev)) for idx, dev in enumerate(devices)
                if int(dev.get("max_output_channels", 0)) > 0
                and fragment.casefold() in str(dev.get("name", "")).casefold()
            ]
            if matches:
                break
        if not matches:
            available = "\n".join(
                f"  - {idx}: {dev.get('name')} (saídas={dev.get('max_output_channels')})"
                for idx, dev in enumerate(devices) if int(dev.get("max_output_channels", 0)) > 0
            )
            raise RuntimeError(
                f"O dispositivo '{device_name}' não apareceu no sounddevice/PortAudio.\n"
                "Saídas disponíveis:\n" + (available or "  (nenhuma)")
            )

        self.device_index, info = matches[0]
        self.channels = 2 if int(info.get("max_output_channels", 0)) >= 2 else 1
        self.sample_rate = int(round(float(info.get("default_samplerate") or 48_000)))
        # Prefer the device's default rate; fall back to a rate PortAudio accepts.
        supported_rate = None
        for rate in (self.sample_rate, *OUTPUT_SAMPLE_RATE_CANDIDATES):
            try:
                sd.check_output_settings(
                    device=self.device_index, samplerate=rate,
                    channels=self.channels, dtype="float32"
                )
                supported_rate = int(rate)
                break
            except Exception:
                continue
        if supported_rate is None:
            raise RuntimeError(f"PortAudio não encontrou uma taxa de saída compatível para {info.get('name')}.")
        self.sample_rate = supported_rate

        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=OUTPUT_QUEUE_BLOCKS)
        self.current: np.ndarray | None = None
        self.current_offset = 0
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=OUTPUT_BLOCK_FRAMES,
            latency="low",
            callback=self._callback,
        )
        self.stream.start()
        print(
            f"[SAÍDA STREAM] {info.get('name')} | {self.sample_rate} Hz | "
            f"{self.channels} canal(is) | buffer limitado a ~{OUTPUT_QUEUE_BLOCKS * OUTPUT_BLOCK_FRAMES / self.sample_rate:.2f}s"
        )

    def _callback(self, outdata, frames, _time_info, _status):
        outdata.fill(0)
        written = 0
        with self.lock:
            while written < frames:
                if self.current is None or self.current_offset >= len(self.current):
                    try:
                        self.current = self.blocks.get_nowait()
                        self.current_offset = 0
                    except queue.Empty:
                        break
                available = len(self.current) - self.current_offset
                take = min(frames - written, available)
                if take <= 0:
                    self.current = None
                    continue
                outdata[written:written + take] = self.current[self.current_offset:self.current_offset + take]
                self.current_offset += take
                written += take
                if self.current_offset >= len(self.current):
                    self.current = None
                    self.current_offset = 0

    def clear_pending(self) -> None:
        """Drop queued samples and any not-yet-consumed block; audio already in hardware may finish briefly."""
        with self.lock:
            self.current = None
            self.current_offset = 0
            while True:
                try:
                    self.blocks.get_nowait()
                except queue.Empty:
                    break

    def enqueue_audio(self, samples: np.ndarray, sample_rate: int, captured_at: float, epoch: int,
                      state: dict) -> bool:
        if samples is None or len(samples) == 0:
            return True
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.reshape(-1)
        if sample_rate != self.sample_rate:
            from math import gcd
            divisor = gcd(int(self.sample_rate), int(sample_rate))
            data = resample_poly(data, self.sample_rate // divisor, int(sample_rate) // divisor).astype(np.float32)
        if self.channels == 2:
            data = np.column_stack((data, data)).astype(np.float32, copy=False)
        else:
            data = data[:, None].astype(np.float32, copy=False)

        for start in range(0, len(data), OUTPUT_BLOCK_FRAMES):
            if self.stop_event.is_set():
                return False
            if epoch != state.get("epoch", 0):
                return False
            block = np.ascontiguousarray(data[start:start + OUTPUT_BLOCK_FRAMES])
            while not self.stop_event.is_set():
                if epoch != state.get("epoch", 0):
                    return False
                try:
                    # Backpressure is intentional: the audio callback drains this small queue
                    # continuously. Do not discard a sentence merely because it takes >1.5 s to play.
                    self.blocks.put(block, timeout=0.08)
                    break
                except queue.Full:
                    continue
        return True

    def has_pending(self) -> bool:
        with self.lock:
            return self.current is not None or not self.blocks.empty()

    def close(self) -> None:
        try:
            self.stream.stop()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass


def list_devices() -> None:
    print("\nDISPOSITIVOS DE ENTRADA (microfones e entradas virtuais):")
    for mic in sc.all_microphones(include_loopback=True):
        print(f"  - {mic.name}")
    print("\nDISPOSITIVOS DE SAÍDA (soundcard):")
    for speaker in sc.all_speakers():
        marker = " [PADRÃO]" if speaker.name == sc.default_speaker().name else ""
        print(f"  - {speaker.name}{marker}")
    try:
        import sounddevice as sd
        print("\nDISPOSITIVOS DE SAÍDA (PortAudio/sounddevice para reprodução contínua):")
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_output_channels", 0)) > 0:
                print(f"  - {idx}: {dev['name']} (saídas={dev['max_output_channels']}, taxa padrão={dev['default_samplerate']})")
    except Exception as exc:
        print(f"\n[AVISO] Não foi possível listar sounddevice: {exc}")


def find_input_device(name_fragment: str | None):
    microphones = sc.all_microphones(include_loopback=True)
    if name_fragment:
        matches = [m for m in microphones if name_fragment.casefold() in m.name.casefold()]
    else:
        matches = [m for m in microphones if "cable output" in m.name.casefold()]
    if not matches:
        names = "\n".join(f"  - {m.name}" for m in microphones)
        raise RuntimeError(f"Não encontrei entrada contendo '{name_fragment or 'CABLE Output'}'.\n{names}")
    return matches[0]


def find_output_device(name_fragment: str | None):
    if not name_fragment:
        return sc.default_speaker()
    speakers = sc.all_speakers()
    matches = [s for s in speakers if name_fragment.casefold() in s.name.casefold()]
    if not matches:
        names = "\n".join(f"  - {s.name}" for s in speakers)
        raise RuntimeError(f"Não encontrei saída contendo '{name_fragment}'.\n{names}")
    return matches[0]


def push_latest(target_queue: queue.Queue, item) -> None:
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
            print("[TEMPO REAL] Trecho antigo descartado na captura; priorizando áudio recente.")
        except queue.Full:
            pass


def capture_worker(microphone, audio_queue: queue.Queue, stop_event: threading.Event,
                   state: dict, player: StreamingAudioPlayer) -> None:
    block_frames = int(INPUT_RATE * CAPTURE_BLOCK_SECONDS)
    buffer: list[np.ndarray] = []
    duration = 0.0
    silence_duration = 0.0
    active = False
    pause_logged = False
    buffer_started_at: float | None = None
    buffer_has_overlap = False
    overlap_samples_count = int(INPUT_RATE * 0.28)
    print(f"[ÁUDIO] Capturando de: {microphone.name}")
    print("[ÁUDIO] Pipeline por etapas ativo. Reproduza um vídeo; Ctrl+C encerra.")
    try:
        with microphone.recorder(samplerate=INPUT_RATE) as recorder:
            while not stop_event.is_set():
                block = recorder.record(numframes=block_frames)
                if block is None or len(block) == 0:
                    continue
                block = np.asarray(block, dtype=np.float32)
                rms = float(np.sqrt(np.mean(np.square(block))))
                now = time.monotonic()
                is_active = rms >= RMS_THRESHOLD
                if is_active:
                    state["last_audio_at"] = now
                    if state.get("paused", False):
                        state["resume_at"] = now
                        state["paused"] = False
                        # Keep the currently speaking phrase intact. Queued/prepared segments from
                        # before resume are rejected by their captured_at timestamp downstream.
                        pause_logged = False
                        print("[TEMPO REAL] Áudio retomado; descartando trabalho antigo, mantendo a frase já em reprodução.")
                elif (not state.get("paused", False)
                      and now - state.get("last_audio_at", now) >= PAUSE_DROP_AFTER_SECONDS):
                    state["paused"] = True
                    # Do not clear the playback device here: that chopped the current TTS phrase
                    # mid-sentence. The active phrase may finish; workers won't start old queued
                    # phrases while paused, and pre-pause tasks are rejected after resume.
                    while True:
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    if not pause_logged:
                        print("[TEMPO REAL] Silêncio prolongado: suspendendo novas falas; a frase atual pode terminar.")
                        pause_logged = True

                if is_active or active:
                    if not buffer:
                        buffer_started_at = now
                    buffer.append(block)
                    duration += len(block) / INPUT_RATE
                if is_active:
                    active = True
                    silence_duration = 0.0
                elif active:
                    silence_duration += len(block) / INPUT_RATE

                due_to_silence = active and silence_duration >= SILENCE_TO_SPLIT_SECONDS
                due_to_length = active and duration >= MAX_CHUNK_SECONDS
                if due_to_silence or due_to_length:
                    samples = np.concatenate(buffer, axis=0) if buffer else np.empty((0, 1), dtype=np.float32)
                    if duration >= MIN_CHUNK_SECONDS and len(samples):
                        item = AudioChunk(samples, buffer_started_at or time.monotonic(),
                                          len(samples) / INPUT_RATE, state.get("epoch", 0),
                                          buffer_has_overlap, speech_boundary=due_to_silence)
                        push_latest(audio_queue, item)

                    # When the hard length cap cuts continuous speech, retain a short overlap
                    # for the next Whisper window. A post-ASR deduper removes exact repeated words.
                    carry_overlap = due_to_length and is_active and len(samples) > overlap_samples_count
                    if carry_overlap:
                        tail = samples[-overlap_samples_count:].copy()
                        buffer = [tail]
                        duration = len(tail) / INPUT_RATE
                        buffer_started_at = now - duration
                        buffer_has_overlap = True
                        silence_duration = 0.0
                        active = True
                    else:
                        buffer = []
                        buffer_started_at = None
                        duration = 0.0
                        silence_duration = 0.0
                        active = False
                        buffer_has_overlap = False
    except Exception as exc:
        print(f"[ERRO DE CAPTURA] {type(exc).__name__}: {exc}")
        stop_event.set()

def language_name(language: str) -> str:
    return {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
        "pt": "Portuguese", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ru": "Russian",
        "nl": "Dutch", "ar": "Arabic",
    }.get((language or "en").lower(), (language or "en").upper())


def preflight_whisper_runtime(model: WhisperModel) -> None:
    """Force one tiny real inference at startup so CUDA DLL failures are caught before capture."""
    # Faster-Whisper can instantiate the CTranslate2 model before cuBLAS is actually loaded.
    # Consuming the generator is essential: transcribe() performs most work lazily.
    duration = 2.0
    t = np.arange(int(WHISPER_RATE * duration), dtype=np.float32) / WHISPER_RATE
    probe = (0.002 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    segments, _info = model.transcribe(
        probe, language="en", beam_size=1, vad_filter=False,
        condition_on_previous_text=False, without_timestamps=True,
    )
    list(segments)


def translation_prompt(text: str, source_language: str) -> str:
    src_name = language_name(source_language)
    src_code = (source_language or "en").lower()
    return (
        f"You are a professional {src_name} ({src_code}) to Brazilian Portuguese (pt-BR) translator. "
        "Translate the source speech naturally and faithfully into Brazilian Portuguese. "
        "Return only the translation, with no explanation, analysis, preamble, or commentary. "
        "Keep names and technical terms accurate. "
        "Preserve the source sentence and clause boundaries: keep the terminal punctuation "
        "(. ! ?) and natural clause punctuation (semicolons, colons, commas) so the spoken "
        "translation keeps the same rhythm and is never cut mid-sentence.\n\n"
        f"Source speech:\n{text}"
    )


def clean_translation(raw: str, source_text: str) -> str:
    translated = (raw or "").strip()
    if "</think>" in translated.lower():
        lowered = translated.lower()
        translated = translated[lowered.rfind("</think>") + len("</think>"):].strip()
    elif "<think>" in translated.lower():
        raise RuntimeError("Resposta contém um bloco de raciocínio sem fechamento; trecho descartado.")
    translated = re.sub(r"</?think\s*>", "", translated, flags=re.IGNORECASE).strip()
    translated = re.sub(r"\s+", " ", translated)
    if not translated:
        raise RuntimeError("O modelo retornou uma tradução vazia.")
    source_words = max(1, len(source_text.split()))
    if len(translated.split()) > max(45, source_words * 4):
        raise RuntimeError("Resposta longa demais para o trecho; possível raciocínio vazado, descartando.")
    return translated


def extract_ready_units(buffer: str, force: bool = False) -> tuple[list[str], str]:
    """Yield punctuation-delimited units, splitting a long run-on only at a real clause boundary."""
    units: list[str] = []
    remaining = buffer
    while remaining.strip():
        # Prefer complete sentences; keep the punctuation with the phrase. A closing
        # quote/bracket may follow the terminal mark.
        match = re.search(r"[.!?…](?:[\"'”’)]*)(?=\s|$)", remaining)
        if match:
            end = match.end()
            unit = remaining[:end].strip()
            remaining = remaining[end:].lstrip()
            if unit:
                units.append(unit)
            continue
        # If the fragment already ends a sentence, keep it intact until the stream sends
        # the rest of the next sentence or signals completion. Never apply the fallback to
        # a complete sentence merely because its final token has no trailing space.
        if re.search(r"[.!?…](?:[\"'”’)]*)$", remaining.strip()):
            break
        words = list(re.finditer(r"\S+", remaining))
        if len(words) >= MAX_STREAM_UNIT_WORDS:
            # Very long run-on without a terminal mark: split only at a strong clause
            # boundary so no sentence is chopped in the middle. If none exists yet, keep
            # waiting for punctuation or for the stream to finish (force).
            split = find_clause_split(remaining)
            if split is not None:
                unit = remaining[:split].strip()
                remaining = remaining[split:].lstrip()
                if unit:
                    units.append(unit)
                continue
            break
        break
    if force and remaining.strip():
        units.append(remaining.strip())
        remaining = ""
    return units, remaining


# Clause separators, strongest first. Splitting at one of these keeps each part a
# natural spoken clause instead of cutting a sentence in the middle of a clause.
_CLAUSE_MARKS = ";:—–,"


def find_clause_split(text: str, min_head_words: int = 8, min_tail_words: int = 4) -> int | None:
    """Return the index just after the best clause boundary in `text`, or None.

    Only a boundary that leaves a meaningful clause on both sides is used, so a
    long sentence is never chopped in the middle of a clause.
    """
    for sep in _CLAUSE_MARKS:
        index = text.rfind(sep)
        while index != -1:
            head = text[:index].strip()
            tail = text[index + 1:].strip()
            if len(head.split()) >= min_head_words and len(tail.split()) >= min_tail_words:
                return index + 1  # keep the separator with the head
            index = text.rfind(sep, 0, index)
    return None


def split_last_sentence(text: str) -> tuple[str, str]:
    """Split `text` at its last complete sentence boundary.

    Returns (head, tail): `head` ends at a terminal mark and is safe to commit;
    `tail` is whatever speech came after it. If there is no usable boundary (or
    the text already ends at a terminal mark), returns (text, "").
    """
    matches = list(re.finditer(r"[.!?…](?:[\"'”’)]*)(?=\s|$)", text))
    if not matches:
        return text, ""
    last = matches[-1]
    after = text[last.end():].strip()
    if not after:
        return text, ""
    return text[:last.end()].strip(), after


def _ends_with_punctuation(text: str) -> bool:
    return bool(re.search(r"[.!?…][\"'”’)]*$", text))


def part_trailing_pause(text: str) -> float:
    """Seconds of silence to leave after a spoken part so the next part does not start abruptly."""
    stripped = (text or "").rstrip()
    if not stripped:
        return 0.0
    last = stripped[-1]
    if last in ".!?…":
        return TTS_PAUSE_AFTER_SENTENCE
    if last in ";:":
        return TTS_PAUSE_AFTER_CLAUSE
    return TTS_PAUSE_AFTER_FRAGMENT


def translate_stream_ollama(text: str, source_language: str, stop_event: threading.Event) -> Iterator[str]:
    prompt = translation_prompt(text, source_language)
    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0, "num_predict": 128},
        "keep_alive": "10m",
    }
    parts: list[str] = []
    buffer = ""
    source_words = max(1, len(text.split()))
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(5, 10)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if stop_event.is_set():
                response.close()
                return
            if not line:
                continue
            packet = json.loads(line)
            if packet.get("error"):
                raise RuntimeError(str(packet["error"]))
            delta = packet.get("message", {}).get("content") or ""
            if not delta:
                if packet.get("done"):
                    break
                continue
            parts.append(delta)
            full_so_far = "".join(parts).strip()
            # A hard cap limits runaway output. Typical translated segments are well below this.
            if len(full_so_far.split()) > max(45, source_words * 4):
                response.close()
                raise RuntimeError("O modelo excedeu o tamanho esperado; saída bloqueada.")
            buffer += delta
            # A reasoning tag is unexpected for TranslateGemma. If one appears, fail closed
            # before emitting any later clause from this response.
            if re.search(r"</?think\s*>", "".join(parts), flags=re.IGNORECASE):
                raise RuntimeError("O modelo emitiu marcador de raciocínio; saída bloqueada.")
            units, buffer = extract_ready_units(buffer)
            for unit in units:
                clean = re.sub(r"\s+", " ", unit).strip()
                if clean:
                    yield clean
        units, buffer = extract_ready_units(buffer, force=True)
        for unit in units:
            if re.search(r"</?think\s*>", unit, flags=re.IGNORECASE):
                raise RuntimeError("O modelo emitiu marcador de raciocínio; saída bloqueada.")
            clean = re.sub(r"\s+", " ", unit).strip()
            if clean:
                yield clean


def translate_complete(text: str, source_language: str) -> str:
    # Used by the diagnostic command; the live path consumes the streaming chunks above.
    stop_event = threading.Event()
    pieces = list(translate_stream_ollama(text, source_language, stop_event))
    return clean_translation(" ".join(pieces), text)


def as_stereo(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return np.column_stack((data, data))
    if data.ndim == 2 and data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    if data.ndim == 2 and data.shape[1] > 2:
        return data[:, :2]
    return data


def initialize_tts(tts_workers: int = 1):
    """Load Kokoro on CPU with a bounded ONNX thread pool; reserve GPU for ASR/Ollama."""
    print("[MODELO] Carregando Kokoro TTS em CPU e processador fonético pt-BR...")
    _fallback = espeak.EspeakFallback(british=False)
    g2p = EspeakG2P(language="pt-br")
    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        logical_cpus = os.cpu_count() or 10
        options.intra_op_num_threads = max(2, min(8, logical_cpus // max(2, tts_workers)))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(KOKORO_MODEL), sess_options=options, providers=["CPUExecutionProvider"]
        )
        kokoro = Kokoro.from_session(session, str(KOKORO_VOICES))
        print(f"[RECURSOS] Kokoro: CPUExecutionProvider, {options.intra_op_num_threads} threads intra-op; TTS workers={tts_workers}.")
    except Exception as exc:
        print(f"[AVISO] Não consegui configurar a sessão CPU do Kokoro explicitamente ({exc}); usando inicialização padrão.")
        kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    return kokoro, g2p


def test_voice_output(speaker) -> int:
    try:
        kokoro, g2p = initialize_tts()
        text = "Olá! Este é um teste de áudio do tradutor local. Se você está ouvindo esta frase, a saída de voz está funcionando."
        phonemes, _ = g2p(text)
        samples, sample_rate = kokoro.create(phonemes, "pf_dora", is_phonemes=True)
        samples = as_stereo(samples)
        print(f"[DIAGNÓSTICO] Voz gerada: {len(samples) / sample_rate:.1f}s, taxa={sample_rate} Hz")
        print(f"[DIAGNÓSTICO] Reproduzindo no dispositivo: {speaker.name}")
        speaker.play(samples, samplerate=sample_rate)
        return 0
    except Exception as exc:
        print(f"[ERRO NO TESTE DE VOZ] {type(exc).__name__}: {exc}")
        return 2


def monitor_input(microphone, seconds: int = 10) -> int:
    print(f"[DIAGNÓSTICO] Medindo a entrada '{microphone.name}' por {seconds}s; reproduza um vídeo.")
    try:
        with microphone.recorder(samplerate=INPUT_RATE) as recorder:
            for second in range(1, seconds + 1):
                block = np.asarray(recorder.record(numframes=INPUT_RATE), dtype=np.float32)
                if block.size == 0:
                    print(f"  {second:02d}s: sem amostras")
                else:
                    print(f"  {second:02d}s: RMS={np.sqrt(np.mean(np.square(block))):.5f} | pico={np.max(np.abs(block)):.5f}")
        return 0
    except Exception as exc:
        print(f"[ERRO NO TESTE DE ENTRADA] {type(exc).__name__}: {exc}")
        return 2


def test_stream_output(device_name: str, tts_speed: float) -> int:
    """Test continuous output plus the new streamed Kokoro producer/consumer path."""
    stop_event = threading.Event()
    player = None
    producer = None
    try:
        kokoro, g2p = initialize_tts(1)
        player = StreamingAudioPlayer(device_name, stop_event)
        state = {"epoch": 0, "paused": False}
        text = "Olá! Este é o teste da reprodução contínua. Se você ouvir a frase inteira, o caminho de áudio novo está funcionando."
        event = SpeechEvent(1, 0, text, False, time.monotonic(), 0)
        part = AudioPart(event=event, audio_chunks=queue.Queue(maxsize=4),
                         done=threading.Event(), cancel=threading.Event())
        g2p_lock = threading.Lock()
        producer = threading.Thread(
            target=tts_synthesis_job,
            args=(part, kokoro, g2p, g2p_lock, stop_event, state, tts_speed),
            name="test-tts-producer", daemon=True,
        )
        print("[DIAGNÓSTICO STREAM] Gerando blocos de voz em paralelo com o consumidor de áudio...")
        producer.start()
        while not part.done.is_set() or not part.audio_chunks.empty():
            try:
                samples, sample_rate = part.audio_chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            if not player.enqueue_audio(samples, sample_rate, event.captured_at, event.epoch, state):
                print("[ERRO DIAGNÓSTICO STREAM] A saída contínua rejeitou um bloco de áudio.")
                return 2
        producer.join(timeout=2.0)
        # Let PortAudio drain its hardware buffer after the producer has finished.
        expected_seconds = len(text.split()) * 0.28 / max(0.5, tts_speed)
        time.sleep(min(8.0, max(1.0, expected_seconds)))
        if part.error:
            print(f"[ERRO DIAGNÓSTICO STREAM] {part.error}")
            return 2
        print("[DIAGNÓSTICO STREAM] Reprodução contínua concluída. O áudio foi audível no dispositivo escolhido?")
        return 0
    except Exception as exc:
        print(f"[ERRO DIAGNÓSTICO STREAM] {type(exc).__name__}: {exc}")
        return 2
    finally:
        stop_event.set()
        if producer and producer.is_alive():
            producer.join(timeout=1.0)
        if player is not None:
            player.close()


def transcribe_chunk(chunk_order: int, chunk: AudioChunk, whisper_model: WhisperModel,
                     source_language: str, state: dict) -> ASRResult:
    """Independent ASR job. Result ordering is restored by asr_dispatcher."""
    started = time.perf_counter()
    if (chunk.epoch != state.get("epoch", 0) or state.get("paused", False)
            or chunk.captured_at < state.get("resume_at", 0.0)):
        return ASRResult(chunk_order, chunk, "", "?", 0.0, "stale-before-asr")
    age = time.monotonic() - chunk.captured_at
    if age > MAX_ITEM_AGE_SECONDS:
        return ASRResult(chunk_order, chunk, "", "?", 0.0, "stale-before-asr")
    try:
        mono = chunk.samples.mean(axis=1) if chunk.samples.ndim == 2 else chunk.samples
        audio_16k = resample_poly(np.asarray(mono, dtype=np.float32), WHISPER_RATE, INPUT_RATE).astype(np.float32)
        language = None if source_language == "auto" else source_language
        segments, info = whisper_model.transcribe(
            audio_16k, language=language, beam_size=1, vad_filter=True,
            condition_on_previous_text=False, no_speech_threshold=0.6,
        )
        original_text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - started
        detected = getattr(info, "language", "?") or source_language
        if detected == "?" and source_language != "auto":
            detected = source_language
        return ASRResult(chunk_order, chunk, original_text, detected, elapsed)
    except Exception as exc:
        return ASRResult(chunk_order, chunk, "", "?", time.perf_counter() - started,
                         f"{type(exc).__name__}: {exc}")


def deduplicate_overlap(previous_text: str, current_text: str, max_words: int = 4) -> str:
    """Remove only an exact repeated word prefix caused by adjacent overlapping ASR windows."""
    previous_words = list(re.finditer(r"\S+", previous_text or ""))
    current_words = list(re.finditer(r"\S+", current_text or ""))
    if not previous_words or not current_words:
        return (current_text or "").strip()

    def norm(token: str) -> str:
        return re.sub(r"[^\w']", "", token, flags=re.UNICODE).casefold()

    max_overlap = min(max_words, len(previous_words), len(current_words))
    for count in range(max_overlap, 0, -1):
        left = [norm(m.group()) for m in previous_words[-count:]]
        right = [norm(m.group()) for m in current_words[:count]]
        if left == right and all(left):
            start = current_words[count - 1].end()
            return (current_text or "")[start:].lstrip(" ,;:—–-\t\r\n")
    return (current_text or "").strip()


def asr_dispatcher(audio_queue: queue.Queue, transcript_queue: queue.Queue, whisper_model: WhisperModel,
                   source_language: str, stop_event: threading.Event, state: dict,
                   sequence_meta: dict, sequence_lock: threading.Lock, workers: int) -> None:
    """Runs bounded ASR jobs in order, then combines partial windows into coherent utterances."""
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr")
    inflight: dict[Future, tuple[int, AudioChunk]] = {}
    completed: dict[int, ASRResult] = {}
    next_chunk_order = 1
    next_chunk_to_emit = 1
    next_transcript_seq = 1
    last_raw_text_by_epoch: dict[int, str] = {}
    pending = {
        "text": "", "source_started_at": 0.0, "ready_at": 0.0,
        "duration": 0.0, "epoch": 0, "language": "?", "words": 0,
    }
    grace_deadline: float | None = None

    def reset_pending() -> None:
        nonlocal grace_deadline
        grace_deadline = None
        pending.update({"text": "", "source_started_at": 0.0, "ready_at": 0.0,
                        "duration": 0.0, "epoch": 0, "language": "?", "words": 0})

    def commit_pending(reason: str, can_split: bool) -> None:
        """Send the assembled utterance to translation.

        With `can_split`, the text is committed only up to its last complete
        sentence, and the trailing fragment is kept as the start of the next
        utterance. This guarantees a sentence is never cut in the middle.
        """
        nonlocal next_transcript_seq
        text = str(pending["text"]).strip()
        if not text:
            reset_pending()
            return
        carry = ""
        if can_split:
            head, tail = split_last_sentence(text)
            if tail:
                text, carry = head, tail
        source_started_at = float(pending["source_started_at"])
        ready_at = float(pending["ready_at"])
        duration = float(pending["duration"])
        epoch = int(pending["epoch"])
        language = str(pending["language"])
        reset_pending()

        stale = (state.get("paused", False)
                 or source_started_at < state.get("resume_at", 0.0)
                 or time.monotonic() - source_started_at > MAX_SOURCE_AGE_SECONDS
                 or time.monotonic() - ready_at > MAX_ITEM_AGE_SECONDS)
        if stale:
            if carry:
                # The carried tail is newer; seed a fresh utterance with it.
                pending.update({
                    "text": carry,
                    "source_started_at": ready_at,
                    "ready_at": ready_at,
                    "duration": 0.0,
                    "epoch": epoch,
                    "language": language,
                    "words": len(carry.split()),
                })
            return

        seq = next_transcript_seq
        transcript = Transcript(seq, text, language, ready_at, duration, epoch, source_started_at)
        enqueued = False
        while not stop_event.is_set() and not enqueued:
            if (epoch != state.get("epoch", 0) or state.get("paused", False)
                    or source_started_at < state.get("resume_at", 0.0)):
                break
            if (time.monotonic() - ready_at > MAX_ITEM_AGE_SECONDS
                    or time.monotonic() - source_started_at > MAX_SOURCE_AGE_SECONDS):
                print(f"[TEMPO REAL] Trecho #{seq} venceu aguardando tradução; descartado.")
                break
            with sequence_lock:
                try:
                    transcript_queue.put_nowait(transcript)
                except queue.Full:
                    pass
                else:
                    sequence_meta[seq] = {
                        "captured_at": ready_at, "source_started_at": source_started_at,
                        "epoch": epoch, "text": text,
                    }
                    enqueued = True
            if not enqueued:
                time.sleep(0.01)
        if enqueued:
            next_transcript_seq += 1
            print(f"\n[ORIGINAL/{language}] {text}")
            print(f"[TEMPO] ASR/segmentação: {duration:.2f}s de fala, {reason}; ID={seq}; ASR workers={workers}")
        if carry:
            pending.update({
                "text": carry,
                "source_started_at": ready_at,
                "ready_at": ready_at,
                "duration": 0.0,
                "epoch": epoch,
                "language": language,
                "words": len(carry.split()),
            })

    try:
        while not stop_event.is_set() or inflight:
            # On detected silence/pause, drop the not-yet-committed ASR text. Active playback
            # is allowed to finish its current phrase; pre-pause utterances are stale after resume.
            if state.get("paused", False):
                reset_pending()
                while True:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break

            if stop_event.is_set():
                while True:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break

            # A mid-sentence pause with no continuation within the grace window is a
            # real stop: commit the utterance (at its last sentence boundary).
            if grace_deadline is not None and time.monotonic() > grace_deadline:
                grace_deadline = None
                commit_pending("pausa natural confirmada", can_split=True)

            while not stop_event.is_set() and len(inflight) < workers:
                try:
                    chunk: AudioChunk = audio_queue.get_nowait()
                except queue.Empty:
                    break
                if (chunk.epoch != state.get("epoch", 0) or state.get("paused", False)
                        or chunk.captured_at < state.get("resume_at", 0.0)):
                    continue
                age = time.monotonic() - (chunk.captured_at + chunk.duration)
                if age > MAX_ITEM_AGE_SECONDS:
                    print(f"[TEMPO REAL] Entrada descartada antes do ASR (atraso {age:.1f}s).")
                    continue
                order = next_chunk_order
                next_chunk_order += 1
                future = executor.submit(transcribe_chunk, order, chunk, whisper_model, source_language, state)
                inflight[future] = (order, chunk)

            if inflight:
                done, _ = wait(tuple(inflight), timeout=0.025, return_when=FIRST_COMPLETED)
                for future in done:
                    order, chunk = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = ASRResult(order, chunk, "", "?", 0.0, f"{type(exc).__name__}: {exc}")
                    completed[result.chunk_order] = result
            else:
                time.sleep(0.01)

            while not stop_event.is_set() and next_chunk_to_emit in completed:
                result = completed.pop(next_chunk_to_emit)
                next_chunk_to_emit += 1
                chunk = result.chunk
                if result.error:
                    if result.error != "stale-before-asr":
                        print(f"[ERRO ASR #{result.chunk_order}] {result.error}")
                    continue
                if (chunk.epoch != state.get("epoch", 0) or state.get("paused", False)
                        or chunk.captured_at < state.get("resume_at", 0.0)):
                    continue
                if time.monotonic() - (chunk.captured_at + chunk.duration) > MAX_ITEM_AGE_SECONDS:
                    print(f"[TEMPO REAL] ASR #{result.chunk_order} concluiu tarde; descartando resultado antigo.")
                    continue

                raw_text = result.text.strip()
                text = raw_text
                if chunk.overlap:
                    text = deduplicate_overlap(last_raw_text_by_epoch.get(chunk.epoch, ""), raw_text)
                    if text != raw_text:
                        print(f"[ASR] Removi palavras repetidas na sobreposição do trecho #{result.chunk_order}.")
                if raw_text:
                    last_raw_text_by_epoch[chunk.epoch] = raw_text

                # A chunk with no recognized words can still mark a natural pause. If the
                # assembled text is mid-sentence, hold it briefly for a possible continuation
                # instead of splitting the sentence; only a real stop is committed.
                if not text.strip():
                    if chunk.speech_boundary and int(pending["words"]) >= 2:
                        if (float(pending["duration"]) >= CONTEXT_DURATION_SECONDS
                                or int(pending["words"]) >= CONTEXT_WORDS):
                            commit_pending("limite de contexto", can_split=True)
                        elif _ends_with_punctuation(str(pending["text"])):
                            commit_pending("pausa natural", can_split=False)
                        elif grace_deadline is None:
                            grace_deadline = time.monotonic() + GRACE_PAUSE_SECONDS
                            print("[TEMPO REAL] Pausa no meio de frase: aguardando possível continuação...")
                    continue

                # Start a new utterance, or extend the previous partial transcript. The tiny one-word
                # fragments at a window edge are held for context instead of being translated alone.
                if not pending["text"] or int(pending["epoch"]) != chunk.epoch:
                    if pending["text"]:
                        commit_pending("mudança de segmento", can_split=True)
                    grace_deadline = None
                    pending.update({
                        "text": text,
                        "source_started_at": chunk.captured_at,
                        "ready_at": chunk.captured_at + chunk.duration,
                        "duration": chunk.duration,
                        "epoch": chunk.epoch,
                        "language": result.language,
                        "words": len(text.split()),
                    })
                else:
                    # A new speech chunk resumes an utterance that was waiting in grace:
                    # it is a continuation, so cancel the commit deadline.
                    grace_deadline = None
                    prior = str(pending["text"]).rstrip()
                    # A one/two-word edge fragment followed by a capitalized phrase is likely a
                    # cut between sentences; preserve the boundary rather than gluing grammar together.
                    separator = ". " if int(pending["words"]) <= 2 and text[:1].isupper() and not re.search(r"[.!?…]$", prior) else " "
                    pending["text"] = f"{prior}{separator}{text.lstrip()}"
                    pending["ready_at"] = chunk.captured_at + chunk.duration
                    pending["duration"] = float(pending["duration"]) + max(0.0, chunk.duration - (0.28 if chunk.overlap else 0.0))
                    pending["words"] = len(str(pending["text"]).split())
                    pending["language"] = result.language or pending["language"]

                current_text = str(pending["text"]).strip()
                word_count = int(pending["words"])
                duration = float(pending["duration"])
                if _ends_with_punctuation(current_text):
                    commit_pending("pontuação de frase", can_split=False)
                elif word_count >= CONTEXT_WORDS or duration >= CONTEXT_DURATION_SECONDS:
                    commit_pending("limite de contexto", can_split=True)
                elif chunk.speech_boundary and word_count >= 2:
                    if grace_deadline is None:
                        grace_deadline = time.monotonic() + GRACE_PAUSE_SECONDS
                        print("[TEMPO REAL] Pausa no meio de frase: aguardando possível continuação...")
                elif not chunk.speech_boundary and word_count >= 4:
                    # A full window ended but the speaker keeps going: commit the sentences
                    # that are already complete so translation starts sooner, and carry the
                    # unfinished tail into the next utterance.
                    head, tail = split_last_sentence(current_text)
                    if tail and len(head.split()) >= 2:
                        commit_pending("frase concluída na janela", can_split=True)

    finally:
        executor.shutdown(wait=False, cancel_futures=True)

def put_speech_event(events_queue: queue.Queue, event: SpeechEvent, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            events_queue.put(event, timeout=0.10)
            return True
        except queue.Full:
            continue
    return False


def translate_job(transcript: Transcript, events_queue: queue.Queue, stop_event: threading.Event,
                  state: dict) -> None:
    part = 0
    error_message: str | None = None
    try:
        if (transcript.epoch != state.get("epoch", 0) or state.get("paused", False)
                or (transcript.source_started_at if transcript.source_started_at is not None else transcript.captured_at) < state.get("resume_at", 0.0)):
            return
        if transcript.language == "pt":
            units = [transcript.text]
            for unit in units:
                if unit.strip():
                    put_speech_event(events_queue, SpeechEvent(transcript.seq, part, unit.strip(), False,
                                      transcript.captured_at, transcript.epoch,
                                      source_started_at=transcript.source_started_at), stop_event)
                    part += 1
        else:
            t0 = time.perf_counter()
            for unit in translate_stream_ollama(transcript.text, transcript.language, stop_event):
                if stop_event.is_set():
                    return
                if (transcript.epoch != state.get("epoch", 0) or state.get("paused", False)
                        or (transcript.source_started_at if transcript.source_started_at is not None else transcript.captured_at) < state.get("resume_at", 0.0)):
                    break
                put_speech_event(events_queue, SpeechEvent(transcript.seq, part, unit, False,
                                  transcript.captured_at, transcript.epoch,
                                  source_started_at=transcript.source_started_at), stop_event)
                part += 1
            if part:
                print(f"[PORTUGUÊS #{transcript.seq}] tradução em fluxo | primeira/última emissão ~{time.perf_counter() - t0:.2f}s")
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"[ERRO TRADUÇÃO #{transcript.seq}] {error_message}")
    finally:
        # Exactly one end marker per sequence: the reorder stage must never wait for a missing part.
        put_speech_event(events_queue, SpeechEvent(transcript.seq, part, "", True,
                          transcript.captured_at, transcript.epoch, error_message,
                          source_started_at=transcript.source_started_at), stop_event)


def translation_dispatcher(transcript_queue: queue.Queue, events_queue: queue.Queue,
                           stop_event: threading.Event, state: dict, workers: int) -> None:
    """ThreadPoolExecutor grows threads only as jobs arrive, up to the allowed Ollama parallelism."""
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="translator")
    futures: set[Future] = set()
    try:
        while not stop_event.is_set() or futures:
            while not stop_event.is_set() and len(futures) < workers:
                try:
                    item: Transcript = transcript_queue.get_nowait()
                except queue.Empty:
                    break
                futures.add(executor.submit(translate_job, item, events_queue, stop_event, state))
            if futures:
                done, remaining = wait(futures, timeout=0.04, return_when=FIRST_COMPLETED)
                futures = set(remaining)
                for future in done:
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[ERRO WORKER DE TRADUÇÃO] {type(exc).__name__}: {exc}")
            else:
                time.sleep(0.025)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _enqueue_part_chunk(part: AudioPart, samples: np.ndarray, sample_rate: int,
                        stop_event: threading.Event, state: dict) -> bool:
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    # Smaller frames allow the playback consumer to start promptly and bound per-part audio ahead.
    frame_samples = 4800  # 200 ms at Kokoro's 24 kHz output rate
    for offset in range(0, len(data), frame_samples):
        while not stop_event.is_set():
            if part.cancel.is_set() or part.event.epoch != state.get("epoch", 0):
                return False
            try:
                part.audio_chunks.put((data[offset:offset + frame_samples].copy(), sample_rate), timeout=0.05)
                break
            except queue.Full:
                continue
        else:
            return False
    return True


async def _stream_kokoro(kokoro: Kokoro, phonemes: str, part: AudioPart,
                         state: dict, stop_event: threading.Event, speed: float) -> None:
    emitted = False
    if hasattr(kokoro, "create_stream"):
        try:
            async for samples, sample_rate in kokoro.create_stream(
                phonemes, voice="pf_dora", speed=speed, lang="pt-br", is_phonemes=True,
                sentence_pause=TTS_SENTENCE_PAUSE, clause_pause=TTS_CLAUSE_PAUSE,
            ):
                if stop_event.is_set() or part.cancel.is_set() or part.event.epoch != state.get("epoch", 0):
                    return
                emitted = True
                if not _enqueue_part_chunk(part, samples, sample_rate, stop_event, state):
                    return
            _enqueue_part_trailing_pause(part, stop_event, state)
            return
        except Exception as exc:
            if emitted:
                raise RuntimeError(f"Kokoro streaming falhou depois de emitir áudio: {exc}") from exc
            print(f"[AVISO TTS] create_stream falhou antes de emitir; usando fallback em lote: {exc}")
    samples, sample_rate = kokoro.create(
        phonemes, "pf_dora", speed=speed, is_phonemes=True,
        sentence_pause=TTS_SENTENCE_PAUSE, clause_pause=TTS_CLAUSE_PAUSE,
    )
    if _enqueue_part_chunk(part, samples, sample_rate, stop_event, state):
        _enqueue_part_trailing_pause(part, stop_event, state)


def _enqueue_part_trailing_pause(part: AudioPart, stop_event: threading.Event,
                                 state: dict) -> None:
    pause = part_trailing_pause(part.event.text)
    if pause <= 0:
        return
    silence = np.zeros(int(pause * TTS_SAMPLE_RATE), dtype=np.float32)
    _enqueue_part_chunk(part, silence, TTS_SAMPLE_RATE, stop_event, state)


def tts_synthesis_job(part: AudioPart, kokoro: Kokoro, g2p: EspeakG2P,
                      g2p_lock: threading.Lock, stop_event: threading.Event,
                      state: dict, tts_speed: float) -> None:
    event = part.event
    started = time.perf_counter()
    try:
        if (stop_event.is_set() or part.cancel.is_set() or event.epoch != state.get("epoch", 0)
                or state.get("paused", False)
                or (event.source_started_at if event.source_started_at is not None else event.captured_at) < state.get("resume_at", 0.0)):
            return
        age = time.monotonic() - event.captured_at
        source_started_at = event.source_started_at if event.source_started_at is not None else event.captured_at
        if age > MAX_ITEM_AGE_SECONDS or time.monotonic() - source_started_at > MAX_SOURCE_AGE_SECONDS:
            print(f"[TEMPO REAL] TTS #{event.seq}.{event.part} expirou antes da síntese (idade {age:.2f}s).")
            return
        # Misaki/espeak is protected by a short lock; neural synthesis itself can run concurrently.
        with g2p_lock:
            phonemes, _ = g2p(event.text)
        if not phonemes.strip() or part.cancel.is_set():
            return
        asyncio.run(_stream_kokoro(kokoro, phonemes, part, state, stop_event, tts_speed))
        print(f"[TEMPO] TTS synthesis #{event.seq}.{event.part}: {time.perf_counter() - started:.2f}s; idade inicial={age:.2f}s")
    except Exception as exc:
        part.error = f"{type(exc).__name__}: {exc}"
        print(f"[ERRO TTS #{event.seq}.{event.part}] {part.error}")
    finally:
        part.done.set()


def tts_playback_worker(parts_queue: queue.Queue, player: StreamingAudioPlayer,
                        stop_event: threading.Event, state: dict) -> None:
    """Single ordered audio consumer; TTS workers can synthesize future parts concurrently."""
    while not stop_event.is_set() or not parts_queue.empty():
        try:
            part: AudioPart = parts_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        event = part.event
        source_started_at = event.source_started_at if event.source_started_at is not None else event.captured_at
        stale = (event.epoch != state.get("epoch", 0) or state.get("paused", False)
                 or source_started_at < state.get("resume_at", 0.0)
                 or time.monotonic() - event.captured_at > MAX_ITEM_AGE_SECONDS
                 or time.monotonic() - source_started_at > MAX_SOURCE_AGE_SECONDS)
        if stale:
            part.cancel.set()
            # Do not clear the device queue here: it can still contain the tail (a few hundred ms)
            # of the previous phrase. Skip this part without truncating the phrase already spoken.
            # Drain the bounded future-part buffer so its worker cannot remain blocked.
            while not part.done.is_set() or not part.audio_chunks.empty():
                try:
                    part.audio_chunks.get(timeout=0.03)
                except queue.Empty:
                    if stop_event.is_set():
                        break
            continue

        first_chunk = True
        while not stop_event.is_set():
            if part.cancel.is_set() or event.epoch != state.get("epoch", 0):
                part.cancel.set()
                player.clear_pending()
                break
            try:
                item = part.audio_chunks.get(timeout=0.05)
            except queue.Empty:
                if part.done.is_set():
                    break
                continue
            samples, sample_rate = item
            if first_chunk:
                # If the pause happened while this part was still waiting for its first audio block,
                # do not start an old sentence after the video has been paused. Once playback of a
                # phrase has started, it is allowed to finish normally.
                if (state.get("paused", False)
                        or source_started_at < state.get("resume_at", 0.0)):
                    part.cancel.set()
                    break
                first_chunk = False
                lag = time.monotonic() - source_started_at
                print(f"[PORTUGUÊS #{event.seq}.{event.part}] {event.text}")
                print(f"[LATÊNCIA] primeiro bloco de voz na saída após {lag:.2f}s desde o início da fala de origem.")
            if not player.enqueue_audio(samples, sample_rate, event.captured_at, event.epoch, state):
                player.clear_pending()
                part.cancel.set()
                break
        # If this part was canceled or the process is stopping, drain remaining blocks so a
        # producer blocked on its bounded queue can finish and release its worker.
        if part.cancel.is_set() or event.epoch != state.get("epoch", 0) or stop_event.is_set():
            while not part.done.is_set() or not part.audio_chunks.empty():
                try:
                    part.audio_chunks.get(timeout=0.02)
                except queue.Empty:
                    if stop_event.is_set():
                        break


def ordered_tts_dispatcher(events_queue: queue.Queue, parts_queue: queue.Queue,
                           kokoro: Kokoro, g2p: EspeakG2P, stop_event: threading.Event,
                           state: dict, sequence_meta: dict, sequence_lock: threading.Lock,
                           tts_speed: float, workers: int) -> None:
    """Reorders text events, dispatches bounded concurrent TTS jobs, keeps playback order strict."""
    pending: dict[int, dict[int, SpeechEvent]] = {}
    expected_seq = 1
    expected_part = 0
    scheduled_sequences: set[int] = set()
    dropped_sequences: set[int] = set()
    g2p_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts")
    futures: set[Future] = set()
    try:
        while not stop_event.is_set() or futures:
            if not stop_event.is_set():
                try:
                    event: SpeechEvent = events_queue.get(timeout=0.025)
                    if event.seq >= expected_seq:
                        pending.setdefault(event.seq, {})[event.part] = event
                except queue.Empty:
                    pass

            with sequence_lock:
                meta = sequence_meta.get(expected_seq)
            if (meta and expected_seq not in scheduled_sequences
                    and (meta.get("source_started_at", meta["captured_at"]) < state.get("resume_at", 0.0)
                         or time.monotonic() - meta["captured_at"] > MAX_ITEM_AGE_SECONDS
                         or time.monotonic() - meta.get("source_started_at", meta["captured_at"]) > MAX_SOURCE_AGE_SECONDS)):
                pending.pop(expected_seq, None)
                with sequence_lock:
                    sequence_meta.pop(expected_seq, None)
                print(f"[TEMPO REAL] ID={expected_seq} expirou antes de iniciar a voz; avançando.")
                expected_seq += 1
                expected_part = 0
                continue

            segment_events = pending.get(expected_seq, {})
            while expected_part in segment_events:
                event = segment_events.pop(expected_part)
                expected_part += 1
                if event.final:
                    if event.error:
                        print(f"[AVISO] ID={event.seq}: {event.error}")
                    pending.pop(expected_seq, None)
                    scheduled_sequences.discard(expected_seq)
                    dropped_sequences.discard(expected_seq)
                    with sequence_lock:
                        sequence_meta.pop(expected_seq, None)
                    expected_seq += 1
                    expected_part = 0
                    break
                if not event.text.strip():
                    continue
                if event.seq in dropped_sequences:
                    continue
                source_started_at = event.source_started_at if event.source_started_at is not None else event.captured_at
                if (event.epoch != state.get("epoch", 0) or state.get("paused", False)
                        or source_started_at < state.get("resume_at", 0.0)):
                    dropped_sequences.add(event.seq)
                    continue
                age = time.monotonic() - event.captured_at
                if (age > MAX_ITEM_AGE_SECONDS
                        or time.monotonic() - source_started_at > MAX_SOURCE_AGE_SECONDS):
                    dropped_sequences.add(event.seq)
                    print(f"[TEMPO REAL] ID={event.seq}.{event.part} expirou antes de entrar no TTS; trecho descartado.")
                    continue

                # Bounded parts_queue gives backpressure; each part queue is also bounded.
                part = AudioPart(event=event, audio_chunks=queue.Queue(maxsize=4),
                         done=threading.Event(), cancel=threading.Event())
                enqueued = False
                while not stop_event.is_set() and not enqueued:
                    if (event.epoch != state.get("epoch", 0) or state.get("paused", False)
                            or source_started_at < state.get("resume_at", 0.0)):
                        break
                    try:
                        parts_queue.put(part, timeout=0.05)
                        enqueued = True
                    except queue.Full:
                        continue
                if enqueued:
                    scheduled_sequences.add(event.seq)
                    futures.add(executor.submit(tts_synthesis_job, part, kokoro, g2p,
                                                g2p_lock, stop_event, state, tts_speed))
            if futures:
                done = {future for future in futures if future.done()}
                futures.difference_update(done)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

def resolve_worker_count(value: str) -> int:
    if value != "auto":
        return max(1, min(4, int(value)))
    # Ollama defaults to one parallel request per model. Honour an explicit server configuration.
    configured = os.getenv("OLLAMA_NUM_PARALLEL", "1").strip()
    try:
        return max(1, min(3, int(configured)))
    except ValueError:
        return 1




def resolve_asr_worker_count(value: str, device: str) -> int:
    if value != "auto":
        return max(1, min(3, int(value)))
    # Avoid duplicating the Whisper model on GPU by default; CPU has room for concurrent ASR workers.
    return 1 if device == "cuda" else (2 if (os.cpu_count() or 8) >= 12 else 1)


def resolve_tts_worker_count(value: str) -> int:
    if value != "auto":
        return max(1, min(3, int(value)))
    # Kokoro stays on CPU; create workers lazily and keep a conservative ceiling.
    return 2 if (os.cpu_count() or 8) >= 8 else 1

def main() -> int:
    parser = argparse.ArgumentParser(description="Tradutor local de áudio do navegador para voz pt-BR (pipeline por etapas).")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input", default=None, help="Trecho do nome da entrada; padrão: CABLE Output.")
    parser.add_argument("--output", default=None, help="Trecho do nome da saída; padrão: saída de áudio do Windows.")
    parser.add_argument("--language", default="auto", help="Idioma da fala original: auto, en, es, fr etc.")
    parser.add_argument("--whisper-model", default="base", help="Modelo ASR; base prioriza velocidade, small prioriza precisão.")
    parser.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--translation-workers", default="auto", help="auto ou 1-4; auto respeita OLLAMA_NUM_PARALLEL.")
    parser.add_argument("--asr-workers", default="auto", help="auto ou 1-3; auto evita duplicar o modelo na GPU e usa 2 no CPU com mais de 12 processadores lógicos.")
    parser.add_argument("--tts-workers", default="auto", help="auto ou 1-3; workers Kokoro em CPU, com saída e ordem controladas.")
    parser.add_argument("--tts-speed", type=float, default=1.2, help="Velocidade da voz; padrão 1.2 para reduzir o atraso.")
    parser.add_argument("--test-voice", action="store_true", help="Testa o caminho antigo soundcard.play.")
    parser.add_argument("--test-stream-output", action="store_true", help="Testa o caminho contínuo sounddevice/Kokoro usado na tradução ao vivo.")
    parser.add_argument("--monitor-input", action="store_true")
    parser.add_argument("--test-translation", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0
    if args.translation_workers != "auto":
        try:
            parsed = int(args.translation_workers)
            if not 1 <= parsed <= 4:
                raise ValueError
        except ValueError:
            parser.error("--translation-workers aceita 'auto' ou um número entre 1 e 4.")
    for arg_name, arg_value in (("--asr-workers", args.asr_workers), ("--tts-workers", args.tts_workers)):
        if arg_value != "auto":
            try:
                parsed = int(arg_value)
                if not 1 <= parsed <= 3:
                    raise ValueError
            except ValueError:
                parser.error(f"{arg_name} aceita 'auto' ou um número entre 1 e 3.")
    if args.test_translation:
        sample = "And once you see something populated, you can see the content."
        print(f"[MODELO] Testando tradução via streaming com {OLLAMA_MODEL}...")
        try:
            print("[ORIGINAL/en] " + sample)
            print("[PORTUGUÊS] " + translate_complete(sample, "en"))
            return 0
        except Exception as exc:
            print(f"[ERRO TESTE TRADUÇÃO] {type(exc).__name__}: {exc}")
            return 2

    if not KOKORO_MODEL.exists() or not KOKORO_VOICES.exists():
        print("[ERRO] Faltam os arquivos do Kokoro ao lado de tradutor_local.py:")
        print(f"  {KOKORO_MODEL}\n  {KOKORO_VOICES}")
        return 2
    try:
        microphone = find_input_device(args.input)
        speaker = find_output_device(args.output)
    except Exception as exc:
        print(f"[ERRO] {exc}")
        return 2
    print(f"[SAÍDA] Saída escolhida: {speaker.name}")
    if args.test_voice:
        return test_voice_output(speaker)
    if args.test_stream_output:
        return test_stream_output(speaker.name, args.tts_speed)
    if args.monitor_input:
        return monitor_input(microphone)

    stop_event = threading.Event()
    initial_time = time.monotonic()
    state = {"last_audio_at": initial_time, "epoch": 0, "paused": False, "resume_at": initial_time}
    sequence_meta: dict[int, dict] = {}
    sequence_lock = threading.Lock()

    whisper_device = "cpu"
    whisper_compute = "int8"
    whisper_model = None
    asr_worker_count = 1
    if args.whisper_device in ("auto", "cuda"):
        asr_worker_count = resolve_asr_worker_count(args.asr_workers, "cuda")
        print(f"[MODELO] Tentando faster-whisper '{args.whisper_model}' na GPU NVIDIA (FP16), num_workers={asr_worker_count}...")
        try:
            candidate = WhisperModel(args.whisper_model, device="cuda", compute_type="float16",
                                     num_workers=asr_worker_count)
            print("[MODELO] Verificando uma inferência real na GPU (detecta DLLs carregadas tardiamente)...")
            preflight_whisper_runtime(candidate)
            whisper_model = candidate
            whisper_device, whisper_compute = "cuda", "float16"
        except Exception as exc:
            whisper_model = None
            if args.whisper_device == "cuda":
                print(f"[ERRO] Whisper não passou no teste real de inferência CUDA: {exc}")
                print("O faster-whisper requer cuBLAS para CUDA 12 e cuDNN 9 no PATH. Use --whisper-device cpu ou corrija essas DLLs.")
                return 2
            print(f"[AVISO] GPU para Whisper não passou no teste; fallback para CPU INT8 antes de iniciar a captura: {exc}")
    if whisper_model is None:
        try:
            asr_worker_count = resolve_asr_worker_count(args.asr_workers, "cpu")
            logical_cpus = os.cpu_count() or 12
            cpu_threads = max(2, min(6, logical_cpus // (asr_worker_count * 2)))
            whisper_model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8",
                                         cpu_threads=cpu_threads, num_workers=asr_worker_count)
            print(f"[RECURSOS] ASR CPU: {cpu_threads} threads por worker, num_workers={asr_worker_count}.")
        except Exception as exc:
            print(f"[ERRO] Não consegui carregar o Whisper: {exc}")
            return 2
    print(f"[MODELO] Whisper pronto: dispositivo={whisper_device}, precisão={whisper_compute}, workers ASR={asr_worker_count}.")
    try:
        tts_worker_count = resolve_tts_worker_count(args.tts_workers)
        kokoro, g2p = initialize_tts(tts_worker_count)
        player = StreamingAudioPlayer(speaker.name, stop_event)
    except Exception as exc:
        print(f"[ERRO] Não consegui iniciar Kokoro/saída contínua: {type(exc).__name__}: {exc}")
        print("Se o erro indicar dispositivo não encontrado, rode --list-devices e use o nome de saída correto.")
        return 2

    try:
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=3).json().get("models", [])
        names = [item.get("name", "") for item in tags]
        if not any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") for name in names):
            print(f"[ERRO] Modelo não encontrado: execute ollama pull {OLLAMA_MODEL}")
            player.close()
            return 2
    except Exception as exc:
        print(f"[ERRO] Ollama não respondeu em 127.0.0.1:11434: {exc}")
        player.close()
        return 2

    worker_count = resolve_worker_count(args.translation_workers)
    if args.translation_workers == "auto":
        print(f"[POOL] Tradução adaptativa: máximo de {worker_count} worker(s), conforme OLLAMA_NUM_PARALLEL.")
    else:
        worker_count = int(args.translation_workers)
        print(f"[POOL] Tradução configurada para até {worker_count} worker(s).")
        if worker_count > 1 and os.getenv("OLLAMA_NUM_PARALLEL", "1") == "1":
            print("[AVISO] Ollama pode serializar chamadas. Para concorrência real, configure OLLAMA_NUM_PARALLEL no processo servidor e reinicie o Ollama.")

    audio_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=3)
    transcript_queue: queue.Queue[Transcript] = queue.Queue(maxsize=3)
    speech_events: queue.Queue[SpeechEvent] = queue.Queue(maxsize=48)
    audio_parts: queue.Queue[AudioPart] = queue.Queue(maxsize=4)

    print(f"[POOL] ASR={asr_worker_count} worker(s); tradução={worker_count} worker(s); TTS={tts_worker_count} worker(s).")
    print("[POOL] Os executores criam threads sob demanda até esses limites; não criamos cópias ilimitadas dos modelos.")
    threads = [
        threading.Thread(target=asr_dispatcher,
                         args=(audio_queue, transcript_queue, whisper_model, args.language, stop_event,
                               state, sequence_meta, sequence_lock, asr_worker_count),
                         name="ASR-dispatcher", daemon=True),
        threading.Thread(target=translation_dispatcher,
                         args=(transcript_queue, speech_events, stop_event, state, worker_count),
                         name="translation-dispatcher", daemon=True),
        threading.Thread(target=ordered_tts_dispatcher,
                         args=(speech_events, audio_parts, kokoro, g2p, stop_event, state, sequence_meta,
                               sequence_lock, args.tts_speed, tts_worker_count),
                         name="TTS-dispatcher", daemon=True),
        threading.Thread(target=tts_playback_worker,
                         args=(audio_parts, player, stop_event, state),
                         name="ordered-audio-playback", daemon=True),
        threading.Thread(target=capture_worker,
                         args=(microphone, audio_queue, stop_event, state, player),
                         name="audio-capture", daemon=True),
    ]
    print("[PIPELINE] Captura, ASR, tradução, síntese e reprodução em workers independentes.")
    print("[PIPELINE] Tradução+TTS em streaming, ASR com pool ordenado, reprodução estritamente em sequência; buffers limitados.")
    try:
        for thread in threads:
            thread.start()
        while all(thread.is_alive() for thread in threads):
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[ENCERRANDO] Limpando buffers e interrompendo o pipeline...")
    finally:
        stop_event.set()
        player.clear_pending()
        for thread in threads:
            thread.join(timeout=1.5)
        player.close()
    print("[FIM] Tradutor encerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
