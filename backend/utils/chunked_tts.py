"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.
"""

import logging
import re
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("voicebox.chunked-tts")

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
    }
)

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence-end (``.!?`` not preceded by an abbreviation and not
    inside brackets) → clause boundary (``;:,—``) → whitespace → hard cut.

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to split at the last real sentence ending
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            # Absolute fallback: hard cut but avoid splitting inside a tag
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last sentence-ending punctuation in *text*.

    Skips periods that follow common abbreviations (``Dr.``, ``Mr.``, etc.)
    and periods inside bracket tags (``[laugh]``).  Also handles CJK
    sentence-ending punctuation (``。！？``).
    """
    best = -1
    # ASCII sentence ends
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        char = text[pos]
        # Skip periods after abbreviations
        if char == ".":
            # Walk backwards to find the preceding word
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            # Skip decimal numbers (digit immediately before the period)
            if word_start >= 0 and text[word_start].isdigit():
                continue
        # Skip if we're inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    # CJK sentence-ending punctuation
    for m in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if m.start() > best:
            best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation."""
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = m.start()
        # Skip if inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    for m in _PARA_TAG_RE.finditer(text):
        if m.start() < pos < m.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result


# ── ASR round-trip verification ──────────────────────────────────────
#
# Autoregressive TTS fails by alignment drift: double-read sentences,
# skipped words, invented words. An ASR round-trip (generate → transcribe →
# compare against the source text) catches exactly this class; published
# results show best-of-N with an ASR gate drives catastrophic failure from
# ~27% to ~0% at N=4 (arXiv 2606.18323). The same technique powers
# ElevenLabs Studio's Auto-Regenerate ("missing or additional words").

VERIFY_WER_THRESHOLD = 0.35
VERIFY_MAX_ATTEMPTS = 3  # 1 original + up to 2 retries
_SEED_RETRY_STRIDE = 1009  # prime, far outside the per-chunk +i stride

_WHISPER_LANGS = {"en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"}


def _normalize_for_wer(text: str) -> List[str]:
    """Lowercase, strip [tags]/punctuation, drop digit-bearing tokens.

    Digit tokens are dropped from BOTH sides because TTS normalizes numbers
    ("20" → "twenty") — a legitimate read would otherwise count as an error.
    """
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if not any(ch.isdigit() for ch in w)]


def _word_error_rate(ref: List[str], hyp: List[str]) -> float:
    """Word-level Levenshtein distance over the reference length."""
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


async def _transcribe_for_verify(
    audio: np.ndarray, sample_rate: int, language: str
) -> str | None:
    """Whisper round-trip via the in-process STT backend.

    Returns None when no Whisper model is cached — verification never
    triggers a model download mid-generation.
    """
    from ..services import transcribe as transcribe_service

    whisper = transcribe_service.get_whisper_model()
    size = None
    if whisper.is_loaded():
        size = whisper.model_size
    else:
        for candidate in ("base", "small", "turbo", "medium", "large"):
            try:
                if whisper._is_model_cached(candidate):
                    size = candidate
                    break
            except Exception:
                continue
    if size is None:
        return None

    import os
    import tempfile

    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, np.asarray(audio, dtype=np.float32), sample_rate)
        tmp.close()
        lang = language if language in _WHISPER_LANGS else None
        return await whisper.transcribe(tmp.name, lang, size)
    except Exception as e:
        logger.warning("Verify transcription failed (%s) — skipping check", e)
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


async def _generate_chunk_verified(
    backend,
    chunk_text: str,
    voice_prompt: dict,
    language: str,
    chunk_seed: int | None,
    instruct: str | None,
    params: dict | None,
    trim_fn,
    verify: bool,
    report: dict | None,
) -> Tuple[np.ndarray, int, float | None]:
    """Generate one chunk, optionally ASR-verified with seed-reroll retries.

    Returns (audio, sample_rate, wer) — wer is None when verification was
    off or unavailable. On repeated failure the LOWEST-WER take is kept.
    """
    ref_words = _normalize_for_wer(chunk_text) if verify else []
    attempts = VERIFY_MAX_ATTEMPTS if (verify and ref_words) else 1

    best: tuple | None = None  # (wer, audio, sr)
    for attempt in range(attempts):
        seed_n = (
            (chunk_seed + attempt * _SEED_RETRY_STRIDE)
            if chunk_seed is not None
            else None
        )
        audio, sr = await backend.generate(
            chunk_text, voice_prompt, language, seed_n, instruct, params
        )
        if trim_fn is not None:
            audio = trim_fn(audio, sr)

        if attempts == 1:
            return audio, sr, None

        hyp_text = await _transcribe_for_verify(audio, sr, language)
        if hyp_text is None:
            # Whisper unavailable — verification is off for this request.
            if report is not None:
                report["available"] = False
            return audio, sr, None

        wer = _word_error_rate(ref_words, _normalize_for_wer(hyp_text))
        if best is None or wer < best[0]:
            best = (wer, audio, sr)
        if wer <= VERIFY_WER_THRESHOLD:
            break
        if attempt < attempts - 1:
            if report is not None:
                report["retries"] = report.get("retries", 0) + 1
            logger.info(
                "Chunk failed verify (WER %.2f > %.2f) — retrying with rerolled seed. heard=%r",
                wer,
                VERIFY_WER_THRESHOLD,
                (hyp_text or "")[:120],
            )

    assert best is not None
    return best[1], best[2], best[0]


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
    params: dict | None = None,
    verify: bool = False,
    report: dict | None = None,
) -> Tuple[np.ndarray, int]:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()`` with zero overhead.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()`` verbatim.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).
    params : dict | None
        Per-engine expressiveness/sampling overrides, forwarded to
        ``backend.generate()`` (each backend sanitizes its own subset).
    verify : bool
        ASR round-trip verification per chunk (see module notes above).
        Silently unavailable when no Whisper model is cached.
    report : dict | None
        Out-param: mutated with verification metadata —
        ``{enabled, available, chunks, retries, worst_wer, passed}``.

    Returns
    -------
    (audio, sample_rate) : Tuple[np.ndarray, int]
    """
    chunks = split_text_into_chunks(text, max_chunk_chars)

    if report is not None:
        report.update(
            {
                "enabled": bool(verify),
                "available": bool(verify),
                "chunks": len(chunks),
                "retries": 0,
                "worst_wer": None,
                "passed": None,
            }
        )

    if len(chunks) > 1:
        logger.info(
            "Splitting %d chars into %d chunks (max %d chars each)",
            len(text),
            len(chunks),
            max_chunk_chars,
        )

    audio_chunks: List[np.ndarray] = []
    sample_rate: int | None = None
    worst_wer: float | None = None

    for i, chunk_text in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(
                "Generating chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk_text)
            )
        # Vary the seed per chunk to avoid correlated RNG artefacts,
        # but keep it deterministic so the same (text, seed) pair
        # always produces the same output.
        chunk_seed = (seed + i) if seed is not None else None

        chunk_audio, chunk_sr, wer = await _generate_chunk_verified(
            backend,
            chunk_text,
            voice_prompt,
            language,
            chunk_seed,
            instruct,
            params,
            trim_fn,
            verify,
            report,
        )
        if wer is not None:
            worst_wer = wer if worst_wer is None else max(worst_wer, wer)

        audio_chunks.append(np.asarray(chunk_audio, dtype=np.float32))
        if sample_rate is None:
            sample_rate = chunk_sr

    if report is not None:
        report["worst_wer"] = round(worst_wer, 3) if worst_wer is not None else None
        if report["enabled"] and report["available"] and worst_wer is not None:
            report["passed"] = worst_wer <= VERIFY_WER_THRESHOLD

    if len(audio_chunks) == 1:
        return audio_chunks[0], sample_rate

    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return audio, sample_rate
