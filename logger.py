"""
logger.py
Loguru-based structured logging for the Pipecat Voice Agent.

Matches bot (1).py's loguru setup with bracket-tagged stage messages:
  [TURN]     turn id, state transitions
  [STT]      transcription results
  [LLM]      token stream, latency
  [TTS]      synthesis lifecycle
  [TOOL]     tool calls and results
  [VAD]      speech start / end
  [PIPELINE] pipeline lifecycle events

Every pipeline stage gets a named logger via get_logger("stage").
All existing %-style call sites (info("msg %s", val)) work unchanged.
"""

import sys
from pathlib import Path

from loguru import logger

from config import settings

# ── Console / file formats ────────────────────────────────────────────────────
_CONSOLE_FMT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[stage]:<12}</cyan> | "
    "{message}"
)
_FILE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{extra[stage]:<12} | "
    "{message}"
)

_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT  = 5
_WARNING_NO    = 30                 # loguru WARNING level number

# ── Stage name map ────────────────────────────────────────────────────────────
LOGGERS: dict[str, str] = {
    "audio_in":     "AUDIO-IN",
    "vad":          "VAD",
    "stt":          "STT",
    "llm":          "LLM",
    "tts":          "TTS",
    "audio_out":    "AUDIO-OUT",
    "pipeline":     "PIPELINE",
    "agent":        "AGENT",
    "echo_gate":    "ECHO",
    "filler":       "FILLER",
    "filter":       "FILTER",
    "normalizer":   "NORMALIZER",
    "phonetic":     "PHONETIC",
    "latency":      "LATENCY",
    "typing_sound": "TYPING",
    "tools":        "TOOLS",
}


class _StageLogger:
    """
    Thin loguru wrapper with %-format compatibility.
    Drop-in for logging.Logger — all existing call sites work unchanged.
    """

    def __init__(self, stage: str) -> None:
        self._log = logger.bind(stage=stage)

    @staticmethod
    def _fmt(msg: str, args: tuple) -> str:
        return msg % args if args else str(msg)

    def debug(self, msg: str, *args, **_) -> None:
        self._log.opt(depth=1).debug(self._fmt(msg, args))

    def info(self, msg: str, *args, **_) -> None:
        self._log.opt(depth=1).info(self._fmt(msg, args))

    def warning(self, msg: str, *args, **_) -> None:
        self._log.opt(depth=1).warning(self._fmt(msg, args))

    def error(self, msg: str, *args, **_) -> None:
        self._log.opt(depth=1).error(self._fmt(msg, args))

    def exception(self, msg: str, *args, **_) -> None:
        self._log.opt(depth=1, exception=True).error(self._fmt(msg, args))


def setup_logging() -> None:
    """Configure loguru — call once at startup. Matches bot (1).py setup."""
    # Default stage for Pipecat / third-party loguru messages (no stage bound)
    logger.configure(extra={"stage": "PIPECAT"})

    # Remove loguru's default stderr sink
    logger.remove()

    # Console — our stage-tagged logs at INFO+, Pipecat internals at WARNING+ only
    def _console_filter(record: dict) -> bool:
        stage = record["extra"].get("stage", "PIPECAT")
        if stage != "PIPECAT":
            return record["level"].no >= 20   # INFO+ for our logs
        return record["level"].no >= _WARNING_NO  # WARNING+ for Pipecat/untagged

    logger.add(
        sys.stderr,
        level="DEBUG",
        format=_CONSOLE_FMT,
        colorize=True,
        filter=_console_filter,
    )

    # File — full DEBUG trace for post-mortem analysis, rotating 10 MB × 5
    Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.LOG_FILE,
        level="DEBUG",
        format=_FILE_FMT,
        rotation=_MAX_LOG_BYTES,
        retention=_BACKUP_COUNT,
        encoding="utf-8",
    )

    # Silence noisy stdlib loggers that bypass loguru
    import logging as _stdlib
    for _name in ["httpx", "httpcore", "urllib3", "openai", "websockets"]:
        _stdlib.getLogger(_name).setLevel(_stdlib.WARNING)

    logger.bind(stage="SYSTEM").info(
        "Logging initialised — level=INFO  file={}", settings.LOG_FILE
    )


def get_logger(stage: str) -> _StageLogger:
    """Return a stage-tagged logger. Compatible with logging.Logger call sites."""
    name = LOGGERS.get(stage, stage.upper())
    return _StageLogger(name)


# ── Convenience helpers (used by processors.py and agent.py) ─────────────────

def log_pipeline_event(event: str, detail: str = "") -> None:
    logger.bind(stage="PIPELINE").info(f"[{event}] {detail}")


def log_vad_event(speech_detected: bool) -> None:
    _log = logger.bind(stage="VAD")
    if speech_detected:
        _log.info("[VAD] Speech START detected")
    else:
        _log.info("[VAD] Speech END")


def log_stt_result(transcript: str, is_final: bool) -> None:
    tag = "FINAL" if is_final else "partial"
    logger.bind(stage="STT").info(f"[STT] [{tag}]  {transcript!r}")


def log_llm_prompt(messages: list) -> None:
    logger.bind(stage="LLM").debug(f"[LLM] Sending {len(messages)} message(s)")


def log_llm_token(token: str) -> None:
    logger.bind(stage="LLM").debug(f"[LLM] Token: {token!r}")


def log_llm_complete(full_response: str) -> None:
    logger.bind(stage="LLM").info(f"[LLM] Complete — {len(full_response)} chars")


def log_tts_chunk(chunk_bytes: int) -> None:
    logger.bind(stage="TTS").debug(f"[TTS] Audio chunk: {chunk_bytes} bytes")


def log_tts_complete() -> None:
    logger.bind(stage="TTS").info("[TTS] Synthesis complete")
