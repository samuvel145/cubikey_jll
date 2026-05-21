"""
pipeline/processors.py
Custom Pipecat frame processors for logging and tracing.

Each processor sits between two pipeline stages and logs the frames
passing through without modifying them (transparent pass-through).
"""

import asyncio
import dataclasses
import random
import re
import time

from audio_smoother import fade_in, fade_out                   # AUDIO-SMOOTH-v1
from pronunciation_normalizer import normalize as tts_normalize # PRON-NORM-v1

from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from logger import (
    get_logger,
    log_llm_complete,
    log_llm_token,
    log_stt_result,
    log_tts_chunk,
    log_tts_complete,
    log_vad_event,
)

audio_log = get_logger("audio_in")

_echo_log = get_logger("echo_gate")


class EchoCancelGate(FrameProcessor):
    """
    Placed immediately after transport.input().
    Drops AudioRawFrames while the bot's TTS is playing so the
    microphone cannot pick up the speaker output and cause the STT
    to transcribe the bot's own voice.

    TTSSpeakingTracker (placed after the TTS service) calls
    set_bot_speaking() directly to open/close the gate.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking: bool = False

    def set_bot_speaking(self, active: bool) -> None:
        if active != self._bot_speaking:
            _echo_log.debug("EchoCancelGate: bot_speaking=%s", active)
        self._bot_speaking = active

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Barge-in: user spoke while bot was playing — open the gate right here
        # at position 2 (immediately after transport.input) so the very next
        # AudioRawFrame passes through to STT. Waiting until TypingSoundProcessor
        # (position 6) means ~6 pipeline hops of audio are dropped first.
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            if self._bot_speaking:
                _echo_log.debug("EchoCancelGate: barge-in — opening gate immediately")
                self._bot_speaking = False
        if isinstance(frame, AudioRawFrame) and self._bot_speaking:
            return  # swallow mic audio while bot is speaking
        await self.push_frame(frame, direction)


class PostSpeechGate(FrameProcessor):
    """
    Placed between stt_log and echo_vad in the pipeline.

    Drops TranscriptionFrames that arrive within grace_secs of the bot
    finishing speech.  Prevents speaker ring-down, audio echo, or ambient
    noise from triggering a false LLM turn immediately after the bot speaks.

    TTSSpeakingTracker calls start_grace_period() on BotStoppedSpeakingFrame.
    """

    def __init__(self, grace_secs: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._grace_secs = grace_secs
        self._grace_until: float = 0.0

    def start_grace_period(self) -> None:
        self._grace_until = time.monotonic() + self._grace_secs
        _echo_log.debug("PostSpeechGate: grace started (%.1fs)", self._grace_secs)

    def in_grace_period(self) -> bool:
        return time.monotonic() < self._grace_until

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and self.in_grace_period():
            _echo_log.debug(
                "PostSpeechGate: dropped during grace: %r", frame.text[:40]
            )
            return  # silently drop — prevents false LLM turn from ring-down
        await self.push_frame(frame, direction)


class TTSSpeakingTracker(FrameProcessor):
    """
    Placed after transport.output() in the pipeline.

    Controls the echo-cancel gate (so mic is muted during bot audio),
    triggers PostSpeechGate grace period after bot finishes speaking,
    and logs per-turn latency.

    Gate logic:
      BotStartedSpeakingFrame  → close gate immediately
      BotStoppedSpeakingFrame  → schedule gate-open after 200 ms tail delay
                                  + start PostSpeechGate grace period (1 s)

    Latency:
      TTSStartedFrame          → stamp bot_started; set _real_tts_active
      BotStoppedSpeakingFrame  → log summary only when _real_tts_active
    """

    def __init__(
        self,
        gate: "EchoCancelGate",
        post_speech_gate: "PostSpeechGate | None" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gate = gate
        self._post_speech_gate = post_speech_gate
        self._stop_handle = None
        self._real_tts_active: bool = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            # Real Cartesia synthesis started — stamp latency and arm the flag
            self._real_tts_active = True
            _TurnLatency.stamp("bot_started")
            e2e_ms = (
                (_TurnLatency.bot_started - _TurnLatency.vad_start) * 1000
                if _TurnLatency.vad_start else 0
            )
            _echo_log.info("TTS synthesis started  e2e=%.0fms", e2e_ms)

        elif isinstance(frame, BotStartedSpeakingFrame):
            # Transport started playing audio (keyboard OR real TTS) — close gate
            if self._stop_handle:
                self._stop_handle.cancel()
                self._stop_handle = None
            self._gate.set_bot_speaking(True)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            loop = asyncio.get_event_loop()
            if self._stop_handle:
                self._stop_handle.cancel()
            # 200 ms tail — absorbs speaker ring-down; cancelled if TTS restarts sooner
            self._stop_handle = loop.call_later(
                0.2, lambda: self._gate.set_bot_speaking(False)
            )
            # 1 s grace — drops any TranscriptionFrame caused by ring-down or noise
            if self._post_speech_gate:
                self._post_speech_gate.start_grace_period()
            if self._real_tts_active:
                _TurnLatency.stamp("bot_stopped")
                _TurnLatency.log_summary()
                self._real_tts_active = False

        elif isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            # Barge-in cleanup: cancel pending tail timer and confirm gate is open.
            # EchoCancelGate already opened the gate at position 2; this just
            # cancels the delayed call_later so it can't re-close the gate later.
            if self._stop_handle:
                self._stop_handle.cancel()
                self._stop_handle = None
            self._gate.set_bot_speaking(False)

        await self.push_frame(frame, direction)

class EchoCancelVADProcessor(FrameProcessor):
    """
    Lightweight replacement for TypingSoundProcessor when typing sound is disabled.
    Closes the echo gate the moment VAD detects the user has stopped speaking so
    that the mic is silenced during the STT → LLM → TTS latency gap.
    Reopens the gate instantly if the user interrupts (starts speaking again).
    TTSSpeakingTracker handles the gate during actual bot audio playback.
    """

    def __init__(self, gate: EchoCancelGate, **kwargs):
        super().__init__(**kwargs)
        self._gate = gate

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            self._gate.set_bot_speaking(True)
        elif isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._gate.set_bot_speaking(False)
        await self.push_frame(frame, direction)


_typing_log = get_logger("typing_sound")

# Phrases where playing an ack phrase is inappropriate (farewells, plain closings).
# Normalized to lowercase with punctuation stripped before matching.
_CLOSING_PHRASES: frozenset[str] = frozenset({
    # Farewells
    "bye", "goodbye", "good bye", "see you", "see you later", "take care",
    "bye bye", "ok bye", "okay bye", "thanks bye", "thank you bye",
    # Pure thanks (end of call)
    "thank you", "thanks", "thank you so much", "thanks a lot", "many thanks",
    "okay thank you", "ok thank you", "okay thanks", "ok thanks",
    "thank you very much", "thanks very much",
    # Call-back deferrals
    "i will call you later", "will call you later", "call you later",
    "i will call back", "will call back", "i ll call back",
    "i will get back to you", "will get back to you",
    "i ll get back", "i will get back",
    # That's all
    "that is all", "that's all", "nothing else", "nothing more", "nothing right now",
    "no thanks", "no thank you", "not right now",
})


def _is_closing_phrase(text: str) -> bool:
    """Return True when the transcript is a farewell/closing that needs no ack phrase."""
    normalized = re.sub(r"[.!?,]", "", text.lower()).strip()
    return normalized in _CLOSING_PHRASES


class TypingSoundProcessor(FrameProcessor):
    """
    Thin signal router placed after STT in the pipeline.

    Trigger order (whichever arrives first wins):
      TranscriptionFrame (non-empty, non-closing)
        → start ack phrase + keyboard immediately — faster than Silero silence
          detection, which typically fires 600 ms–1 s after Azure STT returns.
      VADUserStoppedSpeakingFrame
        → fallback trigger if STT has not yet returned (WS/server mode where
          Silero fires before Azure STT).  No-op if ack already started.

    Special cases:
      Closing phrase ("Thank you", "Bye", …)
        → cancel ack, reopen echo gate, let LLM reply naturally.
      Ghost STT (TranscriptionFrame arrives >5 s after last VAD start)
        → Azure STT buffered audio from bot speech; ignore silently.
      Empty TranscriptionFrame
        → background noise false-alarm; cancel and reopen.
      VADUserStartedSpeakingFrame
        → user interrupted; cancel keyboard, reopen gate if no audio pushed yet.
    """

    def __init__(
        self,
        typing_sound_gate: "TypingSoundGate",
        echo_gate: "EchoCancelGate",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._typing_gate = typing_sound_gate
        self._echo_gate = echo_gate
        # Set by TranscriptionFrame handler to prevent VADUserStoppedSpeakingFrame
        # from (re-)starting the ack phrase when we've already decided to skip it.
        self._skip_next_kb: bool = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            if self._skip_next_kb:
                # TranscriptionFrame already decided to skip (closing / ghost / empty)
                self._skip_next_kb = False
            else:
                # Fallback: STT hasn't fired yet — start ack now
                self._echo_gate.set_bot_speaking(True)
                self._typing_gate.start_kb()

        elif isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._skip_next_kb = False
            audio_was_started = self._typing_gate.audio_started
            self._typing_gate.stop_kb("user speaking again")
            if not audio_was_started:
                # Ack task cancelled before pushing audio — BotStartedSpeakingFrame
                # never fired, so TTSSpeakingTracker won't reopen the gate.
                self._echo_gate.set_bot_speaking(False)

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()

            # Ghost: TranscriptionFrame arrived long after the last VAD start — Azure
            # STT is finalising buffered audio from during bot speech.
            is_ghost = (
                _TurnLatency.vad_start == 0.0
                or (time.monotonic() - _TurnLatency.vad_start) > 5.0
            )

            if not text:
                # Empty = background noise; cancel and reopen
                self._skip_next_kb = True
                self._typing_gate.stop_kb("empty transcription")
                self._echo_gate.set_bot_speaking(False)

            elif is_ghost:
                age = (time.monotonic() - _TurnLatency.vad_start) if _TurnLatency.vad_start else 0
                _typing_log.warning("[TYPING] Ghost STT ignored  age=%.1fs  text=%r", age, text)
                self._skip_next_kb = True

            elif _is_closing_phrase(text):
                # Closing / farewell — cancel ack, let agent reply normally
                _typing_log.info("[TYPING] Closing phrase — skipping ack: %r", text)
                self._skip_next_kb = True
                self._typing_gate.stop_kb("closing phrase")
                self._echo_gate.set_bot_speaking(False)

            else:
                # Real turn — start ack immediately (faster than Silero silence)
                self._echo_gate.set_bot_speaking(True)
                self._typing_gate.start_kb()

        await self.push_frame(frame, direction)


class TypingSoundGate(FrameProcessor):
    """
    Placed just before transport.output() (between tts_log and transport).

    Owns the keyboard PCM loop.  Pushes plain AudioRawFrame *downstream*
    (straight to transport.output()) from an asyncio task, so keyboard
    audio never travels upstream through STT or any other processor.

    Seamless handoff:
      start_kb()         — called by TypingSoundProcessor on VADUserStoppedSpeakingFrame
      stop_kb()          — called on first real TTSAudioRawFrame from Cartesia
                           (the keyboard's last chunk finishes, then TTS takes
                           over with zero silence gap)
      stop_kb() (early)  — called by TypingSoundProcessor if user speaks again
    """

    def __init__(
        self,
        keyboard_pcm: bytes,
        sample_rate: int,
        ack_pcms: list[bytes] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pcm = keyboard_pcm
        self._sample_rate = sample_rate
        self._kb_task: asyncio.Task | None = None
        self._tts_seen: bool = False  # True once first real TTS frame passes
        self._ack_pcms: list[bytes] = list(ack_pcms) if ack_pcms else []
        self._ack_pool: list[bytes] = []  # shuffled draw pool, refilled when empty
        self._audio_started: bool = False  # True once first OutputAudioRawFrame is pushed

    def _next_ack(self) -> bytes | None:
        if not self._ack_pcms:
            return None
        if not self._ack_pool:
            self._ack_pool = list(self._ack_pcms)
            random.shuffle(self._ack_pool)
        return self._ack_pool.pop()

    @property
    def audio_started(self) -> bool:
        """True once the first OutputAudioRawFrame has actually been pushed."""
        return self._audio_started

    def start_kb(self) -> None:
        if self._kb_task and not self._kb_task.done():
            return
        self._tts_seen = False
        self._audio_started = False
        _typing_log.info("[TYPING] sound START — masking STT+LLM+TTS latency")
        if self._ack_pcms:
            self._kb_task = asyncio.create_task(self._ack_then_keyboard_loop())
        else:
            self._kb_task = asyncio.create_task(self._keyboard_loop())

    def stop_kb(self, reason: str = "stop") -> None:
        if self._kb_task and not self._kb_task.done():
            _typing_log.info("[TYPING] sound STOP — %s", reason)
            self._kb_task.cancel()
        self._kb_task = None

    def is_active(self) -> bool:
        return self._kb_task is not None and not self._kb_task.done()

    async def _ack_then_keyboard_loop(self) -> None:
        """Play one random ack phrase then seamlessly enter the keyboard loop.

        CancelledError propagates naturally during ack (stop_kb / TTS handoff).
        _keyboard_loop swallows CancelledError during the keyboard phase.
        """
        ack = self._next_ack()
        if ack:
            _typing_log.info("[ACK] playing acknowledgment phrase")
            self._audio_started = True
            await self.push_frame(
                OutputAudioRawFrame(audio=ack, sample_rate=self._sample_rate, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            # Wait for the phrase to finish playing before starting keyboard.
            # Duration = bytes / (sample_rate × 2 bytes per sample)
            duration = len(ack) / (self._sample_rate * 2)
            await asyncio.sleep(duration)
        await self._keyboard_loop()

    async def _keyboard_loop(self) -> None:
        chunk = self._sample_rate * 2 * 20 // 1000  # 20 ms of 16-bit mono PCM
        pos, n = 0, len(self._pcm)
        self._audio_started = True
        try:
            while True:
                end  = pos + chunk
                data = (self._pcm[pos:end] if end <= n
                        else self._pcm[pos:] + self._pcm[:end - n])
                pos  = end % n
                # OutputAudioRawFrame is required by LocalAudioOutputTransport
                # (transport_destination attribute comes through DataFrame→Frame).
                # Pushed directly downstream so keyboard audio never enters STT.
                await self.push_frame(
                    OutputAudioRawFrame(audio=data, sample_rate=self._sample_rate, num_channels=1),
                    FrameDirection.DOWNSTREAM,
                )
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # First real Cartesia audio chunk: stop keyboard, then let TTS through
        if isinstance(frame, TTSAudioRawFrame) and not self._tts_seen:
            self._tts_seen = True
            if self.is_active():
                self.stop_kb("real TTS audio — seamless handoff")
        elif isinstance(frame, TTSStoppedFrame):
            self._tts_seen = False  # ready for next turn
        await self.push_frame(frame, direction)


_func_log = get_logger("filter")

# Matches a complete <function=name>{...}</function> block
_FUNC_COMPLETE_RE = re.compile(r"<function=[^>]+>.*?</function>", re.DOTALL)

# The marker we're trying to detect across streamed tokens
_FUNC_MARKER = "<function="


class FunctionCallFilter(FrameProcessor):
    """
    Buffers streamed LLMTextFrame tokens and strips any raw function-call
    markup (``<function=...>...</function>``) before it reaches TTS.

    Because the LLM may stream tokens one character at a time (``<``, ``f``,
    ``u``, ``n``, …), this filter accumulates ALL tokens in a buffer and
    only emits text once it's confirmed NOT to be the start of a function
    call pattern.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_func: bool = False    # True while inside <function=...>...</function>
        self._buf: str = ""            # rolling buffer of un-emitted text

    def _flush_buffer(self) -> str | None:
        """
        Process the buffer and return any safe-to-emit text.
        Returns None if nothing should be emitted yet.
        """
        # ── Inside a function call: swallow everything until </function> ──
        if self._in_func:
            if "</function>" in self._buf:
                _, after = self._buf.split("</function>", 1)
                _func_log.debug("FunctionCallFilter: dropped function call block")
                self._buf = after
                self._in_func = False
                # Recursively process any text after the closing tag
                if self._buf:
                    return self._flush_buffer()
                return None
            # Still waiting for closing tag
            return None

        # ── Check for a COMPLETE function call block in buffer ────────────
        match = _FUNC_COMPLETE_RE.search(self._buf)
        if match:
            before = self._buf[:match.start()]
            after = self._buf[match.end():]
            _func_log.debug("FunctionCallFilter: stripped complete call")
            self._buf = after
            result = before.strip()
            if self._buf:
                more = self._flush_buffer()
                if more:
                    result = (result + " " + more).strip() if result else more
            return result if result else None

        # ── Check if buffer contains the START of a function call ─────────
        if _FUNC_MARKER in self._buf:
            idx = self._buf.index(_FUNC_MARKER)
            before = self._buf[:idx]
            self._buf = self._buf[idx:]
            self._in_func = True
            _func_log.debug("FunctionCallFilter: detected function call start")
            return before.strip() if before.strip() else None

        # ── Check if the buffer ENDS with a partial prefix of "<function=" ─
        # e.g. buffer ends with "<" or "<func" — we can't tell yet if it's
        # the start of a function call, so hold those chars back.
        marker = _FUNC_MARKER
        for i in range(min(len(self._buf), len(marker)), 0, -1):
            if self._buf.endswith(marker[:i]):
                # Hold back the partial match, emit everything before it
                emit = self._buf[:-i]
                self._buf = self._buf[-i:]
                return emit if emit.strip() else None

        # ── No function-call markers at all — safe to emit everything ─────
        emit = self._buf
        self._buf = ""
        return emit if emit else None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            self._buf += frame.text
            result = self._flush_buffer()
            if result:
                await self.push_frame(LLMTextFrame(text=result), direction)
            return

        # On end of LLM response, flush any remaining buffered text
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._buf and not self._in_func:
                await self.push_frame(LLMTextFrame(text=self._buf), direction)
            self._buf = ""
            self._in_func = False

        await self.push_frame(frame, direction)


_norm_log = get_logger("normalizer")


class TextNormalizerProcessor(FrameProcessor):
    """
    Converts comma-formatted Indian numbers in LLM output to spoken words
    before the frame reaches TTS.

    Examples:
      ₹1,00,00,000  →  "1 crore"
      ₹50,00,000    →  "50 lakhs"
      ₹20,00,000    →  "20 lakhs"
      50,000        →  "50 thousand"

    Buffers tokens into complete sentences before normalising so that
    prices like ₹20,00,000 streamed as "₹20" + ",00,000" across multiple
    tokens are always seen as a whole.
    """

    # Currency prefix (₹ / Rs / Rs.) + Indian comma-formatted number
    _CURRENCY_RE = re.compile(
        r'[₹]\s*(\d{1,3}(?:,\d{2,3})+)|'
        r'(?:Rs\.?\s*)(\d{1,3}(?:,\d{2,3})+)',
        re.IGNORECASE,
    )
    # Bare comma-separated Indian numbers not already preceded by a currency symbol
    _NUMBER_RE = re.compile(r'(?<![₹\d,])(\d{1,3}(?:,\d{2,3})+)(?!\d)')

    # FILLER-CONTEXT-v1: flush only on ? and ! — NOT on period.
    # Sentences ending with "." accumulate into the next ?/! chunk so Cartesia
    # receives them together with internal periods as natural prosodic pause cues.
    # Trailing period is stripped in _normalise so Cartesia never says "dot".
    _SENTENCE_END_RE = re.compile(r'[!?]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buf: str = ""
        self._buf_started_at: float = 0.0  # for Tier-3 time-based flush

    @staticmethod
    def _to_words(num_str: str) -> str:
        try:
            n = int(num_str.replace(',', ''))
        except ValueError:
            return num_str
        if n >= 10_000_000:
            val = n / 10_000_000
            return f"{int(val)} crore" if val == int(val) else f"{val:.1f} crore"
        if n >= 100_000:
            val = n / 100_000
            return f"{int(val)} lakh" if val == int(val) else f"{val:.1f} lakh"
        if n >= 1_000:
            val = n / 1_000
            return f"{int(val)} thousand" if val == int(val) else f"{val:.1f} thousand"
        return str(n)

    def _normalise(self, text: str) -> str:
        # Strip markdown formatting (LLM sometimes ignores the no-markdown instruction)
        text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)   # **bold** / *italic*
        text = re.sub(r"#{1,6}\s*", "", text)                      # ## heading markers
        text = re.sub(r"\n+", " ", text)                           # newlines → space
        text = re.sub(r"(?:^| )\d+\.\s+", " ", text).strip()     # "1. item" list markers
        # FILLER-CONTEXT-v1 period handling:
        # Step 1 — abbreviation dots: single letter + dot → space (T. Nagar → T Nagar)
        text = re.sub(r'(?<=[A-Za-z])(?<![A-Za-z]{2})\.(?!\d)', ' ', text)
        # Step 2 — strip trailing period only — Cartesia says "dot" if chunk ends with "."
        #   Internal periods are kept so Cartesia uses them as natural sentence-pause cues.
        text = re.sub(r'\.\s*$', '', text)
        text = re.sub(r'[/\\|]', ' ', text)             # slashes → space
        text = re.sub(r" +", " ", text).strip()
        # Currency and number normalisation
        def _repl_currency(m: re.Match) -> str:
            return self._to_words(m.group(1) or m.group(2))
        text = self._CURRENCY_RE.sub(_repl_currency, text)
        text = self._NUMBER_RE.sub(lambda m: self._to_words(m.group(1)), text)
        text = tts_normalize(text)                             # PRON-NORM-v1
        return text

    async def _flush(self, direction: FrameDirection) -> None:
        if self._buf:
            normalised = self._normalise(self._buf)
            if normalised != self._buf:
                _norm_log.debug("Normalised: %r → %r", self._buf, normalised)
            await self.push_frame(LLMTextFrame(text=normalised), direction)
            self._buf = ""
            self._buf_started_at = 0.0

    async def _flush_to_comma(self, direction: FrameDirection) -> None:
        """Tier-2 latency split: flush up to the last comma, keep remainder."""
        idx = self._buf.rfind(",")
        if idx < 20:   # guard: don't split a very short prefix
            return
        to_send = self._buf[:idx + 1]
        self._buf = self._buf[idx + 1:].lstrip()
        self._buf_started_at = time.monotonic() if self._buf else 0.0
        normalised = self._normalise(to_send)
        if normalised:
            _norm_log.debug("Comma-flush: %r", normalised[:60])
            await self.push_frame(LLMTextFrame(text=normalised), direction)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            if not self._buf_started_at:
                self._buf_started_at = time.monotonic()
            self._buf += frame.text
            # Tier 1 — sentence-ending punctuation: flush immediately
            # Guard: buffer must have a space (≥2 words) — prevents flushing a single
            # letter like "T" when the next token is "." (would cause letter-spelling).
            if self._SENTENCE_END_RE.search(frame.text) and ' ' in self._buf:
                await self._flush(direction)
            # Tier 2 — long buffer with comma: flush at last comma position
            # Mirrors bot_ev LOW_LATENCY_TTS_COMMA_CHARS logic.
            elif len(self._buf) > 60 and ',' in self._buf:
                await self._flush_to_comma(direction)
            # Tier 3 — time-based: flush anything after 0.4 s at a word boundary
            # Ensures Cartesia starts within 400 ms of first LLM token regardless
            # of sentence structure. Mirrors bot_ev LOW_LATENCY_TTS_TIME_SECONDS.
            elif (
                len(self._buf) > 30
                and self._buf[-1] in (' ', ',')
                and (time.monotonic() - self._buf_started_at) > 0.4
            ):
                await self._flush(direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._flush(direction)

        await self.push_frame(frame, direction)


class VADLogProcessor(FrameProcessor):
    """Logs VAD speech-start / speech-end frames and resets the turn latency clock."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            _TurnLatency.reset()   # t=0 for this turn
            log_vad_event(speech_detected=True)
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            log_vad_event(speech_detected=False)
        await self.push_frame(frame, direction)


# ── Shared turn state + per-turn latency tracker ─────────────────────────────

_lat_log = get_logger("latency")


class _TurnLatency:
    """
    Module-level stopwatch. Every processor stamps its milestone here.
    TTSSpeakingTracker prints the full summary when the bot stops speaking.

    All times are from time.monotonic(); deltas logged in milliseconds.
    """
    vad_start:       float = 0.0   # VADUserStartedSpeakingFrame
    stt_done:        float = 0.0   # TranscriptionFrame received from STT
    phonetic_done:   float = 0.0   # PhoneticCorrectorProcessor pushed frame
    llm_first_token: float = 0.0   # First LLMTextFrame
    llm_done:        float = 0.0   # LLMFullResponseEndFrame
    tts_first_chunk: float = 0.0   # First TTSAudioRawFrame
    bot_started:     float = 0.0   # BotStartedSpeakingFrame (actual playback)
    bot_stopped:     float = 0.0   # BotStoppedSpeakingFrame

    @classmethod
    def reset(cls) -> None:
        cls.vad_start       = time.monotonic()
        cls.stt_done        = 0.0
        cls.phonetic_done   = 0.0
        cls.llm_first_token = 0.0
        cls.llm_done        = 0.0
        cls.tts_first_chunk = 0.0
        cls.bot_started     = 0.0
        cls.bot_stopped     = 0.0

    @classmethod
    def stamp(cls, milestone: str) -> None:
        setattr(cls, milestone, time.monotonic())

    @classmethod
    def log_summary(cls) -> None:
        if not cls.vad_start:
            return
        ref = cls.vad_start

        def ms(t: float) -> str:
            return f"{(t - ref) * 1000:.0f}ms" if t else "—"

        def diff(a: float, b: float) -> str:
            return f"{(b - a) * 1000:.0f}ms" if (a and b) else "—"

        _lat_log.info(
            "\n"
            "┌─────────────────────────────────────────┐\n"
            "│           TURN LATENCY REPORT           │\n"
            "├─────────────────────────────┬───────────┤\n"
            "│ VAD speech detected         │ t=0       │\n"
            "│ STT result received         │ t=%-7s │\n"
            "│ Phonetic correction done    │ t=%-7s │\n"
            "│ LLM first token             │ t=%-7s │\n"
            "│ LLM generation complete     │ t=%-7s │\n"
            "│ TTS first audio chunk       │ t=%-7s │\n"
            "│ Bot started speaking        │ t=%-7s │\n"
            "│ Bot stopped speaking        │ t=%-7s │\n"
            "├─────────────────────────────┴───────────┤\n"
            "│ STT latency        : %-8s             │\n"
            "│ Phonetic correction: %-8s             │\n"
            "│ LLM first token    : %-8s             │\n"
            "│ LLM full response  : %-8s             │\n"
            "│ TTS to first chunk : %-8s             │\n"
            "│ E2E (VAD→audio)    : %-8s             │\n"
            "│ Bot speaking time  : %-8s             │\n"
            "└─────────────────────────────────────────┘",
            ms(cls.stt_done), ms(cls.phonetic_done), ms(cls.llm_first_token),
            ms(cls.llm_done), ms(cls.tts_first_chunk), ms(cls.bot_started), ms(cls.bot_stopped),
            diff(cls.vad_start,       cls.stt_done),
            diff(cls.stt_done,        cls.phonetic_done),
            diff(cls.stt_done,        cls.llm_first_token),
            diff(cls.llm_first_token, cls.llm_done),
            diff(cls.llm_done,        cls.tts_first_chunk),
            diff(cls.vad_start,       cls.bot_started),
            diff(cls.bot_started,     cls.bot_stopped),
        )


class _TurnState:
    """Module-level shared state so processors at different pipeline positions
    can coordinate turn IDs and pipeline state."""
    turn_id: int = 0
    state: str = "idle"

    @classmethod
    def transition(cls, new_state: str) -> None:
        if new_state != cls.state:
            get_logger("agent").info("[STATE] %s → %s", cls.state, new_state)
            cls.state = new_state


class STTLogProcessor(FrameProcessor):
    """Bolt-style STT + turn tracking.
    Must be placed BEFORE context_aggregator.user() so it sees
    TranscriptionFrame before the aggregator consumes it.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._interim_count: int = 0  # resets per turn

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            # Interim results confirm Azure STT IS receiving intelligible audio.
            # Absence of any interim before a final means audio was silence/noise/garbled.
            self._interim_count += 1
            get_logger("stt").debug(
                "[STT]  interim #%d: %r", self._interim_count, (frame.text or "")[:60]
            )

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                _TurnLatency.stamp("stt_done")
                stt_ms = (_TurnLatency.stt_done - _TurnLatency.vad_start) * 1000 if _TurnLatency.vad_start else 0
                _TurnState.turn_id += 1
                _TurnState.transition("listening")
                get_logger("agent").info("[TURN] id=%d", _TurnState.turn_id)
                get_logger("agent").info(
                    "[STT]  final=%r  stt_latency=%.0fms  interim_count=%d",
                    text, stt_ms, self._interim_count,
                )
            else:
                # Empty final transcription = background noise / too short / audio format issue.
                # interim_count=0 here strongly suggests the audio reaching STT was silence
                # or garbled (wrong format) — not recognisable as speech at all.
                get_logger("stt").warning(
                    "[STT]  empty final result  interim_count=%d  "
                    "(0 = audio was silence/noise/garbled; >0 = speech cut off early)",
                    self._interim_count,
                )
            self._interim_count = 0  # reset for next turn

        await self.push_frame(frame, direction)


class LLMLogProcessor(FrameProcessor):
    """Logs LLM response start, streaming tokens, and completion."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            get_logger("llm").info("🤖 LLM response stream started")
        elif isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text)
            log_llm_token(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            log_llm_complete("".join(self._buffer))
            self._buffer = []
        await self.push_frame(frame, direction)


class TTSLogProcessor(FrameProcessor):
    """Logs TTS audio chunks and synthesis lifecycle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._first_chunk_logged = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            self._first_chunk_logged = False
            get_logger("tts").info("TTS synthesis started")
        elif isinstance(frame, TTSAudioRawFrame):
            if not self._first_chunk_logged:
                _TurnLatency.stamp("tts_first_chunk")
                tts_ms = (
                    (_TurnLatency.tts_first_chunk - _TurnLatency.llm_done) * 1000
                    if _TurnLatency.llm_done else 0
                )
                get_logger("tts").info("TTS first chunk ready  tts_latency=%.0fms", tts_ms)
                self._first_chunk_logged = True
            log_tts_chunk(len(frame.audio))
        elif isinstance(frame, TTSStoppedFrame):
            log_tts_complete()
        await self.push_frame(frame, direction)


class AudioSmootherProcessor(FrameProcessor):  # AUDIO-SMOOTH-v1
    """
    Eliminates click/pop at TTS chunk boundaries with PCM16 fade-in/out.

    Strategy — one-chunk-behind buffer:
      TTSStartedFrame  → reset state
      TTSAudioRawFrame → fade-in the first chunk; push the PREVIOUS pending
                         chunk unmodified, hold the current one as pending
      TTSStoppedFrame  → flush pending with fade-out applied, then pass frame

    The buffer adds at most one audio chunk of latency (~20 ms) — inaudible.
    Revert keyword: AUDIO-SMOOTH-v1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_frame: TTSAudioRawFrame | None = None
        self._chunk_index: int = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._pending_frame = None
            self._chunk_index = 0
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            audio = frame.audio
            if self._chunk_index == 0:
                audio = fade_in(audio)
            # push the previous held frame, hold the current one
            if self._pending_frame is not None:
                await self.push_frame(self._pending_frame, direction)
            try:
                self._pending_frame = dataclasses.replace(frame, audio=audio)
            except Exception:
                self._pending_frame = frame  # fallback: push unmodified
            self._chunk_index += 1

        elif isinstance(frame, TTSStoppedFrame):
            # flush pending with fade-out, then pass TTSStoppedFrame
            if self._pending_frame is not None:
                try:
                    faded = dataclasses.replace(
                        self._pending_frame,
                        audio=fade_out(self._pending_frame.audio),
                    )
                except Exception:
                    faded = self._pending_frame
                await self.push_frame(faded, direction)
                self._pending_frame = None
            self._chunk_index = 0
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


class AudioInputLogProcessor(FrameProcessor):
    """Logs raw microphone frame counts (at DEBUG level to avoid spam)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frame_count = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            if self._frame_count % 50 == 0:   # log every 50 frames (~1 sec)
                audio_log.debug("🎙  %d audio frames captured", self._frame_count)
        await self.push_frame(frame, direction)


class STTAudioGateMonitor(FrameProcessor):
    """
    Placed between EchoCancelGate and AzureSTTService in the WS pipeline.

    Counts AudioRawFrames that reach STT and logs the count when a VAD-stop
    event passes through.  If the count is 0 when VAD stops, the gate was
    closed the whole time and no audio reached STT — the agent will be silent.

    Only watches VADUser* frames (directly from the VAD processor) — NOT the
    secondary UserStartedSpeakingFrame / UserStoppedSpeakingFrame emitted by
    LLMUserAggregator, which would produce false "0 frames" reports.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stt_gate_log = get_logger("stt")
        self._frame_count: int = 0
        self._byte_count: int = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            self._byte_count += len(frame.audio)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._frame_count = 0
            self._byte_count = 0
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            duration_ms = self._byte_count / (16000 * 2) * 1000  # 16kHz 16-bit mono
            if self._frame_count == 0:
                self._stt_gate_log.warning(
                    "[STT-GATE] 0 audio frames reached STT — echo gate was CLOSED "
                    "during entire user turn. STT will return empty."
                )
            else:
                self._stt_gate_log.info(
                    "[STT-GATE] %d frames (%.0f ms of audio) passed to STT",
                    self._frame_count, duration_ms,
                )
            self._frame_count = 0
            self._byte_count = 0
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────────────
# Phonetic correction: Soundex + Metaphone
# Pure-Python, no external deps, ~microseconds per call — zero latency impact.
# ─────────────────────────────────────────────────────────────────────────────

_SDEX: dict[str, str] = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6',
}


def _soundex(word: str) -> str:
    w = re.sub(r'[^A-Z]', '', word.upper())
    if not w:
        return '0000'
    result = w[0]
    prev = _SDEX.get(w[0], '0')
    for ch in w[1:]:
        code = _SDEX.get(ch, '0')
        if code != '0' and code != prev:
            result += code
        prev = code
        if len(result) == 4:
            break
    return result.ljust(4, '0')


def _metaphone(word: str) -> str:
    w = re.sub(r'[^A-Z]', '', word.upper())
    if not w:
        return ''
    if w[:2] in ('AE', 'GN', 'KN', 'PN', 'WR'):
        w = w[1:]
    if len(w) > 1 and w.endswith('E'):
        w = w[:-1]
    result: list[str] = []
    i = 0
    while i < len(w):
        c = w[i]
        if i > 0 and c == w[i - 1] and c != 'C':
            i += 1
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ''
        if c in 'AEIOU':
            if i == 0:
                result.append(c)
        elif c == 'B':
            if not (nxt == '' and i > 0 and w[i - 1] == 'M'):
                result.append('B')
        elif c == 'C':
            if nxt in 'EIY':
                result.append('S')
            elif w[i:i + 2] == 'CH':
                result.append('X'); i += 1
            elif w[i:i + 2] == 'CK':
                result.append('K'); i += 1
            else:
                result.append('K')
        elif c == 'D':
            if w[i:i + 2] == 'DG' and nxt in 'EIY':
                result.append('J'); i += 1
            else:
                result.append('T')
        elif c == 'G':
            if nxt == 'H':
                if i == 0 or w[i - 1] not in 'AEIOU':
                    result.append('K')
                i += 1
            elif nxt == 'N':
                pass
            elif nxt in 'EIY':
                result.append('J')
            else:
                result.append('K')
        elif c == 'H':
            if nxt in 'AEIOU' and (i == 0 or w[i - 1] not in 'AEIOU'):
                result.append('H')
        elif c in 'FJLMNR':
            result.append(c)
        elif c == 'K':
            if i == 0 or w[i - 1] != 'C':
                result.append('K')
        elif c == 'P':
            if nxt == 'H':
                result.append('F'); i += 1
            else:
                result.append('P')
        elif c == 'Q':
            result.append('K')
        elif c == 'S':
            if w[i:i + 2] == 'SH' or w[i:i + 3] in ('SIA', 'SIO'):
                result.append('X')
            else:
                result.append('S')
        elif c == 'T':
            if w[i:i + 2] == 'TH':
                result.append('0'); i += 1
            elif w[i:i + 3] in ('TIA', 'TIO'):
                result.append('X')
            else:
                result.append('T')
        elif c == 'V':
            result.append('F')
        elif c == 'W':
            if nxt in 'AEIOU':
                result.append('W')
        elif c == 'X':
            result.append('KS')
        elif c == 'Y':
            if nxt in 'AEIOU':
                result.append('Y')
        elif c == 'Z':
            result.append('S')
        i += 1
    return ''.join(result)


def _phonetic_score(token: str, candidate: str) -> int:
    """Score 0-100: exact=100, both algorithms=80, metaphone only=60, soundex only=40."""
    if token.upper() == candidate.upper():
        return 100
    sdx = _soundex(token) == _soundex(candidate)
    meta = _metaphone(token) == _metaphone(candidate)
    if sdx and meta:
        return 80
    if meta:
        return 60
    if sdx:
        return 40
    return 0


_LOCATION_LIST: list[str] = [
    # Chennai
    "Anna Nagar", "T Nagar", "Adyar", "Velachery", "Porur", "Perambur",
    "Tambaram", "Sholinganallur", "Pallavaram", "Chromepet", "Medavakkam",
    "Ambattur", "Avadi", "Mogappair", "Nungambakkam", "Mylapore",
    "Kodambakkam", "Guindy", "Thoraipakkam", "Perungudi", "Siruseri",
    "Kelambakkam", "Maraimalai Nagar", "Perungalathur", "Navallur", "Padur",
    # Bengaluru
    "Whitefield", "Koramangala", "Indiranagar", "Sarjapur", "Bellandur",
    "Marathahalli", "HSR Layout", "JP Nagar", "Bannerghatta", "Electronic City",
    "Yelahanka", "Hebbal", "Rajajinagar", "Malleshwaram", "Jayanagar",
    "BTM Layout", "Basavanagudi", "Devanahalli", "Kengeri", "Banashankari",
    "Vijayanagar", "Frazer Town",
    # Hyderabad
    "Gachibowli", "Kondapur", "Madhapur", "Hitech City", "Banjara Hills",
    "Jubilee Hills", "Kukatpally", "Miyapur", "Kompally", "Secunderabad",
    "Ameerpet", "Begumpet", "Manikonda", "Narsingi", "Tellapur",
    "Nallagandla", "Kokapet", "Nanakramguda", "Financial District",
    "Gandipet", "Tolichowki", "Mehdipatnam", "LB Nagar", "Dilsukhnagar",
    "Uppal", "Bachupally",
]

_NAME_LIST: list[str] = [
    "Aarav", "Aditya", "Akash", "Amith", "Anand", "Anil", "Anita", "Anitha",
    "Anjali", "Arjun", "Arun", "Ashok", "Bala", "Balaji", "Bharath",
    "Chitra", "Deepa", "Deepak", "Divya", "Ganesh", "Gopal", "Harish",
    "Haritha", "Hari", "Karthik", "Kavitha", "Kumar", "Lakshmi", "Lavanya",
    "Madhavan", "Mahesh", "Meena", "Meenakshi", "Mohan", "Muthu",
    "Nithya", "Pooja", "Prabhu", "Pradeep", "Prasad", "Priya", "Rajesh",
    "Rajan", "Ramesh", "Rekha", "Rohit", "Sanjay", "Santhosh", "Senthil",
    "Shobha", "Sridhar", "Suresh", "Swathi", "Uma", "Usha", "Vani",
    "Venkat", "Vijay", "Vikram", "Vinay", "Vishal", "Yuvaraj",
]

# Pure function words — never phonetically correct these
_STOPWORDS: frozenset[str] = frozenset({
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she',
    'they', 'them', 'it', 'its', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'the',
    'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'how', 'when', 'where',
    'why', 'all', 'some', 'any', 'both', 'yes', 'no', 'not', 'just',
    'also', 'so', 'then', 'than', 'too', 'very', 'here', 'there', 'now',
    'please', 'okay', 'ok', 'sir', 'hi', 'hello', 'name', 'want',
})

_phon_log = get_logger("phonetic")


class PhoneticCorrectorProcessor(FrameProcessor):
    """
    Intercepts TranscriptionFrame between STT and the context aggregator.
    Corrects misheard location and customer name tokens using Soundex+Metaphone.

    Context rules (derived from the last assistant message in LLMContext):
      - NAME context  : bot last asked for the user's name → score names at >=60
      - LOCATION context: bot last asked for area/city → score locations at >=60
      - PASSIVE       : always correct locations at >=80 regardless of context

    Highest-scoring candidate wins. Bigrams are tried before single tokens so
    multi-word places like "Anna Nagar" beat individual-word false matches.
    """

    _NAME_TRIGGERS = (
        'your name', 'may i have', 'name please', 'good name',
        'who am i speaking', 'your good name',
    )
    _LOC_TRIGGERS = (
        'which area', 'which location', 'preferred location', 'which city',
        'where are you', 'looking in', 'looking at', 'interested in',
        'area are you', 'location are you', 'city are you',
        'what area', 'what location', 'what city',
    )

    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        # Pre-split location list once at startup
        self._multi_locs: list[tuple[str, list[str]]] = [
            (loc, loc.split()) for loc in _LOCATION_LIST if ' ' in loc
        ]
        self._single_locs: list[str] = [loc for loc in _LOCATION_LIST if ' ' not in loc]

    def _last_assistant_text(self) -> str:
        for msg in reversed(self._context.messages):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, list):
                    return ' '.join(
                        c.get('text', '') for c in content if isinstance(c, dict)
                    )
                return str(content)
        return ''

    def _detect_context(self) -> tuple[bool, bool]:
        text = self._last_assistant_text().lower()
        awaiting_name = any(t in text for t in self._NAME_TRIGGERS)
        awaiting_location = any(t in text for t in self._LOC_TRIGGERS)
        return awaiting_name, awaiting_location

    @staticmethod
    def _best_match(token: str, candidates: list[str], threshold: int) -> tuple[str | None, int]:
        best_score, best = 0, None
        for c in candidates:
            s = _phonetic_score(token, c)
            if s > best_score:
                best_score, best = s, c
        return (best, best_score) if best_score >= threshold else (None, 0)

    def _correct(self, text: str, awaiting_name: bool, awaiting_location: bool) -> str:
        words = text.split()
        if not words:
            return text

        result = list(words)
        skip = [False] * len(words)
        loc_threshold = 60 if awaiting_location else 80

        # ── Bigram pass (multi-word locations: "Anna Nagar", "Hitech City" …) ──
        for i in range(len(words) - 1):
            if skip[i]:
                continue
            best_score, best = 0, None
            for loc, parts in self._multi_locs:
                if len(parts) == 2:
                    s = (_phonetic_score(words[i], parts[0]) +
                         _phonetic_score(words[i + 1], parts[1])) // 2
                    if s > best_score:
                        best_score, best = s, loc
            if best and best_score >= loc_threshold:
                _phon_log.debug(
                    "Bigram %r+%r → %r (score=%d)", words[i], words[i + 1], best, best_score
                )
                result[i] = best
                result[i + 1] = ''
                skip[i] = skip[i + 1] = True

        # ── Single-token pass ──────────────────────────────────────────────────
        for i, word in enumerate(words):
            if skip[i]:
                continue
            if len(word) <= 2 or re.search(r'\d', word) or word.lower() in _STOPWORDS:
                continue

            corrected = None

            # Name correction — only when bot explicitly asked for name
            if awaiting_name:
                corrected, score = self._best_match(word, _NAME_LIST, 60)
                if corrected:
                    _phon_log.debug("Name %r → %r (score=%d)", word, corrected, score)

            # Location correction
            if not corrected:
                corrected, score = self._best_match(word, self._single_locs, loc_threshold)
                if corrected:
                    _phon_log.debug(
                        "Location %r → %r (score=%d, thr=%d)", word, corrected, score, loc_threshold
                    )

            if corrected:
                result[i] = corrected

        return ' '.join(w for w in result if w)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            t0 = time.monotonic()
            awaiting_name, awaiting_location = self._detect_context()
            corrected = self._correct(frame.text, awaiting_name, awaiting_location)
            elapsed_us = (time.monotonic() - t0) * 1_000_000
            _TurnLatency.stamp("phonetic_done")
            if corrected != frame.text:
                _phon_log.info(
                    "[PHONETIC] %r → %r  (%.0fµs)", frame.text, corrected, elapsed_us
                )
                try:
                    frame = dataclasses.replace(frame, text=corrected)
                except Exception:
                    pass
            else:
                _phon_log.debug("[PHONETIC] no change (%.0fµs)", elapsed_us)
        await self.push_frame(frame, direction)


class GatherHintProcessor(FrameProcessor):
    """
    Placed between PhoneticCorrectorProcessor and context_aggregator.user().
    On each non-empty TranscriptionFrame, calls update_fn() to refresh the
    [GATHER STATE] system message so the LLM always knows what to ask next.
    update_fn is a zero-argument callable supplied by agent.py.
    """

    def __init__(self, update_fn, **kwargs):
        super().__init__(**kwargs)
        self._update_fn = update_fn

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._update_fn()
        await self.push_frame(frame, direction)


_conv_log = get_logger("agent")


class ConversationLogProcessor(FrameProcessor):
    """
    Bolt-style LLM + TTS logging (position 7, after func_filter).
    TranscriptionFrame never reaches here — STTLogProcessor handles that.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._agent_buf: list[str] = []
        self._llm_start_ts: float = 0.0
        self._llm_first_token_ts: float = 0.0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # ── LLM starts generating ───────────────────────────────────────────
        if isinstance(frame, LLMFullResponseStartFrame):
            self._agent_buf = []
            self._llm_start_ts = time.monotonic()
            self._llm_first_token_ts = 0.0
            _TurnState.transition("thinking")

        # ── First LLM token + accumulate ────────────────────────────────────
        elif isinstance(frame, LLMTextFrame) and frame.text:
            if not self._llm_first_token_ts:
                self._llm_first_token_ts = time.monotonic()
                _TurnLatency.stamp("llm_first_token")
                first_token_ms = (self._llm_first_token_ts - self._llm_start_ts) * 1000
                _conv_log.info("[LLM]  first_token=%.0fms", first_token_ms)
            self._agent_buf.append(frame.text)

        # ── LLM done → log full response + TTS ──────────────────────────────
        elif isinstance(frame, LLMFullResponseEndFrame):
            _TurnLatency.stamp("llm_done")
            full_text = "".join(self._agent_buf).strip()
            if full_text:
                total_ms = (time.monotonic() - self._llm_start_ts) * 1000
                _conv_log.info(
                    "[LLM]  done | total=%.0fms chars=%d",
                    total_ms, len(full_text),
                )
                _conv_log.info("[TTS]  speaking | %r", full_text[:120])
                _TurnState.transition("speaking")
                _conv_log.info(
                    "[TURN_SUMMARY] turn_id=%d llm=%.2fs",
                    _TurnState.turn_id, total_ms / 1000,
                )
            self._agent_buf = []

        await self.push_frame(frame, direction)


# CALLBACK-TAG-v1 ──────────────────────────────────────────────────────────────
_IMMEDIATE_CALLBACK_URGENCY: frozenset[str] = frozenset([
    "right now", "right away", "straight away", "immediately",
    "asap", "as soon as possible", "at once", "instant", "instantly",
    "this moment", "this instant",
])

_CALLBACK_INTENT: frozenset[str] = frozenset([
    "callback", "call back", "consultant", "connect", "transfer",
    "specialist", "call me", "reach me",
])


def _is_immediate_callback(text: str) -> bool:
    t = text.lower()
    return (
        any(u in t for u in _IMMEDIATE_CALLBACK_URGENCY)
        and any(c in t for c in _CALLBACK_INTENT)
    )
# END CALLBACK-TAG-v1 ───────────────────────────────────────────────────────────


# Words that follow "I am" / "I'm" that are NOT names — used in both _detect_intent
# and _generate_acknowledgment to prevent "I am driving" → "Nice to meet you Driving"
_NAME_EXCL = (
    r"looking|searching|interested|here|calling|from|in|a\b|an\b|the\b"
    r"|just|also|still|not|very|now|ready|driving|busy|going|working"
    r"|talking|unable|sorry|sure|you|we|they|this|that|there|afraid"
    r"|available|free|getting|coming|trying|using|checking|waiting"
)


class LatencyFillerProcessor(FrameProcessor):
    """
    Injects context-aware acknowledgments immediately after user speech ends to mask LLM latency.

    When a TranscriptionFrame arrives (user finished speaking), this processor:
    1. Analyzes user speech intent and content
    2. Generates a relevant acknowledgment phrase based on context
    3. Immediately pushes a TTSSpeakFrame downstream (plays before LLM reply)
    4. Logs the acknowledgment being used

    Pass context= (LLMContext) to enable context-aware name detection (single-word names
    after "May I have your name" are correctly identified as name_giving).
    """

    # Phrases in the last assistant message that indicate the bot was asking for a name
    _AWAITING_NAME_TRIGGERS: tuple[str, ...] = (
        'your name', 'may i have', 'name please', 'good name',
        'who am i speaking', 'your good name', 'say your name',
        "didn't catch", 'repeat your name', 'could you repeat',
    )

    def __init__(self, task=None, context=None, **kwargs):
        super().__init__(**kwargs)
        self._task = task
        self._context = context
        self._filler_log = get_logger("filler")

    def _last_assistant_text(self) -> str:
        """Return the text of the most recent assistant message, or '' if unavailable."""
        if not self._context:
            return ""
        for msg in reversed(self._context.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                return str(content)
        return ""

    def _awaiting_name(self) -> bool:
        """True if the last assistant turn was asking for the caller's name."""
        last = self._last_assistant_text().lower()
        return any(t in last for t in self._AWAITING_NAME_TRIGGERS)

    def _detect_intent(self, text: str) -> str:  # FILLER-CONTEXT-v1
        text_lower = text.lower()

        # Name giving — BEFORE confirmation so "Yeah, I'm Sam" / "I am Sam" → name not confirmation
        # Negative lookahead (_NAME_EXCL) prevents "I am driving" → "Nice to meet you Driving"
        # it's/that's patterns removed — too ambiguous ("it's correct" → spurious name match)
        if re.search(
            rf"\bi'?m\s+(?!{_NAME_EXCL})\w+"
            rf"|\bi\s+am\s+(?!{_NAME_EXCL})\w+"
            r"|\bmy name\b|\bname is\b|\bcall me\b",
            text_lower,
        ):
            return "name_giving"

        # Gratitude / farewell — check early so "thanks" doesn't fall to providing_info
        _gratitude = {"thank", "thanks", "thank you", "thankyou", "cheers", "bye", "goodbye", "take care"}
        if any(w in text_lower for w in _gratitude):
            return "gratitude"

        # Action request — BEFORE question so "can you connect/book..." fires correctly
        if any(w in text_lower for w in ["book", "schedule", "call back", "callback", "send",
                                          "whatsapp", "connect", "consultant", "transfer", "speak to"]):
            return "action_request"

        # Question intent
        if any(w in text_lower for w in ["what", "how", "where", "which", "can you", "tell me", "show me"]):
            return "question"

        # Confirmation/affirmation — word-boundary check to prevent "ok" matching "looking"
        # Skip to providing_info if the user is ALSO giving real information
        # ("Yeah, Velachery", "Yes, 2 crore", "Okay, 3BHK") — those deserve specific ack phrases.
        _conf_words = {"yes", "yeah", "correct", "right", "exactly", "sure", "ok", "okay", "alright"}
        _word_set = set(re.sub(r"[.,!?']", " ", text_lower).split())
        if _word_set & _conf_words:
            _entities = self._extract_entities(text)
            if not any(_entities.values()):
                return "confirmation"
            # Entities present — fall through to providing_info for a specific ack phrase

        # Negation
        if any(w in text_lower for w in ["no", "not", "don't", "doesn't", "didn't"]):
            return "negation"

        # Correction
        if any(w in text_lower for w in ["actually", "change", "different", "instead", "rather"]):
            return "correction"

        # Providing information — property, budget, location phrases
        if any(w in text_lower for w in [
            "i want", "looking for", "looking in", "looking at", "i am", "i'm looking",
            "searching", "interested", "need", "my budget", "my name",
            "apartment", "flat", "villa", "plot", "house", "bhk", "crore", "lakh", "around",
        ]):
            return "providing_info"

        # Bare single word — distinguish place name from personal name
        words = text_lower.split()
        if len(words) == 1:
            bare = words[0].strip(".,!?")
            _PLACE_SUFFIXES = (
                "pakkam", "nallur", "attur", "atur", "alur", "puram", "pattu",
                "vakkam", "mangalam", "eri", "akkam", "nagar", "aram", "ery", "thur",
            )
            _SHORT_PLACES = frozenset({
                "porur", "adyar", "omr", "ecr", "gst", "egmore", "guindy",
                "kilpauk", "padi", "avadi", "mylapore", "saidapet", "velachery",
                "tambaram", "chromepet", "ambattur", "perambur", "mogappair",
            })
            if any(bare.endswith(s) for s in _PLACE_SUFFIXES) or bare in _SHORT_PLACES:
                return "providing_info"

        return "providing_info"

    def _extract_entities(self, text: str) -> dict:
        """Extract key entities from user speech."""
        entities = {}
        text_lower = text.lower()
        
        # Cities
        cities = ["chennai", "bengaluru", "hyderabad", "bangalore"]
        for city in cities:
            if city in text_lower:
                entities["city"] = city.capitalize()
                break
        
        # Property types
        property_types = ["apartment", "villa", "plot", "flat"]
        for ptype in property_types:
            if ptype in text_lower:
                entities["property_type"] = ptype.capitalize()
                break
        
        # BHK
        bhk_pattern = re.search(r'\d\s*bhk', text_lower)
        if bhk_pattern:
            entities["bhk"] = bhk_pattern.group().upper()
        
        # Budget — detect word form AND numeric form.
        # Use [\d,]+ so "10,000 lakhs" captures "10,000" (not just "000").
        if "crore" in text_lower:
            crore_match = re.search(r'([\d,]+\.?\d*)\s*crore', text_lower)
            if crore_match:
                entities["budget"] = f"{crore_match.group(1).replace(',', '')} crore"
        elif "lakh" in text_lower or "lac" in text_lower:
            lakh_match = re.search(r'([\d,]+)\s*(?:lakh|lac)', text_lower)
            if lakh_match:
                entities["budget"] = f"{lakh_match.group(1).replace(',', '')} lakh"
        elif re.search(r'\d{1,3}(?:,\d{2,3}){2,}', text_lower):
            # Numeric Indian format: 2,00,00,000 or 50,00,000 etc.
            entities["budget"] = "that"
        
        # Location/area — FILLER-CONTEXT-v1
        # Abbreviations that .title() mangles — use exact display form
        _AREA_DISPLAY_CASE: dict[str, str] = {
            "omr": "OMR",
            "ecr": "ECR",
            "gst": "GST Road",
        }
        areas = [
            "t nagar", "velachery", "omr", "ecr", "anna nagar", "porur", "adyar",
            "mylapore", "sholinganallur", "tambaram", "chromepet", "ambattur",
            "perambur", "mogappair", "nungambakkam", "kodambakkam", "guindy",
            "thoraipakkam", "perungudi", "siruseri", "kelambakkam", "pallavaram",
            "medavakkam", "egmore", "kilpauk", "saidapet", "avadi", "poonamallee",
            "madipakkam", "kotturpuram", "besant nagar", "thiruvanmiyur",
            "perumbakkam", "navalur", "padur", "karapakkam", "oragadam",
        ]
        for area in areas:
            if area in text_lower:
                entities["area"] = _AREA_DISPLAY_CASE.get(area, area.title())
                break
        # Bare single-word input that matches a place suffix → treat as area
        if "area" not in entities:
            words = text_lower.split()
            if len(words) == 1:
                bare = words[0].strip(".,!?")
                _PLACE_SUFFIXES = (
                    "pakkam", "nallur", "attur", "atur", "alur", "puram", "pattu",
                    "vakkam", "mangalam", "eri", "akkam", "nagar", "aram", "ery", "thur",
                )
                if any(bare.endswith(s) for s in _PLACE_SUFFIXES):
                    entities["area"] = words[0].strip(".,!?").title()
        
        return entities

    def _generate_acknowledgment(self, text: str) -> str:  # FILLER-CONTEXT-v1
        intent = self._detect_intent(text)
        entities = self._extract_entities(text)

        area  = entities.get("area", "")
        ptype = entities.get("property_type", "").lower()
        budget = entities.get("budget", "")
        bhk   = entities.get("bhk", "")

        # ── Context-aware single-word name override ──────────────────────────
        # When the bot just asked for the caller's name and the response is 1–2
        # words (e.g. "Sam", "Ravi Kumar"), treat the last content word as a name.
        # This handles the case where STT returned "My name is." (cut off) and
        # the name arrives as a standalone utterance on the NEXT turn.
        if self._awaiting_name() and intent not in ("gratitude", "action_request"):
            words_raw = [w.strip(".,!?") for w in text.split() if w.strip(".,!?")]
            if len(words_raw) <= 2:
                content_words = [w for w in words_raw if w.lower() not in _STOPWORDS]
                if content_words:
                    name_candidate = content_words[-1].capitalize()
                    # Guard: don't treat property-type keywords as names
                    _PROP_WORDS = frozenset({"apartment", "villa", "plot", "flat", "house", "bhk"})
                    if name_candidate.lower() not in _PROP_WORDS:
                        return f"Nice to meet you {name_candidate}"

        # ── Name giving ───────────────────────────────────────────────────────
        if intent == "name_giving":
            # The lookahead (?!_FUNC_WORDS) prevents capturing stopwords/auxiliaries as names.
            # "My name is." → optional-skip of "is" would capture "is" without the guard.
            _FUNC_WORDS = r"(?:is|was|will|are|be|a|an|the|just|also|from)\b"
            name_match = re.search(
                rf"\bi'?m\s+(?!{_NAME_EXCL})(\w+)"
                rf"|\bi\s+am\s+(?!{_NAME_EXCL})(\w+)"
                rf"|\bmy name(?:\s+is)?\s+(?!{_FUNC_WORDS})(\w+)"
                r"|\bthis is\s+(\w+)"
                r"|\bcall me\s+(\w+)"
                r"|\bname is\s+(\w+)",
                text.lower(),
            )
            if name_match:
                name = next(g for g in name_match.groups() if g).capitalize()
                return f"Nice to meet you {name}"
            # No name word extracted (e.g. "My name is." — STT cut off before name).
            # Return empty so the LLM can re-ask cleanly without a confusing nameless greeting.
            return ""

        # ── Gratitude / farewell ─────────────────────────────────────────────
        if intent == "gratitude":
            return random.choice(["My pleasure", "Happy to help", "Glad I could help"])

        # ── Confirmation ──────────────────────────────────────────────────────
        if intent == "confirmation":
            if area or ptype or budget or bhk:
                return random.choice(["Let me pull that up", "One moment"])
            return random.choice(["Sure", "Of course", "Noted"])

        # ── Negation ──────────────────────────────────────────────────────────
        if intent == "negation":
            return random.choice(["Let me try different options", "Let me adjust that"])

        # ── Action request ────────────────────────────────────────────────────
        if intent == "action_request":
            return random.choice(["Let me arrange that", "On it"])

        # ── Correction ────────────────────────────────────────────────────────
        if intent == "correction":
            return random.choice(["Let me update that", "Noted"])

        # ── Providing info — context-aware ────────────────────────────────────
        if intent == "providing_info":
            if area and ptype and bhk:
                return f"Let me find {bhk} {ptype} options in {area}"
            if area and ptype:
                return f"Let me find {ptype} options in {area}"
            if area and bhk:
                return f"Checking {bhk} options in {area}"
            if area:
                return random.choice([
                    f"Oh, you're looking in {area}",
                    f"Got it, {area}",
                ])
            if ptype:
                return f"Let me pull {ptype} options for you"
            if budget:  # BUDGET-ACK-v1: all phrases use punctuation for Cartesia prosody + "your" not "that"
                return random.choice([
                    "Got it. Checking within your budget",
                    "Sure. Searching within your budget",
                    "Noted. Let me find options in your range",
                ])
            if bhk:
                return f"Checking {bhk} options for you"
            # Budget mentioned by keyword but amount not parseable (e.g. "my budget is 10,000")
            text_lower = text.lower()
            if any(w in text_lower for w in ("budget", "price range", "my range")):
                return random.choice(["Noted", "Got it"])
            return random.choice(["Let me check that", "One moment"])

        # ── Question / generic ────────────────────────────────────────────────
        return random.choice(["Let me check that", "One moment"])

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # When user speech ends, inject acknowledgment immediately
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if user_text:
                # CALLBACK-TAG-v1: tag immediate-callback requests so the LLM
                # skips date/time collection and submits right away.
                if _is_immediate_callback(user_text):
                    self._filler_log.info(
                        "[FILLER] Immediate callback — tagging transcript: %r", user_text[:80]
                    )
                    try:
                        frame = dataclasses.replace(frame, text=f"[IMMEDIATE_CALLBACK] {user_text}")
                    except Exception:
                        pass
                # END CALLBACK-TAG-v1

                acknowledgment = self._generate_acknowledgment(user_text)
                acknowledgment = acknowledgment.rstrip('.')
                if acknowledgment:
                    self._filler_log.info("[FILLER] Acknowledgment: %r (user said: %r)", acknowledgment, user_text[:60])
                    ack_frame = TTSSpeakFrame(text=acknowledgment, append_to_context=False)
                    await self.push_frame(ack_frame, direction)
                else:
                    self._filler_log.debug("[FILLER] No acknowledgment for: %r", user_text[:60])

        await self.push_frame(frame, direction)
