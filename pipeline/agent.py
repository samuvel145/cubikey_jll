"""
pipeline/agent.py
JLL Voice Sales Agent — local terminal pipeline.

Replaces the general-purpose agent.py.
Wires: Mic -> Deepgram STT -> Groq LLM (JLL tools) -> Cartesia TTS -> Speaker

Tool call flow (derived from IntegrationToolHandler in bot (1).py):
  LLM emits function_call -> llm.register_function handler -> jll_client HTTP call
  -> result returned to Pipecat -> LLM speaks the result

References:
  - bot (1).py: pipeline assembly, TranscriptionLogger, IntegrationToolHandler
  - integration (1).js: proxy routes, field names
"""

from __future__ import annotations

import json
import logging

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import StartFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.processors.audio.vad_processor import VADProcessor

from config import settings
from logger import get_logger, log_pipeline_event
from pipeline import jll_client
from pipeline.prompts import build_gather_hint, build_system_prompt
from pipeline.processors import AudioSmootherProcessor, ConversationLogProcessor, EchoCancelGate, EchoCancelVADProcessor, FunctionCallFilter, LatencyFillerProcessor, PhoneticCorrectorProcessor, PostSpeechGate, STTAudioGateMonitor, STTLogProcessor, TextNormalizerProcessor, TTSLogProcessor, TTSSpeakingTracker, VADLogProcessor, _TurnLatency  # AUDIO-SMOOTH-v1: AudioSmootherProcessor added
from pipeline.tools import TOOL_SCHEMAS, JLLToolHandler

log = get_logger("agent")

# Filler phrases spoken by TTS the moment a tool call fires,
# so there is no silence while the API runs.
_TOOL_FILLERS: dict[str, str] = {
    "search_properties":   "Let me pull up those listings for you.",
    "get_property_details": "Let me get the details on that.",
    "areas_by_budget":     "Let me check which areas fit your budget.",
    "submit_callback":     "Booking that callback for you now.",
    "schedule_site_visit": "Scheduling your site visit now.",
}


async def run_agent() -> None:
    log.info("[bold cyan]JLL Voice Agent starting…[/bold cyan]")
    log_pipeline_event("INIT", "Building pipeline components")

    # ── Transport (mic + speaker) ─────────────────────────────────────────────
    log_pipeline_event("TRANSPORT", "Initialising LocalAudioTransport")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.SAMPLE_RATE,
            audio_out_sample_rate=settings.SAMPLE_RATE,
            audio_in_channels=settings.CHANNELS,
            audio_out_channels=settings.CHANNELS,
            # Keep True so mic stays active. Pipecat 1.1.0 LocalAudioTransport
            # may not properly re-enable mic input when set to False.
            audio_in_passthrough=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    # Lowered from 0.7 → more tolerant of softer speech
                    confidence=0.5,
                    start_secs=0.2,
                    stop_secs=float(settings.SILENCE_THRESHOLD_MS) / 1000,
                    # Lowered to 0.2 → ensures mic is picked up after bot stops
                    min_volume=0.2,
                )
            ),
            input_device_index=settings.AUDIO_INPUT_DEVICE_INDEX,
            output_device_index=settings.AUDIO_OUTPUT_DEVICE_INDEX,
        ),
    )

    # ── STT — Azure Speech (en-IN) ────────────────────────────────────
    log_pipeline_event("STT", f"Initialising Azure Speech region={settings.AZURE_SPEECH_REGION}")
    from pipecat.transcriptions.language import Language
    stt = AzureSTTService(
        api_key=settings.AZURE_STT_KEY,
        region=settings.AZURE_SPEECH_REGION,
        sample_rate=settings.SAMPLE_RATE,
        settings=AzureSTTService.Settings(language=Language.EN_IN),
    )

    # ── LLM — Azure OpenAI (gpt-4o-mini) ────────────────────────────────
    log_pipeline_event("LLM", f"Initialising Azure OpenAI deployment={settings.AZURE_OPENAI_DEPLOYMENT}")
    llm = AzureLLMService(
        api_key=settings.AZURE_OPENAI_API_KEY,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        settings=AzureLLMService.Settings(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        ),
    )

    # ── TTS — Cartesia sonic-2 ────────────────────────────────────────
    log_pipeline_event("TTS", f"Initialising Cartesia voice_id={settings.CARTESIA_VOICE_ID[:8]}...")
    from pipecat.services.cartesia.tts import CartesiaTTSSettings
    tts = CartesiaTTSService(
        api_key=settings.CARTESIA_API_KEY,
        sample_rate=settings.SAMPLE_RATE,
        settings=CartesiaTTSSettings(
            voice=settings.CARTESIA_VOICE_ID,
            model="sonic-2",
        ),
    )

    # ── LLM Context (system prompt + tool schemas) ────────────────────────────
    system_prompt = build_system_prompt(settings.JLL_ASSISTANT_NAME)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=ToolsSchema(
            standard_tools=[],
            custom_tools={AdapterType.OPENAI: TOOL_SCHEMAS},
        ),
    )
    context_aggregator = LLMContextAggregatorPair(context=context)

    # ── Tool handler ──────────────────────────────────────────────────────────
    tool_handler = JLLToolHandler()

    # ── Pipeline assembly ─────────────────────────────────────────────────────
    log_pipeline_event("PIPELINE", "Assembling pipeline stages")
    func_filter        = FunctionCallFilter()
    text_normalizer    = TextNormalizerProcessor()
    stt_log            = STTLogProcessor()
    latency_filler     = LatencyFillerProcessor()
    echo_gate          = EchoCancelGate()
    conv_log           = ConversationLogProcessor()
    tts_log            = TTSLogProcessor()
    vad_log            = VADLogProcessor()
    post_speech_gate   = PostSpeechGate(grace_secs=1.0)
    tts_tracker        = TTSSpeakingTracker(gate=echo_gate, post_speech_gate=post_speech_gate)
    phonetic_corrector = PhoneticCorrectorProcessor(context=context)
    echo_vad           = EchoCancelVADProcessor(gate=echo_gate)
    audio_smoother     = AudioSmootherProcessor()              # AUDIO-SMOOTH-v1

    pipeline = Pipeline(
        [
            transport.input(),               # 1.  Raw mic
            echo_gate,                       # 2.  Drop mic frames while bot speaks
            vad_log,                         # 3.  Reset latency clock on VAD speech start
            stt,                             # 4.  Azure STT → TranscriptionFrame
            stt_log,                         # 5.  STT log + stt_latency stamp
            latency_filler,                  # 6.  Inject filler words to mask latency
            post_speech_gate,                # 7.  Drop transcriptions within 1 s of bot stopping
            echo_vad,                        # 8.  Close echo gate on VAD stop, reopen on start
            phonetic_corrector,              # 9.  Phonetic correction for names + locations
            context_aggregator.user(),       # 10. Accumulate user turn
            llm,                             # 11. Azure OpenAI LLM
            func_filter,                     # 12. Drop function-call markup
            conv_log,                        # 13. LLM log + llm_first_token / llm_done stamps
            text_normalizer,                 # 14. Number normalisation + pronunciation
            tts,                             # 15. Cartesia TTS
            tts_log,                         # 16. TTS first chunk stamp
            audio_smoother,                  # 17. PCM fade-in/out (AUDIO-SMOOTH-v1)
            transport.output(),              # 18. Speaker
            tts_tracker,                     # 19. Echo gate control + latency report
            context_aggregator.assistant(),  # 20. Store assistant turn
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Set task reference on latency_filler so it can queue frames
    latency_filler._task = task

    # ── Tool call handlers ────────────────────────────────────────────────────
    # Queue a filler phrase the moment the tool fires so TTS plays while the
    # API runs â€” eliminates the silence gap between LLM tool call and result.
    def _make_tool_handler(tool_name: str):
        async def _handler(params) -> None:
            args = params.arguments
            _update_gather_hint(context, tool_handler)
            filler = _TOOL_FILLERS.get(tool_name)
            if filler:
                await task.queue_frame(TTSSpeakFrame(text=filler, append_to_context=False))
            t0 = __import__("time").monotonic()
            result_text = await tool_handler.handle(tool_name, args)
            elapsed = __import__("time").monotonic() - t0
            log.info(
                "[TOOL] %-22s | %s | %.2fs",
                tool_name,
                json.dumps({k: v for k, v in args.items() if k in ("city", "location", "property_type", "min_price", "max_price")}, ensure_ascii=False),
                elapsed,
            )
            log.info("[TOOL-RESULT] %s", result_text[:120])
            await params.result_callback(result_text)
        return _handler

    for schema in TOOL_SCHEMAS:
        func_name: str = schema["function"]["name"]
        llm.register_function(func_name, _make_tool_handler(func_name))

    # ── Startup: let LLM speak the opening from system_prompt.txt ────────────
    @task.event_handler("on_pipeline_started")
    async def _trigger_opening(t: PipelineTask, _frame: StartFrame) -> None:
        log_pipeline_event("GREET", "Triggering opening from system_prompt.txt")
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await t.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "[BEGIN]"}],
                run_llm=True,
            )
        )

    # ── Runner ────────────────────────────────────────────────────────────────
    log_pipeline_event("READY", "Pipeline assembled — starting runner")
    log.info(
        "[bold green]✅ JLL Agent ready.[/bold green] "
        "Hear the greeting, then speak. Press [bold]Ctrl+C[/bold] to exit."
    )

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        log_pipeline_event("SHUTDOWN", "KeyboardInterrupt received")
        log.info("[yellow]Shutting down…[/yellow]")
    finally:
        log_pipeline_event("CLEANUP", "Pipeline task cancelled")
        await task.cancel()
        await jll_client.close_client()
        log.info("Agent stopped cleanly.")


async def run_agent_ws(websocket, stream_sid: str = "") -> None:
    """WebSocket pipeline for phone/remote clients (Exotel/Vodafone protocol).

    Audio flows over WebSocket as base64-encoded PCM JSON frames:
      {"event": "media", "media": {"payload": "<base64_pcm>"}}
    This is the same Exotel Media Streams format used by Vodafone India.
    ExotelFrameSerializer handles resampling between pipeline rate and 8 kHz.
    """
    # Reset latency tracker so this session's opening greeting doesn't inherit
    # vad_start from the previous session (which produced bogus 200s+ E2E numbers).
    _TurnLatency.reset()

    log.info("[bold cyan]JLL Voice Agent (WebSocket) starting…[/bold cyan]")
    log_pipeline_event("INIT", "Building WebSocket pipeline components")

    # ── Transport — FastAPI WebSocket + Exotel serializer ────────────────────
    serializer = ExotelFrameSerializer(stream_sid=stream_sid)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.SAMPLE_RATE,
            audio_out_sample_rate=settings.SAMPLE_RATE,
            audio_in_channels=settings.CHANNELS,
            audio_out_channels=settings.CHANNELS,
            audio_in_passthrough=True,  # pass frames downstream for VADProcessor
            serializer=serializer,
            # Split large ack-phrase / TTS frames into 20 ms chunks so the
            # carrier (Exotel/Vodafone) receives properly sized payloads and
            # the _write_audio_sleep clock stays accurate.
            fixed_audio_packet_size=640,  # 20 ms × 16 kHz × 2 bytes = 640 bytes
        ),
    )

    # ── VAD — explicit processor (WebSocket transport has no built-in VAD) ───
    # Phone audio has more background noise than a clean mic, so use a higher
    # confidence threshold to avoid false-triggers on line hiss/static.
    # stop_secs=0.5 gives Azure STT enough audio to finalise before the gate
    # closes; the previous 0.2s caused premature gate-close and empty STT results.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.5,
                min_volume=0.3,
            )
        )
    )

    # ── STT ───────────────────────────────────────────────────────────────────
    from pipecat.transcriptions.language import Language
    stt = AzureSTTService(
        api_key=settings.AZURE_STT_KEY,
        region=settings.AZURE_SPEECH_REGION,
        sample_rate=settings.SAMPLE_RATE,
        settings=AzureSTTService.Settings(language=Language.EN_IN),
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = AzureLLMService(
        api_key=settings.AZURE_OPENAI_API_KEY,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        settings=AzureLLMService.Settings(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        ),
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    from pipecat.services.cartesia.tts import CartesiaTTSSettings
    tts = CartesiaTTSService(
        api_key=settings.CARTESIA_API_KEY,
        sample_rate=settings.SAMPLE_RATE,
        settings=CartesiaTTSSettings(
            voice=settings.CARTESIA_VOICE_ID,
            model="sonic-2",
        ),
    )

    # ── Context ───────────────────────────────────────────────────────────────
    system_prompt = build_system_prompt(settings.JLL_ASSISTANT_NAME)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=ToolsSchema(
            standard_tools=[],
            custom_tools={AdapterType.OPENAI: TOOL_SCHEMAS},
        ),
    )
    context_aggregator = LLMContextAggregatorPair(context=context)
    tool_handler = JLLToolHandler()

    # ── Processors ────────────────────────────────────────────────────────────
    func_filter        = FunctionCallFilter()
    text_normalizer    = TextNormalizerProcessor()
    stt_log            = STTLogProcessor()
    latency_filler     = LatencyFillerProcessor()
    echo_gate          = EchoCancelGate()
    conv_log           = ConversationLogProcessor()
    tts_log            = TTSLogProcessor()
    vad_log            = VADLogProcessor()
    post_speech_gate   = PostSpeechGate(grace_secs=0.3)  # was 1.0 — too aggressive, dropped fast user replies
    tts_tracker        = TTSSpeakingTracker(gate=echo_gate, post_speech_gate=post_speech_gate)
    phonetic_corrector = PhoneticCorrectorProcessor(context=context)
    echo_vad           = EchoCancelVADProcessor(gate=echo_gate)
    audio_smoother     = AudioSmootherProcessor()              # AUDIO-SMOOTH-v1
    stt_gate_monitor   = STTAudioGateMonitor()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),               # 1.  WebSocket audio in
            vad,                             # 2.  Silero VAD
            echo_gate,                       # 3.  Drop mic frames while bot speaks
            stt_gate_monitor,                # 4.  Confirm audio is reaching STT (diagnostic)
            vad_log,                         # 5.  Reset latency clock on VAD speech start
            stt,                             # 6.  Azure STT → TranscriptionFrame
            stt_log,                         # 7.  STT log + stt_latency stamp
            latency_filler,                  # 8.  Inject filler words to mask latency
            post_speech_gate,                # 9.  Drop transcriptions within 0.3 s of bot stopping
            echo_vad,                        # 10. Close echo gate on VAD stop, reopen on start
            phonetic_corrector,              # 11. Phonetic correction for names + locations
            context_aggregator.user(),       # 12. Accumulate user turn
            llm,                             # 13. Azure OpenAI LLM
            func_filter,                     # 14. Drop function-call markup
            conv_log,                        # 15. LLM log
            text_normalizer,                 # 16. Number normalisation + pronunciation
            tts,                             # 17. Cartesia TTS
            tts_log,                         # 18. TTS first chunk stamp
            audio_smoother,                  # 19. PCM fade-in/out (AUDIO-SMOOTH-v1)
            transport.output(),              # 20. WebSocket audio out
            tts_tracker,                     # 21. Echo gate control + latency report
            context_aggregator.assistant(),  # 22. Store assistant turn
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Set task reference on latency_filler so it can queue frames
    latency_filler._task = task

    # ── Tool handlers ─────────────────────────────────────────────────────────
    def _make_tool_handler(tool_name: str):
        async def _handler(params) -> None:
            args = params.arguments
            _update_gather_hint(context, tool_handler)
            filler = _TOOL_FILLERS.get(tool_name)
            if filler:
                await task.queue_frame(TTSSpeakFrame(text=filler, append_to_context=False))
            t0 = __import__("time").monotonic()
            result_text = await tool_handler.handle(tool_name, args)
            elapsed = __import__("time").monotonic() - t0
            log.info(
                "[TOOL] %-22s | %s | %.2fs",
                tool_name,
                json.dumps({k: v for k, v in args.items() if k in ("city", "location", "property_type", "min_price", "max_price")}, ensure_ascii=False),
                elapsed,
            )
            log.info("[TOOL-RESULT] %s", result_text[:120])
            await params.result_callback(result_text)
        return _handler

    for schema in TOOL_SCHEMAS:
        func_name: str = schema["function"]["name"]
        llm.register_function(func_name, _make_tool_handler(func_name))

    # ── Opening greeting ──────────────────────────────────────────────────────
    @task.event_handler("on_pipeline_started")
    async def _trigger_opening(t: PipelineTask, _frame: StartFrame) -> None:
        log_pipeline_event("GREET", "Triggering opening from system_prompt.txt")
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await t.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "[BEGIN]"}],
                run_llm=True,
            )
        )

    # ── Runner ────────────────────────────────────────────────────────────────
    log_pipeline_event("READY", "WebSocket pipeline assembled — waiting for audio")
    log.info("[bold green]✅ JLL Agent (WS) ready.[/bold green] stream_sid=%s", stream_sid)

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except Exception as exc:
        log.warning("[WS] Pipeline ended: %s", exc)
    finally:
        log_pipeline_event("CLEANUP", "WebSocket session ended")
        await task.cancel()
        log.info("[WS] Agent session closed. stream_sid=%s", stream_sid)


def _update_gather_hint(context: LLMContext, tool_handler: JLLToolHandler) -> None:
    """
    Inject a gather-state hint into the system message so the LLM always
    knows what has been collected and what to ask next.
    Mirrors GatherStateHint processor in bot (1).py.
    """
    hint = build_gather_hint(tool_handler.gathered)
    messages = context.messages

    # Replace existing gather hint system message if present
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and "[GATHER STATE]" in msg.get("content", ""):
            messages[i] = {"role": "system", "content": f"[GATHER STATE]\n{hint}"}
            return

    # Insert after the main system prompt
    messages.insert(1, {"role": "system", "content": f"[GATHER STATE]\n{hint}"})
