import os
import asyncio
import tempfile
import uvicorn
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import requests

# from opentelemetry import trace
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OPENTELEMETRY SETUP (DISABLED) ---
# OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
# SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "whisper-server")

# resource = Resource.create({
#     "service.name": SERVICE_NAME,
# })

# tracer_provider = TracerProvider(resource=resource)
# otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
# span_processor = BatchSpanProcessor(otlp_exporter)
# tracer_provider.add_span_processor(span_processor)
# trace.set_tracer_provider(tracer_provider)
# tracer = trace.get_tracer(__name__)

# RequestsInstrumentor().instrument()

app = FastAPI(title="Harden Whisper STT Service")
# FastAPIInstrumentor.instrument_app(app)

# --- CONFIGURATION (ENV DRIVEN) ---
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_MAX_JOBS = int(os.getenv("WHISPER_MAX_JOBS", "1"))
TRANSLATE_TIMEOUT = float(os.getenv("TRANSLATE_TIMEOUT", "10.0"))
TRANSLATE_PROVIDER = os.getenv("TRANSLATE_PROVIDER", "google").lower() # google | libre | off
LIBRE_TRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://localhost:5000/translate")

# --- INITIAL PROMPT FOR BETTER CONTEXT ---
# This helps the model stay in the right language and style (Chinese Drama aka Dracin)
DRACIN_PROMPT = "Mandarin Chinese (simplified) drama dialogue. Casual, colloquial. 电视剧普通话对白，日常口语。"

# --- SINGLETON MODEL & CONCURRENCY CONTROL ---
logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on {WHISPER_DEVICE}...")
model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

whisper_semaphore = asyncio.Semaphore(WHISPER_MAX_JOBS)

class TranslationManager:
    @staticmethod
    def _sync_translate(text: str, target_lang: str, source_lang: str = "auto") -> str:
        """Blocking translation call to be run in a thread."""
        # Normalize source_lang for Google (zh -> zh-CN)
        s_lang = source_lang
        if s_lang == "zh": s_lang = "zh-CN"
        
        # Simple retry logic (up to 2 times)
        for attempt in range(2):
            try:
                if TRANSLATE_PROVIDER == "google":
                    return GoogleTranslator(source=s_lang, target=target_lang).translate(text)
                elif TRANSLATE_PROVIDER == "libre":
                    resp = requests.post(
                        LIBRE_TRANSLATE_URL,
                        json={"q": text, "source": s_lang, "target": target_lang, "format": "text"},
                        timeout=TRANSLATE_TIMEOUT
                    )
                    if resp.status_code == 200:
                        return resp.json().get("translatedText", text)
            except Exception as e:
                logger.warning(f"Translation attempt {attempt+1} failed: {e}")
                if attempt == 1: raise e
                time.sleep(0.5)
        return text

    @classmethod
    async def translate_async(cls, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """Non-blocking translation with hard timeout and thread offloading."""
        if TRANSLATE_PROVIDER == "off" or not text.strip():
            return text
        
        try:
            logger.info(f"Translating {source_lang} -> {target_lang}: {text[:30]}...")
            translated = await asyncio.wait_for(
                asyncio.to_thread(cls._sync_translate, text, target_lang, source_lang),
                timeout=TRANSLATE_TIMEOUT
            )
            return translated
        except asyncio.TimeoutError:
            logger.error(f"Translation timed out after {TRANSLATE_TIMEOUT}s for text: {text[:30]}...")
        except Exception as e:
            logger.error(f"Translation failed ({TRANSLATE_PROVIDER}): {e}")
        
        return text # Fallback to original text

def is_valid_segment(s) -> bool:
    text = s.text.strip()
    if not text: 
        return False
    
    # Lenient check for single-character languages (Chinese, etc)
    if len(text) < 1: 
        return False
    
    # Relaxed thresholds for better sensitivity
    if s.no_speech_prob > 0.8: # More lenient (was 0.45)
        return False 
    if s.avg_logprob < -1.2:  # More lenient (was -0.9)
        return False
    if (s.end - s.start) < 0.2: # More lenient (was 0.3)
        return False
    return True

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{msecs:03}"

def apply_casual_rewrite(text: str) -> str:
    # Case-insensitive replacements for casual Indonesian
    replacements = [
        ("tidak dapat", "nggak bisa"), # Check longer phrases first
        ("tidak", "nggak"),
        ("saya", "aku"),
        ("anda", "kamu"),
        ("akan", "bakal"),
        ("kamu tidak", "kamu nggak"),
        ("aku tidak", "aku nggak")
    ]
    
    # Simple replace - can be improved with regex borders if needed, 
    # but basic replace is often sufficient for casualizing subtitles.
    import re
    
    # Iterate through replacement pairs
    for old, new in replacements:
        # Use regex to replace case-insensitively
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        text = pattern.sub(new, text)
            
    return text

async def process_and_translate_segments(segments, target_lang: str, needs_translation: bool, source_lang: str = "auto"):
    logger.info(f"Processing {len(segments)} raw segments. Needs translate: {needs_translation}")
    
    # 1. First pass: filter out invalid segments and collect text
    valid_data = []
    for i, s in enumerate(segments):
        if is_valid_segment(s):
            valid_data.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip()
            })
        else:
            logger.debug(f"Segment {i} filtered: text='{s.text.strip()}', no_speech={s.no_speech_prob:.2f}, logprob={s.avg_logprob:.2f}")
    
    if not valid_data:
        logger.warning("No valid segments found after filtering.")
        return []

    # 2. Sequential Translation (One by One)
    if needs_translation:
        logger.info(f"Translating {len(valid_data)} segments sequentially...")
        for i, item in enumerate(valid_data):
            # Translate one by one
            translated = await TranslationManager.translate_async(item["text"], target_lang, source_lang)
            if translated:
                item["text"] = translated

    # 3. Casual Rewrite & 4. Merging Short Segments
    processed_segments = []
    if not valid_data:
        return []
        
    current_seg = valid_data[0]
    current_seg["text"] = apply_casual_rewrite(current_seg["text"])
    
    # We iterate through the REST of the segments
    for next_seg in valid_data[1:]:
        next_seg["text"] = apply_casual_rewrite(next_seg["text"])
        
        duration = current_seg["end"] - current_seg["start"]
        
        # Merge if current segment is too short (< 0.6s)
        # Check if the gap isn't too large (> 1s gap probably shouldn't merge)
        gap = next_seg["start"] - current_seg["end"]
        if duration < 0.6 and gap < 1.0:
            # Merge text with a space
            current_seg["text"] += " " + next_seg["text"]
            # Extend end time
            current_seg["end"] = next_seg["end"]
        else:
            processed_segments.append(current_seg)
            current_seg = next_seg
            
    processed_segments.append(current_seg)

    # 5. Add Delay (0.25s)
    # Ensure start >= 0
    final_segments = []
    for s in processed_segments:
        s["start"] += 0.25
        s["end"] += 0.25
        final_segments.append(s)

    logger.info(f"Final processed segments after merge: {len(final_segments)}")
    return final_segments

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form(None, alias="model"),
    response_format: str = Form("json"),
    language: str = Form(None),
    task: str = Form("transcribe")
):
    tmp_path = None
    start_time = time.time()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            payload = await file.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            tmp.write(payload)
            tmp_path = tmp.name

        # target_lang is what the user wants (e.g., 'id' or 'en')
        target_lang = language if language else "id"
        # Whisper native 'translate' only supports target='en'
        use_whisper_native_translate = (task == "translate" and target_lang == "en")
        
        async with whisper_semaphore:
            logger.info(f"Processing job for file {file.filename} (Target: {target_lang})")
            
            # We ALWAYS use auto-detection for the source language to avoid 
            # feeding the target language (e.g. 'id') as the source language to Whisper.
            segments_gen, info = model.transcribe(
                tmp_path,
                task="translate" if use_whisper_native_translate else "transcribe",
                language=None, # Auto-detect source language (e.g., detects 'zh')
                initial_prompt=DRACIN_PROMPT,
                beam_size=5,   # Increased for better Chinese recognition
                temperature=0,
                word_timestamps=True, # Improves sync accuracy significantly
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000, speech_pad_ms=400) # Relaxed VAD: 1s silence to split, 400ms pad
            )
            raw_segments = list(segments_gen)
            source_lang = info.language
            logger.info(f"Detected source language: {source_lang}")

        # If we used native translate to 'en', we don't need further translation
        # Unless the source was already English, or we want a different target.
        needs_external_translate = False
        if target_lang != source_lang:
            if not use_whisper_native_translate:
                needs_external_translate = True
            elif target_lang != "en":
                # This case shouldn't happen with current logic but for safety:
                needs_external_translate = True

        processed_data = await process_and_translate_segments(
            raw_segments, 
            target_lang, 
            needs_external_translate,
            source_lang
        )

        if response_format == "vtt":
            content = ["WEBVTT\n"]
            for r in processed_data:
                start = format_timestamp(r["start"])
                end = format_timestamp(r["end"])
                content.append(f"{start} --> {end}\n{r['text']}\n")
            return PlainTextResponse("\n".join(content))
        
        full_text = " ".join([r["text"] for r in processed_data])
        return {"text": full_text.strip()}

    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Internal server error during processing")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to delete temp file {tmp_path}: {e}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", "9002")),
        timeout_keep_alive=60 
    )
