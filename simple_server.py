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
import re
import copy

HAN_RE = re.compile(r'[\u4e00-\u9fff]')

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

whisper_lock = asyncio.Lock()

class TranslationManager:
    @staticmethod
    def _sync_translate(text: str, target_lang: str) -> str:
        """Blocking translation call with source='auto'."""
        text = text.strip()
        if not text: return text
        
        last_exception = None
        for attempt in range(3):
            try:
                if TRANSLATE_PROVIDER == "google":
                    if attempt > 0: time.sleep(0.5 * attempt)
                    result = GoogleTranslator(source="auto", target=target_lang).translate(text)
                    if result and HAN_RE.search(result) and attempt < 2:
                        continue
                    return result
                elif TRANSLATE_PROVIDER == "libre":
                    resp = requests.post(
                        LIBRE_TRANSLATE_URL,
                        json={"q": text, "source": "auto", "target": target_lang, "format": "text"},
                        timeout=TRANSLATE_TIMEOUT
                    )
                    if resp.status_code == 200:
                        return resp.json().get("translatedText", text)
            except Exception as e:
                last_exception = e
                if attempt < 2: time.sleep(0.5)
        
        if last_exception:
            logger.error(f"Translation failed: {last_exception}")
        return text

    @classmethod
    async def translate_batch_async(cls, texts: List[str], target_lang: str) -> List[str]:
        """Translates multiple texts in batches to reduce API overhead."""
        if TRANSLATE_PROVIDER == "off" or not texts:
            return texts
            
        batch_size = 8
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Use a more robust delimiter
            delimiter = f"\n\n--- SEG {int(time.time())} ---\n\n"
            joined_text = delimiter.join(batch)
            
            try:
                translated_joined = await asyncio.wait_for(
                    asyncio.to_thread(cls._sync_translate, joined_text, target_lang),
                    timeout=TRANSLATE_TIMEOUT
                )
                
                # Split and clean.
                translated_segments = [s.strip() for s in translated_joined.split(delimiter.strip())]
                # Filter empty segments
                translated_segments = [s for s in translated_segments if s]
                
                if len(translated_segments) == len(batch):
                    results.extend(translated_segments)
                else:
                    logger.warning(f"Batch mismatch: expected {len(batch)}, got {len(translated_segments)}. Falling back to individual.")
                    for txt in batch:
                        results.append(cls._sync_translate(txt, target_lang))
            except Exception as e:
                logger.error(f"Batch translation error: {e}. Falling back to individual.")
                for txt in batch:
                    results.append(cls._sync_translate(txt, target_lang))
                
        return results

def is_valid_segment(s) -> bool:
    text = s.text.strip()
    if not text: 
        return False
    
    # Lenient check for single-character languages (Chinese, etc)
    if len(text) < 1: 
        return False
    
    # Relaxed thresholds for better sensitivity (APPROVED)
    if s.no_speech_prob > 0.99: 
        return False 
    if s.avg_logprob < -2.0:  
        return False
    if (s.end - s.start) < 0.1: 
        return False
    return True

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{msecs:03}"



def regroup_by_words(segments, max_chars=40, max_dur=2.5, max_gap=0.6):
    """
    Reconstruct segments from word timestamps to effectively 'split' long lines 
    and 'merge' short ones naturally based on pauses and punctuation.
    """
    new_segments = []
    current_words = []
    current_len = 0
    current_start = 0.0
    last_end = 0.0
    
    # Flatten all words from VALID segments
    all_words = []
    for s in segments:
        if hasattr(s, 'words') and s.words:
            all_words.extend(s.words)
    
    if not all_words:
        # Fallback if no word info available
        for s in segments:
            new_segments.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip()
            })
        return new_segments

    for w in all_words:
        # initialize start of new segment
        if not current_words:
            current_start = w.start
        
        # Check for meaningful gap (silence)
        is_gap = (w.start - last_end) > max_gap if last_end > 0 else False
        
        # Check length / duration
        # Note: w.word usually includes leading space for latin languages
        w_len = len(w.word)
        is_long = (current_len + w_len) > max_chars
        is_toolong_dur = (w.end - current_start) > max_dur
        
        # Check punctuation (Sentence endings)
        text_so_far = "".join(current_words).strip()
        has_punctuation = text_so_far and text_so_far[-1] in ['.', '?', '!', '。', '？', '！']
        
        # DECIDE TO SPLIT
        if current_words and (is_gap or is_long or is_toolong_dur or has_punctuation):
            new_segments.append({
                "start": current_start,
                "end": last_end,
                "text": text_so_far
            })
            current_words = []
            current_len = 0
            current_start = w.start
        
        current_words.append(w.word)
        current_len += w_len
        last_end = w.end

    # Flush last chunk
    if current_words:
        new_segments.append({
            "start": current_start,
            "end": last_end,
            "text": "".join(current_words).strip()
        })
        
    return new_segments

async def process_segments_for_langs(segments, target_langs: List[str], needs_translation: bool):
    valid_raw_segments = [s for s in segments if is_valid_segment(s)]
    if not valid_raw_segments: return {}

    # 1. Regroup based on Words (Natural Flow) - Done ONCE
    processed_data_template = regroup_by_words(valid_raw_segments, max_chars=42, max_dur=3.5, max_gap=1.0)
    logger.info(f"Regrouped into {len(processed_data_template)} natural segments.")

    results = {}
    for lang in target_langs:
        # Deep copy template for each language
        processed_data = copy.deepcopy(processed_data_template)
        
        # 2. Translate the NEW segments
        if needs_translation:
            texts_to_translate = [item["text"] for item in processed_data]
            translated_texts = await TranslationManager.translate_batch_async(texts_to_translate, lang)
            for i, translated in enumerate(translated_texts):
                processed_data[i]["text"] = translated

        # 3. Smart Duration Enforcement (Readability Fix)
        final_segments = []
        min_duration = 1.5 

        for i, s in enumerate(processed_data):
            next_start = processed_data[i+1]["start"] if i < len(processed_data) - 1 else float('inf')
            current_dur = s["end"] - s["start"]
            available_room = max(0, (next_start - 0.1) - s["end"])
            
            if current_dur < min_duration:
                needed = min_duration - current_dur
                s["end"] += min(needed, available_room)
                
            final_segments.append(s)
        
        results[lang] = final_segments

    return results

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

        # target_langs is what the user wants (e.g., 'id' or 'en' or 'id,en')
        target_langs = language.split(",") if language else ["id"]
        # Whisper native 'translate' is NEVER used
        use_whisper_native_translate = False
        
        if whisper_lock.locked():
            raise HTTPException(
                status_code=429,
                detail="Server busy, try again later"
            )
            
        async with whisper_lock:
            logger.info(f"Processing job for file {file.filename} (Targets: {target_langs})")
            
            # We ALWAYS use auto-detection for the source language to avoid 
            # feeding the target language (e.g. 'id') as the source language to Whisper.
            segments_gen, info = model.transcribe(
                tmp_path,
                task="transcribe",
                language="zh", # Force source language to Chinese
                initial_prompt=DRACIN_PROMPT,
                beam_size=5,   # Increased for better Chinese recognition
                temperature=0,
                word_timestamps=True, # Improves sync accuracy significantly
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=3000, 
                    speech_pad_ms=800, 
                    threshold=0.3
                ) # Optimized Sensitive VAD
            )
            raw_segments = list(segments_gen)
            source_lang = info.language
            logger.info(f"Detected source language: {source_lang}")

        # ALWAYS translate externally if provider is not off
        needs_external_translate = (TRANSLATE_PROVIDER != "off")

        multi_processed_data = await process_segments_for_langs(
            raw_segments, 
            target_langs, 
            needs_external_translate
        )

        if response_format == "vtt":
            if len(target_langs) == 1:
                lang = target_langs[0]
                content = ["WEBVTT\n"]
                for r in multi_processed_data.get(lang, []):
                    start = format_timestamp(r["start"])
                    end = format_timestamp(r["end"])
                    content.append(f"{start} --> {end}\n{r['text']}\n")
                return PlainTextResponse("\n".join(content))
            else:
                # Return multiple VTTs in JSON
                vtt_results = {}
                for lang in target_langs:
                    content = ["WEBVTT\n"]
                    for r in multi_processed_data.get(lang, []):
                        start = format_timestamp(r["start"])
                        end = format_timestamp(r["end"])
                        content.append(f"{start} --> {end}\n{r['text']}\n")
                    vtt_results[lang] = "\n".join(content)
                return vtt_results
        
        if len(target_langs) == 1:
            lang = target_langs[0]
            full_text = " ".join([r["text"] for r in multi_processed_data.get(lang, [])])
            return {"text": full_text.strip()}
        else:
            json_results = {}
            for lang in target_langs:
                full_text = " ".join([r["text"] for r in multi_processed_data.get(lang, [])])
                json_results[lang] = {"text": full_text.strip()}
            return json_results

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
