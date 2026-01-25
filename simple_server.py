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
# Enhanced prompt with common drama phrases to improve recognition
# This helps Whisper understand the context and style of Chinese drama dialogue
DRACIN_PROMPT = """中国电视剧对白。普通话，简体中文。日常口语，情感丰富。
常见词汇：你好，谢谢，对不起，没关系，怎么了，为什么，我爱你，别走。
Chinese TV drama dialogue. Mandarin, simplified. Casual, emotional, colloquial."""

# --- SINGLETON MODEL & CONCURRENCY CONTROL ---
logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on {WHISPER_DEVICE}...")
model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

whisper_lock = asyncio.Lock()



class TranslationManager:
    """
    Subtitle Reliability Translation Manager
    
    HARD RULES (NEVER VIOLATED):
    1. Subtitles must NEVER be empty. Empty subtitle = fatal bug.
    2. Batch translation ONLY for short texts (<80 characters).
    3. Segments >120 chars MUST be split by Chinese punctuation before translation.
    4. Empty/short results trigger individual retry.
    5. Final fallback is ALWAYS original zh-CN text.
    6. Never trust batch output length, delimiters, or ordering.
    7. Determinism > Performance.
    """
    
    # Chinese sentence-ending punctuation for safe splitting
    CHINESE_PUNCTUATION = re.compile(r'([，。！？；：、])')
    
    @staticmethod
    def split_long_text(text: str, max_length: int = 120) -> List[str]:
        """
        Split long Chinese text by punctuation for safe translation.
        
        RULE: Segments >120 chars MUST be split before translation.
        Returns list of chunks, each ending at natural punctuation boundary.
        """
        text = text.strip()
        if not text:
            return []
        
        # If short enough, return as-is
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by Chinese punctuation while preserving it
        parts = TranslationManager.CHINESE_PUNCTUATION.split(text)
        
        # Check if we actually have punctuation (split produces >1 part)
        has_punctuation = len(parts) > 1
        
        if has_punctuation:
            # Use punctuation-based splitting
            for part in parts:
                if not part:
                    continue
                    
                # If adding this part would exceed limit and we have content, save chunk
                if current_chunk and len(current_chunk + part) > max_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = part
                else:
                    current_chunk += part
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            # No punctuation found - fallback to character-based splitting
            logger.warning(f"No punctuation found in long text, using character-based split")
            for i in range(0, len(text), max_length):
                chunk = text[i:i + max_length]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        # Safety: if no chunks created (shouldn't happen), return original
        if not chunks:
            logger.warning(f"split_long_text failed to create chunks, returning original text")
            return [text]
        
        logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
        return chunks
    
    @staticmethod
    def _sync_translate_safe(text: str, target_lang: str, retry_count: int = 2) -> str:
        """
        Safe blocking translation with empty-result detection and retry.
        
        RULE: If result is empty or suspiciously short, retry individually.
        RULE: Final fallback is ALWAYS original text.
        """
        text = text.strip()
        if not text:
            return text
        
        original_text = text
        min_expected_length = max(1, len(text) // 10)  # Expect at least 10% of original length
        
        for attempt in range(retry_count):
            try:
                if TRANSLATE_PROVIDER == "google":
                    result = GoogleTranslator(source="zh-CN", target=target_lang).translate(text)
                    
                    # CRITICAL: Validate result is not empty or suspiciously short
                    if result and len(result.strip()) >= min_expected_length:
                        return result.strip()
                    else:
                        logger.warning(f"Translation attempt {attempt+1} returned empty/short result for: {text[:50]}...")
                        
                elif TRANSLATE_PROVIDER == "libre":
                    resp = requests.post(
                        LIBRE_TRANSLATE_URL,
                        json={"q": text, "source": "zh", "target": target_lang, "format": "text"},
                        timeout=TRANSLATE_TIMEOUT
                    )
                    if resp.status_code == 200:
                        result = resp.json().get("translatedText", "")
                        if result and len(result.strip()) >= min_expected_length:
                            return result.strip()
                        else:
                            logger.warning(f"LibreTranslate returned empty/short result")
                            
            except Exception as e:
                logger.error(f"Translation attempt {attempt+1} failed: {e}")
        
        # FINAL FALLBACK: Return original text (NEVER empty)
        logger.warning(f"All translation attempts failed, returning original: {original_text[:50]}...")
        return original_text
    
    @classmethod
    async def safe_translate(cls, text: str, target_lang: str) -> str:
        """
        Async wrapper for safe translation with text splitting.
        
        RULE: Long text (>120 chars) is split before translation.
        RULE: Result is NEVER empty.
        """
        text = text.strip()
        if not text:
            return text
        
        # Split long text by punctuation
        chunks = cls.split_long_text(text, max_length=120)
        
        if len(chunks) == 1:
            # Short text - translate directly
            return await asyncio.to_thread(cls._sync_translate_safe, chunks[0], target_lang)
        else:
            # Long text - translate chunks individually and rejoin
            translated_chunks = []
            for chunk in chunks:
                translated = await asyncio.to_thread(cls._sync_translate_safe, chunk, target_lang)
                translated_chunks.append(translated)
            
            result = " ".join(translated_chunks)
            
            # SAFETY: Ensure result is not empty
            if not result.strip():
                logger.error(f"Chunk translation resulted in empty string, returning original")
                return text
            
            return result.strip()
    
    @classmethod
    async def translate_batch_async(cls, texts: List[str], target_lang: str) -> List[str]:
        """
        Batch translation with strict 1:1 mapping validation.
        
        RULES:
        - Batch ONLY for short texts (<80 chars)
        - Long texts (>80 chars) translated individually
        - Strict validation: input count MUST equal output count
        - Empty results trigger individual retry
        - Never trust delimiter-based splitting
        """
        if TRANSLATE_PROVIDER == "off" or not texts:
            return texts
        
        results = []
        
        # Separate short and long texts
        short_texts = []
        short_indices = []
        long_texts = []
        long_indices = []
        
        for i, text in enumerate(texts):
            text_len = len(text.strip())
            if text_len == 0:
                # Empty text - keep as-is
                results.append("")
            elif text_len < 80:
                # Short text - can batch
                short_texts.append(text)
                short_indices.append(i)
            else:
                # Long text - must translate individually
                long_texts.append(text)
                long_indices.append(i)
        
        # Initialize results array with None placeholders
        final_results = [None] * len(texts)
        
        # Handle long texts individually (RULE: >80 chars = individual)
        logger.info(f"Translating {len(long_texts)} long texts individually")
        for text, idx in zip(long_texts, long_indices):
            translated = await cls.safe_translate(text, target_lang)
            final_results[idx] = translated
        
        # Handle short texts in small batches
        if short_texts:
            logger.info(f"Translating {len(short_texts)} short texts in batches")
            batch_size = 10  # Smaller batch for reliability
            
            for i in range(0, len(short_texts), batch_size):
                batch = short_texts[i:i + batch_size]
                batch_indices = short_indices[i:i + batch_size]
                
                # Try batch translation with delimiter
                DELIMITER = " ||| "
                joined_text = DELIMITER.join(batch)
                
                try:
                    translated_joined = await asyncio.wait_for(
                        asyncio.to_thread(cls._sync_translate_safe, joined_text, target_lang),
                        timeout=TRANSLATE_TIMEOUT + (len(batch) * 0.5)
                    )
                    
                    parts = translated_joined.split(DELIMITER)
                    parts = [p.strip() for p in parts]
                    
                    # STRICT VALIDATION: Count must match
                    if len(parts) == len(batch):
                        # Validate no empty results
                        valid = True
                        for j, part in enumerate(parts):
                            if not part or len(part) < 1:
                                logger.warning(f"Batch result {j} is empty, invalidating batch")
                                valid = False
                                break
                        
                        if valid:
                            # Success - use batch results
                            for j, (part, idx) in enumerate(zip(parts, batch_indices)):
                                final_results[idx] = part
                        else:
                            # Empty results detected - retry individually
                            logger.warning(f"Empty results in batch, retrying individually")
                            for text, idx in zip(batch, batch_indices):
                                translated = await cls.safe_translate(text, target_lang)
                                final_results[idx] = translated
                    else:
                        # Count mismatch - retry individually
                        logger.warning(f"Batch count mismatch: expected {len(batch)}, got {len(parts)}. Retrying individually.")
                        for text, idx in zip(batch, batch_indices):
                            translated = await cls.safe_translate(text, target_lang)
                            final_results[idx] = translated
                            
                except Exception as e:
                    logger.error(f"Batch translation error: {e}. Falling back to individual.")
                    for text, idx in zip(batch, batch_indices):
                        translated = await cls.safe_translate(text, target_lang)
                        final_results[idx] = translated
        
        # FINAL VALIDATION: Ensure 1:1 mapping and no None values
        for i, result in enumerate(final_results):
            if result is None:
                logger.error(f"Result {i} is None! Using original text as fallback.")
                final_results[i] = texts[i]
            elif not result.strip():
                logger.error(f"Result {i} is empty! Using original text as fallback.")
                final_results[i] = texts[i]
        
        # STRICT CHECK: Output count MUST equal input count
        if len(final_results) != len(texts):
            logger.critical(f"FATAL: Output count ({len(final_results)}) != Input count ({len(texts)})")
            # Emergency fallback: return original texts
            return texts
        
        logger.info(f"Successfully translated {len(texts)} texts with 1:1 mapping")
        return final_results

def is_valid_segment(s) -> bool:
    text = s.text.strip()
    if not text: 
        return False
    
    if len(text) == 1 and not HAN_RE.search(text):
        return False

    # STRICTER thresholds to reduce hallucinations
    if s.no_speech_prob > 0.92:
        return False
    if s.avg_logprob < -1.4:
        return False
    if (s.end - s.start) < 0.2:
        return False
    return True

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{msecs:03}"



def regroup_by_words(segments, max_chars=35, max_dur=2.0, max_gap=0.4):
    new_segments = []
    current_words = []
    current_len = 0
    current_start = 0.0
    last_end = 0.0

    all_words = []
    fallback_segments = []

    for s in segments:
        if hasattr(s, "words") and s.words:
            all_words.extend(s.words)
        else:
            if s.text.strip():
                fallback_segments.append({
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip()
                })

    # 🔴 Kalau TIDAK ADA word timestamps sama sekali
    if not all_words:
        return fallback_segments

    # 🔵 Regroup berbasis word timestamps
    for w in all_words:
        if not current_words:
            current_start = w.start

        is_gap = (w.start - last_end) > max_gap if last_end > 0 else False
        w_len = len(w.word)
        is_long = (current_len + w_len) > max_chars
        is_toolong_dur = (w.end - current_start) > max_dur

        text_so_far = "".join(current_words).strip()
        has_punctuation = text_so_far and text_so_far[-1] in ['.', '?', '!', '。', '？', '！']

        if current_words and (is_gap or is_long or is_toolong_dur or has_punctuation):
            new_segments.append({
                "start": current_start,
                "end": last_end,
                "text": text_so_far
            })
            current_words = []
            current_len = 0
            current_start = w.start

        if current_words:
            prev_word = current_words[-1]
            if w.word == prev_word:
                # skip exact repetition
                last_end = max(last_end, w.end)
                continue

        # skip overlapping timestamps
        if w.start < last_end - 0.05:
            last_end = max(last_end, w.end)
            continue

        current_words.append(w.word)
        current_len += w_len
        last_end = w.end

    if current_words:
        new_segments.append({
            "start": current_start,
            "end": last_end,
            "text": "".join(current_words).strip()
        })

    # 🔵 Gabungkan fallback TANPA tabrakan waktu
    merged = new_segments[:]

    for fb in fallback_segments:
        overlap = False
        for seg in new_segments:
            if not (fb["end"] <= seg["start"] or fb["start"] >= seg["end"]):
                overlap = True
                break
        if not overlap:
            merged.append(fb)

    merged.sort(key=lambda x: x["start"])
    return merged

async def process_segments_for_langs(segments, target_langs: List[str], needs_translation: bool):
    valid_raw_segments = [s for s in segments if is_valid_segment(s)]
    if not valid_raw_segments: return {}

    # 1. Regroup based on Words (Natural Flow) - Done ONCE
    # Optimized for Chinese: shorter chars, tighter timing for better sync
    processed_data_template = regroup_by_words(valid_raw_segments, max_chars=22, max_dur=2.5, max_gap=0.4)
    logger.info(f"Regrouped into {len(processed_data_template)} natural segments.")

    results = {}
    for lang in target_langs:
        # Deep copy template for each language
        processed_data = copy.deepcopy(processed_data_template)
        
        # 2. Translate the NEW segments
        if needs_translation:
            texts_to_translate = [item["text"] for item in processed_data]
            original_texts = texts_to_translate[:]  # Keep backup for safety
            
            translated_texts = await TranslationManager.translate_batch_async(texts_to_translate, lang)
            
            # CRITICAL VALIDATION: Ensure 1:1 mapping and no empty subtitles
            if len(translated_texts) != len(texts_to_translate):
                logger.critical(f"FATAL: Translation count mismatch! Expected {len(texts_to_translate)}, got {len(translated_texts)}")
                # Emergency: use original texts
                translated_texts = original_texts
            
            for i, translated in enumerate(translated_texts):
                # RULE: Subtitles must NEVER be empty
                if not translated or not translated.strip():
                    logger.error(f"Empty subtitle detected at index {i}! Using original text: {original_texts[i][:50]}...")
                    processed_data[i]["text"] = original_texts[i]
                else:
                    processed_data[i]["text"] = translated

        # 3. Smart Duration Enforcement (Readability Fix)
        final_segments = []
        

        for i, s in enumerate(processed_data):
            next_start = processed_data[i+1]["start"] if i < len(processed_data) - 1 else float('inf')
            current_dur = s["end"] - s["start"]
            available_room = max(0, (next_start - 0.2) - s["end"])  # More gap before next

            text_len = len(s["text"].strip())

            # 🔥 PER-SUBTITLE MIN DURATION
            if text_len <= 12:
                min_duration = 0.7
            elif text_len <= 22:
                min_duration = 1.0
            else:
                min_duration = 1.3
            
            # Only extend if VERY short and there's room
            if current_dur < min_duration and available_room > 0.3:
                needed = min_duration - current_dur
                s["end"] += min(needed, available_room)

            if s["end"] <= s["start"]:
                continue

            if len(s["text"]) <= 1 and (s["end"] - s["start"]) < 0.3:
                continue
                
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
                vad_filter=True,  # ENABLED - critical for reducing hallucinations
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Shorter - more responsive
                    speech_pad_ms=300,  # Less padding - tighter timing
                    threshold=0.35  # Higher threshold - less sensitive to noise
                )
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
