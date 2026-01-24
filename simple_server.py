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
        Safe blocking translation.
        
        FIX: TRANSLATION COMPLETENESS
        - Accept ANY non-empty result.
        - No minimum length validation (short translations are valid).
        - Retry only on empty or error.
        """
        text = text.strip()
        if not text:
            return text
        
        original_text = text
        
        for attempt in range(retry_count):
            try:
                if TRANSLATE_PROVIDER == "google":
                    result = GoogleTranslator(source="zh-CN", target=target_lang).translate(text)
                    
                    # VALIDATION: Only check for empty
                    if result and result.strip():
                        return result.strip()
                    else:
                        logger.warning(f"Translation attempt {attempt+1} returned empty result for: {text[:50]}...")
                        
                elif TRANSLATE_PROVIDER == "libre":
                    resp = requests.post(
                        LIBRE_TRANSLATE_URL,
                        json={"q": text, "source": "zh", "target": target_lang, "format": "text"},
                        timeout=TRANSLATE_TIMEOUT
                    )
                    if resp.status_code == 200:
                        result = resp.json().get("translatedText", "")
                        if result and result.strip():
                            return result.strip()
                        else:
                            logger.warning(f"LibreTranslate returned empty result")
                            
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
        Batch translation with AGGRESSIVE per-line fallback for CN drama.
        
        FIX 1: TRANSLATION COMPLETENESS
        - Every input MUST produce output (never empty)
        - Batch failure → immediate per-line retry
        - ANY empty result → invalidate entire batch → retry all individually
        - Delimiter breaks → per-line fallback
        - Google merges lines → per-line fallback
        
        RULES:
        - Batch ONLY for short texts (<80 chars)
        - Long texts (>80 chars) translated individually
        - Zero tolerance for empty results
        """
        if TRANSLATE_PROVIDER == "off" or not texts:
            return texts
        
        # Initialize results array with None placeholders
        final_results = [None] * len(texts)
        
        # Separate short and long texts
        short_texts = []
        short_indices = []
        long_texts = []
        long_indices = []
        
        for i, text in enumerate(texts):
            text_len = len(text.strip())
            if text_len == 0:
                # Empty input - preserve as empty
                final_results[i] = ""
            elif text_len < 80:
                # Short text - eligible for batching
                short_texts.append(text)
                short_indices.append(i)
            else:
                # Long text - MUST translate individually
                long_texts.append(text)
                long_indices.append(i)
        
        # Handle long texts individually (RULE: >80 chars = individual)
        if long_texts:
            logger.info(f"Translating {len(long_texts)} long texts individually")
            for text, idx in zip(long_texts, long_indices):
                translated = await cls.safe_translate(text, target_lang)
                final_results[idx] = translated
        
        # Handle short texts with AGGRESSIVE batch validation
        if short_texts:
            logger.info(f"Translating {len(short_texts)} short texts in batches")
            batch_size = 10  # Small batch for CN drama reliability
            
            for i in range(0, len(short_texts), batch_size):
                batch = short_texts[i:i + batch_size]
                batch_indices = short_indices[i:i + batch_size]
                
                batch_success = False
                
                # TRY batch translation (optimistic path)
                DELIMITER = " ||| "
                joined_text = DELIMITER.join(batch)
                
                try:
                    translated_joined = await asyncio.wait_for(
                        asyncio.to_thread(cls._sync_translate_safe, joined_text, target_lang),
                        timeout=TRANSLATE_TIMEOUT + (len(batch) * 0.5)
                    )
                    
                    parts = translated_joined.split(DELIMITER)
                    parts = [p.strip() for p in parts]
                    
                    # AGGRESSIVE VALIDATION: Count + Content
                    if len(parts) == len(batch):
                        # Check EVERY result for emptiness
                        all_valid = True
                        for j, part in enumerate(parts):
                            # FIX: Only check for empty, ignore length ratio
                            if not part or not part.strip():
                                logger.warning(f"Batch item {j} is empty: '{batch[j][:30]}...' -> '{part}'")
                                all_valid = False
                                break
                        
                        if all_valid:
                            # SUCCESS - use batch results
                            for part, idx in zip(parts, batch_indices):
                                final_results[idx] = part
                            batch_success = True
                            logger.debug(f"Batch of {len(batch)} succeeded")
                        else:
                            logger.warning(f"Batch validation failed: empty/short results detected")
                    else:
                        logger.warning(f"Batch count mismatch: {len(batch)} → {len(parts)}")
                        
                except Exception as e:
                    logger.error(f"Batch translation exception: {e}")
                
                # FALLBACK: Per-line retry if batch failed
                if not batch_success:
                    logger.warning(f"Falling back to per-line translation for {len(batch)} items")
                    for text, idx in zip(batch, batch_indices):
                        translated = await cls.safe_translate(text, target_lang)
                        final_results[idx] = translated
        
        # FINAL SAFETY NET: Catch any remaining None/empty
        for i, result in enumerate(final_results):
            if result is None:
                logger.error(f"CRITICAL: Result {i} is None after all retries! Using original.")
                final_results[i] = texts[i]
            elif not result.strip() and texts[i].strip():  # Non-empty input became empty
                logger.error(f"CRITICAL: Result {i} is empty for non-empty input! Using original.")
                final_results[i] = texts[i]
        
        # STRICT CHECK: Output count MUST equal input count
        if len(final_results) != len(texts):
            logger.critical(f"FATAL: Count mismatch {len(final_results)} != {len(texts)}, returning originals")
            return texts
        
        logger.info(f"Translation complete: {len(texts)} items, 100% coverage")
        return final_results

def is_valid_segment(s) -> bool:
    text = s.text.strip()
    if not text: 
        return False
    
    if len(text) < 2 and not HAN_RE.search(text):
        return False
    
    # STRICTER thresholds to reduce hallucinations
    if s.no_speech_prob > 0.85:  # Much stricter - reject likely non-speech
        return False 
    if s.avg_logprob < -1.0:     # Much stricter - reject low confidence
        return False
    if (s.end - s.start) < 0.3:  # Longer minimum - reject very short segments
        return False
    return True

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{msecs:03}"



def regroup_by_words(segments, max_chars=20, max_dur=1.5, max_gap=0.25):
    """
    FIX 2: SPEECH-SUBTITLE SYNC (CN Drama Optimized)
    
    Regroup subtitles using word timestamps for natural dialogue flow.
    
    Parameters tuned for fast-paced Chinese drama:
    - max_chars=20: Shorter lines (was 35) → prevents long blocks
    - max_dur=1.5: Tighter timing (was 2.0) → faster subtitle changes
    - max_gap=0.25: Smaller silence (was 0.4) → more responsive breaks
    
    Result: Natural, easy-to-read subtitles that match CN drama pacing
    """
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

    # No word timestamps → use fallback segments
    if not all_words:
        return fallback_segments

    # Regroup based on word timestamps
    for w in all_words:
        if not current_words:
            current_start = w.start

        is_gap = (w.start - last_end) > max_gap if last_end > 0 else False
        w_len = len(w.word)
        is_long = (current_len + w_len) > max_chars
        is_toolong_dur = (w.end - current_start) > max_dur

        text_so_far = "".join(current_words).strip()
        has_punctuation = text_so_far and text_so_far[-1] in ['.', '?', '!', '。', '？', '！']

        # Break on: gap, length, duration, or punctuation
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

    # Add final segment
    if current_words:
        new_segments.append({
            "start": current_start,
            "end": last_end,
            "text": "".join(current_words).strip()
        })

    # Merge fallback segments (no timestamp overlap)
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
    # Optimized for Chinese: max_chars=24, max_dur=1.8, max_gap=0.3
    processed_data_template = regroup_by_words(valid_raw_segments, max_chars=24, max_dur=1.8, max_gap=0.3)
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

        # 3. TEXT-AWARE SUBTITLE DURATION (FIX 3)
        final_segments = []
        
        # Reading speed model
        MIN_DURATION = 1.0       # HARD MINIMUM 1.0s
        SAFETY_GAP = 0.05        # Minimal gap to prevent overlap but keep flow

        for i, s in enumerate(processed_data):
            next_start = processed_data[i+1]["start"] if i < len(processed_data) - 1 else float('inf')
            
            # 1. Base Duration Calculation
            text_len = len(s["text"].strip())

            if text_len > 30 and (s["end"] - s["start"]) < 1.2:
                s["end"] = s["start"] + 1.2
            
            # Adaptive CPS: Lower CPS (more time) for longer text to aid comfort
            cps = 10.0

            if text_len <= 10:
                cps = 14.0
            elif text_len <= 20:
                cps = 11.0
            elif text_len <= 30:
                cps = 9.0
            else:
                cps = 7.5
                        
            required_dur = max(MIN_DURATION, text_len / cps)
            
            # 2. Extend to meet requirement
            current_end = s["end"]
            current_dur = current_end - s["start"]
            
            if current_dur < required_dur:
                wanted_end = s["start"] + required_dur

                hard_limit = next_start - 0.01 if next_start != float("inf") else s["start"] + required_dur

                new_end = min(wanted_end, hard_limit)

                # HARD MINIMUM DURATION GUARANTEE
                if new_end < s["start"] + MIN_DURATION:
                    new_end = min(s["start"] + MIN_DURATION, hard_limit)

                # Only extend, never shorten
                s["end"] = max(current_end, new_end)

            # 3. Gap Bridging (Flow Optimization)
            if next_start != float("inf"):
                gap = (next_start - SAFETY_GAP) - s["end"]
                if 0 < gap < 0.3:
                    s["end"] = next_start - SAFETY_GAP
                 
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
                    threshold=0.5  # Higher threshold - less sensitive to noise
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
