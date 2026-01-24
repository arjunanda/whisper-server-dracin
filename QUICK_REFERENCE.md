# Subtitle Reliability - Quick Reference

## What Changed

### TranslationManager Class (simple_server.py)

#### New Methods

1. **`split_long_text(text, max_length=120)`**
   - Splits long Chinese text by punctuation
   - Fallback to character-based split if no punctuation
   - Ensures chunks never exceed max_length

2. **`safe_translate(text, target_lang)`**
   - Async wrapper with automatic text splitting
   - Guarantees non-empty results
   - Fallback to original text on failure

3. **`_sync_translate_safe(text, target_lang, retry_count=2)`**
   - Replaces old `_sync_translate`
   - Validates result length (min 10% of original)
   - Retries on empty/short results
   - Always returns original text as final fallback

#### Modified Methods

1. **`translate_batch_async(texts, target_lang)`**
   - Now separates short (<80 chars) and long (≥80 chars) texts
   - Long texts translated individually via `safe_translate`
   - Short texts batched with smaller batch size (10 instead of 15)
   - Strict validation: count match + no empty results
   - Individual retry on batch failure

### process_segments_for_langs Function

**Added validation**:

- Keeps backup of original texts
- Validates translation count matches input count
- Checks each result for emptiness
- Replaces empty results with original Chinese text
- Logs all failures

## Key Thresholds

| Threshold         | Value           | Purpose                               |
| ----------------- | --------------- | ------------------------------------- |
| Batch limit       | 80 chars        | Max length for batch translation      |
| Split threshold   | 120 chars       | Max length before splitting           |
| Batch size        | 10 items        | Number of items per batch             |
| Min result length | 10% of original | Minimum acceptable translation length |
| Retry count       | 2 attempts      | Number of retries on failure          |

## Decision Flow

```
Input Text
    │
    ├─ Empty? → Keep empty
    │
    ├─ <80 chars? → Batch translation
    │   │
    │   ├─ Success? → Use result
    │   └─ Failure? → Individual retry
    │
    └─ ≥80 chars? → Individual translation
        │
        ├─ >120 chars? → Split by punctuation
        │   │
        │   ├─ Has punctuation? → Split at punctuation
        │   └─ No punctuation? → Character-based split
        │
        ├─ Translate each chunk
        │
        ├─ Rejoin chunks
        │
        ├─ Result empty? → Use original
        └─ Result valid? → Use translation
```

## Error Handling

### Batch Translation Failures

1. Delimiter split count mismatch → Individual retry
2. Empty result in batch → Individual retry
3. Exception during batch → Individual retry

### Individual Translation Failures

1. Empty result → Retry (up to 2 times)
2. Short result (<10% original) → Retry
3. All retries failed → Use original text
4. Exception → Use original text

### Final Validation

1. Result is None → Use original text
2. Result is empty string → Use original text
3. Count mismatch → Use all original texts

## Logging Levels

### INFO

- Text splitting: chunk count
- Batch processing: item counts
- Successful translations

### WARNING

- No punctuation found (using char split)
- Batch count mismatch
- Empty results in batch
- Translation retry attempts

### ERROR

- Empty subtitle detected (using fallback)
- Batch recovery failed
- Translation attempts failed

### CRITICAL

- Fatal count mismatch (emergency fallback)

## Testing

Run the test suite:

```bash
cd /home/rbz/dracin-api/whisper-server
python3 test_reliability_standalone.py
```

Expected output:

```
✓ ALL TESTS PASSED

Hard Rules Verified:
1. ✓ No empty subtitles
2. ✓ Batch only for short texts (<80 chars)
3. ✓ Long texts (>120 chars) split before translation
4. ✓ Empty results trigger fallback
5. ✓ Final fallback is original text
6. ✓ 1:1 mapping guaranteed
7. ✓ Deterministic behavior
```

## Monitoring Checklist

After deployment, monitor for:

- [ ] Zero empty subtitles in production
- [ ] Long dialogue (>120 chars) successfully translated
- [ ] No "CRITICAL" log messages
- [ ] Batch success rate (should be >90% for short texts)
- [ ] Individual translation fallback rate (should be <10%)
- [ ] Overall translation coverage (should be 100%)

## Rollback Plan

If issues occur:

1. Check logs for CRITICAL/ERROR messages
2. Verify test suite still passes
3. If needed, revert to previous version
4. File bug report with:
   - Input text that failed
   - Expected vs actual output
   - Relevant log messages

## Performance Impact

**Expected changes**:

- Slightly more API calls (due to splitting)
- Slightly longer processing time (due to validation)
- **Zero missing subtitles** (worth the trade-off)

**Benchmarks** (approximate):

- Short text (<80 chars): ~Same speed (batched)
- Long text (80-120 chars): ~Same speed (individual)
- Very long text (>120 chars): ~2x slower (split + translate chunks)

## FAQ

**Q: Why 80 chars for batch limit?**
A: Google Translate becomes unreliable with long text in batch mode. 80 chars is a safe threshold based on testing.

**Q: Why 120 chars for split threshold?**
A: Balances translation quality (shorter is better) with API efficiency (fewer calls).

**Q: What if there's no punctuation?**
A: Character-based split at 120 chars as fallback.

**Q: What if translation fails completely?**
A: Original Chinese text is shown (better than empty subtitle).

**Q: Can I adjust the thresholds?**
A: Yes, but test thoroughly. The current values are optimized for Chinese drama dialogue.

**Q: Does this work for other languages?**
A: The logic works, but punctuation regex is Chinese-specific. Adapt `CHINESE_PUNCTUATION` for other languages.
