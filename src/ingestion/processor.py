import os
import sys
import time
import uuid
import json 
import logfire

# Safety net for the changelog-file incident: if a single file still produces
# more chunks than this after quality filtering, don't blindly embed all of
# them — log it and cap the file instead of burning the whole rate-limit
# budget (or worse, the daily quota) on one file.
MAX_CHUNKS_PER_FILE = 100

# Seconds to sleep between files to ease pressure on the embedding API.
SLEEP_BETWEEN_FILES = 3

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config.config import settings

from src.retrieval.embeddings import embed_texts, get_embedding_dim

from src.ingestion.loaders.pdf import parse_pdf     
from src.ingestion.loaders.html import parse_html
from src.ingestion.loaders.text import parse_text
from src.ingestion.chunking.splitter import chunk_text
from src.ingestion.text_cleanup import strip_markup_noise

# Add this function near the top of processor.py, after imports

_NAV_PATTERNS = (
    "llms.txt", "/index", "sitemap", "documentation index",
    "fetch the complete", "AccordionGroup", "<Accordion",
    "theme={\"theme\":",
)

def _code_line_ratio(text: str) -> float:
    """
    Fraction of lines that look like code, rather than prose.
    A chunk can legitimately contain a short inline example; what we want to
    reject is chunks that are *mostly* code fences/API signatures with little
    surrounding explanation (previously the "```python" in text substring
    check killed the whole chunk even when it was 90% prose with one snippet).
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    code_like = 0
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            code_like += 1
            continue
        if in_fence:
            code_like += 1
            continue
        # Heuristics for code-ish lines outside fences: indentation, or
        # syntax-heavy lines with little natural-language content.
        if line.startswith(("    ", "\t")):
            code_like += 1
        elif stripped and sum(ch in "{}()[]=<>;" for ch in stripped) / len(stripped) > 0.15:
            code_like += 1
    return code_like / len(lines)


def _is_quality_chunk(text: str, min_words: int = 40, max_code_ratio: float = 0.6) -> bool:
    """
    Returns False for chunks that are navigational/structural noise:
      - Too short to carry answer signal
      - Known nav page markers
      - Mostly URLs (link-dump tables)
      - Mostly code with little surrounding prose (density-based, not a
        blunt "contains a code fence" check, so a short inline example
        inside an otherwise-explanatory chunk survives)
    """
    words = text.split()
    if len(words) < min_words:
        return False
    lowered = text.lower()
    if any(pat.lower() in lowered for pat in _NAV_PATTERNS):
        return False
    # Reject if >30% of "words" look like URLs
    url_count = sum(1 for w in words if w.startswith("http") or w.startswith("https"))
    if url_count / len(words) > 0.30:
        return False
    if _code_line_ratio(text) > max_code_ratio:
        return False
    return True

# Local folder where parsed + chunked JSON metadata is saved
PROCESSED_DATA_DIR = "processed_data"

# Lazy Qdrant client — instantiated on first use so a missing env var
# raises a clear error at call time, not silently at import time.
_qdrant_client: QdrantClient | None = None

def _get_qdrant() -> QdrantClient:
    """Return (and lazily create) the shared Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        if not settings.QDRANT_URL:
            raise EnvironmentError("QDRANT_CLUSTER_ENDPOINT is not set in your .env file.")
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _qdrant_client


def _with_retries(fn, *args, attempts: int = 3, delay: float = 2.0, **kwargs):
    """
    Retry a callable a few times on transient network/DNS failures.
    Seen twice now: a Qdrant call failing with 'No address associated with
    hostname' or 'Name or service not known' — both DNS resolution blips,
    not code bugs. This wraps the handful of Qdrant calls most likely to
    hit that (wipe, create, upsert) so a one-off network hiccup doesn't
    take down the whole run or leave the collection half-wiped.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            msg = str(e)
            transient = (
                "No address associated" in msg
                or "Name or service not known" in msg
                or "Temporary failure in name resolution" in msg
                or "Connection" in msg
            )
            if not transient or attempt == attempts:
                raise
            logfire.warning(
                f"Transient network error on attempt {attempt}/{attempts} "
                f"({msg}) — retrying in {delay}s."
            )
            time.sleep(delay)
    raise last_exc


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def parse_and_chunk_file(file_path: str, filename: str, source_type: str, force: bool = False):
    """
    Parse → clean → chunk → quality-filter → save locally.
    Returns the list of chunks (str), or None if the file was skipped
    (unsupported type, empty, all-noise, or already processed and not forced).
    Does NOT embed or touch Qdrant — that happens in a second pass, after
    cross-file duplicates have been identified across the whole directory.
    """
    with logfire.span("Parsing File", file=filename, source=source_type):
        base_name_check = os.path.splitext(filename)[0]
        existing_path = os.path.join(PROCESSED_DATA_DIR, source_type, f"{base_name_check}.json")
        if not force and os.path.exists(existing_path):
            logfire.info(f"Loading already-parsed chunks for {filename} from {existing_path}.")
            with open(existing_path, "r", encoding="utf-8") as f:
                return json.load(f).get("chunks", [])

        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            full_text = parse_pdf(file_path)
        elif ext in ("html", "htm"):
            full_text = parse_html(file_path)
        elif ext in ("txt", "md"):
            full_text = parse_text(file_path)
        elif ext in ("docx", "pptx"):
            from src.ingestion.loaders.office import parse_office
            full_text = parse_office(file_path)
        else:
            logfire.warning(f"Skipping unsupported file type: {filename}")
            return None

        if not full_text or not full_text.strip():
            logfire.warning(f"No text extracted from {filename} — skipping.")
            return None

        full_text = strip_markup_noise(full_text)
        if not full_text.strip():
            logfire.warning(f"{filename} was entirely markup noise after cleanup — skipping.")
            return None

        chunks = chunk_text(full_text)
        if not chunks:
            return None

        before = len(chunks)
        chunks = [c for c in chunks if _is_quality_chunk(c)]
        logfire.info(f"Quality filter: {before} → {len(chunks)} chunks kept from {filename}.")
        if not chunks:
            logfire.warning(f"All chunks filtered out for {filename} — skipping.")
            return None

        if len(chunks) > MAX_CHUNKS_PER_FILE:
            logfire.warning(
                f"{filename} produced {len(chunks)} chunks after filtering "
                f"(> cap of {MAX_CHUNKS_PER_FILE}) — likely a changelog/reference "
                f"dump rather than prose docs. Truncating to first {MAX_CHUNKS_PER_FILE}."
            )
            chunks = chunks[:MAX_CHUNKS_PER_FILE]

        return chunks


def _dedupe_across_files(file_chunks: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Given {filename: [chunks]}, drop any chunk whose exact text appears in
    more than one distinct file. Keeps the first occurrence (by filename
    sort order) and removes it from every other file. This is what catches
    the widget/boilerplate-HTML chunk that gets scraped verbatim onto every
    page of a docs site (e.g. an "Ask AI" embed) — a within-file duplicate
    check can't see this, since each file only has it once.
    """
    text_to_files: dict[str, list[str]] = {}
    for filename, chunks in file_chunks.items():
        for chunk in chunks:
            text_to_files.setdefault(chunk, []).append(filename)

    cross_file_dupe_texts = {
        text for text, files in text_to_files.items() if len(set(files)) > 1
    }
    if cross_file_dupe_texts:
        logfire.warning(
            f"Found {len(cross_file_dupe_texts)} chunk(s) duplicated verbatim "
            f"across multiple files — dropping all but a canonical copy."
        )

    keep_owner: dict[str, str] = {}
    for filename in sorted(file_chunks):
        for chunk in file_chunks[filename]:
            if chunk in cross_file_dupe_texts and chunk not in keep_owner:
                keep_owner[chunk] = filename

    result: dict[str, list[str]] = {}
    for filename, chunks in file_chunks.items():
        kept = []
        for chunk in chunks:
            if chunk in cross_file_dupe_texts and keep_owner[chunk] != filename:
                continue  # dropped: a duplicate living in another file
            kept.append(chunk)
        result[filename] = kept
    return result


def _embedded_marker_path(source_type: str, filename: str) -> str:
    base_name = os.path.splitext(filename)[0]
    return os.path.join(PROCESSED_DATA_DIR, source_type, f"{base_name}.embedded")


def _is_already_embedded(source_type: str, filename: str) -> bool:
    return os.path.exists(_embedded_marker_path(source_type, filename))


def _mark_embedded(source_type: str, filename: str):
    path = _embedded_marker_path(source_type, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def embed_and_index_chunks(chunks: list[str], filename: str, source_type: str, force: bool = False):
    """Save final chunk list locally, then embed and upsert into Qdrant.

    Skips files that already have an `.embedded` marker from a previous
    successful run (resume-skip) — the dedup restructure means we now see
    every file's final chunk list before embedding any of them, so this
    marker is what preserves the original "don't re-embed on retry"
    guarantee. Note: if a later run adds new files that introduce a new
    cross-file duplicate against an already-embedded file, that already-
    embedded file's Qdrant copy won't be retroactively updated — rerun with
    a full --wipe if you need dedup to be exact across incremental runs.
    """
    with logfire.span("Vectorizing & Indexing", file=filename):
        if not chunks:
            logfire.warning(f"No chunks left for {filename} after dedup — nothing to index.")
            return

        if not force and _is_already_embedded(source_type, filename):
            logfire.info(f"Skipping embed for {filename} — already indexed in a previous run.")
            return

        processed_data = {"filename": filename, "source_type": source_type, "chunks": chunks}
        base_name = os.path.splitext(filename)[0]
        local_path = save_processed_locally(processed_data, source_type, base_name)
        logfire.info(f"Saved processed data → {local_path}")

        embeddings = embed_texts(chunks)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": filename,
                    "source_type": source_type,
                    "label": "true" if source_type == "true" else "noise",
                },
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        _with_retries(
            _get_qdrant().upsert,
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )
        logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
        _mark_embedded(source_type, filename)


def process_directory(dir_path: str, source_type: str):
    """
    Recursively process every file under dir_path (all nested sub-folders
    included), in two phases:
      1. Parse + chunk + quality-filter every file (cheap, no API calls).
      2. Drop chunks duplicated verbatim across files, THEN embed + index
         only the survivors.
    Cross-file dedup only sees files within one call to this function — if
    you ingest langgraph_docs and langchain_docs as two separate
    process_directory calls, a chunk duplicated between them (rather than
    within one of them) won't be caught. Combine both into one call (or one
    parent directory with both as subfolders) if you want dedup across both.
    """
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        all_files = [
            os.path.join(root, filename)
            for root, _dirs, files in os.walk(dir_path)
            for filename in files
        ]
        logfire.info(f"Found {len(all_files)} files (recursive) under {dir_path}.")

        # Phase 1: parse + chunk + quality-filter (no embedding yet)
        failed_files = []
        file_chunks: dict[str, list[str]] = {}
        for file_path in all_files:
            filename = os.path.basename(file_path)
            try:
                chunks = parse_and_chunk_file(file_path, filename, source_type)
                if chunks:
                    file_chunks[filename] = chunks
            except Exception as e:
                logfire.error(f"Skipping {filename} after parse failure: {e}")
                failed_files.append(filename)

        # Phase 2: cross-file exact-duplicate removal
        file_chunks = _dedupe_across_files(file_chunks)

        # Phase 3: embed + index the survivors, one file at a time so a
        # rate-limit failure on one file doesn't take down the rest.
        for filename, chunks in file_chunks.items():
            try:
                embed_and_index_chunks(chunks, filename, source_type)
            except Exception as e:
                logfire.error(f"Skipping {filename} after embed failure: {e}")
                failed_files.append(filename)
            time.sleep(SLEEP_BETWEEN_FILES)

        if failed_files:
            logfire.warning(
                f"{len(failed_files)} file(s) failed and were skipped: {failed_files}"
            )


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if _with_retries(_get_qdrant().collection_exists, settings.QDRANT_COLLECTION):
                    _with_retries(_get_qdrant().delete_collection, settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

                # Clear resume-skip markers too — otherwise every file looks
                # "already embedded" against a collection that's now empty,
                # and the whole ingest silently does nothing.
                cleared = 0
                for root, _dirs, files in os.walk(PROCESSED_DATA_DIR):
                    for fname in files:
                        if fname.endswith(".embedded"):
                            os.remove(os.path.join(root, fname))
                            cleared += 1
                if cleared:
                    logfire.info(f"Cleared {cleared} stale '.embedded' marker(s) after wipe.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not _with_retries(_get_qdrant().collection_exists, settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            _get_qdrant().create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )
            # Re-create payload index on 'label' for filtered queries
            from qdrant_client.models import PayloadSchemaType
            _get_qdrant().create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="label",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logfire.info("Payload index on 'label' created.")

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m src.ingestion.processor DATA --wipe
    #   python -m src.ingestion.processor DATA/true_data true

    # Configure logfire only when running this file directly, not on import
    logfire.configure(service_name="enterprise-ingestion-service")

    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")