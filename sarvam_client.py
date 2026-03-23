"""
Sarvam AI API client — Document Intelligence (OCR) & Chat Completions.

Document Intelligence API is async / job-based:
  1) Create job
  2) Get upload URL
  3) Upload file (to Azure Blob presigned URL)
  4) Start job
  5) Poll status
  6) Download results
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── API key validation at import time ────────────────────────────────────
if not config.SARVAM_API_KEY:
    log.warning("SARVAM_API_KEY is empty — check your .env file")

# ── HTTP session with retries ────────────────────────────────────────────
_session = requests.Session()
_retries = Retry(
    total=config.MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT"],
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _safe_json(resp: requests.Response) -> dict:
    """Safely parse JSON even if server returns text/html on errors."""
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


def _raise_for_status_with_details(resp: requests.Response, context: str):
    """Raise HTTPError with extra response text for easier debugging."""
    if 200 <= resp.status_code < 300:
        return
    detail = resp.text[:2000] if resp.text else ""
    raise requests.HTTPError(
        f"{context} failed: {resp.status_code} {resp.reason}. Response: {detail}",
        response=resp
    )


def _headers_json() -> dict[str, str]:
    """Standard Sarvam API headers for OCR / Doc Intelligence endpoints."""
    if not config.SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set. Add it to your .env file.")

    return {
        "api-subscription-key": config.SARVAM_API_KEY,
        "Content-Type": "application/json",
    }


def _headers_chat() -> dict[str, str]:
    """Headers for /v1/chat/completions — OpenAI-compatible endpoint.

    Sarvam's chat endpoint accepts both api-subscription-key and
    Authorization: Bearer.  Send both so it works regardless of
    which auth scheme the account tier requires.
    """
    if not config.SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set. Add it to your .env file.")

    return {
        "api-subscription-key": config.SARVAM_API_KEY,
        "Authorization": f"Bearer {config.SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }


def _guess_content_type(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".jpg") or fn.endswith(".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


# ═══════════════════════════════════════════════════════════════════════════
# Document Intelligence (OCR / Vision)
# ═══════════════════════════════════════════════════════════════════════════

def _create_job() -> str:
    """Step 1 — create an OCR job, return job_id."""
    url = config.SARVAM_DOC_INTEL_URL

    payload = {
        "job_parameters": {
            "language": config.SARVAM_OCR_LANGUAGE,
            "output_format": config.SARVAM_OCR_OUTPUT_FORMAT,
        }
    }

    resp = _session.post(url, headers=_headers_json(), json=payload, timeout=config.REQUEST_TIMEOUT)
    _raise_for_status_with_details(resp, "Create OCR job")

    data = _safe_json(resp)
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Create OCR job returned no job_id. Response: {data}")

    log.info("OCR job created: %s", job_id)
    return job_id


def _get_upload_url(job_id: str, filename: str) -> str:
    """Step 2 — get a presigned upload URL for the file."""
    url = f"{config.SARVAM_DOC_INTEL_URL}/upload-files"
    payload = {"job_id": job_id, "files": [filename]}

    resp = _session.post(url, headers=_headers_json(), json=payload, timeout=config.REQUEST_TIMEOUT)
    _raise_for_status_with_details(resp, "Get upload URL")

    data = _safe_json(resp)

    upload_urls = data.get("upload_urls", {})
    if filename not in upload_urls:
        raise RuntimeError(f"Upload URL missing for {filename}. Response: {data}")

    upload_url = upload_urls[filename].get("file_url")
    if not upload_url:
        raise RuntimeError(f"file_url missing for {filename}. Response: {data}")

    log.info("Got upload URL for %s", filename)
    return upload_url


def _upload_file(upload_url: str, file_bytes: bytes, filename: str):
    """
    Step 3 — PUT file bytes to the presigned URL.

    IMPORTANT:
    Sarvam gives an Azure Blob SAS URL. Azure often requires:
      - x-ms-blob-type: BlockBlob
      - Content-Type must match file
    """
    content_type = _guess_content_type(filename)

    upload_headers = {
        "Content-Type": content_type,
        "x-ms-blob-type": "BlockBlob",
    }

    # Use plain requests.put (NOT the retry session) — Azure SAS URLs
    # must not carry session-level headers or retry stale tokens.
    resp = requests.put(
        upload_url,
        data=file_bytes,
        headers=upload_headers,
        timeout=config.REQUEST_TIMEOUT,
    )

    _raise_for_status_with_details(resp, "Upload file to Azure presigned URL")

    log.info("File uploaded successfully (%s)", filename)


def _start_job(job_id: str):
    """Step 4 — start processing."""
    url = f"{config.SARVAM_DOC_INTEL_URL}/{job_id}/start"
    resp = _session.post(url, headers=_headers_json(), timeout=config.REQUEST_TIMEOUT)
    _raise_for_status_with_details(resp, "Start OCR job")
    log.info("OCR job started: %s", job_id)


def _poll_job(job_id: str) -> str:
    """Step 5 — poll until job completes. Returns final state."""
    url = f"{config.SARVAM_DOC_INTEL_URL}/{job_id}/status"
    terminal_states = {"Completed", "PartiallyCompleted", "Failed"}
    elapsed = 0

    while elapsed < config.OCR_MAX_WAIT:
        resp = _session.get(url, headers=_headers_json(), timeout=config.REQUEST_TIMEOUT)
        _raise_for_status_with_details(resp, "Poll OCR job status")

        data = _safe_json(resp)
        state = data.get("job_state", "Unknown")

        log.info("OCR job %s — state: %s (elapsed %ds)", job_id, state, elapsed)
        if state in terminal_states:
            return state

        time.sleep(config.OCR_POLL_INTERVAL)
        elapsed += config.OCR_POLL_INTERVAL

    raise TimeoutError(f"OCR job {job_id} did not complete within {config.OCR_MAX_WAIT}s")


def _download_results(job_id: str) -> Dict[str, str]:
    """Step 6 — download output zip, return {filename: content} mapping."""
    url = f"{config.SARVAM_DOC_INTEL_URL}/{job_id}/download-files"
    resp = _session.post(url, headers=_headers_json(), timeout=config.REQUEST_TIMEOUT)
    _raise_for_status_with_details(resp, "Download OCR results (get URLs)")

    data = _safe_json(resp)
    download_urls = data.get("download_urls", {})

    results: Dict[str, str] = {}
    for fname, info in download_urls.items():
        dl_url = info.get("file_url")
        if not dl_url:
            continue

        dl_resp = _session.get(dl_url, timeout=config.REQUEST_TIMEOUT * 3)
        _raise_for_status_with_details(dl_resp, f"Download OCR file {fname}")

        if fname.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
                for name in sorted(zf.namelist()):
                    if not name.startswith("__") and not name.startswith("."):
                        results[name] = zf.read(name).decode("utf-8", errors="replace")
        else:
            # Might be text or json
            try:
                results[fname] = dl_resp.text
            except Exception:
                results[fname] = dl_resp.content.decode("utf-8", errors="replace")

    log.info("Downloaded %d result file(s) for job %s", len(results), job_id)
    return results


def ocr_pdf(pdf_bytes: bytes, pdf_filename: str) -> Dict[str, str]:
    """
    Full OCR pipeline for a PDF.

    Returns dict mapping output filenames to their markdown/text content.
    """
    job_id = _create_job()
    upload_url = _get_upload_url(job_id, pdf_filename)
    _upload_file(upload_url, pdf_bytes, pdf_filename)
    _start_job(job_id)
    state = _poll_job(job_id)

    if state == "Failed":
        raise RuntimeError(f"OCR job {job_id} failed.")

    if state == "PartiallyCompleted":
        log.warning("OCR job %s partially completed — some pages may have failed.", job_id)

    return _download_results(job_id)


# ═══════════════════════════════════════════════════════════════════════════
# Chat Completions
# ═══════════════════════════════════════════════════════════════════════════

def chat_completion(
    messages: list[dict[str, str]],
    model: str = config.SARVAM_CHAT_MODEL,
    temperature: float = config.SARVAM_CHAT_TEMPERATURE,
    max_tokens: int = config.SARVAM_CHAT_MAX_TOKENS,
) -> dict[str, Any]:
    """Call Sarvam chat completions endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    resp = _session.post(
        config.SARVAM_CHAT_URL,
        headers=_headers_chat(),
        json=payload,
        timeout=config.REQUEST_TIMEOUT,
    )

    _raise_for_status_with_details(resp, "Chat completion")

    data = _safe_json(resp)
    log.info("Chat completion usage: %s", data.get("usage", {}))
    return data


def chat_answer(messages: list[dict[str, str]], **kwargs) -> str:
    """Convenience: returns just the assistant message text."""
    data = chat_completion(messages, **kwargs)
    return data["choices"][0]["message"]["content"]
