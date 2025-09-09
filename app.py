# citation_verifier_app.py
# ------------------------------------------------------------
# A Gradio app that extracts citations from a PDF/DOCX paper and
# verifies each citation using your existing Agent + WebSearchTool.
#
# Deps:
#   pip install gradio PyPDF2 python-docx pandas
# ------------------------------------------------------------

from agents import Agent, WebSearchTool, trace, Runner, gen_trace_id, function_tool, RunConfig
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from IPython.display import display, Markdown

import gradio as gr
import PyPDF2
from docx import Document as DocxDocument
import tempfile
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pathlib
import urllib.parse
import html
import json

# Try cloudscraper for 403/anti-bot walls (optional)
try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except Exception:
    HAS_CLOUDSCRAPER = False

# -------------------------
# 0) Load API keys
# -------------------------
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set (and this is optional)")

if groq_api_key:
    print(f"Groq API Key exists and begins {groq_api_key[:4]}")
else:
    print("Groq API Key not set (and this is optional)")

# ---------------------------------------------------
# 1) Search agent and paper metadata extraction agent
# ---------------------------------------------------
INSTRUCTIONS = (
    "You are a research assistant. Given a citation from an academic paper, you search the web for that citation "
    "and produce a concise summary of the results. The summary must show whether you found an exact match for the doi "
    "or title of the citation, or if you found a close match, or if you did not find a match. In the case that you find "
    "an exact match or a close match, include a link to the reference you found. It is vital you do not make up a reference "
    "if it does not exist. You are specifically trying to verify the validity of the citation. Do not include any additional "
    "commentary other than the summary itself."
)

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)

# ---- Paper metadata extraction ----
META_INSTRUCTIONS = """\
Extract overall paper metadata from the provided full text. Return ONLY a JSON object:
{
  "title": "<string or null>",
  "authors": "<semicolon-separated authors or null>",
  "doi": "<doi or null>",
  "abstract": "<abstract text or null>"
}
Guidelines:
- Detect DOI via pattern (starts with "10." and has a slash) if present in front matter.
- Title is usually the largest/top text; if ambiguous, best-effort.
- Authors: preserve order; join with semicolons.
- Abstract: text under 'Abstract' heading if present; omit headings/labels.
- If a field is unknown, return null.
"""

metadata_agent = Agent(
    name="Paper metadata agent",
    instructions=META_INSTRUCTIONS,
    tools=[],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="none"),
)

ABSTRACT_HEAD = re.compile(r"^\s*abstract\s*:?\s*$", re.IGNORECASE | re.MULTILINE)

async def extract_paper_metadata(full_text: str) -> Dict[str, str]:
    # Try LLM first
    try:
        txt = await run_agent(metadata_agent, f"TEXT START\n{full_text}\nTEXT END")
        data = json.loads(txt)
        meta = {
            "title": (data.get("title") or "").strip(),
            "authors": (data.get("authors") or "").strip(),
            "doi": (data.get("doi") or "").strip(),
            "abstract": (data.get("abstract") or "").strip(),
        }
        # If abstract is missing, try a quick heuristic
        if not meta["abstract"]:
            meta["abstract"] = heuristic_abstract(full_text)
        return meta
    except Exception:
        # Fallback heuristics
        doi = (DOI_REGEX.search(full_text).group(0) if DOI_REGEX.search(full_text) else "")
        title = heuristic_title(full_text)
        authors = ""  # could be enhanced later
        abstract = heuristic_abstract(full_text)
        return {"title": title, "authors": authors, "doi": doi, "abstract": abstract}

def heuristic_title(full_text: str) -> str:
    head = full_text.strip().splitlines()[:40]  # first 40 lines
    # choose first non-empty line with decent length
    for line in head:
        s = line.strip()
        if 20 <= len(s) <= 200 and not s.lower().startswith(("abstract", "introduction")):
            return s
    return ""

def heuristic_abstract(full_text: str) -> str:
    # Look for "Abstract" heading; return next ~1800 chars or until a blank line gap
    m = ABSTRACT_HEAD.search(full_text[: max(40000, len(full_text))])
    if not m:
        return ""
    after = full_text[m.end():]
    # stop at first big section break
    stop = re.search(r"\n\s*\n\s*[A-Z][A-Za-z ]{3,40}\n", after)
    chunk = after[: stop.start()] if stop else after[:1800]
    return re.sub(r"\s+", " ", chunk).strip()

# -------------------------
# 2) Citation data models
# -------------------------
class Citation(BaseModel):
    raw: str
    authors: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None

class SearchResult(BaseModel):
    match: str  # "exact" | "close" | "none"
    url: Optional[str] = None
    matched_title: Optional[str] = None
    note: Optional[str] = None  # optional diagnostic

# -------------------------
# 3) Extraction agent setup
# -------------------------
EXTRACT_INSTRUCTIONS = """\
You extract individual reference entries (citations) from the provided References/Bibliography text of an academic paper.

Return ONLY JSON in this exact schema (a JSON array of objects). Do not add commentary:

[
  {
    "raw": "<the full text of the citation as given>",
    "authors": "<authors string or null>",
    "year": "<4-digit year or null>",
    "title": "<title or null>",
    "journal": "<journal or null>",
    "volume": "<volume or null>",
    "issue": "<issue or null>",
    "pages": "<pages or null>",
    "doi": "<doi if present, else null>"
  },
  ...
]

Extraction guidelines:
- Separate each citation into its own object.
- Preserve the full 'raw' string (minus line-break artifacts).
- Parse DOI if present (pattern usually starts with '10.' and includes a slash).
- Titles: if unclear, best-effort. It's okay to leave fields null when uncertain.
- Handle numeric lists (e.g., [1], 1., 2.) and author-year styles.
- Limit to 300 citations maximum.
"""

extract_agent = Agent(
    name="Citation extraction agent",
    instructions=EXTRACT_INSTRUCTIONS,
    tools=[],  # no web tools needed
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="none"),
)

# ----------------------------------------
# 4) Helpers: read file, locate references
# ----------------------------------------
def read_pdf_text(path: str) -> str:
    text_chunks = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_chunks.append(page.extract_text() or "")
            except Exception:
                text_chunks.append("")
    return "\n".join(text_chunks)

def read_docx_text(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_file_text(filepath: str) -> str:
    lower = filepath.lower()
    if lower.endswith(".pdf"):
        return read_pdf_text(filepath)
    elif lower.endswith(".docx"):
        return read_docx_text(filepath)
    elif lower.endswith(".doc"):
        # python-docx does not support legacy .doc; you can convert externally to .docx or PDF first.
        raise ValueError("Legacy .doc format not supported. Please upload .docx or PDF.")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

REF_HEADINGS = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.IGNORECASE | re.MULTILINE)

def extract_references_section(full_text: str) -> str:
    """
    Heuristic: find a 'References'/'Bibliography'/'Works Cited' heading and return from there to the end.
    If not found, fallback to the last ~30% of the document (often where refs appear).
    """
    match = REF_HEADINGS.search(full_text)
    if match:
        start = match.start()
        return full_text[start:]
    # fallback: last 30%
    n = len(full_text)
    return full_text[int(n * 0.7):]

def _escape_md(s: str) -> str:
    """Escape Markdown table-breaking characters."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("|", r"\|").replace("\n", " ").replace("\r", " ")
    s = s.replace("[", r"\[").replace("]", r"\]")
    return s

def _rows_to_markdown_table(rows, with_links: bool) -> str:
    """
    Build a Markdown table. If with_links=True, include a clickable 'Link' column.
    Columns: Title, Authors, Year, Journal, DOI, Matched Title, Link, Note
    """
    if not rows:
        return "_No items in this bucket._"

    headers = ["Title", "Authors", "Year", "Journal", "DOI", "Matched Title"]
    if with_links:
        headers += ["Link"]
    headers += ["Note"]

    lines = []
    # header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        title = _escape_md(r.get("Title") or "")
        authors = _escape_md(r.get("Authors") or "")
        year = _escape_md(r.get("Year") or "")
        journal = _escape_md(r.get("Journal") or "")
        doi = _escape_md(r.get("DOI") or "")
        matched_title = _escape_md(r.get("Matched Title") or "")
        note = _escape_md(r.get("Note") or "")

        row_cells = [title, authors, year, journal, doi, matched_title]

        if with_links:
            url = r.get("URL") or ""
            link_cell = f"[Open]({url})" if url else ""
            row_cells.append(link_cell)

        row_cells.append(note)
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)

def build_report_md(meta: Dict[str, str], exact_rows, close_rows, none_rows) -> str:
    def md_link(u: str) -> str:
        return f"[Open]({u})" if u else ""

    def section_for(name: str, rows, linky: bool) -> str:
        if not rows:
            return f"## {name}\n\n_No items._\n"
        lines = [f"## {name}\n"]
        for r in rows:
            raw = (r.get("Raw") or "").strip()
            url = r.get("URL") if linky else None
            link = md_link(url)
            if link:
                lines.append(f"- {raw}\n  - {link}")
            else:
                lines.append(f"- {raw}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    title = meta.get("title") or "Unknown"
    authors = meta.get("authors") or "Unknown"
    doi = meta.get("doi") or "Unknown"
    abstract = meta.get("abstract") or "_No abstract detected._"

    header = [
        "# Citation Verification Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Paper metadata",
        f"**Title:** {title}",
        f"**Authors:** {authors}",
        f"**DOI:** {doi}",
        "",
        "## Abstract",
        abstract,
        "",
    ]

    body = [
        section_for("Exact matches", exact_rows, linky=True),
        section_for("Close matches", close_rows, linky=True),
        section_for("Not found",    none_rows,  linky=False),
    ]

    return "\n".join(header + body)

def save_report_md(text: str, suffix=".md") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(f.name, "w", encoding="utf-8") as out:
        out.write(text)
    return f.name

def reset_for_new_search(pct: float = 0.0, text: str = "Starting…"):
    """
    Return a 10-tuple of gr.update(...) matching your btn.click outputs:
      0 exact_md, 1 close_md, 2 none_md,
      3 summary_out,
      4 exact_tab, 5 close_tab, 6 none_tab,
      7 status_md,
      8 download_btn,
      9 progress_html
    Clears the 3 tabs, summary, resets tab labels to (0), clears prior download,
    and shows a fresh progress bar + status.
    """
    return (
        gr.update(value=""),  # exact_md
        gr.update(value=""),  # close_md
        gr.update(value=""),  # none_md
        gr.update(value=""),  # summary_out
        gr.update(label="✅ Exact matches (0)"),
        gr.update(label="⚠️ Close matches (0)"),
        gr.update(label="❌ Not found (0)"),
        gr.update(value=f"**Status:** {text}"),
        gr.update(value=None),                        # clear previous report
        gr.update(value=render_progress(pct, text)),  # reset progress bar
    )

PDF_MIME = "application/pdf"

def _user_agent() -> dict:
    # realistic desktop UA + sane defaults
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def _is_pdf_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    return (ctype == PDF_MIME) or resp.url.lower().endswith(".pdf")

def _download_to_tempfile(resp: requests.Response, suffix: str = ".pdf") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as out:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                out.write(chunk)
    return tmp.name

def _robust_request(method: str, url: str, *, referer: str | None = None,
                    stream: bool = False, timeout: int = 25,
                    session: requests.Session | None = None,
                    headers_extra: dict | None = None,
                    accept: str | None = None) -> requests.Response:
    headers = dict(_user_agent())
    if referer:
        headers["Referer"] = referer
    if accept:
        headers["Accept"] = accept
    if headers_extra:
        headers.update(headers_extra)

    s = session or requests.Session()
    resp = s.request(method, url, headers=headers, allow_redirects=True, stream=stream, timeout=timeout)

    # On 403, try cloudscraper if available
    if resp.status_code == 403 and HAS_CLOUDSCRAPER:
        scr = cloudscraper.create_scraper(browser={'custom': 'chrome'})
        resp = scr.request(method, url, headers=headers, allow_redirects=True, stream=stream, timeout=timeout)

    return resp

def _extract_pdf_from_html(soup: BeautifulSoup) -> str | None:
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        return meta.get("content")
    for tag in soup.find_all(["a", "link"]):
        href = (tag.get("href") or "").strip()
        if href.lower().endswith(".pdf"):
            return href
    return None

def _absolutize(base_url: str, maybe_relative: str) -> str:
    from urllib.parse import urljoin
    return urljoin(base_url, maybe_relative)

# ---------- Crossref + Unpaywall fallbacks ----------
def _crossref_pdf_url(doi_core: str, timeout: int = 20) -> str | None:
    api = f"https://api.crossref.org/works/{urllib.parse.quote(doi_core, safe='')}"
    r = requests.get(api, headers=_user_agent(), timeout=timeout)
    if r.ok:
        msg = r.json().get("message", {})
        for link in msg.get("link", []):
            if (link.get("content-type") or "").lower() == PDF_MIME and link.get("URL"):
                return link["URL"]
    return None

def _unpaywall_pdf_url(doi_core: str, timeout: int = 20) -> str | None:
    # Requires an email; set env var UNPAYWALL_EMAIL="you@example.com"
    email = os.getenv("UNPAYWALL_EMAIL")
    if not email:
        return None
    api = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi_core, safe='')}?email={urllib.parse.quote(email)}"
    r = requests.get(api, headers=_user_agent(), timeout=timeout)
    if r.ok:
        msg = r.json()
        oa = msg.get("best_oa_location") or {}
        return oa.get("url_for_pdf") or oa.get("url")
    return None

# ---------- PNAS-specific heuristic ----------
def _pnas_pdf_candidates(landing_or_doi_url: str) -> list[str]:
    # e.g. https://www.pnas.org/doi/10.1073/pnas.2507345122
    # PDFs commonly:
    #   /doi/pdf/10.1073/pnas.XXXX
    #   /doi/pdfdirect/10.1073/pnas.XXXX   (skips HTML frame/download page)
    if "www.pnas.org/doi/" in landing_or_doi_url:
        base = landing_or_doi_url
        return [
            base.replace("/doi/", "/doi/pdf/"),
            base.replace("/doi/", "/doi/pdfdirect/"),
        ]
    return []

def read_pdf_text_from_bytes_url(url: str, referer: str | None = None, timeout: int = 25) -> str:
    resp = _robust_request("GET", url, referer=referer, stream=True, timeout=timeout,
                           accept="application/pdf")
    resp.raise_for_status()
    if not _is_pdf_response(resp):
        raise ValueError(f"URL did not return a PDF: {url}")
    tmp_path = _download_to_tempfile(resp, suffix=".pdf")
    try:
        return read_pdf_text(tmp_path)
    finally:
        try:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

# ---------- DOI normalization + resolution ----------
def normalize_doi(doi_str: str) -> str:
    s = (doi_str or "").strip()
    s = re.sub(r'^(doi:)\s*', '', s, flags=re.I)
    s = re.sub(r'^(https?://(dx\.)?doi\.org/)', '', s, flags=re.I)
    if not s:
        raise ValueError("DOI is empty.")
    if not re.match(r'^10\.\d{4,9}/\S+$', s, flags=re.I):
        raise ValueError("That does not look like a DOI (e.g., 10.1038/...).")
    return s

def resolve_doi_to_landing(doi_core: str, timeout: int = 25) -> str:
    doi_url = f"https://doi.org/{doi_core}"
    # HEAD first to follow redirects cheaply
    head = _robust_request("HEAD", doi_url, accept="text/html,application/xhtml+xml", timeout=timeout)
    head.raise_for_status()
    return head.url

def _crossref_pdf_url(doi_core: str, timeout: int = 20) -> str | None:
    api = f"https://api.crossref.org/works/{urllib.parse.quote(doi_core, safe='')}"
    r = requests.get(api, headers=_user_agent(), timeout=timeout)
    if r.ok:
        msg = r.json().get("message", {})
        for link in msg.get("link", []):
            if (link.get("content-type") or "").lower() == PDF_MIME and link.get("URL"):
                return link["URL"]
    return None

def _biorxiv_pdf_from_landing(landing_url: str) -> str | None:
    # typical landing: https://www.biorxiv.org/content/10.1101/515643v4
    # PDF:             https://www.biorxiv.org/content/10.1101/515643v4.full.pdf
    try:
        if "biorxiv.org" in landing_url and "/content/" in landing_url:
            u = landing_url
            # strip known endings to get the base
            for tail in (".full", ".full.pdf", ".abstract", ".pdf"):
                if u.endswith(tail):
                    u = u[: -len(tail)]
            if not u.endswith(".full.pdf"):
                u = u + ".full.pdf"
            return u
    except Exception:
        pass
    return None

def load_url_text(url: str, timeout: int = 25) -> str:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    s = requests.Session()

    # 1) Try GET landing
    resp = _robust_request("GET", url, stream=False, timeout=timeout, session=s)
    resp.raise_for_status()

    # 1a) Direct PDF?
    if _is_pdf_response(resp):
        tmp_path = _download_to_tempfile(resp, suffix=".pdf")
        try:
            return read_pdf_text(tmp_path)
        finally:
            try: pathlib.Path(tmp_path).unlink(missing_ok=True)
            except Exception: pass

    # 2) If HTML, try to discover a PDF link in-page
    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_link = _extract_pdf_from_html(soup)
    if pdf_link:
        pdf_url = _absolutize(resp.url, pdf_link)
        return read_pdf_text_from_bytes_url(pdf_url, referer=resp.url, timeout=timeout)

    # 3) PNAS heuristic: derive pdf/pdfdirect variants
    if "www.pnas.org/doi/" in resp.url:
        for cand in _pnas_pdf_candidates(resp.url):
            try:
                return read_pdf_text_from_bytes_url(cand, referer=resp.url, timeout=timeout)
            except Exception:
                continue  # try next candidate

    # 4) Fallback: extract readable HTML text
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    chunks = []
    for sel in ["article", "main", "div", "section", "p", "h1", "h2", "h3"]:
        for el in soup.select(sel):
            t = " ".join(el.get_text(" ", strip=True).split())
            if t:
                chunks.append(t)
    if not chunks:
        chunks.append(" ".join(soup.get_text(" ", strip=True).split()))
    return "\n".join(chunks)

def load_doi_text(doi: str, timeout: int = 25) -> str:
    core = normalize_doi(doi)
    referer = f"https://doi.org/{core}"

    # 1) Resolve to landing
    try:
        landing = resolve_doi_to_landing(core, timeout=timeout)
    except Exception:
        # Crossref/Unpaywall direct PDF fallbacks
        pdf = _crossref_pdf_url(core, timeout=timeout) or _unpaywall_pdf_url(core, timeout=timeout)
        if pdf:
            return read_pdf_text_from_bytes_url(pdf, referer=referer, timeout=timeout)
        raise

    # 2) Try loading landing URL (this applies PNAS heuristic internally)
    try:
        return load_url_text(landing, timeout=timeout)
    except requests.HTTPError as http_err:
        if http_err.response is not None and http_err.response.status_code == 403:
            # Publisher blocked us; try direct PDF fallbacks
            pdf = _crossref_pdf_url(core, timeout=timeout) or _unpaywall_pdf_url(core, timeout=timeout)
            if pdf:
                return read_pdf_text_from_bytes_url(pdf, referer=referer, timeout=timeout)
            # As last resort for PNAS, try constructed PDF links
            if "pnas.org" in landing:
                for cand in _pnas_pdf_candidates(landing):
                    try:
                        return read_pdf_text_from_bytes_url(cand, referer=referer, timeout=timeout)
                    except Exception:
                        pass
        # Re-raise if all fallbacks failed
        raise

# --- tiny sanitizer for Crossref abstracts (often JATS/XML) ---
def _strip_tags(s: str) -> str:
    if not s:
        return ""
    # quick-n-dirty tag strip; keeps text
    return re.sub(r"<[^>]+>", "", s)

# ---------- Crossref: metadata + references ----------
def crossref_metadata_and_refs(doi_core: str, timeout: int = 25):
    """Returns (meta_dict, references_list) or (None, None) if not found."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi_core, safe='')}"
    r = requests.get(url, headers=_user_agent(), timeout=timeout)
    if not r.ok:
        return None, None
    msg = r.json().get("message", {})

    # metadata
    title = " ".join(msg.get("title") or []) if msg.get("title") else ""
    # authors as "Last, First" joined with semicolons
    authors = []
    for a in msg.get("author", []) or []:
        given = a.get("given", "")
        family = a.get("family", "")
        nm = (f"{family}, {given}".strip(", ").strip()) or a.get("name", "")
        if nm:
            authors.append(nm)
    authors_str = "; ".join(authors)
    doi = (msg.get("DOI") or "").lower()
    abstract = _strip_tags(msg.get("abstract") or "")

    meta = {"title": title, "authors": authors_str, "doi": doi, "abstract": abstract}

    # references (array of dicts with varying keys)
    refs = msg.get("reference", []) or []
    return meta, refs

# ---------- OpenAlex: fall back if Crossref lacks references ----------
def openalex_refs(doi_core: str, timeout: int = 25):
    """Return a list of reference dicts with at least 'raw' and maybe 'doi'/'title'."""
    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi_core, safe='')}"
    r = requests.get(url, headers=_user_agent(), timeout=timeout)
    if not r.ok:
        return []
    data = r.json()
    # We may only get IDs of referenced works; resolve a small subset or make raw strings.
    refs_out = []
    for wid in data.get("referenced_works", []) or []:
        # Resolve each referenced work to get title + doi (avoid hammering: cap at e.g. 400)
        if len(refs_out) >= 400:
            break
        rr = requests.get(f"https://api.openalex.org/{wid}", headers=_user_agent(), timeout=timeout)
        if not rr.ok:
            continue
        w = rr.json()
        title = (w.get("title") or "").strip()
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        raw = title
        refs_out.append({"unstructured": raw, "DOI": doi, "article-title": title})
    return refs_out

# ---------- Semantic Scholar: another fallback for references ----------
def semanticscholar_refs(doi_core: str, timeout: int = 25):
    """Return a list of reference dicts with 'unstructured' and 'DOI' when available."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi_core, safe='')}" \
          f"?fields=title,abstract,authors,externalIds,references.title,references.externalIds"
    r = requests.get(url, headers=_user_agent(), timeout=timeout)
    if not r.ok:
        return []
    data = r.json()
    refs_out = []
    for ref in data.get("references", []) or []:
        title = (ref.get("title") or "").strip()
        ext = ref.get("externalIds") or {}
        doi = (ext.get("DOI") or "").strip()
        raw = title
        refs_out.append({"unstructured": raw, "DOI": doi, "article-title": title})
    return refs_out

# ---------- Lift API ref dicts into your Citation objects ----------
def refs_to_citations(refs: list) -> list[Citation]:
    out = []
    for r in refs:
        raw = r.get("unstructured") or r.get("article-title") or r.get("journal-title") or ""
        title = r.get("article-title") or None
        doi = (r.get("DOI") or "") or None
        # Some Crossref refs store 'year', 'volume', 'issue', 'first-page' etc.
        year = (str(r.get("year")).strip() if r.get("year") else None)
        journal = r.get("journal-title")
        pages = r.get("first-page")
        out.append(Citation(
            raw=raw.strip() or (title or ""),
            authors=None,
            year=year,
            title=title,
            journal=journal,
            volume=r.get("volume"),
            issue=r.get("issue"),
            pages=pages,
            doi=doi
        ))
    return out

# Bundle: try Crossref → OpenAlex → Semantic Scholar
def doi_api_pipeline(doi_str: str, timeout: int = 25):
    """
    Returns (meta_dict, citations_list).
    meta_dict keys: title, authors, doi, abstract
    citations_list: List[Citation]
    """
    core = normalize_doi(doi_str)

    meta, refs = crossref_metadata_and_refs(core, timeout=timeout)
    # If Crossref missing refs, fall back to OpenAlex, then Semantic Scholar
    if not refs:
        refs = openalex_refs(core, timeout=timeout)
        if meta is None:
            meta = {"title": "", "authors": "", "doi": core, "abstract": ""}
    if not refs:
        refs = semanticscholar_refs(core, timeout=timeout)

    citations = refs_to_citations(refs) if refs else []
    # Ensure meta dict exists
    if not meta:
        meta = {"title": "", "authors": "", "doi": core, "abstract": ""}

    return meta, citations

# -------------------------
# 5) Agent runners (robust to different Agent APIs)
# -------------------------
async def run_agent(agent: Agent, prompt: str) -> str:
    """
    Run an Agents SDK Agent with a single input string and return its text output.
    Supports both Runner.run(agent, ...) and Runner.run(starting_agent=..., ...).
    """
    trace_id = gen_trace_id()
    try:
        # Preferred signature (current docs/cookbooks)
        result = await Runner.run(
            agent,
            input=prompt,
            run_config=RunConfig(trace_id=trace_id),
        )
    except TypeError:
        # Fallback for builds that require the 'starting_agent' kw
        result = await Runner.run(
            starting_agent=agent,
            input=prompt,
            run_config=RunConfig(trace_id=trace_id),
        )

    # Normalize the return to text
    out = getattr(result, "final_output", None)
    if out is None:
        # Some builds may return a simple object/string
        out = extract_agent_text(result)
    return out or ""

def extract_agent_text(obj: Any) -> str:
    """
    Normalize various possible return structures into a string.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    # Common patterns: dict with 'content' or object with .content
    for key in ("content", "text", "output"):
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    if hasattr(obj, "content"):
        return getattr(obj, "content")
    # last resort: stringify
    return str(obj)

def render_progress(pct: float, text: str) -> str:
    """
    Render a theme-aware horizontal progress bar with text next to it.
    pct: 0..1
    """
    pct = max(0, min(100, int(round(pct * 100))))
    return f"""
    <div class="cv-prog-wrap" style="display:flex;align-items:center;gap:10px;width:100%;">
      <div aria-label="Progress" class="cv-prog-bar"
           style="flex:1;height:12px;border-radius:6px;overflow:hidden;
                  background: var(--block-border-color, rgba(127,127,127,.2));">
        <div style="height:100%;width:{pct}%;
                    background: linear-gradient(90deg,
                      var(--color-accent, #3b82f6),
                      var(--color-accent-foreground, #60a5fa));
                    transition: width .15s;"></div>
      </div>
      <div class="cv-prog-text" style="min-width:240px;font-size:.95em;opacity:.92;">
        {text} ({pct}%)
      </div>
    </div>
    """

# -------------------------
# 6) Parsing utilities (fallbacks)
# -------------------------
DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

def simple_heurstic_split(ref_text: str) -> List[str]:
    """
    Fallback splitter if the extraction agent fails: split references by common numbering formats
    and by blank lines, then clean chunks.
    """
    # Normalize line breaks
    t = re.sub(r"[ \t]+", " ", ref_text)
    # Split on patterns like [1], 1. 2) etc., and on double newlines
    parts = re.split(r"(?:\n\s*(?:\[\d+\]|\d{1,3}[.)]))|\n{2,}", t)
    # re.split creates None/empty entries; clean + rejoin short fragments
    candidates = [p.strip() for p in parts if p and p.strip()]
    # Merge very short lines with the next line
    merged = []
    buf = ""
    for c in candidates:
        if len(c) < 60:
            buf = (buf + " " + c).strip()
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(c)
    if buf:
        merged.append(buf)
    # De-dup extremely short/noisy
    return [m for m in merged if len(m) > 40]

def heuristic_parse_citation(raw: str) -> Citation:
    doi = None
    m = DOI_REGEX.search(raw)
    if m:
        doi = m.group(0)
    # Year: (2021) or 2021.
    y = re.search(r"(19|20)\d{2}", raw)
    year = y.group(0) if y else None
    return Citation(raw=raw, doi=doi, year=year)

# -------------------------
# 7) Extraction: ask the extraction agent, fallback to heuristics
# -------------------------
async def extract_citations(ref_text: str) -> List[Citation]:
    prompt = (
        "Extract citations from the following References/Bibliography text. "
        "Return ONLY the JSON array, no commentary.\n\n"
        "=== TEXT START ===\n"
        f"{ref_text}\n"
        "=== TEXT END ==="
    )
    txt = await run_agent(extract_agent, prompt)
    # Try to parse JSON
    citations: List[Citation] = []
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            for item in data[:300]:
                try:
                    citations.append(Citation(**{
                        "raw": item.get("raw") or "",
                        "authors": item.get("authors"),
                        "year": item.get("year"),
                        "title": item.get("title"),
                        "journal": item.get("journal"),
                        "volume": item.get("volume"),
                        "issue": item.get("issue"),
                        "pages": item.get("pages"),
                        "doi": item.get("doi"),
                    }))
                except Exception:
                    # Skip broken rows
                    continue
    except Exception:
        # Fallback: simple splitter + light parsing
        chunks = simple_heurstic_split(ref_text)
        citations = [heuristic_parse_citation(c) for c in chunks]
    # Ensure unique-ish set (dedupe by raw)
    seen = set()
    uniq = []
    for c in citations:
        key = re.sub(r"\s+", " ", c.raw.strip()).lower()
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq

# -------------------------
# 8) Verification via search agent
# -------------------------
def build_search_prompt(c: Citation) -> str:
    """
    Tight prompt to force JSON result from the search agent.
    """
    basis = f"DOI: {c.doi}" if c.doi else f"TITLE: {c.title or '[title missing]'}"
    additional = []
    if c.authors: additional.append(f"AUTHORS: {c.authors}")
    if c.year:    additional.append(f"YEAR: {c.year}")
    if c.journal: additional.append(f"JOURNAL: {c.journal}")
    extra = "\n".join(additional)

    return (
        "You are verifying a single citation. Search for the DOI if available; otherwise, search for an exact title match.\n"
        "Return ONLY a single JSON object with these keys:\n"
        '{\n'
        '  "match": "exact" | "close" | "none",\n'
        '  "url": "<best url if found, else null>",\n'
        '  "matched_title": "<the title you matched or null>",\n'
        '  "note": "<short reason or null>"\n'
        '}\n'
        "NO commentary outside JSON.\n\n"
        f"CITATION BASIS:\n{basis}\n{extra}\n\n"
        f"RAW:\n{c.raw}\n"
    )

async def verify_one(c: Citation, sem: asyncio.Semaphore) -> Tuple[Citation, SearchResult]:
    prompt = build_search_prompt(c)
    async with sem:
        out = await run_agent(search_agent, prompt)
    # Parse the returned JSON object
    try:
        data = json.loads(out.strip())
        sr = SearchResult(
            match=(data.get("match") or "none").lower(),
            url=data.get("url"),
            matched_title=data.get("matched_title"),
            note=data.get("note")
        )
    except Exception:
        # If we cannot parse JSON, classify as unknown/none with note
        sr = SearchResult(match="none", url=None, matched_title=None, note="Unparseable search response")
    return c, sr

async def verify_all(citations: List[Citation], progress_cb=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Verify all citations concurrently with a small semaphore (avoid rate limits).
    Reports progress as: 'Searching citations… (k/N)'.
    Returns dict with buckets: exact, close, none.
    """
    sem = asyncio.Semaphore(5)
    tasks = [verify_one(c, sem) for c in citations]
    total = len(tasks) or 1
    done = 0

    if progress_cb:
        progress_cb(0.0, f"Searching citations… (0/{total})")

    results: List[Tuple[Citation, SearchResult]] = []
    for fut in asyncio.as_completed(tasks):
        c, sr = await fut
        results.append((c, sr))
        done += 1
        if progress_cb:
            progress_cb(done / total, f"Searching citations… ({done}/{total})")

    # Bucketize
    buckets = {"exact": [], "close": [], "none": []}
    for c, sr in results:
        row = {
            "Title": c.title,
            "Authors": c.authors,
            "Year": c.year,
            "Journal": c.journal,
            "DOI": c.doi,
            "Matched Title": sr.matched_title,
            "URL": sr.url,
            "Note": sr.note,
            "Raw": c.raw,
        }
        if sr.match == "exact":
            buckets["exact"].append(row)
        elif sr.match == "close":
            buckets["close"].append(row)
        else:
            buckets["none"].append(row)

    return buckets


# -------------------------
# 9) Gradio UI
# -------------------------
def to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Title","Authors","Year","Journal","DOI","Matched Title","URL","Note","Raw"])
    return pd.DataFrame(rows)

# --- Progress bar renderer (if you don't already have it) ---
def render_progress(pct: float, text: str) -> str:
    pct = max(0, min(100, int(round(pct * 100))))
    return f"""
    <div class="cv-prog-wrap" style="display:flex;align-items:center;gap:10px;width:100%;">
      <div aria-label="Progress" class="cv-prog-bar"
           style="flex:1;height:12px;border-radius:6px;overflow:hidden;
                  background: var(--block-border-color, rgba(127,127,127,.2));">
        <div style="height:100%;width:{pct}%;
                    background: linear-gradient(90deg,
                      var(--color-accent, #3b82f6),
                      var(--color-accent-foreground, #60a5fa));
                    transition: width .15s;"></div>
      </div>
      <div class="cv-prog-text" style="min-width:240px;font-size:.95em;opacity:.92;">
        {text} ({pct}%)
      </div>
    </div>
    """

# --- Tuple builder used by your async generator to stream progress ---
def progress_update(pct: float, text: str):
    """
    Returns the 10 updates in the SAME ORDER as your btn.click outputs:
      0 exact_md, 1 close_md, 2 none_md,
      3 summary_out,
      4 exact_tab, 5 close_tab, 6 none_tab,
      7 status_md,
      8 download_btn,
      9 progress_html
    Only status + progress bar are changed; others are left unchanged.
    """
    return (
        gr.update(), gr.update(), gr.update(),        # exact_md, close_md, none_md (unchanged)
        gr.update(),                                   # summary_out (unchanged)
        gr.update(), gr.update(), gr.update(),         # tab labels (unchanged)
        gr.update(value=f"**Status:** {text}"),        # status_md
        gr.update(),                                   # download_btn (unchanged)
        gr.update(value=render_progress(pct, text)),   # progress_html
    )

# --- Optional: clear UI at start of a new run (if you use it at the top) ---
def reset_for_new_search(pct: float = 0.0, text: str = "Starting…"):
    return (
        gr.update(value=""),  # exact_md
        gr.update(value=""),  # close_md
        gr.update(value=""),  # none_md
        gr.update(value=""),  # summary_out
        gr.update(label="✅ Exact matches (0)"),
        gr.update(label="⚠️ Close matches (0)"),
        gr.update(label="❌ Not found (0)"),
        gr.update(value=f"**Status:** {text}"),
        gr.update(value=None),                        # clear previous report
        gr.update(value=render_progress(pct, text)),  # reset progress bar
    )

async def handle_file(file: gr.File, url: str, doi: str):

    def empty_results(msg: str):
        empty_md = "_No items in this bucket._"
        return (
            gr.update(value=empty_md), gr.update(value=empty_md), gr.update(value=empty_md),
            msg,
            gr.update(label="✅ Exact matches (0)"),
            gr.update(label="⚠️ Close matches (0)"),
            gr.update(label="❌ Not found (0)"),
            gr.update(value=f"**Status:** {msg}"),
            gr.update(value=None),
            gr.update(value=render_progress(0.0, msg)),
        )

    # --- init everything up front so later checks don't crash ---
    used_api_refs: bool = False
    meta: dict | None = None
    citations: list[Citation] | None = None
    full_text: str | None = None

    # Clear UI & start progress
    yield reset_for_new_search(0.02, "Starting…")

    src_url = (url or "").strip()
    src_doi = (doi or "").strip()

    # --------- choose source: file → url → doi ----------
    if file is not None:
        yield progress_update(0.04, "Reading file…")
        try:
            full_text = load_file_text(file.name)
        except Exception as e:
            yield empty_results(f"Error reading file: {e}")
            return

    elif src_url:
        yield progress_update(0.04, "Fetching URL…")
        try:
            full_text = load_url_text(src_url)
        except Exception as e:
            yield empty_results(f"Error fetching URL: {e}")
            return

    elif src_doi:
        yield progress_update(0.04, "Resolving DOI…")
        try:
            # try publisher route (PDF/HTML); may 403
            full_text = load_doi_text(src_doi)
        except Exception as e:
            # use API-based references to avoid 403 walls
            yield progress_update(0.08, "Publisher blocked. Fetching references via Crossref/OpenAlex…")
            meta_api, citations_api = doi_api_pipeline(src_doi)
            if not citations_api:
                yield empty_results(f"Error resolving DOI (and no references via APIs): {e}")
                return
            used_api_refs = True
            meta = meta_api
            citations = citations_api
    else:
        yield empty_results("Please upload a PDF/DOCX, enter a URL, or enter a DOI.")
        return

    # --------- metadata + references/citations extraction ----------
    if not used_api_refs:
        # metadata from document text
        yield progress_update(0.10, "Extracting paper metadata…")
        meta = await extract_paper_metadata(full_text or "")

        # locate References/Bibliography in the text
        yield progress_update(0.18, "Locating References/Bibliography…")
        refs = extract_references_section(full_text or "")
        if not refs or len(refs.strip()) < 50:
            yield empty_results("Could not locate References. For tough publishers, try entering a DOI (uses Crossref/OpenAlex).")
            return

        # extract structured citations from the section
        yield progress_update(0.30, "Extracting citations…")
        citations = await extract_citations(refs)
        if not citations:
            yield empty_results("No citations detected. Try DOI input to use Crossref/OpenAlex references.")
            return

    # safety: ensure both are set before continuing
    meta = meta or {"title": "", "authors": "", "doi": (src_doi or ""), "abstract": ""}
    citations = citations or []
    if not citations:
        yield empty_results("No citations to verify.")
        return

    # announce count & move on with your existing verification steps
    N = len(citations)
    yield progress_update(0.34, f"Found {N} citations.")
    yield progress_update(0.35, f"Searching citations… (0/{N})")

    # 5) Verify citations (unchanged, streaming progress)
    sem = asyncio.Semaphore(5)
    tasks = [verify_one(c, sem) for c in citations]
    done = 0
    buckets = {"exact": [], "close": [], "none": []}

    async for c, sr in _as_completed_stream(tasks):
        row = {
            "Title": c.title, "Authors": c.authors, "Year": c.year, "Journal": c.journal,
            "DOI": c.doi, "Matched Title": sr.matched_title, "URL": sr.url, "Note": sr.note, "Raw": c.raw
        }
        bucket = "exact" if sr.match == "exact" else "close" if sr.match == "close" else "none"
        buckets[bucket].append(row)
        done += 1
        prog = 0.35 + 0.55 * (done / N)
        yield progress_update(prog, f"Searching citations… ({done}/{N})")

    # 6) Render results (unchanged)
    yield progress_update(0.92, "Rendering results…")
    exact_rows, close_rows, none_rows = buckets["exact"], buckets["close"], buckets["none"]
    exact_md_str = _rows_to_markdown_table(exact_rows, with_links=True)
    close_md_str = _rows_to_markdown_table(close_rows, with_links=True)
    none_md_str  = _rows_to_markdown_table(none_rows,  with_links=False)

    summary = (
        f"**Total citations:** {N}  |  "
        f"**Exact:** {len(exact_rows)}  |  "
        f"**Close:** {len(close_rows)}  |  "
        f"**No match:** {len(none_rows)}"
    )

    # 7) Build report (unchanged)
    yield progress_update(0.96, "Building report…")
    report_md = build_report_md(meta, exact_rows, close_rows, none_rows)
    report_path = save_report_md(report_md, suffix=".md")

    # 8) Final update (unchanged)
    yield (
        gr.update(value=exact_md_str),
        gr.update(value=close_md_str),
        gr.update(value=none_md_str),
        summary,
        gr.update(label=f"✅ Exact matches ({len(exact_rows)})"),
        gr.update(label=f"⚠️ Close matches ({len(close_rows)})"),
        gr.update(label=f"❌ Not found ({len(none_rows)})"),
        gr.update(value="**Status:** Done."),
        gr.update(value=report_path),
        gr.update(value=render_progress(1.0, "Done."))
    )

# Helper to stream results as each task completes
async def _as_completed_stream(tasks):
    for fut in asyncio.as_completed(tasks):
        yield await fut

with gr.Blocks(title="Citation Verifier") as demo:

    gr.HTML(
    """
<div class="cv-hero" role="region" aria-label="About Citation Verifier">
  <div class="cv-emoji">🔎</div>
  <div class="cv-text">
    <h1>Citation Verifier</h1>
    <p>
      Upload a scientific paper (PDF or DOCX) or paste a paper URL (PDF, DOI, arXiv, publisher page). 
      The app extracts your References/Bibliography, searches each citation (DOI or exact title), 
      and sorts results into Exact, Close, or Not found.
    </p>
    <ol>
      <li><strong>Upload</strong> your PDF/DOCX.</li>
      <li><strong>Click</strong> <em>Verify Citations</em>.</li>
      <li><strong>Watch progress</strong> (extraction count + per-citation search).</li>
      <li><strong>Review</strong> results in the tabs; links are clickable.</li>
      <li><strong>Download</strong> a Markdown report.</li>
    </ol>
    <p class="cv-note">Tip: Works best with text-based PDFs; OCR scans first if needed.</p>
  </div>
</div>

<style>
  /* Base / shared styles */
  .cv-hero {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 16px 18px;
    border: 1px solid;
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,.06);
    color: inherit;                /* inherit text color from theme */
    backdrop-filter: saturate(120%) blur(2px);
  }
  .cv-emoji { font-size: 40px; line-height: 1; user-select: none; }
  .cv-text h1 { margin: 0 0 6px 0; font-size: 20px; }
  .cv-text p, .cv-text li { margin: 0 0 8px 0; color: inherit; opacity: 0.92; }
  .cv-text ol { margin: 0 0 6px 18px; padding: 0; }
  .cv-note { margin-top: 6px; opacity: 0.85; }

  /* Light mode */
  @media (prefers-color-scheme: light) {
    .cv-hero {
      background: linear-gradient(180deg, #f8fafc, #eef2f7);
      border-color: rgba(0,0,0,.08);
    }
  }

  /* Dark mode */
  @media (prefers-color-scheme: dark) {
    .cv-hero {
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
      border-color: rgba(255,255,255,.16);
      box-shadow: 0 1px 2px rgba(0,0,0,.4);
    }
    .cv-text p, .cv-text li, .cv-note { opacity: 0.88; }
  }

  @media (max-width: 640px) {
    .cv-hero { flex-direction: column; }
    .cv-emoji { font-size: 34px; }
  }
</style>
"""
    )

    with gr.Row():
        file_in = gr.File(label="Upload PDF or DOCX", file_types=[".pdf", ".docx"], type="filepath")
    with gr.Row():
        url_in = gr.Textbox(label="Or paste a paper URL (PDF, DOI, arXiv, publisher page)", placeholder="https://doi.org/10.1038/s41586-024-XXXXX")
    with gr.Row():
        doi_in = gr.Textbox(
            label="Or enter a DOI",
            placeholder="10.1038/s41586-024-XXXXX",
        )
    with gr.Row(equal_height=True):
        btn = gr.Button("Verify Citations", variant="primary", scale=2)
        clear_btn = gr.Button("Clear Input Fields", variant="secondary", scale=1)  # half width

    # Status line under the button
    status_md = gr.Markdown("_Idle._")

    # Custom progress bar + text
    progress_html = gr.HTML(render_progress(0.0, "Idle"))

    with gr.Tabs() as tabs:
        with gr.TabItem("✅ Exact matches (0)") as exact_tab:
            exact_md = gr.Markdown()
        with gr.TabItem("⚠️ Close matches (0)") as close_tab:
            close_md = gr.Markdown()
        with gr.TabItem("❌ Not found (0)") as none_tab:
            none_md = gr.Markdown()

    summary_out = gr.Markdown()

    # New: download report button (value set at runtime)
    download_btn = gr.DownloadButton(label="Download report (Markdown)", value=None)

    btn.click(
        handle_file,
        inputs=[file_in, url_in, doi_in],   # <-- now three inputs
        outputs=[
            exact_md, close_md, none_md,
            summary_out,
            exact_tab, close_tab, none_tab,
            status_md,
            download_btn,
            progress_html
        ],
    )

    def clear_all():
        # Clear inputs AND wipe results/labels/report/progress
        return (
            None, "", "",                               # inputs
            *reset_for_new_search(0.0, "Idle")          # returns the 10 outputs in your app
        )

    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[
            file_in, url_in, doi_in,                    # inputs
            exact_md, close_md, none_md,                # 0..2
            summary_out,                                # 3
            exact_tab, close_tab, none_tab,             # 4..6
            status_md,                                  # 7
            download_btn,                               # 8
            progress_html                               # 9
        ],
    )


if __name__ == "__main__":
    # Set share=True if you want a public link from Gradio
    demo.launch()
