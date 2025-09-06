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

# -------------------------
# 4) Helpers: read file, locate references
# -------------------------
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

async def handle_file(file: gr.File):
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

    def progress_update(pct: float, text: str):
        return (
            gr.update(), gr.update(), gr.update(),
            gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(value=f"**Status:** {text}"),
            gr.update(),
            gr.update(value=render_progress(pct, text)),
        )

    # 1) Reading file
    # yield progress_update(0.02, "Reading file…")
    yield reset_for_new_search(0.02, "Reading file…")
    try:
        full_text = load_file_text(file.name)
    except Exception as e:   # ✅ fixed: no extra ')'
        yield empty_results(f"Error reading file: {e}")  # ✅ yield, not return value
        return                                              # ✅ plain return to end stream

    # 2) Paper metadata
    yield progress_update(0.10, "Extracting paper metadata…")
    meta = await extract_paper_metadata(full_text)

    # 3) Locate references
    yield progress_update(0.18, "Locating References/Bibliography…")
    refs = extract_references_section(full_text)
    if not refs or len(refs.strip()) < 50:
        yield empty_results("Could not locate a References/Bibliography section. Try a cleaner PDF or a DOCX.")
        return

    # 4) Extract citations
    yield progress_update(0.30, "Extracting citations…")
    citations = await extract_citations(refs)
    if not citations:
        yield empty_results("No citations detected. If the PDF is scanned or image-based, OCR it first.")
        return
    N = len(citations)
    yield progress_update(0.34, f"Found {N} citations.")
    yield progress_update(0.35, f"Searching citations… (0/{N})")

    # 5) Verify citations concurrently, update progress for each completion
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

    # 6) Render results
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

    # 7) Build report
    yield progress_update(0.96, "Building report…")
    report_md = build_report_md(meta, exact_rows, close_rows, none_rows)
    report_path = save_report_md(report_md, suffix=".md")

    # 8) Final update (populate all outputs)
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
      Upload a scientific paper (PDF or DOCX). The app extracts your References/Bibliography,
      searches each citation (DOI or exact title), and sorts results into Exact, Close, or Not found.
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
        btn = gr.Button("Verify Citations", variant="primary")

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
        inputs=[file_in],
        outputs=[
            exact_md, close_md, none_md,     # 0..2
            summary_out,                     # 3
            exact_tab, close_tab, none_tab,  # 4..6
            status_md,                       # 7
            download_btn,                    # 8
            progress_html                    # 9  <-- NEW
        ],
    )

if __name__ == "__main__":
    # Set share=True if you want a public link from Gradio
    demo.launch()
