# 🔎 Citation Verifier

Verify the validity of **citations in scientific and academic papers**.  
Upload a PDF/DOCX or paste a **URL** or **DOI**, and the app:

- extracts the paper’s **References/Bibliography** (or uses public APIs for DOI-only),
- searches for each citation (preferring **DOI**, else **exact title**),
- buckets results into **✅ Exact**, **⚠️ Close**, and **❌ Not found**,  
- shows **clickable links** for Exact/Close matches,
- displays a **live progress bar** (extraction count + per-citation status),
- and lets you **download a Markdown report** (paper metadata + abstract + grouped citations).

---

## 1) What it does

- **Input sources**: file upload (**PDF/DOCX**), **URL** (publisher/arXiv/DOI page), or **DOI only** (e.g., `10.1038/...`).
- **Extraction**: finds the References/Bibliography section and structures citations (authors, title, year, DOI…).
- **Verification**:
  - uses the DOI when present; otherwise searches exact titles;
  - classifies results as **Exact**, **Close**, or **Not found**;
  - includes links for Exact/Close.
- **Reporting**: one-click **Markdown report** with paper metadata (title, authors, DOI, abstract) and all citations grouped by outcome.
- **Resilience**: DOI fetch falls back to **Crossref/OpenAlex/Semantic Scholar** metadata/references when publishers block bots (403).

---

## 2) How to use (with example)

1. **Open the app** (see the Hugging Face link below).
2. In the “About” box, review the quick steps.
3. Provide **one** of the following:
   - **Upload**: drag-and-drop a `.pdf` or `.docx`.
   - **URL**: paste a paper page (e.g., `https://www.pnas.org/doi/10.1073/pnas.2507345122`).
   - **DOI**: paste just the DOI (e.g., `10.1101/515643`).
4. Click **Verify Citations**.
5. Watch the **progress row**:
   - “Extracting citations…”, then “**Found N citations**”,
   - “**Searching citations… (k/N)**” as each verification completes.
6. Browse the three **tabs**:
   - **Exact matches (N)** – verified citations with a link.
   - **Close matches (M)** – likely matches; check manually.
   - **Not found (K)** – possible errors or missing items.
7. Click **Download report (Markdown)** to save a full summary.

**Example**  
Paste the DOI `10.1101/515643` (bioRxiv). Even if the publisher blocks direct PDF download, the app will pull references via Crossref/OpenAlex and proceed with verification.

---

## 3) Live app (Hugging Face)

👉 **[Open the Citation Verifier on Hugging Face](https://huggingface.co/spaces/your-org/citation-verifier)**  
*(Replace with your actual Space URL if different.)*

---

## 4) Install & run locally

### Minimum requirements
- **Python** ≥ 3.12  
- macOS / Linux / Windows
- Internet connection (for verification + DOI/metadata lookups)

### Quick start

```bash
# 1) (optional) create a fresh virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Upgrade pip
python -m pip install -U pip

# 3) Install dependencies
pip install -U gradio PyPDF2 python-docx pandas requests beautifulsoup4

# Optional (improves success rate against 403 walls):
pip install cloudscraper

# If you’re using the OpenAI Agents SDK for the search agent:
# (Make sure you’re using the correct 'agents' package)
pip install openai-agents

# 4) (optional) set environment variables
# For better open-access PDF discovery when publishers block bots:
# PowerShell (Windows):
setx UNPAYWALL_EMAIL "you@example.com"
# macOS/Linux (bash/zsh):
export UNPAYWALL_EMAIL="you@example.com"

# You will also need a .env file containing you OpenAI API key:
OPENAI_API_KEY=sk-your-key
UNPAYWALL_EMAIL=you@example.com

# 5) Run the app
python app.py
```

## Notes

- If you see TensorFlow/Gym errors, you likely installed the wrong `agents` package (an RL library). Uninstall `agents` and install **`openai-agents`** instead.
- For `.doc` (legacy Word) or scanned PDFs, convert to `.docx` / text-based PDF or run OCR first for best results.

---

## Update history

### v0.6 — 2025-09-09
- Added **DOI-only** input with **Crossref/OpenAlex/Semantic Scholar** fallback to avoid 403s.
- Strengthened URL fetching and added publisher heuristics (e.g., PNAS, bioRxiv).

### v0.5
- New **Clear Input Fields** button; resets inputs and status/progress.
- Startup **reset** clears all tabs and counts before new runs.

### v0.4
- Compact **custom progress bar** with text next to it; shows “Found N citations” and per-citation “(k/N)”.
- **Tabbed interface** with dynamic counts and **clickable links** for Exact/Close.

### v0.3
- **Downloadable Markdown report** (paper metadata + abstract; grouped citations with links).
- Dark/light theme **hero** header box.

### v0.2
- URL ingestion with PDF preference; HTML fallback if no PDF found.

### v0.1
- Initial release: upload PDF/DOCX, extract references, verify citations, bucket results.

---

## About

Developed by **Cornell Genomics Innovation Hub (2025)**.  
If you use this tool in your lab/classroom, a shout-out is appreciated!
