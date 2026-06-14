"""Download and parse the Michigan SNAP eligibility manual into structured sections.

The corpus is a real, scoped subset of Michigan's Bridges Eligibility Manual (BEM)
and Bridges Administrative Manual (BAM): the sections a caseworker actually reaches
for when determining food assistance (FAP, Michigan's name for SNAP). Using the
real manual matters because its structure is the thing we exploit: every page is
stamped with its section number, so a citation can point to an exact BEM/BAM
section.

This module is deterministic and idempotent: it downloads each section PDF (if not
already present), strips the repeated page boilerplate, and writes one clean JSON
record per section to data/manual_sections/. Those JSON files are committed, so the
rest of the pipeline runs without re-downloading.

Run:  python data/ingest.py
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

DATA = Path(__file__).resolve().parent
PDF_DIR = DATA / "manual_pdfs"
OUT_DIR = DATA / "manual_sections"
BASE_URL = "https://dhhs.michigan.gov/OLMWEB/EX/BP/Public/{manual}/{number}.pdf"

# The scoped FAP/SNAP corpus. Titles are filled in from each PDF's own header.
CORPUS: list[tuple[str, str]] = [
    ("BEM", "212"),  # Food Assistance Program group composition
    ("BEM", "213"),  # Categorical eligibility
    ("BEM", "400"),  # Assets
    ("BEM", "500"),  # Income overview
    ("BEM", "501"),  # Income from employment
    ("BEM", "502"),  # Income from self-employment
    ("BEM", "503"),  # Income, unearned
    ("BEM", "550"),  # FAP income budgeting
    ("BEM", "554"),  # FAP allowable expenses and expense budgeting
    ("BEM", "556"),  # Computing the food assistance budget
    ("BAM", "130"),  # Verification and collateral contacts
]

# Page boilerplate that repeats on every page and must be stripped.
HEADER_MARKER = re.compile(r"^B[AE]M\s+\d+\s+\d+\s+of\s+\d+\b")
BPB_LINE = re.compile(r"^BPB\s+\d{4}-\d+")
DATE_LINE = re.compile(r"^\d{1,2}-\d{1,2}-\d{4}\s*$")
BOILERPLATE_EXACT = {
    "BRIDGES ELIGIBILITY MANUAL STATE OF MICHIGAN",
    "DEPARTMENT OF HEALTH & HUMAN SERVICES",
    "BRIDGES ADMINISTRATIVE MANUAL STATE OF MICHIGAN",
    "STATE OF MICHIGAN",
}
CROSS_REF = re.compile(r"\b(B[AE]M)\s+(\d{3})\b")


@dataclass
class Section:
    manual: str
    number: str
    section_id: str  # e.g. "BEM 502"
    title: str
    url: str
    n_pages: int
    text: str
    cross_references: list[str]


def download(manual: str, number: str) -> Path:
    """Download a section PDF if missing. Tries urllib, falls back to curl."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDF_DIR / f"{manual}_{number}.pdf"
    if dest.exists() and dest.stat().st_size > 2000:
        return dest
    url = BASE_URL.format(manual=manual, number=number)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        dest.write_bytes(data)
    except Exception:
        # Some environments block urllib but allow curl; use it as a fallback.
        subprocess.run(
            ["curl", "-sL", "--max-time", "60", "-o", str(dest), url],
            check=True,
        )
    if dest.stat().st_size < 2000:
        raise RuntimeError(f"Download failed or too small: {url}")
    return dest


def parse_title(first_page_lines: list[str], section_id: str) -> str:
    """Extract the section title from the first page's header marker and spillover."""
    title_parts: list[str] = []
    started = False
    for ln in first_page_lines:
        if HEADER_MARKER.match(ln):
            # title begins after "BEM N X of Y "
            rest = re.sub(r"^B[AE]M\s+\d+\s+\d+\s+of\s+\d+\s*", "", ln).strip()
            if rest:
                title_parts.append(rest)
            started = True
            continue
        if started:
            if BPB_LINE.match(ln) or DATE_LINE.match(ln) or not ln.strip():
                break
            title_parts.append(ln.strip())
    title = re.sub(r"\s+", " ", " ".join(title_parts)).strip()
    return title or section_id


def strip_boilerplate(raw: str) -> str:
    """Drop the repeating page header/footer lines, keep the body text."""
    out: list[str] = []
    skipping_header = False
    for ln in raw.splitlines():
        s = ln.strip()
        if HEADER_MARKER.match(s):
            # entering a page header block: skip until we pass the date line
            skipping_header = True
            continue
        if skipping_header:
            if DATE_LINE.match(s):
                skipping_header = False
            continue
        if s in BOILERPLATE_EXACT or BPB_LINE.match(s):
            continue
        out.append(ln.rstrip())
    # collapse runs of blank lines
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def parse_section(manual: str, number: str, pdf_path: Path) -> Section:
    reader = PdfReader(str(pdf_path))
    section_id = f"{manual} {number}"
    first_lines = (reader.pages[0].extract_text() or "").splitlines()
    title = parse_title(first_lines, section_id)
    raw = "\n".join(p.extract_text() or "" for p in reader.pages)
    body = strip_boilerplate(raw)
    refs = sorted({f"{m} {n}" for m, n in CROSS_REF.findall(body) if f"{m} {n}" != section_id})
    return Section(
        manual=manual,
        number=number,
        section_id=section_id,
        title=title,
        url=BASE_URL.format(manual=manual, number=number),
        n_pages=len(reader.pages),
        text=body,
        cross_references=refs,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for manual, number in CORPUS:
        pdf = download(manual, number)
        section = parse_section(manual, number, pdf)
        out = OUT_DIR / f"{manual}_{number}.json"
        out.write_text(json.dumps(asdict(section), indent=2))
        summary.append((section.section_id, section.title, section.n_pages, len(section.text)))
        print(f"{section.section_id:8} | {section.n_pages:2}p | {len(section.text):6} chars | {section.title}")
    print(f"\nWrote {len(summary)} sections to {OUT_DIR}")


if __name__ == "__main__":
    main()
