"""Crop a figure or table out of the paper's PDF, located by its caption, for the reader's popover.

No model involved: find the caption block ("Figure 2:" / "Table 1."), then grow a region from it
over nearby images and vector drawings (rules, plots), upward for figures and downward for tables
(the usual caption placement), trying the other direction when nothing is found.
"""

import re

import pymupdf

from . import papers
from .config import DATA_DIR

CROPS_DIR = DATA_DIR / "crops"
_KIND_WORDS = {"figure": r"(?:Figure|Fig\.?)", "table": r"Table"}
GAP = 28  # max vertical gap (pt) between drawn pieces of one figure/table
TEXT_GAP = 14  # max gap for text bridging them (sub-captions, table rows)
CAPTION_RE = re.compile(r"^\s*(?:Figure|Fig\.?|Table)\s*[A-Z]?\d+(?:\.\d+)*\s*[:.|]", re.I)
PAD = 6


def crops_dir(paper: dict):
    """Named like the PDF (papers.pdf_path): by its hash, shared by every user's copy."""
    return CROPS_DIR / papers.pdf_path(paper).stem


_NUMBER = re.compile(r"[A-Z]?\.?\d{1,4}(?:\.\d{1,4}){0,3}", re.I)  # "3", "A2", "A.2", "4.1.2"


def valid_number(number: str) -> bool:
    """A figure or table number as captions print them; anything else (a path, say) is refused."""
    return bool(_NUMBER.fullmatch(number))


def crop_path(paper: dict, kind: str, number: str):
    if kind not in _KIND_WORDS or not valid_number(number):
        raise ValueError(f"Not a figure or table: {kind} {number!r}")
    return crops_dir(paper) / f"{kind}-{number}.png"


def crop(paper: dict, kind: str, number: str, page_hint: int | None) -> bytes | None:
    """PNG of the figure/table, cached on disk; None when it can't be located."""
    path = crop_path(paper, kind, number)
    if path.exists():
        return path.read_bytes() or None  # an empty file caches "not found"
    png = _render(paper, kind, number, page_hint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png or b"")
    return png


def _render(paper, kind, number, page_hint):
    pdf = papers.pdf_path(paper)
    if kind not in _KIND_WORDS or not pdf.exists():
        return None
    caption_re = re.compile(rf"^\s*{_KIND_WORDS[kind]}\s*{re.escape(number)}\s*[:.|]", re.I)
    with pymupdf.open(pdf) as doc:
        # The page recorded by the refs index first, then its neighbours, then everything.
        order = [p - 1 for p in (page_hint, page_hint and page_hint + 1, page_hint and page_hint - 1) if p]
        order += [i for i in range(doc.page_count) if i not in order]
        for i in order:
            if not 0 <= i < doc.page_count:
                continue
            page = doc[i]
            for block in page.get_text("blocks"):
                if block[6] == 0 and caption_re.match(block[4]):
                    rect = _region(page, pymupdf.Rect(block[:4]), kind)
                    if rect:
                        return page.get_pixmap(clip=rect, dpi=200).tobytes("png")
    return None


def _region(page, caption, kind):
    pr = page.rect
    # Column the caption sits in; full-width captions allow content across the whole text width.
    col = pymupdf.Rect(caption.x0 - 12, 0, caption.x1 + 12, pr.height)
    pieces = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 1 and r.height < 1:
            continue
        # Skip running header/footer rules and page-sized frames.
        if r.height < 2 and (r.y1 < pr.height * 0.09 or r.y0 > pr.height * 0.92):
            continue
        if r.height > pr.height * 0.8:
            continue
        # Rules are zero-height (or zero-width) rects, which PyMuPDF treats as empty and drops
        # from unions; give them a hairline of thickness.
        pieces.append(r + (-0.5, -0.5, 0.5, 0.5))
    pieces = [r for r in pieces if r.x1 > col.x0 and r.x0 < col.x1 and not r.intersects(caption)]
    # Short text (sub-captions, table rows, axis labels) bridges gaps between drawn parts but never
    # defines the extent; running paragraphs (long lines) are never included.
    bridges = []
    for b in page.get_text("blocks"):
        r, text = pymupdf.Rect(b[:4]), b[4].strip()
        lines = max(text.count("\n") + 1, 1)
        if b[6] == 0 and text and len(text) / lines < 45 and r.x1 > col.x0 and r.x0 < col.x1 \
                and not r.intersects(caption) and not CAPTION_RE.match(text):
            bridges.append(r)
    candidates = [(r, GAP, True) for r in pieces] + [(r, TEXT_GAP, False) for r in bridges]
    first = "down" if kind == "table" else "up"
    for direction in (first, "up" if first == "down" else "down"):
        found = _grow(caption, candidates, direction)
        if found:
            break
    else:
        if kind != "table":
            return None
        # Borderless table: just its rows, chained closely below the caption.
        found = _grow(caption, [(r, TEXT_GAP, True) for r in bridges], "down")
        if not found:
            return None
    region = found | caption
    region.x0, region.x1 = min(region.x0, caption.x0), max(region.x1, caption.x1)
    return (region + (-PAD, -PAD, PAD, PAD)) & pr


def _grow(caption, candidates, direction):
    """Grow from the caption edge one way over (rect, max_gap, is_anchor) candidates; the result
    reaches only as far as the outermost anchor (drawing/image), so bridging text can't extend it."""
    edge = caption.y0 if direction == "up" else caption.y1
    region, anchor_edge = None, None
    remaining = list(candidates)
    progress = True
    while progress:
        progress = False
        for item in list(remaining):
            r, gap, anchor = item
            near = (edge - r.y1 < gap and r.y1 <= caption.y0 + 2) if direction == "up" else \
                   (r.y0 - edge < gap and r.y0 >= caption.y1 - 2)
            if not near:
                continue
            remaining.remove(item)
            progress = True
            edge = min(edge, r.y0) if direction == "up" else max(edge, r.y1)
            if anchor:
                region = r if region is None else region | r
                anchor_edge = r.y0 if anchor_edge is None else (min(anchor_edge, r.y0) if direction == "up" else max(anchor_edge, r.y1))
            elif region is not None:
                region |= r
    if region is None:
        return None
    # Drop bridging text beyond the outermost anchor.
    if direction == "up":
        region.y0 = max(region.y0, anchor_edge)
    else:
        region.y1 = min(region.y1, anchor_edge if anchor_edge > region.y0 else region.y1)
    return region
