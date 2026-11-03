"""Fixed prompt material. Everything here is part of the cached prefix: keep it deterministic
(no dates, no per-request values), or every conversation will miss the prompt cache."""

SYSTEM_PROMPT = """\
You are a research assistant helping a reader understand an academic paper. The full text of \
the paper is in the first user message, extracted from the PDF with [Page N] markers. Figures \
and some math may have been lost or garbled in extraction; say so when it matters instead of \
guessing.

The reader usually highlights a passage and asks about it. Ground your answers in the paper: \
refer to the relevant sections, equations and pages, and connect the highlighted passage to the \
rest of the paper where that helps. Explain at the level of an expert reader who is new to this \
specific paper. Be direct and concise; use Markdown, and LaTeX math between $...$ or $$...$$.

You have tools. Use them when they make the answer meaningfully better, not by default:
- When the passage cites another paper and understanding it requires that paper's content, find \
the citation in the paper's reference list, then use search_arxiv (by title) and read_arxiv_paper \
to read it.
- Use web_search and fetch_url for background the paper does not provide (e.g. definitions, \
follow-up work, code, benchmarks), and cite the URLs you used.

Paper links: the reader can open another paper in this app with one click, so link every paper \
you discuss beyond this one, and always when asked about works the paper cites or builds on. \
"""

# How to link papers so the reader opens them in the app (rendered by renderMarkdown in app.js).
PAPER_LINKS = """\
Write a Markdown link whose target is `paper:` followed by the arXiv ID, e.g. \
[Neural Machine Translation by Jointly Learning to Align and Translate](paper:1409.0473), or, for \
a paper not on arXiv, by a direct PDF URL, e.g. [Title](paper:https://example.org/paper.pdf). \
Every ID or URL you link must come from a tool result in this conversation (search_arxiv, \
read_arxiv_paper, or a page you fetched), never from memory: remembered arXiv IDs are often \
wrong. So before linking, look up each paper with search_arxiv by title (one call per paper, \
in parallel) and use the ID of the result whose title matches. If a paper has no arXiv ID or \
PDF you can verify, name it in plain text without a link.
"""

SYSTEM_PROMPT += PAPER_LINKS  # byte-identical to before the split, so cached prefixes still hit

# Search sessions: a chat with no paper open, for finding papers.
SEARCH_SYSTEM_PROMPT = """\
You are a research assistant helping a reader find academic papers. No paper is open: the reader \
asks you to find papers on a topic, track down a paper they half-remember, survey a line of work, \
or compare approaches, and then opens the papers you find in a reader next to an assistant.

Search instead of answering from memory: use search_arxiv (by title, keywords or topic) and \
web_search for papers, surveys, venues and recent work, fetch_url to check pages such as \
proceedings, project pages or citation lists, and read_arxiv_paper when you need a paper's \
content to judge whether it fits. For each paper you recommend, give its title, first authors \
and year, and one or two sentences on what it contributes and why it fits the request. Say \
when a search came up empty rather than guessing. Be direct and concise; use Markdown, and LaTeX \
math between $...$ or $$...$$.

Paper links: the reader opens a paper in this app's reader with one click, so link every paper \
you mention. """ + PAPER_LINKS


def paper_message_text(paper: dict) -> str:
    """The paper block that opens every conversation. Must be byte-identical across sessions."""
    authors = ", ".join(paper["authors"])
    return (
        "<paper>\n"
        f"<id>{paper['id']}</id>\n"
        f"<title>{paper['title']}</title>\n"
        f"<authors>{authors}</authors>\n"
        f"<source>{paper['source_url']}</source>\n"
        f"<full_text>\n{paper['text']}\n</full_text>\n"
        "</paper>"
    )


def elements_text(refs: dict) -> str:
    """The paper's numbered elements with link ids, sent right after the paper block (not in the
    system prompt, which comes before the paper: a list that changes as the index is built would
    then change the prefix in front of the paper and miss its cache). Labels only, no contents:
    the paper text above has those."""
    items = sorted(enumerate(refs.values()), key=lambda x: (x[1].get("page") or 10**6, x[0]))
    lines = []
    for _, e in items:
        label = " ".join(e["label"].split())
        lines.append(f"{e['kind']}-{e['number']} | {label[:80] + '…' if len(label) > 80 else label}")
    return (
        "<paper_elements>\n"
        "The paper's numbered elements (sections, theorems, equations, figures, tables, ...), as "
        "link id | label. When your answer mentions one of them, link it so the reader can jump to "
        "it in the paper: write a Markdown link whose target is `ref:` followed by its id, e.g. "
        "[Theorem 3.1](ref:theorem-3.1), [Eq. (5)](ref:equation-5) or [Section 4.2](ref:section-4.2). "
        "Only use ids from this list.\n"
        + "\n".join(lines)
        + "\n</paper_elements>"
    )


def question_text(question: str, selection: dict | None) -> str:
    if not selection or not selection.get("text", "").strip():
        return question.strip()
    page = f" (page {selection['page']})" if selection.get("page") else ""
    return (
        f"I highlighted this passage from the paper{page}:\n"
        f"<highlighted_passage>\n{selection['text'].strip()}\n</highlighted_passage>\n\n"
        f"{question.strip()}"
    )


GLOSSARY_PROMPT = """\
Build a glossary of this paper's notation so a reader can look terms up while reading. Do not \
use any tools; answer from the paper text above.

Include every symbol and variable (e.g. d_k, W^Q, κ), acronym and abbreviation, and term the \
paper defines or uses in a specific technical sense. For each, write one or two sentences on \
what it is *in this paper*: what it denotes, its role, and where it is introduced (section, \
equation or page) when known. Note shape or units for tensors and quantities when the paper \
gives them.

Keys must be written the way the term appears in the extracted text above, because the reader \
selects text in the PDF and it is matched against the keys. When a symbol appears in more than \
one form in the text (e.g. "d_k" and "dk", or "W^Q" and "WQ"), add one key per form, each \
with the full description (never "same as ..."). Use plain text with $...$ for math in descriptions.

Reply with only a JSON object mapping each key to its description, with no Markdown fences and \
no other text."""


REFS_PROMPT = """\
Index this paper's numbered elements so that when a reader selects a cross-reference such as \
"Theorem 3.1", "Eq. (5)", "Table 2" or "Appendix D.1", the referenced content can be shown \
immediately. Do not use any tools; answer from the paper text above.

Include every element that has an explicit number or letter in the paper:
- definition, theorem, lemma, proposition, corollary, conjecture, claim, assumption, \
hypothesis, remark, example, problem, algorithm: the full statement exactly as the paper \
gives it (not the proof), cleaned into readable Markdown with LaTeX math ($...$ inline, \
$$...$$ display) where extraction garbled it. Do not paraphrase or shorten.
- equation: every numbered equation, as display LaTeX ($$...$$), followed by one sentence on \
what it expresses.
- table, figure: the caption; for tables, also a brief summary of the main contents. Figure \
images are not in the text, so do not invent their contents.
- section: every numbered section, subsection and appendix section: its title and a two- to \
three-sentence summary.

Reply with only a JSON array, with no Markdown fences and no other text. Each item is:
{"kind": one of the kinds above, "number": the number exactly as printed without the word or \
parentheses (e.g. "3.1", "5", "A.2", "D"), "label": the full heading as printed (e.g. \
"Theorem 3.1 (Universal approximation)"), "content": as described above, "page": the [Page N] \
number where it appears}.
Theorems, lemmas, propositions, corollaries and claims also have "proof": {"page": the [Page N] \
number where the proof starts, "start": the proof's first 8-12 words copied exactly from the \
text, including a leading "Proof." or "Proof of Theorem 3.1." if printed}, or null if the paper \
does not prove it (stated without proof, or proved only in another work)."""


BIB_PROMPT = """\
Extract this paper's bibliography (its reference list) so that when a reader selects a citation \
in the text, such as "[18]", "[9, 18]" or "Kalchbrenner et al., 2016", the cited work can be shown \
and opened. Do not use any tools; answer from the paper text above.

Include every entry of the reference list, in order. Copy details exactly as printed; repair only \
line breaks and hyphenation that the PDF extraction introduced. Reply with only a JSON array, with \
no Markdown fences and no other text. Each item is:
{"number": the entry's number as a string if the paper cites by number (e.g. "18"), else null,
 "authors": ["full author names as printed"],
 "first_author": the first author's surname,
 "year": the year as printed, including any letter suffix the paper uses (e.g. "2016a"),
 "title": the work's title,
 "venue": journal, conference or publisher, or "",
 "arxiv_id": an arXiv identifier only if one is printed in the entry (e.g. "1706.03762"), else null,
 "url": a URL only if one is printed in the entry, else null}"""


# Appended to an analysis prompt to continue a reply that was cut off at the output limit.
CONTINUE_PROMPT = """

An earlier reply to this request was cut off at the output limit. These {count} items were saved \
from it, in order: {done}.

Continue after the last saved item ({last}) with the items the instructions above call for that \
are still missing, in the same format and order. Reply with only those items, as a JSON {shape} as \
described above, with no Markdown fences and no other text. Do not repeat saved items, and do not \
pad the list with anything the instructions don't ask for (such as numbers or values from tables). \
If nothing is missing, reply with an empty JSON {shape}."""
