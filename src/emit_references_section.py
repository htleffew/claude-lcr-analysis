"""Read references.bib, emit a manually-typed References section in APA style.

The LCR preprint uses inline APA citations rather than \\cite{} commands;
this script produces the matching alphabetical bibliography block to drop
into the .tex.

Outputs the LaTeX fragment to stdout. Pipe to a file or paste into the .tex.
"""
import re
import sys
from pathlib import Path


BIB = Path(__file__).resolve().parent.parent / "paper" / "references.bib"

# Citations actually present in the prose (cross-checked against the .tex).
# Quiroz-Gutierrez (2026) is in the .bib but not cited in the LCR paper.
CITED_KEYS = {
    "Anthropic2024Character",
    "Bai2022b",
    "BeauchampChildress2019",
    "Bommasani2021",
    "BraunClarke2019",
    "Charmaz2014",
    "Christiano2017",
    "Fitzpatrick2017",
    "GabbardLester2003",
    "Gutheil1993",
    "Inkster2018",
    "Joiner2005",
    "Krippendorff2018",
    "Leffew2025",
    "Liang2023",
    "Mill2003",
    "Olah2020",
    "Ouyang2022",
    "PopeVasquez2016",
    "Smith2009",
    "Templeton2024",
    "Vaidyam2019",
}


def parse_bibtex(text):
    """Return list of (key, entry_type, fields_dict)."""
    entries = []
    # Find each @type{key, ... }
    for m in re.finditer(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", text, re.DOTALL):
        etype = m.group(1).lower()
        key = m.group(2).strip()
        body = m.group(3)
        fields = {}
        # Match field = {value} OR field = "value" with potential nested braces
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"]([^{}\"]*(?:\{[^{}]*\}[^{}\"]*)*)[}\"]\s*,?", body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append((key, etype, fields))
    return entries


def format_authors_apa(author_field):
    """Convert BibTeX author field to APA inline form: 'Last, F. M., & Last, F. M.'"""
    if not author_field:
        return ""
    authors = [a.strip() for a in re.split(r"\s+and\s+", author_field)]
    formatted = []
    for a in authors:
        if "," in a:
            last, rest = a.split(",", 1)
            initials = "".join(
                f"{p[0]}. " for p in rest.strip().split() if p
            ).strip()
            formatted.append(f"{last.strip()}, {initials}")
        else:
            parts = a.split()
            if len(parts) == 1:
                formatted.append(parts[0])
            else:
                last = parts[-1]
                initials = "".join(f"{p[0]}. " for p in parts[:-1]).strip()
                formatted.append(f"{last}, {initials}")
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, \\& {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", \\& {formatted[-1]}"


def format_entry_apa(key, etype, f):
    """Produce one APA-formatted reference entry as a LaTeX paragraph."""
    authors = format_authors_apa(f.get("author", ""))
    year = f.get("year", "n.d.")
    if etype == "misc" and "month" in f and f.get("year"):
        year = f"{f['year']}"  # APA uses just year in inline
    title = f.get("title", "")
    # Strip {} grouping marks BibTeX uses for protecting capitalization
    title = re.sub(r"\{(.+?)\}", r"\1", title)

    if etype == "article":
        journal = f.get("journal", "")
        vol = f.get("volume", "")
        num = f.get("number", "")
        pages = f.get("pages", "").replace("--", "-")
        bits = [f"{authors} ({year}).", f"{title}.", f"\\emph{{{journal}}}"]
        if vol:
            vp = f"\\emph{{{vol}}}"
            if num:
                vp += f"({num})"
            if pages:
                vp += f", {pages}"
            bits.append(vp + ".")
        else:
            bits[-1] = bits[-1] + "."
        return " ".join(bits)

    if etype == "book":
        publisher = f.get("publisher", "")
        edition = f.get("edition", "")
        edstr = f" ({edition} ed.)" if edition else ""
        return f"{authors} ({year}). \\emph{{{title}}}{edstr}. {publisher}."

    if etype == "incollection":
        booktitle = f.get("booktitle", "")
        publisher = f.get("publisher", "")
        pages = f.get("pages", "").replace("--", "-")
        pp = f" (pp. {pages})" if pages else ""
        return f"{authors} ({year}). {title}. In \\emph{{{booktitle}}}{pp}. {publisher}."

    if etype == "misc":
        how = f.get("howpublished", "")
        url = f.get("url", "")
        note = ""
        if how and url:
            note = f"\\emph{{{how}}}. \\url{{{url}}}"
        elif how:
            note = f"\\emph{{{how}}}."
        elif url:
            note = f"\\url{{{url}}}"
        return f"{authors} ({year}). {title}. {note}".strip()

    if etype == "unpublished":
        note = f.get("note", "")
        return f"{authors} ({year}). \\emph{{{title}}}. {note}."

    # Fallback
    return f"{authors} ({year}). {title}."


def main():
    text = BIB.read_text(encoding="utf-8")
    entries = parse_bibtex(text)

    cited = [(k, t, f) for k, t, f in entries if k in CITED_KEYS]
    cited.sort(key=lambda e: (e[2].get("author", "").lower(), e[2].get("year", "")))

    missing = CITED_KEYS - {k for k, _, _ in cited}
    if missing:
        print(f"% WARNING: cited keys not found in references.bib: {missing}",
              file=sys.stderr)

    print(r"\section*{References}")
    print(r"\small")
    print(r"\begin{hangparas}{0.4in}{1}")
    # Fallback if hangparas not loaded: use itemize with negative left margin
    # The class doesn't load hangparas; switch to indent-style paragraphs.
    # Simpler: use \par-separated paragraphs with first-line indent reversed.
    print(r"\end{hangparas}")

    # Emit using a more portable list-free approach with manual hanging indent
    sys.stdout.flush()


def emit_portable():
    text = BIB.read_text(encoding="utf-8")
    entries = parse_bibtex(text)
    cited = [(k, t, f) for k, t, f in entries if k in CITED_KEYS]
    # Sort by author surname (strip leading {{ }} group-author braces) then year
    cited.sort(key=lambda e: (re.sub(r"[{}]", "", e[2].get("author", "")).lower(),
                              e[2].get("year", "")))

    out = []
    out.append(r"\newpage")
    out.append(r"\section*{References}")
    out.append("")
    for key, etype, fields in cited:
        line = format_entry_apa(key, etype, fields)
        # APA 7th Edition requires a 0.5in hanging indent for reference lists.
        out.append(r"\hangindent=0.5in \hangafter=1 " + line.rstrip() + r"\par")
    return "\n".join(out)


if __name__ == "__main__":
    print(emit_portable())
