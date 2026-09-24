"""
Strip non-prose markup (SVG, script, style blocks, and iframe/embed widget
remnants) from extracted text before it reaches the chunker.

Root cause this fixes: markdown scraped from docs sites sometimes embeds raw
HTML for icons, diagrams, or page widgets (e.g. an "Ask AI" chat embed) —
these are the source of the SVG-path-as-prose and the duplicate-boilerplate-
across-files problems found during spot-checking. Regex is used deliberately
over a full HTML parser here since input is markdown with occasional HTML
islands, not a full HTML document.
"""

import re

_SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_IFRAME_RE = re.compile(r"<iframe\b[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL)

# Common docs-site widget boilerplate seen leaking into scraped content
# (e.g. "Ask AI" chat embeds). Add more signatures here as you find them —
# this is what would have caught the identical iframeCache chunk that
# appeared verbatim across two unrelated files.
_WIDGET_SIGNATURES = (
    "iframeCache",
    "lastActiveAt",
)


def strip_markup_noise(text: str) -> str:
    """Remove SVG/script/style/iframe blocks and known widget boilerplate."""
    text = _SVG_RE.sub(" ", text)
    text = _SCRIPT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    text = _IFRAME_RE.sub(" ", text)

    # Drop any remaining line that contains a known widget signature — this
    # catches fragments where the closing tag got separated from the
    # opening tag by chunking, so the regexes above miss it.
    if any(sig in text for sig in _WIDGET_SIGNATURES):
        lines = text.splitlines()
        lines = [l for l in lines if not any(sig in l for sig in _WIDGET_SIGNATURES)]
        text = "\n".join(lines)

    # Collapse the whitespace holes left behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()