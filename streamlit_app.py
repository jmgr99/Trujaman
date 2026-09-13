"""
Judeo-Arabic / Hebrew word tagger — Streamlit version.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this file, requirements.txt and ja_he_clf.joblib
               to a GitHub repo, then point share.streamlit.io at it.
"""

import csv
import html
import io
import re
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "ja_he_clf.joblib"

# Same cleaning as in training (cell 23) — do not change without retraining.
STRIP = re.compile("[\u05c4\u05bf\u05f3\u05f4\u0027\u0022,.;:()\\[\\]\\-\u2013\u2014?!\u2026]")
HEB_LETTER = re.compile("[\u05d0-\u05ea]")


@st.cache_resource
def load_model():
    """Load the classifier once and keep it in memory between visitors."""
    return joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def tokenize(text):
    """Split text into tokens, keeping every token, flagging which are taggable."""
    tokens = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for raw in line.split():
            norm = STRIP.sub("", raw)
            tokens.append(
                {
                    "line": line_no,
                    "word": raw,
                    "norm": norm,
                    "taggable": bool(HEB_LETTER.search(norm)),
                    "lang": None,
                }
            )
    return tokens


def classify(tokens, clf, context_crosses_lines=True):
    """Fill in tokens[i]['lang'] for every taggable token."""
    idx = [i for i, t in enumerate(tokens) if t["taggable"]]
    if not idx:
        return tokens

    norm = [tokens[i]["norm"] for i in idx]
    line_no = [tokens[i]["line"] for i in idx]

    feats = []
    for k, w in enumerate(norm):
        if context_crosses_lines:
            prev = norm[k - 1] if k > 0 else ""
            nxt = norm[k + 1] if k + 1 < len(norm) else ""
        else:
            prev = norm[k - 1] if k > 0 and line_no[k - 1] == line_no[k] else ""
            nxt = norm[k + 1] if k + 1 < len(norm) and line_no[k + 1] == line_no[k] else ""
        feats.append(f"{prev} ~{w}~ {nxt}")

    for k, y in zip(idx, clf.predict(feats)):
        tokens[k]["lang"] = "HE" if y == 1 else "JA"

    return tokens


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

STYLE = """
<style>
  .stTextArea textarea {
    direction: rtl;
    text-align: right;
    font-family: "SBL Hebrew", "Ezra SIL", "Taamey Frank CLM",
                 "Frank Ruehl CLM", "David", serif;
    font-size: 1.15rem;
    line-height: 1.9;
  }
  .tagger-sheet {
    --paper: #f5f3ef;
    --ink: #1f1d1a;
    --rule: #d9d4ca;
    --he: #2e5e7e;
    --ja: #8a6a2f;
    background: var(--paper);
    color: var(--ink);
    padding: 1.5rem 1.25rem;
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow-x: auto;
  }
  .tagger-line {
    display: flex;
    flex-direction: row-reverse;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: flex-start;
    gap: 0.45rem 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--rule);
  }
  .tagger-line:last-child { border-bottom: none; }
  .tagger-num {
    font: 400 0.7rem/1 ui-monospace, monospace;
    color: #9b958a;
    min-width: 2ch;
    text-align: left;
    padding-bottom: 0.9rem;
  }
  .tagger-word {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.15rem;
  }
  .tagger-word .form {
    font-family: "SBL Hebrew", "Ezra SIL", "Taamey Frank CLM",
                 "Frank Ruehl CLM", "David", serif;
    font-size: 1.45rem;
    line-height: 1.5;
    direction: rtl;
  }
  .tagger-word .tag {
    font: 600 0.62rem/1 ui-monospace, monospace;
    letter-spacing: 0.06em;
    padding: 0.12rem 0.3rem;
    border-radius: 2px;
    color: #fff;
  }
  .tagger-word.he .form { border-bottom: 2px solid var(--he); }
  .tagger-word.he .tag  { background: var(--he); }
  .tagger-word.ja .form { border-bottom: 2px solid var(--ja); }
  .tagger-word.ja .tag  { background: var(--ja); }
  .tagger-word.skip .form { color: #a29b90; }
</style>
"""


def render(tokens):
    """Turn classified tokens into the HTML shown on the page."""
    lines = {}
    for t in tokens:
        lines.setdefault(t["line"], []).append(t)

    out = []
    for line_no in sorted(lines):
        cells = [f'<span class="tagger-num">{line_no}</span>']
        for t in lines[line_no]:
            form = html.escape(t["word"])
            if t["lang"]:
                cells.append(
                    f'<span class="tagger-word {t["lang"].lower()}">'
                    f'<span class="form">{form}</span>'
                    f'<span class="tag">{t["lang"]}</span></span>'
                )
            else:
                cells.append(
                    f'<span class="tagger-word skip">'
                    f'<span class="form">{form}</span></span>'
                )
        out.append(f'<div class="tagger-line">{"".join(cells)}</div>')

    return f'<div class="tagger-sheet">{"".join(out)}</div>'


def to_csv(tokens):
    """Return the tagged words as CSV text, BOM included so Excel reads Hebrew."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["line", "word", "lang"])
    for t in tokens:
        if t["lang"]:
            writer.writerow([t["line"], t["word"], t["lang"]])
    return "\ufeff" + buf.getvalue()


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

st.set_page_config(page_title="Trujaman: Judeo-Arabic & Hebrew word tagger", layout="wide")
st.html(STYLE)

st.title("Judeo-Arabic / Hebrew word tagger")
st.write(
    "Paste a Hebrew-script text. Every word is labelled **HE** (Hebrew) or "
    "**JA** (Judeo-Arabic)."
)

try:
    clf = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(
        f"The model file could not be loaded from {MODEL_PATH.name}. "
        f"Python reported: {exc}"
    )
    st.stop()

text = st.text_area(
    "Text",
    height=240,
    placeholder="פקאל לה אלפילסוף, ליס ענד אללה רצׄי ולא בגׄץׄ",
)

col_a, col_b = st.columns([3, 2])

if st.button("Tag the text", type="primary"):
    if not text.strip():
        st.warning("Paste some text above, then tag it.")
    else:
        st.session_state["tokens"] = classify(tokenize(text), clf, cross_lines)

tokens = st.session_state.get("tokens")

if tokens:
    he = sum(1 for t in tokens if t["lang"] == "HE")
    ja = sum(1 for t in tokens if t["lang"] == "JA")
    total = he + ja

    if not total:
        st.info("No Hebrew-script words found in that text.")
    else:
        skipped = sum(1 for t in tokens if not t["taggable"])
        note = (
            f" {skipped} token(s) held no Hebrew letters and were left untagged, in grey."
            if skipped
            else ""
        )
        st.write(
            f"**{he}** Hebrew and **{ja}** Judeo-Arabic words out of {total} tagged "
            f"— {100 * he / total:.1f}% Hebrew.{note}"
        )
        st.html(render(tokens))

        safe = re.sub(r"[^\w.\- ]", "_", name).strip() or "tagged"
        st.download_button(
            "Download the tagged words as CSV",
            data=to_csv(tokens),
            file_name=f"{safe}.csv",
            mime="text/csv",
        )
