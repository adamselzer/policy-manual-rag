"""Shared visual language for the app.

A restrained system, not a redesign: a considered neutral palette, a serif display
face paired with a humanist sans and a mono for codes and citations, and the
framework chrome removed. The same treatment is used across the four Safety-Net AI
apps so the suite reads as one body of work.
"""

from __future__ import annotations

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink:#16202A; --paper:#FBFAF8; --panel:#F1EFEA; --line:#E4E0D7;
  --accent:#1F6F5C; --accent-ink:#16433A; --muted:#6B7280;
}

/* remove framework chrome for an app-like surface */
[data-testid="stToolbar"], #MainMenu, [data-testid="stDecoration"],
header[data-testid="stHeader"], footer { display:none !important; }

html, body, .stApp, [class*="css"] { font-family:'Inter', system-ui, sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
.block-container { padding-top:2.2rem; max-width:1060px; }

/* header */
.sn-eyebrow { font-size:11.5px; letter-spacing:0.16em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin-bottom:.35rem; }
.sn-title { font-family:'Newsreader', ui-serif, Georgia, serif; font-weight:500;
  font-size:2.5rem; line-height:1.08; letter-spacing:-0.01em; margin:0 0 .4rem 0; color:var(--ink); }
.sn-sub { color:var(--muted); font-size:0.98rem; line-height:1.5; max-width:62ch; margin:0; }
.sn-rule { height:1px; background:var(--line); border:0; margin:1.1rem 0 1.4rem 0; }

/* section headings stay sans for a crisp hierarchy under the serif title */
h2, h3 { font-family:'Inter', system-ui, sans-serif !important; font-weight:600 !important;
  letter-spacing:-0.01em; color:var(--ink); }

/* codes, ids, citations read as artifacts */
code, kbd, .sn-mono { font-family:'JetBrains Mono', ui-monospace, monospace !important;
  font-size:0.86em; background:rgba(31,111,92,0.08); color:var(--accent-ink);
  padding:.05em .35em; border-radius:4px; }

/* accent the primary button + form focus without shouting */
.stButton>button[kind="primary"], .stButton>button:focus { border-color:var(--accent); }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within { border-color:var(--accent) !important; }

/* sidebar as a quiet panel */
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }

/* footer */
.sn-foot { color:var(--muted); font-size:12px; border-top:1px solid var(--line);
  margin-top:2.5rem; padding-top:.9rem; }
.sn-foot b { color:var(--accent-ink); font-weight:600; }
</style>
"""


def apply_theme(st) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def header(st, eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="sn-eyebrow">{eyebrow}</div>'
        f'<h1 class="sn-title">{title}</h1>'
        f'<p class="sn-sub">{subtitle}</p>'
        f'<hr class="sn-rule"/>',
        unsafe_allow_html=True,
    )


def footer(st, note: str) -> None:
    st.markdown(
        f'<div class="sn-foot">{note} · part of the <b>Safety-Net AI</b> portfolio</div>',
        unsafe_allow_html=True,
    )
