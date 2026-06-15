"""Shared visual language for the app — the Civil design system (Streamlit form).

A restrained derivative of the U.S. Web Design System: an open civic sans (Source
Sans 3) with a Source Serif 4 display face and Roboto Mono for citations, an
evergreen primary, a warm-paper canvas, and the framework chrome removed. The hero
front-end (benefits-intake-agent/web) carries the full component system; here the
same tokens give the suite a consistent feel. See that repo's DESIGN_SYSTEM.md.
"""

from __future__ import annotations

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=Roboto+Mono:wght@400;500&display=swap');

:root {
  --ink:#14181F; --paper:#FBFAF8; --panel:#F4F2EC; --line:#E4E0D7;
  --accent:#1F6F5C; --accent-ink:#143F35; --muted:#697078;
  --sans:'Source Sans 3', system-ui, sans-serif; --serif:'Source Serif 4', Georgia, serif;
  --mono:'Roboto Mono', ui-monospace, monospace;
}

[data-testid="stToolbar"], #MainMenu, [data-testid="stDecoration"],
header[data-testid="stHeader"], footer { display:none !important; }

html, body, .stApp, [class*="css"] { font-family:var(--sans); color:var(--ink); }
.stApp { background:var(--paper); }
.block-container { padding-top:2.2rem; max-width:1060px; }

.sn-eyebrow { font-size:11.5px; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin-bottom:.35rem; }
.sn-title { font-family:var(--serif); font-weight:600;
  font-size:2.4rem; line-height:1.1; letter-spacing:-0.01em; margin:0 0 .4rem 0; color:var(--ink); }
.sn-sub { color:var(--muted); font-size:0.98rem; line-height:1.55; max-width:62ch; margin:0; }
.sn-rule { height:1px; background:var(--line); border:0; margin:1.1rem 0 1.4rem 0; }

h2, h3 { font-family:var(--sans) !important; font-weight:600 !important;
  letter-spacing:-0.01em; color:var(--ink); }

code, kbd, .sn-mono { font-family:var(--mono) !important;
  font-size:0.86em; background:rgba(31,111,92,0.08); color:var(--accent-ink);
  padding:.05em .35em; border-radius:4px; }

.stButton>button[kind="primary"], .stButton>button:focus { border-color:var(--accent); }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within { border-color:var(--accent) !important; }
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }

.sn-foot { color:var(--muted); font-size:12px; border-top:1px solid var(--line);
  margin-top:2.5rem; padding-top:.9rem; }
.sn-foot b { color:var(--accent-ink); font-weight:600; }

/* case study */
.cs-lead { font-family:var(--serif); font-size:1.45rem; line-height:1.42; color:var(--ink); margin:.4rem 0 .8rem; }
.cs h2 { font-size:1.05rem; font-weight:600; margin:1.6rem 0 .6rem; }
.cs p { font-size:0.97rem; line-height:1.65; color:#3D4654; margin:0 0 .7rem; max-width:64ch; }
.cs .dec { border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; margin:.55rem 0; background:#fff; max-width:64ch; }
.cs .dec b { font-size:0.97rem; color:var(--ink); }
.cs .dec p { margin:.35rem 0 0; font-size:0.9rem; }
.cs .alt { color:var(--muted); font-size:0.85rem; margin-top:.35rem; }
.cs ul { max-width:64ch; } .cs li { font-size:0.95rem; line-height:1.6; color:#3D4654; margin-bottom:.4rem; }
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


HUB_URL = "https://adamselzer.github.io/safety-net-ai/"


def footer(st, note: str) -> None:
    st.markdown(
        f'<div class="sn-foot">{note} · part of the '
        f'<a href="{HUB_URL}" target="_blank"><b>Safety-Net AI</b></a> portfolio</div>',
        unsafe_allow_html=True,
    )


def case_study(st, html: str) -> None:
    """Render a case-study walkthrough from a composed HTML string."""
    st.markdown(f'<div class="cs">{html}</div>', unsafe_allow_html=True)
