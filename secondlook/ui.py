"""Visual language for the app: a dark ledger, green for approvals, red for declines, rubber-stamp decisions."""

import html

import plotly.graph_objects as go
import plotly.io as pio

PAPER = "#0A0D0B"
PANEL = "#111714"
INK = "#E6EEE8"
MUTED = "#8C9C92"
RULE = "#25302A"
APPROVE = "#2FBF71"
DECLINE = "#EF5350"
BASELINE = "#66766D"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Special+Elite&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
/* The rule above also matches Streamlit's icon spans; give them their icon font back. */
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
[data-testid="stTable"] th {{ word-break: normal; overflow-wrap: normal; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1, h2, h3 {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; letter-spacing: -0.015em; }}
h2 {{ font-weight: 700; font-size: 1.45rem; }}
h3 {{ font-weight: 650; font-size: 1.1rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 1rem; padding: 0.6rem 0.1rem; }}
.stTabs [aria-selected="true"] {{ color: {APPROVE}; }}
:focus-visible {{ outline: 2px solid {APPROVE} !important; outline-offset: 2px; }}

/* masthead */
.sl-mast {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 2.5rem; align-items: end;
  border-bottom: 2px solid {INK}; padding-bottom: 1.3rem; margin-bottom: 0.4rem; }}
.sl-mast h1 {{ font-size: 2.9rem; line-height: 1.05; font-weight: 700; margin: 0 0 0.6rem 0; display: flex; align-items: center; gap: 0.7rem; }}
.sl-mast h1::before {{ content: ''; width: 0.55em; height: 0.55em; background: {APPROVE}; border-radius: 2px; }}
.sl-mast p {{ font-size: 1.06rem; line-height: 1.55; color: {INK}; margin: 0; max-width: 34em; }}
.sl-ledger {{ border-top: 1px solid {RULE}; }}
.sl-ledger div {{ display: flex; justify-content: space-between; gap: 1rem; padding: 0.5rem 0;
  border-bottom: 1px solid {RULE}; font-size: 0.95rem; }}
.sl-ledger b {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.12rem; color: {APPROVE}; }}
.sl-ledger span {{ color: {MUTED}; }}

/* case file */
.sl-case {{ background: {PANEL}; border: 1px solid {RULE}; border-left: 4px solid {APPROVE}; border-radius: 4px;
  padding: 1.2rem 1.4rem 1rem; position: relative; }}
.sl-case h3 {{ margin: 0 0 0.2rem 0; font-size: 1.35rem; }}
.sl-case .sub {{ color: {MUTED}; font-size: 0.92rem; margin-bottom: 0.9rem; }}
.sl-facts {{ display: grid; grid-template-columns: 1fr 1fr; column-gap: 2rem; }}
.sl-facts div {{ display: flex; justify-content: space-between; border-bottom: 1px dotted {RULE};
  padding: 0.32rem 0; font-size: 0.93rem; }}
.sl-facts span {{ color: {MUTED}; }}
.sl-facts b {{ font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: 0.9rem; text-align: right; }}
.sl-stamps {{ display: flex; gap: 1.2rem; margin-top: 1.1rem; flex-wrap: wrap; }}
.sl-stampbox {{ flex: 1; min-width: 210px; }}
.sl-stampbox .who {{ font-size: 0.85rem; color: {MUTED}; margin-bottom: 0.35rem; }}
.sl-stamp {{ display: inline-block; font-family: 'Special Elite', 'Courier New', monospace; font-size: 1.55rem;
  letter-spacing: 0.08em; padding: 0.15rem 0.7rem; border: 3px solid currentColor; border-radius: 6px;
  transform: rotate(-4deg); opacity: 0.88; }}
.sl-stamp.ok {{ color: {APPROVE}; }}
.sl-stamp.no {{ color: {DECLINE}; transform: rotate(3deg); }}
.sl-stampbox .risk {{ font-size: 0.95rem; margin-top: 0.45rem; font-variant-numeric: tabular-nums; }}
@media (prefers-reduced-motion: no-preference) {{
  .sl-stamp.fresh {{ animation: sl-press .38s cubic-bezier(.2,.9,.3,1.3) both; }}
  @keyframes sl-press {{ from {{ transform: scale(1.6) rotate(-10deg); opacity: 0; }} }}
}}

/* counter-offer slip */
.sl-slip {{ border: 2px dashed {APPROVE}; background: rgba(47, 191, 113, 0.08); padding: 0.9rem 1.1rem; margin: 0.6rem 0 0.4rem;
  font-size: 0.97rem; line-height: 1.5; }}
.sl-slip.none {{ border-color: {DECLINE}; background: rgba(239, 83, 80, 0.08); }}
.sl-slip.clear {{ border-color: {RULE}; background: {PANEL}; }}
.sl-note {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.5; }}

/* queue */
div[data-testid="stRadio"] label p {{ font-variant-numeric: tabular-nums; }}
.sl-legend {{ font-size: 0.86rem; color: {MUTED}; margin: -0.2rem 0 0.5rem; }}

@media (max-width: 860px) {{
  .sl-mast {{ grid-template-columns: 1fr; gap: 1rem; }}
  .sl-mast h1 {{ font-size: 2.2rem; }}
  .sl-facts {{ grid-template-columns: 1fr; }}
}}
</style>
"""


def esc(x) -> str:
    return html.escape(str(x))


def stamp(approved: bool, fresh=False) -> str:
    cls = "ok" if approved else "no"
    word = "APPROVED" if approved else "DECLINED"
    return f'<span class="sl-stamp {cls}{" fresh" if fresh else ""}">{word}</span>'


def stamp_pair(base_risk, base_ok, sl_risk, sl_ok, base_cut, sl_cut, fresh=False) -> str:
    return f"""
<div class="sl-stamps">
  <div class="sl-stampbox"><div class="who">External credit score</div>{stamp(base_ok, fresh)}
    <div class="risk">Default risk <b>{base_risk:.1%}</b> <span class="sl-note">(cut-off {base_cut:.1%})</span></div></div>
  <div class="sl-stampbox"><div class="who">Second Look</div>{stamp(sl_ok, fresh)}
    <div class="risk">Default risk <b>{sl_risk:.1%}</b> <span class="sl-note">(cut-off {sl_cut:.1%})</span></div></div>
</div>"""


def facts(rows) -> str:
    cells = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in rows)
    return f'<div class="sl-facts">{cells}</div>'


def ledger(rows) -> str:
    cells = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in rows)
    return f'<div class="sl-ledger">{cells}</div>'


def register_plotly_theme():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[APPROVE, BASELINE, DECLINE, "#2F7D5B"],
        xaxis=dict(gridcolor=RULE, zerolinecolor=MUTED, linecolor=RULE),
        yaxis=dict(gridcolor=RULE, zerolinecolor=MUTED, linecolor=RULE),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    pio.templates["ledger"] = t
    pio.templates.default = "ledger"
