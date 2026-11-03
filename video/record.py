"""Screen-record the real app, scene by scene, in time with the narration.

Run (app must be running):  python video/record.py http://localhost:8502
Writes video/build/screen.webm and video/build/offset.txt (seconds to trim from the start).
"""

import json
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
BUILD = Path(__file__).resolve().parent / "build"
W, H = 1280, 720

CARD = """
(html) => {
  const d = document.createElement('div');
  d.id = 'sl-card';
  d.style.cssText = 'position:fixed;inset:0;z-index:99999;display:flex;flex-direction:column;justify-content:center;'
    + 'padding:0 110px;background:linear-gradient(135deg,#0b1d3a,#123f6b);color:#fff;font-family:Segoe UI,Arial,sans-serif;'
    + 'opacity:0;transition:opacity .6s';
  d.innerHTML = html;
  document.body.appendChild(d);
  requestAnimationFrame(() => d.style.opacity = 1);
}
"""

TITLE = """
<div style="font-size:72px;font-weight:700">🔎 Second Look</div>
<div style="font-size:30px;margin-top:18px;opacity:.92">A second opinion for loan applicants the credit bureau has never heard of</div>
<div style="font-size:20px;margin-top:56px;opacity:.75">Neural Storm 2026 · Track 3: Human Impact · HI-04 Alternative Credit Scoring</div>
<div style="font-size:20px;margin-top:8px;opacity:.75">Aryan Mashalkar</div>
"""

CLOSE = """
<div style="font-size:64px;font-weight:700">🔎 Second Look</div>
<div style="font-size:26px;margin-top:30px">Live demo: <b>second-look-credit.streamlit.app</b></div>
<div style="font-size:26px;margin-top:10px">Code, tests and metrics: <b>github.com/AryanMashalkar/second-look</b></div>
<div style="font-size:18px;margin-top:56px;opacity:.7">Data: Home Credit Default Risk (Kaggle). Demo profiles are synthetic. Voiceover is AI-generated (ElevenLabs).</div>
"""


def scroll_to(locator, block="center"):
    locator.evaluate(f"e => e.scrollIntoView({{behavior: 'smooth', block: '{block}'}})")


def main():
    timeline = json.loads((BUILD / "segments.json").read_text())
    starts = {s["name"]: s["start"] for s in timeline["segments"]}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(BUILD / "raw"),
                                  record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        video_start = time.monotonic()
        page.goto(URL)
        page.wait_for_selector("text=Ranking quality", timeout=60_000)
        page.wait_for_selector(".stPlotlyChart", timeout=60_000)
        page.wait_for_timeout(2500)

        page.evaluate(CARD, TITLE)
        page.wait_for_timeout(700)
        t0 = time.monotonic()

        def at(name):
            """Sleep until this scene's narration starts."""
            delay = t0 + starts[name] - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        at("overview")
        page.evaluate("() => { const d = document.getElementById('sl-card'); d.style.opacity = 0; setTimeout(() => d.remove(), 700); }")

        at("approve")
        scroll_to(page.locator(".stPlotlyChart").first)

        at("caution")
        box = page.locator('[data-testid="stSelectbox"]').filter(has_text="Open profile")
        scroll_to(box, "start")
        page.wait_for_timeout(600)
        box.click()
        page.wait_for_timeout(400)
        page.keyboard.type("SL-1001", delay=60)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1800)
        scroll_to(page.locator(".stPlotlyChart").first)
        page.wait_for_timeout(6500)
        scroll_to(page.get_by_text("Counter-offer:"))

        at("whatif")
        page.get_by_text("What if the loan terms change?").click()
        page.wait_for_timeout(600)
        slider = page.get_by_role("slider").first
        scroll_to(slider)
        page.wait_for_timeout(500)
        slider.focus()
        for _ in range(2):
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(700)

        at("decide")
        scroll_to(page.get_by_text("Officer decision"), "start")
        page.wait_for_timeout(700)
        page.get_by_text("Approve with counter-offer").click()
        page.wait_for_timeout(400)
        page.get_by_label("Reason (required, shared with the applicant if declined)").click()
        page.keyboard.type("Stable income; offered a 10% smaller loan.", delay=35)
        page.wait_for_timeout(300)
        page.get_by_role("button", name="Record decision").click()
        page.wait_for_timeout(1500)
        scroll_to(page.get_by_text("Download decision log"))

        at("evidence")
        page.get_by_role("tab", name="Evidence").click()
        page.wait_for_timeout(300)
        scroll_to(page.get_by_role("tab", name="Evidence"), "start")
        page.wait_for_timeout(5000)
        scroll_to(page.locator(".stPlotlyChart").nth(1))

        at("fairness")
        scroll_to(page.get_by_text("Every gender and age group is approved more often"))

        at("close")
        page.evaluate(CARD, CLOSE)
        time.sleep(timeline["segments"][-1]["duration"] + 1.5)

        offset = t0 - video_start
        video_path = page.video.path()
        ctx.close()
        browser.close()

    shutil.move(video_path, BUILD / "screen.webm")
    (BUILD / "offset.txt").write_text(f"{offset:.3f}")
    print(f"recorded, trim {offset:.2f}s")


if __name__ == "__main__":
    main()
