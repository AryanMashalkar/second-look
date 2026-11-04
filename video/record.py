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
<div style="font-size:18px;margin-top:56px;opacity:.7">Data: Home Credit Default Risk (Kaggle). Demo profiles are synthetic. Voiceover is AI-generated (Microsoft neural TTS).</div>
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
        page.wait_for_selector("text=Review queue", timeout=60_000)
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

        def tab(name):
            page.get_by_role("tab", name=name, exact=True).click()
            page.wait_for_timeout(500)

        at("overview")
        page.evaluate("() => { const d = document.getElementById('sl-card'); d.style.opacity = 0; setTimeout(() => d.remove(), 700); }")

        at("approve")
        scroll_to(page.locator(".sl-case").first, "start")
        page.wait_for_timeout(9000)
        scroll_to(page.locator(".stPlotlyChart").first)

        at("caution")
        case = page.get_by_text("SL-1031", exact=True)
        scroll_to(case)
        page.wait_for_timeout(700)
        case.click()
        page.wait_for_timeout(1500)
        scroll_to(page.locator(".sl-case").first, "start")
        page.wait_for_timeout(8000)
        tab("Counter-offer and what-if")
        scroll_to(page.locator(".sl-slip").first)

        at("whatif")
        slider = page.get_by_role("slider").first
        scroll_to(slider)
        page.wait_for_timeout(400)
        slider.focus()
        for _ in range(2):
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(600)

        at("decide")
        tab("Decide")
        scroll_to(page.get_by_text("Approve with counter-offer"))
        page.wait_for_timeout(500)
        page.get_by_text("Approve with counter-offer").click()
        page.wait_for_timeout(300)
        page.get_by_label("Reason (required, shared with the applicant if declined)").click()
        page.keyboard.type("Late payments noted; offered a 10% smaller loan.", delay=30)
        page.get_by_role("button", name="Record decision").click()
        page.wait_for_timeout(1500)
        scroll_to(page.get_by_text("Download decision log"))

        at("trial")
        tab("Try an applicant")
        scroll_to(page.get_by_text("Score an applicant of your own"), "start")
        page.wait_for_timeout(1500)
        late = page.locator('[data-testid="stSlider"]').filter(has_text="Share of them paid late").get_by_role("slider")
        late.focus()
        for _ in range(4):  # a few big steps, letting each rerun finish so the page never sits dimmed
            for _ in range(8):
                page.keyboard.press("ArrowRight")
            page.wait_for_timeout(1400)

        at("evidence")
        tab("Evidence")
        policy = page.locator('[data-testid="stSlider"]').filter(has_text="Approve this share").get_by_role("slider")
        scroll_to(policy, "start")
        page.wait_for_timeout(1500)
        policy.focus()
        for _ in range(2):
            for _ in range(4):
                page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(1800)

        at("fairness")
        tab("Fairness")
        scroll_to(page.get_by_text("Approval rates by group"), "start")
        page.wait_for_timeout(4000)
        page.get_by_text("Approval rate among people who repaid").click()

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
