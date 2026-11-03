"""Capture README screenshots from a running app (streamlit run streamlit_app.py).

Run:  python scripts/screenshots.py [http://localhost:8501]
Needs Playwright with Chromium installed.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parent.parent / "docs"


def settle(page, ms=2500):
    page.wait_for_selector("text=Second Look", timeout=60_000)
    page.wait_for_timeout(ms)


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(URL)
        settle(page, 5000)
        page.screenshot(path=OUT / "queue.png", full_page=False)

        page.get_by_text("What if the loan terms change?").click()
        page.wait_for_timeout(1500)
        page.get_by_text("Officer decision").scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        page.screenshot(path=OUT / "explain.png", full_page=False)

        page.get_by_role("tab", name="Evidence").click()
        page.wait_for_timeout(3000)
        page.screenshot(path=OUT / "evidence.png")
        # Charts on the Evidence tab: 0 is the reason chart on the queue tab.
        policy_chart = page.locator(".stPlotlyChart").nth(1)
        policy_chart.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        policy_chart.screenshot(path=OUT / "policy.png")
        fairness_note = page.get_by_text("Every gender and age group is approved more often")
        fairness_note.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        page.screenshot(path=OUT / "fairness.png")
        browser.close()


if __name__ == "__main__":
    main()
