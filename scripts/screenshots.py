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
    page.wait_for_selector("text=Review queue", timeout=60_000)
    page.wait_for_selector(".stPlotlyChart", timeout=60_000)
    page.wait_for_timeout(ms)


def to_top(locator, ms=900):
    locator.evaluate("e => e.scrollIntoView({block: 'start'})")
    locator.page.wait_for_timeout(ms)


def tab(page, name, ms=2500):
    page.get_by_role("tab", name=name, exact=True).click()
    page.wait_for_timeout(ms)


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(URL)
        settle(page, 4000)
        page.screenshot(path=OUT / "queue.png")

        # A case the external score approves but Second Look flags: reasons, then the counter-offer.
        page.get_by_text("SL-1031", exact=True).click()
        page.wait_for_timeout(2500)
        to_top(page.locator(".sl-case").first)
        page.screenshot(path=OUT / "case.png")
        to_top(page.get_by_text("Each bar is the exact contribution"))
        page.screenshot(path=OUT / "explain.png")
        tab(page, "Counter-offer and what-if", 1500)
        to_top(page.locator(".sl-slip").first)
        page.screenshot(path=OUT / "counteroffer.png")

        tab(page, "Try an applicant")
        to_top(page.get_by_text("Score an applicant of your own"))
        page.screenshot(path=OUT / "try.png")

        tab(page, "Evidence", 3000)
        to_top(page.get_by_text("Set your own lending policy"), 1500)
        page.screenshot(path=OUT / "policy.png")

        tab(page, "Fairness", 3000)
        to_top(page.get_by_text("Approval rates by group"))
        page.screenshot(path=OUT / "fairness.png")
        browser.close()


if __name__ == "__main__":
    main()
