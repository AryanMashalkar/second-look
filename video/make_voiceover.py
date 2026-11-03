"""Narration for the demo video: one audio clip per scene, plus sentence-timed captions.

Run:  python video/make_voiceover.py
Writes video/build/seg_XX.mp3, video/build/segments.json and video/build/captions.srt.
The voice is synthetic and disclosed in the README: ElevenLabs when ELEVENLABS_API_KEY is set,
otherwise Microsoft neural TTS via edge-tts.
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import edge_tts

ELEVEN_VOICE = "cjVigY5qzO86Huf0OWal"  # Eric - Smooth, Trustworthy (premade library voice)
ELEVEN_MODEL = "eleven_multilingual_v2"
VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+4%"
GAP = 0.5  # seconds of silence after each scene
BUILD = Path(__file__).resolve().parent / "build"

# Written for the ear: short sentences, numbers as people say them.
SCENES = [
    ("title", "One in seven loan applicants in our data has no credit bureau record. "
              "Lenders still judge them with an outside credit score, and for these people that score is at its weakest. "
              "Second Look gives them a second opinion."),
    ("overview", "It's a review queue for loan officers. "
                 "On eight thousand eight hundred real applicants the model never saw, it ranks risk better than the external score. "
                 "And at the same default rate, it approves six hundred and sixty more people per ten thousand. "
                 "Six hundred and ten of them paid back."),
    ("approve", "Here's someone the external score would decline. "
                "Second Look puts their risk at two and a half percent, and approves. "
                "The chart says why. Sixteen years in the same job, higher education, "
                "and a hundred and nine instalments already paid to this lender. "
                "Green bars lower the risk, red bars raise it, and every bar is the model's actual math."),
    ("caution", "Now the other way round. The external score approves this applicant, but Second Look sees only two years in the current job, "
                "and earlier loans granted at about half of what was asked. "
                "So instead of a flat no, it suggests a counter-offer: a ten percent smaller loan brings the risk under the line. "
                "It only changes the loan terms, never who the person is."),
    ("whatif", "The officer can try other terms and watch the risk update live."),
    ("decide", "But the model never decides. The officer picks an action, writes a reason, "
               "and every decision lands in a log they can export."),
    ("evidence", "The Evidence tab shows the receipts. Every number comes from real held-out applicants, with confidence intervals. "
                 "At every approval rate, Second Look's approved group defaults less often. "
                 "And an ablation shows most of the gain comes from the lender's own repayment records."),
    ("fairness", "Gender and age are not inputs, but we audit them anyway. "
                 "Every group gets approved more often. The age gap does get wider, and we report that instead of hiding it."),
    ("close", "Second Look. Built in forty-eight hours with LightGBM and Streamlit, on real Home Credit data. "
              "The code, the tests and every metric are on GitHub. Thanks for watching."),
]


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def synth_elevenlabs(text: str, path: Path, key: str):
    """ElevenLabs with character timestamps; returns sentence boundaries as (start, end, text)."""
    body = json.dumps({
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.15, "use_speaker_boost": True},
    }).encode()
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/with-timestamps?output_format=mp3_44100_128",
        data=body, headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    path.write_bytes(base64.b64decode(data["audio_base64"]))
    al = data["alignment"]
    starts, ends = al["character_start_times_seconds"], al["character_end_times_seconds"]
    sentences = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        a, b = m.start(), m.end() - 1
        while a < b and text[a].isspace():
            a += 1
        if a >= len(starts) or not text[a:b + 1].strip():
            continue
        sentences.append((starts[a], ends[min(b, len(ends) - 1)], text[a:b + 1].strip()))
    return sentences


async def synth(text: str, path: Path):
    """Write the audio and return sentence boundaries as (start, end, text)."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        return synth_elevenlabs(text, path, key)
    comm = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="SentenceBoundary")
    sentences = []
    with path.open("wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "SentenceBoundary":
                start = chunk["offset"] / 1e7
                sentences.append((start, start + chunk["duration"] / 1e7, chunk["text"]))
    return sentences


def split_caption(text: str, start: float, end: float, max_chars=70):
    """Break a long sentence into caption-sized pieces, timed by character share."""
    words, pieces, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            pieces.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    pieces.append(line)
    total = sum(len(p) for p in pieces)
    t, out = start, []
    for p in pieces:
        dt = (end - start) * len(p) / total
        out.append((t, t + dt, p))
        t += dt
    return out


async def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    segments, captions, t = [], [], 0.0
    for i, (name, text) in enumerate(SCENES):
        path = BUILD / f"seg_{i:02}.mp3"
        sentences = await synth(text, path)
        dur = duration(path)
        for s, e, txt in sentences:
            captions.extend(split_caption(txt, t + s, t + e))
        segments.append({"name": name, "file": path.name, "start": round(t, 3), "duration": round(dur, 3)})
        print(f"{name:10s} {dur:5.1f}s")
        t += dur + GAP
    (BUILD / "segments.json").write_text(json.dumps({"total": round(t, 3), "gap": GAP, "segments": segments}, indent=1))
    with (BUILD / "captions.srt").open("w", encoding="utf-8") as f:
        for n, (s, e, txt) in enumerate(captions, 1):
            f.write(f"{n}\n{srt_time(s)} --> {srt_time(e)}\n{txt}\n\n")
    print(f"total {t:.1f}s, {len(captions)} captions")


if __name__ == "__main__":
    asyncio.run(main())
