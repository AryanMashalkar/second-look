"""Mix the screen recording, narration and captions into the final demo video.

Run after make_voiceover.py and record.py:  python video/assemble.py
Writes docs/second-look-demo.mp4. Needs ffmpeg on PATH.
"""

import json
import subprocess
from pathlib import Path

BUILD = Path(__file__).resolve().parent / "build"
OUT = BUILD.parent.parent / "docs" / "second-look-demo.mp4"
CAPTION_STYLE = (
    "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,BorderStyle=3,"
    "OutlineColour=&H80000000,Outline=1,Shadow=0,MarginV=36"
)


def main():
    meta = json.loads((BUILD / "segments.json").read_text())
    offset = (BUILD / "offset.txt").read_text().strip()
    segments = meta["segments"]

    inputs = ["-ss", offset, "-i", "screen.webm"]
    chains = []
    for i, seg in enumerate(segments, 1):
        inputs += ["-i", seg["file"]]
        ms = int(seg["start"] * 1000)
        chains.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(segments) + 1))
    graph = ";".join(
        [
            "[0:v]scale=1280:720,fps=30,tpad=stop_mode=clone:stop_duration=10,"
            f"subtitles=captions.srt:force_style='{CAPTION_STYLE}'[v]",
            *chains,
            f"{mix}amix=inputs={len(segments)}:normalize=0,apad[a]",
        ]
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
         "-t", f"{meta['total'] + 1:.2f}", "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(OUT)],
        cwd=BUILD, check=True,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
