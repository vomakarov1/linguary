#!/usr/bin/env python3
"""Fetch and freeze the ten Project Gutenberg passages used by the experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


SOURCE_URL = "https://www.gutenberg.org/cache/epub/1952/pg1952.txt"
BOOK_URL = "https://www.gutenberg.org/ebooks/1952"
SOURCE_SHA256 = "20c7788eeae2fe3a0efd51a1f08b6723366a4be0778d65db219149a3522651d4"
SOURCE_UPDATED = "2026-09-16"
OUTPUT = Path(__file__).with_name("cases.json")


CASES = (
    {
        "id": "yw-01",
        "paragraph_range": [8, 11],
        "expected_fact": "The narrator suspects John's medical beliefs may be slowing her recovery.",
        "reference_quotes": ["perhaps that is one reason I do not get well faster."],
    },
    {
        "id": "yw-02",
        "paragraph_range": [12, 18],
        "expected_fact": "The narrator believes congenial work would help her.",
        "reference_quotes": [
            "Personally, I believe that congenial work, with excitement and change, would do me good."
        ],
    },
    {
        "id": "yw-03",
        "paragraph_range": [32, 36],
        "expected_fact": "The room was previously a nursery or playground, as shown by the barred windows and rings.",
        "reference_quotes": [
            "It was nursery first and then playground and gymnasium, I should judge; for the windows are barred for little children, and there are rings and things in the walls."
        ],
    },
    {
        "id": "yw-04",
        "paragraph_range": [36, 39],
        "expected_fact": "The wallpaper's pattern is difficult to follow and irritates the narrator.",
        "reference_quotes": [
            "It is dull enough to confuse the eye in following, pronounced enough to constantly irritate, and provoke study, and when you follow the lame, uncertain curves for a little distance they suddenly commit suicide—plunge off at outrageous angles, destroy themselves in unheard-of contradictions."
        ],
    },
    {
        "id": "yw-05",
        "paragraph_range": [48, 51],
        "expected_fact": "The narrator feels she has become a burden to John.",
        "reference_quotes": [
            "I meant to be such a help to John, such a real rest and comfort, and here I am a comparative burden already!"
        ],
    },
    {
        "id": "yw-06",
        "paragraph_range": [62, 64],
        "expected_fact": "John tells the narrator to treat the people she thinks she sees as products of her imagination.",
        "reference_quotes": [
            "John has cautioned me not to give way to fancy in the least.",
            "He says that with my imaginative power and habit of story-making a nervous weakness like mine is sure to lead to all manner of excited fancies, and that I ought to use my will and good sense to check the tendency."
        ],
    },
    {
        "id": "yw-07",
        "paragraph_range": [154, 157],
        "expected_fact": "At night, the narrator sees the wallpaper's outer pattern as bars with a woman behind it.",
        "reference_quotes": [
            "At night in any kind of light, in twilight, candlelight, lamplight, and worst of all by moonlight, it becomes bars! The outside pattern I mean, and the woman behind it is as plain as can be."
        ],
    },
    {
        "id": "yw-08",
        "paragraph_range": [161, 165],
        "expected_fact": "The narrator has begun to fear John.",
        "reference_quotes": ["The fact is, I am getting a little afraid of John."],
    },
    {
        "id": "yw-09",
        "paragraph_range": [190, 193],
        "expected_fact": "The narrator believes the woman behind the wallpaper is shaking its front pattern.",
        "reference_quotes": ["The front pattern does move—and no wonder! The woman behind shakes it!"],
    },
    {
        "id": "yw-10",
        "paragraph_range": [238, 255],
        "expected_fact": "The narrator refuses to leave the room or let anyone enter until John returns.",
        "reference_quotes": [
            "I don’t want to go out, and I don’t want to have anybody come in, till John comes."
        ],
    },
)


def fetch_source() -> bytes:
    request = Request(SOURCE_URL, headers={"User-Agent": "linguary-model-source-anchors/1.0"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    actual = sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        raise SystemExit(
            f"Project Gutenberg source changed: expected SHA-256 {SOURCE_SHA256}, got {actual}; "
            "review the new source edition before updating the benchmark."
        )
    return raw


def source_paragraphs(raw: bytes) -> list[str]:
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start < 0 or end <= start:
        raise SystemExit("Could not find Project Gutenberg book boundary markers.")
    body = text[start:end]
    return [part.strip() for part in re.split(r"\n(?:[ \t]*\n)+", body) if part.strip()]


def freeze_cases(raw: bytes) -> dict[str, object]:
    paragraphs = source_paragraphs(raw)
    frozen: list[dict[str, object]] = []
    for case in CASES:
        first, stop = case["paragraph_range"]
        selected = paragraphs[first:stop]
        if len(selected) != stop - first:
            raise SystemExit(f"Source excerpt range is incomplete for {case['id']}.")
        text = "\n\n".join(
            " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
            for paragraph in selected
        )
        reference_quotes = []
        for quote in case["reference_quotes"]:
            offsets = [match.start() for match in re.finditer(re.escape(quote), text)]
            if not offsets:
                raise SystemExit(f"Reference quote is missing from {case['id']}: {quote}")
            reference_quotes.append({"text": quote, "offsets": offsets})
        frozen.append(
            {
                "id": case["id"],
                "block_id": "b1",
                "text": text,
                "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "expected_anchor_count": 1,
                "expected_fact": case["expected_fact"],
                "reference_quotes": reference_quotes,
                "source_paragraph_range": case["paragraph_range"],
            }
        )
    return {
        "schema_version": 1,
        "source": {
            "title": "The Yellow Wallpaper",
            "author": "Charlotte Perkins Gilman",
            "project_gutenberg_ebook_id": 1952,
            "ebook_url": BOOK_URL,
            "plain_text_url": SOURCE_URL,
            "source_updated": SOURCE_UPDATED,
            "source_sha256": sha256(raw).hexdigest(),
            "paragraph_indexing": "0-based paragraphs starting with the opening Project Gutenberg marker paragraph",
            "extraction": "Collapsed source line wraps to spaces within each paragraph; retained paragraph breaks as blank lines.",
        },
        "cases": frozen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = freeze_cases(fetch_source())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(manifest['cases'])} frozen cases at {args.output}.")


if __name__ == "__main__":
    main()
