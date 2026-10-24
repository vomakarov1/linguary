"""Builds a small synthetic EPUB3 book for end-to-end testing.

The script creates `synthetic_story.epub` in the same directory: a short original English
story across three chapters, containing italics and bold text, one footnote (with internal link
and backlink), one small PNG image (generated inline, without external files), one list, and
one blockquote. Proper nouns and an invented term recur across chapters to allow testing
glossary assembly on the file.

Usage:

    uv run python tests/fixtures/epub/make_synthetic.py

Rebuilds the file deterministically — safe to run repeatedly.
"""

from __future__ import annotations

import struct
import zipfile
import zlib
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "synthetic_story.epub"

BOOK_ID = "urn:uuid:8f2b6e2a-3c9d-4e2e-9a2f-6a5c6b7d8e9f"
BOOK_TITLE = "The Lamp at Duskmere"
BOOK_AUTHOR = "Linguary Test Fixtures"
BOOK_LANGUAGE = "en"


def make_tiny_png() -> bytes:
    """Constructs a valid 8x8 PNG (indexed gradient) without third-party libraries."""

    width = height = 8

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB, no interlace

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type: none
        for x in range(width):
            raw += bytes((30 + x * 20, 40 + y * 20, 120))
    idat = zlib.compress(bytes(raw), level=9)

    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

STYLE_CSS = """body { font-family: serif; }
i, em { font-style: italic; }
b, strong { font-weight: bold; }
blockquote { margin-left: 2em; font-style: italic; }
.footnote { font-size: 0.9em; }
"""

CHAPTER_1 = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head>
  <title>Chapter 1: The Road to Duskmere</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <h1>Chapter 1: The Road to Duskmere</h1>
  <p><i>Wren Calder</i> had walked for three days when the hedgerows finally opened onto the
  valley of <b>Duskmere</b>, and the town looked smaller than every story had promised.</p>
  <p>A cart driver at the crossroads warned that the last mile was best walked before nightfall,
  since the marsh lights near Duskmere were known to lead travellers astray.</p>
  <p>Wren carried very little, but what was carried mattered: a knapsack held together by three
  kinds of stitching, and inside it, wrapped in wool, the <b>starglass</b><a epub:type="noteref"
  href="#fn1" id="fnref1">[1]</a> that had belonged to a grandmother nobody in the family liked
  to talk about.</p>
  <p>The knapsack held, in order:</p>
  <ol>
    <li>a brass pocket compass that pointed roughly north, on a good day;</li>
    <li>a heel of dried bread wrapped in cloth;</li>
    <li>a small tin lantern, unlit, saved for the marsh;</li>
    <li>the starglass itself, cold and faintly humming.</li>
  </ol>
  <p>By the time the road dipped toward the marsh, the sky over Duskmere had turned the colour of
  a bruise, and Wren Calder began, for the first time on the whole journey, to walk <i>quickly</i>.</p>
  <p>Somewhere ahead, a lantern answered the tin one before it was even lit, swinging once, twice,
  from a window that had no business being that high above the marsh.</p>
  <aside epub:type="footnote" id="fn1" class="footnote">
    <p>[1] A starglass is a hand-sized lens ground from marsh quartz; Duskmere folklore holds that
    it shows the holder not what is there, but what is about to be. <a href="#fnref1">↩</a></p>
  </aside>
</body>
</html>
"""

CHAPTER_2 = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <title>Chapter 2: The Keeper of Duskmere</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <h1>Chapter 2: The Keeper of Duskmere</h1>
  <p>The lantern belonged to Selene Row, who had kept the marsh gate of Duskmere for longer than
  anyone in town cared to count, and who did not seem at all surprised to see Wren Calder standing
  in the reeds with a starglass held out like a question.</p>
  <p>&#8220;You held it wrapped in wool,&#8221; Selene said, not quite a greeting. &#8220;That was
  the right instinct. It does not like to be looked at directly, not before Duskmere has looked at
  <i>you</i> first.&#8221;</p>
  <p>She led the way along a boardwalk that Wren had not noticed from the road, past a row of
  drowned fence posts and a single leaning marker carved with a symbol shaped like a lit window.</p>
  <figure>
    <img src="images/marker.png" alt="A small carved marker shaped like a lit window, drawn in blue and orange tones."/>
    <figcaption>The marker at the edge of the boardwalk into Duskmere.</figcaption>
  </figure>
  <p>Inside the gatehouse, Selene Row set two cups of something hot and unfamiliar on the table and
  asked, plainly, what a starglass from that particular family was doing so far from home.</p>
  <p>Wren answered with the only true thing there was to say: that the glass had started humming
  three nights before anyone in the family died, every single time, for four generations, and that
  this time it had started humming for <b>no one</b> that Wren could name.</p>
  <p>Selene Row did not look away from the starglass again for a long while, and when she finally
  spoke, her voice had lost its dry, gatekeeper's edge entirely.</p>
</body>
</html>
"""

CHAPTER_3 = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <title>Chapter 3: The Light Beneath the Hollow</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <h1>Chapter 3: The Light Beneath the Hollow</h1>
  <p>Duskmere had a saying about the starglass, and Selene Row recited it the way people recite
  things they had rather not believe:</p>
  <blockquote>
    <p>What the glass has not yet shown you, it is already carrying for someone else.</p>
  </blockquote>
  <p>Wren Calder turned the starglass over once in the lantern light, and for the first time on the
  whole journey, it did not feel cold at all.</p>
  <p>Selene explained, in the plain way of someone who had explained it before, that a starglass
  humming for no one meant only that the someone had not arrived at Duskmere <i>yet</i>, and that
  the marsh gate had a long habit of waiting exactly as long as it took.</p>
  <p>So Wren Calder did the only sensible thing left: unwrapped the starglass fully, set it on the
  gatehouse sill facing the marsh, and waited with Selene Row for whoever the light was for.</p>
  <p>Somewhere out past the drowned fence posts, a second lantern answered the tin one, swinging
  once, twice, exactly the way the first one had.</p>
  <p>Neither of them moved to open the door. In Duskmere, Selene said, you let the marsh finish its
  sentence before you answer it, and that, more than any starglass, was the only thing worth
  <b>knowing</b> about the town at all.</p>
</body>
</html>
"""

NAV_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head><title>Navigation</title></head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Contents</h1>
    <ol>
      <li><a href="chapter1.xhtml">Chapter 1: The Road to Duskmere</a></li>
      <li><a href="chapter2.xhtml">Chapter 2: The Keeper of Duskmere</a></li>
      <li><a href="chapter3.xhtml">Chapter 3: The Light Beneath the Hollow</a></li>
    </ol>
  </nav>
</body>
</html>
"""

CONTENT_OPF = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">{BOOK_ID}</dc:identifier>
    <dc:title>{BOOK_TITLE}</dc:title>
    <dc:creator>{BOOK_AUTHOR}</dc:creator>
    <dc:language>{BOOK_LANGUAGE}</dc:language>
    <dc:rights>Synthetic text written for the Linguary test suite; not a public-domain
    reproduction.</dc:rights>
    <meta property="dcterms:modified">2026-08-22T00:00:00Z</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter3" href="chapter3.xhtml" media-type="application/xhtml+xml"/>
    <item id="style" href="style.css" media-type="text/css"/>
    <item id="marker-image" href="images/marker.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
    <itemref idref="chapter2"/>
    <itemref idref="chapter3"/>
  </spine>
</package>
"""


def build() -> None:
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with zipfile.ZipFile(OUTPUT_PATH, "w") as epub:
        # mimetype comes first and uncompressed, as required by the EPUB specification.
        epub.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED)

        def add(name: str, content: str | bytes) -> None:
            data = content.encode("utf-8") if isinstance(content, str) else content
            epub.writestr(name, data, zipfile.ZIP_DEFLATED)

        add("META-INF/container.xml", CONTAINER_XML)
        add("OEBPS/content.opf", CONTENT_OPF)
        add("OEBPS/nav.xhtml", NAV_XHTML)
        add("OEBPS/style.css", STYLE_CSS)
        add("OEBPS/chapter1.xhtml", CHAPTER_1)
        add("OEBPS/chapter2.xhtml", CHAPTER_2)
        add("OEBPS/chapter3.xhtml", CHAPTER_3)
        add("OEBPS/images/marker.png", make_tiny_png())

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Built: {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
