# Test EPUB set

Fixtures for developing and checking the EPUB importer/exporter (phase 2, PRD § 28.1). Four
public-domain books from Project Gutenberg (US) and one small synthetic
book, which the script in this same directory rebuilds.

All Gutenberg files were downloaded via `curl -L https://www.gutenberg.org/ebooks/{id}.epub...` and
checked not by the URL name (Gutenberg changes the delivery format), but by the actual contents of
`content.opf` — the EPUB version and the manifest listing below were confirmed by `inspect_fixtures.py`, not by
assumption.

## Contents

| File | Book | Author | Gutenberg ID | License | Size |
|---|---|---|---|---|---|
| `great_gatsby.epub` | The Great Gatsby | F. Scott Fitzgerald | [64317](https://www.gutenberg.org/ebooks/64317) | Public domain in the USA | ~345 KB |
| `beowulf.epub` | Beowulf: An Anglo-Saxon Epic Poem (trans. J. Lesslie Hall) | anonymous, trans. J. Lesslie Hall | [16328](https://www.gutenberg.org/ebooks/16328) | Public domain in the USA | ~285 KB |
| `alice_rackham.epub` | Alice's Adventures in Wonderland, illustrated by Arthur Rackham | Lewis Carroll | [28885](https://www.gutenberg.org/ebooks/28885) | Public domain in the USA | ~1.4 MB |
| `dracula.epub` | Dracula | Bram Stoker | [345](https://www.gutenberg.org/ebooks/345) | Public domain in the USA | ~588 KB |
| `synthetic_story.epub` | The Lamp at Duskmere (synthetic) | Linguary Test Fixtures | — | synthetic text, no license required | ~5 KB |

The whole set is about 2.6 MB; each file stays within ~1.5 MB (the task budget is up to ~3 MB per file
and ~15 MB for the set).

## Which profile each file covers

Profiles come from PRD § 28.1. The figures below are the actual output of `inspect_fixtures.py` at the time the
set was built (the full output is in the P2.T4 task report).

### `great_gatsby.epub` — a plain EPUB3-novel
- `package/@version = 3.0`.
- Ordinary chapters (Roman numerals I–IX) are marked by headings `<h2>` **inside one shared
  XHTML-document** of the book body — Gutenberg assembled the book as 3 spine-documents (title, body,
  closing matter), not one document per chapter. This is a normal, common form of an edition:
  the importer must be able to split the book into chapters by headings, not only along
  spine-document boundaries.
- A short novel (~50 thousand words) is a convenient baseline file for the first import runs.

### `beowulf.epub` — EPUB2 plus footnotes and internal links (+ poetry, a bonus profile)
- `package/@version = 2.0` — the only file in the set of this exact version; the EPUB2 profile
  is covered by it.
- Extensive footnotes following Gutenberg: a pair of links `id="N.FNREF.k"` →
  `href="....xhtml#N.FNDEF.k"` (jump to the footnote) and a return link `href="....xhtml#N.FNREF.k"`
  inside the footnote text (jump back to the quoted place). Note: Gutenberg almost
  always writes `href="файл.xhtml#якорь"` (path + fragment), rather than a bare `href="#якорь"` — even
  for links inside the same document. The importer must resolve both variants.
- The text is a verse translation of the epic and keeps the line and stanza layout — it covers
  the desired (not required) “poetry” profile from PRD § 28.1.

### `alice_rackham.epub` — a book with images
- 38 raster images in the manifest (illustrations by Arthur Rackham), not only the cover.
- `package/@version = 3.0`.
- The heaviest file in the set (~1.4 MB): almost all of the weight is images, and the text
  part is small. If the importer must not embed copies of every resource at once, keep
  that in mind on the first runs of this file.

### `dracula.epub` — complex inline formatting (+ dialogue, a bonus profile)
- 502 elements `<i>/<em>/<b>/<strong>` — italics are used for thoughts, titles of publications,
  foreign phrases, `<strong>` — for separate emphasis in diary entries.
- Epistolary form: chapters are letters, diary entries, telegrams, and shorthand
  notes of several narrators, so one book places different voices side by side, along with
  date and signature markup. A good check that the importer does not treat the whole book as a single
  stream of neutral prose.
- There is a lot of direct speech — it covers the desired “abundance of dialogue” profile from PRD § 28.1.

## Synthetic book

`synthetic_story.epub` — “The Lamp at Duskmere”, a short original story in English
(not taken from any other work; these fixtures are the author), assembled by the script `make_synthetic.py` from this
same directory. Rebuild:

```sh
uv run python tests/fixtures/epub/make_synthetic.py
```

The script deterministically overwrites `synthetic_story.epub`. Structure:

- valid EPUB3: `mimetype` first and uncompressed, `META-INF/container.xml`, `OEBPS/content.opf`,
  `OEBPS/nav.xhtml`;
- 3 chapters (`chapter1.xhtml` — `chapter3.xhtml`) of 6 paragraphs of English text — a coherent
  short story, not `lorem ipsum`;
- italics (`<i>`) and bold (`<b>`) in every chapter;
- one footnote in chapter 1 (`epub:type="footnote"`) with a link to it (`epub:type="noteref"`) and
  a back link `↩` from the footnote text — both forms use a bare `href="#id"`, unlike
  the Gutenberg files above, where links always carry the file name;
  the importer must resolve both styles the same way;
- one small image (`images/marker.png`, 8×8 PNG, generated in the script itself through
  `zlib`/`struct`, with no external files or libraries);
- one list (`<ol>` in chapter 1) and one quotation (`<blockquote>` in chapter 3);
- proper names and an invented term that repeat across chapters, to check glossary assembly:
  character **Wren Calder**, place **Duskmere**, invented object **starglass**.

The finished size is about 5 KB, as the fast end-to-end test required.

## Checking the set

`inspect_fixtures.py` prints for each `*.epub` in this directory: name, size, EPUB version
(`package/@version`), the number of spine documents, whether nav/toc is present, the number of manifest images,
the number of internal links (`<a href="...">`, where the value contains `#`) and the number of elements
`<i>/<em>/<b>/<strong>`. Before `content.opf` is read, each archive is checked for integrity
through `zipfile.ZipFile.testzip()`.

```sh
uv run python tests/fixtures/epub/inspect_fixtures.py
```

The script reads the files directly as zip-archives (`zipfile` + `lxml`) and does not rely on the project's importer —
it had not been written yet when this set was assembled.

The test “every file in the set imports without `FATAL`” (PRD § 28.1) is not part of this package —
it will be added by the package that writes the importer of EPUB.
