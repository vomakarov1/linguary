"""Prints a summary for each EPUB in the test fixture suite.

Reads files as standard zip archives (`zipfile` + `lxml`) without relying on the project's
importer — at the time this script was written, the importer did not exist yet, and the fixture
suite itself was assembled specifically for its future development.

For each `*.epub` in this directory, prints: name, size, EPUB version (`package/@version`
from `content.opf`), number of spine documents, presence of nav (EPUB3) or toc.ncx (EPUB2),
number of manifest images, number of internal links, and number of italic/bold elements.

An internal link here is `<a href="...">` containing `#` (has a fragment). Real books from
Project Gutenberg almost always reference footnotes as `href="chapter.xhtml#fn1"` (path + fragment)
rather than bare `href="#fn1"` — both remain internal links within the book, and are counted together.

Prior to parsing `content.opf`, each file is checked for integrity via `zipfile.ZipFile.testzip()` —
a corrupted or truncated archive would invalidate all tests relying on it.

Usage:

    uv run python tests/fixtures/epub/inspect_fixtures.py
"""

from __future__ import annotations

import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

FIXTURES_DIR = Path(__file__).parent

OPF_NS = "http://www.idpf.org/2007/opf"
XHTML_NS = "http://www.w3.org/1999/xhtml"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


@dataclass
class BookReport:
    name: str
    size_bytes: int
    testzip_ok: bool
    epub_version: str | None = None
    spine_doc_count: int | None = None
    has_nav_or_toc: bool = False
    image_count: int = 0
    internal_link_count: int = 0
    italic_bold_count: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def find_opf_path(zf: zipfile.ZipFile) -> str:
    container = etree.fromstring(zf.read("META-INF/container.xml"))
    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None:
        raise ValueError("META-INF/container.xml does not contain rootfile")
    full_path = rootfile.get("full-path")
    if not full_path:
        raise ValueError("rootfile has no full-path attribute")
    return full_path


def inspect_book(path: Path) -> BookReport:
    report = BookReport(name=path.name, size_bytes=path.stat().st_size, testzip_ok=False)

    try:
        with zipfile.ZipFile(path) as zf:
            bad_file = zf.testzip()
            report.testzip_ok = bad_file is None
            if bad_file is not None:
                report.error = f"corrupted file inside archive: {bad_file}"
                return report

            opf_path = find_opf_path(zf)
            opf_root = etree.fromstring(zf.read(opf_path))
            opf_dir = "/".join(opf_path.split("/")[:-1])

            report.epub_version = opf_root.get("version")

            manifest = opf_root.find(f"{{{OPF_NS}}}manifest")
            spine = opf_root.find(f"{{{OPF_NS}}}spine")
            if manifest is None or spine is None:
                report.error = "content.opf missing manifest or spine"
                return report

            items_by_id = {
                item.get("id"): item for item in manifest.findall(f"{{{OPF_NS}}}item")
            }

            report.spine_doc_count = len(spine.findall(f"{{{OPF_NS}}}itemref"))

            has_nav = any(
                "nav" in (item.get("properties") or "").split()
                for item in items_by_id.values()
            )
            has_ncx_toc = spine.get("toc") is not None
            report.has_nav_or_toc = has_nav or has_ncx_toc

            report.image_count = sum(
                1
                for item in items_by_id.values()
                if (item.get("media-type") or "").startswith("image/")
            )

            html_hrefs = [
                item.get("href")
                for item in items_by_id.values()
                if (item.get("media-type") or "") in ("application/xhtml+xml", "text/html")
            ]

            for href in html_hrefs:
                doc_path = f"{opf_dir}/{href}" if opf_dir else href
                try:
                    doc_bytes = zf.read(doc_path)
                except KeyError:
                    report.warnings.append(f"manifest document not found in archive: {doc_path}")
                    continue
                try:
                    doc_root = etree.fromstring(doc_bytes, etree.XMLParser(recover=True))
                except etree.XMLSyntaxError as exc:
                    report.warnings.append(f"{href}: failed to parse XHTML ({exc})")
                    continue
                if doc_root is None:
                    report.warnings.append(f"{href}: empty XHTML parse result")
                    continue

                for anchor in doc_root.iter():
                    tag = etree.QName(anchor).localname if isinstance(anchor.tag, str) else None
                    if tag == "a":
                        href_value = anchor.get("href") or ""
                        if "#" in href_value:
                            report.internal_link_count += 1
                    elif tag in ("i", "em", "b", "strong"):
                        report.italic_bold_count += 1

    except (KeyError, ValueError, etree.XMLSyntaxError, zipfile.BadZipFile) as exc:
        report.error = str(exc)

    return report


def format_report(report: BookReport) -> str:
    size_kb = report.size_bytes / 1024
    lines = [f"=== {report.name} ({size_kb:.1f} KB) ==="]
    lines.append(f"  testzip: {'OK' if report.testzip_ok else 'FAIL'}")
    if report.error:
        lines.append(f"  ERROR: {report.error}")
        return "\n".join(lines)

    lines.append(f"  EPUB version (package/@version): {report.epub_version}")
    lines.append(f"  documents in spine: {report.spine_doc_count}")
    lines.append(f"  nav/toc: {'yes' if report.has_nav_or_toc else 'no'}")
    lines.append(f"  images in manifest: {report.image_count}")
    lines.append(f"  internal links (href with '#'): {report.internal_link_count}")
    lines.append(f"  i/em/b/strong elements: {report.italic_bold_count}")
    for warning in report.warnings:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


def main() -> int:
    epub_files = sorted(FIXTURES_DIR.glob("*.epub"))
    if not epub_files:
        print(f"No *.epub files found in {FIXTURES_DIR}", file=sys.stderr)
        return 1

    had_errors = False
    for path in epub_files:
        report = inspect_book(path)
        print(format_report(report))
        print()
        if report.error or not report.testzip_ok:
            had_errors = True

    print(f"Total files: {len(epub_files)}")
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
