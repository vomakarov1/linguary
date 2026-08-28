"""Печатает сводку по каждому EPUB из набора тестовых фикстур.

Читает файлы как обычные zip-архивы (`zipfile` + `lxml`), не полагаясь на импортёр
проекта — на момент написания скрипта импортёра ещё нет, а сам набор фикстур собирается
именно для его будущей разработки.

Для каждого `*.epub` в этой директории печатает: имя, размер, версию EPUB (`package/@version`
из `content.opf`), число документов в spine, наличие nav (EPUB3) или toc.ncx (EPUB2), число
изображений в манифесте, число внутренних ссылок и число элементов курсива/полужирного.

Внутренняя ссылка здесь — это `<a href="...">`, чьё значение содержит `#` (есть фрагмент).
Реальные книги Project Gutenberg почти всегда ссылаются на сноски в форме
`href="chapter.xhtml#fn1"` (путь + фрагмент), а не голым `href="#fn1"` — оба варианта
одинаково остаются ссылками внутри самой книги, поэтому считаются вместе.

Перед разбором `content.opf` каждый файл проверяется на целостность через
`zipfile.ZipFile.testzip()` — битый или обрезанный архив в наборе фикстур обесценил бы
все тесты, которые на него полагаются.

Запуск:

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
        raise ValueError("META-INF/container.xml не содержит rootfile")
    full_path = rootfile.get("full-path")
    if not full_path:
        raise ValueError("rootfile без атрибута full-path")
    return full_path


def inspect_book(path: Path) -> BookReport:
    report = BookReport(name=path.name, size_bytes=path.stat().st_size, testzip_ok=False)

    try:
        with zipfile.ZipFile(path) as zf:
            bad_file = zf.testzip()
            report.testzip_ok = bad_file is None
            if bad_file is not None:
                report.error = f"повреждён файл внутри архива: {bad_file}"
                return report

            opf_path = find_opf_path(zf)
            opf_root = etree.fromstring(zf.read(opf_path))
            opf_dir = "/".join(opf_path.split("/")[:-1])

            report.epub_version = opf_root.get("version")

            manifest = opf_root.find(f"{{{OPF_NS}}}manifest")
            spine = opf_root.find(f"{{{OPF_NS}}}spine")
            if manifest is None or spine is None:
                report.error = "content.opf без manifest или spine"
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
                    report.warnings.append(f"документ манифеста не найден в архиве: {doc_path}")
                    continue
                try:
                    doc_root = etree.fromstring(doc_bytes, etree.XMLParser(recover=True))
                except etree.XMLSyntaxError as exc:
                    report.warnings.append(f"{href}: не удалось разобрать XHTML ({exc})")
                    continue
                if doc_root is None:
                    report.warnings.append(f"{href}: пустой результат разбора XHTML")
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
    lines = [f"=== {report.name} ({size_kb:.1f} КБ) ==="]
    lines.append(f"  testzip: {'OK' if report.testzip_ok else 'FAIL'}")
    if report.error:
        lines.append(f"  ОШИБКА: {report.error}")
        return "\n".join(lines)

    lines.append(f"  версия EPUB (package/@version): {report.epub_version}")
    lines.append(f"  документов в spine: {report.spine_doc_count}")
    lines.append(f"  nav/toc: {'есть' if report.has_nav_or_toc else 'нет'}")
    lines.append(f"  изображений в манифесте: {report.image_count}")
    lines.append(f"  внутренних ссылок (href с '#'): {report.internal_link_count}")
    lines.append(f"  элементов i/em/b/strong: {report.italic_bold_count}")
    for warning in report.warnings:
        lines.append(f"  предупреждение: {warning}")
    return "\n".join(lines)


def main() -> int:
    epub_files = sorted(FIXTURES_DIR.glob("*.epub"))
    if not epub_files:
        print(f"В {FIXTURES_DIR} не найдено ни одного *.epub", file=sys.stderr)
        return 1

    had_errors = False
    for path in epub_files:
        report = inspect_book(path)
        print(format_report(report))
        print()
        if report.error or not report.testzip_ok:
            had_errors = True

    print(f"Всего файлов: {len(epub_files)}")
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
