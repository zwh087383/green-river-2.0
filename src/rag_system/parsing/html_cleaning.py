from __future__ import annotations

from pathlib import Path


def clean_html_file(path: Path) -> str:
    """Extract a cleaner HTML body for document parsing."""

    html = path.read_text(encoding="utf-8", errors="ignore")
    return clean_html(html)


def clean_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html

    soup = BeautifulSoup(html, "html.parser")
    for selector in [
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "[role='navigation']",
        "[aria-label='breadcrumb']",
        ".breadcrumb",
        ".breadcrumbs",
        ".toc",
        ".toctree-wrapper",
        ".sphinxsidebar",
    ]:
        for node in soup.select(selector):
            node.decompose()

    main = soup.find("main") or soup.find("article") or _densest_body_node(soup)
    if main is None:
        main = soup.body or soup
    return str(main)


def _densest_body_node(soup: object) -> object | None:
    body = getattr(soup, "body", None)
    if body is None:
        return None

    candidates = []
    for node in body.find_all(["main", "article", "section", "div"], recursive=True):
        text = node.get_text(" ", strip=True)
        if len(text) < 200:
            continue
        link_text = " ".join(link.get_text(" ", strip=True) for link in node.find_all("a"))
        density = len(text) - len(link_text) * 1.5
        candidates.append((density, node))

    if not candidates:
        return body
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
