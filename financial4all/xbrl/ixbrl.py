# financial4all/xbrl/ixbrl.py
"""
Inline XBRL (iXBRL) extraction from HTML documents.

Extracts XBRL facts, contexts, and units from HTML files that embed
Inline XBRL markup (ix:nonFraction, ix:nonNumeric, ix:context, ix:unit)
and builds a minimal XBRL instance document for parsing by the main XBRL parser.
"""

from typing import Optional, Union

from financial4all.core import log

# iXBRL namespace
IXBRL_NS = "http://www.xbrl.org/2013/inlineXBRL"
XBRLI_NS = "http://www.xbrl.org/2003/instance"
XHTML_NS = "http://www.w3.org/1999/xhtml"


def extract_ixbrl_to_xml(html_content: Union[str, bytes]) -> Optional[str]:
    """
    Extract XBRL from Inline XBRL HTML and return as standalone XML instance.

    Parses HTML, finds ix:context, ix:unit, ix:nonFraction, ix:nonNumeric
    elements, and builds a minimal xbrl:instance document compatible with
    XBRL.from_xml().

    Args:
        html_content: Raw HTML content (string or bytes)

    Returns:
        XML string of XBRL instance document, or None if extraction fails
    """
    try:
        from lxml import etree
    except ImportError:
        log.warning("lxml required for iXBRL extraction. Install lxml.")
        return None

    if isinstance(html_content, bytes):
        html_content = html_content.decode("utf-8", errors="replace")

    try:
        parser = etree.HTMLParser(recover=True, encoding="utf-8")
        root = etree.fromstring(html_content.encode("utf-8"), parser)
    except Exception as e:
        log.debug(f"Failed to parse HTML for iXBRL: {e}")
        return None

    # Namespace map for finding elements
    ns = {"ix": IXBRL_NS, "xbrli": XBRLI_NS}

    # Collect contexts - ix:context or embedded context elements
    contexts = []
    for ctx in root.iter(f"{{{IXBRL_NS}}}context"):
        try:
            ctx_str = etree.tostring(ctx, encoding="unicode", method="xml")
            contexts.append(ctx_str)
        except Exception:
            continue

    # Also find xbrli:context (sometimes used)
    for ctx in root.iter(f"{{{XBRLI_NS}}}context"):
        try:
            ctx_str = etree.tostring(ctx, encoding="unicode", method="xml")
            if ctx_str not in str(contexts):
                contexts.append(ctx_str)
        except Exception:
            continue

    # Collect units
    units = []
    for unit in root.iter(f"{{{IXBRL_NS}}}unit"):
        try:
            unit_str = etree.tostring(unit, encoding="unicode", method="xml")
            units.append(unit_str)
        except Exception:
            continue
    for unit in root.iter(f"{{{XBRLI_NS}}}unit"):
        try:
            unit_str = etree.tostring(unit, encoding="unicode", method="xml")
            if unit_str not in str(units):
                units.append(unit_str)
        except Exception:
            continue

    # Collect numeric and non-numeric facts
    facts = []
    for elem in root.iter():
        tag = elem.tag
        if not isinstance(tag, str):
            continue
        if f"{{{IXBRL_NS}}}" in tag:
            local = tag.split("}")[-1] if "}" in tag else tag
            if local in ("nonFraction", "nonNumeric", "fraction", "denominator", "numerator"):
                try:
                    fact_str = etree.tostring(elem, encoding="unicode", method="xml")
                    facts.append(fact_str)
                except Exception:
                    continue

    if not facts and not contexts:
        log.debug("No iXBRL content found in HTML")
        return None

    # Build minimal XBRL instance document
    ns_decls = [
        'xmlns="http://www.xbrl.org/2003/instance"',
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"',
        'xmlns:xbrli="http://www.xbrl.org/2003/instance"',
        'xmlns:xlink="http://www.w3.org/1999/xlink"',
    ]

    parts = [f'<?xml version="1.0" encoding="UTF-8"?>']
    parts.append(f'<xbrl {" ".join(ns_decls)}>')

    for ctx in contexts:
        parts.append(ctx)

    for unit in units:
        parts.append(unit)

    for fact in facts:
        parts.append(fact)

    parts.append("</xbrl>")

    return "\n".join(parts)
