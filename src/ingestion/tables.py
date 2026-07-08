"""
RAG Table Reformatter
=======================
Converts HTML table strings (output of ``DocumentParser``) into natural-
language prose optimised for semantic embedding and retrieval.

Two-strategy approach
---------------------
1. **Rule-based** (fast) — for simple tables (≤ configurable max rows×cols):
   generates a structured sentence per row using column headers as labels.

2. **LLM-based** (thorough) — for complex tables with merged cells, nested
   headers, or dense numeric content: asks the Groq LLM to produce a concise
   prose summary that preserves all key information.

Both outputs are stored alongside the original HTML in the returned
``ReformattedTable`` so downstream components can choose which to embed.

Haystack 2.x contract
----------------------
``run(html_tables: list[str]) -> {"tables": list[ReformattedTable]}``

Usage::

    from src.ingestion.tables import TableReformatter

    reformatter = TableReformatter()
    result = reformatter.run(html_tables=["<table>...</table>"])
    for tbl in result["tables"]:
        print(tbl.natural_language)
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from haystack import component
from haystack.dataclasses import ChatMessage

from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector
from src.utils.groq_client import RotatableGroqGenerator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output DTO
# ---------------------------------------------------------------------------


@dataclass
class ReformattedTable:
    """
    All representations of a single table.

    Attributes:
        html:              Original HTML string.
        natural_language:  Primary NL representation (rule-based or LLM).
        llm_summary:       Richer LLM-generated summary (may be None if
                           rule-based was sufficient).
        num_rows:          Data rows (excluding header).
        num_cols:          Number of columns.
        headers:           Parsed column header strings.
        strategy_used:     ``"rule_based"`` or ``"llm"``.
        metadata:          Arbitrary extra attributes.
    """

    html: str
    natural_language: str
    llm_summary: str | None = None
    num_rows: int = 0
    num_cols: int = 0
    headers: list[str] = field(default_factory=list)
    strategy_used: str = "rule_based"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_text(self) -> str:
        """Return the richest available text representation."""
        return self.llm_summary or self.natural_language


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class TableReformatter:
    """
    Converts HTML tables to natural language.

    Args:
        llm_threshold_cells: Tables with more than this many cells
                             (rows × cols) are sent to the LLM.
        llm_model:           Groq model for LLM-based reformatting.
        max_llm_table_chars: HTML tables larger than this character count
                             are truncated before being sent to the LLM.
        use_llm:             Master switch — set ``False`` to disable LLM
                             even for complex tables.
    """

    def __init__(
        self,
        llm_threshold_cells: int = 30,
        llm_model: str | None = None,
        max_llm_table_chars: int = 8000,
        use_llm: bool = True,
    ) -> None:
        self.llm_threshold_cells = llm_threshold_cells
        self.max_llm_table_chars = max_llm_table_chars
        self.use_llm = use_llm
        self._metrics = MetricsCollector.get_instance()

        from src.config import get_settings
        cfg = get_settings()
        model = llm_model or cfg.groq.fast_model
        self._llm = RotatableGroqGenerator(model=model) if use_llm else None

        log.info(
            "TableReformatter initialised",
            extra={
                "llm_threshold_cells": llm_threshold_cells,
                "use_llm": use_llm,
                "model": model if use_llm else "disabled",
            },
        )

    # ------------------------------------------------------------------
    # Haystack run()
    # ------------------------------------------------------------------

    @component.output_types(tables=list)
    def run(
        self,
        html_tables: list[str],
        context: str | None = None,
    ) -> dict[str, list[ReformattedTable]]:
        """
        Reformat a list of HTML table strings.

        Args:
            html_tables: Raw HTML strings (one per table).
            context:     Optional surrounding document text used as
                         system context in LLM prompts.

        Returns:
            ``{"tables": list[ReformattedTable]}`` in input order.
        """
        if not html_tables:
            return {"tables": []}

        results: list[ReformattedTable] = []
        for html in html_tables:
            with timed_operation("tables.reformat", log):
                result = self._reformat_one(html, context)
            results.append(result)
            self._metrics.record_event("tables.processed")
        return {"tables": results}

    # ------------------------------------------------------------------
    # Single-table processing
    # ------------------------------------------------------------------

    def _reformat_one(self, html: str, context: str | None) -> ReformattedTable:
        """Dispatch to rule-based or LLM strategy based on table size."""
        parsed = _parse_html_table(html)
        rows = parsed["rows"]
        headers = parsed["headers"]
        num_rows = len(rows)
        num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
        total_cells = num_rows * num_cols

        rule_nl = _rows_to_nl(headers, rows)

        tbl = ReformattedTable(
            html=html,
            natural_language=rule_nl,
            num_rows=num_rows,
            num_cols=num_cols,
            headers=headers,
            strategy_used="rule_based",
        )

        if self.use_llm and self._llm and total_cells > self.llm_threshold_cells:
            tbl.llm_summary = self._llm_summarise(html, context)
            tbl.strategy_used = "llm"
            tbl.natural_language = tbl.llm_summary  # prefer LLM for large tables

        return tbl

    def _llm_summarise(self, html: str, context: str | None) -> str:
        """Ask the LLM to produce a factual prose summary of the table."""
        truncated_html = html[: self.max_llm_table_chars]

        system_prompt = textwrap.dedent("""\
            You are a technical document analyst. Your task is to convert an
            HTML table into a clear, factual prose summary.

            Rules:
            - Preserve all key numerical values, units, and labels.
            - Group related rows if it aids clarity.
            - Mention column headers and what they represent.
            - Write in plain English, 3-8 sentences.
            - Do NOT add information not present in the table.
        """)

        user_content = f"HTML Table:\n\n{truncated_html}"
        if context:
            user_content += f"\n\nSurrounding document context:\n{context[:400]}"

        try:
            result = self._llm.run(
                messages=[
                    ChatMessage.from_system(system_prompt),
                    ChatMessage.from_user(user_content),
                ],
                generation_kwargs={"max_tokens": 400, "temperature": 0.1},
            )
            return result["replies"][0].content.strip()
        except Exception as exc:
            log.warning("LLM table reformatting failed", extra={"error": str(exc)})
            self._metrics.record_event("tables.llm_error")
            # Return rule-based NL as fallback
            parsed = _parse_html_table(html)
            return _rows_to_nl(parsed["headers"], parsed["rows"])


# ---------------------------------------------------------------------------
# HTML parsing helpers (no external deps beyond stdlib)
# ---------------------------------------------------------------------------


def _parse_html_table(html: str) -> dict[str, Any]:
    """
    Parse an HTML table string into headers and data rows.

    Uses ``html.parser`` from stdlib to avoid BeautifulSoup dependency.
    Falls back to regex-based extraction if parsing fails.

    Returns:
        ``{"headers": list[str], "rows": list[list[str]]}``
    """
    try:
        from html.parser import HTMLParser

        class _TableParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.headers: list[str] = []
                self.rows: list[list[str]] = []
                self._in_header = False
                self._in_cell = False
                self._current_row: list[str] = []
                self._current_text: list[str] = []
                self._in_thead = False

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag == "thead":
                    self._in_thead = True
                elif tag == "tr":
                    self._current_row = []
                elif tag in ("th", "td"):
                    self._in_cell = True
                    self._in_header = tag == "th" or self._in_thead
                    self._current_text = []

            def handle_endtag(self, tag: str) -> None:
                if tag == "thead":
                    self._in_thead = False
                elif tag == "tr":
                    if self._current_row:
                        if self._in_thead and not self.headers:
                            self.headers = self._current_row[:]
                        else:
                            self.rows.append(self._current_row[:])
                    self._current_row = []
                elif tag in ("th", "td"):
                    cell_text = " ".join(self._current_text).strip()
                    if self._in_header and not self.rows:
                        if self._in_thead or tag == "th":
                            self.headers.append(cell_text)
                        else:
                            self._current_row.append(cell_text)
                    else:
                        self._current_row.append(cell_text)
                    self._in_cell = False

            def handle_data(self, data: str) -> None:
                if self._in_cell:
                    cleaned = data.strip()
                    if cleaned:
                        self._current_text.append(cleaned)

        p = _TableParser()
        p.feed(html)

        # If we got no explicit <th> headers, treat first row as headers
        if not p.headers and p.rows:
            p.headers = p.rows.pop(0)

        return {"headers": p.headers, "rows": p.rows}

    except Exception as exc:
        log.warning("HTML table parser failed, using regex fallback", extra={"error": str(exc)})
        return _regex_parse_table(html)


def _regex_parse_table(html: str) -> dict[str, Any]:
    """Regex-based table extraction as a last resort."""
    cell_pattern = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    tag_pattern = re.compile(r"<[^>]+>")

    rows_html = row_pattern.findall(html)
    all_rows: list[list[str]] = []
    for row_html in rows_html:
        cells = cell_pattern.findall(row_html)
        cleaned = [tag_pattern.sub("", c).strip() for c in cells]
        if any(cleaned):
            all_rows.append(cleaned)

    headers: list[str] = all_rows.pop(0) if all_rows else []
    return {"headers": headers, "rows": all_rows}


def _rows_to_nl(headers: list[str], rows: list[list[str]]) -> str:
    """
    Convert parsed table data into a natural-language string.

    Produces one sentence per row using the pattern:
    ``"Row N: Header1 is Value1; Header2 is Value2."``
    """
    if not rows:
        return "The table contains no data rows."

    lines: list[str] = []

    if headers:
        lines.append(f"Table with columns: {', '.join(headers)}.")

    for i, row in enumerate(rows, start=1):
        if headers and len(headers) == len(row):
            pairs = "; ".join(
                f"{h} is {v}" for h, v in zip(headers, row) if v
            )
        else:
            pairs = "; ".join(v for v in row if v)

        if pairs:
            lines.append(f"Row {i}: {pairs}.")

    return " ".join(lines)
