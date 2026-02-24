"""Clinical guideline retrieval for evidence-based recommendations.

This module previously contained a hardcoded guideline database with
paraphrased/summarized text. That has been removed because the text
was not directly quoted from published guidelines and could be misleading.

The retrieve_guidelines() function now returns an empty list.
A production system should use a licensed, curated guideline database
or direct API access to guideline publishers.
"""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GuidelineSnippet:
    """A snippet from a clinical guideline."""

    title: str
    source: str
    text: str
    relevance_score: float
    finding_type: str
    citation: str


# Guideline database removed — the previous entries were paraphrased summaries,
# not direct quotes from published guidelines. A production system should use
# a licensed guideline database or API.
GUIDELINE_DATABASE: list[dict] = []


async def retrieve_guidelines(
    finding_types: list[str],
    max_snippets: int = 3,
) -> list[GuidelineSnippet]:
    """
    Retrieve relevant clinical guideline snippets based on finding types.

    Currently returns an empty list. A production system should connect
    to a licensed guideline database or API.
    """
    logger.info("Guideline retrieval skipped (no guideline database configured)")
    return []


def get_guideline_text_for_prompt(snippets: list[GuidelineSnippet]) -> str:
    """Format guideline snippets for inclusion in inference prompt."""
    if not snippets:
        return ""

    lines = ["Relevant Clinical Guidelines:"]
    for snippet in snippets:
        lines.append(f"\n[{snippet.source}]")
        lines.append(snippet.text)

    return "\n".join(lines)


async def get_all_guideline_sources() -> list[str]:
    """Get list of all guideline sources in the database."""
    return []
