"""
evaluator.py - Deterministic citation and answer quality grader.
Evaluates answers against data/questions.json ground-truth rules:
  - "correct": question answerable, valid citations [<paper_id>, p.<page>], >= 2 keywords matched
  - "wrong": question answerable, but failed keywords or invalid citations
  - "refused_correctly": trap question properly refused
  - "refused_wrongly": answerable query falsely refused OR trap query unfaithfully answered
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[(1706\.03762v7|2410\.00037v1|2507\.07955v2),\s*p\.(\d+)\]")

REFUSAL_PHRASES = [
    "cannot answer",
    "does not contain",
    "not contain",
    "not reported",
    "none of the three",
    "not in the provided",
    "not in the paper",
    "not provide",
    "not mention",
    "neither moshi nor h-net",
    "neither",
]

def contains_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)

def extract_citations(text: str) -> List[Tuple[str, int]]:
    """Returns list of (paper_id, page_num)."""
    matches = CITATION_REGEX.findall(text)
    return [(m[0], int(m[1])) for m in matches]

def grade_response(
    question_spec: Dict[str, Any],
    response_text: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Grades response_text against question_spec.
    Returns (grade, details_dict) where grade is one of:
      ["correct", "wrong", "refused_correctly", "refused_wrongly"]
    """
    expected = question_spec.get("expected_outcome", "correct")
    has_refusal = contains_refusal(response_text)
    citations = extract_citations(response_text)
    
    # 1. Handle Trap Questions
    if expected == "refused_correctly":
        if has_refusal:
            return "refused_correctly", {
                "reason": "Correctly refused unanswerable question",
                "citations_found": citations,
            }
        else:
            return "refused_wrongly", {
                "reason": "Failed to refuse unanswerable trap question (hallucination)",
                "citations_found": citations,
            }

    # 2. Handle Answerable Questions
    if has_refusal:
        return "refused_wrongly", {
            "reason": "Refused an answerable question",
            "citations_found": citations,
        }

    # Must contain citations in format [<paper_id>, p.<page>]
    if not citations:
        return "wrong", {
            "reason": "Missing required [<paper_id>, p.<page>] citation",
            "citations_found": [],
        }

    # Check keyword / number matches
    required_keywords = question_spec.get("required_keywords", [])
    min_keywords = question_spec.get("min_keywords_matched", 2)
    
    lower_text = response_text.lower()
    matched_keywords = [
        kw for kw in required_keywords
        if kw.lower() in lower_text
    ]

    if len(matched_keywords) < min_keywords:
        return "wrong", {
            "reason": f"Only matched {len(matched_keywords)}/{min_keywords} required keywords: {matched_keywords}",
            "citations_found": citations,
            "matched_keywords": matched_keywords,
        }

    return "correct", {
        "reason": f"Matched {len(matched_keywords)} keywords and provided valid citations",
        "citations_found": citations,
        "matched_keywords": matched_keywords,
    }
