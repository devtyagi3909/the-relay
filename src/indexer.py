"""
indexer.py - Section-aware PDF parser, chunker, and hybrid BM25 + TF-IDF reranking retriever.
Generates indices for 3 settings:
  1. small_256_k3_norerank: 256 tok (32 overlap), k=3, rerank=False
  2. balanced_512_k5_rerank: 512 tok (64 overlap), k=5, rerank=True
  3. large_1024_k8_rerank: 1024 tok (128 overlap), k=8, rerank=True
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Section header regex as specified in the assignment
SECTION_REGEX = re.compile(r"^(?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z0-9\s\-:]+$")
NUM_ONLY_REGEX = re.compile(r"^(?:\d+(?:\.\d+)*)$")
TABLE_START_REGEX = re.compile(r"^(Table\s+\d+[:\.\s][^\n]*)", re.IGNORECASE)

PAPERS = [
    {"paper_id": "1706.03762v7", "filename": "1706.03762v7.pdf"},
    {"paper_id": "2410.00037v1", "filename": "2410.00037v1.pdf"},
    {"paper_id": "2507.07955v2", "filename": "2507.07955v2.pdf"},
]

SETTINGS = {
    "small_256_k3_norerank": {
        "piece_size": 256,
        "overlap": 32,
        "pieces_fetched": 3,
        "rerank": False,
        "pool": 3,
    },
    "balanced_512_k5_rerank": {
        "piece_size": 512,
        "overlap": 64,
        "pieces_fetched": 5,
        "rerank": True,
        "pool": 12,
    },
    "large_1024_k8_rerank": {
        "piece_size": 1024,
        "overlap": 128,
        "pieces_fetched": 8,
        "rerank": True,
        "pool": 16,
    },
}

def estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ~ 0.75 words, or 1 word ~ 1.33 tokens)."""
    words = text.split()
    return max(1, int(len(words) * 1.33))

def parse_pdf_pages(pdf_path: Path, paper_id: str) -> List[Dict[str, Any]]:
    """
    Parses PDF page-by-page, detecting section headers and preserving table headers.
    Returns a list of page dicts with extracted blocks and current section info.
    """
    doc = pymupdf.open(str(pdf_path))
    pages_data = []
    current_section = "Abstract / Preamble"

    for page_idx in range(len(doc)):
        page_num = page_idx + 1  # 1-indexed
        page = doc[page_idx]
        text = page.get_text("text")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Find any tables on this page for the Table Preservation Rule
        table_context = ""
        for i, line in enumerate(lines):
            t_match = TABLE_START_REGEX.match(line)
            if t_match:
                caption = t_match.group(1)
                header_row = lines[i + 1] if i + 1 < len(lines) else ""
                table_context = f"[Table Context: {caption} | Columns: {header_row}]\n"
                break

        # Check for section headers (handling two-line format e.g. "3.2.1" on line i, "Scaled..." on line i+1)
        i = 0
        cleaned_blocks = []
        while i < len(lines):
            line = lines[i]
            # Match standard single-line section header
            if SECTION_REGEX.match(line):
                current_section = line
                cleaned_blocks.append((current_section, line, table_context))
            # Match split two-line section header (e.g. Vaswani line i = "3.2.1", line i+1 = "Scaled Dot-Product Attention")
            elif NUM_ONLY_REGEX.match(line) and i + 1 < len(lines) and re.match(r"^[A-Z][A-Za-z0-9\s\-:]+$", lines[i + 1]):
                combined = f"{line} {lines[i + 1]}"
                if SECTION_REGEX.match(combined):
                    current_section = combined
                    cleaned_blocks.append((current_section, combined, table_context))
                    i += 1
                else:
                    cleaned_blocks.append((current_section, line, table_context))
            elif line.lower() in ["abstract", "references", "conclusion", "conclusions", "background", "related work"]:
                current_section = line.title()
                cleaned_blocks.append((current_section, line, table_context))
            else:
                cleaned_blocks.append((current_section, line, table_context))
            i += 1

        pages_data.append({
            "paper_id": paper_id,
            "page": page_num,
            "blocks": cleaned_blocks,
            "table_context": table_context,
        })

    return pages_data

def chunk_pages_data(pages_data: List[Dict[str, Any]], piece_size_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
    """
    Chunks parsed paper text using sliding window over words with table preservation rule.
    """
    words_per_token = 0.75
    target_words = max(10, int(piece_size_tokens * words_per_token))
    overlap_words = max(2, int(overlap_tokens * words_per_token))
    stride_words = max(5, target_words - overlap_words)

    all_chunks = []
    chunk_counter = 0

    for pdata in pages_data:
        paper_id = pdata["paper_id"]
        page_num = pdata["page"]
        table_context = pdata["table_context"]

        # Group lines by section on this page
        current_sec = "General"
        sec_lines = []
        page_sections = []

        for sec, line, _ in pdata["blocks"]:
            if sec != current_sec:
                if sec_lines:
                    page_sections.append((current_sec, sec_lines))
                current_sec = sec
                sec_lines = [line]
            else:
                sec_lines.append(line)
        if sec_lines:
            page_sections.append((current_sec, sec_lines))

        # Chunk within each section on the page
        for section_name, lines in page_sections:
            text = " ".join(lines)
            words = text.split()
            if not words:
                continue

            # Slide window
            w_idx = 0
            while w_idx < len(words):
                window_words = words[w_idx : w_idx + target_words]
                chunk_text = " ".join(window_words)

                # Table Preservation Rule: If numeric rows detected or table on page, prepend table context
                has_numeric_row = bool(re.search(r"\b\d+\.\d+\b", chunk_text) or re.search(r"\b\d{2,}\b", chunk_text))
                if table_context and has_numeric_row and "[Table Context:" not in chunk_text:
                    final_text = f"{table_context}{chunk_text}"
                else:
                    final_text = chunk_text

                chunk_counter += 1
                token_est = estimate_tokens(final_text)
                chunk_id = f"{paper_id}_p{page_num}_c{chunk_counter:04d}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "page": page_num,
                    "section": section_name,
                    "text": final_text,
                    "token_est": token_est,
                })

                if w_idx + target_words >= len(words):
                    break
                w_idx += stride_words

    return all_chunks

def build_indices(papers_dir: Path, output_chunks_dir: Path) -> Dict[str, Path]:
    output_chunks_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    paper_files = [
        ("1706.03762v7", papers_dir / "1706.03762v7.pdf"),
        ("2410.00037v1", papers_dir / "2410.00037v1.pdf"),
        ("2507.07955v2", papers_dir / "2507.07955v2.pdf"),
    ]

    # Pre-parse all papers
    parsed_all_papers = {}
    for pid, ppath in paper_files:
        if not ppath.exists():
            raise FileNotFoundError(f"Missing PDF: {ppath}. Run fetch_papers first.")
        logger.info(f"Parsing {pid} ({ppath.name})...")
        parsed_all_papers[pid] = parse_pdf_pages(ppath, pid)

    for setting_name, cfg in SETTINGS.items():
        all_chunks = []
        for pid, _ in paper_files:
            chunks = chunk_pages_data(
                parsed_all_papers[pid],
                piece_size_tokens=cfg["piece_size"],
                overlap_tokens=cfg["overlap"],
            )
            all_chunks.extend(chunks)

        out_path = output_chunks_dir / f"{setting_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c) + "\n")

        logger.info(f"Generated index '{setting_name}': {len(all_chunks)} chunks -> {out_path}")
        results[setting_name] = out_path

    return results

class HybridRetriever:
    """
    Search engine over pre-indexed JSONL chunks with BM25 + TF-IDF n-gram and entity overlap boost.
    """
    def __init__(self, chunks_path: Path, setting_config: Dict[str, Any]):
        self.chunks_path = Path(chunks_path)
        self.config = setting_config
        self.chunks: List[Dict[str, Any]] = []
        self._load_chunks()
        self._init_bm25()
        if self.config.get("rerank", False):
            self._init_tfidf()

    def _load_chunks(self):
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase alphanumeric tokenization
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def _init_bm25(self):
        tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _init_tfidf(self):
        from sklearn.pipeline import FeatureUnion
        corpus_texts = [f"{c['section']} {c['text']}" for c in self.chunks]
        self.tfidf = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), analyzer="word", max_features=20000, sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=20000, sublinear_tf=True)),
        ])
        self.tfidf_matrix = self.tfidf.fit_transform(corpus_texts)

    def search(
        self,
        query: str,
        paper_id_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes search. Returns (retrieved_chunks, max_bm25_norm).
        """
        k = top_k or self.config["pieces_fetched"]
        tokens = self._tokenize(query)
        if not tokens:
            return [], 0.0

        bm25_scores = np.array(self.bm25.get_scores(tokens))
        raw_max = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 0.0
        # Normalization: typical BM25 scores for well-matching 256-512 tok passages range from 10 to 30
        max_bm25_norm = min(1.0, raw_max / 25.0)

        # Apply paper filter if specified
        if paper_id_filter:
            mask = np.array([c["paper_id"] == paper_id_filter for c in self.chunks])
            bm25_scores = np.where(mask, bm25_scores, -1e9)

        if not self.config.get("rerank", False):
            # Pure BM25
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            results = []
            for idx in top_indices:
                if bm25_scores[idx] <= -1e8:
                    continue
                chunk = dict(self.chunks[idx])
                chunk["score"] = round(float(bm25_scores[idx]), 3)
                results.append(chunk)
            return results, round(max_bm25_norm, 4)

        # Re-ranking path
        pool_size = self.config.get("pool", k * 2)
        candidate_indices = np.argsort(bm25_scores)[::-1][:pool_size]
        candidate_indices = [idx for idx in candidate_indices if bm25_scores[idx] > -1e8]

        if not candidate_indices:
            return [], 0.0

        # Query TF-IDF similarity
        q_vec = self.tfidf.transform([query])
        cand_tfidf = self.tfidf_matrix[candidate_indices]
        sims = cosine_similarity(q_vec, cand_tfidf).flatten()

        # Entity boost: reward chunks that match numbers or acronyms in the query
        query_entities = set(re.findall(r"\b[A-Z0-9\.\-]{2,}\b|\b\d+\b", query))
        bm25_cands = bm25_scores[candidate_indices]
        cand_max_bm25 = np.max(bm25_cands) if len(bm25_cands) > 0 and np.max(bm25_cands) > 0 else 1.0

        final_scores = []
        for i, idx in enumerate(candidate_indices):
            chunk_text = self.chunks[idx]["text"]
            # Count exact entity matches
            ent_matches = sum(1 for e in query_entities if e in chunk_text)
            ent_boost = min(0.3, ent_matches * 0.08)

            norm_bm25 = max(0.0, bm25_cands[i] / cand_max_bm25)
            tfidf_sim = max(0.0, float(sims[i]))
            composite_score = 0.50 * norm_bm25 + 0.35 * tfidf_sim + 0.15 * ent_boost
            final_scores.append((composite_score, idx))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        top_k_items = final_scores[:k]

        results = []
        for score, idx in top_k_items:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results, round(max_bm25_norm, 4)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    papers_dir = root_dir / "data" / "papers"
    chunks_dir = root_dir / "data" / "chunks"

    generated = build_indices(papers_dir, chunks_dir)
    print(f"Successfully generated indices for all 3 settings in {chunks_dir}:")
    for s, p in generated.items():
        print(f"  - {s}: {p.stat().st_size} bytes")

if __name__ == "__main__":
    main()
