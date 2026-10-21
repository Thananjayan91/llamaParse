import os
import re
import json
import math
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from bs4 import BeautifulSoup

import config

# Try loading fastembed for local dense vector embeddings
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
    _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    HAS_FASTEMBED = False
    _embedding_model = None

# Optional LLM integrations
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class HybridSemanticRetriever:
    """
    True Hybrid Semantic Retriever combining:
    1. Local Dense Vector Embeddings (FastEmbed BGE-small) - semantic similarity, synonyms, paraphrasing.
    2. BM25 Frequency Scoring - exact terminology and keyword matches.
    3. Markdown & Table-Aware Chunking - preserves structured tables and section contexts intact.
    """
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.node_embeddings: Optional[np.ndarray] = None
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf_dict: Dict[str, float] = collections.defaultdict(float)

    def _tokenize(self, text: str) -> List[str]:
        # Strip HTML tags before tokenizing for BM25
        clean_text = re.sub(r"<[^>]+>", " ", text)
        return re.findall(r"\w+", clean_text.lower())

    def _extract_table_summary(self, html_table: str) -> str:
        """Converts HTML table into natural language rows for dense embedding."""
        try:
            soup = BeautifulSoup(html_table, "html.parser")
            rows = soup.find_all("tr")
            if not rows:
                return ""
            
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))

            lines = []
            for r in rows[1:]:
                cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
                if not cells:
                    continue
                row_label = cells[0]
                for idx, val in enumerate(cells[1:], 1):
                    col_header = headers[idx] if idx < len(headers) else f"Col {idx}"
                    lines.append(f"{row_label} in {col_header} is {val}")
            
            return ". ".join(lines)
        except Exception:
            return ""

    def _chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = doc.get("doc_id")
        filename = doc.get("original_filename", "Document")
        # Prefer Markdown over plain text (Markdown preserves LlamaParse tables and layouts)
        content = doc.get("markdown", "") or doc.get("text", "")
        if not content.strip():
            return []

        chunks = []
        # Split document by HTML tables so tables stay 100% intact
        parts = re.split(r"(<table.*?>.*?</table>)", content, flags=re.DOTALL | re.IGNORECASE)

        current_header = filename
        for p in parts:
            p_strip = p.strip()
            if not p_strip:
                continue

            # Update current section header if headers found
            headers = re.findall(r"(?:^|\n)(#{1,4}\s+[^\n]+)", p_strip)
            if headers:
                current_header = headers[-1].strip("# ").strip()

            # If this part is a table:
            if re.match(r"^<table", p_strip, re.IGNORECASE):
                table_nl_summary = self._extract_table_summary(p_strip)
                chunk_text = f"### Section: {current_header}\n\n{p_strip}"
                embed_text = f"Section: {current_header}. Table contents: {table_nl_summary}. {chunk_text[:500]}"
                chunks.append({
                    "text": chunk_text,
                    "embed_text": embed_text,
                    "is_table": True,
                    "header": current_header
                })
            else:
                # Regular text: split by headers or paragraphs
                sections = re.split(r"(?=\n#{1,3}\s+)", p_strip)
                for s in sections:
                    s = s.strip()
                    if not s:
                        continue
                    if len(s) > 1200:
                        # Split large sections into overlapping blocks
                        step = 800
                        for start_idx in range(0, len(s), step):
                            block = s[start_idx:start_idx + 1000].strip()
                            if block:
                                chunks.append({
                                    "text": block,
                                    "embed_text": block,
                                    "is_table": False,
                                    "header": current_header
                                })
                    else:
                        chunks.append({
                            "text": s,
                            "embed_text": s,
                            "is_table": False,
                            "header": current_header
                        })

        processed_nodes = []
        for idx, c in enumerate(chunks):
            tokens = self._tokenize(c["text"])
            processed_nodes.append({
                "node_id": f"{doc_id}_chunk_{idx}",
                "doc_id": doc_id,
                "filename": filename,
                "text": c["text"],
                "embed_text": c.get("embed_text", c["text"]),
                "is_table": c.get("is_table", False),
                "header": c.get("header", ""),
                "chunk_index": idx,
                "tokens": tokens,
                "token_count": len(tokens)
            })
        return processed_nodes

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.nodes = []
        for doc in documents:
            doc_nodes = self._chunk_document(doc)
            self.nodes.extend(doc_nodes)

        N = len(self.nodes)
        if N == 0:
            self.node_embeddings = None
            return

        # Precompute BM25 statistics
        self.doc_lengths = [n["token_count"] for n in self.nodes]
        self.avg_doc_len = sum(self.doc_lengths) / N if N > 0 else 0

        doc_freq = collections.defaultdict(int)
        for n in self.nodes:
            unique_terms = set(n["tokens"])
            for t in unique_terms:
                doc_freq[t] += 1

        for t, df in doc_freq.items():
            self.idf_dict[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # Precompute Dense Vector Embeddings
        if HAS_FASTEMBED and _embedding_model is not None:
            embed_texts = [n["embed_text"] for n in self.nodes]
            raw_embs = list(_embedding_model.embed(embed_texts))
            embs = np.array(raw_embs)
            # Normalize to unit length for fast cosine similarity via dot product
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.node_embeddings = embs / norms
        else:
            self.node_embeddings = None

    def retrieve(self, query: str, top_k: int = 5, doc_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.nodes:
            return []

        query_tokens = self._tokenize(query)
        k1 = 1.5
        b = 0.75

        # 1. Compute BM25 Scores
        bm25_scores = np.zeros(len(self.nodes))
        for idx, node in enumerate(self.nodes):
            term_counts = collections.Counter(node["tokens"])
            doc_len = node["token_count"]
            score = 0.0
            for q in query_tokens:
                if q in term_counts:
                    tf = term_counts[q]
                    idf = self.idf_dict.get(q, 0.1)
                    denom = tf + k1 * (1 - b + b * (doc_len / (self.avg_doc_len or 1)))
                    score += idf * ((tf * (k1 + 1)) / denom)
            
            # Substring match bonus
            if query.lower() in node["text"].lower():
                score += 3.0
            bm25_scores[idx] = score

        # Normalize BM25
        max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 0
        if max_bm25 > 0:
            bm25_norm = bm25_scores / max_bm25
        else:
            bm25_norm = bm25_scores

        # 2. Compute Dense Vector Similarities
        dense_scores = np.zeros(len(self.nodes))
        if HAS_FASTEMBED and _embedding_model is not None and self.node_embeddings is not None:
            q_emb = list(_embedding_model.embed([query]))[0]
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm
                dense_scores = np.dot(self.node_embeddings, q_emb)
                # Clip to [0, 1]
                dense_scores = np.clip(dense_scores, 0, 1)

        # 3. Hybrid Ranking (70% Dense Semantic + 30% Keyword BM25)
        if HAS_FASTEMBED and self.node_embeddings is not None:
            hybrid_scores = 0.70 * dense_scores + 0.30 * bm25_norm
        else:
            hybrid_scores = bm25_norm

        scored_nodes = []
        for idx, node in enumerate(self.nodes):
            if doc_filter and node["doc_id"] not in doc_filter:
                continue
            scored_nodes.append((float(hybrid_scores[idx]), node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, node in scored_nodes[:top_k]:
            results.append({
                "node_id": node["node_id"],
                "doc_id": node["doc_id"],
                "filename": node["filename"],
                "text": node["text"],
                "header": node.get("header", ""),
                "is_table": node.get("is_table", False),
                "score": round(float(score), 4),
                "chunk_index": node["chunk_index"]
            })
        return results


class RAGService:
    def __init__(self):
        self.retriever = HybridSemanticRetriever()
        self.openai_key = config.OPENAI_API_KEY
        self.gemini_key = config.GEMINI_API_KEY
        self.refresh_index()

    def refresh_index(self):
        """Loads all parsed files from PARSED_DIR and rebuilds index."""
        from services.llama_service import llama_service
        parsed_docs = llama_service.list_parsed_documents()
        full_docs = []
        for d in parsed_docs:
            doc_data = llama_service.get_parsed_document(d["doc_id"])
            if doc_data:
                full_docs.append({
                    "doc_id": d["doc_id"],
                    "original_filename": d.get("original_filename", "Document"),
                    "text": doc_data["text"],
                    "markdown": doc_data["markdown"]
                })
        self.retriever.index_documents(full_docs)

    def _extract_table_answer(self, query: str, retrieved_nodes: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extracts exact cell values from HTML/Markdown tables when a query asks for a specific year and metric.
        """
        # Look for years (e.g. 2017, 2018, 2021) in the query
        year_match = re.search(r"\b(20\d{2})\b", query)
        target_year = year_match.group(1) if year_match else None

        # Clean query keywords
        q_clean = re.sub(r"\b20\d{2}\b", "", query).lower().strip()
        q_keywords = [w for w in re.findall(r"\w+", q_clean) if len(w) > 2 and w not in ["what", "when", "much", "total", "were", "show", "give", "tell"]]

        for node in retrieved_nodes:
            text = node["text"]
            if "<table" not in text.lower():
                continue

            try:
                soup = BeautifulSoup(text, "html.parser")
                for table in soup.find_all("table"):
                    rows = table.find_all("tr")
                    if not rows:
                        continue

                    # Search header for year
                    header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
                    year_col_idx = None
                    if target_year:
                        for idx, h in enumerate(header_cells):
                            if target_year in h:
                                year_col_idx = idx
                                break

                    # Look for matching row
                    for r in rows[1:]:
                        cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
                        if not cells:
                            continue
                        row_name = cells[0]
                        row_lower = row_name.lower()
                        # If row matches keywords
                        if any(k in row_lower for k in q_keywords):
                            val = cells[year_col_idx] if (year_col_idx is not None and year_col_idx < len(cells)) else "See Table"
                            
                            answer = (
                                f"### Direct Answer:\n\n"
                                f"- **Metric**: {row_name}\n"
                            )
                            if target_year:
                                answer += f"- **Year**: **{target_year}**\n"
                            answer += (
                                f"- **Value**: **{val}**\n\n"
                                f"---\n\n"
                                f"#### Full Reference Table:\n\n"
                                f"{str(table)}\n"
                            )
                            return answer
            except Exception:
                continue

        return None

    def query(self, prompt: str, doc_ids: Optional[List[str]] = None, top_k: int = 5, provider: str = "auto") -> Dict[str, Any]:
        """
        Retrieves relevant context chunks and synthesizes an answer.
        """
        self.refresh_index()
        retrieved_nodes = self.retriever.retrieve(prompt, top_k=top_k, doc_filter=doc_ids)

        if not retrieved_nodes:
            return {
                "answer": "No relevant context found in the parsed documents for your question. Please ensure documents are uploaded and parsed.",
                "sources": [],
                "provider_used": "none"
            }

        # Context assembly
        context_str = "\n\n---\n\n".join([
            f"[Source: {n['filename']} (Node {n['node_id']}) - Relevance Score: {n['score']}]\n{n['text']}"
            for n in retrieved_nodes
        ])

        answer = ""
        provider_used = "extractive_rag"

        # 1. Try Gemini
        if (provider == "gemini" or (provider == "auto" and self.gemini_key)) and self.gemini_key:
            # Try new google-genai SDK first
            if HAS_GOOGLE_GENAI:
                try:
                    client = genai.Client(api_key=self.gemini_key)
                    system_prompt = (
                        "You are an expert RAG analyst. Answer the user question accurately using the provided context from documents.\n"
                        "Read all tables carefully (matching rows and column headers accurately). Cite document names.\n\n"
                        f"Context:\n{context_str}\n\n"
                        f"Question: {prompt}\n\nAnswer:"
                    )
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=system_prompt
                    )
                    answer = resp.text
                    provider_used = "gemini-2.5-flash"
                except Exception as e_new:
                    # Fallback to gemini-1.5-flash
                    try:
                        resp = client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=system_prompt
                        )
                        answer = resp.text
                        provider_used = "gemini-1.5-flash"
                    except Exception:
                        pass

            if not answer and HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.gemini_key)
                    model = legacy_genai.GenerativeModel("gemini-1.5-flash")
                    system_prompt = (
                        "You are an expert RAG analyst. Answer the user question accurately using the provided context from documents.\n"
                        "Read all tables carefully (matching rows and column headers accurately). Cite document names.\n\n"
                        f"Context:\n{context_str}\n\n"
                        f"Question: {prompt}\n\nAnswer:"
                    )
                    resp = model.generate_content(system_prompt)
                    answer = resp.text
                    provider_used = "gemini-1.5-flash"
                except Exception as e_leg:
                    pass

        # 2. Try OpenAI
        elif (provider == "openai" or (provider == "auto" and self.openai_key)) and self.openai_key:
            if HAS_OPENAI:
                try:
                    client = openai.OpenAI(api_key=self.openai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a precise RAG analyst. Answer the user question accurately using the context chunks. When reading tables, align columns and rows accurately."},
                            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"}
                        ]
                    )
                    answer = resp.choices[0].message.content
                    provider_used = "openai-gpt-4o-mini"
                except Exception as e:
                    pass

        # 3. Intelligent Local Engine (when no external LLM API key is configured)
        if not answer:
            provider_used = "Hybrid Semantic RAG (Local Engine)"
            # Check if this query is asking for a table cell/metric
            table_answer = self._extract_table_answer(prompt, retrieved_nodes)
            if table_answer:
                answer = table_answer
            else:
                top_nodes = retrieved_nodes[:3]
                answer = (
                    f"### Retrieved Relevant Context:\n\n"
                    f"Here are the most relevant sections identified by the semantic retrieval engine for **'{prompt}'**:\n\n"
                )
                for idx, n in enumerate(top_nodes, 1):
                    header_info = f" ({n['header']})" if n.get("header") else ""
                    answer += f"#### {idx}. From `{n['filename']}`{header_info} — Similarity: {n['score']}\n\n"
                    answer += f"{n['text']}\n\n---\n\n"

        return {
            "query": prompt,
            "answer": answer,
            "sources": retrieved_nodes,
            "provider_used": provider_used,
            "total_chunks_indexed": len(self.retriever.nodes)
        }

rag_service = RAGService()
