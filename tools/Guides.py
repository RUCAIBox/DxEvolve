# tools/guideline_search.py

import os
import json
from typing import Any, Dict, List, Optional, Type, DefaultDict
from collections import defaultdict
import torch
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


GUIDELINE_JSON_PATH = "./guidence/guidence_new.json"
DEFAULT_INDEX_DIR = "./guidence/bge-guidence-new.faiss"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"

EXTRACT_PROMPT = """Extract diagnostic guidance from ONE guideline chunk.

Clinical query:
{query}

Year: {year}
Title: {title}

Chunk:
{content}

Return points only. Max 8.
Each point must:
- Be explicitly for confirming or ruling out the suspected disease in the query.
- Mention a test (lab or imaging with region/modality) AND what result/finding/threshold means.
- No treatment/management/procedures.

Tagging:
- Start each point with [GENERAL] or [RARE].
- Use [RARE] only for special subgroups/unusual scenarios, and then append: "Rare context: <who/when>".

If nothing diagnostic, output: NO_DIAGNOSTIC_POINTS
"""


MERGE_PROMPT = """Merge and deduplicate diagnostic points across chunks.

Clinical query:
{query}

Points:
{bullets}

Return up to 10 points.
Keep only diagnostic/test-ordering points (tests + findings/thresholds). Drop any treatment/procedure content.
Preserve [RARE] and its "Rare context:" when present.
"""

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


class GuidelineStore:
    """
    Thin wrapper around LangChain FAISS for guideline retrieval.
    Index text for each guideline is:
      f"Title: {title}\\nAbstract: {abstract}"
    Metadata stores the full guideline dict (year, title, abstract, content).
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_INDEX_DIR,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ):
        self.persist_dir = persist_dir
        self.embeddings = embeddings or self._build_default_embeddings()
        self.vs: Optional[FAISS] = None

        if os.path.isdir(self.persist_dir):
            try:
                self.vs = FAISS.load_local(
                    self.persist_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                raise
        else:
            raise FileNotFoundError

    @staticmethod
    def _build_default_embeddings() -> HuggingFaceEmbeddings:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBED_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_from_json(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("guidence.json must be a List[Dict[str, str]].")

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            year = _safe_str(item.get("year"))
            title = _safe_str(item.get("title"))
            abstract = _safe_str(item.get("abstract"))
            content = _safe_str(item.get("content"))

            text = f"Title: {title}\nAbstract: {abstract}".strip()
            meta = {
                "year": year,
                "title": title,
                "abstract": abstract,
                "content": content,
            }

            if text:
                texts.append(text)
                metadatas.append(meta)

        if not texts:
            raise ValueError("No valid guideline entries found to index.")

        self.vs = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def save(self) -> None:
        if self.vs is None:
            raise ValueError("Vector store is empty. Build or load it first.")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vs.save_local(self.persist_dir)

    def similarity_search(self, query: str, k: int = 5):
        if self.vs is None:
            return []
        return self.vs.similarity_search_with_score(query, k=k)


class GuidelineSearchInput(BaseModel):
    action_input: str = Field(
        description=(
            "Compact keyword-style clinical query for guideline retrieval. "
            "Include suspected condition, diagnostic task (diagnosis, labs, imaging), "
            "and 2–6 key findings (symptom location, fever, WBC, LFT, CT, ultrasound). "
            "Avoid long patient narratives."
        )
    )


class GuidelineSearchTool(BaseTool):
    name: str = "Guideline Search"
    description: str = (
        "Search medical guidelines via dense retrieval over title+abstract. "
        "Then extract only query-relevant diagnostic criteria and test-ordering points."
    )
    args_schema: Type[BaseModel] = GuidelineSearchInput

    guideline_store: Any
    llm: Any
    tags: Optional[Dict[str, str]] = None

    k: int = 5
    max_guidelines: int = 5

    # context control
    reserve_output_tokens: int = 512
    min_chunk_tokens: int = 80000
    max_chunks_per_guideline: int = 24

    class Config:
        arbitrary_types_allowed = True

    def _wrap_with_tags(self, system_text: str, user_text: str) -> str:
        if not self.tags:
            return f"{system_text}\n\n{user_text}".strip()

        t = self.tags
        sys_s = t.get("system_tag_start", "")
        sys_e = t.get("system_tag_end", "")
        usr_s = t.get("user_tag_start", "")
        usr_e = t.get("user_tag_end", "")
        ai_s = t.get("ai_tag_start", "")
        return f"{sys_s}{system_text}{sys_e}{usr_s}{user_text}{usr_e}{ai_s}"

    def _call_llm(self, prompt: str) -> str:
        # CustomLLM._call(prompt, stop=...) needs stop
        try:
            return str(self.llm._call(prompt, stop=[]))
        except TypeError:
            # fallback if your llm signature differs
            return str(self.llm._call(prompt))

    def _call_llm_batch(self, prompts: List[str]) -> List[str]:
        return self.llm._call_batch(prompts)

    def _get_tokenizer(self):
        return getattr(self.llm, "tokenizer", None)

    def _token_len(self, text: str) -> int:
        tok = self._get_tokenizer()
        if tok is None:
            # fallback: rough char heuristic
            return max(1, len(text) // 4)
        return len(tok.encode(text, add_special_tokens=False))

    def _chunk_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        tok = self._get_tokenizer()
        if tok is None:
            # fallback: split by chars
            max_chars = max(800, max_tokens * 4)
            return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        ids = tok.encode(text, add_special_tokens=False)
        chunks: List[str] = []
        for i in range(0, len(ids), max_tokens):
            part_ids = ids[i : i + max_tokens]
            chunks.append(tok.decode(part_ids, skip_special_tokens=True))
        return chunks

    def _max_content_tokens_for_prompt(
        self,
        query: str,
        year: str,
        title: str,
        abstract: str,
        system_text: str,
    ) -> int:
        max_ctx = getattr(self.llm, "max_context_length", None) or 8192

        # estimate prompt tokens without content filled
        skeleton = EXTRACT_PROMPT.format(
            query=query, year=year, title=title, abstract=abstract, content=""
        )
        wrapped = self._wrap_with_tags(system_text, skeleton)
        base_tokens = self._token_len(wrapped)

        avail = max_ctx - base_tokens - int(self.reserve_output_tokens)
        return min(self.min_chunk_tokens, avail)

    def _run(self, action_input: str) -> str:
        query = (action_input or "").strip()
        if not query:
            return "Empty query."

        results = self.guideline_store.similarity_search(query, k=self.k)
        if not results:
            return "No relevant guideline found."

        system_text = "You are a medical guideline extractor. Output only concise bullets."
        blocks: List[str] = []

        for rank, (doc, score) in enumerate(results[: self.max_guidelines], 1):
            meta: Dict = getattr(doc, "metadata", {}) or {}
            year = str(meta.get("year", "") or "")
            title = str(meta.get("title", "Unknown title") or "Unknown title")
            abstract = str(meta.get("abstract", "") or "")
            content = str(meta.get("content", "") or getattr(doc, "page_content", "") or "")

            # decide chunk size by tokenizer context
            max_content_tokens = self._max_content_tokens_for_prompt(
                query=query,
                year=year,
                title=title,
                abstract=abstract,
                system_text=system_text,
            )

            content_chunks = self._chunk_by_tokens(content, max_content_tokens)
            content_chunks = content_chunks[: self.max_chunks_per_guideline]

            chunk_bullets: List[str] = []
            for cidx, chunk in enumerate(content_chunks, 1):
                user_text = EXTRACT_PROMPT.format(
                    query=query,
                    year=year,
                    title=title,
                    abstract=abstract,
                    content=chunk,
                )
                prompt = self._wrap_with_tags(system_text, user_text)
                extracted = self._call_llm(prompt).strip()

                lines = [ln.strip() for ln in extracted.splitlines() if ln.strip()]
                extracted = "\n".join(lines[:12]).strip()
                if extracted:
                    chunk_bullets.append(extracted)

            # merge bullets from chunks (also context-safe)
            merged = ""
            if len(chunk_bullets) == 0:
                merged = "No extractable diagnostic/test-ordering points."
            elif len(chunk_bullets) == 1:
                merged = chunk_bullets[0]
            else:
                bullets_text = "\n\n".join(f"Chunk {i}:\n{b}" for i, b in enumerate(chunk_bullets, 1))
                merge_user = MERGE_PROMPT.format(query=query, bullets=bullets_text)
                merge_prompt = self._wrap_with_tags(system_text, merge_user)

                # if merge prompt too long, trim bullets by keeping first N lines
                if self._token_len(merge_prompt) > (getattr(self.llm, "max_context_length", 8192) - self.reserve_output_tokens):
                    flat_lines = []
                    for b in chunk_bullets:
                        flat_lines.extend([ln for ln in b.splitlines() if ln.strip()])
                    flat_lines = flat_lines[:40]
                    bullets_text = "\n".join(flat_lines)
                    merge_user = MERGE_PROMPT.format(query=query, bullets=bullets_text)
                    merge_prompt = self._wrap_with_tags(system_text, merge_user)

                merged = self._call_llm(merge_prompt).strip()
                merged_lines = [ln.strip() for ln in merged.splitlines() if ln.strip()]
                merged = "\n".join(merged_lines[:14]).strip()

            blocks.append(
                f"Guideline {rank} (score={score:.4f})\n"
                f"Title: {title}\n"
                f"Year: {year}\n"
                f"Key points:\n{merged}"
            )

        return "\n\n---\n\n".join(blocks)

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("GuidelineSearchTool does not support async.")

    def _run(self, action_input: str) -> str:
        query = (action_input or "").strip()
        if not query:
            return "Empty query."

        # return "Not available."

        results = self.guideline_store.similarity_search(query, k=self.k)
        if not results:
            return "No relevant guideline found."

        system_text = "You are a medical guideline extractor. Output only concise bullets."

        # 保存每条 guideline 的元信息, 方便最后拼 blocks
        guideline_meta: Dict[int, Dict[str, str]] = {}

        # (rank, cidx, prompt)
        extract_jobs: List[Tuple[int, int, str]] = []

        # 1) 为所有检索结果构建 chunk prompts, 一次性 batch 抽取
        for rank, (doc, score) in enumerate(results[: self.max_guidelines], 1):
            meta: Dict = getattr(doc, "metadata", {}) or {}
            year = str(meta.get("year", "") or "")
            title = str(meta.get("title", "Unknown title") or "Unknown title")
            abstract = str(meta.get("abstract", "") or "")
            content = str(meta.get("content", "") or getattr(doc, "page_content", "") or "")

            guideline_meta[rank] = {
                "score": f"{float(score):.4f}",
                "year": year,
                "title": title,
                "abstract": abstract,
            }

            max_content_tokens = self._max_content_tokens_for_prompt(
                query=query,
                year=year,
                title=title,
                abstract=abstract,
                system_text=system_text,
            )

            content_chunks = self._chunk_by_tokens(content, max_content_tokens)
            content_chunks = content_chunks[: self.max_chunks_per_guideline]

            for cidx, chunk in enumerate(content_chunks, 1):
                user_text = EXTRACT_PROMPT.format(
                    query=query,
                    year=year,
                    title=title,
                    abstract=abstract,
                    content=chunk,
                )
                prompt = self._wrap_with_tags(system_text, user_text)
                extract_jobs.append((rank, cidx, prompt))

        if not extract_jobs:
            return "No relevant guideline found."

        extract_prompts = [p for _, _, p in extract_jobs]
        extract_outs = self._call_llm_batch(extract_prompts)

        # 2) 还原每个 guideline 的 chunk bullets
        # rank -> {cidx -> cleaned_bullets}
        chunk_bullets_by_rank: Dict[int, Dict[int, str]] = defaultdict(dict)

        for (rank, cidx, _), out in zip(extract_jobs, extract_outs):
            extracted = (out or "").strip()
            lines = [ln.strip() for ln in extracted.splitlines() if ln.strip()]
            extracted = "\n".join(lines[:12]).strip()
            if extracted:
                chunk_bullets_by_rank[rank][cidx] = extracted

        # 3) 为需要 merge 的 guideline 构建 merge prompts, 再 batch 一次
        merge_jobs: List[Tuple[int, str]] = []
        merged_by_rank: Dict[int, str] = {}

        for rank in range(1, min(self.max_guidelines, len(results)) + 1):
            bullets_map = chunk_bullets_by_rank.get(rank, {})
            if not bullets_map:
                merged_by_rank[rank] = "No extractable diagnostic/test-ordering points."
                continue

            # 按 cidx 排序, 保持稳定输出
            chunk_bullets = [bullets_map[i] for i in sorted(bullets_map.keys())]

            if len(chunk_bullets) == 1:
                merged_by_rank[rank] = chunk_bullets[0]
                continue

            bullets_text = "\n\n".join(
                f"Chunk {i}:\n{b}" for i, b in enumerate(chunk_bullets, 1)
            )
            merge_user = MERGE_PROMPT.format(query=query, bullets=bullets_text)
            merge_prompt = self._wrap_with_tags(system_text, merge_user)

            # merge prompt 太长就压缩 bullets
            if self._token_len(merge_prompt) > (
                getattr(self.llm, "max_context_length", 8192) - self.reserve_output_tokens
            ):
                flat_lines: List[str] = []
                for b in chunk_bullets:
                    flat_lines.extend([ln for ln in b.splitlines() if ln.strip()])
                flat_lines = flat_lines[:40]
                bullets_text = "\n".join(flat_lines)
                merge_user = MERGE_PROMPT.format(query=query, bullets=bullets_text)
                merge_prompt = self._wrap_with_tags(system_text, merge_user)

            merge_jobs.append((rank, merge_prompt))

        if merge_jobs:
            merge_prompts = [p for _, p in merge_jobs]
            merge_outs = self._call_llm_batch(merge_prompts)

            for (rank, _), out in zip(merge_jobs, merge_outs):
                merged = (out or "").strip()
                merged_lines = [ln.strip() for ln in merged.splitlines() if ln.strip()]
                merged = "\n".join(merged_lines[:14]).strip()
                merged_by_rank[rank] = merged or "No extractable diagnostic/test-ordering points."

        # 4) 拼最终 blocks
        blocks: List[str] = []
        for rank in range(1, min(self.max_guidelines, len(results)) + 1):
            meta = guideline_meta.get(rank, {})
            blocks.append(
                f"Guideline {rank} (score={meta.get('score','0.0000')})\n"
                f"Title: {meta.get('title','Unknown title')}\n"
                f"Year: {meta.get('year','')}\n"
                f"Key points:\n{merged_by_rank.get(rank,'No extractable diagnostic/test-ordering points.')}"
            )

        return "\n\n---\n\n".join(blocks)


def main() -> None:
    store = GuidelineStore(persist_dir=DEFAULT_INDEX_DIR)
    store.build_from_json(GUIDELINE_JSON_PATH)
    store.save()
    print(f"Saved guideline FAISS index to: {DEFAULT_INDEX_DIR}")


if __name__ == "__main__":
    main()
