# 新增：经验库封装
from typing import Optional, List, Dict, Any, Type
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# 新增：经验检索工具
import copy
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class ExperienceStore:
    """
    Thin wrapper around LangChain FAISS vector store to add and query experience notes.
    """
    def __init__(self, embeddings=None, persist_path: Optional[str] = None):
        # 默认使用本地句嵌入模型；可替换为任意 embeddings（OpenAIEmbeddings 等）
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vs: Optional[FAISS] = None
        self.persist_path = persist_path
        self.retrieval_records: List[Dict[str, Any]] = []
        self._next_exp_seq: int = 0

        # 若有持久化目录，尝试加载
        if self.persist_path:
            try:
                self.vs = FAISS.load_local(
                    self.persist_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                try:
                    self._next_exp_seq = int(getattr(self.vs.index, "ntotal", 0))
                except Exception:
                    self._next_exp_seq = 0
            except Exception:
                self.vs = None

    # def init_exp(self, experiences: List[str]):
    #     for text in experiences:
    #         text = (text or "").strip()
    #         if not text:
    #             continue
    #         if self.vs is None:
    #             self.vs = FAISS.from_texts([text], self.embeddings, metadatas=None)
    #         else:
    #             self.vs.add_texts([text], metadatas=None)

    def add_experience(self, metadata: Optional[Dict[str, Any]] = None):
        summary = metadata.get("summary", "").strip()
        text = summary
        parts = summary.split("\n")
        for p in parts:
            if p.startswith("Experience Pattern:"):
                text = p[len("Experience Pattern: ") :].strip()
                break

        if not text:
            return
        metadata["exp_seq"] = int(metadata.get("exp_seq") or self._next_exp_seq)
        self._next_exp_seq = max(self._next_exp_seq, metadata["exp_seq"] + 1)
        metadatas = [metadata or {}]
        if self.vs is None:
            self.vs = FAISS.from_texts([text], self.embeddings, metadatas=metadatas)
        else:
            self.vs.add_texts([text], metadatas=metadatas)

    def record_retrieval(self, query: str, results_metadata: List[Dict[str, Any]]):
        rec = {
            "query": str(query),
            "results": results_metadata,
        }
        self.retrieval_records.append(rec)

    def get_records(self):
        res = copy.deepcopy(self.retrieval_records)
        self.retrieval_records = []
        return res
            
    def save(self):
        if self.persist_path and (self.vs is not None):
            self.vs.save_local(self.persist_path)

    def similarity_search(self, query: str, k: int = 5):
        if not self.vs:
            return []
        results = self.vs.similarity_search_with_score(query, k=k)
        # 返回 (Document, score)
        metas: List[Dict[str, Any]] = []
        for (doc, score) in results:
            md = dict(getattr(doc, "metadata", {}) or {})
            md["score"] = float(score)
            metas.append(md)
        self.record_retrieval(query, metas)
        return results

    def filter_by_exp_seq(self, start: Optional[int], end: Optional[int], new_persist_path: Optional[str] = None):

        # return a NEW store (persist_path=None), do not overwrite the original index on disk
        new_store = ExperienceStore(embeddings=self.embeddings, persist_path=new_persist_path)
        new_store.retrieval_records = []
        new_store._next_exp_seq = 1

        if self.vs is None:
            return new_store

        ids = list(getattr(self.vs, "index_to_docstore_id", []) or [])
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for doc_id in ids:
            real_id = self.vs.index_to_docstore_id.get(doc_id)
            doc = self.vs.docstore.search(real_id)
            if doc is None:
                continue
            md = dict(getattr(doc, "metadata", {}) or {})
            seq = md.get("exp_seq", None)
            try:
                seq = int(seq)
            except Exception:
                continue

            if start is not None and seq < start:
                continue
            if end is not None and seq >= end:
                continue

            text = (getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue

            texts.append(text)
            metas.append(md)

        if texts:
            new_store.vs = FAISS.from_texts(texts, self.embeddings, metadatas=metas)
            max_seq = max(int(m.get("exp_seq", 0)) for m in metas)
            new_store._next_exp_seq = max_seq + 1

        return new_store

class ExperienceSearchInput(BaseModel):
    action_input: str = Field(description="Text query for retrieving experience notes.")

class ExperienceSearchTool(BaseTool):
    name: str = "Experience Search"
    description: str = "Retrieve concise diagnostic experience from similar past clinical cases."
    args_schema: Type[BaseModel] = ExperienceSearchInput
    k: int = 10

    # ✅ 将 store 显式声明为字段（避免 __init__ 动态赋值触发 Pydantic 错误）
    experience_store: ExperienceStore

    class Config:
        # pydantic v1 config（v2 会忽略，不影响）
        arbitrary_types_allowed = True

    def _run(self, action_input: str) -> str:
        results = self.experience_store.similarity_search(action_input, k=self.k)
        if not results:
            # 没有匹配经验时, 返回空字符串, 让上层自己决定提示
            return "This is the first case. No experience found."

        cases = []
        for i, (doc, _) in enumerate(results, 1):
            # 优先用 metadata 里的 summary
            summary = ""
            if getattr(doc, "metadata", None):
                summary = (doc.metadata.get("summary") or "").strip()
            

            # 如果 summary 为空, 可以选择 fallback 到 page_content, 也可以直接跳过
            if not summary:
                # fallback 版本:
                summary = (getattr(doc, "page_content", "") or "").strip()

            if not summary:
                continue

            cases.append(f"Case {i}:\n{summary}")

        # 如果全是空的, 也返回空串
        if not cases:
            return "This is the first case. No experience found."
        tails = ""
        if len(cases) < self.k:
            tails = f"\n\nOnly {len(cases)} experiences in the knowledge base."

        return "\n\n-----\n\n".join(cases) + tails

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("ExperienceSearchTool does not support async.")