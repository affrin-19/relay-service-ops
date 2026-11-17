import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, OPENAI_API_KEY


class LocalFallbackEmbeddingFunction(chromadb.EmbeddingFunction):
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'is', 'to', 'in', 'of', 'it', 'for', 'on', 'with',
        'as', 'at', 'by', 'this', 'that', 'i', 'you', 'my', 'your', 'we', 'be', 'do',
        'does', 'did', 'have', 'has', 'had', 'can', 'could', 'should', 'would', 'how',
        'what', 'where', 'when', 'why', 'who', 'me', 'all', 'from', 'so',
        'not', 'no', 'are', 'were', 'was', 'been', 'being', 'having', 'into', 'out', 'up', 'down', 'cannot', 'cant'
    }

    DOMAIN_BOOSTS = {
        'vpn': 4.0, 'acmeconnect': 4.0, 'gateway': 3.0,
        'outlook': 4.5, 'email': 3.5, 'exchange': 3.0, 'webmail': 3.0, 'mailbox': 3.0,
        'password': 4.0, 'lockout': 4.0, 'sso': 3.0, 'okta': 3.0, 'reset': 3.0,
        'wifi': 4.0, 'hardware': 3.0, 'laptop': 3.0, 'monitor': 3.0, 'screen': 2.5,
        'opening': 3.0, 'open': 3.0, 'launch': 3.0, 'crash': 3.0, 'safe': 3.0, 'hang': 2.5, 'freeze': 2.5,
        'disconnect': 2.5, 'disconnected': 2.5, 'credentials': 2.5, 'token': 2.5
    }

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        import re
        import math
        import zlib
        embeddings = []
        for text in input_texts:
            vec = [0.0] * 128
            words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
            for w in words:
                if w in self.STOP_WORDS:
                    continue
                weight = self.DOMAIN_BOOSTS.get(w, 1.0)
                bucket = zlib.crc32(w.encode("utf-8")) % 128
                vec[bucket] += weight
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings


class USMVectorStore:
    COLLECTION_NAME = "usm_knowledge_base"

    def __init__(self, persist_dir: Optional[str] = None, api_key: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIRECTORY
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self._get_or_create_collection()

    def _get_embedding_function(self):
        use_openai_emb = os.getenv("USE_OPENAI_EMBEDDINGS", "False").lower() == "true"
        effective_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if use_openai_emb and effective_key and effective_key.strip().startswith("sk-"):
            try:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=effective_key.strip(),
                    model_name=EMBEDDING_MODEL_NAME
                )
            except Exception:
                pass
        return LocalFallbackEmbeddingFunction()

    def _get_or_create_collection(self):
        ef = self._get_embedding_function()
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

    def set_api_key(self, new_api_key: str):
        self.api_key = new_api_key.strip()
        self.collection = self._get_or_create_collection()

    def is_document_indexed(self, filename: str) -> bool:
        try:
            results = self.collection.get(where={"source": filename}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        overwrite: bool = False
    ) -> Tuple[int, str]:
        if not chunks:
            return 0, "No chunks provided to index."

        source_doc = chunks[0]["metadata"].get("source", "unknown")

        if self.is_document_indexed(source_doc):
            if not overwrite:
                return 0, f"Document '{source_doc}' is already indexed. Enable overwrite to re-index."
            self.delete_document(source_doc)

        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            chunk_id = f"{meta.get('source', 'doc')}_p{meta.get('page', 1)}_c{idx + 1}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(meta)

        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            return len(ids), f"Successfully indexed {len(ids)} chunks from '{source_doc}'."
        except Exception as e:
            raise RuntimeError(f"ChromaDB ingestion failed for '{source_doc}': {str(e)}")

    def similarity_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        count = self.collection.count()
        if count == 0:
            return []

        k = min(top_k, count)
        retrieved = []

        try:
            results = self.collection.query(
                query_texts=[clean_query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            if results and results.get("documents") and len(results["documents"][0]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, dists):
                    similarity_score = max(0.0, round(1.0 - dist, 4))
                    retrieved.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity_score": similarity_score,
                        "source": meta.get("source", "Unknown Document"),
                        "page": meta.get("page", 1)
                    })
        except Exception:
            pass

        # Fall back to keyword matching if vector query missed results
        if len(retrieved) < k:
            try:
                import re
                raw_data = self.collection.get(include=["documents", "metadatas"])
                all_docs = raw_data.get("documents", [])
                all_metas = raw_data.get("metadatas", [])

                query_words = set(re.findall(r"[a-zA-Z0-9_-]+", clean_query.lower()))
                query_words = {w for w in query_words if w not in LocalFallbackEmbeddingFunction.STOP_WORDS and len(w) > 2}

                existing_texts = {r["text"] for r in retrieved}
                scored_fallbacks = []

                for doc, meta in zip(all_docs, all_metas):
                    if doc in existing_texts:
                        continue
                    doc_lower = doc.lower()
                    match_score = 0.0
                    for qw in query_words:
                        if qw in doc_lower:
                            weight = LocalFallbackEmbeddingFunction.DOMAIN_BOOSTS.get(qw, 1.5)
                            match_score += weight

                    if match_score > 0.0:
                        norm_score = min(0.85, 0.35 + (match_score * 0.1))
                        scored_fallbacks.append({
                            "text": doc,
                            "metadata": meta,
                            "distance": round(1.0 - norm_score, 4),
                            "similarity_score": round(norm_score, 4),
                            "source": meta.get("source", "Unknown Document"),
                            "page": meta.get("page", 1),
                            "_match_score": match_score
                        })

                scored_fallbacks.sort(key=lambda x: x["_match_score"], reverse=True)
                for item in scored_fallbacks[: (k - len(retrieved))]:
                    del item["_match_score"]
                    retrieved.append(item)

            except Exception:
                pass

        retrieved.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return retrieved[:k]

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        try:
            data = self.collection.get(include=["metadatas"])
            metas = data.get("metadatas", [])
            summary: Dict[str, Dict[str, Any]] = {}

            for m in metas:
                src = m.get("source", "Unknown")
                if src not in summary:
                    summary[src] = {
                        "source": src,
                        "chunk_count": 0,
                        "total_pages": m.get("total_pages", 1)
                    }
                summary[src]["chunk_count"] += 1

            return list(summary.values())
        except Exception:
            return []

    def delete_document(self, filename: str) -> bool:
        try:
            self.collection.delete(where={"source": filename})
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self._get_or_create_collection()
            return True
        except Exception:
            return False

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0
