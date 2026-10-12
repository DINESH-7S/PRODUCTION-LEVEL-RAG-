import os
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from schemas import QueryResponse, ContextPassage, RAGSearchResult
from parser import EmbeddingGenerator, MODEL_DIMENSIONS, DEFAULT_ENCODER
from vector_db import QdrantEngine

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

RAG_PROMPT_SYSTEM = """You are a highly capable document intelligence assistant. Your objective is to formulate precise answers by carefully parsing and synthesizing the retrieved context details listed below.

INSTRUCTIONS:
1. Cross-reference data points from multiple passages to build structured, comprehensive, and accurate answers. Identify core themes and links.
2. If the provided reference context does not contain enough detailed information, explicitly declare: "The provided documents do not contain enough information on this topic." Do not extrapolate, but do present any partial facts that are validated by the context.
3. Absolutely do not invent details or mention claims that aren't backed up by the provided reference text.
4. Cite the corresponding document name and page number for your statements (e.g., [Document: filename.pdf, Page: X]). Ensure inline references are placed throughout your explanation.

{length_instruction}
"""

def generate_queries(question: str, model_id: str = "llama-3.1-8b-instant") -> List[str]:
    """Uses LLM to output variations of the search query to improve recall."""
    if not groq_key:
        return [question, f"Details about {question}", f"Overview of {question}"]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a search query rewriting expert. Produce 3 alternative versions of the user's question to retrieve relevant information. Return ONLY a valid JSON list of 3 strings."
                },
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=200,
        )
        data = response.choices[0].message.content.strip()
        if data.startswith("```"):
            data = data.split("```")[1].strip()
            if data.startswith("json"):
                data = data[4:].strip()
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return [str(item) for item in parsed] + [question]
    except Exception:
        pass
    return [question, f"Details about {question}", f"Overview of {question}"]

def execute_duckduckgo_fallback(query: str) -> List[ContextPassage]:
    """Queries DuckDuckGo HTML search for external text snippets when local confidence is low."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(request, timeout=5) as resp:
            raw_html = resp.read().decode("utf-8")

        soup = BeautifulSoup(raw_html, "html.parser")
        snippets = soup.find_all("a", class_="result__snippet", limit=3)
        results = []
        for idx, snip in enumerate(snippets):
            txt = snip.get_text(strip=True)
            if txt:
                results.append(
                    ContextPassage(
                        content=txt,
                        doc_id="Web Search (DuckDuckGo Fallback)",
                        page_num=1,
                        seq_index=idx,
                        relevance_score=0.5
                    )
                )
        if results:
            return results
    except Exception:
        pass

    return [
        ContextPassage(
            content="No relevant web context could be retrieved.",
            doc_id="Web Search Fallback",
            page_num=1,
            seq_index=0,
            relevance_score=0.0
        )
    ]

class AdaptiveRAGRetriever(BaseRetriever):
    """Custom LangChain retriever mapping custom search strategies and CRAG fallback."""
    db_engine: Any
    strategy: str = "dense"
    top_k: int = 5
    embedding_model: str = DEFAULT_ENCODER
    enable_crag: bool = True
    fallback_threshold: float = 0.02

    cached_results: Optional[RAGSearchResult] = None
    web_fallback_triggered: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        variant_vectors = None
        if self.strategy == "multi_query":
            variants = generate_queries(query)
            variant_vectors = EmbeddingGenerator.embed_texts(variants, model_name=self.embedding_model)
            query_vector = variant_vectors[0]
        else:
            query_vector = EmbeddingGenerator.embed_texts([query], model_name=self.embedding_model)[0]

        search_res = self.db_engine.search(
            query=query,
            query_vector=query_vector,
            strategy=self.strategy,
            top_k=self.top_k,
            variants_vectors=variant_vectors
        )

        self.cached_results = search_res
        passages = search_res.passages
        self.web_fallback_triggered = False

        highest_score = max([p.relevance_score for p in passages], default=0.0)

        # Thresholds per search strategy
        threshold_map = {
            "dense": 0.02,
            "mmr": 0.02,
            "hybrid": 0.01,
            "reranker": -5.0,
            "multi_query": 0.02,
        }
        active_threshold = threshold_map.get(self.strategy.lower(), self.fallback_threshold)

        if self.enable_crag and (not passages or highest_score < active_threshold):
            web_passages = execute_duckduckgo_fallback(query)
            if web_passages:
                passages = web_passages
                self.web_fallback_triggered = True

        truncated_passages = passages[:self.top_k]
        self.cached_results.passages = truncated_passages

        docs = []
        for idx, passage in enumerate(truncated_passages, 1):
            docs.append(
                Document(
                    page_content=passage.content,
                    metadata={
                        "source": passage.doc_id,
                        "page": passage.page_num,
                        "score": passage.relevance_score,
                        "chunk_id": passage.seq_index,
                        "idx": idx
                    }
                )
            )
        return docs

def format_passages(documents: List[Document]) -> str:
    """Formats document objects into text context string for Prompt ingestion."""
    blocks = []
    for doc in documents:
        idx = doc.metadata.get("idx", 1)
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        header = f"[Passage {idx} | Document: {source} | Page: {page}]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)

class RAGCoordinator:
    """Manages the full RAG pipeline execution."""
    def __init__(self, collection_name: str = "docs"):
        self.collection_name = collection_name

    def process_query(
        self,
        question: str,
        strategy: str = "dense",
        top_k: int = 5,
        embedding_model: str = DEFAULT_ENCODER,
        chat_history: Optional[List[Dict[str, str]]] = None,
        enable_crag: bool = True,
        answer_length: str = "Detailed Analysis"
    ) -> QueryResponse:
        """Executes search, contextual prompt construction, and ChatGroq generation."""
        if not groq_key:
            return QueryResponse(
                answer="⚠️ **Configuration Error:** Groq API Key is not set.",
                sources=[],
                context_count=0,
                passages=[],
                match_confidence=0.0,
                web_fallback_used=False
            )

        if embedding_model not in MODEL_DIMENSIONS:
            embedding_model = DEFAULT_ENCODER

        v_dim = MODEL_DIMENSIONS[embedding_model]
        db_engine = QdrantEngine(collection_name=self.collection_name, vector_dim=v_dim)

        retriever = AdaptiveRAGRetriever(
            db_engine=db_engine,
            strategy=strategy,
            top_k=top_k,
            embedding_model=embedding_model,
            enable_crag=enable_crag
        )

        length_instruction = (
            "Give a very concise answer in a few sentences."
            if answer_length == "Concise Summary"
            else "Provide a comprehensive, high-depth analysis."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_PROMPT_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Provided Context Passages:\n\n{context}\n\nUser Question: {question}")
        ])

        llm = ChatGroq(
            api_key=groq_key,
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            max_tokens=1024
        )

        chain = prompt | llm | StrOutputParser()

        # Format conversation memory from input history
        memory = []
        if chat_history:
            for turn in chat_history[-4:]:
                content = turn.get("content", "")
                if not content or "⚠️" in content or "API Error" in content:
                    continue
                if turn.get("role") == "user":
                    memory.append(HumanMessage(content=content))
                elif turn.get("role") == "assistant":
                    memory.append(AIMessage(content=content))

        docs = retriever.invoke(question)
        context_str = format_passages(docs)

        try:
            answer = chain.invoke({
                "context": context_str,
                "question": question,
                "chat_history": memory,
                "length_instruction": length_instruction
            })
        except Exception as err:
            try:
                # Use Llama 3.1 8b as backup LLM
                fallback_llm = ChatGroq(
                    api_key=groq_key,
                    temperature=0.3,
                    model_name="llama-3.1-8b-instant",
                    max_tokens=1024
                )
                fallback_chain = prompt | fallback_llm | StrOutputParser()
                answer = fallback_chain.invoke({
                    "context": context_str,
                    "question": question,
                    "chat_history": memory,
                    "length_instruction": length_instruction
                })
            except Exception as b_err:
                answer = f"⚠️ **Groq API Error:** {err}"

        # Re-map LangChain docs to ContextPassage list
        passages = []
        for doc in docs:
            passages.append(
                ContextPassage(
                    content=doc.page_content,
                    doc_id=doc.metadata.get("source", "Unknown"),
                    page_num=doc.metadata.get("page", 1),
                    seq_index=doc.metadata.get("chunk_id", 0),
                    relevance_score=doc.metadata.get("score", 0.0)
                )
            )

        sources_list = list({f"{p.doc_id} (Page {p.page_num})" for p in passages})
        highest_score = max([p.relevance_score for p in passages], default=0.0)

        if not passages:
            answer = "I don't know based on the provided documents."

        return QueryResponse(
            answer=answer,
            sources=sources_list,
            context_count=len(passages),
            passages=passages,
            match_confidence=round(float(highest_score), 4),
            web_fallback_used=retriever.web_fallback_triggered
        )
