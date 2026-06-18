---
layout: post
title: "How Agentic RAG Works: Part I"
date: 2026-06-18
categories: [AI, RAG, Agents, Deep-Dive]
---

*From single-pass retrieval to self-correcting agents — how RAG grows a brain.*

---

## 1. Quick Recap: What Plain RAG Does

Before understanding Agentic RAG, you need to understand what it replaces.

Standard RAG is a fixed, linear pipeline:

```
User question
     │
     ▼
Embed question  ──►  Retrieve top-k chunks  ──►  Build prompt  ──►  LLM  ──►  Answer
```

It runs **once**, in one direction, with no feedback. The model gets whatever chunks the retriever found — good or bad — and produces an answer. There is no mechanism to:

- Check whether the retrieved chunks were actually relevant
- Ask a follow-up retrieval if the first one missed
- Break a complex question into sub-questions
- Decide that retrieval isn't needed at all for a simple question

For simple factual queries, this works well. For anything complex — multi-part questions, ambiguous queries, questions that require synthesizing information across multiple topics — it falls short.

---

## 2. What Agentic RAG Is

Agentic RAG wraps the RAG pipeline inside an **agent loop**. Instead of a fixed sequence, the model becomes a decision-maker:

```
User question
     │
     ▼
┌─────────────────────────────────────────────────┐
│                   AGENT LOOP                    │
│                                                 │
│   Observe ──► Think ──► Act ──► Observe ──► ... │
│                                                 │
│   Actions available:                            │
│     • retrieve(query)   → fetch chunks          │
│     • rewrite(query)    → improve the question  │
│     • grade(chunk)      → score relevance       │
│     • web_search(query) → fallback to internet  │
│     • answer(response)  → return final answer   │
└─────────────────────────────────────────────────┘
```

The model decides **when** to retrieve, **what** to retrieve, **whether the result is good enough**, and **whether to try again**. Retrieval becomes one tool among many, not a fixed step in a pipeline.

---

## 3. Why the Agent Loop Changes Everything

In a standard agent (think ReAct), the model alternates between reasoning and acting:

```
Thought: I need to find information about X.
Action: retrieve("X")
Observation: [chunk 1, chunk 2, chunk 3]
Thought: chunk 2 is relevant but chunk 1 is off-topic. I still need Y.
Action: retrieve("Y specifically")
Observation: [chunk 4, chunk 5]
Thought: Now I have enough to answer.
Action: answer("Based on chunks 2, 4, 5...")
```

Each observation feeds back into the model's reasoning. The agent can:

- **Loop** — retrieve multiple times until it has what it needs
- **Branch** — take different paths depending on what it finds
- **Self-correct** — detect a bad retrieval and try a different query
- **Combine tools** — mix retrieval with web search, SQL, or code execution

This is fundamentally different from a pipeline. A pipeline is a recipe. An agent is a problem-solver.

---

## 4. Core Patterns in Agentic RAG

Agentic RAG is not one technique — it's a family of patterns. Here are the four most important ones.

### Pattern 1: Routing (Adaptive-RAG)

Not every question needs retrieval. A question like "what is 2 + 2?" doesn't need a vector database lookup. A question about a specific internal document does.

Adaptive-RAG adds a **router** at the front of the pipeline:

```
User question
     │
     ▼
┌─────────────┐
│   Router    │  ◄── small classifier model
└──────┬──────┘
       │
  ┌────┴────────────────┐──────────────┐
  ▼                     ▼              ▼
No RAG              Single-step    Multi-step
(direct answer)     RAG            RAG
                    (one           (agent
                    retrieval)     loop)
```

The router classifies the query by complexity and sends it down the appropriate path. Simple questions get fast direct answers. Hard questions get the full agent loop. This reduces latency and cost significantly on mixed workloads.

---

### Pattern 2: Retrieval Grading (Corrective RAG)

The retriever doesn't always return relevant chunks. CRAG adds a **grader** — a small model or LLM call — that scores each retrieved chunk before it reaches the main model:

```
Retrieve top-k chunks
        │
        ▼
┌───────────────┐
│  Relevance    │  scores each chunk: RELEVANT / IRRELEVANT / AMBIGUOUS
│  Grader       │
└──────┬────────┘
       │
  ┌────┴──────────────────┐
  ▼                       ▼
Relevant chunks        No relevant chunks found
  │                       │
  ▼                       ▼
Send to LLM           Web search fallback
                          │
                          ▼
                     Combine web results
                     with any partial
                     good chunks
                          │
                          ▼
                      Send to LLM
```

The grader acts as a quality gate. If chunks are poor, the system falls back to web search rather than feeding the LLM useless context. This makes the pipeline robust to retrieval failures instead of silently producing hallucinated answers.

---

### Pattern 3: Query Rewriting

The user's raw question is often a poor retrieval query. "What did they decide about the auth system?" is hard to match against a vector database. Rewriting it to "authentication system architecture decision" dramatically improves recall.

```
User question: "What did they decide about the auth?"
       │
       ▼
┌──────────────┐
│   Rewriter   │  LLM rewrites for retrieval
└──────┬───────┘
       │
       ▼
Rewritten query: "authentication system decision records"
       │
       ▼
   Retriever  ──►  Much better chunks
```

Rewriting can happen once upfront or iteratively — if retrieval still fails after one rewrite, the agent rewrites again with a different strategy.

---

### Pattern 4: Self-RAG — Retrieve, Reflect, Critique

Self-RAG is the most sophisticated pattern. The model is trained to emit special **reflection tokens** throughout generation:

| Token | Meaning |
|---|---|
| `[Retrieve]` | "I need to look something up before continuing" |
| `[IsREL]` | "This retrieved chunk is relevant to my question" |
| `[IsREL] No` | "This chunk is not relevant — discard it" |
| `[IsSUP]` | "My generated statement is supported by the context" |
| `[IsSUP] No` | "My statement contradicts the context — retract it" |
| `[IsUSE]` | "My overall response is useful" |

```
Question: "What causes transformer attention to scale quadratically?"

Model generates: "Transformer attention scales quadratically because..."
                          [Retrieve] ← model decides it needs a source
                              │
                    retrieve("transformer attention complexity")
                              │
                         [chunk returned]
                              │
                         [IsREL] Yes  ← chunk is relevant
                         [IsSUP] Yes  ← statement is supported
                              │
        "...each token attends to every other token, resulting in
         O(n²) memory and compute with respect to sequence length."
                         [IsUSE] Yes  ← final answer is useful
```

The model actively monitors its own generation. If it produces a claim not supported by the retrieved context, it catches and corrects it mid-stream rather than returning a hallucinated answer.

---

## 5. Multi-Hop Retrieval

Some questions can't be answered with a single retrieval. "What are the tradeoffs between the storage backends used by the top three vector databases?" requires:

1. Retrieve: which are the top three vector databases?
2. For each, retrieve: what storage backend does it use?
3. For each, retrieve: what are the tradeoffs of that backend?
4. Synthesize across all results.

This is **multi-hop retrieval** — each retrieval step depends on the output of the previous one:

```
Complex question
       │
       ▼
Sub-question ①: retrieve("top three vector databases")
       │
       ▼
Sub-question ②: retrieve("Pinecone storage backend")
       │
       ▼
Sub-question ③: retrieve("Weaviate storage backend")
       │
       ▼
Sub-question ④: retrieve("Milvus storage backend")
       │
       ▼
Synthesize: "Pinecone uses... Weaviate uses... Milvus uses..."
```

A flat single-pass RAG system cannot do this. Multi-hop requires the agent loop to carry intermediate results forward and generate new queries based on what it already found.

---

## 6. Agentic RAG vs. Plain RAG — Side by Side

| Capability | Plain RAG | Agentic RAG |
|---|---|---|
| Retrieval timing | Once, at query start | Any time during reasoning |
| Number of retrievals | Fixed (1) | Dynamic (1 to N) |
| Query used for retrieval | Raw user question | Rewritten, decomposed, or generated |
| Handles bad chunks | No — uses them anyway | Yes — grades and retries |
| Multi-hop questions | No | Yes |
| Can skip retrieval | No | Yes (via routing) |
| Self-correction | No | Yes (Self-RAG) |
| Latency | Low (one pass) | Higher (multiple LLM calls) |
| Cost | Low | Higher |

The tradeoff is straightforward: more intelligence costs more latency and money. The right choice depends on your query complexity and quality requirements.

---

## 7. How It's Built: LangGraph as the Backbone

The most common implementation pattern today uses **LangGraph** — a framework that models the agent as a directed graph of nodes and edges.

Each node is a function (retrieve, grade, rewrite, generate). Each edge is a condition (if grade is IRRELEVANT, go to web_search; otherwise go to generate).

```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)

graph.add_node("retrieve",      retrieve_node)
graph.add_node("grade_chunks",  grade_node)
graph.add_node("rewrite_query", rewrite_node)
graph.add_node("web_search",    web_search_node)
graph.add_node("generate",      generate_node)

graph.add_conditional_edges(
    "grade_chunks",
    decide_next_step,       # returns "generate", "rewrite", or "web_search"
    {
        "generate":   "generate",
        "rewrite":    "rewrite_query",
        "web_search": "web_search",
    }
)
```

The graph handles control flow. Each node only needs to know its own job. The state object carries retrieved chunks, grades, and the current query between nodes.

---

## 8. When to Use Agentic RAG

Agentic RAG is not always the right tool. Use the following as a guide:

| Use case | Recommendation |
|---|---|
| Simple factual Q&A over a small corpus | Plain RAG is sufficient |
| Questions that frequently miss on first retrieval | Add query rewriting |
| Workload with mixed simple/complex questions | Add routing (Adaptive-RAG) |
| Retrieval quality is variable or corpus is noisy | Add retrieval grading (CRAG) |
| Questions require synthesizing multiple topics | Add multi-hop + agent loop |
| Hallucination is unacceptable | Add self-reflection (Self-RAG) |
| Mix of internal docs + live web data | Add web search fallback |

Start simple. Add agentic components only where a specific failure mode demands it.

---

## Summary

| Concept | What it means |
|---|---|
| Plain RAG | One-pass: retrieve → prompt → generate |
| Agentic RAG | Agent loop: model decides when and what to retrieve |
| Routing | Classify query complexity; skip retrieval for simple questions |
| Retrieval grading | Score chunks for relevance before passing to LLM |
| Query rewriting | Rewrite user question into a better retrieval query |
| Self-RAG | Model emits reflection tokens to grade its own outputs mid-generation |
| Multi-hop | Chain retrievals — each step informs the next query |
| LangGraph | Graph-based framework for implementing these patterns |

Agentic RAG closes the gap between what a language model knows and what it can reliably answer. By treating retrieval as a dynamic, self-correcting loop rather than a fixed pipeline step, it enables a class of question-answering that plain RAG simply cannot support.

---

## Resources

### Foundational Papers

| Paper | Authors | Link |
|---|---|---|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al., 2020 | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) |
| Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Asai et al., 2023 | [arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) |
| FLARE: Active Retrieval Augmented Generation | Jiang et al., 2023 | [arxiv.org/abs/2305.06983](https://arxiv.org/abs/2305.06983) |
| Corrective Retrieval Augmented Generation (CRAG) | Yan et al., 2024 | [arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884) |
| Adaptive-RAG | Jeong et al., 2024 | [arxiv.org/abs/2403.14403](https://arxiv.org/abs/2403.14403) |

### Blog Posts

| Post | Source |
|---|---|
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Lilian Weng |
| [Agentic RAG with LlamaIndex](https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex) | LlamaIndex Blog |
| [Agentic RAG with LangGraph](https://blog.langchain.dev/agentic-rag-with-langgraph/) | LangChain Blog |
| [RAG from Scratch](https://blog.langchain.dev/rag-from-scratch/) | LangChain Blog |

### Repositories

| Repository | Purpose |
|---|---|
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Graph-based agent orchestration — the standard for implementing agentic RAG patterns |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Core RAG primitives: loaders, splitters, retrievers, chains |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | High-level RAG framework with first-class agentic support |
| [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Pipeline-based RAG framework for production deployments |

---

*Part II will cover implementation: building a CRAG + Adaptive-RAG pipeline end-to-end with LangGraph, a local embedding model, and ChromaDB.*
