import json
from typing import Dict, List

import requests

from contextwalker.agent.neighbors import NEIGHBOR_TOOL, fetch_neighbor_chunks
from contextwalker.config import (
    MAX_AGENT_STEPS,
    MAX_NEIGHBOR_RADIUS,
    MAX_TOOL_CALLS,
    OLLAMA_CHAT_URL,
    OLLAMA_MODEL,
)
from contextwalker.schema import Chunk


def build_initial_agent_context(
    final_results: List[Dict],
    chunk_lookup: Dict[int, Chunk],
) -> str:
    blocks = []

    for rank, result in enumerate(final_results, start=1):
        chunk_id = result["chunk_id"]
        chunk = chunk_lookup[chunk_id]
        block = f"""
============================================================
INITIAL RETRIEVAL RESULT #{rank}
============================================================

Chunk ID: {chunk.chunk_id}

Page: {chunk.page}

RRF Score:
{result.get("rrf_score", "N/A")}

Reranker Score:
{result.get("reranker_score", "N/A")}

Contextual Description:

{chunk.context}

Original Text:

{chunk.text}
"""
        blocks.append(block)

    return "\n\n".join(blocks)


def agent_answer(
    query: str,
    final_results: List[Dict],
    chunk_lookup: Dict[int, Chunk],
    max_agent_steps: int = MAX_AGENT_STEPS,
) -> str:
    initial_context = build_initial_agent_context(final_results, chunk_lookup)
    all_chunk_ids = sorted(chunk_lookup.keys())
    min_chunk_id = all_chunk_ids[0]
    max_chunk_id = all_chunk_ids[-1]
    tool_call_count = 0

    system_prompt = f"""
You are an autonomous document exploration agent
inside a Retrieval Augmented Generation system.

The user asks a question.

You initially receive the five best chunks found by
hybrid vector search, BM25, Reciprocal Rank Fusion,
and neural reranking.

THE TOP FIVE CHUNKS ARE ONLY STARTING POINTS.

You are NOT restricted to those chunks.

You have access to:

fetch_neighbor_chunks(chunk_id, radius)

The document has chunk IDs from approximately:

{min_chunk_id}

through:

{max_chunk_id}


============================================================
HOW YOU SHOULD EXPLORE
============================================================

You may call fetch_neighbor_chunks repeatedly.

A chunk returned by a previous tool call can itself
become the center of another tool call.

For example:

Initial chunk 100

→ fetch chunk 100 radius 2

returns:

98, 99, 100, 101, 102


If chunk 102 looks more relevant:

→ fetch chunk 102 radius 3


If chunk 105 then looks important:

→ fetch chunk 105 radius 4


You can therefore WALK THROUGH THE DOCUMENT:

100
→ 102
→ 105
→ 109
→ 113
→ ...

You can move:

- forward
- backward
- between sections
- around multiple initial retrieval results

You may revisit an area if necessary.

Do not assume that the answer must be contained
inside the original Top 5.


============================================================
WHEN TO CONTINUE EXPLORING
============================================================

Continue using the tool when:

- a chunk looks partially relevant
- the explanation appears to continue
- the beginning of the explanation is missing
- a list continues into another chunk
- a procedure spans multiple chunks
- a definition references earlier material
- a section heading suggests more relevant text nearby
- a chunk says "continued", "following", "above",
  "below", "next", or similar
- you have clues but not enough evidence
- nearby chunks may provide the exact answer


============================================================
WHEN TO STOP
============================================================

Stop exploring when:

- you have sufficient evidence to directly answer
  the user's question

OR

- continued neighboring exploration is clearly
  unlikely to find useful information


============================================================
RADIUS
============================================================

Radius can be:

1
2
3
4
5

Use whatever radius makes sense.

You do NOT always have to start at radius 1.

You may use radius 5 when broader local context is useful.


============================================================
IMPORTANT RULES
============================================================

Use ONLY information from:

1. initial retrieved chunks
2. chunks obtained through the tool

Do not fabricate facts.

Do not answer merely because one chunk looks vaguely
related.

Explore neighboring chunks when doing so could materially
improve the answer.

The goal is to find enough documentary evidence before
answering.

When giving the final answer:

- directly answer the user's question
- synthesize information across relevant chunks
- mention page numbers where useful
- do not discuss hidden reasoning
- do not expose chain-of-thought
- do not invent unsupported information

If the evidence remains insufficient after reasonable
exploration, clearly state that the document does not
contain enough information.
"""

    user_prompt = f"""
USER QUESTION:

{query}


============================================================
INITIAL TOP-5 SEARCH RESULTS
============================================================

{initial_context}


Use these as STARTING POINTS.

You may freely explore neighboring chunks around any
chunk ID returned during exploration.

Continue retrieving neighboring evidence until you have
enough information to answer the question accurately.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(max_agent_steps):
        print("\n" + "=" * 70)
        print(f"[AGENT] Reasoning step {step + 1}/{max_agent_steps}")
        print(f"[AGENT] Total tool calls: {tool_call_count}/{MAX_TOOL_CALLS}")
        print("=" * 70)

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": [NEIGHBOR_TOOL],
            "stream": False,
            "options": {"temperature": 0},
        }
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        assistant_message = result["message"]
        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            final_answer = assistant_message.get("content", "")
            if final_answer.strip():
                return final_answer
            return "The agent stopped without producing a final answer."

        for tool_call in tool_calls:
            if tool_call_count >= MAX_TOOL_CALLS:
                print("[AGENT] Maximum tool calls reached.")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum "
                            "allowed number of document "
                            "tool calls. Use all evidence "
                            "collected so far and provide "
                            "the best supported final answer."
                        ),
                    }
                )
                break

            function = tool_call["function"]
            function_name = function["name"]
            arguments = function.get("arguments", {})

            # Some Ollama versions may return arguments as a JSON string.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}

            if function_name == "fetch_neighbor_chunks":
                try:
                    chunk_id = int(arguments.get("chunk_id"))
                except Exception:
                    chunk_id = -1

                try:
                    radius = int(arguments.get("radius", 1))
                except Exception:
                    radius = 1

                radius = max(1, min(radius, MAX_NEIGHBOR_RADIUS))
                tool_call_count += 1
                print("\n[TOOL CALL]")
                print(f"Center chunk: {chunk_id}")
                print(f"Radius: ±{radius}")

                tool_result = fetch_neighbor_chunks(
                    chunk_id=chunk_id,
                    radius=radius,
                    chunk_lookup=chunk_lookup,
                )
                returned_chunks = tool_result.get("chunks", [])

                if returned_chunks:
                    ids = [item["chunk_id"] for item in returned_chunks]
                    print(f"[TOOL RESULT] Returned chunk IDs: {ids}")
                else:
                    print(f"[TOOL RESULT] {tool_result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": "fetch_neighbor_chunks",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        if tool_call_count >= MAX_TOOL_CALLS:
            final_payload = {
                "model": OLLAMA_MODEL,
                "messages": messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "No more tool calls are "
                            "available. Based only "
                            "on the evidence collected "
                            "so far, provide the final "
                            "supported answer."
                        ),
                    }
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
            final_response = requests.post(
                OLLAMA_CHAT_URL,
                json=final_payload,
                timeout=300,
            )
            final_response.raise_for_status()
            final_data = final_response.json()
            return final_data["message"].get("content", "")

    print("\n[AGENT] Maximum reasoning steps reached.")
    messages.append(
        {
            "role": "user",
            "content": (
                "You have reached the maximum "
                "document exploration steps. "
                "Do not call additional tools. "
                "Using all evidence collected "
                "so far, provide the final "
                "supported answer to the "
                "original user question."
            ),
        }
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
    response.raise_for_status()
    final_result = response.json()
    return final_result["message"].get("content", "")

