import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:3b-cloud")
KNOWLEDGE_MODEL = os.getenv("OLLAMA_KNOWLEDGE_MODEL", "gpt-oss:120b-cloud")
ALLOW_EXTERNAL_KNOWLEDGE = os.getenv("ALLOW_EXTERNAL_KNOWLEDGE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


RECURSIVE_MAX_ROUNDS = max(0, _env_int("RECURSIVE_MAX_ROUNDS", 1))
CONTEXT_CHARS_PER_CHUNK = max(200, _env_int("CONTEXT_CHARS_PER_CHUNK", 700))

_CONFIDENCE_LEVELS = {"low": 0, "medium": 1, "high": 2}


def query_llm(messages: list, format: str = None, model: Optional[str] = None) -> Optional[str]:
    """
    Sends a chat request to the Ollama API.
    Falls back to a simple response if Ollama is unavailable.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": model or MODEL, "messages": messages, "stream": False}
    if format:
        payload["format"] = format

    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        print(f"Warning: Could not connect to Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        print(f"LLM Error: {e}")
        return None


def _parse_json_response(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Best-effort parser for JSON payloads returned by local models.
    """
    if not raw:
        return None

    text = raw.strip()
    candidates = [text]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalize_confidence(value: Any, default: str = "Low") -> str:
    if value is None:
        return default

    token = str(value).strip().lower()
    if "high" in token:
        return "High"
    if "med" in token:
        return "Medium"
    if "low" in token:
        return "Low"
    return default


def _confidence_rank(label: str) -> int:
    return _CONFIDENCE_LEVELS.get(str(label).strip().lower(), 0)


def _confidence_from_rank(rank: int) -> str:
    if rank >= 2:
        return "High"
    if rank == 1:
        return "Medium"
    return "Low"


def _most_conservative_confidence(*values: str) -> str:
    ranks = [_confidence_rank(v) for v in values if v]
    if not ranks:
        return "Low"
    return _confidence_from_rank(min(ranks))


def _extract_key_points(answer: str, fallback: Optional[List[str]] = None) -> List[str]:
    if fallback and isinstance(fallback, list):
        clean = [str(item).strip() for item in fallback if str(item).strip()]
        if clean:
            return clean[:3]

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    bullets = [line.lstrip("-* ").strip() for line in lines if line.startswith("-") or line.startswith("*")]
    if bullets:
        return bullets[:3]

    if not answer.strip():
        return []

    sentence = answer.split(".")[0].strip()
    return [sentence] if sentence else ["See analysis above"]


def _build_context_text(context: list) -> str:
    blocks = []
    for idx, item in enumerate(context, start=1):
        source = item.get("source", "unknown")
        role = item.get("role")
        entity = item.get("entity")
        content = str(item.get("content", "")).replace("\n", " ").strip()
        snippet = content[:CONTEXT_CHARS_PER_CHUNK]
        role_part = f" role={role}" if role else ""
        entity_part = f" entity={entity}" if entity else ""
        blocks.append(f"[{idx}] source={source}{role_part}{entity_part} text={snippet}")
    return "\n".join(blocks)


def _ask_structured(messages: list) -> Optional[Dict[str, Any]]:
    raw = query_llm(messages, format="json")
    parsed = _parse_json_response(raw)
    if parsed:
        return parsed

    raw = query_llm(messages)
    return _parse_json_response(raw)


def _ask_structured_with_model(messages: list, model: str) -> Optional[Dict[str, Any]]:
    raw = query_llm(messages, format="json", model=model)
    parsed = _parse_json_response(raw)
    if parsed:
        return parsed

    raw = query_llm(messages, model=model)
    return _parse_json_response(raw)


def _draft_answer(query: str, context_text: str) -> Dict[str, Any]:
    prompt = f"""
You are a precise financial analyst.
Use ONLY the context below. Do not invent facts.
Treat entries with source=memory as conversational hints, not factual evidence.
Financial facts, numbers, and claims must be grounded in non-memory sources.

Context:
{context_text}

Query:
{query}

Return strict JSON with this schema:
{{
  "answer": "string",
  "key_points": ["string", "string", "string"],
  "confidence": "Low|Medium|High"
}}

If the context is insufficient, say that explicitly in "answer" and set confidence to "Low".
"""
    messages = [{"role": "user", "content": prompt}]
    parsed = _ask_structured(messages)

    if parsed:
        answer = str(parsed.get("answer", "")).strip()
        key_points = _extract_key_points(answer, parsed.get("key_points"))
        confidence = _normalize_confidence(parsed.get("confidence"), default="Low")
        if answer:
            return {"answer": answer, "key_points": key_points, "confidence": confidence}

    fallback_response = query_llm(messages) or ""
    return {
        "answer": fallback_response.strip(),
        "key_points": _extract_key_points(fallback_response),
        "confidence": "Low",
    }


def _critique_answer(query: str, context_text: str, answer: str) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are a strict verifier.
Check whether the candidate answer is fully supported by the context.
Memory entries are lower-trust notes and do not count as evidence for factual claims unless corroborated by non-memory context.

Context:
{context_text}

Query:
{query}

Candidate answer:
{answer}

Return strict JSON with this schema:
{{
  "revise": false,
  "confidence": "Low|Medium|High",
  "issues": ["unsupported claim or missing point"]
}}

Set "revise" to true if anything is unsupported, contradictory, or missing from the query.
"""
    messages = [{"role": "user", "content": prompt}]
    parsed = _ask_structured(messages)
    if not parsed:
        return None

    issues = parsed.get("issues")
    if not isinstance(issues, list):
        issues = []
    issues = [str(item).strip() for item in issues if str(item).strip()]

    revise = bool(parsed.get("revise", False))
    confidence = _normalize_confidence(parsed.get("confidence"), default="Low")

    if issues and not revise:
        revise = True

    return {"revise": revise, "confidence": confidence, "issues": issues[:6]}


def _revise_answer(query: str, context_text: str, previous_answer: str, issues: List[str]) -> Optional[Dict[str, Any]]:
    issues_text = "\n".join([f"- {item}" for item in issues]) if issues else "- Improve grounding and completeness."
    prompt = f"""
You are revising an answer to maximize factual grounding.
Use ONLY the context and fix all listed issues.
Treat source=memory as non-authoritative context.

Context:
{context_text}

Query:
{query}

Previous answer:
{previous_answer}

Verifier issues:
{issues_text}

Return strict JSON with this schema:
{{
  "answer": "string",
  "key_points": ["string", "string", "string"],
  "confidence": "Low|Medium|High"
}}
"""
    messages = [{"role": "user", "content": prompt}]
    parsed = _ask_structured(messages)
    if not parsed:
        return None

    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        return None

    return {
        "answer": answer,
        "key_points": _extract_key_points(answer, parsed.get("key_points")),
        "confidence": _normalize_confidence(parsed.get("confidence"), default="Low"),
    }


def _looks_insufficient(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    markers = [
        "insufficient context",
        "not in the context",
        "not provided in the context",
        "does not include",
        "does not provide",
        "does not explicitly disclose",
        "not explicitly",
        "cannot be determined",
        "not available in the provided",
        "no standalone",
    ]
    return any(marker in a for marker in markers)


def _is_factoid_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    cue_tokens = [
        "what is",
        "how much",
        "revenue",
        "net income",
        "earnings",
        "eps",
        "q1",
        "q2",
        "q3",
        "q4",
        "2023",
        "2024",
    ]
    hits = sum(1 for token in cue_tokens if token in q)
    return hits >= 2


def _extract_company_hints(context: list) -> List[str]:
    pattern = re.compile(
        r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,6}\s+(?:Inc\.?|Ltd\.?|LLC|Corp\.?|Corporation|plc|PLC))"
    )
    hints = []
    for item in context:
        text = str(item.get("content", ""))[:1200]
        for match in pattern.findall(text):
            name = " ".join(match.split())
            if name and name not in hints:
                hints.append(name)
    return hints[:6]


def _query_mentions_any_company(query: str, companies: List[str]) -> bool:
    q = (query or "").lower()
    if not q:
        return False
    for c in companies:
        base = c.lower()
        if base in q:
            return True
        first_token = base.split(" ")[0]
        if first_token and first_token in q:
            return True
    return False


def _query_has_specific_entity_hint(query: str) -> bool:
    q = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    tokens = [t for t in q.split() if t]
    if not tokens:
        return False
    generic = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "how",
        "much",
        "revenue",
        "earnings",
        "income",
        "net",
        "q1",
        "q2",
        "q3",
        "q4",
        "in",
        "for",
        "of",
        "the",
        "a",
        "an",
        "year",
        "fiscal",
        "quarter",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "2026",
    }
    return any(token not in generic and not token.isdigit() for token in tokens)


def _knowledge_fallback(query: str) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are a financial fact assistant.
Answer from general public knowledge when possible.
If uncertain, say you are unsure instead of guessing.

Query:
{query}

Return strict JSON with this schema:
{{
  "answer": "string",
  "key_points": ["string", "string", "string"],
  "confidence": "Low|Medium|High"
}}
"""
    messages = [{"role": "user", "content": prompt}]
    parsed = _ask_structured_with_model(messages, model=KNOWLEDGE_MODEL)
    if not parsed:
        return None

    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        return None

    confidence = _normalize_confidence(parsed.get("confidence"), default="Low")
    if _confidence_rank(confidence) > _confidence_rank("Medium"):
        confidence = "Medium"

    labeled_answer = (
        f"{answer}\n\n"
        "Note: This value is from model general knowledge and may not be present in uploaded documents."
    )

    return {
        "answer": labeled_answer,
        "key_points": _extract_key_points(answer, parsed.get("key_points")),
        "confidence": confidence,
    }


def summarize_text(text: str) -> str:
    """
    Generates a concise summary of the provided text.
    """
    prompt = f"""
    You are an expert financial research assistant.
    Summarize the following text efficiently, preserving key numerical data, dates, and factual claims.
    Do not add any external information.

    Text:
    {text}

    Summary:
    """
    messages = [{"role": "user", "content": prompt}]
    result = query_llm(messages)
    return result if result else "Failed to generate summary. Make sure Ollama is running."


def analyze_query(query: str, context: list) -> Dict[str, Any]:
    """
    Analyzes a user query based on retrieved document chunks.
    Returns a structured JSON response.
    """
    if not context:
        return {"answer": "No documents found to analyze.", "key_points": [], "confidence": "Low"}

    context_text = _build_context_text(context)

    company_hints = _extract_company_hints(context)
    context_entities = [
        str(item.get("entity", "")).strip()
        for item in context
        if str(item.get("entity", "")).strip()
    ]
    unique_context_entities = []
    for entity in context_entities:
        if entity not in unique_context_entities:
            unique_context_entities.append(entity)

    distinct_sources = {
        str(item.get("source", "")).strip()
        for item in context
        if str(item.get("source", "")).strip() and str(item.get("source", "")).strip().lower() != "memory"
    }
    known_companies = []
    for c in company_hints + unique_context_entities:
        if c and c not in known_companies:
            known_companies.append(c)
    explicit_company_in_query = _query_mentions_any_company(query, known_companies)

    if (
        _is_factoid_query(query)
        and (
            (
                len(company_hints) > 1
                and not explicit_company_in_query
            )
            or (
                len(distinct_sources) > 1
                and len(unique_context_entities) != 1
                and not explicit_company_in_query
            )
        )
    ):
        sample = ", ".join(company_hints[:3])
        if not sample:
            sample = ", ".join(unique_context_entities[:3]) or ", ".join(sorted(list(distinct_sources))[:3])
        return {
            "answer": (
                f"Your question is ambiguous across multiple companies in the retrieved context ({sample}). "
                "Please specify the company name or select a single source document."
            ),
            "key_points": [
                "Multiple company entities were detected in retrieved context.",
                "Factoid financial queries need explicit company disambiguation.",
                "Select one source or include company name in the question.",
            ],
            "confidence": "Low",
        }

    current = _draft_answer(query, context_text)

    if not current.get("answer"):
        return {
            "answer": "Error: Could not connect to Ollama. Make sure it's running with: ollama serve",
            "key_points": [],
            "confidence": "Low",
        }

    # Recursive self-check loop (bounded by RECURSIVE_MAX_ROUNDS).
    revision_confidence: Optional[str] = None
    for _ in range(RECURSIVE_MAX_ROUNDS):
        critique = _critique_answer(query, context_text, current.get("answer", ""))
        if not critique:
            break

        if not critique.get("revise", False):
            break

        critique_confidence = critique.get("confidence", "Low")
        revision_confidence = (
            _most_conservative_confidence(revision_confidence, critique_confidence)
            if revision_confidence
            else _normalize_confidence(critique_confidence, default="Low")
        )

        revised = _revise_answer(
            query=query,
            context_text=context_text,
            previous_answer=current.get("answer", ""),
            issues=critique.get("issues", []),
        )
        if not revised:
            break

        revised["confidence"] = _most_conservative_confidence(
            revised.get("confidence", "Low"),
            critique_confidence,
        )

        if revised.get("answer", "").strip() == current.get("answer", "").strip():
            current = revised
            break

        current = revised

    final_confidence = _normalize_confidence(current.get("confidence", "Low"), default="Low")
    if revision_confidence:
        final_confidence = _most_conservative_confidence(final_confidence, revision_confidence)

    needs_external = _looks_insufficient(current.get("answer", "")) or (
        final_confidence == "Low" and _is_factoid_query(query)
    )
    if ALLOW_EXTERNAL_KNOWLEDGE and needs_external:
        fallback_query = query
        if len(unique_context_entities) == 1 and not explicit_company_in_query:
            fallback_query = f"{query} for {unique_context_entities[0]}"
        fallback = _knowledge_fallback(fallback_query)
        if fallback and not _looks_insufficient(fallback.get("answer", "")):
            current = fallback
            final_confidence = _normalize_confidence(fallback.get("confidence", "Low"), default="Low")

    return {
        "answer": current.get("answer", ""),
        "key_points": current.get("key_points", []) or _extract_key_points(current.get("answer", "")),
        "confidence": final_confidence,
    }
