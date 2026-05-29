"""OpenAPI 3.x and GraphQL SDL schema trimmer using BM25 relevance scoring."""
from __future__ import annotations

import json
import re
from typing import Any


def _bm25_score_text(query_terms: list[str], text: str) -> float:
    """Lightweight BM25-style score (no IDF — single-document variant for ranking)."""
    import math
    from collections import Counter

    stopwords = frozenset(
        "a an the is are was were be been have has had do does will would could "
        "should may might can i you he she it we they this that and or but not of "
        "in on at to for with by from as into about".split()
    )
    doc_terms = [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in stopwords]
    tf = Counter(doc_terms)
    dl = len(doc_terms)
    if dl == 0:
        return 0.0
    k1, b, avgdl = 1.5, 0.75, 20.0
    score = 0.0
    for term in set(query_terms):
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        score += (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
    return score


def _score_operation(path: str, method: str, op: dict, query_terms: list[str]) -> float:
    text = f"{method} {path} {op.get('summary', '')} {op.get('description', '')} {' '.join(op.get('tags', []))}"
    return _bm25_score_text(query_terms, text)


def _strip_operation(op: dict) -> dict:
    """Remove low-value fields from a kept operation."""
    out = {}
    for k, v in op.items():
        if k in ("examples", "example", "x-codeSamples") or k.startswith("x-"):
            continue
        if k == "description" and isinstance(v, str) and len(v) > 80:
            out[k] = v[:80] + "…"
        elif k == "responses":
            # Drop 422 response schemas; keep others
            out[k] = {
                code: _strip_response(resp)
                for code, resp in (v or {}).items()
                if code != "422"
            }
        else:
            out[k] = v
    return out


def _strip_response(resp: dict) -> dict:
    out = {}
    for k, v in resp.items():
        if k in ("examples", "example") or k.startswith("x-"):
            continue
        if k == "description" and isinstance(v, str) and len(v) > 80:
            out[k] = v[:80] + "…"
        else:
            out[k] = v
    return out


def trim_openapi(schema: dict, task: str, top_k: int = 10) -> dict:
    """Return a minimal OpenAPI dict with only the top-K operations relevant to *task*."""
    stopwords = frozenset(
        "a an the is are was were be have has had do does will would could "
        "should may might can i you he she it we they this that and or but not of "
        "in on at to for with by from as into about".split()
    )
    query_terms = [w for w in re.findall(r"\b\w+\b", task.lower()) if w not in stopwords]

    paths = schema.get("paths") or {}
    HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}

    scored: list[tuple[float, str, str, dict]] = []
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method not in HTTP_METHODS or not isinstance(op, dict):
                continue
            if op.get("deprecated"):
                continue
            score = _score_operation(path, method, op, query_terms)
            scored.append((score, path, method, op))

    scored.sort(key=lambda x: x[0], reverse=True)
    kept = scored[:top_k]

    # Reconstruct minimal paths dict
    new_paths: dict[str, dict] = {}
    for _, path, method, op in kept:
        new_paths.setdefault(path, {})[method] = _strip_operation(op)

    # Build trimmed schema — keep info, servers, components/schemas (stripped)
    out: dict[str, Any] = {}
    for k in ("openapi", "info", "servers"):
        if k in schema:
            out[k] = schema[k]
    out["paths"] = new_paths

    # Keep components/schemas but strip extension keys
    components = schema.get("components") or {}
    if components:
        out["components"] = {
            k: {
                name: {ck: cv for ck, cv in comp.items() if not ck.startswith("x-")}
                for name, comp in section.items()
                if isinstance(comp, dict)
            }
            for k, section in components.items()
            if isinstance(section, dict)
        }

    return out


def trim_graphql(sdl: str, task: str) -> str:
    """Strip long docstrings and @deprecated fields from GraphQL SDL."""
    # Remove multi-line docstrings longer than 80 chars
    def _trim_docstring(m: re.Match) -> str:
        content = m.group(1)
        if len(content.strip()) > 80:
            return '"""' + content.strip()[:80] + '…"""'
        return m.group(0)

    result = re.sub(r'"""([\s\S]*?)"""', _trim_docstring, sdl)

    # Remove @deprecated lines (field declarations followed by @deprecated)
    result = re.sub(r"[ \t]+\w[\w!:\[\] ]*@deprecated[^\n]*\n?", "", result)

    return result


def trim_schema(text: str, task: str, top_k: int = 10) -> str:
    """
    Auto-detect format (OpenAPI JSON, OpenAPI YAML, or GraphQL SDL) and trim.

    Returns the trimmed schema as a JSON string for OpenAPI or plain text for GraphQL.
    """
    stripped = text.strip()

    # Try JSON first
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and ("paths" in data or "openapi" in data or "swagger" in data):
                return json.dumps(trim_openapi(data, task, top_k), indent=2)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try YAML (optional dependency)
    if re.match(r"^openapi:|^swagger:", stripped, re.IGNORECASE):
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(stripped)
            if isinstance(data, dict):
                trimmed = trim_openapi(data, task, top_k)
                try:
                    return yaml.dump(trimmed, allow_unicode=True, sort_keys=False)
                except Exception:
                    return json.dumps(trimmed, indent=2)
        except ImportError:
            pass

    # GraphQL SDL heuristic
    if re.search(r"\btype\s+\w+\s*\{|\bquery\b|\bmutation\b|\bschema\b\s*\{", stripped, re.I):
        return trim_graphql(stripped, task)

    # Unknown format — return as-is
    return text
