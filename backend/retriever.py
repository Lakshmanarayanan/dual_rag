import re
import networkx as nx
from langchain_community.vectorstores import FAISS

# ==========================================
# GUARDRAILS
# ==========================================

# Known prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"override\s+your\s+rules",
    r"reveal\s+your\s+system\s+instructions",
    r"jailbreak",
]


def validate_user_input(
    user_input: str, max_length: int = 500
) -> tuple[bool, str]:
    """Validates input text against safety policies and injection threats.

    Returns:
        (is_safe: bool, sanitized_text_or_error_message: str)
    """
    clean_input = user_input.strip()

    # 1. Length check
    if not clean_input:
        return False, "Input cannot be empty."

    if len(clean_input) > max_length:
        return (
            False,
            f"Input exceeds maximum allowed character length ({max_length}).",
        )

    # 2. Prompt Injection & Jailbreak check
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, clean_input, re.IGNORECASE):
            return (
                False,
                "Security Alert: Input contains prohibited system modification patterns.",
            )

    return True, clean_input


def validate_llm_output(output_text: str) -> str:
    """Sanitizes model output to prevent accidental leaks or inappropriate content.

    Returns:
        sanitized_output: str
    """
    # 1. Redact accidental API Key leaks (e.g., sk-...)
    sanitized = re.sub(
        r"sk-[a-zA-Z0-9T3BlbkFJ]{20,}", "[REDACTED_API_KEY]", output_text
    )

    # 2. Fallback check for empty response
    if not sanitized.strip():
        return (
            "I'm sorry, but I couldn't generate a valid response for that query."
        )

    return sanitized


# ==========================================
# HYBRID RETRIEVAL
# ==========================================


def retrieve_hybrid_context(
    query: str, vector_db: FAISS, kg: nx.DiGraph, k: int = 3
):
    """Gathers both vector similarity passages and graph node relationships."""
    # 1. FAISS Vector Search
    vector_results = vector_db.similarity_search(query, k=k)
    vector_context = "\n---\n".join([doc.page_content for doc in vector_results])

    # 2. NetworkX Graph Search
    matched_nodes = [
        node for node in kg.nodes() if node.lower() in query.lower()
    ]
    graph_triples = []

    for node in matched_nodes:
        for neighbor in kg.successors(node):
            edge_data = kg.get_edge_data(node, neighbor)
            relation = edge_data.get("relation", "RELATED_TO")
            graph_triples.append(f"({node}) --[{relation}]--> ({neighbor})")

    graph_context = (
        "\n".join(graph_triples)
        if graph_triples
        else "No matching structural entities found in graph."
    )

    return vector_context, graph_context