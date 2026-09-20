import re


# ---------------------------------------------------------------------
# 1. GRAPH CONTEXT RETRIEVAL (SAFE TRAVERSAL)
# ---------------------------------------------------------------------
def retrieve_graph_context(user_query: str, kg) -> str:
    """Safely extracts node-relationship triples matching user query terms."""
    if kg is None or len(kg.nodes()) == 0:
        return "No Knowledge Graph context available."

    relevant_triples = []
    query_terms = set(re.findall(r"\w+", user_query.lower()))

    for node in kg.nodes():
        node_str = str(node).lower()
        # Check if any query term matches the node name
        if any(term in node_str for term in query_terms if len(term) > 2):
            # Safe neighbor retrieval: checks successors (DiGraph) first, then fallback to neighbors
            if hasattr(kg, "successors"):
                neighbors = list(kg.successors(node))
            elif hasattr(kg, "neighbors"):
                neighbors = list(kg.neighbors(node))
            else:
                neighbors = []

            for neighbor in neighbors:
                edge_data = kg.get_edge_data(node, neighbor) or {}
                
                # Handle MultiDiGraph / MultiGraph vs standard DiGraph
                if isinstance(edge_data, dict) and 0 in edge_data:
                    relation = edge_data[0].get("relation", "is related to")
                elif isinstance(edge_data, dict):
                    relation = edge_data.get("relation", "is related to")
                else:
                    relation = "is related to"

                relevant_triples.append(
                    f"({node}) -[{relation}]-> ({neighbor})"
                )

    if not relevant_triples:
        return "No direct entity matches found in Knowledge Graph."

    # Return up to 15 unique relationship triples
    unique_triples = list(dict.fromkeys(relevant_triples))[:15]
    return "\n".join(unique_triples)


# ---------------------------------------------------------------------
# 2. HYBRID RETRIEVAL (VECTOR + GRAPH)
# ---------------------------------------------------------------------
def retrieve_hybrid_context(user_query: str, vector_db, kg, k: int = 4):
    """Combines FAISS vector similarity search with NetworkX graph context."""
    # 1. Retrieve Vector Context from FAISS
    vector_context = "No Vector Context available."
    if vector_db is not None:
        try:
            docs = vector_db.similarity_search(user_query, k=k)
            vector_context = "\n\n".join(
                [f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(docs)]
            )
        except Exception as e:
            vector_context = f"Vector search error: {str(e)}"

    # 2. Retrieve Graph Context safely
    graph_context = retrieve_graph_context(user_query, kg)

    return vector_context, graph_context


# ---------------------------------------------------------------------
# 3. GUARDRAILS
# ---------------------------------------------------------------------
def validate_user_input(prompt: str, max_length: int = 500):
    """Simple input guardrail for length and basic prompt injection checks."""
    if not prompt or not prompt.strip():
        return False, "Query cannot be empty."

    if len(prompt) > max_length:
        return (
            False,
            f"Query exceeds the maximum allowed length of {max_length} characters.",
        )

    forbidden_keywords = [
        "ignore previous instructions",
        "drop database",
        "system prompt",
    ]
    for kw in forbidden_keywords:
        if kw in prompt.lower():
            return False, f"Query contains prohibited input pattern: '{kw}'"

    return True, prompt.strip()


def validate_llm_output(output_text: str) -> str:
    """Output guardrail ensuring no sensitive tokens are exposed."""
    sensitive_patterns = [r"sk-[a-zA-Z0-9]{32,}", r"OPENAI_API_KEY"]

    cleaned_text = output_text
    for pattern in sensitive_patterns:
        cleaned_text = re.sub(
            pattern, "[REDACTED_SENSITIVE_DATA]", cleaned_text
        )

    return cleaned_text