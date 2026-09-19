import networkx as nx
from langchain_community.vectorstores import FAISS


def retrieve_hybrid_context(query: str, vector_db: FAISS, kg: nx.DiGraph, k: int = 3):
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