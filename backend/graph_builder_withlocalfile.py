import json
import networkx as nx
import openai


def extract_triples_from_chunk(text: str, client: openai.OpenAI):
    """Uses LLM to extract entity-relationship triples from text."""
    prompt = f"""Extract main entities and their relationships from the text.
Return ONLY a valid raw JSON array of objects with keys "source", "relation", and "target".

Text:
{text}

JSON Output:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Triple extraction error: {e}")
        return []


def build_networkx_graph(chunks, client: openai.OpenAI, max_chunks: int = 10) -> nx.DiGraph:
    """Builds a NetworkX directed graph from document chunks."""
    print("🕸️ Extracting triples and building NetworkX Knowledge Graph...")
    kg = nx.DiGraph()

    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f" Processing chunk {i + 1}/{min(len(chunks), max_chunks)}...")
        triples = extract_triples_from_chunk(chunk.page_content, client)

        for triple in triples:
            source = triple.get("source")
            target = triple.get("target")
            relation = triple.get("relation", "RELATED_TO")

            if source and target:
                kg.add_node(source)
                kg.add_node(target)
                kg.add_edge(source, target, relation=relation)

    print(f"✅ Knowledge Graph constructed ({kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges).")
    return kg