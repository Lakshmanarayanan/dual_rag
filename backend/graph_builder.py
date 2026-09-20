import json
import networkx as nx

def extract_triples_from_chunk(chunk_text: str, client) -> list:
    system_prompt = """You are a knowledge graph builder. Extract key entity relationships from the provided text.
Return strictly valid JSON in the following format:
{
  "triples": [
    {"subject": "EntityA", "predicate": "RELATION", "object": "EntityB"}
  ]
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text:\n{chunk_text}"}
            ],
            temperature=0.0
        )
        
        raw_content = response.choices[0].message.content.strip()
        data = json.loads(raw_content)
        return data.get("triples", [])

    except Exception as e:
        print(f"Triple extraction error: {e}")
        return []


def build_networkx_graph(chunks, client, max_chunks=5):
    """
    Constructs a NetworkX DiGraph from document chunks.
    """
    # ❌ OLD: G = nx.Graph()
    # ✅ FIX: Use DiGraph to support directional traversal methods like successors()
    G = nx.DiGraph()
    
    print("🕸️ Extracting triples and building NetworkX Knowledge Graph...")
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f" Processing chunk {i+1}/{min(len(chunks), max_chunks)}...")
        triples = extract_triples_from_chunk(chunk.page_content, client)
        
        for t in triples:
            sub = t.get("subject")
            pred = t.get("predicate")
            obj = t.get("object")
            
            if sub and pred and obj:
                G.add_edge(sub, obj, relation=pred)
                
    return G