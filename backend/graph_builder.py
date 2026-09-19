import json
import networkx as nx

def extract_triples_from_chunk(chunk_text: str, client) -> list:
    """
    Extracts knowledge graph entities and relationships (subject, predicate, object) from a text chunk.
    """
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
            response_format={"type": "json_object"},  # 👈 Enforces raw JSON output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text:\n{chunk_text}"}
            ],
            temperature=0.0
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # Parse JSON output safely
        data = json.loads(raw_content)
        return data.get("triples", [])

    except json.JSONDecodeError:
        # Fallback if markdown fences are included
        cleaned_content = raw_content.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned_content)
            return data.get("triples", [])
        except Exception:
            return []
    except Exception as e:
        print(f"Error extracting triples: {e}")
        return []


def build_networkx_graph(chunks, client, max_chunks=5):
    """
    Constructs a NetworkX MultiDiGraph from document chunks.
    """
    G = nx.Graph()
    
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