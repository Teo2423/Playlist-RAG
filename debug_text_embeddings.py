import torch
from transformers import ClapModel, ClapProcessor
import numpy as np
from qdrant_client import QdrantClient

def debug_text_embeddings():
    """
    Debugging específico para verificar si los embeddings de texto son diferentes
    """
    print("🔍 DEBUGGING TEXT EMBEDDINGS")
    print("=" * 50)
    
    # Cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ClapModel.from_pretrained("laion/larger_clap_music").to(device)
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
    
    # Textos de prueba DIFERENTES
    test_prompts = [
        "upbeat electronic dance music",
        "slow melancholic piano ballad", 
        "heavy metal with distorted guitars",
        "jazz fusion with saxophone",
        "classical orchestral symphony"
    ]
    
    print("\n📊 GENERATING AND COMPARING TEXT EMBEDDINGS")
    embeddings = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{i+1}. Prompt: '{prompt}'")
        
        # Generar embedding
        inputs = processor(text=[prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs)
            query_embedding = text_embed.cpu().numpy().flatten()
        
        # Normalizar
        embeddings.append(query_embedding)
        
        print(f"   Embedding shape: {query_embedding.shape}")
        print(f"   Embedding mean: {np.mean(query_embedding):.6f}")
        print(f"   Embedding std: {np.std(query_embedding):.6f}")
        print(f"   First 5 values: {query_embedding[:5]}")
    
    # Comparar similitudes entre embeddings
    print(f"\n🔍 SIMILARITY MATRIX")
    print("   ", end="")
    for i in range(len(test_prompts)):
        print(f"{i+1:8d}", end="")
    print()
    
    for i in range(len(embeddings)):
        print(f"{i+1}. ", end="")
        for j in range(len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j])
            print(f"{similarity:.4f}  ", end="")
        print()
    
    # Test búsqueda en Qdrant
    print(f"\n🔍 TESTING QDRANT SEARCHES")
    client = QdrantClient(path="./qdrant_storage")
    collection_name = "music_full_songs_embeddings"
    
    for i, (prompt, embedding) in enumerate(zip(test_prompts, embeddings)):
        print(f"\n{i+1}. Searching for: '{prompt}'")
        
        search_results = client.search(
            collection_name=collection_name,
            query_vector=embedding.tolist(),
            limit=5
        )
        
        print(f"   Results: {len(search_results)}")
        track_ids = [result.payload.get('track_id', 'N/A') for result in search_results]
        similarities = [result.score for result in search_results]
        
        print(f"   Track IDs: {track_ids}")
        print(f"   Similarities: {[f'{s:.4f}' for s in similarities]}")

if __name__ == "__main__":
    debug_text_embeddings() 