from qdrant_client import QdrantClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_audio_embeddings():
    """
    Analiza los embeddings de AUDIO en la base de datos
    """
    print("🎵 ANALIZANDO EMBEDDINGS DE AUDIO EN QDRANT")
    print("=" * 50)
    
    client = QdrantClient(path="./qdrant_storage")
    collection_name = "music_chunks_embeddings"
    
    # Obtener una muestra de embeddings de audio
    print("📥 Obteniendo embeddings de audio...")
    search_results = client.search(
        collection_name=collection_name,
        query_vector=[0.0] * 512,  # Vector dummy
        limit=50,  # Tomar 50 muestras
        with_vectors=True
    )
    
    audio_embeddings = []
    track_ids = []
    filenames = []
    
    for result in search_results:
        if result.vector:
            audio_embeddings.append(result.vector)
            track_ids.append(result.payload.get('track_id', 'N/A'))
            filenames.append(result.payload.get('filename', 'N/A'))
    
    print(f"✅ Obtenidos {len(audio_embeddings)} embeddings de audio")
    
    # Analizar estadísticas de cada embedding
    print(f"\n📊 ESTADÍSTICAS DE EMBEDDINGS DE AUDIO:")
    print("Track_ID | Filename | Mean | Std | Min | Max")
    print("-" * 80)
    
    for i, (emb, track_id, filename) in enumerate(zip(audio_embeddings[:10], track_ids[:10], filenames[:10])):
        emb_array = np.array(emb)
        print(f"{track_id} | {filename[:20]:<20} | {np.mean(emb_array):.6f} | {np.std(emb_array):.6f} | {np.min(emb_array):.3f} | {np.max(emb_array):.3f}")
    
    # Calcular similitudes entre embeddings de audio
    print(f"\n🔍 SIMILITUDES ENTRE EMBEDDINGS DE AUDIO:")
    
    # Tomar primeros 10 para matriz de similitud
    sample_embeddings = np.array(audio_embeddings[:10])
    similarity_matrix = cosine_similarity(sample_embeddings)
    
    print("Matriz de similitudes (diagonal debe ser 1.0, resto debe variar):")
    print("   ", end="")
    for i in range(10):
        print(f"{i:7d}", end="")
    print()
    
    for i in range(10):
        print(f"{i}: ", end="")
        for j in range(10):
            print(f"{similarity_matrix[i][j]:6.3f} ", end="")
        print()
    
    # Estadísticas de la matriz de similitud
    upper_triangle = np.triu(similarity_matrix, k=1)  # Sin diagonal
    non_diagonal = upper_triangle[upper_triangle != 0]
    
    print(f"\n📈 ESTADÍSTICAS DE SIMILITUDES:")
    print(f"   Similitud promedio: {np.mean(non_diagonal):.6f}")
    print(f"   Desviación estándar: {np.std(non_diagonal):.6f}")
    print(f"   Mínima similitud: {np.min(non_diagonal):.6f}")
    print(f"   Máxima similitud: {np.max(non_diagonal):.6f}")
    
    # DIAGNÓSTICO
    if np.mean(non_diagonal) > 0.95:
        print(f"\n🚨 PROBLEMA CRÍTICO: Embeddings de audio son demasiado similares!")
        print(f"   Los embeddings de audio de canciones diferentes son casi idénticos")
        print(f"   Esto explica por qué todas las búsquedas devuelven las mismas canciones")
    elif np.std(non_diagonal) < 0.05:
        print(f"\n⚠️  PROBLEMA: Poca variabilidad en similitudes de audio")
    else:
        print(f"\n✅ Embeddings de audio parecen normales")
    
    # Comparar con algunas estadísticas individuales
    print(f"\n🎼 VARIABILIDAD DE ESTADÍSTICAS INDIVIDUALES:")
    means = [np.mean(emb) for emb in audio_embeddings]
    stds = [np.std(emb) for emb in audio_embeddings]
    
    print(f"   Variación en medias: {np.std(means):.8f}")
    print(f"   Variación en desv. std: {np.std(stds):.8f}")
    
    if np.std(means) < 1e-5:
        print(f"   🚨 CRÍTICO: Todas las canciones tienen medias casi idénticas")
    
    if np.std(stds) < 1e-5:
        print(f"   🚨 CRÍTICO: Todas las canciones tienen desv. std casi idénticas")

if __name__ == "__main__":
    analyze_audio_embeddings() 