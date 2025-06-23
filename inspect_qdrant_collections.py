import os
from qdrant_client import QdrantClient

def inspect_qdrant_collections():
    """
    Inspecciona todas las colecciones disponibles en Qdrant.
    """
    qdrant_storage_path = "./qdrant_storage"
    
    print("🔍 INSPECTING QDRANT COLLECTIONS")
    print("="*50)
    print(f"📁 Qdrant storage path: {qdrant_storage_path}")
    
    if not os.path.exists(qdrant_storage_path):
        print("❌ Qdrant storage path does not exist!")
        return
    
    try:
        # Inicializar cliente
        client = QdrantClient(path=qdrant_storage_path)
        
        # Obtener todas las colecciones
        collections_response = client.get_collections()
        
        if not collections_response.collections:
            print("❌ No collections found in Qdrant!")
            return
        
        print(f"\n📊 Found {len(collections_response.collections)} collections:")
        print("-" * 80)
        
        for i, collection in enumerate(collections_response.collections, 1):
            print(f"{i}. Collection Name: '{collection.name}'")
            
            # Obtener información detallada
            try:
                collection_info = client.get_collection(collection.name)
                points_count = collection_info.points_count
                vector_size = collection_info.config.params.vectors.size
                distance = collection_info.config.params.vectors.distance
                
                print(f"   📈 Points: {points_count:,}")
                print(f"   📐 Vector size: {vector_size}")
                print(f"   📏 Distance metric: {distance}")
                
                # Verificar si tiene puntos
                if points_count > 0:
                    print("   ✅ Collection has data")
                else:
                    print("   ⚠️  Collection is empty")
                    
            except Exception as e:
                print(f"   ❌ Error getting collection info: {str(e)}")
            
            print()
        
        # Verificar colecciones esperadas
        expected_collections = [
            "music_chunks_embeddings",
            "music_songs_embeddings", 
            "music_full_songs_embeddings",
            "music_weighted_embeddings",
            "music_representative_embeddings"
        ]
        
        print("🎯 EXPECTED COLLECTIONS CHECK:")
        print("-" * 50)
        
        available_names = [col.name for col in collections_response.collections]
        
        for expected in expected_collections:
            if expected in available_names:
                collection_info = client.get_collection(expected)
                status = f"✅ EXISTS ({collection_info.points_count:,} points)"
            else:
                status = "❌ MISSING"
            
            print(f"{expected:35} -> {status}")
        
    except Exception as e:
        print(f"❌ Error inspecting collections: {str(e)}")

if __name__ == "__main__":
    inspect_qdrant_collections() 