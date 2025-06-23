import os
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

class SongSimilarityMatrixAnalyzer:
    """
    Analiza matrices de similitud entre canciones calculando S = E @ E.T
    y comparando los triángulos superiores usando correlación de Spearman.
    Optimizado para manejar grandes conjuntos de datos con memoria limitada.
    """
    
    def __init__(self, qdrant_storage_path="./qdrant_storage"):
        self.qdrant_storage_path = qdrant_storage_path
        self.client = QdrantClient(path=qdrant_storage_path)
        
        # Configuración de colecciones de canciones (sin chunks)
        self.song_collections = {
            "simple_avg": "music_songs_embeddings",
            "weighted_avg": "music_weighted_embeddings", 
            "representative": "music_representative_embeddings",
            "full_songs": "music_full_songs_embeddings"
        }
        
        print(f"🎵 Song Similarity Matrix Analyzer initialized")
        print(f"📁 Using Qdrant storage: {qdrant_storage_path}")
    
    def verify_song_collections(self):
        """Verifica que las colecciones de canciones existan."""
        print("\n🔍 Verifying song collections availability...")
        
        try:
            collections = self.client.get_collections()
            available_collections = [col.name for col in collections.collections]
            
            verified_collections = {}
            for name, collection_name in self.song_collections.items():
                if collection_name in available_collections:
                    collection_info = self.client.get_collection(collection_name)
                    points_count = collection_info.points_count
                    verified_collections[name] = {
                        'collection_name': collection_name,
                        'points_count': points_count
                    }
                    print(f"   ✅ {name:15} -> {collection_name} ({points_count} points)")
                else:
                    print(f"   ❌ {name:15} -> {collection_name} (NOT FOUND)")
            
            return verified_collections
            
        except Exception as e:
            print(f"❌ Error verifying collections: {str(e)}")
            return {}
    
    def extract_song_embeddings(self, collection_name, max_songs=None):
        """
        Extrae embeddings de canciones de una colección.
        """
        print(f"   📥 Extracting song embeddings from {collection_name}...")
        
        try:
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count
            
            if max_songs is not None:
                limit = min(max_songs, total_points)
                print(f"      📊 Extracting {limit} out of {total_points} songs")
            else:
                limit = total_points
                print(f"      📊 Extracting all {total_points} songs")
            
            batch_size = 1000
            all_embeddings = []
            track_ids = []
            
            for offset in tqdm(range(0, limit, batch_size), 
                              desc=f"      Loading {collection_name}", 
                              leave=False):
                
                current_batch_size = min(batch_size, limit - offset)
                
                search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=[0.0] * 512,
                    limit=current_batch_size,
                    offset=offset,
                    with_vectors=True
                )
                
                batch_embeddings = []
                batch_track_ids = []
                for result in search_results:
                    if result.vector:
                        batch_embeddings.append(result.vector)
                        # Extraer track_id del payload
                        track_id = result.payload.get('track_id', f'unknown_{len(track_ids)}')
                        batch_track_ids.append(track_id)
                
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    track_ids.extend(batch_track_ids)
            
            if not all_embeddings:
                return None, None
            
            embeddings_matrix = np.array(all_embeddings, dtype=np.float32)  # Usar float32 para ahorrar memoria
            
            print(f"      ✅ Extracted {embeddings_matrix.shape[0]} songs of dimension {embeddings_matrix.shape[1]}")
            
            return embeddings_matrix, track_ids
            
        except Exception as e:
            print(f"      ❌ Error extracting from {collection_name}: {str(e)}")
            return None, None
    
    def normalize_to_l1(self, embeddings_matrix):
        """
        Normaliza las filas de la matriz a norma L1.
        """
        print("      🔧 Normalizing embeddings to L1 norm...")
        
        # Calcular norma L1 por fila
        l1_norms = np.sum(np.abs(embeddings_matrix), axis=1, keepdims=True)
        
        # Evitar división por cero
        l1_norms = np.where(l1_norms == 0, 1, l1_norms)
        
        # Normalizar
        normalized_embeddings = embeddings_matrix / l1_norms
        
        print(f"         ✅ Normalized {normalized_embeddings.shape[0]} songs")
        return normalized_embeddings
    
    def estimate_memory_requirements(self, n_songs):
        """
        Estima los requerimientos de memoria para la matriz de similitud.
        """
        # Matriz completa en float32: n_songs x n_songs x 4 bytes
        full_matrix_gb = (n_songs * n_songs * 4) / (1024**3)
        
        # Solo triángulo superior: aproximadamente la mitad
        triangle_gb = full_matrix_gb / 2
        
        return full_matrix_gb, triangle_gb
    
    def compute_similarity_matrix_chunked(self, embeddings, chunk_size=1000):
        """
        Calcula la matriz de similitud por chunks para manejar memoria limitada.
        Solo calcula y almacena el triángulo superior.
        """
        n_songs = embeddings.shape[0]
        
        print(f"      🧮 Computing similarity matrix ({n_songs}x{n_songs}) in chunks...")
        
        # Estimar memoria
        full_gb, triangle_gb = self.estimate_memory_requirements(n_songs)
        print(f"         💾 Estimated memory: {full_gb:.2f} GB (full), {triangle_gb:.2f} GB (triangle)")
        
        # Decidir estrategia basada en memoria estimada
        available_ram_gb = 8  # Asumir 8GB disponibles, ajustar según sistema
        
        if triangle_gb < available_ram_gb * 0.5:  # Usar solo 50% de RAM disponible
            print(f"         ⚡ Small matrix - computing directly")
            return self._compute_similarity_direct(embeddings)
        else:
            print(f"         🔄 Large matrix - using chunked computation")
            return self._compute_similarity_chunked_impl(embeddings, chunk_size)
    
    def _compute_similarity_direct(self, embeddings):
        """
        Calcula la matriz de similitud directamente para matrices pequeñas.
        """
        # Calcular matriz completa
        similarity_matrix = embeddings @ embeddings.T
        
        # Extraer solo triángulo superior (sin diagonal)
        n_songs = similarity_matrix.shape[0]
        upper_triangle = []
        
        for i in range(n_songs):
            for j in range(i+1, n_songs):
                upper_triangle.append(similarity_matrix[i, j])
        
        return np.array(upper_triangle, dtype=np.float32)
    
    def _compute_similarity_chunked_impl(self, embeddings, chunk_size):
        """
        Implementación chunked para matrices grandes.
        """
        n_songs = embeddings.shape[0]
        upper_triangle = []
        
        # Procesar por chunks del triángulo superior
        for i in tqdm(range(0, n_songs, chunk_size), desc="         Computing chunks", leave=False):
            i_end = min(i + chunk_size, n_songs)
            
            # Chunk de filas
            chunk_i = embeddings[i:i_end]
            
            for j in range(i, n_songs, chunk_size):
                j_end = min(j + chunk_size, n_songs)
                
                # Chunk de columnas
                chunk_j = embeddings[j:j_end]
                
                # Calcular similitud del bloque
                block_sim = chunk_i @ chunk_j.T
                
                # Extraer solo la parte del triángulo superior
                for ii in range(block_sim.shape[0]):
                    for jj in range(block_sim.shape[1]):
                        global_i = i + ii
                        global_j = j + jj
                        
                        # Solo triángulo superior sin diagonal
                        if global_i < global_j:
                            upper_triangle.append(block_sim[ii, jj])
                
                # Limpiar memoria
                del chunk_j, block_sim
                gc.collect()
            
            # Limpiar memoria
            del chunk_i
            gc.collect()
        
        return np.array(upper_triangle, dtype=np.float32)
    
    def analyze_song_collection(self, collection_name, name, max_songs=None):
        """
        Analiza una colección de canciones completa.
        """
        print(f"\n🎵 Processing song collection: {name}")
        
        # Extraer embeddings
        embeddings, track_ids = self.extract_song_embeddings(collection_name, max_songs)
        
        if embeddings is None:
            print(f"      ❌ Failed to extract embeddings from {name}")
            return None
        
        # Normalizar a L1
        normalized_embeddings = self.normalize_to_l1(embeddings)
        
        # Calcular matriz de similitud (solo triángulo superior)
        upper_triangle = self.compute_similarity_matrix_chunked(normalized_embeddings)
        
        print(f"      ✅ Collection {name} processed: {len(upper_triangle)} similarity values")
        
        # Limpiar memoria
        del embeddings, normalized_embeddings
        gc.collect()
        
        return {
            'upper_triangle': upper_triangle,
            'n_songs': len(track_ids),
            'track_ids': track_ids[:100],  # Solo guardar los primeros 100 para referencia
            'collection_name': collection_name
        }
    
    def compute_spearman_correlations(self, collection_results):
        """
        Calcula correlaciones de Spearman entre los triángulos superiores.
        """
        print(f"\n📊 Computing Spearman correlations between similarity matrices...")
        
        collection_names = list(collection_results.keys())
        correlations = {}
        
        # Verificar que todas las colecciones tengan el mismo número de canciones
        n_songs_per_collection = {name: result['n_songs'] for name, result in collection_results.items()}
        unique_n_songs = set(n_songs_per_collection.values())
        
        if len(unique_n_songs) > 1:
            print(f"⚠️  Warning: Collections have different numbers of songs: {n_songs_per_collection}")
            print("   Using minimum number of songs for comparison...")
            
            # Encontrar el mínimo número de canciones
            min_n_songs = min(n_songs_per_collection.values())
            expected_triangle_size = (min_n_songs * (min_n_songs - 1)) // 2
            
            # Truncar todos los triángulos al mismo tamaño
            for name, result in collection_results.items():
                if len(result['upper_triangle']) > expected_triangle_size:
                    result['upper_triangle'] = result['upper_triangle'][:expected_triangle_size]
                    print(f"   Truncated {name} to {expected_triangle_size} values")
        
        # Calcular correlaciones por pares
        for i, name1 in enumerate(collection_names):
            for j, name2 in enumerate(collection_names):
                if i < j:
                    triangle1 = collection_results[name1]['upper_triangle']
                    triangle2 = collection_results[name2]['upper_triangle']
                    
                    # Asegurar misma longitud
                    min_len = min(len(triangle1), len(triangle2))
                    triangle1_truncated = triangle1[:min_len]
                    triangle2_truncated = triangle2[:min_len]
                    
                    # Correlación de Spearman
                    spearman_corr, spearman_p = spearmanr(triangle1_truncated, triangle2_truncated)
                    
                    pair_key = f"{name1}_vs_{name2}"
                    correlations[pair_key] = {
                        'spearman_correlation': float(spearman_corr),
                        'spearman_p_value': float(spearman_p),
                        'comparison_size': min_len,
                        'significance': 'significant' if spearman_p < 0.05 else 'not_significant'
                    }
                    
                    print(f"   {name1:15} vs {name2:15}: ρ = {spearman_corr:7.4f} (p = {spearman_p:.6f}) [{min_len:,} values]")
        
        return correlations
    
    def run_complete_similarity_analysis(self, max_songs_per_collection=None):
        """
        Ejecuta el análisis completo de matrices de similitud.
        """
        print("🎵 SONG SIMILARITY MATRIX ANALYSIS")
        print("🔬 Computing S = E @ E.T and comparing upper triangles")
        print("="*60)
        
        # Verificar colecciones
        verified_collections = self.verify_song_collections()
        if not verified_collections:
            print("❌ No song collections available for analysis")
            return None
        
        # Analizar cada colección
        collection_results = {}
        
        for name, info in verified_collections.items():
            result = self.analyze_song_collection(
                info['collection_name'], 
                name, 
                max_songs_per_collection
            )
            
            if result is not None:
                collection_results[name] = result
                print(f"      ✅ Collection {name} analysis completed")
            else:
                print(f"      ❌ Failed to analyze {name}")
        
        if len(collection_results) < 2:
            print("❌ Need at least 2 collections for correlation analysis")
            return None
        
        # Calcular correlaciones de Spearman
        correlations = self.compute_spearman_correlations(collection_results)
        
        # Resultado final
        results = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'max_songs_per_collection': max_songs_per_collection,
                'collections_analyzed': len(collection_results),
                'analysis_type': 'song_similarity_matrices'
            },
            'collection_results': {name: {
                'n_songs': result['n_songs'],
                'collection_name': result['collection_name'],
                'triangle_size': len(result['upper_triangle']),
                'triangle_stats': {
                    'mean': float(np.mean(result['upper_triangle'])),
                    'std': float(np.std(result['upper_triangle'])),
                    'min': float(np.min(result['upper_triangle'])),
                    'max': float(np.max(result['upper_triangle']))
                }
            } for name, result in collection_results.items()},
            'spearman_correlations': correlations
        }
        
        return results
    
    def create_similarity_visualizations(self, results, output_folder):
        """Crea visualizaciones de las correlaciones de similitud."""
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            correlations = results['spearman_correlations']
            
            if correlations:
                # Extraer nombres de colecciones
                collection_names = set()
                for pair in correlations.keys():
                    names = pair.split('_vs_')
                    collection_names.update(names)
                
                collection_names = sorted(list(collection_names))
                n_cols = len(collection_names)
                
                # Crear matriz de correlaciones
                correlation_matrix = np.eye(n_cols)
                
                for i, name1 in enumerate(collection_names):
                    for j, name2 in enumerate(collection_names):
                        if i != j:
                            pair_key = f"{name1}_vs_{name2}" if f"{name1}_vs_{name2}" in correlations else f"{name2}_vs_{name1}"
                            if pair_key in correlations:
                                corr = correlations[pair_key]['spearman_correlation']
                                correlation_matrix[i, j] = corr
                
                # Crear heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, 
                           xticklabels=collection_names, 
                           yticklabels=collection_names,
                           annot=True, cmap='RdBu_r', center=0, 
                           square=True, fmt='.4f',
                           cbar_kws={'label': 'Spearman Correlation'})
                plt.title('Spearman Correlations Between Song Similarity Matrices\n(Upper Triangle Vectors)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, 'similarity_matrix_correlations.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Crear gráfico de distribución de correlaciones
                plt.figure(figsize=(10, 6))
                correlation_values = [data['spearman_correlation'] for data in correlations.values()]
                plt.hist(correlation_values, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Spearman Correlation')
                plt.ylabel('Frequency')
                plt.title('Distribution of Spearman Correlations Between Similarity Matrices')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, 'correlation_distribution.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"   🎨 Similarity visualizations saved to {output_folder}")
            
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {str(e)}")

def main():
    """Función principal del analizador de matrices de similitud."""
    print("🎵 SONG SIMILARITY MATRIX ANALYZER")
    print("🔬 Analyzes similarity matrices S = E @ E.T between song collections")
    print("📊 Compares upper triangles using Spearman correlation")
    print("💾 Optimized for large datasets with limited RAM")
    print()
    
    analyzer = SongSimilarityMatrixAnalyzer()
    
    # Configuración
    MAX_SONGS = None  # None = usar todas las canciones, o especificar un número
    
    print(f"⚙️ Configuration:")
    print(f"   Max songs per collection: {'All available' if MAX_SONGS is None else MAX_SONGS}")
    print(f"   Memory optimization: Enabled")
    print(f"   Chunked computation: Auto-enabled for large matrices")
    
    # Ejecutar análisis
    results = analyzer.run_complete_similarity_analysis(max_songs_per_collection=MAX_SONGS)
    
    if results:
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"./similarity_matrix_analysis_{timestamp}"
        
        os.makedirs(output_folder, exist_ok=True)
        
        results_file = os.path.join(output_folder, "similarity_matrix_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Crear visualizaciones
        analyzer.create_similarity_visualizations(results, output_folder)
        
        # Crear reporte
        report_file = os.path.join(output_folder, "similarity_matrix_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SONG SIMILARITY MATRIX ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: {results['analysis_info']['timestamp']}\n")
            f.write(f"Collections Analyzed: {results['analysis_info']['collections_analyzed']}\n")
            f.write(f"Max Songs per Collection: {results['analysis_info']['max_songs_per_collection'] or 'All'}\n\n")
            
            f.write("COLLECTION SUMMARY:\n")
            f.write("-"*30 + "\n")
            for name, data in results['collection_results'].items():
                f.write(f"{name}:\n")
                f.write(f"  Songs: {data['n_songs']:,}\n")
                f.write(f"  Triangle size: {data['triangle_size']:,}\n")
                f.write(f"  Similarity stats:\n")
                f.write(f"    Mean: {data['triangle_stats']['mean']:.6f}\n")
                f.write(f"    Std:  {data['triangle_stats']['std']:.6f}\n")
                f.write(f"    Range: [{data['triangle_stats']['min']:.6f}, {data['triangle_stats']['max']:.6f}]\n\n")
            
            f.write("SPEARMAN CORRELATIONS BETWEEN SIMILARITY MATRICES:\n")
            f.write("-"*50 + "\n")
            for pair, data in results['spearman_correlations'].items():
                significance = "***" if data['spearman_p_value'] < 0.001 else "**" if data['spearman_p_value'] < 0.01 else "*" if data['spearman_p_value'] < 0.05 else ""
                f.write(f"{pair:30}: ρ = {data['spearman_correlation']:7.4f} (p = {data['spearman_p_value']:.6f}) {significance}\n")
                f.write(f"{'':32}   Comparison size: {data['comparison_size']:,} values\n\n")
        
        print(f"\n🎉 Similarity matrix analysis completed successfully!")
        print(f"📁 Results saved to: {output_folder}")
        print(f"📄 Full results: {results_file}")
        print(f"📝 Summary report: {report_file}")
        
        # Mostrar resumen en consola
        print(f"\n📊 SPEARMAN CORRELATIONS SUMMARY:")
        for pair, data in results['spearman_correlations'].items():
            correlation = data['spearman_correlation']
            p_value = data['spearman_p_value']
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"   {pair}: ρ = {correlation:.4f} (p = {p_value:.6f}) {significance}")
            
    else:
        print("❌ Similarity matrix analysis failed")

if __name__ == "__main__":
    main() 