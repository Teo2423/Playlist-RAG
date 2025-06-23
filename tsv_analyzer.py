"""
Análisis exploratorio del archivo TSV de canciones musicales.

Este módulo analiza la estructura, completitud y patrones en los datos
para identificar oportunidades de nuevas métricas para evaluación de embeddings RAG.

Autor: Sistema de evaluación de embeddings RAG Musical
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

class TSVAnalyzer:
    """
    Analizador de datos del archivo TSV de canciones.
    """
    
    def __init__(self, tsv_file_path: str = "valid.tsv"):
        """
        Inicializa el analizador cargando el archivo TSV.
        
        Args:
            tsv_file_path: Ruta al archivo TSV
        """
        print(f"🔄 Loading TSV file: {tsv_file_path}")
        self.df = pd.read_csv(tsv_file_path, sep='\t')
        print(f"✅ Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        
        # Preprocesar listas
        self._preprocess_list_columns()
        
    def _preprocess_list_columns(self):
        """Convierte strings de listas a listas reales."""
        print("🔄 Preprocessing list columns...")
        
        def safe_eval(x):
            if pd.isna(x) or x == '[]':
                return []
            try:
                return ast.literal_eval(x)
            except:
                return []
        
        list_columns = ['genres', 'instruments', 'moods']
        for col in list_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(safe_eval)
        
        print("✅ List columns preprocessed")
    
    def analyze_data_completeness(self) -> Dict[str, Any]:
        """
        Analiza la completitud de datos en cada columna.
        
        Returns:
            Dict con estadísticas de completitud
        """
        print("\n📊 ANALYZING DATA COMPLETENESS")
        print("="*60)
        
        completeness = {}
        total_rows = len(self.df)
        
        for col in self.df.columns:
            if col in ['genres', 'instruments', 'moods']:
                # Para listas, contar cuántas no están vacías
                non_empty = sum(1 for x in self.df[col] if x)
                empty = total_rows - non_empty
                completeness[col] = {
                    'total': total_rows,
                    'non_empty': non_empty,
                    'empty': empty,
                    'completeness_pct': (non_empty / total_rows) * 100,
                    'type': 'list'
                }
            else:
                # Para otros tipos, usar pandas isna
                non_null = self.df[col].notna().sum()
                null = total_rows - non_null
                completeness[col] = {
                    'total': total_rows,
                    'non_null': non_null,
                    'null': null,
                    'completeness_pct': (non_null / total_rows) * 100,
                    'type': 'scalar'
                }
        
        # Imprimir resultados
        print(f"{'Column':<15} {'Complete':<10} {'Missing':<10} {'Completeness':<12} {'Type'}")
        print("-" * 60)
        
        for col, stats in completeness.items():
            if stats['type'] == 'list':
                complete = stats['non_empty']
                missing = stats['empty']
            else:
                complete = stats['non_null']
                missing = stats['null']
            
            pct = stats['completeness_pct']
            print(f"{col:<15} {complete:<10} {missing:<10} {pct:>8.1f}%    {stats['type']}")
        
        return completeness
    
    def analyze_list_columns_distribution(self) -> Dict[str, Any]:
        """
        Analiza la distribución de elementos en columnas de lista.
        
        Returns:
            Dict con estadísticas de distribución
        """
        print("\n🎭 ANALYZING LIST COLUMNS DISTRIBUTION")
        print("="*60)
        
        list_stats = {}
        
        for col in ['genres', 'instruments', 'moods']:
            if col not in self.df.columns:
                continue
                
            print(f"\n📂 {col.upper()} Analysis:")
            
            # Estadísticas básicas
            all_items = []
            lengths = []
            
            for item_list in self.df[col]:
                if item_list:
                    all_items.extend(item_list)
                    lengths.append(len(item_list))
                else:
                    lengths.append(0)
            
            # Contar elementos únicos
            item_counts = Counter(all_items)
            
            stats = {
                'total_entries': len(self.df[col]),
                'non_empty_entries': len([x for x in self.df[col] if x]),
                'empty_entries': len([x for x in self.df[col] if not x]),
                'unique_items': len(item_counts),
                'total_item_occurrences': len(all_items),
                'avg_items_per_entry': np.mean(lengths) if lengths else 0,
                'max_items_per_entry': max(lengths) if lengths else 0,
                'min_items_per_entry': min(lengths) if lengths else 0,
                'most_common_items': item_counts.most_common(10),
                'item_distribution': dict(item_counts)
            }
            
            list_stats[col] = stats
            
            # Imprimir estadísticas
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   Non-empty: {stats['non_empty_entries']} ({stats['non_empty_entries']/stats['total_entries']*100:.1f}%)")
            print(f"   Empty: {stats['empty_entries']} ({stats['empty_entries']/stats['total_entries']*100:.1f}%)")
            print(f"   Unique items: {stats['unique_items']}")
            print(f"   Total occurrences: {stats['total_item_occurrences']}")
            print(f"   Avg items per entry: {stats['avg_items_per_entry']:.2f}")
            print(f"   Range: {stats['min_items_per_entry']} - {stats['max_items_per_entry']} items")
            
            print(f"   Top 10 most common:")
            for item, count in stats['most_common_items']:
                print(f"      {item}: {count} times ({count/stats['total_item_occurrences']*100:.1f}%)")
        
        return list_stats
    
    def analyze_numerical_columns(self) -> Dict[str, Any]:
        """
        Analiza columnas numéricas.
        
        Returns:
            Dict con estadísticas numéricas
        """
        print("\n🔢 ANALYZING NUMERICAL COLUMNS")
        print("="*60)
        
        numerical_cols = ['durationInSec', 'chunk_nr']
        numerical_stats = {}
        
        for col in numerical_cols:
            if col not in self.df.columns:
                continue
            
            print(f"\n📊 {col.upper()} Statistics:")
            
            stats = {
                'count': self.df[col].count(),
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'q25': self.df[col].quantile(0.25),
                'q75': self.df[col].quantile(0.75),
                'unique_values': self.df[col].nunique()
            }
            
            numerical_stats[col] = stats
            
            print(f"   Count: {stats['count']}")
            print(f"   Mean: {stats['mean']:.2f}")
            print(f"   Std: {stats['std']:.2f}")
            print(f"   Min: {stats['min']:.2f}")
            print(f"   Max: {stats['max']:.2f}")
            print(f"   Median: {stats['median']:.2f}")
            print(f"   Q25-Q75: {stats['q25']:.2f} - {stats['q75']:.2f}")
            print(f"   Unique values: {stats['unique_values']}")
        
        return numerical_stats
    
    def analyze_id_columns(self) -> Dict[str, Any]:
        """
        Analiza columnas de IDs para entender cardinalidad.
        
        Returns:
            Dict con estadísticas de IDs
        """
        print("\n🆔 ANALYZING ID COLUMNS")
        print("="*60)
        
        id_cols = ['id', 'artist_id', 'album_id']
        id_stats = {}
        
        for col in id_cols:
            if col not in self.df.columns:
                continue
            
            print(f"\n🏷️  {col.upper()} Analysis:")
            
            stats = {
                'total_entries': len(self.df[col]),
                'unique_values': self.df[col].nunique(),
                'duplicates': len(self.df[col]) - self.df[col].nunique(),
                'most_common': self.df[col].value_counts().head(10)
            }
            
            id_stats[col] = stats
            
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   Unique values: {stats['unique_values']}")
            print(f"   Duplicates: {stats['duplicates']}")
            print(f"   Uniqueness: {stats['unique_values']/stats['total_entries']*100:.1f}%")
            
            if stats['duplicates'] > 0:
                print(f"   Most frequent values:")
                for value, count in stats['most_common'].head(5).items():
                    print(f"      {value}: {count} times")
        
        return id_stats
    
    def analyze_chunk_distribution(self) -> Dict[str, Any]:
        """
        Analiza la distribución de chunks por canción.
        
        Returns:
            Dict con estadísticas de chunks
        """
        print("\n🧩 ANALYZING CHUNK DISTRIBUTION")
        print("="*60)
        
        if 'chunk_nr' not in self.df.columns:
            print("❌ No chunk_nr column found")
            return {}
        
        # Agrupar por ID de canción para ver cuántos chunks tiene cada una
        chunk_stats = {}
        
        # Contar chunks por canción
        chunks_per_song = self.df.groupby('id')['chunk_nr'].nunique()
        
        stats = {
            'total_songs': len(chunks_per_song),
            'total_chunks': len(self.df),
            'avg_chunks_per_song': chunks_per_song.mean(),
            'max_chunks_per_song': chunks_per_song.max(),
            'min_chunks_per_song': chunks_per_song.min(),
            'chunks_distribution': chunks_per_song.value_counts().sort_index()
        }
        
        chunk_stats = stats
        
        print(f"   Total unique songs: {stats['total_songs']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Average chunks per song: {stats['avg_chunks_per_song']:.2f}")
        print(f"   Chunks per song range: {stats['min_chunks_per_song']} - {stats['max_chunks_per_song']}")
        
        print(f"\n   Chunks per song distribution:")
        for chunks, count in stats['chunks_distribution'].head(10).items():
            print(f"      {chunks} chunks: {count} songs ({count/stats['total_songs']*100:.1f}%)")
        
        return chunk_stats
    
    def suggest_new_metrics(self, completeness: Dict, list_stats: Dict, 
                          numerical_stats: Dict, id_stats: Dict, 
                          chunk_stats: Dict) -> List[Dict[str, str]]:
        """
        Sugiere nuevas métricas basadas en el análisis de datos.
        
        Args:
            completeness: Estadísticas de completitud
            list_stats: Estadísticas de columnas de lista
            numerical_stats: Estadísticas numéricas
            id_stats: Estadísticas de IDs
            chunk_stats: Estadísticas de chunks
            
        Returns:
            Lista de sugerencias de métricas
        """
        print("\n💡 SUGGESTED NEW METRICS")
        print("="*60)
        
        suggestions = []
        
        # Basado en completitud de datos
        if completeness.get('genres', {}).get('completeness_pct', 0) < 50:
            suggestions.append({
                'name': 'Data Completeness Score',
                'description': 'Penalizar embeddings que seleccionan canciones con metadatos incompletos',
                'rationale': f"Solo {completeness.get('genres', {}).get('completeness_pct', 0):.1f}% de canciones tienen géneros",
                'implementation': 'Calcular porcentaje de canciones con datos completos en la playlist'
            })
        
        # Basado en distribución de chunks
        if chunk_stats:
            suggestions.append({
                'name': 'Chunk Coherence Metric',
                'description': 'Medir si los chunks seleccionados provienen de canciones coherentes',
                'rationale': f"Hay {chunk_stats['total_chunks']} chunks de {chunk_stats['total_songs']} canciones",
                'implementation': 'Penalizar playlists que mezclan chunks de la misma canción de manera incoherente'
            })
            
            suggestions.append({
                'name': 'Chunk Diversity Metric',
                'description': 'Medir diversidad de chunks vs canciones completas',
                'rationale': 'Chunks pueden dar más granularidad pero menos cohesión',
                'implementation': 'Ratio de canciones únicas vs total de chunks en playlist'
            })
        
        # Basado en duración
        if 'durationInSec' in numerical_stats:
            duration_stats = numerical_stats['durationInSec']
            suggestions.append({
                'name': 'Duration Consistency Metric',
                'description': 'Medir consistencia de duración en la playlist',
                'rationale': f"Duraciones van de {duration_stats['min']:.0f}s a {duration_stats['max']:.0f}s",
                'implementation': 'Penalizar playlists con duraciones muy dispares usando coef. de variación'
            })
            
            suggestions.append({
                'name': 'Playlist Flow Metric',
                'description': 'Medir transiciones suaves entre canciones',
                'rationale': 'Playlists buenas tienen transiciones naturales de duración',
                'implementation': 'Calcular varianza de diferencias consecutivas de duración'
            })
        
        # Basado en distribución de artistas/álbumes
        if 'artist_id' in id_stats:
            artist_stats = id_stats['artist_id']
            suggestions.append({
                'name': 'Artist Clustering Metric',
                'description': 'Detectar si hay clusters de canciones del mismo artista',
                'rationale': f"{artist_stats['duplicates']} canciones duplicadas de artistas",
                'implementation': 'Medir distribución espacial de canciones por artista en la playlist'
            })
        
        # Basado en datos semánticos disponibles
        for col in ['genres', 'instruments', 'moods']:
            if col in list_stats:
                stats = list_stats[col]
                if stats['non_empty_entries'] > 0:
                    suggestions.append({
                        'name': f'{col.capitalize()} Rarity Score',
                        'description': f'Valorar canciones con {col} poco comunes',
                        'rationale': f"Hay {stats['unique_items']} {col} únicos, algunos muy raros",
                        'implementation': f'Usar frecuencia inversa de {col} para dar puntos a diversidad'
                    })
        
        # Métricas de calidad de embeddings específicas
        suggestions.append({
            'name': 'Metadata Richness Score',
            'description': 'Preferir canciones con metadatos ricos y completos',
            'rationale': 'Canciones con más metadatos dan mejor contexto semántico',
            'implementation': 'Sumar tags únicos por canción y promediar en playlist'
        })
        
        suggestions.append({
            'name': 'Semantic Density Metric',
            'description': 'Medir densidad semántica de tags en la playlist',
            'rationale': 'Playlists con alta densidad semántica son más interesantes',
            'implementation': 'Ratio de tags únicos vs total de tags en playlist'
        })
        
        suggestions.append({
            'name': 'Cross-Taxonomic Coherence',
            'description': 'Medir coherencia entre géneros, moods e instrumentos',
            'rationale': 'Combinaciones coherentes indican mejor comprensión semántica',
            'implementation': 'Medir correlaciones típicas entre taxonomías en el dataset'
        })
        
        # Imprimir sugerencias
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i:2d}. 🎯 {suggestion['name']}")
            print(f"    📝 Description: {suggestion['description']}")
            print(f"    🔍 Rationale: {suggestion['rationale']}")
            print(f"    ⚙️  Implementation: {suggestion['implementation']}")
        
        return suggestions
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Ejecuta análisis completo del TSV.
        
        Returns:
            Dict con todos los resultados del análisis
        """
        print("🚀 STARTING COMPLETE TSV ANALYSIS")
        print("="*60)
        
        results = {}
        
        # 1. Análisis de completitud
        results['completeness'] = self.analyze_data_completeness()
        
        # 2. Análisis de distribución de listas
        results['list_distribution'] = self.analyze_list_columns_distribution()
        
        # 3. Análisis de columnas numéricas
        results['numerical_analysis'] = self.analyze_numerical_columns()
        
        # 4. Análisis de IDs
        results['id_analysis'] = self.analyze_id_columns()
        
        # 5. Análisis de chunks
        results['chunk_analysis'] = self.analyze_chunk_distribution()
        
        # 6. Sugerencias de métricas
        results['metric_suggestions'] = self.suggest_new_metrics(
            results['completeness'],
            results['list_distribution'],
            results['numerical_analysis'],
            results['id_analysis'],
            results['chunk_analysis']
        )
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"📊 Found {len(results['metric_suggestions'])} metric suggestions")
        
        return results
    
    def save_analysis_report(self, results: Dict[str, Any], filename: str = "tsv_analysis_report.txt"):
        """
        Guarda el reporte de análisis en un archivo.
        
        Args:
            results: Resultados del análisis
            filename: Nombre del archivo de salida
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("TSV ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Resumen ejecutivo
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            f.write(f"Total rows: {len(self.df)}\n")
            f.write(f"Total columns: {len(self.df.columns)}\n")
            
            # Completitud
            f.write(f"\nData completeness:\n")
            for col, stats in results['completeness'].items():
                f.write(f"  {col}: {stats['completeness_pct']:.1f}%\n")
            
            # Sugerencias
            f.write(f"\nMETRIC SUGGESTIONS ({len(results['metric_suggestions'])}):\n")
            f.write("-"*40 + "\n")
            for i, suggestion in enumerate(results['metric_suggestions'], 1):
                f.write(f"{i}. {suggestion['name']}\n")
                f.write(f"   {suggestion['description']}\n")
                f.write(f"   Rationale: {suggestion['rationale']}\n\n")
        
        print(f"📄 Analysis report saved to: {filename}")


def main():
    """
    Función principal para ejecutar el análisis.
    """
    # Crear analizador
    analyzer = TSVAnalyzer("valid.tsv")
    
    # Ejecutar análisis completo
    results = analyzer.run_complete_analysis()
    
    # Guardar reporte
    analyzer.save_analysis_report(results)
    
    return results


if __name__ == "__main__":
    analysis_results = main() 