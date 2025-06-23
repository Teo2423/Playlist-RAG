"""
Métricas semánticas para evaluación de embeddings en RAG musical.

Este módulo implementa métricas que evalúan la calidad semántica de las playlists
generadas, complementando las métricas técnicas de audio existentes.

Métricas implementadas:
1. Cobertura del prompt: Mide qué tan bien las canciones cubren las palabras clave del prompt
2. Diversidad de artistas/álbumes: Usa índice de Gini-Simpson para medir diversidad
3. Entropía de géneros/moods/instrumentos: Usa entropía de Shannon para medir variedad semántica

Autor: Sistema de evaluación de embeddings RAG Musical

NO SE USAN TODAS LAS METRICAS EN EL PROYECTO, SOLO LAS NECESARIAS.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from math import log2
import ast
from typing import List, Dict, Any, Tuple, Set
import warnings
warnings.filterwarnings("ignore")

class SemanticMetricsCalculator:
    """
    Calculadora de métricas semánticas para evaluación de playlists musicales.
    """
    
    def __init__(self, tsv_file_path: str = "valid.tsv"):
        """
        Inicializa el calculador cargando los datos de canciones.
        
        Args:
            tsv_file_path: Ruta al archivo TSV con información de canciones
        """
        print(f"🔄 Loading song data from {tsv_file_path}...")
        self.songs_df = pd.read_csv(tsv_file_path, sep='\t')
        print(f"✅ Loaded {len(self.songs_df)} songs")
        
        # Preprocesar las listas de strings
        self._preprocess_lists()
        
        # Palabras clave para extracción del prompt
        self.genre_keywords = self._extract_all_genres()
        self.mood_keywords = self._extract_all_moods()  
        self.instrument_keywords = self._extract_all_instruments()
        
        print(f"📊 Found {len(self.genre_keywords)} genres, {len(self.mood_keywords)} moods, {len(self.instrument_keywords)} instruments")
    
    def _preprocess_lists(self):
        """Convierte strings de listas a listas reales."""
        print("🔄 Preprocessing list columns...")
        
        def safe_eval(x):
            """Safely evaluate string representation of list."""
            if pd.isna(x) or x == '[]':
                return []
            try:
                return ast.literal_eval(x)
            except:
                return []
        
        self.songs_df['genres'] = self.songs_df['genres'].apply(safe_eval)
        self.songs_df['instruments'] = self.songs_df['instruments'].apply(safe_eval)
        self.songs_df['moods'] = self.songs_df['moods'].apply(safe_eval)
        
        print("✅ Lists preprocessed")
    
    def _extract_all_genres(self) -> Set[str]:
        """Extrae todos los géneros únicos del dataset."""
        genres = set()
        for genre_list in self.songs_df['genres']:
            genres.update(genre_list)
        return genres
    
    def _extract_all_moods(self) -> Set[str]:
        """Extrae todos los moods únicos del dataset.""" 
        moods = set()
        for mood_list in self.songs_df['moods']:
            moods.update(mood_list)
        return moods
    
    def _extract_all_instruments(self) -> Set[str]:
        """Extrae todos los instrumentos únicos del dataset."""
        instruments = set()
        for instrument_list in self.songs_df['instruments']:
            instruments.update(instrument_list)
        return instruments
    
    def extract_keywords_from_prompt(self, prompt: str) -> Dict[str, Set[str]]:
        """
        Extrae palabras clave del prompt categorizándolas en géneros, moods e instrumentos.
        
        Args:
            prompt: Texto del prompt de la playlist
            
        Returns:
            Dict con sets de keywords por categoría
        """
        prompt_lower = prompt.lower()
        
        # Encontrar matches de cada categoría
        found_genres = {g for g in self.genre_keywords if g.lower() in prompt_lower}
        found_moods = {m for m in self.mood_keywords if m.lower() in prompt_lower}  
        found_instruments = {i for i in self.instrument_keywords if i.lower() in prompt_lower}
        
        # También buscar variantes comunes
        genre_variants = {
            'rock': ['rock', 'rocks', 'rocker'],
            'pop': ['pop', 'pops', 'popular'],
            'jazz': ['jazz', 'jazzy'],
            'electronic': ['electronic', 'electro', 'edm', 'techno', 'house'],
            'classical': ['classical', 'classic'],
            'folk': ['folk', 'folklore'],
            'ambient': ['ambient', 'atmospheric'],
            'metal': ['metal', 'metallic', 'heavy'],
        }
        
        mood_variants = {
            'happy': ['happy', 'happiness', 'joyful', 'upbeat', 'cheerful', 'uplifting'],
            'sad': ['sad', 'sadness', 'melancholic', 'melancholy', 'depressing'],
            'energetic': ['energetic', 'energy', 'powerful', 'dynamic'],
            'calm': ['calm', 'calming', 'peaceful', 'relaxing', 'chill'],
            'dark': ['dark', 'darkness', 'mysterious', 'ominous'],
            'romantic': ['romantic', 'romance', 'love', 'loving'],
        }
        
        instrument_variants = {
            'guitar': ['guitar', 'guitars', 'guitarist', 'acoustic', 'electric'],
            'piano': ['piano', 'pianos', 'pianist', 'keyboard'],
            'drums': ['drums', 'drummer', 'percussion', 'beat', 'rhythm'],
            'violin': ['violin', 'violins', 'strings', 'orchestra'],
            'synthesizer': ['synthesizer', 'synth', 'electronic', 'digital'],
        }
        
        # Buscar variantes
        for base_genre, variants in genre_variants.items():
            if any(variant in prompt_lower for variant in variants):
                if base_genre in self.genre_keywords:
                    found_genres.add(base_genre)
                    
        for base_mood, variants in mood_variants.items():
            if any(variant in prompt_lower for variant in variants):
                if base_mood in self.mood_keywords:
                    found_moods.add(base_mood)
                    
        for base_instrument, variants in instrument_variants.items():
            if any(variant in prompt_lower for variant in variants):
                if base_instrument in self.instrument_keywords:
                    found_instruments.add(base_instrument)
        
        return {
            'genres': found_genres,
            'moods': found_moods,
            'instruments': found_instruments
        }
    
    def calculate_prompt_coverage(self, playlist_songs: List[Dict], prompt: str) -> Dict[str, float]:
        """
        Métrica 1: Cobertura del prompt
        
        Extrae palabras clave del prompt y calcula qué porcentaje de canciones
        las cubren en sus metadatos (géneros, moods, instrumentos).
        
        Args:
            playlist_songs: Lista de canciones en la playlist
            prompt: Texto del prompt original
            
        Returns:
            Dict con métricas de cobertura
        """
        if not playlist_songs or len(playlist_songs) == 0:
            return {'prompt_coverage': 0.0, 'covered_songs': 0, 'total_keywords': 0}
            
        # Extraer keywords del prompt
        keywords = self.extract_keywords_from_prompt(prompt)
        all_keywords = keywords['genres'] | keywords['moods'] | keywords['instruments']
        
        if not all_keywords:
            return {'prompt_coverage': 0.0, 'covered_songs': 0, 'total_keywords': 0}
        
        # Obtener información detallada de las canciones
        song_ids = [song['track_id'] for song in playlist_songs]
        song_info = self.songs_df[self.songs_df['id'].isin(song_ids)]
        
        covered_songs = 0
        
        for _, song in song_info.iterrows():
            # Unificar todas las etiquetas de la canción
            song_tags = set()
            song_tags.update(song['genres'])
            song_tags.update(song['moods'])
            song_tags.update(song['instruments'])
            
            # Verificar si la canción cubre al menos una keyword
            if any(keyword.lower() in [tag.lower() for tag in song_tags] for keyword in all_keywords):
                covered_songs += 1
        
        coverage = covered_songs / len(playlist_songs)
        
        return {
            'prompt_coverage': coverage,
            'covered_songs': covered_songs,
            'total_songs': len(playlist_songs),
            'total_keywords': len(all_keywords),
            'keywords_by_category': {
                'genres': list(keywords['genres']),
                'moods': list(keywords['moods']),
                'instruments': list(keywords['instruments'])
            }
        }
    
    def calculate_diversity_index(self, playlist_songs: List[Dict], dimension: str = 'artist') -> Dict[str, float]:
        """
        Métrica 2: Diversidad de artistas o álbumes usando índice de Gini-Simpson
        
        D = 1 - Σ(pi²) donde pi es la proporción de canciones del artista/álbum i
        
        Args:
            playlist_songs: Lista de canciones en la playlist
            dimension: 'artist' o 'album'
            
        Returns:
            Dict con índice de diversidad
        """
        if not playlist_songs or len(playlist_songs) == 0:
            return {'diversity_index': 0.0, 'unique_entities': 0}
        
        # Obtener información de las canciones
        song_ids = [song['track_id'] for song in playlist_songs]
        song_info = self.songs_df[self.songs_df['id'].isin(song_ids)]
        
        if len(song_info) == 0:
            return {'diversity_index': 0.0, 'unique_entities': 0}
        
        # Seleccionar la columna apropiada
        column_map = {'artist': 'artist_id', 'album': 'album_id'}
        if dimension not in column_map:
            raise ValueError(f"Dimension must be 'artist' or 'album', got {dimension}")
        
        column = column_map[dimension]
        
        # Contar ocurrencias
        entity_counts = song_info[column].value_counts()
        total_songs = len(song_info)
        
        # Calcular proporciones
        proportions = entity_counts / total_songs
        
        # Calcular índice de Gini-Simpson: D = 1 - Σ(pi²)
        gini_simpson = 1 - sum(p**2 for p in proportions)
        
        return {
            'diversity_index': gini_simpson,
            'unique_entities': len(entity_counts),
            'total_songs': total_songs,
            'entity_distribution': dict(zip(entity_counts.index, proportions.values))
        }
    
    def calculate_shannon_entropy(self, playlist_songs: List[Dict], taxonomies: List[str] = None) -> Dict[str, float]:
        """
        Métrica 3: Entropía de Shannon para géneros/moods/instrumentos
        
        H = -Σ(pt * log2(pt)) donde pt es la proporción del tag t
        
        Args:
            playlist_songs: Lista de canciones en la playlist
            taxonomies: Lista de taxonomías a incluir ['genres', 'moods', 'instruments']
                       Si es None, incluye todas
                       
        Returns:
            Dict con entropías por taxonomía y global
        """
        if not playlist_songs or len(playlist_songs) == 0:
            return {'shannon_entropy': 0.0, 'taxonomies': {}}
        
        if taxonomies is None:
            taxonomies = ['genres', 'moods', 'instruments']
        
        # Obtener información de las canciones
        song_ids = [song['track_id'] for song in playlist_songs]
        song_info = self.songs_df[self.songs_df['id'].isin(song_ids)]
        
        if len(song_info) == 0:
            return {'shannon_entropy': 0.0, 'taxonomies': {}}
        
        results = {'taxonomies': {}}
        all_tags = []
        
        # Calcular entropía por taxonomía
        for taxonomy in taxonomies:
            tags_in_taxonomy = []
            
            for _, song in song_info.iterrows():
                tags_in_taxonomy.extend(song[taxonomy])
            
            if not tags_in_taxonomy:
                results['taxonomies'][taxonomy] = {
                    'entropy': 0.0,
                    'unique_tags': 0,
                    'total_tags': 0
                }
                continue
            
            # Contar frecuencias
            tag_counts = Counter(tags_in_taxonomy)
            total_tags = len(tags_in_taxonomy)
            
            # Calcular entropía de Shannon
            entropy = 0.0
            for count in tag_counts.values():
                proportion = count / total_tags
                if proportion > 0:  # Evitar log(0)
                    entropy -= proportion * log2(proportion)
            
            results['taxonomies'][taxonomy] = {
                'entropy': entropy,
                'unique_tags': len(tag_counts),
                'total_tags': total_tags,
                'tag_distribution': dict(tag_counts)
            }
            
            # Agregar al conjunto global
            all_tags.extend(tags_in_taxonomy)
        
        # Calcular entropía global (unión de todas las taxonomías)
        if all_tags:
            global_tag_counts = Counter(all_tags)
            total_global_tags = len(all_tags)
            
            global_entropy = 0.0
            for count in global_tag_counts.values():
                proportion = count / total_global_tags
                if proportion > 0:
                    global_entropy -= proportion * log2(proportion)
            
            results['shannon_entropy'] = global_entropy
            results['global_stats'] = {
                'unique_tags': len(global_tag_counts),
                'total_tags': total_global_tags
            }
        else:
            results['shannon_entropy'] = 0.0
            results['global_stats'] = {'unique_tags': 0, 'total_tags': 0}
        
        return results
    
    def calculate_all_semantic_metrics(self, playlist_songs: List[Dict], prompt: str) -> Dict[str, Any]:
        """
        Calcula todas las métricas semánticas para una playlist.
        
        Args:
            playlist_songs: Lista de canciones en la playlist
            prompt: Texto del prompt original
            
        Returns:
            Dict con todas las métricas semánticas
        """
        if not playlist_songs:
            return {
                'prompt_coverage': {'prompt_coverage': 0.0},
                'artist_diversity': {'diversity_index': 0.0},
                'album_diversity': {'diversity_index': 0.0},
                'semantic_entropy': {'shannon_entropy': 0.0}
            }
        
        results = {}
        
        # 1. Cobertura del prompt
        results['prompt_coverage'] = self.calculate_prompt_coverage(playlist_songs, prompt)
        
        # 2. Diversidad de artistas
        results['artist_diversity'] = self.calculate_diversity_index(playlist_songs, 'artist')
        
        # 3. Diversidad de álbumes  
        results['album_diversity'] = self.calculate_diversity_index(playlist_songs, 'album')
        
        # 4. Entropía semántica
        results['semantic_entropy'] = self.calculate_shannon_entropy(playlist_songs)
        
        return results
    
    def print_metrics_summary(self, metrics: Dict[str, Any], prompt: str = ""):
        """
        Imprime un resumen legible de las métricas calculadas.
        
        Args:
            metrics: Dict con métricas calculadas
            prompt: Prompt original (opcional)
        """
        print("\n" + "="*60)
        print("📊 SEMANTIC METRICS SUMMARY")
        print("="*60)
        
        if prompt:
            print(f"🎯 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print()
        
        # Cobertura del prompt
        pc = metrics.get('prompt_coverage', {})
        print(f"1️⃣ PROMPT COVERAGE")
        print(f"   📈 Coverage: {pc.get('prompt_coverage', 0):.3f} ({pc.get('covered_songs', 0)}/{pc.get('total_songs', 0)} songs)")
        print(f"   🏷️  Keywords found: {pc.get('total_keywords', 0)}")
        if 'keywords_by_category' in pc:
            kbc = pc['keywords_by_category']
            print(f"      • Genres: {', '.join(kbc.get('genres', [])) or 'None'}")
            print(f"      • Moods: {', '.join(kbc.get('moods', [])) or 'None'}")
            print(f"      • Instruments: {', '.join(kbc.get('instruments', [])) or 'None'}")
        
        # Diversidad de artistas
        ad = metrics.get('artist_diversity', {})
        print(f"\n2️⃣ ARTIST DIVERSITY (Gini-Simpson)")
        print(f"   🎭 Diversity Index: {ad.get('diversity_index', 0):.3f}")
        print(f"   👥 Unique Artists: {ad.get('unique_entities', 0)}")
        
        # Diversidad de álbumes
        alb = metrics.get('album_diversity', {})
        print(f"\n3️⃣ ALBUM DIVERSITY (Gini-Simpson)")
        print(f"   💿 Diversity Index: {alb.get('diversity_index', 0):.3f}")
        print(f"   📀 Unique Albums: {alb.get('unique_entities', 0)}")
        
        # Entropía semántica
        se = metrics.get('semantic_entropy', {})
        print(f"\n4️⃣ SEMANTIC ENTROPY (Shannon)")
        print(f"   🌐 Global Entropy: {se.get('shannon_entropy', 0):.3f}")
        
        taxonomies = se.get('taxonomies', {})
        for tax_name, tax_data in taxonomies.items():
            print(f"   📂 {tax_name.capitalize()}: {tax_data.get('entropy', 0):.3f} ({tax_data.get('unique_tags', 0)} unique)")
        
        print("="*60)

    def print_playlist_songs_info(self, playlist_songs: List[Dict]):
        """
        Imprime información detallada de las canciones en la playlist.
        
        Args:
            playlist_songs: Lista de canciones en la playlist
        """
        if not playlist_songs:
            print("❌ No songs in playlist")
            return
        
        song_ids = [song['track_id'] for song in playlist_songs]
        song_info = self.songs_df[self.songs_df['id'].isin(song_ids)]
        
        print(f"\n🎵 PLAYLIST SONGS DETAILS ({len(song_info)} songs)")
        print("="*80)
        
        for i, (_, song) in enumerate(song_info.iterrows(), 1):
            # Encontrar la similaridad correspondiente
            similarity = next((s['similarity'] for s in playlist_songs if s['track_id'] == song['id']), 'N/A')
            
            print(f"\n{i:2d}. 🎶 Song ID: {song['id']} | Similarity: {similarity}")
            print(f"    ⏱️  Duration: {song['durationInSec']:.1f}s")
            print(f"    👤 Artist ID: {song['artist_id']} | 💿 Album ID: {song['album_id']}")
            print(f"    📂 Chunk: {song['chunk_nr']}")
            
            # Géneros
            genres = song['genres'] if song['genres'] else []
            genres_str = ', '.join(genres) if genres else "None"
            print(f"    🎭 Genres: {genres_str}")
            
            # Moods
            moods = song['moods'] if song['moods'] else []
            moods_str = ', '.join(moods) if moods else "None"
            print(f"    😊 Moods: {moods_str}")
            
            # Instrumentos
            instruments = song['instruments'] if song['instruments'] else []
            instruments_str = ', '.join(instruments) if instruments else "None"
            print(f"    🎸 Instruments: {instruments_str}")
            
            # Tags totales
            all_tags = set(genres + moods + instruments)
            print(f"    🏷️  Total unique tags: {len(all_tags)}")
        
        print("="*80)


def test_semantic_metrics():
    """
    Función de prueba para las métricas semánticas.
    """
    print("🧪 TESTING SEMANTIC METRICS")
    print("="*50)
    
    # Inicializar calculadora
    calc = SemanticMetricsCalculator()
    
    # Ejemplo de playlist ficticia para testing
    test_playlist = [
        {'track_id': 973758, 'similarity': 0.85},
        {'track_id': 1396664, 'similarity': 0.80},
        {'track_id': 1397366, 'similarity': 0.75},
        {'track_id': 199625, 'similarity': 0.70},
        {'track_id': 205437, 'similarity': 0.65},
        {'track_id': 212535, 'similarity': 0.60},
        {'track_id': 250160, 'similarity': 0.55},
        {'track_id': 250990, 'similarity': 0.50},
        {'track_id': 391418, 'similarity': 0.45},
        {'track_id': 1400508, 'similarity': 0.40}
    ]
    
    test_prompt = "Create a dreamy Christmas playlist with peaceful and relaxing electronic music"
    
    print(f"🎯 Test Prompt: {test_prompt}")
    
    # Imprimir características de las canciones de prueba
    calc.print_playlist_songs_info(test_playlist)
    
    # Mostrar keywords extraídas del prompt
    keywords = calc.extract_keywords_from_prompt(test_prompt)
    print(f"\n🔍 KEYWORDS EXTRACTED FROM PROMPT")
    print("="*50)
    print(f"🎭 Genres: {', '.join(keywords['genres']) if keywords['genres'] else 'None'}")
    print(f"😊 Moods: {', '.join(keywords['moods']) if keywords['moods'] else 'None'}")
    print(f"🎸 Instruments: {', '.join(keywords['instruments']) if keywords['instruments'] else 'None'}")
    all_keywords = keywords['genres'] | keywords['moods'] | keywords['instruments']
    print(f"🏷️  Total keywords: {len(all_keywords)} -> {', '.join(all_keywords)}")
    
    # Calcular métricas
    print(f"\n🧮 CALCULATING METRICS...")
    metrics = calc.calculate_all_semantic_metrics(test_playlist, test_prompt)
    
    # Mostrar resultados
    calc.print_metrics_summary(metrics, test_prompt)
    
    # Análisis adicional
    print(f"\n🔬 DETAILED ANALYSIS")
    print("="*50)
    
    # Análisis de cobertura por canción
    song_ids = [song['track_id'] for song in test_playlist]
    song_info = calc.songs_df[calc.songs_df['id'].isin(song_ids)]
    
    print(f"📊 Coverage Analysis per Song:")
    covered_count = 0
    for i, (_, song) in enumerate(song_info.iterrows(), 1):
        song_tags = set()
        song_tags.update(song['genres'])
        song_tags.update(song['moods'])
        song_tags.update(song['instruments'])
        
        # Verificar si cubre keywords
        covers_keywords = any(keyword.lower() in [tag.lower() for tag in song_tags] for keyword in all_keywords)
        covered_count += covers_keywords
        
        matching_keywords = [kw for kw in all_keywords if kw.lower() in [tag.lower() for tag in song_tags]]
        status = "✅ COVERS" if covers_keywords else "❌ NO COVER"
        
        print(f"   {i:2d}. Song {song['id']}: {status}")
        if matching_keywords:
            print(f"       Matches: {', '.join(matching_keywords)}")
    
    print(f"\n📈 Summary: {covered_count}/{len(song_info)} songs cover prompt keywords")
    
    return metrics


if __name__ == "__main__":
    # Ejecutar pruebas
    test_metrics = test_semantic_metrics() 