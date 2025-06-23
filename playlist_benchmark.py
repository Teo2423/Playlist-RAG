import os
import json
import csv
import pandas as pd
import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from datetime import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import openai
from tqdm import tqdm
import re
from pathlib import Path

# Importar métricas semánticas
from semantic_metrics import SemanticMetricsCalculator

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_diverse_prompts(num_prompts=500):
    """
    Genera prompts diversos usando GPT-4o-mini para crear playlists variadas.
    Cada llamada genera UNA descripción única.
    """
    print(f"🎯 Generating {num_prompts} diverse prompts using GPT-4o-mini...")
    print(f"📞 Making {num_prompts} individual API calls for maximum diversity...")
    
    # Categorías expandidas para máxima diversidad
    prompt_categories = [
        # Géneros principales
        "rock subgenres (progressive, alternative, indie, post-punk, shoegaze, etc.)",
        "electronic music styles (house, techno, ambient, dubstep, drum & bass, etc.)",
        "hip-hop and rap variations (trap, boom-bap, conscious rap, drill, etc.)",
        "pop music evolution (synth-pop, indie pop, k-pop, electro-pop, etc.)",
        "jazz styles (fusion, bebop, smooth jazz, free jazz, etc.)",
        "classical music periods and forms",
        "folk and world music traditions",
        "metal subgenres (death, black, progressive, power, etc.)",
        "reggae and caribbean styles",
        "country and americana variations",
        
        # Estados de ánimo específicos
        "melancholic and introspective moods",
        "energetic and uplifting emotions",
        "nostalgic and romantic feelings",
        "dark and mysterious atmospheres",
        "peaceful and meditative states",
        "angry and rebellious emotions",
        "dreamy and ethereal moods",
        "anxious and tense feelings",
        "euphoric and celebratory emotions",
        "contemplative and philosophical moods",
        
        # Actividades y contextos
        "workout and fitness routines",
        "study and concentration sessions",
        "driving and road trips",
        "cooking and food preparation",
        "cleaning and household chores",
        "creative work and artistic activities",
        "social gatherings and parties",
        "relaxation and stress relief",
        "morning routines and wake-up",
        "nighttime and sleep preparation",
        
        # Épocas y décadas específicas
        "1960s psychedelic and counterculture",
        "1970s disco and funk era",
        "1980s new wave and synth era",
        "1990s grunge and alternative explosion",
        "2000s indie and emo revival",
        "2010s electronic and EDM boom",
        "2020s contemporary trends",
        "vintage and retro revivals",
        "future and experimental sounds",
        "timeless and classic compositions",
        
        # Instrumentos y sonoridades
        "guitar-driven compositions (acoustic, electric, fingerpicking)",
        "piano and keyboard-focused pieces",
        "synthesizer and electronic textures",
        "string sections and orchestral arrangements",
        "percussion and rhythm-heavy tracks",
        "brass and wind instruments",
        "vocal harmonies and a cappella",
        "bass-heavy and low-end focused",
        "minimalist and sparse arrangements",
        "complex and layered productions",
        
        # Características técnicas de audio
        "lo-fi and analog warmth",
        "high-fidelity and pristine production",
        "reverb-drenched and spacious sounds",
        "compressed and punchy dynamics",
        "wide stereo and immersive soundscapes",
        "mono and vintage recording techniques",
        "distorted and overdriven tones",
        "clean and crystalline audio",
        "ambient and atmospheric textures",
        "sharp and aggressive mixing",
        
        # Tempos y energías
        "slow and downtempo grooves",
        "moderate and walking pace",
        "fast and high-energy rhythms",
        "varying and dynamic tempo changes",
        "syncopated and off-beat patterns",
        "steady and metronomic pulses",
        "rubato and flexible timing",
        "polyrhythmic and complex meters",
        "simple and straightforward beats",
        "breakbeats and irregular patterns",
        
        # Vocales y estilos de canto
        "powerful and operatic vocals",
        "intimate and whispered singing",
        "rough and gravelly voices",
        "smooth and silky tones",
        "high-pitched and soaring melodies",
        "deep and resonant low voices",
        "harmonized and layered vocals",
        "spoken word and rap delivery",
        "falsetto and head voice techniques",
        "instrumental and wordless music",
        
        # Culturas y regiones
        "Latin American rhythms and styles",
        "African and Afrobeat influences",
        "Asian and Eastern musical traditions",
        "European classical and folk heritage",
        "Middle Eastern and Arabic scales",
        "Indian and subcontinental music",
        "Celtic and Irish traditional music",
        "Nordic and Scandinavian sounds",
        "Mediterranean and Southern European",
        "Indigenous and native musical traditions",
        
        # Ocasiones y eventos
        "wedding and celebration music",
        "funeral and memorial services",
        "holiday and seasonal themes",
        "graduation and achievement ceremonies",
        "romantic dates and intimate moments",
        "friendship and bonding activities",
        "travel and adventure soundtracks",
        "spiritual and religious contexts",
        "protest and social movement songs",
        "background and ambient listening",
        
        # Temáticas líricas
        "love and relationship stories",
        "social justice and political commentary",
        "personal growth and self-discovery",
        "nature and environmental themes",
        "urban life and city experiences",
        "childhood and coming-of-age narratives",
        "loss and grief processing",
        "hope and resilience messages",
        "fantasy and storytelling elements",
        "everyday life and mundane experiences",
        
        # Estilos de producción
        "DIY and bedroom recording aesthetics",
        "studio-polished and commercial production",
        "live and concert recording atmospheres",
        "field recording and found sound elements",
        "sampling and hip-hop production techniques",
        "analog and vintage recording methods",
        "digital and modern production styles",
        "collaborative and band-oriented recordings",
        "solo artist and intimate productions",
        "experimental and avant-garde approaches",
        
        # Elementos musicales específicos
        "catchy hooks and memorable melodies",
        "complex chord progressions and harmony",
        "simple and repetitive structures",
        "improvisation and spontaneous elements",
        "structured and composed arrangements",
        "call-and-response patterns",
        "build-ups and dramatic crescendos",
        "quiet-loud dynamics and contrasts",
        "continuous and flowing compositions",
        "fragmented and episodic structures",
        
        # Contextos sociales y culturales
        "underground and alternative scenes",
        "mainstream and popular culture",
        "academic and intellectual circles",
        "youth and teenage experiences",
        "adult contemporary and mature themes",
        "cross-generational and family-friendly",
        "niche and specialized communities",
        "global and international perspectives",
        "local and regional identity",
        "timeless and universal appeal"
    ]
    
    system_prompt = """You are a music expert who creates varied and creative playlist descriptions. 
    Generate ONE specific, detailed, and unique playlist description that covers different genres, emotions, 
    activities, eras, instruments, and sonic characteristics.
    
    The description should be:
    - Specific and detailed (not generic)
    - Realistic and practical for music recommendation
    - Between 8-25 words
    - In English
    - Musically accurate and descriptive
    - Focused on creating a cohesive playlist theme
    
    Examples of the type of descriptions I need:
    - "Melancholic indie folk with fingerpicked guitars for rainy afternoon reflection"
    - "High-energy 80s synth-pop with driving beats for morning workouts"
    - "Ambient electronic textures with field recordings for deep focus sessions"
    - "Gritty blues-rock with harmonica and slide guitar from the Mississippi Delta"
    - "Minimalist piano compositions with subtle strings for evening meditation"
    - "Afrobeat rhythms with heavy percussion and brass sections for dancing"
    - "Lo-fi hip-hop beats with vinyl crackle for studying and concentration"
    - "Post-punk revival with angular guitars and urgent vocals from the 2000s"
    """
    
    all_prompts = []
    failed_attempts = 0
    
    # Generar prompts uno por uno
    for i in tqdm(range(num_prompts), desc="Generating individual prompts"):
        # Seleccionar categorías aleatorias para este prompt específico
        num_categories = np.random.randint(3, 8)  # Entre 3 y 7 categorías por prompt
        selected_categories = np.random.choice(prompt_categories, size=num_categories, replace=False)
        categories_text = ", ".join(selected_categories)
        
        user_prompt = f"""Create ONE unique playlist description focusing on these musical aspects: {categories_text}.
        
        Make it specific, creative, and different from typical playlist descriptions.
        Return only the description, nothing else."""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.95,  # Máxima creatividad
                max_tokens=150     # Suficiente para una descripción
            )
            
            prompt = response.choices[0].message.content.strip()
            
            # Limpiar el prompt
            prompt = re.sub(r'^\d+[\.\-\)]\s*', '', prompt)  # Remover numeración
            prompt = prompt.strip('"\'')                      # Remover comillas
            prompt = prompt.replace('\n', ' ').strip()        # Una sola línea
            
            # Validar longitud y calidad
            if prompt and len(prompt) > 20 and len(prompt) < 400:
                all_prompts.append(prompt)
                if (i + 1) % 50 == 0:
                    print(f"   ✅ Generated {len(all_prompts)} valid prompts so far...")
            else:
                failed_attempts += 1
                print(f"   ⚠️  Prompt {i+1} rejected (length: {len(prompt)}): '{prompt[:50]}...'")
            
            # Pausa para no sobrecargar la API
            time.sleep(0.3)
            
        except Exception as e:
            failed_attempts += 1
            print(f"   ❌ Error generating prompt {i+1}: {str(e)}")
            time.sleep(1)  # Pausa más larga en caso de error
            continue
    
    # Remover duplicados manteniendo orden
    unique_prompts = list(dict.fromkeys(all_prompts))
    
    print(f"\n✅ Generation completed!")
    print(f"📊 Generated {len(unique_prompts)} unique prompts")
    print(f"❌ Failed attempts: {failed_attempts}")
    print(f"🎯 Success rate: {len(unique_prompts)/(num_prompts)*100:.1f}%")
    
    # Si no tenemos suficientes prompts, mostrar advertencia
    if len(unique_prompts) < num_prompts * 0.8:  # Menos del 80% exitoso
        print(f"⚠️  Warning: Only {len(unique_prompts)} prompts generated out of {num_prompts} requested")
        print(f"💡 Consider running again or adjusting parameters")
    
    return unique_prompts

def process_full_songs_results(search_results):
    """Función para procesar resultados de canciones completas"""
    processed_results = []
    for result in search_results:
        payload = result.payload
        processed_results.append({
            'filename': payload['filename'],
            'track_id': payload.get('track_id', 'N/A'),
            'similarity': result.score,
            'file_path': payload['file_path'],
            'total_duration': payload['total_duration'],
            'embedding_type': 'full_song'
        })
    return processed_results

def get_playlist_from_query(client, processor, model, query_text, search_type, collection_name, target_songs=10):
    """
    Genera una playlist basada en un query de texto y tipo de embedding específico.
    No guarda archivos, solo retorna la información de las canciones.
    ACTUALIZADO: Incluye soporte para full_songs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Generar embedding del texto
        inputs = processor(text=[query_text], return_tensors="pt").to(device)
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs)
            query_embedding = text_embed.cpu().numpy().flatten()
        
        # Buscar según el tipo
        if search_type == "chunks":
            processed_results = search_unique_songs_from_chunks(
                client, collection_name, query_embedding, target_songs
            )
        else:
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=target_songs
            )
            
            if search_type == "songs":
                processed_results = process_song_results(search_results)
            elif search_type == "weighted":
                processed_results = process_weighted_results(search_results)
            elif search_type == "representative":
                processed_results = process_representative_results(search_results)
            elif search_type == "full_songs":
                processed_results = process_full_songs_results(search_results)
            else:
                return None
        
        return processed_results
        
    except Exception as e:
        print(f"❌ Error generating playlist for '{query_text}' with {search_type}: {str(e)}")
        return None

# Reutilizar funciones del playlist_creator.py
def search_unique_songs_from_chunks(client, collection_name, query_embedding, target_unique_songs):
    """Función copiada de playlist_creator.py"""
    unique_songs = {}
    search_limit = 20
    max_search_limit = 200
    
    while len(unique_songs) < target_unique_songs and search_limit <= max_search_limit:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=search_limit
        )
        
        for result in search_results:
            payload = result.payload
            track_id = payload.get('track_id')
            
            if track_id is None:
                continue
                
            if (track_id not in unique_songs or 
                result.score > unique_songs[track_id]['similarity']):
                
                unique_songs[track_id] = {
                    'filename': payload['filename'],
                    'track_id': track_id,
                    'similarity': result.score,
                    'file_path': payload['file_path'],
                    'total_duration': payload['total_duration'],
                    'embedding_type': 'chunk'
                }
        
        if len(unique_songs) < target_unique_songs:
            search_limit += 30
        else:
            break
    
    sorted_results = sorted(unique_songs.values(), key=lambda x: x['similarity'], reverse=True)
    return sorted_results[:target_unique_songs]

def process_song_results(search_results):
    """Función copiada de playlist_creator.py"""
    processed_results = []
    for result in search_results:
        payload = result.payload
        processed_results.append({
            'filename': payload['filename'],
            'track_id': payload.get('track_id', 'N/A'),
            'similarity': result.score,
            'file_path': payload['file_path'],
            'total_duration': payload['total_duration'],
            'embedding_type': 'song_average'
        })
    return processed_results

def process_weighted_results(search_results):
    """Función copiada de playlist_creator.py"""
    processed_results = []
    for result in search_results:
        payload = result.payload
        processed_results.append({
            'filename': payload['filename'],
            'track_id': payload.get('track_id', 'N/A'),
            'similarity': result.score,
            'file_path': payload['file_path'],
            'total_duration': payload['total_duration'],
            'embedding_type': 'weighted_pooling'
        })
    return processed_results

def process_representative_results(search_results):
    """Función copiada de playlist_creator.py"""
    processed_results = []
    for result in search_results:
        payload = result.payload
        processed_results.append({
            'filename': payload['filename'],
            'track_id': payload.get('track_id', 'N/A'),
            'similarity': result.score,
            'file_path': payload['file_path'],
            'total_duration': payload['total_duration'],
            'embedding_type': 'representative_chunk'
        })
    return processed_results

def extract_audio_features(file_path):
    """
    Extrae características de audio necesarias para las métricas.
    CORREGIDO: Analiza toda la canción, no solo los primeros 60 segundos.
    """
    try:
        # Cargar la canción COMPLETA para obtener métricas representativas
        y, sr = librosa.load(file_path, sr=22050, mono=True)  # Reducir sr para eficiencia
        
        if len(y) < sr * 0.5:  # Si la canción es menor a 0.5 segundos, saltar
            return None
        
        # BPM - análisis de toda la canción
        try:
            # Usar hop_length optimizado para canciones completas
            hop_length = 512
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, trim=False)
            bpm = float(tempo)
            
            # Validar BPM razonable
            if bpm < 50 or bpm > 300:
                # Segundo intento con parámetros diferentes
                try:
                    tempo_alt, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024, start_bpm=120)
                    bpm = float(tempo_alt) if 50 <= tempo_alt <= 300 else 120.0
                except:
                    bpm = 120.0
                
        except Exception as e:
            # Si falla el beat tracking, usar estimación por defecto
            bpm = 120.0
        
        # Key (estimación usando chroma de TODA la canción)
        try:
            # Calcular chroma para toda la canción
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            
            # Obtener perfil de tonalidad promediando toda la canción
            key_profile = np.mean(chroma, axis=1)
            
            # Normalizar el perfil
            key_profile = key_profile / np.sum(key_profile)
            
            # Encontrar la tonalidad más fuerte
            key_idx = np.argmax(key_profile)
            key_camelot = key_idx + 1  # Convertir a sistema Camelot (1-12)
            
            # Validar que hay suficiente confianza en la detección
            confidence = key_profile[key_idx]
            if confidence < 0.1:  # Si la confianza es muy baja
                key_camelot = 1  # Default a C major
                
        except Exception as e:
            key_camelot = 1  # Default a C major
        
        # Spectral Centroid (promedio de toda la canción)
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            
            # Validar valor razonable
            if spectral_centroid_mean < 100 or spectral_centroid_mean > 8000:
                spectral_centroid_mean = 2000.0  # Default value
                
        except Exception as e:
            spectral_centroid_mean = 2000.0  # Default value
        
        # Duración real de la canción
        duration = len(y) / sr
        
        return {
            'bpm': bpm,
            'key_camelot': key_camelot,
            'spectral_centroid': spectral_centroid_mean,
            'duration': duration
        }
        
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {str(e)}")
        return None

def calculate_playlist_metrics(playlist_songs, client, collection_name):
    """
    Calcula las métricas de calidad de playlist:
    - Intra-List Similarity (ILS)
    - Key Compactness (KC) 
    - BPM Dispersion (BPMσ)
    - Spectral Centroid Drift (SCD)
    
    ARREGLADO: Formato correcto de filtros de Qdrant
    """
    if not playlist_songs or len(playlist_songs) < 2:
        return None
    
    # Extraer features de audio
    features_list = []
    embeddings_list = []
    
    print(f"   🔍 Extracting features from {len(playlist_songs)} songs...")
    for song in tqdm(playlist_songs, desc="   Analyzing audio", leave=False):
        features = extract_audio_features(song['file_path'])
        if features:
            features_list.append(features)
            
            # Obtener embedding de la canción desde Qdrant - FORMATO CORRECTO
            try:
                # Crear filtro correctamente usando las clases de Qdrant
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="track_id",
                            match=MatchValue(value=song['track_id'])
                        )
                    ]
                )
                
                search_result = client.search(
                    collection_name=collection_name,
                    query_vector=[0.0] * 512,  # Vector dummy, usaremos filter
                    limit=1,
                    with_vectors=True,
                    query_filter=filter_condition
                )
                
                if search_result and len(search_result) > 0 and search_result[0].vector:
                    embeddings_list.append(search_result[0].vector)
                else:
                    print(f"⚠️ No embedding found for track_id: {song['track_id']}")
                    
            except Exception as e:
                print(f"⚠️ Error getting embedding: {str(e)}")
                continue
    
    if len(features_list) < 2 or len(embeddings_list) < 2:
        print(f"⚠️ Insufficient valid features: {len(features_list)} features, {len(embeddings_list)} embeddings")
        return None
    
    # 1. Intra-List Similarity (ILS) - Similitud promedio entre embeddings
    embeddings_array = np.array(embeddings_list)
    similarity_matrix = cosine_similarity(embeddings_array)
    # Tomar solo la parte triangular superior (sin diagonal)
    upper_triangle = np.triu(similarity_matrix, k=1)
    non_zero_similarities = upper_triangle[upper_triangle != 0]
    ils = float(np.mean(non_zero_similarities)) if len(non_zero_similarities) > 0 else 0.0
    
    # 2. Key Compactness (KC) - Dispersión de tonalidades
    keys = [f['key_camelot'] for f in features_list]
    if len(keys) > 1:
        key_distances = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                # Distancia circular en rueda Camelot (12 keys)
                diff = abs(keys[i] - keys[j])
                circular_diff = min(diff, 12 - diff)  # Distancia más corta en círculo
                key_distances.append(circular_diff)
        kc = float(np.mean(key_distances)) if key_distances else 0.0
    else:
        kc = 0.0
    
    # 3. BPM Dispersion (BPMσ) - Dispersión normalizada de BPM
    bpms = [f['bpm'] for f in features_list]
    if len(bpms) > 1:
        bpm_mean = np.mean(bpms)
        bpm_std = np.std(bpms)
        bmp_dispersion = float(bpm_std / bpm_mean) if bpm_mean > 0 else 0.0
    else:
        bmp_dispersion = 0.0
    
    # 4. Spectral Centroid Drift (SCD) - Variabilidad tímbrica
    centroids = [f['spectral_centroid'] for f in features_list]
    scd = float(np.std(centroids)) if len(centroids) > 1 else 0.0
    
    return {
        'ils': ils,
        'key_compactness': kc,
        'bmp_dispersion': bmp_dispersion,
        'spectral_centroid_drift': scd,
        'num_songs_analyzed': len(features_list),
        'avg_bpm': float(np.mean(bpms)) if bpms else 0.0,
        'avg_spectral_centroid': float(np.mean(centroids)) if centroids else 0.0
    }

def extract_semantic_metrics_flat(semantic_metrics):
    """
    Extrae métricas semánticas en formato plano para análisis.
    Copiado de apply_semantic_metrics_to_benchmarks.py
    """
    flat_metrics = {}
    
    # 1. Prompt Coverage
    flat_metrics['prompt_coverage'] = semantic_metrics['prompt_coverage']['prompt_coverage']
    
    # 2. Artist Diversity
    flat_metrics['artist_diversity'] = semantic_metrics['artist_diversity']['diversity_index']
    
    # 3. Album Diversity
    flat_metrics['album_diversity'] = semantic_metrics['album_diversity']['diversity_index']
    
    # 4. Entropías separadas por taxonomía
    semantic_entropy_data = semantic_metrics['semantic_entropy']
    taxonomies = semantic_entropy_data.get('taxonomies', {})
    
    flat_metrics['genre_entropy'] = taxonomies.get('genres', {}).get('entropy', 0.0)
    flat_metrics['mood_entropy'] = taxonomies.get('moods', {}).get('entropy', 0.0)
    flat_metrics['instrument_entropy'] = taxonomies.get('instruments', {}).get('entropy', 0.0)
    
    # 5. Entropía global (opcional)
    flat_metrics['global_semantic_entropy'] = semantic_entropy_data.get('shannon_entropy', 0.0)
    
    return flat_metrics

def calculate_comprehensive_playlist_metrics(playlist_songs, client, collection_name, 
                                           semantic_calculator, prompt_text):
    """
    Calcula métricas completas de playlist: audio + semánticas.
    ACTUALIZADO: Integra métricas semánticas directamente
    """
    if not playlist_songs or len(playlist_songs) < 2:
        return None
    
    # 1. Calcular métricas de audio (existentes)
    audio_metrics = calculate_playlist_metrics(playlist_songs, client, collection_name)
    if not audio_metrics:
        return None
    
    # 2. Calcular métricas semánticas
    try:
        print(f"      🧠 Calculating semantic metrics...")
        semantic_metrics = semantic_calculator.calculate_all_semantic_metrics(
            playlist_songs, prompt_text
        )
        
        # Extraer métricas planas
        flat_semantic_metrics = extract_semantic_metrics_flat(semantic_metrics)
        
        # Combinar todas las métricas
        comprehensive_metrics = {
            # Métricas de audio
            'audio_metrics': audio_metrics,
            
            # Métricas semánticas planas (para análisis estadístico)
            'semantic_metrics': flat_semantic_metrics,
            
            # Métricas semánticas detalladas (para análisis profundo)
            'semantic_metrics_detailed': semantic_metrics,
            
            # Métricas combinadas para análisis unificado
            'combined_metrics': {
                **audio_metrics,  # ILS, key_compactness, bmp_dispersion, spectral_centroid_drift
                **flat_semantic_metrics  # prompt_coverage, artist_diversity, etc.
            }
        }
        
        print(f"      ✅ Semantic metrics: PC={flat_semantic_metrics['prompt_coverage']:.3f}, "
              f"AD={flat_semantic_metrics['artist_diversity']:.3f}")
        
        return comprehensive_metrics
        
    except Exception as e:
        print(f"      ⚠️ Error calculating semantic metrics: {str(e)}")
        # Si fallan las semánticas, devolver solo las de audio
        return {
            'audio_metrics': audio_metrics,
            'semantic_metrics': {
                'prompt_coverage': 0.0,
                'artist_diversity': 0.0,
                'album_diversity': 0.0,
                'genre_entropy': 0.0,
                'mood_entropy': 0.0,
                'instrument_entropy': 0.0,
                'global_semantic_entropy': 0.0
            },
            'semantic_metrics_detailed': None,
            'combined_metrics': audio_metrics
        }

def run_benchmark():
    """
    Ejecuta el benchmark completo comparando las 5 colecciones de embeddings.
    ACTUALIZADO: Incluye métricas semánticas directamente
    """
    print("🚀 STARTING COMPREHENSIVE PLAYLIST QUALITY BENCHMARK")
    print("🧠 Including Audio + Semantic Metrics Analysis")
    print("=" * 60)
    
    # Configuración
    qdrant_storage_path = "./qdrant_storage"
    num_prompts = 200
    target_songs_per_playlist = 10
    
    # Configuración de colecciones - ACTUALIZADA con full_songs
    embedding_configs = {
        "chunks": ("chunks", "music_chunks_embeddings"),
        "simple_avg": ("songs", "music_songs_embeddings"), 
        "weighted_avg": ("weighted", "music_weighted_embeddings"),
        "representative": ("representative", "music_representative_embeddings"),
        "full_songs": ("full_songs", "music_full_songs_embeddings")
    }
    
    # Verificar base de datos
    if not os.path.exists(qdrant_storage_path):
        print("❌ Error: Qdrant database not found.")
        return
    
    # Inicializar componentes
    print("📚 Initializing components...")
    client = QdrantClient(path=qdrant_storage_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        print(f"   🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   🖥️  Using CPU")
    
    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    
    # Inicializar calculadora de métricas semánticas
    print("🧠 Initializing semantic metrics calculator...")
    try:
        semantic_calculator = SemanticMetricsCalculator("valid.tsv")
        print("   ✅ Semantic metrics calculator ready")
    except Exception as e:
        print(f"   ❌ Error initializing semantic calculator: {str(e)}")
        print("   💡 Continuing without semantic metrics...")
        semantic_calculator = None
    
    # Verificar que todas las colecciones existen
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"\n🔍 Checking collections availability:")
        missing_collections = []
        for emb_name, (search_type, collection_name) in embedding_configs.items():
            if collection_name in collection_names:
                collection_info = client.get_collection(collection_name)
                print(f"   ✅ {emb_name:15} -> {collection_name} ({collection_info.points_count} points)")
            else:
                print(f"   ❌ {emb_name:15} -> {collection_name} (NOT FOUND)")
                missing_collections.append(emb_name)
        
        if missing_collections:
            print(f"\n⚠️  WARNING: Missing collections: {missing_collections}")
            print(f"💡 Run indexer.py to create missing collections")
            print(f"🔄 Continuing with available collections...")
            # Remover colecciones faltantes
            for missing in missing_collections:
                del embedding_configs[missing]
        
        if not embedding_configs:
            print(f"❌ Error: No collections available for benchmarking.")
            return
            
    except Exception as e:
        print(f"❌ Error checking collections: {str(e)}")
        return
    
    # Generar prompts
    prompts = generate_diverse_prompts(num_prompts)
    
    # Crear carpeta para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"./benchmark_results_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Guardar prompts generados
    prompts_file = os.path.join(results_folder, "generated_prompts.txt")
    with open(prompts_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"{i:3d}. {prompt}\n")
    print(f"📝 Generated prompts saved to: {prompts_file}")
    
    # Almacenar todos los resultados
    all_results = []
    
    print(f"\n🎵 Processing {len(prompts)} prompts with {len(embedding_configs)} embedding types...")
    print(f"📊 Total playlists to generate: {len(prompts) * len(embedding_configs)}")
    print(f"🎼 Embedding types: {list(embedding_configs.keys())}")
    semantic_status = "✅ WITH semantic metrics" if semantic_calculator else "⚠️ WITHOUT semantic metrics"
    print(f"🧠 Analysis mode: {semantic_status}")
    
    # Procesar cada prompt
    for prompt_idx, prompt_text in enumerate(tqdm(prompts, desc="Processing prompts")):
        print(f"\n🎯 Prompt {prompt_idx + 1}/{len(prompts)}: '{prompt_text}'")
        
        prompt_results = {
            'prompt_id': prompt_idx,
            'prompt_text': prompt_text,
            'embeddings_results': {}
        }
        
        # Probar cada tipo de embedding
        for emb_name, (search_type, collection_name) in embedding_configs.items():
            print(f"   🔍 Testing {emb_name}...")
            
            try:
                # Generar playlist
                playlist = get_playlist_from_query(
                    client, processor, model, prompt_text, 
                    search_type, collection_name, target_songs_per_playlist
                )
                
                if playlist and len(playlist) >= 2:
                    # Calcular métricas completas (audio + semánticas)
                    if semantic_calculator:
                        comprehensive_metrics = calculate_comprehensive_playlist_metrics(
                            playlist, client, collection_name, semantic_calculator, prompt_text
                        )
                    else:
                        # Solo métricas de audio
                        audio_metrics = calculate_playlist_metrics(playlist, client, collection_name)
                        comprehensive_metrics = {
                            'audio_metrics': audio_metrics,
                            'semantic_metrics': None,
                            'combined_metrics': audio_metrics
                        } if audio_metrics else None
                    
                    if comprehensive_metrics:
                        prompt_results['embeddings_results'][emb_name] = {
                            'success': True,
                            'num_songs': len(playlist),
                            'metrics': comprehensive_metrics,
                            'songs': [{'track_id': s['track_id'], 'similarity': s['similarity']} for s in playlist]
                        }
                        
                        # Mostrar métricas clave
                        audio = comprehensive_metrics.get('audio_metrics', {})
                        semantic = comprehensive_metrics.get('semantic_metrics', {})
                        
                        if semantic:
                            print(f"      ✅ Audio: ILS={audio.get('ils', 0):.3f}, KC={audio.get('key_compactness', 0):.3f}")
                            print(f"      🧠 Semantic: PC={semantic.get('prompt_coverage', 0):.3f}, AD={semantic.get('artist_diversity', 0):.3f}")
                        else:
                            print(f"      ✅ Audio only: ILS={audio.get('ils', 0):.3f}, KC={audio.get('key_compactness', 0):.3f}")
                    else:
                        prompt_results['embeddings_results'][emb_name] = {
                            'success': False,
                            'error': 'Error calculating metrics'
                        }
                        print(f"      ❌ Error calculating metrics")
                else:
                    prompt_results['embeddings_results'][emb_name] = {
                        'success': False,
                        'error': 'Insufficient or empty playlist'
                    }
                    print(f"      ❌ Insufficient playlist: {len(playlist) if playlist else 0} songs")
                    
            except Exception as e:
                prompt_results['embeddings_results'][emb_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"      ❌ Error: {str(e)}")
        
        all_results.append(prompt_results)
        
        # Guardar resultados parciales cada 50 prompts
        if (prompt_idx + 1) % 50 == 0:
            partial_file = os.path.join(results_folder, f"partial_results_{prompt_idx + 1}.json")
            with open(partial_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Partial results saved: {partial_file}")
    
    # Guardar resultados finales
    results_file = os.path.join(results_folder, "complete_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Crear análisis estadístico (actualizado para incluir métricas semánticas)
    create_comprehensive_statistical_analysis(all_results, results_folder, semantic_calculator is not None)
    
    print(f"\n🎉 Comprehensive benchmark completed!")
    print(f"📁 Results saved in: {results_folder}")
    print(f"📊 Statistical analysis available at: {results_folder}/statistical_analysis.csv")

def create_comprehensive_statistical_analysis(all_results, results_folder, has_semantic_metrics):
    """
    Crea un análisis estadístico completo incluyendo métricas semánticas.
    ACTUALIZADO: Incluye análisis de métricas semánticas
    """
    print("\n📊 Creating comprehensive statistical analysis...")
    
    # Definir tipos de embeddings
    embedding_types = ['chunks', 'simple_avg', 'weighted_avg', 'representative', 'full_songs']
    
    # Métricas de audio
    audio_metrics_names = ['ils', 'key_compactness', 'bmp_dispersion', 'spectral_centroid_drift']
    
    # Métricas semánticas
    semantic_metrics_names = [
        'prompt_coverage', 'artist_diversity', 'album_diversity',
        'genre_entropy', 'mood_entropy', 'instrument_entropy', 'global_semantic_entropy'
    ]
    
    # Recopilar métricas por tipo de embedding
    audio_metrics_by_embedding = {
        emb_type: {metric: [] for metric in audio_metrics_names} 
        for emb_type in embedding_types
    }
    
    semantic_metrics_by_embedding = {
        emb_type: {metric: [] for metric in semantic_metrics_names} 
        for emb_type in embedding_types
    }
    
    success_counts = {emb: 0 for emb in embedding_types}
    total_counts = {emb: 0 for emb in embedding_types}
    
    # Recopilar datos
    for result in all_results:
        for emb_name, emb_result in result['embeddings_results'].items():
            if emb_name in embedding_types:  # Verificar que la colección existe
                total_counts[emb_name] += 1
                
                if emb_result.get('success', False) and 'metrics' in emb_result:
                    success_counts[emb_name] += 1
                    metrics_data = emb_result['metrics']
                    
                    # Métricas de audio
                    audio_metrics = metrics_data.get('audio_metrics', {})
                    for metric in audio_metrics_names:
                        if metric in audio_metrics:
                            audio_metrics_by_embedding[emb_name][metric].append(audio_metrics[metric])
                    
                    # Métricas semánticas (si disponibles)
                    if has_semantic_metrics:
                        semantic_metrics = metrics_data.get('semantic_metrics', {})
                        for metric in semantic_metrics_names:
                            if metric in semantic_metrics:
                                semantic_metrics_by_embedding[emb_name][metric].append(semantic_metrics[metric])
    
    # Crear DataFrame para análisis
    analysis_data = []
    
    for emb_name in embedding_types:
        if total_counts[emb_name] > 0:  # Solo incluir si se procesó esta colección
            success_rate = success_counts[emb_name] / total_counts[emb_name]
            
            row = {
                'embedding_type': emb_name,
                'success_rate': success_rate,
                'total_attempts': total_counts[emb_name],
                'successful_playlists': success_counts[emb_name]
            }
            
            # Estadísticas para métricas de audio
            for metric_name, values in audio_metrics_by_embedding[emb_name].items():
                prefix = f'audio_{metric_name}'
                if values:
                    row.update({
                        f'{prefix}_mean': np.mean(values),
                        f'{prefix}_std': np.std(values),
                        f'{prefix}_median': np.median(values),
                        f'{prefix}_min': np.min(values),
                        f'{prefix}_max': np.max(values)
                    })
                else:
                    row.update({
                        f'{prefix}_mean': np.nan,
                        f'{prefix}_std': np.nan,
                        f'{prefix}_median': np.nan,
                        f'{prefix}_min': np.nan,
                        f'{prefix}_max': np.nan
                    })
            
            # Estadísticas para métricas semánticas (si disponibles)
            if has_semantic_metrics:
                for metric_name, values in semantic_metrics_by_embedding[emb_name].items():
                    prefix = f'semantic_{metric_name}'
                    if values:
                        row.update({
                            f'{prefix}_mean': np.mean(values),
                            f'{prefix}_std': np.std(values),
                            f'{prefix}_median': np.median(values),
                            f'{prefix}_min': np.min(values),
                            f'{prefix}_max': np.max(values)
                        })
                    else:
                        row.update({
                            f'{prefix}_mean': np.nan,
                            f'{prefix}_std': np.nan,
                            f'{prefix}_median': np.nan,
                            f'{prefix}_min': np.nan,
                            f'{prefix}_max': np.nan
                        })
            
            analysis_data.append(row)
    
    # Guardar análisis
    df = pd.DataFrame(analysis_data)
    analysis_file = os.path.join(results_folder, "comprehensive_statistical_analysis.csv")
    df.to_csv(analysis_file, index=False)
    
    # Crear reporte resumen
    summary_file = os.path.join(results_folder, "comprehensive_summary_report.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("🎵 COMPREHENSIVE PLAYLIST BENCHMARK REPORT\n")
        f.write("🧠 Audio + Semantic Metrics Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📊 Total prompts tested: {len(all_results)}\n")
        f.write(f"🎯 Target songs per playlist: 10\n")
        f.write(f"🎼 Embedding types tested: {len(analysis_data)}\n")
        f.write(f"🧠 Semantic metrics: {'✅ Included' if has_semantic_metrics else '❌ Not available'}\n\n")
        
        f.write("📈 SUCCESS RATES BY EMBEDDING TYPE:\n")
        f.write("-" * 40 + "\n")
        for row in analysis_data:
            emb_name = row['embedding_type']
            success_rate = row['success_rate']
            successful = row['successful_playlists']
            total = row['total_attempts']
            f.write(f"{emb_name:15} {success_rate:6.1%} ({successful}/{total})\n")
        
        # Rankings por categorías
        f.write(f"\n🏆 AUDIO METRICS RANKINGS (lower = better, except ILS):\n")
        f.write("-" * 60 + "\n")
        
        for metric in audio_metrics_names:
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            
            # Calcular ranking
            metric_means = []
            for row in analysis_data:
                mean_val = row.get(f'audio_{metric}_mean')
                if not pd.isna(mean_val):
                    metric_means.append((row['embedding_type'], mean_val))
            
            # Ordenar por valor
            if metric == 'ils':
                metric_means.sort(key=lambda x: x[1], reverse=True)  # ILS: mayor es mejor
            else:
                metric_means.sort(key=lambda x: x[1])  # Otros: menor es mejor
            
            for i, (emb_name, mean_val) in enumerate(metric_means, 1):
                f.write(f"  {i}. {emb_name:15} {mean_val:.4f}\n")
        
        # Rankings semánticos (si disponibles)
        if has_semantic_metrics:
            f.write(f"\n🧠 SEMANTIC METRICS RANKINGS (higher = better):\n")
            f.write("-" * 60 + "\n")
            
            for metric in semantic_metrics_names:
                f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                
                # Calcular ranking
                metric_means = []
                for row in analysis_data:
                    mean_val = row.get(f'semantic_{metric}_mean')
                    if not pd.isna(mean_val):
                        metric_means.append((row['embedding_type'], mean_val))
                
                # Ordenar por valor (mayor es mejor para métricas semánticas)
                metric_means.sort(key=lambda x: x[1], reverse=True)
                
                for i, (emb_name, mean_val) in enumerate(metric_means, 1):
                    f.write(f"  {i}. {emb_name:15} {mean_val:.4f}\n")
        
        f.write(f"\n🎼 EMBEDDING TYPES EXPLANATION:\n")
        f.write("-" * 35 + "\n")
        f.write("• chunks:         Best matching chunks from songs\n")
        f.write("• simple_avg:     Average of all song chunks\n")
        f.write("• weighted_avg:   Weighted average of song chunks\n")
        f.write("• representative: Most representative chunk per song\n")
        f.write("• full_songs:     Complete song embeddings (no chunking)\n")
        
        if has_semantic_metrics:
            f.write(f"\n🧠 SEMANTIC METRICS EXPLANATION:\n")
            f.write("-" * 35 + "\n")
            f.write("• prompt_coverage:    How well playlist matches prompt\n")
            f.write("• artist_diversity:   Variety of artists in playlist\n")
            f.write("• album_diversity:    Variety of albums in playlist\n")
            f.write("• genre_entropy:      Musical genre diversity\n")
            f.write("• mood_entropy:       Emotional/mood diversity\n")
            f.write("• instrument_entropy: Instrumental diversity\n")
            f.write("• global_entropy:     Overall semantic diversity\n")
    
    print(f"✅ Comprehensive statistical analysis completed")
    print(f"📄 Summary report: {summary_file}")
    print(f"📊 Analysis data: {analysis_file}")

if __name__ == "__main__":
    print("🎵 COMPREHENSIVE PLAYLIST QUALITY BENCHMARK")
    print("📊 Comparing 5 types of musical embeddings")
    print("🧠 Including Audio + Semantic Metrics Analysis")
    print("🎼 Including: chunks, simple_avg, weighted_avg, representative, full_songs")
    print()
    
    # Verificar API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("💡 Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    run_benchmark()
