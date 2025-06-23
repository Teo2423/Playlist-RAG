import os
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import torch
from transformers import ClapModel, ClapProcessor
from qdrant_client import QdrantClient
from datetime import datetime
import re

def sanitize_filename(filename):
    """
    Sanitiza un nombre de archivo removiendo caracteres no válidos.
    """
    # Remover caracteres no válidos para nombres de archivo
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Limitar longitud
    if len(filename) > 100:
        filename = filename[:100]
    return filename.strip()

def choose_database_type():
    """
    Permite al usuario elegir entre las 4 colecciones de embeddings disponibles.
    """
    print("\n🎵 SELECCIÓN DE TIPO DE BÚSQUEDA")
    print("=" * 60)
    print("¿En qué base de datos quieres buscar?")
    print()
    
    print("1. 🧩 Embeddings por chunks de 30 segundos")
    print("   - Búsqueda más precisa en partes específicas de las canciones")
    print("   - Encuentra el mejor momento/parte de cada canción")
    print("   - Devuelve canciones completas (encontradas por chunk)")
    print()
    
    print("2. 🎵 Embeddings de promedio simple por canción")  
    print("   - Promedio aritmético simple de todos los chunks")
    print("   - Representa la canción completa como una sola unidad")
    print("   - Rápido y estable")
    print()
    
    print("3. ⚖️  Embeddings de promedio ponderado por energía")
    print("   - Promedio ponderado según la energía de cada chunk")
    print("   - Da más peso a las partes con mayor energía musical")
    print("   - Ideal para encontrar canciones con energía similar")
    print()
    
    print("4. 🎯 Embeddings de chunk representativo")
    print("   - Usa el chunk más cercano al centroide de la canción")
    print("   - Representa la 'esencia' más típica de cada canción")
    print("   - Ideal para encontrar canciones con características similares")
    print()
    
    while True:
        choice = input("Elige una opción (1, 2, 3 o 4): ").strip()
        if choice == "1":
            return "chunks", "music_chunks_embeddings"
        elif choice == "2":
            return "songs", "music_songs_embeddings"
        elif choice == "3":
            return "weighted", "music_weighted_embeddings"
        elif choice == "4":
            return "representative", "music_representative_embeddings"
        else:
            print("❌ Opción inválida. Por favor elige 1, 2, 3 o 4.")

def create_text_based_playlist():
    """
    Crea una playlist basada en una descripción de texto usando embeddings de CLAP.
    Permite elegir entre las 4 colecciones de embeddings disponibles.
    """
    
    qdrant_storage_path = "./qdrant_storage"
    target_sample_rate = 48000  # Para el modelo CLAP
    original_sample_rate = 48000  # ✅ CAMBIO: Usar 48kHz (mismo que CLAP y canciones originales)
    target_unique_songs = 10  # Número de CANCIONES ÚNICAS para la playlist
    
    # Verificar que existe la base de datos
    if not os.path.exists(qdrant_storage_path):
        print("❌ Error: No se encontró la base de datos de Qdrant.")
        print("Ejecuta primero el indexer.py para crear la base de datos de embeddings.")
        return
    
    # Solicitar descripción de la playlist al usuario
    print("🎵 Generador de Playlist Basado en Texto")
    print("=" * 50)
    description = input("Describe la playlist que quieres crear: ").strip()
    
    if not description:
        print("❌ Error: Debes proporcionar una descripción.")
        return
    
    # Elegir tipo de base de datos (ahora con 4 opciones)
    search_type, collection_name = choose_database_type()
    
    print(f"\n🔍 Buscando en {search_type} para: '{description}'")
    print(f"🎯 Objetivo: {target_unique_songs} canciones únicas")
    print(f"📂 Colección: {collection_name}")
    
    # Inicializar cliente Qdrant y modelo CLAP
    print("📚 Cargando modelo y base de datos...")
    try:
        client = QdrantClient(path=qdrant_storage_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ClapModel.from_pretrained("laion/larger_clap_music").to(device)
        processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
    except Exception as e:
        print(f"❌ Error cargando modelo o base de datos: {str(e)}")
        return
    
    # Generar embedding del texto de descripción
    print("🧠 Generando embedding del texto...")
    try:
        inputs = processor(
            text=[description], 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs)
            query_embedding = text_embed.cpu().numpy().flatten()
    except Exception as e:
        print(f"❌ Error generando embedding del texto: {str(e)}")
        return
    
    # Buscar similares en la base de datos según el tipo
    print(f"🔍 Buscando {search_type} similares...")
    try:
        if search_type == "chunks":
            # Para chunks, necesitamos buscar más resultados para garantizar 10 canciones únicas
            processed_results = search_unique_songs_from_chunks(
                client, collection_name, query_embedding, target_unique_songs
            )
        else:
            # Para todos los otros tipos, buscamos directamente el número objetivo
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=target_unique_songs
            )
            
            # Procesar según el tipo de embedding
            if search_type == "songs":
                processed_results = process_song_results(search_results)
            elif search_type == "weighted":
                processed_results = process_weighted_results(search_results)
            elif search_type == "representative":
                processed_results = process_representative_results(search_results)
            else:
                print(f"❌ Tipo de búsqueda desconocido: {search_type}")
                return
                
    except Exception as e:
        print(f"❌ Error buscando en la base de datos: {str(e)}")
        return
    
    if not processed_results:
        print("❌ No se encontraron resultados en la base de datos.")
        return
    
    print(f"✅ Encontradas {len(processed_results)} canciones únicas")
    
    # Crear nombre de carpeta para la playlist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_type_labels = {
        "chunks": "chunks",
        "songs": "simple_avg", 
        "weighted": "weighted_avg",
        "representative": "representative"
    }
    search_type_label = search_type_labels.get(search_type, search_type)
    playlist_name = f"playlist_{search_type_label}_{sanitize_filename(description)}_{timestamp}"
    playlist_folder = f"./playlists/{playlist_name}"
    
    # Crear carpeta de playlist
    os.makedirs(playlist_folder, exist_ok=True)
    print(f"📁 Carpeta de playlist creada: {playlist_folder}")
    
    print(f"\n🎵 Resultados encontrados: {len(processed_results)} elementos")
    print("=" * 100)
    
    # Procesar y guardar archivos
    copied_count = 0
    failed_count = 0
    playlist_info = []
    
    for i, result_info in enumerate(processed_results, 1):
        print(f"{i:2d}. {result_info['display_name']}")
        print(f"    📊 Similitud: {result_info['similarity']:.4f}")
        print(f"    🎵 Track ID: {result_info['track_id']}")
        print(f"    ⏱️  Duración: {result_info['duration_info']}")
        print(f"    🧩 Info: {result_info['chunk_info']}")
        print(f"    📁 Origen: {result_info['file_path']}")
        
        # Verificar que el archivo original existe
        if not os.path.exists(result_info['file_path']):
            print(f"    ❌ Archivo original no encontrado: {result_info['file_path']}")
            failed_count += 1
            print("-" * 100)
            continue
        
        # Crear nombre de destino
        base_name = Path(result_info['filename']).stem
        extension = Path(result_info['filename']).suffix
        
        # Nomenclatura específica según el tipo
        if search_type == "chunks":
            destination_filename = f"{i:02d}_{base_name}_full_foundby_chunk{result_info['chunk_index']:02d}{extension}"
        elif search_type == "songs":
            destination_filename = f"{i:02d}_{base_name}_simple_avg{extension}"
        elif search_type == "weighted":
            destination_filename = f"{i:02d}_{base_name}_weighted_avg{extension}"
        elif search_type == "representative":
            destination_filename = f"{i:02d}_{base_name}_repr_chunk{result_info['chunk_index']:02d}{extension}"
        else:
            destination_filename = f"{i:02d}_{base_name}_full{extension}"
            
        destination_path = os.path.join(playlist_folder, destination_filename)
        
        # Procesar archivo de audio (siempre guarda la canción completa)
        try:
            success = process_audio_file(result_info, destination_path, search_type, original_sample_rate)
            
            if success:
                copied_count += 1
                result_info['playlist_filename'] = destination_filename
                result_info['final_sample_rate'] = original_sample_rate
                playlist_info.append(result_info)
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"    ❌ Error procesando archivo: {str(e)}")
            failed_count += 1
        
        print("-" * 100)
    
    # Crear archivos de información
    create_playlist_info_file(playlist_folder, description, playlist_info, search_type)
    create_m3u_playlist(playlist_folder, playlist_name, playlist_info)
    
    # Resumen final
    print_final_summary(playlist_folder, copied_count, failed_count, description, 
                       original_sample_rate, playlist_info, search_type)

def search_unique_songs_from_chunks(client, collection_name, query_embedding, target_unique_songs):
    """
    Busca en la base de datos de chunks hasta obtener el número objetivo de canciones únicas.
    """
    unique_songs = {}  # track_id -> mejor resultado para esa canción
    search_limit = 20  # Empezar buscando 20 chunks
    max_search_limit = 200  # Límite máximo para evitar búsquedas infinitas
    last_search_results = []  # Para guardar los últimos resultados para el mensaje final
    
    print(f"🔍 Buscando chunks para obtener {target_unique_songs} canciones únicas...")
    
    while len(unique_songs) < target_unique_songs and search_limit <= max_search_limit:
        print(f"   Buscando {search_limit} chunks (canciones únicas encontradas: {len(unique_songs)})")
        
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=search_limit
        )
        
        last_search_results = search_results  # Guardar para el mensaje final
        
        # Procesar resultados y mantener solo el mejor chunk por canción
        for result in search_results:
            payload = result.payload
            track_id = payload.get('track_id')
            
            if track_id is None:
                continue
                
            # Si es la primera vez que vemos esta canción, o si este chunk tiene mejor score
            if (track_id not in unique_songs or 
                result.score > unique_songs[track_id]['similarity']):
                
                chunk_info = f"{payload['chunk_index'] + 1}/{payload['total_chunks']} ({payload['start_time']:.1f}s-{payload['end_time']:.1f}s)"
                duration_info = f"{payload['chunk_duration']:.1f}s (total: {payload['total_duration']:.1f}s)"
                
                unique_songs[track_id] = {
                    'display_name': f"{payload['filename']} - Chunk {payload['chunk_index'] + 1}",
                    'filename': payload['filename'],
                    'track_id': track_id,
                    'similarity': result.score,
                    'file_path': payload['file_path'],
                    'duration_info': duration_info,
                    'chunk_info': chunk_info,
                    'chunk_index': payload['chunk_index'],
                    'start_time': payload['start_time'],
                    'end_time': payload['end_time'],
                    'chunk_duration': payload['chunk_duration'],
                    'total_duration': payload['total_duration'],
                    'embedding_type': 'chunk'
                }
        
        # Si no hemos alcanzado el objetivo, buscar más chunks
        if len(unique_songs) < target_unique_songs:
            search_limit += 30
        else:
            break
    
    # Convertir a lista ordenada por similitud
    sorted_results = sorted(unique_songs.values(), key=lambda x: x['similarity'], reverse=True)
    
    # Tomar solo el número objetivo
    final_results = sorted_results[:target_unique_songs]
    
    print(f"✅ Encontradas {len(final_results)} canciones únicas de {len(last_search_results)} chunks analizados")
    
    return final_results

def process_song_results(search_results):
    """
    Procesa resultados de búsqueda en canciones completas.
    """
    processed_results = []
    
    for result in search_results:
        payload = result.payload
        track_id = payload.get('track_id', 'N/A')
        
        processed_result = {
            'display_name': f"{payload['filename']} (Canción completa)",
            'filename': payload['filename'],
            'track_id': track_id,
            'similarity': result.score,
            'file_path': payload['file_path'],
            'duration_info': f"{payload['total_duration']:.1f}s ({payload['total_chunks']} chunks)",
            'chunk_info': f"Embedding promedio de {payload['total_chunks']} chunks",
            'chunk_index': None,
            'start_time': 0,
            'end_time': payload['total_duration'],
            'chunk_duration': payload['total_duration'],
            'total_duration': payload['total_duration'],
            'embedding_type': 'song_average'
        }
        
        processed_results.append(processed_result)
    
    return processed_results

def process_weighted_results(search_results):
    """
    Procesa resultados de búsqueda en embeddings ponderados por energía.
    """
    processed_results = []
    
    for result in search_results:
        payload = result.payload
        track_id = payload.get('track_id', 'N/A')
        
        # Información específica del pooling ponderado
        method_info = "Promedio ponderado por energía"
        if payload.get('method') == 'uniform_weights':
            method_info = "Promedio ponderado (pesos uniformes)"
        
        max_energy_chunk = payload.get('max_energy_chunk', 0)
        energy_variance = payload.get('energy_variance', 0.0)
        
        processed_result = {
            'display_name': f"{payload['filename']} (Promedio ponderado)",
            'filename': payload['filename'],
            'track_id': track_id,
            'similarity': result.score,
            'file_path': payload['file_path'],
            'duration_info': f"{payload['total_duration']:.1f}s ({payload['total_chunks']} chunks)",
            'chunk_info': f"{method_info} - Chunk más energético: {max_energy_chunk} - Varianza: {energy_variance:.6f}",
            'chunk_index': max_energy_chunk,  # Para compatibilidad
            'start_time': 0,
            'end_time': payload['total_duration'],
            'chunk_duration': payload['total_duration'],
            'total_duration': payload['total_duration'],
            'embedding_type': 'weighted_pooling'
        }
        
        processed_results.append(processed_result)
    
    return processed_results

def process_representative_results(search_results):
    """
    Procesa resultados de búsqueda en embeddings de chunk representativo.
    """
    processed_results = []
    
    for result in search_results:
        payload = result.payload
        track_id = payload.get('track_id', 'N/A')
        
        # Información específica del chunk representativo
        rep_chunk_idx = payload.get('representative_chunk_index', 0)
        distance_to_centroid = payload.get('distance_to_centroid', 0.0)
        rep_start_time = payload.get('representative_start_time', 0.0)
        rep_end_time = payload.get('representative_end_time', 30.0)
        centroid_method = payload.get('centroid_method', 'unknown')
        
        processed_result = {
            'display_name': f"{payload['filename']} (Chunk representativo)",
            'filename': payload['filename'],
            'track_id': track_id,
            'similarity': result.score,
            'file_path': payload['file_path'],
            'duration_info': f"{payload['total_duration']:.1f}s ({payload['total_chunks']} chunks)",
            'chunk_info': f"Chunk {rep_chunk_idx} ({rep_start_time:.1f}s-{rep_end_time:.1f}s) - Distancia: {distance_to_centroid:.6f} - Método: {centroid_method}",
            'chunk_index': rep_chunk_idx,
            'start_time': rep_start_time,
            'end_time': rep_end_time,
            'chunk_duration': rep_end_time - rep_start_time,
            'total_duration': payload['total_duration'],
            'embedding_type': 'representative_chunk'
        }
        
        processed_results.append(processed_result)
    
    return processed_results

def process_audio_file(result_info, destination_path, search_type, original_sample_rate):
    """
    Procesa un archivo de audio según el tipo de búsqueda.
    Siempre guarda la canción COMPLETA, independientemente del tipo de embedding usado.
    """
    try:
        # Mensajes específicos según el tipo
        type_messages = {
            "chunks": f"encontrada por chunk {result_info['chunk_index'] + 1}",
            "songs": "promedio simple de todos los chunks",
            "weighted": f"promedio ponderado (chunk más energético: {result_info['chunk_index']})",
            "representative": f"chunk representativo {result_info['chunk_index']} ({result_info['start_time']:.1f}s-{result_info['end_time']:.1f}s)"
        }
        
        message = type_messages.get(search_type, "embedding avanzado")
        print(f"    🔄 Procesando: {result_info['filename']} ({message})")
        print(f"    🎵 Guardando canción COMPLETA en playlist")
        
        # Cargar audio original COMPLETO (siempre la canción entera)
        print(f"    📥 Cargando audio completo...")
        audio_data, current_sr = librosa.load(result_info['file_path'], sr=None, mono=True)
        
        total_duration = len(audio_data) / current_sr
        print(f"    🎵 Duración total de la canción: {total_duration:.1f}s")
        
        # Convertir sample rate si es necesario
        if current_sr != original_sample_rate:
            print(f"    🔄 Convirtiendo de {current_sr} Hz a {original_sample_rate} Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=current_sr, target_sr=original_sample_rate)
        
        # Guardar archivo completo
        print(f"    💾 Guardando canción completa...")
        sf.write(destination_path, audio_data, original_sample_rate, format='MP3')
        
        # Verificar guardado
        if os.path.exists(destination_path):
            file_size = os.path.getsize(destination_path)
            duration_converted = len(audio_data) / original_sample_rate
            print(f"    ✅ Guardado exitosamente ({file_size} bytes, {duration_converted:.2f}s @ {original_sample_rate}Hz)")
            
            result_info['converted_duration'] = duration_converted
            result_info['saved_full_song'] = True
            return True
        else:
            print(f"    ❌ Error: El archivo no se guardó correctamente")
            return False
                
    except Exception as e:
        print(f"    ❌ Error procesando archivo: {str(e)}")
        return False

def create_playlist_info_file(playlist_folder, description, playlist_info, search_type):
    """
    Crea un archivo de texto con información detallada de la playlist.
    Actualizado para incluir los 4 tipos de embedding.
    """
    info_file_path = os.path.join(playlist_folder, "playlist_info.txt")
    
    # Descripciones de los tipos de embedding
    embedding_descriptions = {
        "chunks": "Chunks de 30s (canciones completas guardadas)",
        "songs": "Promedio simple por canción",
        "weighted": "Promedio ponderado por energía de audio", 
        "representative": "Chunk más representativo por canción"
    }
    
    try:
        with open(info_file_path, 'w', encoding='utf-8') as f:
            f.write("🎵 INFORMACIÓN DE LA PLAYLIST\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"📝 Descripción: {description}\n")
            f.write(f"🔍 Tipo de búsqueda: {embedding_descriptions.get(search_type, search_type)}\n")
            f.write(f"📅 Creada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🎵 Total de canciones: {len(playlist_info)}\n")
            f.write(f"⏱️  Duración total: {sum(song['converted_duration'] for song in playlist_info):.2f} segundos\n")
            f.write(f"🎚️  Sample rate: 48000 Hz (nativo CLAP)\n\n")
            
            if search_type == "chunks":
                f.write(f"📍 Nota: Se usaron chunks para encontrar las mejores canciones, pero se guardaron completas\n\n")
            elif search_type == "weighted":
                f.write(f"⚖️  Nota: Búsqueda basada en promedio ponderado por energía de audio\n\n")
            elif search_type == "representative":
                f.write(f"🎯 Nota: Búsqueda basada en el chunk más representativo de cada canción\n\n")
            
            f.write(f"🎵 ARCHIVOS MP3 EN LA PLAYLIST:\n")
            f.write("-" * 50 + "\n")
            
            for i, song in enumerate(playlist_info, 1):
                f.write(f"{i:2d}. {song['playlist_filename']}\n")
                f.write(f"    Archivo original: {song['filename']}\n")
                f.write(f"    Track ID: {song['track_id']}\n")
                f.write(f"    Similitud: {song['similarity']:.4f}\n")
                f.write(f"    Tipo embedding: {song['embedding_type']}\n")
                f.write(f"    Información: {song['chunk_info']}\n")
                f.write(f"    Duración guardada: {song['converted_duration']:.2f}s\n")
                f.write(f"    Sample rate final: {song['final_sample_rate']} Hz\n")
                f.write(f"    Ruta original: {song['file_path']}\n\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Generado por el sistema de recomendación musical basado en CLAP\n")
            f.write(f"Búsqueda realizada en: {embedding_descriptions.get(search_type, search_type)}\n")
            f.write("Archivos guardados a 48000 Hz (sample rate nativo)\n")
            
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo crear el archivo de información: {str(e)}")

def create_m3u_playlist(playlist_folder, playlist_name, playlist_info):
    """
    Crea un archivo M3U para la playlist que puede ser usado por reproductores de música.
    """
    m3u_file_path = os.path.join(playlist_folder, f"{playlist_name}.m3u")
    
    try:
        with open(m3u_file_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            f.write(f"# Playlist: {playlist_name}\n")
            f.write(f"# Creada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Sample Rate: 48000 Hz\n\n")
            
            for song in playlist_info:
                duration_seconds = int(song['converted_duration'])
                f.write(f"#EXTINF:{duration_seconds},{song['display_name']}\n")
                f.write(f"{song['playlist_filename']}\n\n")
            
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo crear el archivo M3U: {str(e)}")

def print_final_summary(playlist_folder, copied_count, failed_count, description, 
                       original_sample_rate, playlist_info, search_type):
    """
    Imprime el resumen final de la playlist creada.
    Actualizado para los 4 tipos de embedding.
    """
    print(f"\n🎉 ¡Playlist creada exitosamente!")
    print(f"📁 Carpeta: {playlist_folder}")
    
    # Descripciones específicas
    type_descriptions = {
        "chunks": "Encontradas por chunks de 30s, guardadas como canciones completas",
        "songs": "Promedio simple de todos los chunks por canción",
        "weighted": "Promedio ponderado por energía de audio",
        "representative": "Chunk más representativo de cada canción"
    }
    
    print(f"🔍 Método: {type_descriptions.get(search_type, search_type)}")
    print(f"✅ Archivos MP3 procesados y guardados: {copied_count}")
    print(f"❌ Errores: {failed_count}")
    print(f"📝 Descripción: '{description}'")
    print(f"🎚️  Sample rate final: {original_sample_rate} Hz (nativo CLAP)")
    
    if copied_count > 0:
        total_duration = sum(song['converted_duration'] for song in playlist_info)
        print(f"\n🎵 Tu playlist está lista en: {playlist_folder}")
        print(f"⏱️  Duración total: {total_duration:.2f} segundos ({total_duration/60:.1f} minutos)")
        print("📄 Archivos incluidos:")
        print("   - Archivos MP3 de canciones COMPLETAS (48000 Hz)")
        print("   - playlist_info.txt (información detallada)")
        print("   - playlist.m3u (archivo de playlist para reproductores)")
        
        # Listar archivos MP3 procesados con info específica del tipo
        print(f"\n🎵 Archivos MP3 en la playlist:")
        for song in playlist_info:
            duration_label = f"({song['converted_duration']:.1f}s)"
            embedding_type = song.get('embedding_type', 'unknown')
            
            type_labels = {
                'chunk': f"[Completa - encontrada por chunk {song.get('chunk_index', 0) + 1}]",
                'song_average': "[Promedio simple]",
                'weighted_pooling': f"[Promedio ponderado - chunk energético: {song.get('chunk_index', 0)}]",
                'representative_chunk': f"[Chunk representativo {song.get('chunk_index', 0)}]"
            }
            
            type_label = type_labels.get(embedding_type, f"[{embedding_type}]")
            print(f"   {song['playlist_filename']} {duration_label} {type_label}")

if __name__ == "__main__":
    print("🎵 GENERADOR DE PLAYLIST BASADO EN TEXTO")
    print("=" * 50)
    print("Describe tu playlist y elige el tipo de búsqueda musical")
    print()
    
    # Ir directo a crear playlist basada en texto
    create_text_based_playlist()