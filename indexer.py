import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import ClapModel, ClapProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import time
import gc
from scipy.spatial.distance import cosine

def extract_track_id(filename):
    """
    Extrae el track ID del nombre del archivo.
    Ejemplo: '1086592.opus' -> 1086592
    """
    try:
        # Obtener solo el nombre del archivo sin la extensión
        name_without_ext = Path(filename).stem
        # Convertir a entero
        track_id = int(name_without_ext)
        return track_id
    except ValueError:
        # Si no se puede convertir a entero, devolver None
        return None

def chunk_audio(audio_data, sample_rate, chunk_duration=30):
    """
    Divide el audio en chunks de duración específica.
    
    Args:
        audio_data: Array de audio
        sample_rate: Sample rate del audio
        chunk_duration: Duración de cada chunk en segundos (default: 30)
    
    Returns:
        Lista de chunks de audio
    """
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        # Solo agregar chunks que tengan al menos 5 segundos de duración
        if len(chunk) >= int(5 * sample_rate):
            # Si el chunk es más corto que la duración objetivo, rellenamos con ceros
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            chunks.append(chunk)
    
    return chunks

def calculate_chunk_energy_weights(audio_data, chunk_audio_segments, target_sample_rate):
    """
    Calcula pesos de energía para chunks basándose en el audio completo.
    OPTIMIZADO: No recarga el archivo, usa el audio ya cargado.
    """
    try:
        energies = []
        
        for chunk_audio in chunk_audio_segments:
            if len(chunk_audio) > 0:
                # Calcular energía RMS del chunk
                energy = np.sqrt(np.mean(chunk_audio**2))
                energies.append(float(energy))
            else:
                energies.append(0.01)  # Energía mínima para chunks vacíos
        
        # Manejar caso de energías todas cero
        if all(e == 0 for e in energies):
            energies = [1.0] * len(energies)
        
        # Convertir a pesos normalizados
        total_energy = sum(energies)
        if total_energy > 0:
            weights = [e / total_energy for e in energies]
        else:
            weights = [1.0 / len(energies)] * len(energies)
        
        return weights, energies
        
    except Exception as e:
        print(f"      ⚠️ Error calculando pesos de energía: {str(e)}")
        # Usar pesos uniformes como fallback
        num_chunks = len(chunk_audio_segments)
        uniform_weights = [1.0 / num_chunks] * num_chunks
        uniform_energies = [1.0] * num_chunks
        return uniform_weights, uniform_energies

def create_weighted_pooling_embedding(chunk_embeddings, chunk_audio_segments, audio_data, target_sample_rate):
    """
    Crea embedding de pooling ponderado basado en energía de audio.
    """
    try:
        if not chunk_embeddings or len(chunk_embeddings) != len(chunk_audio_segments):
            return None, None
        
        # Calcular pesos de energía
        weights, energies = calculate_chunk_energy_weights(audio_data, chunk_audio_segments, target_sample_rate)
        
        # Validar pesos
        if len(weights) != len(chunk_embeddings):
            return None, None
        
        # Calcular promedio ponderado
        weighted_embedding = np.zeros_like(chunk_embeddings[0])
        for embedding, weight in zip(chunk_embeddings, weights):
            weighted_embedding += embedding * weight
        
        # Normalizar
        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding = weighted_embedding / norm
        else:
            return None, None
        
        # Información adicional
        energy_info = {
            'total_energy': float(sum(energies)),
            'max_energy_chunk': int(np.argmax(energies)),
            'min_energy_chunk': int(np.argmin(energies)),
            'energy_variance': float(np.var(energies)),
            'weights_used': weights[:5]  # Solo los primeros 5 para ahorrar espacio
        }
        
        return weighted_embedding, energy_info
        
    except Exception as e:
        print(f"      ⚠️ Error creando pooling ponderado: {str(e)}")
        return None, None

def create_representative_chunk_embedding(chunk_embeddings):
    """
    Encuentra el chunk más representativo (más cercano al centroide).
    """
    try:
        if not chunk_embeddings:
            return None, None
        
        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0], {
                'representative_chunk_index': 0,
                'distance_to_centroid': 0.0,
                'total_chunks_analyzed': 1,
                'centroid_method': 'single_chunk'
            }
        
        # Calcular centroide
        centroid = np.mean(chunk_embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        else:
            return chunk_embeddings[0], {
                'representative_chunk_index': 0,
                'distance_to_centroid': 0.0,
                'total_chunks_analyzed': len(chunk_embeddings),
                'centroid_method': 'fallback_first'
            }
        
        # Encontrar el más cercano al centroide
        min_distance = float('inf')
        representative_idx = 0
        distances = []
        
        for i, embedding in enumerate(chunk_embeddings):
            distance = cosine(embedding, centroid)
            distances.append(distance)
            
            if distance < min_distance:
                min_distance = distance
                representative_idx = i
        
        representative_embedding = chunk_embeddings[representative_idx]
        
        representative_info = {
            'representative_chunk_index': representative_idx,
            'distance_to_centroid': float(min_distance),
            'total_chunks_analyzed': len(chunk_embeddings),
            'mean_distance_to_centroid': float(np.mean(distances)),
            'std_distance_to_centroid': float(np.std(distances)),
            'centroid_method': 'cosine_distance'
        }
        
        return representative_embedding, representative_info
        
    except Exception as e:
        print(f"      ⚠️ Error creando embedding representativo: {str(e)}")
        return None, None

def get_all_audio_files(base_songs_path):
    """
    Encuentra todos los archivos de audio en todas las carpetas de Songs.
    """
    audio_files = []
    audio_extensions = {'.opus', '.mp3', '.wav', '.flac', '.m4a'}
    
    print("🔍 Explorando carpetas de canciones...")
    
    # Buscar carpetas numeradas (0, 1, 2, ... 21)
    for folder_num in range(22):  # 0 a 21
        folder_path = os.path.join(base_songs_path, str(folder_num))
        
        if os.path.exists(folder_path):
            print(f"   📁 Explorando carpeta {folder_num}...")
            folder_files = []
            
            for file in os.listdir(folder_path):
                if Path(file).suffix.lower() in audio_extensions:
                    full_path = os.path.join(folder_path, file)
                    folder_files.append(full_path)
            
            audio_files.extend(folder_files)
            print(f"      ✅ Encontrados {len(folder_files)} archivos de audio")
        else:
            print(f"      ⚠️  Carpeta {folder_num} no encontrada")
    
    print(f"\n📊 Total de archivos encontrados: {len(audio_files)}")
    return audio_files

def create_complete_music_embeddings_database():
    """
    Crea embeddings para TODAS las canciones en todas las carpetas de Songs.
    Genera TODAS las 5 colecciones en una sola pasada ultra-optimizada:
    1. Chunks individuales (30s cada uno)
    2. Promedios por canción (promedio simple de chunks)
    3. Canciones completas (embedding de toda la canción)
    4. Ponderados por energía (promedio ponderado de chunks)
    5. Representativos (chunk más representativo)
    """
    
    # Configuración de rutas
    base_songs_path = "./Songs"  # Carpeta base
    qdrant_storage_path = "./qdrant_storage"
    chunks_collection_name = "music_chunks_embeddings"
    songs_collection_name = "music_songs_embeddings"
    full_songs_collection_name = "music_full_songs_embeddings"
    weighted_collection_name = "music_weighted_embeddings"
    representative_collection_name = "music_representative_embeddings"
    target_sample_rate = 48000
    chunk_duration = 30  # 30 segundos por chunk
    
    # Configuración optimizada
    db_batch_size = 50    # Batch para inserción en DB
    
    # Verificar que existe la carpeta de datos
    if not os.path.exists(base_songs_path):
        print(f"❌ Error: No se encontró la carpeta {base_songs_path}")
        return
    
    # Crear carpeta para almacenamiento de Qdrant si no existe
    os.makedirs(qdrant_storage_path, exist_ok=True)
    
    # Obtener todos los archivos de audio
    audio_files = get_all_audio_files(base_songs_path)
    
    if not audio_files:
        print("❌ No se encontraron archivos de audio.")
        return
    
    # Inicializar cliente Qdrant
    print(f"\n🚀 Inicializando cliente Qdrant local en {qdrant_storage_path}...")
    client = QdrantClient(path=qdrant_storage_path)
    
    # Cargar modelo y procesador CLAP
    print("Cargando modelo larger-clap-general...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   🎮 GPU detectada: {gpu_name}")
        print(f"   💾 VRAM total: {gpu_memory:.1f} GB")
        
        # Verificar memoria disponible
        torch.cuda.empty_cache()  # Limpiar cache
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   📊 VRAM usada: {memory_allocated:.1f} GB")
        print(f"   📊 VRAM reservada: {memory_reserved:.1f} GB")
    else:
        print(f"   🖥️  Usando CPU (mucho más lento)")
    
    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    
    # Crear las 5 colecciones en Qdrant (eliminar existentes)
    print("🗄️  Creando 5 colecciones en Qdrant...")
    collection_names = [
        chunks_collection_name, 
        songs_collection_name, 
        full_songs_collection_name,
        weighted_collection_name,
        representative_collection_name
    ]
    
    for collection_name in collection_names:
        try:
            client.delete_collection(collection_name)
            print(f" Colección '{collection_name}' eliminada")
        except:
            print(f" Colección '{collection_name}' no existía")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        print(f"  Colección '{collection_name}' creada")
    
    # Estadísticas globales
    total_processed = 0
    total_failed = 0
    total_chunks = 0
    total_duration_processed = 0
    success_counts = {
        'chunks': 0,
        'songs': 0,
        'full_songs': 0,
        'weighted': 0,
        'representative': 0
    }
    
    # Listas para almacenar puntos antes de insertar
    chunk_points = []
    song_points = []
    full_song_points = []
    weighted_points = []
    representative_points = []
    
    # Estimar tiempo
    start_time = time.time()
    
    print(f"\nIniciando procesamiento COMPLETO de {len(audio_files)} archivos...")
    print(f"Configuración:")
    print(f"   - Batch de DB: {db_batch_size} puntos")
    print(f"   - Chunks de: {chunk_duration} segundos")
    print(f"   - Sample rate objetivo: {target_sample_rate} Hz")
    print(f"   - Colecciones: 5 (chunks + songs + full_songs + weighted + representative)")
    
    # Procesar archivos con barra de progreso global
    with tqdm(total=len(audio_files), desc="🎵 Procesando canciones", unit="songs") as pbar:
        
        for audio_file in audio_files:
            try:
                filename = os.path.basename(audio_file)
                track_id = extract_track_id(filename)
                folder_name = os.path.basename(os.path.dirname(audio_file))
                
                # ===== CARGAR AUDIO UNA SOLA VEZ =====
                audio_data, original_sr = librosa.load(audio_file, sr=None, mono=True)
                
                # Resamplear a 48kHz solo si es necesario
                if original_sr != target_sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sample_rate)
                
                total_duration = len(audio_data) / target_sample_rate
                total_duration_processed += total_duration
                
                # ===== 1. PROCESAR CANCIÓN COMPLETA =====
                try:
                    inputs_full = processor(
                        audios=audio_data,
                        sampling_rate=target_sample_rate,
                        return_tensors="pt"
                    ).to(device)
                    
                    with torch.no_grad():
                        full_embed = model.get_audio_features(**inputs_full)
                        full_song_embedding = full_embed.cpu().numpy().flatten()
                    
                    # Normalizar embedding
                    full_song_embedding = full_song_embedding / np.linalg.norm(full_song_embedding)
                    
                    # Crear punto para full_songs
                    full_song_point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=full_song_embedding.tolist(),
                        payload={
                            'file_path': audio_file,
                            'filename': filename,
                            'folder': folder_name,
                            'track_id': track_id,
                            'total_duration': total_duration,
                            'sample_rate': target_sample_rate,
                            'original_sample_rate': original_sr,
                            'embedding_type': 'full_song',
                            'file_size_mb': os.path.getsize(audio_file) / (1024 * 1024)
                        }
                    )
                    full_song_points.append(full_song_point)
                    success_counts['full_songs'] += 1
                    
                except Exception as e:
                    print(f"  Error generando embedding de canción completa: {str(e)}")
                
                # ===== 2. PROCESAR CHUNKS =====
                # Dividir en chunks
                chunk_audio_segments = chunk_audio(audio_data, target_sample_rate, chunk_duration)
                
                # Procesar chunks y almacenar embeddings
                chunk_embeddings = []
                
                for i, chunk in enumerate(chunk_audio_segments):
                    try:
                        # Generar embedding del chunk
                        inputs_chunk = processor(
                            audios=chunk,
                            sampling_rate=target_sample_rate,
                            return_tensors="pt"
                        ).to(device)
                        
                        with torch.no_grad():
                            chunk_embed = model.get_audio_features(**inputs_chunk)
                            embedding = chunk_embed.cpu().numpy().flatten()
                        
                        chunk_embeddings.append(embedding)
                        
                        # Crear punto para la base de datos de chunks
                        start_time_chunk = i * chunk_duration
                        end_time_chunk = min((i + 1) * chunk_duration, total_duration)
                        
                        chunk_point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding.tolist(),
                            payload={
                                'file_path': audio_file,
                                'filename': filename,
                                'folder': folder_name,
                                'track_id': track_id,
                                'chunk_index': i,
                                'start_time': start_time_chunk,
                                'end_time': end_time_chunk,
                                'chunk_duration': end_time_chunk - start_time_chunk,
                                'total_chunks': len(chunk_audio_segments),
                                'total_duration': total_duration,
                                'sample_rate': target_sample_rate,
                                'original_sample_rate': original_sr,
                                'embedding_type': 'chunk'
                            }
                        )
                        chunk_points.append(chunk_point)
                        total_chunks += 1
                        
                    except Exception as e:
                        print(f"      ⚠️ Error procesando chunk {i}: {str(e)}")
                
                if chunk_embeddings:
                    success_counts['chunks'] += len(chunk_embeddings)
                
                # ===== 3. CREAR EMBEDDING PROMEDIO SIMPLE =====
                if chunk_embeddings:
                    try:
                        song_embedding = np.mean(chunk_embeddings, axis=0)
                        song_embedding = song_embedding / np.linalg.norm(song_embedding)
                        
                        song_point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=song_embedding.tolist(),
                            payload={
                                'file_path': audio_file,
                                'filename': filename,
                                'folder': folder_name,
                                'track_id': track_id,
                                'total_chunks': len(chunk_embeddings),
                                'total_duration': total_duration,
                                'sample_rate': target_sample_rate,
                                'original_sample_rate': original_sr,
                                'embedding_type': 'song_average',
                                'chunk_duration': chunk_duration
                            }
                        )
                        song_points.append(song_point)
                        success_counts['songs'] += 1
                        
                    except Exception as e:
                        print(f"      ⚠️ Error creando promedio simple: {str(e)}")
                
                # ===== 4. CREAR EMBEDDING PONDERADO POR ENERGÍA =====
                if chunk_embeddings and len(chunk_embeddings) == len(chunk_audio_segments):
                    try:
                        weighted_embedding, energy_info = create_weighted_pooling_embedding(
                            chunk_embeddings, chunk_audio_segments, audio_data, target_sample_rate
                        )
                        
                        if weighted_embedding is not None:
                            weighted_metadata = {
                                'file_path': audio_file,
                                'filename': filename,
                                'folder': folder_name,
                                'track_id': track_id,
                                'total_chunks': len(chunk_embeddings),
                                'total_duration': total_duration,
                                'sample_rate': target_sample_rate,
                                'original_sample_rate': original_sr,
                                'embedding_type': 'weighted_pooling',
                                'chunk_duration': chunk_duration,
                                **energy_info  # Desempaquetar info de energía
                            }
                            
                            weighted_point = PointStruct(
                                id=str(uuid.uuid4()),
                                vector=weighted_embedding.tolist(),
                                payload=weighted_metadata
                            )
                            weighted_points.append(weighted_point)
                            success_counts['weighted'] += 1
                            
                    except Exception as e:
                        print(f"      ⚠️ Error creando embedding ponderado: {str(e)}")
                
                # ===== 5. CREAR EMBEDDING REPRESENTATIVO =====
                if chunk_embeddings:
                    try:
                        representative_embedding, representative_info = create_representative_chunk_embedding(chunk_embeddings)
                        
                        if representative_embedding is not None:
                            representative_metadata = {
                                'file_path': audio_file,
                                'filename': filename,
                                'folder': folder_name,
                                'track_id': track_id,
                                'total_chunks': len(chunk_embeddings),
                                'total_duration': total_duration,
                                'sample_rate': target_sample_rate,
                                'original_sample_rate': original_sr,
                                'embedding_type': 'representative_chunk',
                                'chunk_duration': chunk_duration,
                                **representative_info  # Desempaquetar info representativa
                            }
                            
                            representative_point = PointStruct(
                                id=str(uuid.uuid4()),
                                vector=representative_embedding.tolist(),
                                payload=representative_metadata
                            )
                            representative_points.append(representative_point)
                            success_counts['representative'] += 1
                            
                    except Exception as e:
                        print(f"      ⚠️ Error creando embedding representativo: {str(e)}")
                
                total_processed += 1
                
                # ===== 6. INSERTAR EN DB SI ES NECESARIO =====
                # Insertar chunks
                if len(chunk_points) >= db_batch_size:
                    client.upsert(collection_name=chunks_collection_name, points=chunk_points)
                    chunk_points = []
                
                # Insertar songs
                if len(song_points) >= db_batch_size:
                    client.upsert(collection_name=songs_collection_name, points=song_points)
                    song_points = []
                
                # Insertar full songs
                if len(full_song_points) >= db_batch_size:
                    client.upsert(collection_name=full_songs_collection_name, points=full_song_points)
                    full_song_points = []
                
                # Insertar weighted
                if len(weighted_points) >= db_batch_size:
                    client.upsert(collection_name=weighted_collection_name, points=weighted_points)
                    weighted_points = []
                
                # Insertar representative
                if len(representative_points) >= db_batch_size:
                    client.upsert(collection_name=representative_collection_name, points=representative_points)
                    representative_points = []
                
                # Limpiar memoria periódicamente
                if total_processed % 10 == 0:
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Monitorear memoria GPU
                if device.type == 'cuda' and total_processed % 25 == 0:
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    
                    if memory_reserved > 3.5:  # Si usa más de 3.5GB de 4GB
                        torch.cuda.empty_cache()
                        gc.collect()
                        tqdm.write(f"🧹 Limpieza de memoria GPU realizada")
                
            except Exception as e:
                total_failed += 1
                tqdm.write(f"❌ Error procesando {audio_file}: {str(e)}")
            
            # Actualizar barra de progreso
            pbar.update(1)
            
            # Actualizar descripción con estadísticas
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            eta = (len(audio_files) - total_processed) / rate if rate > 0 else 0
            
            pbar.set_postfix({
                'Procesadas': total_processed,
                'Fallidas': total_failed,
                'Chunks': total_chunks,
                'Rate': f"{rate:.2f}/s",
                'ETA': f"{eta/60:.1f}min"
            })
    
    # ===== 7. INSERTAR DATOS RESTANTES =====
    print(f"\n💾 Insertando datos restantes en las 5 colecciones...")
    
    # Insertar todos los puntos restantes
    remaining_data = [
        (chunk_points, chunks_collection_name, "chunks"),
        (song_points, songs_collection_name, "promedios"),
        (full_song_points, full_songs_collection_name, "completas"),
        (weighted_points, weighted_collection_name, "ponderadas"),
        (representative_points, representative_collection_name, "representativas")
    ]
    
    for points, collection_name, description in remaining_data:
        if points:
            print(f"   📦 Insertando {len(points)} canciones {description} restantes...")
            client.upsert(collection_name=collection_name, points=points)
    
    # ===== 8. ESTADÍSTICAS FINALES =====
    total_time = time.time() - start_time
    print(f"\n🎉 ¡Procesamiento COMPLETO terminado!")
    print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos ({total_time/3600:.1f} horas)")
    print(f"📊 Estadísticas finales:")
    print(f"   ✅ Canciones procesadas exitosamente: {total_processed}")
    print(f"   ❌ Canciones fallidas: {total_failed}")
    print(f"   🧩 Total de chunks generados: {total_chunks}")
    print(f"   🎵 Total de audio procesado: {total_duration_processed/3600:.1f} horas")
    print(f"   📈 Promedio chunks por canción: {total_chunks/total_processed:.1f}")
    print(f"   📈 Duración promedio por canción: {total_duration_processed/total_processed/60:.1f} minutos")
    print(f"   ⚡ Velocidad promedio: {total_processed/(total_time/60):.2f} canciones/minuto")
    print(f"   🎵 Sample rate procesado: {target_sample_rate}Hz")
    print(f"\n💾 Todas las 5 bases de datos creadas:")
    print(f"   📦 Chunks individuales: '{chunks_collection_name}' ({total_chunks} puntos)")
    print(f"   🎵 Canciones promedio: '{songs_collection_name}' ({success_counts['songs']} puntos)")
    print(f"   🎼 Canciones completas: '{full_songs_collection_name}' ({success_counts['full_songs']} puntos)")
    print(f"   ⚖️  Canciones ponderadas: '{weighted_collection_name}' ({success_counts['weighted']} puntos)")
    print(f"   🎯 Canciones representativas: '{representative_collection_name}' ({success_counts['representative']} puntos)")
    print(f"   📁 Almacenadas en: {qdrant_storage_path}")
    
    # Calcular tasas de éxito
    print(f"\n📈 Tasas de éxito por tipo:")
    for emb_type, count in success_counts.items():
        if emb_type == 'chunks':
            expected = total_chunks
        else:
            expected = total_processed
        
        if expected > 0:
            success_rate = (count / expected) * 100
            print(f"   {emb_type:15}: {success_rate:6.1f}% ({count}/{expected})")
    
    # Limpiar memoria final
    del model
    del processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    print("🚀 ¡Sistema listo para crear playlists con TODAS las opciones de embedding!")
    print("🎯 Ya no necesitas ejecutar advanced_embeddings_generator.py por separado")

def search_similar_music_chunks(query_audio_path, search_type="chunks", top_k=5):
    """
    Busca canciones o chunks similares a un archivo de audio dado.
    ACTUALIZADO: Incluye soporte para weighted y representative
    
    Args:
        query_audio_path: Ruta al archivo de audio de consulta
        search_type: "chunks", "songs", "full_songs", "weighted", "representative"
        top_k: Número de resultados similares a devolver
    """
    collection_mapping = {
        "chunks": "music_chunks_embeddings",
        "songs": "music_songs_embeddings",
        "full_songs": "music_full_songs_embeddings",
        "weighted": "music_weighted_embeddings",
        "representative": "music_representative_embeddings"
    }
    
    qdrant_storage_path = "./qdrant_storage"
    target_sample_rate = 48000
    
    # Seleccionar colección
    if search_type not in collection_mapping:
        print(f"❌ Tipo de búsqueda inválido: {search_type}")
        print(f"   Opciones válidas: {list(collection_mapping.keys())}")
        return
    
    collection_name = collection_mapping[search_type]
    
    # Inicializar cliente y modelo
    print(f"🤖 Cargando modelo y base de datos para búsqueda de {search_type}...")
    client = QdrantClient(path=qdrant_storage_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    
    # Cargar y procesar audio de consulta
    print(f"🎧 Procesando archivo de consulta: {query_audio_path}")
    audio_data, _ = librosa.load(query_audio_path, sr=target_sample_rate, mono=True)
    inputs = processor(
        audios=audio_data,
        sampling_rate=target_sample_rate,
        return_tensors="pt"
    ).to(device)
    
    # Generar embedding
    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)
        query_embedding = audio_embed.cpu().numpy().flatten()
    
    # Normalizar embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Buscar similares
    print(f"🔍 Buscando {search_type} similares...")
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    print(f"\n🎵 {search_type.capitalize()} más similares a {query_audio_path}:")
    print("=" * 100)
    
    for i, result in enumerate(search_results, 1):
        payload = result.payload
        
        print(f"{i}. {payload['filename']} ({payload['embedding_type']})")
        print(f"   📊 Similitud: {result.score:.4f}")
        print(f"   🎵 Track ID: {payload.get('track_id', 'N/A')}")
        print(f"   ⏱️  Duración: {payload['total_duration']:.2f}s ({payload['total_duration']/60:.2f}min)")
        print(f"   📁 Carpeta: {payload['folder']}")
        
        # Información específica por tipo
        if search_type == "chunks":
            print(f"   🧩 Chunk: {payload['chunk_index'] + 1}/{payload['total_chunks']}")
            print(f"   ⏰ Tiempo: {payload['start_time']:.1f}s - {payload['end_time']:.1f}s")
        elif search_type == "weighted":
            print(f"   ⚖️  Energía total: {payload.get('total_energy', 'N/A'):.3f}")
            print(f"   🔥 Chunk más energético: {payload.get('max_energy_chunk', 'N/A')}")
        elif search_type == "representative":
            print(f"   🎯 Chunk representativo: {payload.get('representative_chunk_index', 'N/A')}")
            print(f"   📏 Distancia al centroide: {payload.get('distance_to_centroid', 'N/A'):.4f}")
        
        print("-" * 100)
    
    # Limpiar memoria
    del model
    del processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

if __name__ == "__main__":
    print("🎵 INDEXER COMPLETO DE MÚSICA CON CLAP")
    print("=" * 70)
    print("Este indexer procesará TODAS las canciones y generará:")
    print("  📦 Chunks individuales (30s cada uno)")
    print("  🎵 Promedios por canción (promedio simple de chunks)")
    print("  🎼 Canciones completas (embedding de toda la canción)")
    print("  ⚖️  Embeddings ponderados (promedio ponderado por energía)")
    print("  🎯 Embeddings representativos (chunk más representativo)")
    print("=" * 70)
    print("⚠️  ADVERTENCIA: Esto puede tardar varias horas.")
    print("🚀 ULTRA-OPTIMIZADO: 1 pasada para generar 5 colecciones")
    print("🎯 REEMPLAZA: indexer.py + advanced_embeddings_generator.py")
    print("=" * 70)
    
    # Crear las 5 bases de datos de embeddings en una sola pasada
    create_complete_music_embeddings_database()
    
    # Ejemplos de búsqueda (descomenta para usar)
    # search_similar_music_chunks("./Songs/0/580904.opus", search_type="chunks", top_k=5)
    # search_similar_music_chunks("./Songs/0/580904.opus", search_type="songs", top_k=5)
    # search_similar_music_chunks("./Songs/0/580904.opus", search_type="full_songs", top_k=5)
    # search_similar_music_chunks("./Songs/0/580904.opus", search_type="weighted", top_k=5)
    # search_similar_music_chunks("./Songs/0/580904.opus", search_type="representative", top_k=5)
