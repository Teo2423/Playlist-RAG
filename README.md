# RagMusical 🎵

Un sistema avanzado de recomendación musical basado en **Retrieval-Augmented Generation (RAG)** que utiliza embeddings de audio multimodales para crear playlists personalizadas a partir de descripciones en lenguaje natural.

## 🎓 Contexto Académico

Este proyecto fue desarrollado en el marco de la materia **"Procesamiento de Señales, Audio y Habla"** de la **Universidad de Buenos Aires (UBA)**, bajo la supervisión del **Profesor Pablo Riera**.

> 🤖 **Nota**: Este proyecto fue desarrollado con la asistencia de **Claude-4-Sonnet** de Anthropic.

## 🎯 ¿Qué hace este proyecto?

RagMusical permite:
- **Crear playlists** a partir de descripciones textuales como "música melancólica para estudiar" o "rock energético de los 80s"
- **Comparar diferentes estrategias** de embedding de audio (chunks, promedios, representativos, canciones completas)
- **Evaluar la calidad** de las playlists con métricas de audio y semánticas
- **Analizar similitudes** entre canciones y géneros musicales

## 🗂️ Estructura del Proyecto

```
RagMusical/
├── Songs/              # 🎵 Carpeta con archivos de audio (.opus)
│   ├── 0/, 1/, 2/...  # Subcarpetas organizadas por categorías
├── qdrant_storage/     # 🗄️ Base de datos vectorial local
├── valid.tsv          # 📊 Metadatos de las canciones (dataset Jamendo)
├── indexer.py         # 🔧 Indexador principal - crea embeddings
├── playlist_creator.py # 🎼 Creador de playlists interactivo
├── playlist_benchmark.py # 📊 Benchmark de calidad de playlists
├── semantic_metrics.py # 🧠 Métricas semánticas avanzadas
└── [otros archivos de análisis]
```

## 📊 Dataset

Este proyecto utiliza el **dataset de validación de Jamendo** de Hugging Face:
- **Fuente**: [Jamendo Dataset](https://huggingface.co/datasets/jretzer/jamendo_dataset)
- **Split usado**: `validation` (~5671 canciones)
- **Formato**: Archivos `.opus` con metadatos en `valid.tsv`

### 🎵 Configuración de Canciones

**IMPORTANTE**: Las canciones deben colocarse en la carpeta `Songs/` organizadas en subcarpetas numeradas:

```
Songs/
├── 0/
│   ├── 1031455.opus
│   ├── 1043411.opus
│   └── ...
├── 1/
│   ├── 1022253.opus
│   └── ...
└── ...
```

> ⚠️ **Nota**: La carpeta `Songs/` debe contener todas las canciones que se deseen analizar. En este trabajo se utilizó el split de validación del dataset Jamendo de Hugging Face.

## 🚀 Instalación y Configuración

### 1. Requisitos del Sistema
```bash
# Instalar dependencias de audio (Ubuntu/Debian)
sudo apt-get install ffmpeg libsndfile1

# Para otros sistemas, consultar documentación de librosa
```

### 2. Dependencias de Python
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `torch` - Deep learning
- `transformers` - Modelos CLAP
- `librosa` - Procesamiento de audio
- `qdrant-client` - Base de datos vectorial
- `pandas`, `numpy` - Análisis de datos
- `openai` - API para generación de prompts
- `tqdm` - Barras de progreso

### 3. Variables de Entorno
Crear archivo `.env`:
```bash
OPENAI_API_KEY=tu_api_key_aqui
```

### 4. Preparar Dataset
1. Descargar canciones del dataset Jamendo validation split
2. Colocar archivos `.opus` en `Songs/` organizados por las subcarpetas (archivos .tar de huggingface)
3. Asegurar que `valid.tsv` contiene los metadatos correspondientes

## 🔧 Uso del Sistema

### 1. Indexar Canciones (Obligatorio)
```bash
python indexer.py
```
**¿Qué hace?**
- Procesa todas las canciones en `Songs/`
- Crea 5 tipos diferentes de embeddings
- Guarda vectores en base de datos Qdrant local
- ⏱️ **Tiempo**: ~2-4 horas para 5671 canciones en una GPU Nvidia gtx 1650

### 2. Crear Playlists Interactivas
```bash
python playlist_creator.py
```
**Funciones:**
- Crear playlists desde texto: `"Describe tu playlist ideal"`
- Comparar estrategias de embedding
- Exportar playlists en formatos M3U/JSON
- Interfaz interactiva por consola

### 3. Benchmark de Calidad
```bash
python playlist_benchmark.py
```
**Evalúa:**
- 200 prompts diversos generados automáticamente
- 5 estrategias de embedding diferentes
- Métricas de audio (BPM, tonalidad, timbre)
- Métricas semánticas (diversidad, cobertura)
- Genera reportes estadísticos completos

## 📁 Descripción Detallada de Archivos

### 🔧 Archivos Principales

| Archivo | Descripción | Uso |
|---------|-------------|-----|
| `indexer.py` | **Indexador principal** - Procesa audio y crea embeddings | `python indexer.py` |
| `playlist_creator.py` | **Creador interactivo** - Genera playlists desde texto | `python playlist_creator.py` |
| `playlist_benchmark.py` | **Benchmark completo** - Evalúa calidad de estrategias | `python playlist_benchmark.py` |
| `semantic_metrics.py` | **Métricas semánticas** - Diversidad y coherencia | Importado por otros |

### 📊 Archivos de Análisis

| Archivo | Descripción |
|---------|-------------|
| `song_similarity_matrix_analyzer.py` | Analiza matrices de similitud entre canciones |
| `song_genre_pca_visualizer.py` | Visualiza géneros musicales con PCA |
| `tsv_analyzer.py` | Analiza metadatos del dataset |
| `debug_audio_embeddings.py` | Debug de embeddings de audio |
| `debug_text_embeddings.py` | Debug de embeddings de texto |
| `inspect_qdrant_collections.py` | Inspecciona base de datos Qdrant |

### 🔍 Archivos de Diagnóstico

| Archivo | Descripción |
|---------|-------------|
| `debug_audio_embeddings.py` | Verificación de embeddings de audio |
| `debug_text_embeddings.py` | Verificación de embeddings de texto |
| `inspect_qdrant_collections.py` | Inspección de la base de datos |

## 🎼 Estrategias de Embedding

El sistema implementa 5 estrategias diferentes para generar embeddings musicales:

| Estrategia | Descripción | Ventajas | Uso Recomendado |
|------------|-------------|----------|-----------------|
| **Chunks** | Mejores chunks de 30s por canción | Alta precisión semántica | Búsquedas específicas |
| **Simple Avg** | Promedio simple de todos los chunks | Representación general balanceada | Uso general |
| **Weighted Avg** | Promedio ponderado por similitud | Balance precisión-generalización | Búsquedas complejas |
| **Representative** | Chunk más representativo por canción | Eficiencia computacional | Sistemas con recursos limitados |
| **Full Songs** | Canción completa (no chunks) | Contexto musical completo | Análisis de estructura musical |

## 📊 Métricas de Evaluación

### 🎵 Métricas de Audio
- **ILS (Intra-List Similarity)**: Similitud promedio entre canciones de la playlist
- **Key Compactness**: Coherencia tonal entre canciones
- **BPM Dispersion**: Variabilidad normalizada del tempo
- **Spectral Centroid Drift**: Diversidad tímbrica de la playlist

### 🧠 Métricas Semánticas
- **Prompt Coverage**: Qué tan bien la playlist coincide con la descripción textual
- **Artist Diversity**: Diversidad de artistas en la playlist
- **Album Diversity**: Diversidad de álbumes en la playlist
- **Genre Entropy**: Entropía de diversidad de géneros musicales
- **Mood Entropy**: Entropía de diversidad de estados de ánimo
- **Instrument Entropy**: Entropía de diversidad instrumental

## 🔍 Ejemplos de Uso

### Crear Playlist Simple
```python
# En playlist_creator.py
query = "Música electrónica relajante para trabajar"
# Genera playlist de 10 canciones automáticamente
```

### Ejemplos de Prompts Exitosos
- `"Jazz melancólico con piano para tardes lluviosas"`
- `"Rock alternativo de los 90s con guitarras distorsionadas"`
- `"Música electrónica ambient con texturas atmosféricas"`
- `"Folk acústico con guitarra fingerpicking para meditación"`
- `"Hip-hop instrumental con samples de jazz para estudiar"`

### Comparar Estrategias
El benchmark automáticamente prueba múltiples estrategias con prompts diversos, generando análisis comparativos detallados.

## 📈 Resultados y Análisis

Los benchmarks generan automáticamente:
- **Reportes estadísticos** en formato CSV y TXT
- **Matrices de similitud** entre canciones
- **Análisis comparativo** de las 5 estrategias de embedding

### Estructura de Resultados
```
benchmark_results_YYYYMMDD_HHMMSS/
├── complete_results.json                    # Resultados completos
├── comprehensive_statistical_analysis.csv   # Análisis estadístico
├── comprehensive_summary_report.txt         # Reporte resumen
├── generated_prompts.txt                    # Prompts utilizados
└── partial_results_*.json                   # Resultados parciales
```

### Otros Análisis Disponibles
```
pca_genre_analysis_YYYYMMDD_HHMMSS/
├── pca_genre_comparison.png                 # Comparación visual de géneros
├── pca_genre_results.json                   # Datos del análisis PCA
└── pca_genre_report.txt                     # Reporte del análisis

similarity_matrix_analysis_YYYYMMDD_HHMMSS/
├── similarity_matrix_correlations.png       # Visualización de correlaciones
├── correlation_distribution.png             # Distribución de correlaciones
└── similarity_matrix_report.txt             # Reporte de similitudes
```



### Ajustar Parámetros de Chunking
```python
# En indexer.py:
chunk_duration = 30  # segundos por chunk
overlap = 15         # solapamiento entre chunks
```

## ⚡ Optimizaciones y Rendimiento

- **GPU**: Detección automática y uso de CUDA cuando está disponible
- **Memoria**: Procesamiento por lotes optimizado para grandes datasets
- **Almacenamiento**: Base de datos vectorial Qdrant local de alto rendimiento
- **Paralelización**: Múltiples workers para indexado acelerado
- **Cache**: Reutilización de embeddings calculados

### Configuración de Rendimiento
```python
# Ajustar en indexer.py según hardware:
batch_size = 16      # Reducir si hay problemas de memoria
num_workers = 4      # Ajustar según CPU cores
device = "cuda"      # O "cpu" si no hay GPU
```



## 🎓 Información Académica

### Contexto del Curso
- **Universidad**: Universidad de Buenos Aires (UBA)
- **Materia**: Procesamiento de Señales, Audio y Habla
- **Profesor**: Pablo Riera
- **Enfoque**: Aplicación práctica de técnicas de procesamiento de señales de audio

### Conceptos Aplicados
- **Procesamiento de señales digitales**
- **Análisis espectral de audio**
- **Embeddings multimodales**
- **Recuperación de información musical**
- **Métricas de calidad perceptual**

## 📚 Referencias y Recursos

### Modelos y Datasets
- **CLAP**: [LAION CLAP Model](https://huggingface.co/laion/larger_clap_general)
- **Jamendo**: [Dataset en Hugging Face](https://huggingface.co/datasets/jretzer/jamendo_dataset)
- **Qdrant**: [Base de datos vectorial](https://qdrant.tech/)

### Papers Relacionados
- Contrastive Language-Audio Pre-training (CLAP)
- Music Information Retrieval techniques
- Audio embedding strategies for recommendation systems
