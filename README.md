# 🎶 Playlist RAG

Sistema de recomendación musical basado en **Retrieval-Augmented Generation (RAG)** que utiliza embeddings multimodales de audio y texto para generar playlists personalizadas desde descripciones en lenguaje natural.

---

## 🎓 Contexto Académico

Este proyecto fue desarrollado en el marco de la materia **Procesamiento de Señales, Audio y Habla** de la **Universidad de Buenos Aires (UBA)**, bajo la supervisión del Prof. **Pablo Riera**.

> 🛠️ El código fue desarrollado con la asistencia del modelo **Claude-4-Sonnet (Anthropic)**.

---

## 📁 Estructura del Proyecto

```
RagMusical/
├── Songs/                  # Archivos de audio (.opus)
├── valid.tsv              # Metadatos del dataset Jamendo
├── indexer.py             # Crea e indexa embeddings en Qdrant
├── playlist_creator.py    # Genera playlists desde texto
├── playlist_benchmark.py  # Evalúa calidad de las playlists
├── semantic_metrics.py    # Métricas semánticas (diversidad, cobertura)
└── [archivos de análisis y visualización]
```

> Las canciones deben ubicarse en la carpeta `Songs/`, organizadas en subcarpetas numeradas (`0/`, `1/`, etc.). Cada archivo debe tener formato `.opus` y su nombre debe coincidir con el campo `id` del archivo `valid.tsv`. Ejemplo:
>
> ```
> Songs/
> ├── 0/
> │   ├── 1031455.opus
> │   └── 1043411.opus
> ├── 1/
> │   └── 1022253.opus
> └── ...
> ```
Si se descomprimen los archivos .tar del split val de huggingface se obtiene exactamente la configuracion que se uso en los experimentos (https://huggingface.co/datasets/rkstgr/mtg-jamendo/tree/main/data/val)
---

## 🚀 Uso

### 1. Indexar Canciones
Procesa el dataset y genera embeddings.
```bash
python indexer.py
```

### 2. Crear Playlist desde Texto
Interfaz interactiva por consola para generar playlists personalizadas.
```bash
python playlist_creator.py
```

### 3. Evaluar Calidad de Playlists
Ejecuta el benchmark sobre múltiples prompts.
```bash
python playlist_benchmark.py
```

> Asegurarse de que las canciones estén organizadas en `Songs/` y que `valid.tsv` contenga los metadatos del split de validación del dataset **Jamendo**.

---
