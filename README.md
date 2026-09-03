# Laboratorio 6 — Análisis de redes sociales (YouTube)

CC3084 – Data Science, Semestre II 2026, Universidad del Valle de Guatemala.

Análisis de participación, redes autor–video y contenido a partir de
`youtube_videos.csv` (293 videos) y `youtube_comments.csv` (406 comentarios).

## Estructura del repositorio

```
data/
  raw/            youtube_videos.csv, youtube_comments.csv (datos originales, sin modificar)
  processed/      salidas de la limpieza (videos_clean.csv, comments_clean.csv, merge)
src/
  config.py                  rutas y constantes compartidas
  text_utils.py               limpieza de texto, parseo de conteos y listas
  setup_nltk.py                descarga de stopwords en español
  p1_load_integrate.py         Ejercicio 1: carga e integración
  p2_quality_cleaning.py       Ejercicio 2: calidad, limpieza y preprocesamiento
  p3_eda.py                    Ejercicio 3: análisis exploratorio
  p4_bipartite_network.py      Ejercicio 4: red bipartita autor-video
  run_all.py                   ejecuta 1 a 4 en orden
outputs/
  tables/         tablas generadas (prefijo 01_, 02_, 03_; red en tables/network/)
  figures/        figuras generadas (prefijo 03_, 04_)
report/
  avance_informe.md            informe de avance (ejercicios 1 a 4)
requirements.txt
```

## Cómo ejecutar

Requiere Python 3.11+ (probado con 3.13).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python src/setup_nltk.py
```

Ejecutar todo el pipeline (ejercicios 1 a 4):

```bash
cd src
python run_all.py
```

O cada ejercicio por separado (deben correrse en orden, cada uno depende de
las salidas del anterior):

```bash
cd src
python p1_load_integrate.py
python p2_quality_cleaning.py
python p3_eda.py
python p4_bipartite_network.py
```

Las tablas quedan en `outputs/tables/` y `data/processed/`, y las figuras en
`outputs/figures/`. La tabla de nodos/aristas de la red bipartita queda en
`outputs/tables/network/` (CSV y un `.graphml` para abrir en Gephi si se desea).

## Dependencias principales

pandas, numpy, matplotlib, networkx, nltk (stopwords español), wordcloud,
emoji, unidecode, python-louvain (para la detección de comunidades de la
segunda entrega). Ver `requirements.txt` para versiones exactas.

## Nota metodológica importante

El conjunto `youtube_comments.csv` solo contiene comentarios de **19 de los
293 videos** (6.5%). La red, el análisis de concentración y las conclusiones
de este avance se refieren a esa submuestra de 19 videos y sus autores, no a
la totalidad del conjunto de videos. Esta limitación se documenta en detalle
en `report/avance_informe.md`.

## Enlaces

- Repositorio: https://github.com/anthonylouschwank/Lab6-DS
- Espacio colaborativo del grupo: _pendiente de agregar_
