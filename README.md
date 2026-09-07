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
  config.py                    rutas y constantes compartidas
  text_utils.py                limpieza de texto, parseo de conteos y listas
  setup_nltk.py                descarga de stopwords en español
  p1_load_integrate.py         Ejercicio 1: carga e integración
  p2_quality_cleaning.py       Ejercicio 2: calidad, limpieza y preprocesamiento
  p3_eda.py                    Ejercicio 3: análisis exploratorio
  p4_bipartite_network.py      Ejercicio 4: red bipartita autor-video
  p5_network_projections.py    Ejercicio 5: proyecciones autor-autor / video-video
  p6_topology_fragmentation.py Ejercicio 6: topología y fragmentación
  p7_communities.py            Ejercicio 7: comunidades (Louvain)
  p8_centrality.py             Ejercicio 8: centralidad y nodos puente
  p9_sentiment.py               Ejercicio 9: análisis de sentimiento
  run_all.py                   ejecuta 1 a 9 en orden
tests/
  test_network_analysis.py     controles con redes pequeñas verificables a mano
outputs/
  tables/         tablas generadas (prefijo 01_ a 09_; red en tables/network/)
  figures/        figuras generadas (prefijo 03_ a 09_)
report/
  main.tex / main.pdf          informe final entregado (Ejercicios 1 a 10)
  avance_informe.md            informe de avance entregado el 3 de septiembre (Ejercicios 1 a 4)
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

Ejecutar todo el pipeline (Ejercicios 1 a 9):

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
python p5_network_projections.py
python p6_topology_fragmentation.py
python p7_communities.py
python p8_centrality.py
python p9_sentiment.py
```

Controles automatizados sobre redes pequeñas con resultado verificable a mano:

```bash
python -m unittest discover -s tests -v
```

Las tablas quedan en `outputs/tables/` y `data/processed/`, y las figuras en
`outputs/figures/`. Las tablas de nodos/aristas de cada red (bipartita,
proyecciones, comunidades) quedan en `outputs/tables/network/` (CSV y
`.graphml` para abrir en Gephi).