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
  informe_final.md / .pdf      informe completo (Ejercicios 1 a 10)
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

### Generar el informe en PDF

`report/informe_final.md` es la fuente del informe completo. Para
regenerar el PDF (requiere [pandoc](https://pandoc.org/) y una
distribución LaTeX con `xelatex`, p. ej. MiKTeX o TeX Live; también puede
instalarse `pandoc` vía `pip install pypandoc_binary` sin depender de un
instalador del sistema):

```bash
python -c "import pypandoc; pypandoc.convert_file('report/informe_final.md', 'pdf', outputfile='report/informe_final.pdf', extra_args=['--pdf-engine=xelatex', '--resource-path=report'])"
```

## Dependencias principales

pandas, numpy, matplotlib, networkx + scipy (topología, centralidad,
PageRank), nltk (stopwords español), wordcloud, emoji, unidecode, y
**pysentimiento** (análisis de sentimiento en español, Ejercicio 9) que a
su vez instala `torch` y `transformers` como dependencias — la primera
ejecución de `p9_sentiment.py` descarga el modelo `robertuito-sentiment-
analysis` desde Hugging Face (~500 MB, requiere conexión a internet la
primera vez; luego queda cacheado localmente). Ver `requirements.txt` para
versiones exactas.

## Notas metodológicas importantes

- El conjunto `youtube_comments.csv` solo contiene comentarios de **19 de
  los 293 videos** (6.5%). Toda la red, el análisis de concentración, las
  proyecciones, la detección de comunidades y las conclusiones se refieren
  a esa submuestra de 19 videos y sus autores, no a la totalidad del
  conjunto de videos. Ver el detalle en `report/informe_final.md`.
- Las proyecciones autor-autor y video-video representan co-participación
  o audiencia compartida, **no** amistad, acuerdo, conversación ni
  similitud temática. `reply_count` nunca se usa para crear aristas
  autor-autor: no identifica a los autores de las respuestas.
- La transitividad y el *clustering* de la proyección autor-autor están
  inflados por construcción (proyectar un video muy comentado crea una
  clique completa entre sus comentaristas); la proyección video-video es
  la lectura estructural más confiable y la que se usa para detectar
  comunidades (Ejercicio 7).
- Louvain (Ejercicio 7) usa pesos, resolución 1 y semilla 42 fija para
  reproducibilidad; no garantiza el óptimo global de modularidad.
- Los puntos de articulación (Ejercicio 8) distinguen puentes **críticos**
  (única conexión entre dos partes de la red) de puentes **redundantes**
  (ya existe otra conexión); no todo autor con intermediación > 0 es un
  punto de articulación real.
- El sentimiento (Ejercicio 9) se calcula sobre `texto_original`, no sobre
  `texto_limpio`: limpiar el texto antes de clasificar borraría señales
  (acentos, orden de palabras, formato del emoji) que el modelo usa.

## Enlaces

- Repositorio: https://github.com/anthonylouschwank/Lab6-DS
- Espacio colaborativo del grupo: _pendiente de agregar_
