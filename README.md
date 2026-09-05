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
  p5_network_projections.py    Ejercicio 5: proyecciones
  p6_topology_fragmentation.py Ejercicio 6: topologia y fragmentacion
  p7_communities.py            Ejercicio 7: comunidades
  run_all.py                   ejecuta 1 a 7 en orden
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

Ejecutar todo el pipeline (ejercicios 1 a 7):

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

## Continuación: ejercicios 5 a 7

`python src/run_all.py` ejecuta ahora las etapas **1 a 7**. Para ejecutar
solamente la continuación sobre las salidas existentes de la etapa 4:

```bash
python src/p5_network_projections.py
python src/p6_topology_fragmentation.py
python src/p7_communities.py
```

- `p5_network_projections.py`: proyecciones completas autor–autor (peso =
  videos distintos compartidos) y video–video (peso = autores distintos
  compartidos), tablas de nodos/aristas, GraphML y comparación estructural.
- `p6_topology_fragmentation.py`: métricas, distribución de grados,
  componentes y asignaciones por nodo, top de grados y tablas de periferia.
- `p7_communities.py`: Louvain ponderado sobre videos, membresías, resumen,
  autores y términos frecuentes de hasta tres comunidades principales.

Las nuevas tablas se guardan en `outputs/tables/network/` y las siete
figuras nuevas en `outputs/figures/`, con prefijos `05_`, `06_` y `07_`.
Las proyecciones tienen nombres `author_projection` y `video_projection`.
Las tablas de nodos conservan los aislados y permiten reconstruir las
redes junto con sus tablas de aristas.

### Criterios de interpretación y reproducción

Las proyecciones representan co-participación o audiencia compartida,
sin demostrar amistad, acuerdo, conversación ni similitud temática.
El aislamiento corresponde únicamente a la muestra observada.

La densidad general es `2E / (N(N-1))`; la densidad bipartita se guarda
por separado como `E / (n_autores * n_videos)`. Las componentes se numeran
por tamaño descendente, desempatando por ID. Se considera pequeña una
componente de hasta 3 nodos, grado bajo un grado de hasta 2 y pocos autores
hasta 2 autores por video. Estos umbrales son descriptivos y quedan
identificados mediante columnas booleanas en las tablas de periferia.

La [transitividad](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html)
cuenta el cierre global de tríadas. El
[clustering medio](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html)
promedia los coeficientes locales, incluyendo los ceros. Ambos se calculan
sin pesos solo en las proyecciones; quedan vacíos, como no aplicables, en
la bipartita. En autores, los videos generan grupos completamente
conectados al proyectar, lo que puede elevar estas métricas por construcción.

[Louvain](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
usa pesos, resolución 1 y semilla 42. La modularidad compara la fuerza de
conexión interna con un modelo nulo basado en grados ponderados. Las
comunidades son agrupaciones estructurales, no grupos sociales confirmados;
los videos aislados permanecen como comunidades unitarias. Se registran
la versión de NetworkX y los parámetros en `07_community_metrics.csv`.

Las tres comunidades para caracterización se eligen por cantidad de
comentarios, desempatando por cantidad de videos e ID de comunidad.
`07_community_members.csv` ordena sus videos por participación;
`07_community_authors.csv` identifica autores y su actividad;
`07_community_terms.csv` reúne palabras, bigramas, hashtags y keywords.
Los términos reutilizan el texto limpio existente: describen frecuencias,
no temas confirmados ni sentimiento. Los autores pueden participar en más
de una comunidad, por lo que sus conteos no deben sumarse como únicos globales.

`share_comments` es una proporción entre 0 y 1; la intensidad se expresa
como comentarios por autor, comentarios por video y peso interno de la
proyección. `07_comment_community_sentiment.csv` conserva los IDs de
comentario, video y autor junto con la comunidad, y deja el sentimiento
vacío con estado `pendiente_ejercicio_9`. Los ejercicios 8 a 10 siguen pendientes.

Controles adicionales sobre redes pequeñas con resultados conocidos:

```bash
python -m unittest discover -s tests -v
```
