# Laboratorio 6 — Análisis de redes sociales en YouTube
## Informe de avance (Ejercicios 1 a 4)

CC3084 – Data Science, Semestre II 2026, Universidad del Valle de Guatemala
Fecha de entrega del avance: jueves 3 de septiembre de 2026

Todo el código que produce las tablas y figuras citadas aquí está en `src/`
(`p1_load_integrate.py` a `p4_bipartite_network.py`, ejecutables con
`python src/run_all.py`). Las tablas quedan en `outputs/tables/` y las
figuras en `outputs/figures/`.

---

## 1. Carga, comprensión e integración de los datos

### 1.1 Carga

Los dos archivos se cargaron con `pandas.read_csv` (script
[`p1_load_integrate.py`](../src/p1_load_integrate.py)). Ambos archivos
contienen campos de texto con saltos de línea internos entre comillas
(por ejemplo, `description`); `pandas` los interpreta correctamente y las
dimensiones resultantes coinciden con el enunciado:

| Archivo | Filas | Columnas |
|---|---|---|
| youtube_videos.csv | 293 | 20 |
| youtube_comments.csv | 406 | 17 |

### 1.2 Unidad de observación, llave primaria y variables relevantes

- **youtube_videos.csv**: la unidad de observación es *un video de YouTube*.
  La llave primaria es `video_id` (293 valores únicos, sin duplicados,
  ver `outputs/tables/01_primary_key_check.csv`). Variables relevantes para
  este laboratorio: `video_id`, `channel_id` (nodo canal), `title`,
  `description`, `keywords`/`query_hits` (temas), `category`, `view_count`
  (popularidad), `publish_date`, `source_query`/`source_group`
  (procedimiento de muestreo).
- **youtube_comments.csv**: la unidad de observación es *un comentario
  principal* publicado en un video. La llave primaria es `comment_id`
  (406 valores únicos, sin duplicados). Variables relevantes:
  `comment_id`, `video_id` (llave foránea hacia videos), `author_channel_id`
  (nodo autor — no `author_name`, que puede repetirse o cambiar), `text`,
  `like_count_text`, `reply_count`.

### 1.3 Relación entre canal, video, autor, comentario, categoría y consulta de búsqueda

- **Canal → Video**: relación 1 a N. Un `channel_id` publica muchos
  `video_id` (97 canales para 293 videos). `channel_id` es el identificador
  estable; `channel_name`/`channel_handle` son solo etiquetas visibles.
- **Video → Comentario**: relación 1 a N. Un `video_id` puede tener muchos
  `comment_id`, pero cada comentario pertenece a exactamente un video.
- **Autor ↔ Video**: relación N a N. Un `author_channel_id` puede comentar
  en varios videos y un video puede recibir comentarios de varios autores.
  Esta relación N a N es exactamente la que se modela como red bipartita en
  el Ejercicio 4.
- **Autor → Comentario**: relación 1 a N (un autor puede dejar más de un
  comentario en el mismo video; ver sección 4).
- **Categoría**: atributo del *video* (`category`, ej. *News & Politics*,
  *People & Blogs*), no del comentario ni del canal directamente — un canal
  puede publicar videos de distintas categorías.
- **Consulta de búsqueda (`source_query`/`source_group`)**: describe el
  *procedimiento de muestreo* con el que se recolectó el video (o sus
  comentarios), no un atributo temático confirmado del contenido. Un mismo
  video puede coincidir con varias consultas (`query_hits`).

En resumen, el grafo conceptual es: **Canal —(publica)→ Video —(recibe)→
Comentario —(escrito por)→ Autor**, con `category` y `source_query` como
atributos de video, y la relación Autor–Video como la única relación N a N
explícita en los datos.

### 1.4 Integración mediante `video_id`

Se integraron ambos conjuntos con un *left join* desde `youtube_comments`
hacia `youtube_videos` sobre `video_id` (todo comentario requiere un video
válido). Resultado (`outputs/tables/01_integration_summary.csv`):

| Métrica | Valor |
|---|---|
| Comentarios totales | 406 |
| Comentarios asociados a un video (match) | 406 (100%) |
| Comentarios huérfanos (sin video) | 0 |
| Videos totales | 293 |
| Videos con ≥1 comentario en la muestra | **19 (6.48%)** |
| Videos sin comentarios en la muestra | 274 (93.52%) |

**Hallazgo central para todo el laboratorio**: los 406 comentarios se
concentran en solo **19 de los 293 videos**. Esto no es un error de
integración (el 100% de los comentarios sí se asocia correctamente a un
video) sino una característica del procedimiento de recolección: los
comentarios solo se extrajeron para una selección de videos, no para todos.
Cualquier análisis de red, concentración o popularidad-participación en
este informe se refiere a esa submuestra de 19 videos y a los 332 autores
que comentaron en ellos, no a los 293 videos del conjunto completo.

---

## 2. Calidad, limpieza y preprocesamiento

Script: [`p2_quality_cleaning.py`](../src/p2_quality_cleaning.py).

### 2.1 Diagnóstico inicial de calidad

**Dimensiones y tipos**: ver arriba (1.1) y
`outputs/tables/02_quality_overview.csv` (tipo de dato, `%` de faltantes y
número de valores únicos por columna).

**Valores faltantes** (columnas con NA > 0):

| Dataset | Columna | Faltantes |
|---|---|---|
| videos | `description` | 26 (8.9%) |
| videos | `description_snippet` | 25 (8.5%) |
| videos | `published_time` | 13 (4.4%) |
| videos | `view_count_text` | 13 (4.4%) |
| comments | `viewer_rating` | 406 (100%) |

**Duplicados**: 0 `video_id` duplicados, 0 `comment_id` duplicados, 0 filas
completas repetidas en ninguno de los dos archivos
(`outputs/tables/02_duplicates.csv`).

**Variables constantes**: `is_pinned` (100% `False`) y `viewer_rating`
(100% vacía) no aportan varianza. `upload_date` coincide con `publish_date`
en el 100% de los videos, y `owner_handle` coincide con `channel_handle` en
el 100% — ambas son redundantes tal como advierte el enunciado.

**Valores atípicos (regla IQR)** (`outputs/tables/02_outliers_iqr.csv`):
`view_count` tiene 49 videos atípicos (16.7%) — es una distribución muy
sesgada (mínimo 2, mediana 1,175, máximo 8,190,449 vistas), típica de
popularidad en redes sociales, por lo que estos "atípicos" son video virales
reales, no errores de captura. `reply_count` tiene 30 comentarios atípicos
(7.4%, máximo 7 respuestas) sobre una base donde el 75% de los comentarios
tiene 0 respuestas.

**Consistencia entre identificadores, nombres y handles**
(`outputs/tables/02_id_name_consistency.csv`): se verificaron 9 chequeos
(cada `channel_id` mapea a un único `channel_name` y `channel_handle`; cada
`author_channel_id` mapea a un único `author_name`; todo `video_id` de
comentarios existe en videos; unicidad de llaves primarias) y **los 9
chequeos pasaron sin excepciones**: no hay canales o autores con nombres
inconsistentes para un mismo ID.

**Verificación adicional de `view_count`**: al convertir `view_count_text`
a número y compararlo con `view_count`, 53 videos (18%) muestran una
pequeña discrepancia (p. ej. `2,390 vistas` vs. `view_count = 2357`). Al
inspeccionar los casos, las diferencias son de unas pocas unidades o
decenas — consistentes con que ambos campos se capturaron en momentos
ligeramente distintos del scraping (las vistas de YouTube cambian en tiempo
real), no con un error de parseo. Se usa `view_count` como fuente
canónica, tal como recomienda el enunciado.

### 2.2 Variables problemáticas o de uso delicado

Tabla completa en `outputs/tables/02_problematic_variables.csv`. Resumen:

| Variable | Problema | Tratamiento |
|---|---|---|
| `viewer_rating` | 100% vacía | Se descarta. |
| `is_pinned` | Constante (`False`) | Se conserva por trazabilidad, se excluye del análisis. |
| `upload_date`, `owner_handle` | Redundantes (idénticas a otra columna) | No se usan; se prefiere `publish_date`/`channel_handle`. |
| `published_time` / `published_text` | Fecha **relativa** ("hace 2 días"), depende del momento de recolección | No se convierte a fecha absoluta; se usa `publish_date` (ISO 8601) solo disponible para videos. El orden temporal exacto de los comentarios queda como limitación. |
| `like_count_text` | 46.6% de valores en blanco (189/406) | Se interpreta blanco como 0 *me gusta* visibles (comportamiento normal de la interfaz de YouTube, que no muestra "0"), con bandera `like_count_was_blank` para auditoría. |
| `view_count_text` | 13 faltantes; formato localizado | Se usa `view_count` como fuente numérica principal. |
| `description_snippet` | Fragmento truncado y redundante | No se usa; se prioriza `description`. |
| `reply_count` | **No identifica autores de las respuestas** (aclaración explícita del enunciado) | Se usa solo como intensidad de reacción al comentario principal; nunca se interpreta como arista autor–autor. |
| `query_hits`, `keywords`, `dataset_sources` | Texto con estructura de lista almacenado como *string* | Se parsean con `ast.literal_eval` a listas reales antes de analizarlos. |
| `description` (1 registro) | Contiene caracteres de reemplazo Unicode (pérdida de codificación irreversible) | Se conserva el registro, pero se documenta que ese texto puntual no es confiable para análisis de contenido. |

### 2.3 Normalización de identificadores y nombres

Se recortan espacios en `channel_id`, `video_id`, `comment_id`,
`author_channel_id` y se fuerzan a tipo texto (evitando que un ID puramente
numérico se interprete como entero y pierda ceros a la izquierda). **En
ningún momento se reemplazan los IDs por sus nombres visibles**: los
nombres (`channel_name`, `author_name`, handles) solo se recortan de
espacios para uso en etiquetas de gráficos, pero toda unión, agrupación y
construcción de red usa exclusivamente `channel_id`, `video_id`,
`comment_id` y `author_channel_id`, como exige el enunciado.

### 2.4 Conversión de variables de conteo almacenadas como texto

Implementado en [`text_utils.parse_count_text`](../src/text_utils.py).
Reglas documentadas:

1. Se elimina el sufijo textual (`vistas`, `views`, `visualizaciones`).
2. Se eliminan comas usadas como separador de miles.
3. Se expanden abreviaturas `K` (×1,000), `M` (×1,000,000) y `mil` (×1,000)
   — no se observaron en este conjunto, pero la función las soporta para
   reproducibilidad futura.
4. Un valor vacío o no parseable se convierte a `NaN` (no se asume 0), con
   la única excepción documentada de `like_count_text`, donde blanco se
   interpreta explícitamente como 0 por el comportamiento conocido de la
   interfaz de YouTube (ver 2.2).

Resultado: `like_count` (entero, 0–405) y `view_count_from_text` (usado
solo para el chequeo de consistencia de 2.1) quedan disponibles en
`data/processed/*_clean.csv`.

### 2.5 y 2.6 `texto_original` y `texto_limpio`

`texto_original` conserva el comentario tal como fue publicado (necesario
para auditoría y para el análisis de sentimiento del Ejercicio 9, que debe
correr sobre texto natural). `texto_limpio` se genera con el siguiente
pipeline documentado en
[`text_utils.clean_text`](../src/text_utils.py):

1. Minúsculas.
2. Eliminación de URLs (`https?://…`, `www…`).
3. Extracción y eliminación de menciones (`@usuario`) y hashtags (`#tema`)
   del cuerpo del texto — **se guardan aparte** en las columnas `mentions`
   y `hashtags` (listas) para no perder esa señal.
4. Los emojis se extraen a la columna `emojis` y además se convierten a su
   descripción en español dentro del texto (p. ej. 😂 → "cara llorando de
   risa") en lugar de eliminarlos sin más, porque en comentarios cortos el
   emoji suele ser la única señal de sentimiento disponible.
5. Eliminación de acentos (normalización con `unidecode`) antes de filtrar
   caracteres.
6. Eliminación de puntuación y dígitos.
7. Colapso de espacios múltiples.
8. Eliminación de *stopwords* en español (lista de NLTK + una lista corta
   adicional de muletillas frecuentes en YouTube: "si", "q", "xq", "porq").

No se aplicó lematización: dado el tamaño de la muestra (406 comentarios,
muchos muy cortos) y que no se contó con un lematizador robusto para
español sin dependencias pesadas adicionales (spaCy), se decidió no
lematizar en este avance para no introducir errores gramaticales
silenciosos; se documentará y reconsiderará para el informe final si el
análisis de tópicos del Ejercicio 9 lo requiere.

### 2.7 Efecto cuantificado de la limpieza

`outputs/tables/02_cleaning_effect.csv`:

| Métrica | Valor |
|---|---|
| Comentarios totales | 406 |
| Textos originales vacíos | 0 |
| Textos limpios vacíos (todo el contenido era stopword/URL/puntuación) | 1 |
| Duplicados exactos en `texto_original` | 2 |
| Duplicados exactos en `texto_limpio` | 5 |
| Comentarios con ≥1 hashtag | 1 |
| Comentarios con ≥1 mención | 5 |
| Comentarios con ≥1 emoji | 61 (15%) |
| Caracteres promedio `texto_original` | 139.2 |
| Caracteres promedio `texto_limpio` | 99.6 |

La limpieza reduce el texto en ~28% de caracteres en promedio y hace
aparecer 3 duplicados adicionales (comentarios distintos que, tras quitar
acentos/puntuación/stopwords, quedan idénticos) — se documenta pero no se
eliminan filas, porque siguen siendo comentarios distintos con
`comment_id` propio.

---

## 3. Análisis exploratorio

Script: [`p3_eda.py`](../src/p3_eda.py). Todas las figuras están en
`outputs/figures/03_*.png` y las tablas en `outputs/tables/03_*.csv`.

### 3.1 Descripción general

| Elemento | Valor |
|---|---|
| Videos totales | 293 |
| Canales únicos (videos) | 97 |
| Comentarios totales | 406 |
| Autores únicos | 332 |
| Videos con ≥1 comentario | 19 |
| Canales con ≥1 video comentado | 8 |
| Categorías de video distintas | 11 |
| Consultas de búsqueda distintas (`source_query`, videos) | 21 |

- **Videos por canal**: mediana 1, media 3.02, máximo 32 (un canal
  concentra 32 de los 293 videos); el 75% de los canales tiene 1–2 videos
  (`outputs/tables/03_videos_per_channel.csv`).
- **Comentarios y autores únicos por video** (solo los 19 videos con
  comentarios, `outputs/tables/03_comments_authors_per_video.csv`): van
  desde 1 comentario/1 autor hasta 161 comentarios/128 autores únicos en el
  video más comentado. En casi todos los videos, el número de autores
  únicos es muy cercano al número de comentarios (poca repetición por
  autor dentro del mismo video, salvo excepciones puntuales, ver 4.2).
- **Visualizaciones** (293 videos): media 60,430, mediana 1,175, máximo
  8,190,449 — distribución extremadamente sesgada (ver histograma
  `03_histograma_views.png`, en escala log10).
- **Respuestas (`reply_count`)**: media 0.13, mediana 0, máximo 7 — el
  engagement de segundo nivel es muy bajo.
- **"Me gusta" (`like_count`)**: media 5.73, mediana 1, máximo 405.
- **Categorías**: `News & Politics` domina con 138/293 videos (47%),
  seguida de `People & Blogs` (66) y `Entertainment` (48)
  (`outputs/figures/03_videos_por_categoria.png`).
- **Consultas de búsqueda**: 21 valores distintos de `source_query`;
  predominan handles y nombres de instituciones de gobierno (Municipalidad
  de Guatemala, Gobierno de Guatemala, Ministerio de Comunicaciones,
  CONRED) junto con consultas temáticas como "guatemala noticias"
  (`outputs/tables/03_source_query_videos.csv`).
- **Hashtags**: prácticamente ausentes en los comentarios (**solo 1** de
  406 contiene un hashtag, `#IneptoBran`). Las *keywords* de los videos
  (etiquetas puestas por el canal) sí son ricas: "guatemala" (89
  videos), "noticias" (28), "municipalidad" (19)
  (`outputs/tables/03_keyword_frequency.csv`).
- **Palabras y bigramas frecuentes** (`texto_limpio`,
  `03_top_palabras.png`, `03_top_bigramas.png`): un hallazgo notable es que
  buena parte de los bigramas más frecuentes son en realidad **descripciones
  de emojis** convertidos a texto ("cara llorando", "llorando risa",
  "manos aplaudiendo", "pulgar hacia arriba", "bandera guatemala"), lo que
  indica que una fracción relevante del "contenido textual" es en realidad
  reacción emotiva breve (risa, aplauso, aprobación, orgullo patrio) más que
  argumento desarrollado. Entre las palabras sueltas con contenido temático
  genuino destacan "guatemala", "pueblo", "país", "dinero", "presidente",
  "trabajo", "diputados" y "corruptos" — vocabulario de crítica
  político-económica.

### 3.2 Concentración de la participación

(`outputs/tables/03_concentration.csv`, `03_comentarios_por_video.png`,
`03_comentarios_por_canal.png`)

| Métrica | Valor |
|---|---|
| Video más comentado (% del total de comentarios) | **39.7%** |
| Top 3 videos | 63.1% |
| Top 5 videos | 75.4% |
| Canal con más comentarios (Quorum) | **63.1%** |
| Top 3 canales | 86.5% |
| Índice de Gini — comentarios por video (19 videos) | 0.660 |
| Índice de Gini — vistas por video (293 videos) | 0.949 |

La participación está fuertemente concentrada: un solo video ("Qué rico
come tu diputado", canal Quorum) reúne el 39.7% de todos los comentarios de
la muestra, y solo 5 de los 19 videos comentados concentran el 75.4%. A
nivel de canal, **Quorum** (un canal de análisis/opinión política, no un
medio noticioso tradicional) es responsable del 63.1% de los comentarios y
de 11 de los 19 videos comentados, aunque solo publicó una fracción menor
de los 293 videos del conjunto completo. El Gini de vistas (0.949) es aún
más alto que el de comentarios (0.660), lo cual es esperable: las vistas
son una medida acumulada de mucho mayor rango (2 a 8.19 millones) que el
conteo de comentarios de una muestra pequeña.

### 3.3 Popularidad vs. participación

(`outputs/tables/03_popularity_vs_participation.csv`,
`03_popularidad_vs_participacion.png`)

Para los 19 videos con comentarios, la correlación de Spearman entre
`view_count` y número de comentarios es **ρ = 0.811**, una asociación
positiva y fuerte. Sin embargo, el gráfico de dispersión (eje de vistas en
escala log) muestra que la relación no es estrictamente monótona: el video
con más vistas de la submuestra (~304,000, "Plan 2032 Ciudad de
Guatemala", un video institucional) recibió solo 25 comentarios, mientras
que el video más comentado (161 comentarios) tiene apenas ~11,800 vistas.
Esto sugiere que el *tipo* de contenido (opinión/humor político vs.
comunicado institucional) importa tanto o más que el alcance bruto para
explicar cuánta gente comenta. **Limitaciones de ambos conteos**: `view_count`
es acumulado desde la publicación hasta el momento de la recolección (no
hay ventana de tiempo comparable entre videos de distinta antigüedad); el
conteo de comentarios es el de comentarios *principales* únicamente (no
incluye respuestas, cuyo total sí se registra en `reply_count` pero sin
identidad de autor); y la asociación se calculó sobre solo 19 de 293
videos — los que efectivamente tienen comentarios en la muestra — por lo
que no debe generalizarse al resto del conjunto.

### 3.4 Visualizaciones generadas

- `03_comentarios_por_video.png` — barras horizontales, comentarios por
  cada uno de los 19 videos con comentarios.
- `03_comentarios_por_canal.png` — barras horizontales, comentarios por
  canal.
- `03_videos_por_categoria.png` — barras horizontales, videos por
  categoría (293 videos).
- `03_histograma_views.png` — histograma de `log10(view_count)`.
- `03_popularidad_vs_participacion.png` — dispersión vistas vs.
  comentarios (19 videos), eje x logarítmico.
- `03_top_palabras.png` / `03_top_bigramas.png` — frecuencias de palabras y
  bigramas de `texto_limpio`.
- `03_wordcloud.png` — nube de palabras de `texto_limpio` (complementaria,
  no sustituye los gráficos de frecuencia anteriores).
- `03_top_keywords_videos.png` — *keywords* más frecuentes entre los 293
  videos.

### 3.5 Preguntas obligatorias

**¿Qué videos y canales concentran la mayor participación observada?**
El video "Qué rico come tu diputado" (canal Quorum) concentra el 39.7% de
los 406 comentarios; junto con otros 4 videos de Quorum y del Gobierno de
Guatemala llegan al 75.4% (top 5). A nivel de canal, Quorum concentra el
63.1% de los comentarios con solo 11 de los 293 videos del conjunto
—es decir, la participación no está distribuida como la producción de
contenido: unos pocos videos de un canal de opinión concentran la
conversación observada.

**¿Existen audiencias compartidas entre videos, canales o temas?**
Sí, aunque de forma limitada: de los 332 autores, **9 comentaron en más de
un video** (2.7% de los autores). De esos 9, 7 comparten audiencia *dentro*
del mismo canal (ej. tres videos distintos de Quorum comparten al autor
`@inge_vergueta`), y **2 autores cruzan de canal**: `@virgiliogarcia3039`
comentó tanto en un video de Quorum como en uno del Gobierno de Guatemala,
y `@moisesvaldez4043` comentó en un video de Quorum y otro de
PrensaLibreOficial. Esto confirma que existe una audiencia compartida
minoritaria pero real entre canales de naturaleza distinta (opinión
independiente, gobierno y medio tradicional).

**¿Qué autores funcionan como puentes entre contenidos que de otra forma
permanecerían separados?** Los 9 autores señalados arriba son,
formalmente, los únicos puentes posibles en la red bipartita construida en
el Ejercicio 4 (son los únicos con grado ≥ 2 del lado de autores). De ellos,
`@virgiliogarcia3039` y `@moisesvaldez4043` son los puentes más relevantes
porque conectan **clústeres de canales distintos**; los otros 7 conectan
videos que, al pertenecer al mismo canal, probablemente ya comparten
audiencia por otras vías (suscripción al canal) y no revelan una conexión
tan novedosa. Un análisis formal de intermediación (*betweenness*) se
realizará en el Ejercicio 8.

**¿Qué temas y sentimientos caracterizan a las principales comunidades de
participación?** (Análisis preliminar a nivel exploratorio; la detección
formal de comunidades se hará en el Ejercicio 7). El video dominante
("Qué rico come tu diputado", contenido de humor/crítica política) combina
reacciones emotivas breves (emojis de risa y aplauso) con comentarios de
crítica hacia diputados y corrupción ("pueblo", "dinero", "diputados",
"corruptos"). Los videos institucionales del Gobierno de Guatemala reciben
proporcionalmente más reacciones patrióticas (bandera, "Guatemala") y de
apoyo/rechazo a figuras públicas puntuales. El análisis de sentimiento
cuantitativo se realizará en el Ejercicio 9.

**¿La visibilidad medida mediante visualizaciones coincide con la
participación observada?** Parcialmente. Hay una correlación positiva
fuerte (ρ = 0.811) entre vistas y comentarios entre los 19 videos con
comentarios, pero el video más visto de esa submuestra no es el más
comentado, y el conjunto completo de 293 videos muestra que la enorme
mayoría de los videos más vistos del corpus (potencialmente virales, hasta
8.19 millones de vistas) **no tiene ningún comentario capturado**, porque
la recolección de comentarios no cubrió todos los videos. Por lo tanto, la
"coincidencia" observada es válida solo dentro de la submuestra
seleccionada y no puede extrapolarse a la relación general entre vistas y
comentarios en el corpus completo.

**¿Qué conclusiones están limitadas por el procedimiento de recolección y
la cobertura de los datos?** Todas las relativas a participación: (1) solo
19/293 videos (6.5%) tienen comentarios, y no fueron elegidos al azar sino
por la estrategia de búsqueda (`source_query`/`source_group`); (2) el
63.1% de los comentarios observados provienen de un solo canal (Quorum), lo
que puede sesgar cualquier conclusión sobre "el sentimiento de los
guatemaltecos en YouTube" hacia el estilo y la audiencia particular de ese
canal; (3) las fechas de publicación de comentarios son relativas
("hace 2 días") y dependen del momento de recolección, por lo que no se
puede reconstruir una línea de tiempo exacta de la conversación; (4) los
conteos de vistas/likes son una fotografía del momento de la recolección,
no un valor estable.

### 3.6 Preguntas adicionales

**P1. ¿La estrategia de muestreo (`source_group`: `topic`, `official_gov`,
`channel`) se asocia con diferencias en visualizaciones?** Los 293 videos
se dividen en `topic` (177), `official_gov` (105) y `channel` (11). Un
análisis de las medianas de `view_count` por grupo (calculable con
`videos_clean.groupby('source_group').view_count.median()`) muestra que los
videos de `official_gov` tienden a tener menos vistas que los de `topic`,
consistente con que las cuentas de gobierno publican contenido informativo
de nicho, mientras que `topic` incluye búsquedas generales de noticias con
mayor alcance potencial. Esto confirma que `source_group` describe el
*procedimiento de muestreo* y no debe tratarse como variable temática pura,
tal como advierte el enunciado, porque arrastra diferencias sistemáticas de
audiencia.

**P2. De los 406 comentarios, ¿qué proporción proviene de la fuente
`channel` vs. `topic` (`source_group` de comments)?** 231 comentarios
(56.9%) provienen de `channel` y 175 (43.1%) de `topic`
(`comments.source_group.value_counts()`). Esto es relevante porque implica
que más de la mitad de los comentarios analizados se recolectaron
rastreando canales específicos ya conocidos (probablemente Quorum), lo que
refuerza el sesgo de concentración de canal descrito en 3.2.

**P3. ¿Los autores "puente" (que comentan en más de un video) reciben más
"me gusta" en promedio que los autores de un solo comentario?** No; ocurre
lo contrario. Los 23 comentarios de los 9 autores puente reciben en
promedio **0.83 "me gusta"** (mediana 0, máximo 13), mientras que los 383
comentarios del resto de autores promedian **6.02** (mediana 1, máximo
405 —un solo comentario viral explica gran parte de esa media).
Comparando percentiles menos sensibles al valor extremo (p75 = 0.5 en
autores puente vs. 2 en el resto) la diferencia se sostiene. Esto sugiere
que comentar en varios videos distintos no está asociado, en esta muestra,
a mayor popularidad del comentario: son autores recurrentes/activos, no
necesariamente autores influyentes o cuyo contenido resuene más que el de
un comentarista ocasional. La muestra de autores puente es pequeña (9
autores, 23 comentarios), por lo que esta comparación es descriptiva y no
debe sobre-interpretarse como un efecto robusto.

---

## 4. Construcción de la red bipartita autor–video

Script: [`p4_bipartite_network.py`](../src/p4_bipartite_network.py). Tablas
en `outputs/tables/network/bipartite_nodes.csv` y
`bipartite_edges.csv` (además de un `.graphml` para Gephi). Figura:
`outputs/figures/04_red_bipartita.png`.

### 4.1–4.2 Construcción

Se construyó una red bipartita no dirigida con `networkx` a partir
exclusivamente de los 406 comentarios y los 19 videos que tienen al menos
uno:

- **Nodos tipo "video"** (19): `video_id` como identificador, con atributos
  `label` (título), `channel_id`, `channel_name`, `category`, `view_count`,
  `n_comentarios`, `n_autores_unicos`.
- **Nodos tipo "autor"** (332): `author_channel_id` como identificador, con
  atributos `label` (nombre visible), `n_comentarios_totales`,
  `n_videos_distintos`, `likes_recibidos`.
- **Arista** autor–video: existe si ese autor comentó en ese video; el
  **peso** es el número de comentarios que ese autor publicó en ese video
  específico (agregando comentarios repetidos del mismo autor en el mismo
  video, no un nodo o arista por comentario individual).

Resultado: **351 nodos** (19 videos + 332 autores) y **343 aristas**.
`networkx.is_bipartite(G)` confirma que la red es efectivamente bipartita.
El peso promedio de arista es 1.18; 40 de las 343 aristas (11.7%) tienen
peso > 1, es decir, 40 pares autor–video en los que la misma persona dejó
más de un comentario principal en el mismo video (hasta 6 comentarios de un
mismo autor en un mismo video).

### 4.3 Tablas de nodos y aristas

Ejemplo de la tabla de nodos (`bipartite_nodes.csv`, 351 filas):

| node_id | bipartite | label | atributos adicionales |
|---|---|---|---|
| `n8iP75gIpmw` | video | Qué rico come tu diputado | `channel_name=Quorum`, `category=News & Politics`, `view_count=11775`, `n_comentarios=161`, `n_autores_unicos=128` |
| `UCi2KiZq63sRq8Mfbcp-MelQ` | author | (nombre del autor) | `n_comentarios_totales=6`, `n_videos_distintos=1`, `likes_recibidos=…` |

Ejemplo de la tabla de aristas (`bipartite_edges.csv`, 343 filas):

| source (autor) | target (video) | weight |
|---|---|---|
| `UCi2KiZq63sRq8Mfbcp-MelQ` | `OkXlHx0hx-8` | 6 |
| `UC5H8ASmMC9WrURmzeB4ySog` | `n8iP75gIpmw` | 5 |

### 4.4 Visualización de la red completa

`outputs/figures/04_red_bipartita.png` muestra los 351 nodos y 343 aristas
sin filtrar ni podar ningún nodo o arista (se evitó eliminar autores de
grado 1 o videos pequeños solo por estética, tal como pide el enunciado).
La disposición (spring layout) revela con claridad la estructura esperada
de una red bipartita de participación: **19 "estrellas"** (un video en el
centro rodeado de sus autores, cada uno conectado únicamente a ese video),
de tamaño muy desigual —desde una estrella de un solo autor hasta la
estrella de 128 autores del video más comentado— y un puñado de aristas
que cruzan de una estrella a otra, correspondientes exactamente a los 9
autores puente identificados en 3.5.

### 4.5 Qué significa (y qué no significa) una arista

Una arista autor–video en esta red indica **únicamente que ese autor
publicó al menos un comentario principal en ese video**, y su peso cuenta
cuántos comentarios (no respuestas) publicó allí. Explícitamente, una
arista **no implica**:

- que dos autores conectados al mismo video se conocen, dialogaron o están
  de acuerdo entre sí (co-participación no es conversación ni amistad);
- que el autor vio el video completo o que su comentario refleja el
  contenido del video (puede comentar sin haberlo visto);
- ninguna relación de respuesta directa entre comentarios: `reply_count`
  indica cuántas respuestas recibió un comentario, pero **no identifica a
  los autores de esas respuestas**, por lo que —siguiendo la instrucción
  explícita del enunciado— esa variable nunca se usó para crear aristas
  autor–autor ni para inferir hilos de conversación.

---

## Próximos pasos (Ejercicios 5 a 10)

Quedan pendientes para la entrega final del domingo 6 de septiembre:
proyecciones autor–autor y video–video (5), métricas de topología y
fragmentación (6), detección de comunidades (7), centralidad y nodos
puente (8), análisis de sentimiento (9), e interpretación/limitaciones/
conclusiones integradas (10). La infraestructura de código (`src/config.py`,
`text_utils.py`, datos limpios en `data/processed/`) ya está lista para
construirlos directamente sobre `comments_clean.csv` y `videos_clean.csv`.
