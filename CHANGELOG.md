# CHANGELOG — AutoLayoutDM / RICO → LayoutDM Preprocessing

Todos los cambios notables de este proyecto están documentados aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/).

---

## [Preprocessing iter 4.1] — 2026-03-26

### Modificado
- Refinamiento del notebook `LayoutDM_(preprocesamiento_iter4).ipynb`: ajustes en el pipeline de carga y validación de pantallas.
- Ampliación de la cobertura de casos borde en el parsing de bounds (219 inserciones vs. 107 eliminaciones respecto a iter 4.0).

---

## [Preprocessing iter 4.0] — 2026-03-24

### Añadido
- Nuevo notebook dedicado `LayoutDM_(preprocesamiento_iter4).ipynb` (1 405 líneas).
- Pipeline completo de preprocesamiento: lectura de JSONs RICO → normalización → KMeans → tokens discretos → exportación.
- Script `rico_to_layoutdm_tokens.py` con estrategia de normalización mejorada:
  - **Snapping a resolución canónica**: en lugar de usar los bounds del root directamente, infiere la resolución del dispositivo ajustando al candidato más cercano entre los estándares de RICO `(720×1280, 1080×1920, 1440×2560)`.
  - Prioridad de etiqueta semántica: `componentLabel` > `class` > `"UNKNOWN"`.
  - Documentación completa en docstrings (pipeline, tipos, ejemplos).

---

## [iter 4.5] — 2026-03-24

### Corregido
- `unconditional_sampling`: implementada la fórmula correcta según el paper de LayoutDM para el muestreo sin condición (`notebook LayoutDM_(iter4_blueprint).ipynb`).

---

## [iter 4.4] — 2026-03-23

### Añadido
- Validación completa del diffusion scheduler:
  - `make_exact_schedule_arrays`: construcción exacta de los arrays del schedule.
  - `precompute_Q_mats`: precómputo de las matrices de transición $Q_t$.
  - `inspect_schedule`: inspección de los parámetros del schedule en tiempo de ejecución.
  - `validate_qt_rows`: verificación de que las filas de $Q_t$ sumen 1 (distribuciones válidas).
  - Validación del shuffle aleatorio de elementos por pantalla.

---

## [iter 4.3] — 2026-03-23

### Modificado
- Refactorización mayor del notebook `LayoutDM_(iter4_blueprint).ipynb` (2 003 inserciones / 1 976 eliminaciones): reorganización de celdas y limpieza de código experimental.

---

## [iter 4.2] — 2026-03-23

### Modificado
- Primera ronda de refinamientos sobre el blueprint inicial: ampliación de celdas de modelo y tokenización en `LayoutDM_(iter4_blueprint).ipynb`.

---

## [iter 4.1] — 2026-03-23

### Añadido
- Notebook inicial `LayoutDM_(iter4_blueprint).ipynb` con el blueprint del modelo LayoutDM (1 457 líneas):
  - Arquitectura Transformer para generación de layouts.
  - Integración del proceso de difusión discreta sobre tokens `[N, M, 5]`.

---

## [Preprocessing iter 1 — layoutdm_preprocesamiento.py] — anterior a iter 4

Nota: Esta primera iteracion presenta un problema, hay un error en el indexado entre el preprocesamiento y los ejemplos que se toman para el mapeo del layout

### Añadido
- Script `layoutdm_preprocesamiento.py` (base del pipeline de preprocesamiento):
  - `_walk_nodes`: recorrido pre-order del árbol de UI de RICO.
  - `_normalize_bounds`: interpretación dual de bounds `[x0,y0,x1,y1]` o `[x,y,w,h]`.
  - `infer_base_wh_from_root`: heurística `full_w ≈ x0 + x1` para estimar la resolución completa a partir del root.
  - `_infer_screen_size_from_tree`: fallback — infiere el bbox global de la pantalla recorriendo todos los nodos.
  - `rico_semantic_json_to_elements`: función central de parsing; normaliza coordenadas al rango `[0, 1]` con clamp defensivo.
  - `load_all_screens`: carga y parsea todos los JSONs en orden determinista; registra fallos sin interrumpir el pipeline.
  - `describe_counts` / `choose_M_from_counts`: estadísticas de longitud de secuencia; elige `M` como percentil 95.
  - `split_ids`: partición reproducible train/val/test (80/10/10) con semilla fija.
  - `build_cat2id_from_train`: vocabulario de categorías construido **solo** desde el split de entrenamiento.
  - `fit_kmeans_1d` / `assign_to_nearest_centroid`: discretización 1D de coordenadas con KMeans; centroides ordenados ascendentemente.
  - `_maybe_subsample`: límite de muestras para KMeans (`KMEANS_SAMPLE_LIMIT = 2_000_000`) para evitar OOM.
  - `build_tokens_for_screens`: construcción del tensor `LongTensor [N, M, 5]` con tokens discretos y padding.
  - `decode_tokens_to_xywh` / `sanity_check_decoded`: utilidades de decodificación para verificar la calidad del tokenizado.
  - Exportación de artefactos: `tokens_*.pt`, `centroids_*.pt`, `cat2id.json`, `vocab_meta.json`.

### Configuración por defecto
| Parámetro | Valor | Descripción |
|---|---|---|
| `BINS` | 64 | Clusters KMeans para coordenadas |
| `TRAIN_RATIO` | 0.80 | Proporción del split de entrenamiento |
| `VAL_RATIO` | 0.10 | Proporción del split de validación |
| `TEST_RATIO` | 0.10 | Proporción del split de test |
| `M_PERCENTILE` | 95 | Percentil para elegir la longitud máxima de secuencia |
| `DROP_ROOT` | `True` | Excluir el nodo raíz de los elementos |
| `SEED` | 42 | Semilla global para reproducibilidad |
