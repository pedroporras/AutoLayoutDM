# CHANGELOG — AutoLayoutDM / RICO → LayoutDM Preprocessing

Todos los cambios notables de este proyecto están documentados aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/).

---

## [Trainer iter 4 — layoutdm_trainer.py] — 2026-03-30

### Corregido
- **Bug principal de iter3: `unconditional_sample` usaba direct softmax sampling en lugar del posterior verdadero.**  
  En iter3, el muestreo inverso obtenía `probs = F.softmax(logits[m], dim=-1)` directamente, lo que equivalía a muestrear desde `p_theta(z_0 | z_t)` en lugar de desde `p_theta(z_{t-1} | z_t)`. El modelo no retrocedía correctamente por el proceso de difusión.  
  **Fix**: ahora se usa `compute_theta_posterior(p_theta_z0, zt[:,:,a], Qt, Qbar_prev)` para obtener la distribución marginalizada correcta antes de muestrear.

- **`compute_losses` usaba el softmax crudo del modelo como `p_model` en el KL.**  
  En iter3 la VB loss calculaba KL entre `q_posterior_true(z0,zt)` y `F.softmax(logits)`, es decir comparaba `q(z_{t-1}|z_t,z_0)` (sobre `z_{t-1}`) contra `p_theta(z_0|z_t)` (sobre `z_0`), que son distribuciones sobre vocabularios distintos. El KL no era coherente.  
  **Fix**: `p_model` se calcula ahora como `compute_theta_posterior(p_theta_z0, zt_m, Qt, Qbar_prev)`, produciendo `p_theta(z_{t-1}|z_t)` sobre el mismo espacio que `q_posterior_true`. El KL es ahora matemáticamente correcto.

### Añadido
- **`compute_theta_posterior(p_theta_z0, zt_m, Qt, Qbar_t1)`** — implementación de la marginalización VQ-Diffusion (LayoutDM Eq. 3 / VQ-Diffusion Eq. 7):
  ```
  p_theta(z_{t-1}=v | z_t=s)  ∝  Qt[v,s]  *  (p_theta_z0 @ Qbar_{t-1})[v]
  ```
  - `p_z1_marginal = p_theta_z0 @ Qbar_t1`  → `[B, M, V]`
  - `Qt_col = Qt[:, zt_m].permute(1,2,0)` → `[B, M, V]`
  - `unnorm = Qt_col * p_z1_marginal` → normalizado con `+1e-12`
  - Usada tanto en `compute_losses` (lado modelo del KL) como en `unconditional_sample` (sampling inverso).

### Modificado
- `unconditional_sample`: reemplaza el flujo de sampling por `compute_theta_posterior → torch.multinomial`, coherente con el entrenamiento. El sampling ahora recorre correctamente `p_theta(z_{t-1}|z_t)` en cada paso de denoising.
- `compute_losses`: `p_model` ahora se construye vía `compute_theta_posterior`; el KL se calcula entre `q_posterior_true` (verdadero) y `p_theta(z_{t-1}|z_t)` (modelo), ambos sobre el vocabulario de `z_{t-1}`.
- `main()`: retorna `Qts_all, Qbars_all` adicionalmente a `(model, cfg, train_ds, val_ds, vocab_meta)` para reusar las matrices precalculadas en celdas interactivas sin recomputar.

---

## [Trainer iter 3 — layoutdm_trainer.py] — 2026-03-30

### Corregido
- **Bug crítico: `forward()` usaba flatten por bloques de modalidad en lugar de secuencia intercalada por elemento.**  
  En iter2, `torch.cat(reps, dim=1)` producía una secuencia `[c1..cM, x1..xM, y1..yM, w1..wM, h1..hM]` donde el Transformer veía primero todas las categorías juntas y luego todas las posiciones; nunca podía atender a la categoría y la geometría del **mismo** elemento en la misma ventana de atención.  
  **Fix**: ahora se usa `torch.stack([c_all, x_all, y_all, w_all, h_all], dim=2)` → `reshape(B, M*5, D)` para obtener la secuencia intercalada `[c1,x1,y1,w1,h1, c2,x2,y2,w2,h2, ...]`. El desaplanado de vuelta usa `h.reshape(B, M, 5, D)` con extracción explícita por índice de atributo.

- **Caso borde `t=1` en `Qbars_prev`:** cuando `t==1`, `Qbar_{t-1}` debe ser la matriz identidad (`q(z_0|z_0)=I`). En iter2 esto no estaba manejado explícitamente y podría producir un índice negativo `[t-2]=-1`. Ahora se construye `torch.eye(V)` explícitamente para `t==1`.

- **Validación de rango de `t`:** se añade `if not (1 <= t <= cfg.T): raise ValueError(...)` para detectar tempranamente cualquier error de off-by-one en el muestreo del timestep.

### Añadido
- **`categorical_sample` implementada:**
  ```python
  flat = probs.reshape(-1, probs.size(-1))
  out  = torch.multinomial(flat, num_samples=1).squeeze(-1)
  return out.view(probs.shape[:-1])
  ```
  Era un stub vacío en iter2.

- **`LayoutDMDenoiser.forward()` completamente implementado:** embeddings por modalidad con `elem_pos + attr_pos`, flatten intercalado (corrección del bug de iter2), Transformer encoder, unflatten `[B, M, 5, D]`, extracción explícita de `h_c, h_x, h_y, h_w, h_h`, head por modalidad → logits `[B, M, V_m]`.

- **`compute_losses()` completamente implementado:** produce `q_posterior_true()` para la VB loss (KL), `F.softmax` para la distribución del modelo, `F.cross_entropy` para la aux loss, máscara `valid = ~pad_mask` normalizada por `valid.sum().clamp_min(1.0)`. Retorna `(total_loss, metrics_dict)` con `loss_total`, `loss_vb` y `loss_aux` desagregados como floats para logging.

- **`train_one_epoch()` completamente implementado:** transfiere tensores a device, muestrea `t` uniforme en `[1, T]`, construye `zt` por modalidad via `q_sample_from_Qbar`, maneja el caso `t==1` con identidad para `Qbars_prev`, llama a `compute_losses`, hace `backward/step`. Logging de progreso cada 50 pasos:  
  `step=0050 t=042 loss=1.2345 vb=1.1234 aux=0.1111`

- **`unconditional_sample()` completamente implementado:** inicializa `z_T` con tokens `[MASK]` por modalidad, itera `t=T..1` con `model(zt) → softmax → multinomial`, aplica la **regla de coherencia PAD** con `torch.where` explícito para cada atributo geométrico: si `c == PAD`, fuerza `x/y/w/h` al PAD correspondiente.

- **`main()` con logging por epoch:** `print(f"\n=== Epoch {ep+1}/{epochs} ===")` antes de cada llamada a `train_one_epoch`.

### Modificado
- `precompute_Q_mats`: almacena las matrices en **listas** (`Qts_all[m] = list`, `Qbars_all[m] = list`) en lugar de diccionarios. La indexación usa offset: `Qts_all[m][t-1]`, `Qbars_all[m][t-1]`, `Qbars_all[m][t-2]`.
- `main()`: retorna explícitamente `(model, cfg, train_ds, val_ds, vocab_meta)` para uso interactivo en el notebook Colab.

---

## [Trainer iter 2 — layoutdm_trainer.py] — 2026-03-30

### Corregido
- **Bug crítico: `LayoutTokenDataset` usaba un único `pad_token_id` global para construir `pad_mask`.**  
  En iter1, todos los atributos se comparaban contra el mismo `pad_id`, pero la tokenización de iter3/iter4 tiene `pad_id` distinto por modalidad (`C+1` para categoría, `BINS+1` para geometría). Esto producía que las posiciones PAD de las modalidades geométricas no se enmascararan correctamente.  
  **Fix**: `LayoutTokenDataset` ahora acepta `vocab_meta: Dict[str, Dict[str, int]]`; construye `self.pad_ids = [vocab_meta[m]["pad_id"] for m in modalities]` como tensor `[5]` y usa `tokens.eq(self.pad_ids)` con broadcast para generar `pad_mask [M, 5]` correcto por modalidad.

- **Dataset ficticio eliminado**: `load_or_make_dataset` (generador de tokens aleatorios) reemplazado por carga real de artefactos desde disco. El entrenamiento ya no usa datos sintéticos.

### Añadido
- `load_real_dataset(data_dir, split)`: carga `tokens_{split}.pt` con validación de existencia, tipo y shape `[N, M, 5]`.
- `load_vocab_meta(data_dir)`: carga `vocab_meta.json` con validación de claves requeridas (`vocab_size`, `mask_id`, `pad_id` por modalidad, `M` global).
- `load_real_datasets(cfg, data_dir, shuffle_train_elements)`: orquesta la carga de train + val + `vocab_meta`. Valida consistencia de `M` entre los tensores y el JSON; sobreescribe `cfg.M` desde `vocab_meta["M"]` para garantizar que el modelo use la dimensión real.
- `LayoutTokenDataset._shuffle_valid_elements()`: permuta solo los elementos reales (aquellos cuya categoría ≠ `c.pad_id`), dejando el bloque PAD al final sin tocarlo. Se activa opcionalmente por flag `shuffle_elements=True`.
- `decode_layout(tokens, vocab_meta, id2cat, centroids)`: decodifica un tensor `[M, 5]` de tokens discretos a lista de elementos con coordenadas reales `(x, y, w, h)` usando los centroides KMeans. Filtra tokens PAD/MASK con `safe_centroid_lookup`.
- `safe_centroid_lookup(centroid_tensor, idx)`: lookup de centroide con comprobación de rango; retorna `None` para índices fuera de bounds.
- `render_layout(ax, decoded, canvas_w, canvas_h, title)`: renderiza un layout decodificado sobre un `Axes` de matplotlib con bboxes coloreadas. Usable tanto en modo plot individual como en grids.
- Pipeline de visualización de validación: carga 20 pantallas aleatorias de `tokens_val.pt` (o las primeras `N`) y las renderiza en grid de 4 columnas con `render_layout`. Usa `RANDOM_SEED=42` para reproducibilidad.
- Celdas de setup de Colab: copian artefactos (`tokens_*.pt`, `vocab_meta.json`, `cat2id.json`, `centroids_*.pt`) desde Google Drive a `/content/layoutdm_rico_tokens`.

### Modificado
- `main()`:
  - Usa `load_real_datasets()` en lugar del dataset ficticio.
  - Crea `val_loader` separado para evaluación.
  - El checkpoint incluye `vocab_meta` junto a `cfg.__dict__` y `model.state_dict()`.
  - Retorna `(model, cfg, train_ds, val_ds, vocab_meta)` para uso interactivo en el notebook.
- `cfg.M` ahora se determina en tiempo de ejecución desde `vocab_meta["M"]`; el valor hardcodeado `M=25` de iter1 queda como fallback de debug, no como valor de producción.

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

---

## [Preprocessing iter 2 — layoutdm_preprocesamiento.py] — 2026-03-30

### Corregido
- **Bug crítico: desfase de índices entre preprocesamiento y lookup de ejemplos.**  
  En iter1, `split_ids` se llamaba con `len(json_files)` (total de ficheros JSON, incluyendo los inparsables), pero los tokens se construían solo sobre `screens` (pantallas parseables). Esto producía que el índice del token `[i]` apuntara a una pantalla distinta a la esperada.  
  **Fix**: el split ahora se calcula sobre `len(screens)` (solo las pantallas parseables); la lista `good_ids` captura exactamente los IDs que entraron en los tensores.

- **Bug: `infer_base_wh_from_root` calculaba `base_w = x0 + x1` en lugar de `base_w = x1 - x0`.**  
  La heurística original sumaba las coordenadas en lugar de restar para obtener el ancho real.  
  Corrección intermedia documentada en los comentarios del código; la función fue finalmente eliminada en favor del snapping a resolución canónica (ver *Modificado* abajo).

### Modificado
- `rico_semantic_json_to_elements` refactorizado con nueva estrategia de normalización:
  - Elimina la dependencia de `infer_base_wh_from_root` y de `_infer_screen_size_from_tree`.
  - Infiere la resolución de diseño del dispositivo haciendo **snapping al candidato estándar de RICO más cercano** (`720×1280`, `1080×1920`, `1440×2560`) usando `sum_w = x0 + x1`, `sum_h = y0 + y1` como estimadores.
  - Normalización directa por `design_w` / `design_h` en coordenadas absolutas, consistente con el enfoque "línea amarilla" de las visualizaciones de debug.

### Eliminado
- `_infer_screen_size_from_tree`: ya no forma parte del path principal (comentada en el código).
- `_bounds_to_xywh_norm`: reemplazada por la normalización inline en `rico_semantic_json_to_elements`.
- `infer_base_wh_from_root`: sustituida por el snapping a resoluciones canónicas.

### Añadido
- Patrón de validación `good_ids` / `bad_ids`: antes de hacer el split, se construye la lista de IDs parseables para garantizar coherencia entre índices de tokens y pantallas reales.
- `can_infer_screen_size`: helper de validación para verificar si una pantalla tiene bounds usables antes de incluirla.
- Workflow de debug mejorado: usa `screens = load_all_screens(SEM_DIR)` + `screen_ids = [s["id"] for s in screens]` para mapear correctamente fila de token → ID de pantalla en las celdas de inspección visual.

---

## [Preprocessing iter 3 — layoutdm_preprocesamiento.py] — 2026-03-30

### Añadido
- `main()`: exporta `split_ids.json` con los IDs reales de pantalla por split (train / val / test), para garantizar trazabilidad exacta entre `tokens_train[i]` y el JSON de origen.
- `render_debug_overlays()`: renderiza `K` pantallas con bboxes decodificadas dibujadas sobre los screenshots originales; lee `split_ids.json` en lugar de reconstruir el split desde `os.listdir()`, evitando el desfase de índices del iter1.

### Eliminado
- Código comentado residual de iter1/iter2: funciones `_infer_screen_size_from_tree`, `_bounds_to_xywh_norm`, `infer_base_wh_from_root` y bloques `# old / # legacy` dentro de `rico_semantic_json_to_elements` — el código queda limpio sin rastros de iteraciones anteriores.

---

## [Preprocessing iter 4 — layoutdm_preprocesamiento.py] — 2026-03-30

### Añadido
- `RICO25_LABELS`: whitelist estricta de las 25 categorías semánticas oficiales de LayoutDM/RICO (`CyberAgentAILab/layout-dm`). Solo se conservan elementos cuyo `componentLabel` esté en este conjunto; contenedores, `ViewGroup`, nodos sin etiqueta y clases desconocidas se descartan implícitamente.
- `DISCARD_LONG_SCREENS`: flag que reproduce el comportamiento oficial del paper — las pantallas con `N > M` se descartan en lugar de truncarse. `DISCARD_LONG_SCREENS=False` mantiene el comportamiento de truncado de iteraciones anteriores.
- `NMS_IOU_THRESHOLD` / `PREFER_LEAVES`: filtro NMS opcional para eliminar cajas casi duplicadas (IoU ≥ umbral). Prioriza nodos hoja cuando hay empate. Desactivable fijando `NMS_IOU_THRESHOLD = 1.0`.
- `M_ROUND_BASE`: `M` ahora se redondea al múltiplo de `M_ROUND_BASE` (por defecto 5) más cercano por encima del percentil crudo.
- `round_up_to_multiple()`: utilidad de redondeo hacia arriba a múltiplos.
- `_is_leaf()`: helper que indica si un nodo del árbol UI no tiene hijos.
- `_iou_2d()`: cálculo de IoU entre dos cajas 2D.
- `_nms_filter()`: greedy NMS sobre la lista de elementos crudos; ordena por (leaf_score, -area) para priorizar nodos hoja y más pequeños.
- `vocab_meta.json` ahora incluye:
  - campo `"filter"` con la configuración completa de filtrado (`rico25_labels`, `nms_iou_threshold`, `prefer_leaves`, `discard_long_screens`, `drop_root`) para trazabilidad reproducible del pipeline.
  - campos `M_raw` y `M_round_base` para documentar el valor crudo del percentil y el factor de redondeo usado.

### Modificado
- `rico_semantic_json_to_elements`: refactorizada con tres etapas de filtrado explícitas:
  1. **Label whitelist** (filtro oficial LayoutDM) — descarta nodos cuyo `componentLabel` no está en `RICO25_LABELS`.
  2. **`is_valid` geométrico** (filtro oficial LayoutDM) — descarta elementos parcial o totalmente fuera de los límites de pantalla, o con tamaño degenerado.
  3. **NMS opcional** (extra, no presente en el repo oficial) — elimina cajas casi duplicadas según `NMS_IOU_THRESHOLD`.
- `main()`:
  - Calcula `M` dinámicamente usando `choose_M_from_counts(percentile=M_PERCENTILE)` seguido de `round_up_to_multiple()`, en lugar de usar un valor fijo.
  - Añade paso de descarte de pantallas largas (post-cálculo de `M`) con log del número y porcentaje descartados.
  - `M_PERCENTILE` por defecto cambiado a `50` para facilitar depuración con layouts compactos; el comentario documenta `p90/p95/p99` como alternativas.
