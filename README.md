# AutoLayoutDM

Preprocessing pipeline y entrenador para **generación de layouts de UI móvil** usando difusión discreta sobre el dataset RICO.

El sistema convierte anotaciones semánticas de apps Android en tokens discretos y entrena un modelo **LayoutDM** (Transformer + difusión discreta tipo VQ-Diffusion) capaz de generar layouts plausibles de la forma `(category, x, y, w, h)`.

```
RICO JSON → elementos UI → tokens discretos → LayoutDM → layouts generados
```

---

## Estructura del repositorio

```
rico_layoutdm/
├── layoutdm_preprocesamiento.py   # Builder: RICO → tokens discretos
├── layoutdm_trainer.py            # Entrenador LayoutDM (iter4)
├── rico_to_layoutdm_tokens.py     # Script standalone de conversión
├── LayoutDM_(iter4_blueprint).ipynb         # Notebook Colab — modelo
├── LayoutDM_(preprocesamiento_iter4).ipynb  # Notebook Colab — preprocesamiento
├── DOCUMENTATION.md               # Documentación técnica del builder
└── EXAMPLES.md                    # Ejemplos de entrada/salida por función
docs/
└── main.md                        # Documentación completa del proyecto
CHANGELOG.md                       # Historial de cambios por iteración
```

---

## Qué hace el sistema

### A. Preprocesamiento (`layoutdm_preprocesamiento.py`)

Lee los archivos JSON de `semantic_annotations/` de RICO, extrae los elementos UI de cada pantalla y los convierte a tokens discretos.

**Pipeline:**

1. Recorre el árbol JSON de cada pantalla y filtra por la whitelist oficial de 25 categorías de RICO (`RICO25_LABELS`)
2. Filtra elementos con geometría inválida (fuera de bounds, degenerados)
3. Aplica NMS opcional para eliminar cajas casi duplicadas
4. Calcula `M` (longitud de secuencia) como el percentil p95 del número de elementos, redondeado al múltiplo de 5 más cercano
5. Hace split 80/10/10 reproducible con `SEED=42`
6. Ajusta 4 modelos KMeans independientes sobre train (`x`, `y`, `w`, `h`, `BINS=64` clusters)
7. Discretiza cada elemento y exporta tensores `[N, M, 5]` + artefactos de metadatos

**Artefactos exportados:**

| Archivo | Contenido |
|---|---|
| `tokens_train.pt` | `LongTensor [52956, 55, 5]` |
| `tokens_val.pt` | `LongTensor [6619, 55, 5]` |
| `tokens_test.pt` | `LongTensor [6620, 55, 5]` |
| `centroids_{x,y,w,h}.pt` | `FloatTensor [64]` por modalidad |
| `cat2id.json` | `{ "Button": 0, "Text": 1, ... }` — 25 categorías |
| `vocab_meta.json` | vocab sizes, pad/mask ids, M, bins, seed, configuración de filtrado |
| `split_ids.json` | IDs reales de pantalla por split para trazabilidad |

### B. Entrenador (`layoutdm_trainer.py`)

Entrena un Transformer encoder como denoiser de difusión discreta (LayoutDM).

**Arquitectura:**
- Transformer encoder (4 capas, 8 cabezas, `d_model=512`)
- Embeddings separados por modalidad + positional encodings desacoplados (`elem_pos` + `attr_pos`)
- Flatten intercalado por elemento: `[c1,x1,y1,w1,h1, c2,x2,y2,w2,h2, ...]`
- Un head de proyección por modalidad (5 heads)

**Proceso de difusión:**
- `T=100` pasos, schedule mask-and-replace tipo VQ-Diffusion
- Matrices de transición `Q_t` y acumuladas `Q̄_t` precalculadas por modalidad
- Pérdida: VB loss (KL) + auxiliary loss (CE), `λ=0.1`

**Sampling (iter4 — correcto):**
- Inicializa `z_T` con tokens `[MASK]`
- En cada paso `t=T..1`: calcula `p_theta(z_{t-1}|z_t)` via `compute_theta_posterior` (LayoutDM Eq. 3) y muestrea
- Regla de coherencia PAD: si `category == PAD`, fuerza `x/y/w/h` a sus respectivos `PAD`

---

## Dataset

**RICO** — 66,261 pantallas de apps Android con anotaciones semánticas:
- 66,195 pantallas parseadas correctamente
- 25 categorías de elementos UI (whitelist oficial `CyberAgentAILab/layout-dm`)
- Splits: 52,956 train / 6,619 val / 6,620 test

---

## Configuración principal

```python
# layoutdm_preprocesamiento.py
BINS = 64              # clusters KMeans por coordenada
M_PERCENTILE = 95      # percentil para elegir longitud de secuencia (→ M=55)
DISCARD_LONG_SCREENS = True   # descartar pantallas con N > M (comportamiento oficial del paper)
NMS_IOU_THRESHOLD = 0.85      # umbral NMS para eliminar cajas duplicadas
SEED = 42

# layoutdm_trainer.py — TrainConfig
T = 100                # pasos de difusión
lambda_aux = 0.1       # peso de la aux loss
lr = 5e-4
n_layers = 4
n_heads = 8
d_model = 512
```

---

## Uso rápido (Google Colab)

### 1. Preprocesamiento

```python
# Ajustar rutas en layoutdm_preprocesamiento.py:
RICO_SEMANTIC_DIR = "/content/semantic_annotations/semantic_annotations"
OUT_DIR = "/content/layoutdm_rico_tokens"

# Ejecutar main()
from rico_layoutdm.layoutdm_preprocesamiento import main
main()
```

### 2. Entrenamiento

```python
from rico_layoutdm.layoutdm_trainer import main

model, cfg, train_ds, val_ds, vocab_meta, Qts_all, Qbars_all = main(
    data_dir="/content/layoutdm_rico_tokens",
    batch_size=16,
    epochs=50,
)
```

### 3. Generación de layouts

```python
from rico_layoutdm.layoutdm_trainer import unconditional_sample

samples = unconditional_sample(
    model, cfg, vocab_meta, batch_size=8,
    Qts_all=Qts_all, Qbars_all=Qbars_all,
)
# samples: LongTensor [8, M, 5]  — (category_id, x_id, y_id, w_id, h_id) por elemento
```

---

## Historial de iteraciones

| Iteración | Cambio principal |
|---|---|
| Preprocessing iter1 | Pipeline base RICO → tokens |
| Preprocessing iter2 | Fix bug de índices `split_ids`, snapping a resolución canónica |
| Preprocessing iter3 | Exporta `split_ids.json`, `render_debug_overlays()` |
| Preprocessing iter4 | Whitelist RICO25, filtro NMS, `DISCARD_LONG_SCREENS`, `M_ROUND_BASE` |
| Trainer iter1 | Blueprint con dataset dummy |
| Trainer iter2 | Conecta artefactos reales, `pad_mask` por modalidad, decode/render |
| Trainer iter3 | `forward()` intercalado, todas las funciones implementadas |
| Trainer iter4 | `compute_theta_posterior` — KL y sampling correctos (LayoutDM Eq. 3) |

Ver [CHANGELOG.md](CHANGELOG.md) para el detalle completo de cada iteración.

---

## Documentación

- [docs/main.md](docs/main.md) — documentación completa: dataset, pipeline, decisiones de diseño, diagnóstico de problemas, implementación del entrenador
- [rico_layoutdm/DOCUMENTATION.md](rico_layoutdm/DOCUMENTATION.md) — referencia técnica del builder de tokens
- [rico_layoutdm/EXAMPLES.md](rico_layoutdm/EXAMPLES.md) — ejemplos de entrada/salida por función

---

## Referencias

- [LayoutDM: Discrete Diffusion Model for Geospatial Layout Generation](https://cyberagentailab.github.io/layout-dm/) — CyberAgent AI Lab
- [RICO: A Mobile App Dataset for Building Data-Driven Design Applications](https://interactionmining.org/rico)
- [VQ-Diffusion: Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/abs/2111.14822)
