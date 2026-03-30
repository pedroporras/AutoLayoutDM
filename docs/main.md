# Resumen completo del trabajo realizado con RICO → LayoutDM

## 1. Objetivo general

La meta de todo este trabajo fue entender **LayoutDM** como si fuera un sistema de ingeniería y no como un paper académico, y construir un pipeline completo para usar el dataset **RICO semantic annotations** como entrada.

La idea central es esta:

* tenemos un dataset de interfaces (`RICO`)
* extraemos de cada pantalla una lista de elementos visuales
* convertimos esos elementos a una representación estructurada
* discretizamos esa representación a tokens
* exportamos esos tokens para entrenamiento
* entrenamos un modelo de **diffusion discreto** para generar layouts
* depuramos problemas como solapamientos, alineaciones raras o salidas incoherentes

En otras palabras:

**RICO JSON → elementos UI → tokens discretos → LayoutDM**

---

# 2. Qué problema resuelve LayoutDM

LayoutDM resuelve el problema de **generación de layouts**.

Un layout aquí significa:

* qué elementos hay
* dónde están
* qué tamaño tienen

Por ejemplo, en una pantalla móvil:

* un botón
* un título
* una imagen
* una caja de texto

El modelo no genera la imagen final de la UI.
Lo que genera es la **estructura geométrica**:

* categoría del elemento
* posición
* tamaño

Formalmente, cada elemento se representa como:

`(c, x, y, w, h)`

donde:

* `c` = categoría
* `x, y` = centro del elemento
* `w, h` = ancho y alto

---

# 3. Cómo pensamos el sistema completo

Todo el sistema se puede dividir en dos grandes partes:

## A. Preprocesamiento

Convierte el dataset real (RICO) en tensores listos para entrenar.

## B. Entrenamiento del modelo

Entrena LayoutDM sobre esos tensores discretos.

---

# 4. Qué vimos paso a paso

---

## 4.1. Lectura del dataset RICO

RICO no viene listo para LayoutDM.

Partimos de los archivos en `semantic_annotations/`, donde cada JSON representa una pantalla con su árbol de componentes. Un JSON típico tiene esta pinta:

```json
{
  "class": "com.android.internal.policy.PhoneWindow$DecorView",
  "bounds": [0, 0, 1440, 2560],
  "children": [
    {
      "class": "org.apache.cordova.engine.SystemWebView",
      "componentLabel": "Web View",
      "bounds": [0, 0, 1440, 2392]
    }
  ]
}
```

La lógica del parser:

1. recorrer recursivamente todos los nodos (`children`)
2. leer `bounds` de cada nodo
3. convertirlos a coordenadas normalizadas
4. guardar cada nodo como un elemento estructurado usando `componentLabel` si existe, `class` si no

---

## 4.2. Problema inicial: `Invalid screen size from root bounds`

Al ejecutar el builder apareció este error en algunos archivos:

```python
ValueError: Invalid screen size from root bounds.
```

El script asumía que el root siempre venía como `[x0, y0, x1, y1]`, pero algunos archivos tenían bounds dañados, vacíos o con valores imposibles.

### Qué hicimos

Se reforzó el parser para:

1. intentar normalizar bounds (interpretar como `[x0,y0,x1,y1]` o como `[x,y,w,h]`)
2. inferir el tamaño de pantalla desde el árbol completo cuando el root no era válido
3. si no había forma de inferirlo, saltarse ese archivo y registrar warning

### Resultado real del dataset

* **66 195 screens** se pudieron procesar correctamente
* **66 archivos** no pudieron parsearse porque realmente no tenían geometría utilizable

Esto no es un bug del pipeline, sino una condición real del dataset.

---

## 4.3. Estadísticas del dataset y elección de `M`

`M` es el número máximo de elementos por pantalla. Como cada pantalla tiene distinto número de elementos, `M` define la longitud fija del tensor:

* si una pantalla tiene menos de `M` elementos → se rellena con `PAD`
* si tiene más de `M` → se recorta

Estadísticas reales del conteo de elementos por pantalla:

| Percentil | Valor |
|---|---|
| p50 | 14 |
| p90 | 43 |
| **p95** | **55** |
| max | 423 |

### Decisión

```python
M = p95 = 55
```

Esto es razonable porque:

* captura el 95% de las pantallas sin truncamiento
* evita que unos pocos outliers (hasta 423 elementos) inflen el tensor innecesariamente
* permite un entrenamiento mucho más estable que usar el máximo absoluto

---

## 4.4. Split del dataset

Se dividieron las **pantallas válidas** (las 66 195) en:

| Split | Tamaño |
|---|---|
| train | 52 956 |
| val | 6 619 |
| test | 6 620 |

Proporciones: 80 / 10 / 10, reproducible con `SEED = 42`.

**Importante**: el split se hizo sobre `len(screens)` (pantallas parseables), no sobre `len(json_files)` (total de archivos). Esa distinción fue clave para evitar el bug de índices descrito en la sección 6.

---

## 4.5. Construcción del vocabulario de categorías

A partir del split train se creó `cat2id` usando **solo train**:

```json
{
  "Button": 0,
  "Web View": 1,
  "Text": 2,
  ...
}
```

Resultado real: **25 categorías** en train.

Igual que en las modalidades espaciales, se reservaron ids especiales:

* `mask_id = C`
* `pad_id = C + 1`

donde `C = 25`.

---

## 4.6. Discretización geométrica con KMeans

LayoutDM no usa coordenadas continuas.
Usa coordenadas **discretizadas**.

Para eso se entrenaron 4 KMeans independientes sobre el split de train:

* uno para `x`
* uno para `y`
* uno para `w`
* uno para `h`

Con `BINS = 64` clusters cada uno.

### Por qué 64 bins

Balance entre:

* suficiente resolución espacial
* vocabulario no demasiado grande
* entrenamiento estable

Con 64 bins por modalidad:

* `x_id, y_id, w_id, h_id` quedan en `0..63`
* `mask_id = 64`
* `pad_id = 65`
* vocabulario por modalidad geométrica: `64 + 2 = 66`

---

## 4.7. Exportación de artefactos

El pipeline exportó los siguientes archivos en `OUT_DIR`:

| Archivo | Contenido |
|---|---|
| `tokens_train.pt` | `LongTensor [52956, 55, 5]` |
| `tokens_val.pt` | `LongTensor [6619, 55, 5]` |
| `tokens_test.pt` | `LongTensor [6620, 55, 5]` |
| `centroids_x/y/w/h.pt` | `FloatTensor [64]` por modalidad |
| `cat2id.json` | `{ "Button": 0, ... }` |
| `vocab_meta.json` | vocab sizes, pad/mask ids, M, bins, seed, split ratios |

El formato de `vocab_meta.json` es el contrato entre preprocesamiento y training loop.

---

# 5. Validación del preprocesamiento

---

## 5.1. Validación estadística

Se verificó:

* shape de `train_tokens`: `(52956, 55, 5)` ✅
* dtype: `torch.int64` ✅
* rangos de ids dentro de los vocabs esperados
* consistencia del padding

---

## 5.2. Validación visual

Se generaron overlays sobre screenshots reales, dibujando cajas reconstruidas desde los tokens:

1. tomar una pantalla por `screen_id`
2. decodificar sus tokens usando centroides
3. dibujar los bounding boxes sobre la imagen real
4. verificar si quedaban alineados con los elementos de la UI

### Qué se observó inicialmente

Los layouts parecían mal alineados en varios casos. Esto sugería alguno de estos problemas:

* mala normalización
* centroides incorrectos
* screen size mal calculado
* desalineación entre JSON e imagen
* o error en el mapeo entre token row y pantalla original

---

# 6. Depuración del caso concreto: pantalla `"0"`

Se analizó una pantalla específica con JSON muy simple:

* root: `[0, 0, 1440, 2560]`
* un solo hijo: clase `SystemWebView`, bounds `[0, 0, 1440, 2392]`

---

## 6.1. Lo esperado geométricamente

Esa pantalla debería producir exactamente **1 elemento** con:

* `x = 0.5`
* `y = 0.4671875`
* `w = 1.0`
* `h = 0.934375`

Una caja casi de pantalla completa.

---

## 6.2. Lo que devolvía el decode inicialmente

El `decode_row_to_boxes` devolvía **7 cajas** pequeñas, concentradas arriba:

* `y ≈ 0.064`
* `h ≈ 0.065`
* varias cajas tipo toolbar/lista

Eso no coincidía en nada con el JSON.

---

## 6.3. Diagnóstico real

Se reconstruyó el mapeo exacto entre `screens`, split train y posición en `tokens_train`:

```
idx_in_screens:    0
split_name:        train
pos_in_split_tokens: 44822
real_n (non-pad elems): 1
```

### Conclusión

El problema **no era** la tokenización ni el decode.

El problema era que se estaba inspeccionando el row incorrecto:

* se miraba `train_tokens[0]`
* pero el screen `"0"` estaba en `train_tokens[44822]`

Esto ocurrió porque el split se hizo sobre `screens` válidos, mientras que los scripts de debug usaban índices basados en `json_files` completos, incluyendo los 66 archivos fallidos. Eso desplazaba el índice.

---

## 6.4. Lección aprendida

> En datasets preprocesados **nunca** se debe asumir que `tokens_train[i]` corresponde al archivo `i.json`.

Puede haber archivos descartados, shuffles, splits o filtros previos que desplacen los índices.

### Solución recomendada

Guardar siempre los ids exactos por split:

```python
# En el builder, al final de main():
with open(os.path.join(OUT_DIR, "ids_train.json"), "w", encoding="utf-8") as f:
    json.dump([s["id"] for s in train_screens], f, indent=2)

with open(os.path.join(OUT_DIR, "ids_val.json"), "w", encoding="utf-8") as f:
    json.dump([s["id"] for s in val_screens], f, indent=2)

with open(os.path.join(OUT_DIR, "ids_test.json"), "w", encoding="utf-8") as f:
    json.dump([s["id"] for s in test_screens], f, indent=2)
```

Así el debug y los overlays se hacen por `screen_id`, no por posición asumida.

---

# 7. Aspectos importantes que no se deben olvidar

## 7.1. El preprocesamiento no es solo conversión de formato

También define qué elementos entran, cómo se normalizan, cómo se discretizan, y qué información pierde o conserva el modelo. Eso impacta directamente la calidad de LayoutDM.

## 7.2. `M` es una decisión crítica

* Si `M` es muy bajo: truncas layouts complejos
* Si `M` es muy alto: entrenas con mucho padding y el modelo es menos eficiente

Usar `p95 = 55` fue una decisión equilibrada.

## 7.3. KMeans introduce cuantización

Cuando reconstruyes desde tokens, no obtienes los valores exactos originales: obtienes una aproximación por centroides. Eso es normal. Por eso un pequeño error entre la caja real y la reconstruida en un overlay no siempre significa bug.

## 7.4. La trazabilidad screen_id ↔ token row es obligatoria

Sin esa trazabilidad puedes diagnosticar mal el pipeline y creer que el modelo o el preprocesamiento fallan cuando en realidad estás mirando otro ejemplo.

## 7.5. El schedule de diffusion debe ser el exacto del paper

En el notebook quedó como placeholder `make_transition_params()`. La arquitectura estaba bien, pero no era una réplica exacta hasta reemplazar esa función.

## 7.6. El shuffle de elementos por sample todavía faltaba

El paper busca que el modelo no dependa del orden de los elementos. En una implementación más fiel, el `Dataset` debería hacer shuffle de los slots válidos antes de devolver cada muestra.

## 7.7. LayoutDM no entiende semántica profunda

Solo aprende patrones geométricos y categóricos. No sabe qué es "header" o "CTA" como concepto de producto.

---

# 8. Qué implementamos para el entrenamiento

---

## 8.1. Entrada del modelo

```python
tokens: [B, M, 5]
```

Cada token representa `(category, x, y, w, h)` en forma discreta.

---

## 8.2. Arquitectura

Usamos un **Transformer encoder** como denoiser.

No es autoregresivo, no es decoder-only, no genera token por token. LayoutDM modela el layout completo sin depender de un orden fijo.

---

## 8.3. Proceso de diffusion

Durante entrenamiento:

1. tomamos un layout limpio `z0`
2. elegimos un timestep `t`
3. corrompemos `z0` → `zt`
4. el modelo predice una versión más limpia

Eso se hace con **diffusion discreto**.

---

## 8.4. Modality-wise diffusion

Hay una difusión separada por modalidad (`c`, `x`, `y`, `w`, `h`). Esto evita mezclar vocabularios incompatibles. Una categoría no debería transformarse en un ancho.

---

## 8.5. Loss

El entrenamiento usa dos partes combinadas:

### VB loss

KL entre el posterior verdadero del diffusion y el posterior predicho por el modelo. Es el corazón teórico del modelo.

### Aux loss

Cross entropy para ayudar al modelo a reconstruir `z0`. Estabiliza el entrenamiento.

```text
loss_total = vb_loss + lambda_aux * aux_loss
             con lambda_aux = 0.1
```

---

## 8.6. Máscara de PAD

Los slots PAD no representan elementos reales. Se construyó `pad_mask` para que la loss se calcule **solo sobre tokens válidos**. Si PAD entra en la loss, el modelo aprende una distribución errónea.

---

## 8.7. Unconditional sampling

En inferencia:

1. se inicia todo en `[MASK]`
2. se corre el reverse diffusion de `T` hasta `1`
3. en cada paso el modelo predice distribuciones sobre tokens
4. se samplea categóricamente
5. al final se obtiene `z0`

Esto produce un layout nuevo desde cero.

---

# 9. Explicación separada: Preprocesamiento completo

## Objetivo

Transformar cada pantalla de RICO en una secuencia discreta utilizable por LayoutDM.

## Flujo paso a paso

### Paso 1. Leer cada JSON

Se abre cada archivo en `semantic_annotations/` en orden alfabético (determinista).

### Paso 2. Recorrer el árbol

Se recorren root y `children` recursivamente en pre-order.

### Paso 3. Extraer bounds

De cada nodo se toman `[x0, y0, x1, y1]`.

### Paso 4. Convertir a `(x, y, w, h)` normalizado

```text
x = (x0 + x1) / 2 / screen_w
y = (y0 + y1) / 2 / screen_h
w = (x1 - x0) / screen_w
h = (y1 - y0) / screen_h
```

La resolución de pantalla se infiere haciendo snapping al candidato RICO más cercano (`720×1280`, `1080×1920`, `1440×2560`).

### Paso 5. Determinar la categoría

Prioridad: `componentLabel` > `class` > `"UNKNOWN"`

### Paso 6. Construir lista de elementos por pantalla

```python
{"category": ..., "x": ..., "y": ..., "w": ..., "h": ...}
```

### Paso 7. Calcular estadísticas y elegir `M`

`M = ceil(percentil_95)` sobre el conteo de elementos reales.

### Paso 8. Construir `good_ids` / `bad_ids`

Antes de hacer el split, se filtra la lista a solo las pantallas parseables. El split opera sobre esta lista.

### Paso 9. Dividir train/val/test

Shuffle reproducible con `SEED = 42`, proporción 80/10/10.

### Paso 10. Construir `cat2id`

Solo desde train.

### Paso 11. Ajustar KMeans para x/y/w/h

Solo desde train, con `BINS = 64`.

### Paso 12. Convertir cada elemento a tokens discretos

```python
[c_id, x_id, y_id, w_id, h_id]
```

### Paso 13. Aplicar padding hasta `M`

Posiciones vacías reciben `pad_id` en todas las modalidades.

### Paso 14. Exportar artefactos

Tokens, centroides, `cat2id.json`, `vocab_meta.json`, y opcionalmente `ids_train/val/test.json`.

## Resultado

```python
tokens_train: LongTensor [52956, 55, 5]
```

---

# 10. Explicación separada: Entrenamiento del modelo

## Objetivo

Entrenar un LayoutDM mínimo que aprenda a generar layouts discretos.

## Flujo

### Paso 1. Cargar dataset

`tokens_train.pt`, `tokens_val.pt`, `vocab_meta.json`.

### Paso 2. Construir `LayoutTokenDataset`

Un `Dataset` de PyTorch que devuelve `tokens` por índice.

### Paso 3. Construir el modelo

* embeddings por modalidad
* positional encoding
* Transformer encoder
* head de salida por modalidad

### Paso 4. Precomputar matrices de transición

`Qt` (por timestep) y `Qbar` (acumuladas).

### Paso 5. Loop de entrenamiento

Por cada batch:

* samplear `t`
* corromper `z0` → `zt`
* pasar `zt` por el modelo
* calcular `VB + aux`
* backpropagation

### Paso 6. Guardar checkpoint

### Paso 7. Unconditional sampling para inspección

## Posibles errores durante el entrenamiento

| Error | Causa |
|---|---|
| Mapeo equivocado de datos | Pérdida de trazabilidad `screen_id` ↔ row |
| Flatten/interleave incorrecto | El orden de tokens no coincide con lo esperado por el modelo |
| Dataset con mucho ruido | Demasiados layouts atípicos en train |
| `M` mal elegido | Demasiado truncamiento o demasiado padding |
| Cuantización inadecuada | Bins no capturan bien la distribución geométrica |

---

# 11. Estado actual del proyecto

## Ya resuelto

* parser funcional para la mayoría del dataset (66 195 / 66 261 pantallas)
* manejo de archivos dañados con warning en lugar de crash
* cálculo razonable de `M = 55` (p95)
* split reproducible (52 956 / 6 619 / 6 620)
* vocabulario de 25 categorías desde train
* discretización KMeans train-only con BINS = 64
* exportación de tokens y metadata
* validación de shapes, dtype y vocabulario
* diagnóstico correcto del bug de correspondencia índice ↔ pantalla
* implementación del training loop mínimo con VB + aux loss
* implementación de unconditional sampling

## Pendiente para una réplica más fiel

* guardar `ids_train/val/test.json` en el builder
* usar el schedule exacto del paper (reemplazar `make_transition_params()`)
* añadir shuffle por sample en el Dataset
* validar visualmente los overlays usando `screen_id` como clave
* comparar métricas con el paper

---

# 12. Recomendaciones prácticas para continuar

1. **Guardar siempre los ids por split** (`ids_train.json`, `ids_val.json`, `ids_test.json`) para poder hacer debug por `screen_id`.
2. **Guardar `bad_files.json`** para saber qué pantallas quedaron fuera del pipeline.
3. **Construir el script de debug por `screen_id`**, no por índice posicional en el tensor.
4. **Validar overlays** con: caja real (rojo), caja reconstruida (verde), error absoluto por `x,y,w,h`.
5. **Revisar 20-50 ejemplos** antes de entrenar a gran escala: distribución de conteos, comportamiento del padding, coherencia de categorías.
6. **Ejecutar el preprocesamiento completo** sobre RICO real, revisar las estadísticas de `M` y arrancar con un modelo pequeño.

---

# 13. Conclusión

Se logró construir y depurar el pipeline completo de **RICO → tokens discretos para LayoutDM**.

Lo más importante que descubrimos fue que:

* el preprocesamiento en sí estaba funcionando razonablemente bien,
* pero la inspección visual inicial usaba rows equivocados del tensor,
* lo que generaba una falsa impresión de mala alineación.

La depuración mostró que para trabajar correctamente con LayoutDM no basta con generar tokens: también hay que garantizar una trazabilidad precisa entre archivo fuente, split, posición del tensor y reconstrucción visual.

Con eso resuelto, el siguiente paso natural es conectar estos tokens al entrenamiento de LayoutDM, validar muestras generadas y seguir depurando la calidad de generación.

---

# 14. Resumen final simple

### Preprocesamiento

```
RICO JSON
  → parse tree (66195 screens válidas)
  → flatten elements
  → normalize xywh (snapping a resolución canónica RICO)
  → split 80/10/10 (solo sobre screens parseables)
  → KMeans 1D train-only (BINS=64)
  → discretize
  → pad hasta M=55
  → tokens.pt [N, 55, 5]
```

### Entrenamiento

```
tokens.pt → diffusion corruption (modality-wise) → Transformer denoiser → VB + aux → reverse sampling
```

### Resultado final

Un modelo que aprende a generar layouts en formato:

```python
(c_id, x_id, y_id, w_id, h_id)
```

decodificable a cajas geométricas reales usando los centroides KMeans.

---

# Anexo A: Dataset mínimo para entrenamiento

```python
import os, json, torch
from torch.utils.data import Dataset

class LayoutTokensDataset(Dataset):
    def __init__(self, tokens_path, vocab_meta_path):
        self.tokens = torch.load(tokens_path)
        with open(vocab_meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return {"tokens": self.tokens[idx]}
```

---

# Anexo B: Exportar ids por split en el builder

```python
# Al final de main(), añadir:
with open(os.path.join(OUT_DIR, "ids_train.json"), "w", encoding="utf-8") as f:
    json.dump([s["id"] for s in train_screens], f, indent=2)

with open(os.path.join(OUT_DIR, "ids_val.json"), "w", encoding="utf-8") as f:
    json.dump([s["id"] for s in val_screens], f, indent=2)

with open(os.path.join(OUT_DIR, "ids_test.json"), "w", encoding="utf-8") as f:
    json.dump([s["id"] for s in test_screens], f, indent=2)

with open(os.path.join(OUT_DIR, "bad_files.json"), "w", encoding="utf-8") as f:
    json.dump(bad, f, indent=2)
```
