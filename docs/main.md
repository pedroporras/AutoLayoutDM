# Resumen completo del trabajo realizado con RICO → LayoutDM / UI-Diffuser

## 1. Objetivo general

La meta de todo este trabajo fue construir la parte de **generación de layouts** del pipeline de **UI-Diffuser**, usando:

* **RICO** como dataset real de interfaces Android
* **LayoutDM** como modelo generativo discreto de layouts
* una tokenización de cada elemento UI en la forma `(c, x, y, w, h)`

La idea central es:

* leemos los archivos JSON de anotaciones semánticas de RICO
* extraemos de cada pantalla una lista de elementos visuales
* convertimos cada elemento a una representación estructurada y discreta
* exportamos esos tokens para entrenamiento
* entrenamos un modelo de **diffusion discreto** para generar layouts plausibles
* depuramos problemas como solapamientos, alineaciones raras o salidas incoherentes

La meta no era todavía generar una imagen UI final con Stable Diffusion, sino primero dejar bien la etapa previa: **aprender a generar layouts plausibles**.

En otras palabras:

**RICO JSON → elementos UI → tokens discretos → LayoutDM → layouts generados**

---

# 2. Qué problema resuelve LayoutDM

LayoutDM resuelve el problema de **generación de layouts**.

Un layout aquí significa:

* qué elementos hay
* dónde están
* qué tamaño tienen

Por ejemplo, en una pantalla móvil:

* un botón, un título, una imagen, una caja de texto

El modelo **no genera la imagen final** de la UI. Genera la **estructura geométrica**:

`(c, x, y, w, h)`

donde:

* `c` = categoría del elemento
* `x, y` = centro del bounding box normalizado
* `w, h` = ancho y alto normalizados

---

# 3. Estado inicial del proyecto

## 3.1. Qué ya existía al inicio

Al inicio ya existían varias piezas importantes:

* arquitectura general de LayoutDM (Transformer encoder como denoiser)
* training loop base con pérdida `VB + auxiliary`
* muestreo incondicional básico
* blueprint del modelo (**iter1 del entrenador** — `layoutdm_trainer.py`, primera versión)
* pipeline inicial de preprocesamiento de RICO

Había una base funcional para arrancar, pero el entrenador iter1 usaba datos **completamente ficticios**: tokens aleatorios generados en memoria, sin ninguna conexión con RICO.

## 3.2. Qué faltaba o estaba incompleto

También había brechas importantes:

* **el entrenador iter1 usaba un dataset dummy** — tokens aleatorios en lugar de datos reales de RICO; cualquier resultado en iter1 no tenía validez sobre el dominio real
* no estaba validado que el preprocesamiento fuera compatible con el blueprint
* faltaba el shuffle por elemento
* faltaban sanity checks visuales sólidos
* el schedule del paper no era exacto
* el flatten del modelo no seguía la estructura intercalada del paper
* `M` estaba fijo en `25` en lugar de calcularse desde los datos reales
* `pad_mask` usaba un único `pad_id` global en lugar de uno por modalidad

La conclusión fue clara: antes de optimizar o comparar métricas, había que conectar el entrenador iter1 con datos reales. Eso dio origen al **iter2 del entrenador**.

---

# 4. Cómo pensamos el sistema completo

Todo el sistema se puede dividir en dos grandes partes:

## A. Preprocesamiento

Convierte el dataset real (RICO) en tensores listos para entrenar.

## B. Entrenamiento del modelo

Entrena LayoutDM sobre esos tensores discretos.

---

# 5. Qué vimos paso a paso

---

## 5.1. Lectura del dataset RICO

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
4. guardar cada nodo usando `componentLabel` si existe, `class` si no

---

## 5.2. Problema inicial: `Invalid screen size from root bounds`

Al ejecutar el builder apareció este error en algunos archivos:

```python
ValueError: Invalid screen size from root bounds.
```

El script asumía que el root siempre venía como `[x0, y0, x1, y1]`, pero algunos archivos tenían bounds dañados, vacíos o con valores imposibles.

### Qué hicimos

Se reforzó el parser para:

1. intentar normalizar bounds (interpretar como `[x0,y0,x1,y1]` o como `[x,y,w,h]`)
2. inferir el tamaño de pantalla desde el árbol completo cuando el root no era válido
3. saltarse el archivo y registrar warning si no había forma de inferirlo

### Resultado real del dataset

* **66 195 screens** se pudieron procesar correctamente
* **66 archivos** no pudieron parsearse (geometría inválida — condición real del dataset)

---

## 5.3. Estadísticas del dataset y elección de `M`

`M` es el número máximo de elementos por pantalla — define la longitud fija del tensor. Pantallas con menos de `M` elementos → `PAD`. Con más → truncar.

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

Esto captura el 95% de las pantallas sin truncamiento y evita que outliers (hasta 423 elementos) inflen el tensor. Es un cambio importante respecto al `M=25` fijo del blueprint original.

---

## 5.4. Split del dataset

Se dividieron las **pantallas válidas** (las 66 195) en:

| Split | Tamaño |
|---|---|
| train | 52 956 |
| val | 6 619 |
| test | 6 620 |

Proporciones: 80 / 10 / 10, reproducible con `SEED = 42`.

**Importante**: el split se hizo sobre `len(screens)` (pantallas parseables), no sobre `len(json_files)` (total de archivos). Esa distinción fue clave para evitar el bug de índices descrito en la sección 7.

---

## 5.5. Construcción del vocabulario de categorías

A partir del split train se creó `cat2id` usando **solo train**:

Resultado real: **25 categorías** en train, con ids especiales `mask_id = C` y `pad_id = C + 1`.

---

## 5.6. Discretización geométrica con KMeans

Se entrenaron 4 KMeans independientes sobre el split de train, con `BINS = 64` clusters cada uno.

Con 64 bins por modalidad:

* `x_id, y_id, w_id, h_id` quedan en `0..63`
* `mask_id = 64`, `pad_id = 65`
* vocabulario por modalidad geométrica: `64 + 2 = 66`

---

## 5.7. Exportación de artefactos

| Archivo | Contenido |
|---|---|
| `tokens_train.pt` | `LongTensor [52956, 55, 5]` |
| `tokens_val.pt` | `LongTensor [6619, 55, 5]` |
| `tokens_test.pt` | `LongTensor [6620, 55, 5]` |
| `centroids_x/y/w/h.pt` | `FloatTensor [64]` por modalidad |
| `cat2id.json` | `{ "Button": 0, ... }` |
| `vocab_meta.json` | vocab sizes, pad/mask ids, M, bins, seed, split ratios |
| `split_ids.json` | IDs reales por split (train/val/test) |

El formato de `vocab_meta.json` es el contrato entre preprocesamiento y training loop.

---

# 6. Validación del preprocesamiento

## 6.1. Validación estadística

* shape de `train_tokens`: `(52956, 55, 5)` ✅
* dtype: `torch.int64` ✅
* rangos de ids dentro de los vocabs esperados ✅
* consistencia del padding ✅

## 6.2. Validación visual

Se generaron overlays sobre screenshots reales, dibujando cajas reconstruidas desde los tokens:

1. tomar una pantalla por `screen_id`
2. decodificar sus tokens usando centroides
3. dibujar los bounding boxes sobre la imagen real
4. verificar alineación con los elementos de la UI

### Qué se observó inicialmente

Los layouts parecían mal alineados en varios casos — lo que inicialmente sugería mala normalización, centroides incorrectos o error en el mapeo entre token row y pantalla original.

---

# 7. Depuración del caso concreto: pantalla `"0"`

Se analizó una pantalla con JSON muy simple:

* root: `[0, 0, 1440, 2560]`
* un solo hijo: clase `SystemWebView`, bounds `[0, 0, 1440, 2392]`

**Lo esperado**: 1 elemento, `x=0.5, y=0.467, w=1.0, h=0.934`.

**Lo observado**: `decode_row_to_boxes` devolvía 7 cajas pequeñas concentradas arriba.

### Diagnóstico real

Se reconstruyó el mapeo exacto:

```
idx_in_screens:      0
split_name:          train
pos_in_split_tokens: 44822
real_n (non-pad):    1
```

El problema **no era** la tokenización. Se estaba mirando `train_tokens[0]` cuando el screen `"0"` estaba en `train_tokens[44822]`. Causa: scripts de debug usaban índices basados en `json_files` completos (incluyendo los 66 fallidos), desplazando el índice.

### Lección aprendida

> Nunca asumir que `tokens_train[i]` corresponde al archivo `i.json`.

**Solución implementada**: exportar `split_ids.json` desde `main()` con los IDs reales por split.

---

# 8. Problemas de integración preprocessing ↔ blueprint

Una vez se conectó el dataset real al blueprint del modelo, aparecieron varios problemas. Esto fue bueno: significaba que el pipeline ya estaba ejecutándose de verdad.

## 8.1. `pad_mask` incorrecto

El blueprint asumía un único `pad_id` global, pero el preprocesamiento genera PAD por modalidad:

* `pad_id = C + 1` para categoría
* `pad_id = BINS + 1` para `x/y/w/h`

Solución: construir `pad_mask` iterando solo sobre `["c", "x", "y", "w", "h"]`.

## 8.2. `M` fijo en el blueprint

El modelo usaba `M=25` fijo. Con datos reales, el valor calculado fue `M=55`. Hubo que propagar este valor desde `vocab_meta.json`.

## 8.3. Dataset dummy en el blueprint (iter1) → datos reales (iter2)

El entrenador iter1 incluía una función `load_or_make_dataset` que generaba tokens completamente aleatorios en memoria:

* creaba tensores `[N, M, 5]` con valores enteros al azar dentro del rango de vocabs
* forzaba los últimos 5 slots a `PAD`
* **no tenía ninguna conexión con RICO** — el modelo entrenaba sobre ruido sin sentido

Esto servía para depurar la parte matemática (KL, Qt, Qbar) sin depender de datos externos, pero cualquier generación producida era inválida como evaluación real del modelo.

El entrenador iter2 reemplazó completamente este bloque con:

* `load_real_dataset(data_dir, split)`: carga `tokens_train.pt` / `tokens_val.pt` desde disco con validación de shape
* `load_vocab_meta(data_dir)`: carga `vocab_meta.json` con validación de claves requeridas
* `load_real_datasets(cfg, data_dir)`: orquesta la carga y sobreescribe `cfg.M` desde el JSON para garantizar consistencia con los artefactos reales

A partir de iter2, el modelo entrena sobre los 52 956 layouts reales de RICO.

## 8.4. Flatten no compatible con el paper

El `forward()` organizaba la secuencia por **bloques de modalidad**:

```text
[c1..cM, x1..xM, y1..yM, w1..wM, h1..hM]
```

El Transformer veía primero todas las categorías, luego todas las posiciones `x`, etc. Podía aprender relaciones entre categorías de distintos elementos, pero nunca entre la categoría y la geometría del **mismo** elemento.

El paper espera una secuencia **intercalada por elemento**:

```text
[c1, x1, y1, w1, h1, c2, x2, y2, w2, h2, ...]
```

Así el Transformer entiende que esos 5 tokens pertenecen al mismo elemento y puede modelar coherencia interna (por ejemplo, que un `Button` tenga un cierto rango de anchos y altura).

**La corrección** ajustó la lógica del `forward()` a:

1. embeddings por modalidad: `[B, M, D] × 5`
2. stack: `[B, M, 5, D]`
3. reshape intercalado: `[B, 5M, D]`
4. Transformer sobre la secuencia completa
5. reshape de vuelta: `[B, M, 5, D]`
6. heads por modalidad

Esta fue una corrección estructural fuerte y necesaria.

## 8.5. Shuffle de elementos — validado como correcto

El paper requiere mezclar el orden de los elementos del layout para evitar que el modelo dependa de un orden artificial.

Se comprobó que el shuffle implementado:

* solo mezclaba los elementos reales (nunca PAD),
* mantenía el padding compacto al final,
* no cambiaba el número de elementos válidos,
* no alteraba el contenido de los tokens.

**Resultado**: shuffle correcto — descartado como causa de degeneración.

## 8.6. Errores en la indexación temporal de `Q_t`

Aparecieron varios `IndexError: list index out of range` porque las listas `Qts_all` y `Qbars_all` se indexaban con `t` directamente, cuando están almacenadas desde índice `0`.

Convención correcta:

* `Q_t` → `Qts_all[m][t - 1]`
* `Qbar_t` → `Qbars_all[m][t - 1]`
* `Qbar_{t-1}` → `Qbars_all[m][t - 2]` (si `t == 1`, usar `I`)

## 8.7. Firma incorrecta de `build_Qt`

`mask_id` llegaba como float por un desorden de argumentos posicionales. Se corrigió usando argumentos nombrados.

---

# 9. Entrenamiento exitoso y primera generación

## 9.1. Entrenamiento

Después de corregir los puntos anteriores, el entrenamiento ejecutó end-to-end:

* el modelo sí entrenaba con datos reales
* la loss bajaba sin colapso inmediato
* el pipeline completo corría desde RICO hasta un checkpoint

## 9.2. Primera prueba de generación

Se hizo muestreo incondicional y se renderizaron layouts generados.

### Lo que se observó

**Positivo:**
* categorías plausibles: `Text`, `Image`, `List Item`
* elementos dentro del canvas
* cierta estructura vertical de pantalla móvil

**Problemático:**
* demasiados overlaps
* muchas cajas horizontales largas
* amontonamiento en zonas centrales
* layouts poco naturales

**Conclusión**: como sanity check básico, aceptable. Como calidad final, todavía insuficiente.

## 9.3. Confirmación del problema del flatten

Se renderizaron varios samples y todos repetían el mismo patrón. Eso confirmó que no era una mala muestra aislada sino un problema sistémico.

Al corregir el flatten intercalado, el resultado mejoró ligeramente — pero no fue suficiente por sí solo.

## 9.4. Prueba con `M` reducido

Se probó bajar `M` de `55` a `25`. Los layouts generados mejoraron ligeramente, lo que confirmó que el exceso de padding contribuía al problema. Reducir `M` no fue la solución definitiva, pero simplificó el espacio de error y facilitó observar el efecto de cada corrección posterior.

---

# 10. Diagnóstico actual del proyecto

Esta sección documenta tanto las correcciones estructurales realizadas después de la primera generación como el estado actual del sistema.

## 10.1. Reducción de `M` a 25

Como primera medida exploratoria se redujo `M` de `55` (p95 real del dataset) a `25`. Los layouts generados mejoraron ligeramente — lo que confirmó que el exceso de padding contribuía al problema sin ser la causa principal.

## 10.2. Corrección del flatten intercalado — efecto observado

Después de aplicar la corrección del flatten (sección 8.4):

* desaparecieron algunos patrones repetitivos artificiales,
* mejoró la coherencia entre categoría y geometría del mismo elemento,
* pero la degeneración general del layout persistió.

Conclusión: el flatten era necesario pero no suficiente por sí solo.

## 10.3. Shuffle de elementos — validado como correcto

Se comprobó punto a punto que el shuffle durante el entrenamiento:

* mezclaba solo elementos reales (nunca PAD),
* mantenía el padding compacto al final,
* no alteraba el contenido ni el número de elementos válidos.

**Resultado**: shuffle correcto — descartado como causa de degeneración.

## 10.4. Schedule forward exacto — implementado y validado

Se reemplazó el schedule placeholder por una versión exacta tipo **mask-and-replace** coherente con VQ-Diffusion/LayoutDM:

* `alpha_t`: probabilidad de conservar el token original
* `beta_t`: probabilidad de reemplazar por una clase normal aleatoria
* `gamma_t`: probabilidad de enmascarar con `[MASK]`

Se construyeron matrices `Q_t` válidas por modalidad y se validaron inspeccionando su comportamiento:

| Timestep | `alpha_t` | `gamma_t` | filas de `Q_t` |
|---|---|---|---|
| `t` pequeño | alto | bajo | suman 1 ✅ |
| `t` medio | medio | medio | suman 1 ✅ |
| `t = T` | bajo | alto | suman 1 ✅ |

**Resultado**: schedule matemáticamente consistente con el paper.

## 10.5. Estado después de estas correcciones

Con esto quedan validadas las capas de representación y corrupción:

* flatten intercalado ✅
* shuffle de elementos ✅
* schedule forward exacto ✅
* `M` reducido para depuración ✅

El problema residual ya no apunta a la estructura de datos ni al proceso forward. El sospechoso principal pasa a ser el **reverse sampling**.

## 10.6. Qué ya se puede afirmar

* el dataset real sí está conectado al modelo ✅
* el entrenamiento ya corre con artefactos reales ✅
* el modelo sí aprende algo ✅
* el rendering de samples generados funciona ✅
* flatten intercalado corregido ✅
* shuffle de elementos validado ✅
* schedule forward exacto implementado ✅

## 10.7. Sospechosos actuales de la baja calidad

### a. ~~Reverse sampling mal calibrado~~ — corregido en iter4

En iter3 el sospechoso principal era el sampling inverso: se muestreaba directamente desde `F.softmax(logits[m])` = `p_theta(z_0|z_t)`, ignorando la cadena markoviana. Desde iter4 se usa `compute_theta_posterior` que marginaliza correctamente para obtener `p_theta(z_{t-1}|z_t)` antes de samplear. Este bloqueo está resuelto.

### b. `q_sample()` no validado en la práctica

Aunque el schedule es matemáticamente correcto, no se ha comprobado visualmente que una muestra real se degrade de forma esperada (casi intacta para `t` pequeño, casi toda en `MASK` para `t = T`).

### c. RICO sin filtrado estructural mete ruido

Tomar todos los elementos sin filtrado mínimo puede empeorar la calidad estructural del dataset de entrenamiento.

### d. Entrenamiento todavía insuficiente

Con todas las correcciones activas, puede que simplemente falten epochs para que el modelo converja.

---

# 11. Aspectos importantes que no se deben olvidar

## 11.1. El dataset real era la prioridad correcta

Antes de métricas, constraints o mejoras visuales, había que hacer que el pipeline leyera datos reales y el render confirmara si el sistema aprendía algo. Ese orden fue correcto.

## 11.2. La compatibilidad entre preprocessing y training no era automática

Aunque ambos lados "parecían" correctos en aislamiento, en la práctica hubo que alinear: `M`, `pad_mask`, `vocab_meta`, `Q_t` y `Qbar_t`, estructura del flatten.

## 11.3. Una loss que baja no garantiza buenos layouts

El entrenamiento puede parecer sano numéricamente y aun así producir layouts malos si la secuencia está mal estructurada, el reverse sampling está mal, o el schedule discreto está aproximado.

## 11.4. Los sanity checks visuales son críticos

Renderizar muestras fue lo que permitió detectar que el problema era estructural y sistémico, no un error de implementación aislado.

## 11.5. La trazabilidad screen_id ↔ token row es obligatoria

Sin ella, puedes diagnosticar mal el pipeline y creer que el modelo falla cuando en realidad estás mirando otro ejemplo.

## 11.6. KMeans introduce cuantización — eso es normal

Un pequeño error entre caja real y reconstruida en un overlay no siempre significa bug.

## 11.7. LayoutDM no entiende semántica profunda

Solo aprende patrones geométricos y categóricos. No sabe qué es "header" o "CTA" como concepto de producto.

---

# 12. Qué implementamos para el entrenamiento

## 12.1. Entrada del modelo

```python
tokens: [B, M, 5]
```

Cada token representa `(category, x, y, w, h)` en forma discreta.

## 12.2. Arquitectura

**Transformer encoder** como denoiser, no autoregresivo. LayoutDM modela el layout completo sin depender de un orden fijo.

## 12.3. Difusión discreta (modality-wise)

Hay una difusión separada por modalidad (`c`, `x`, `y`, `w`, `h`). Esto evita mezclar vocabularios incompatibles. Se usan matrices:

* `Q_t`: distribución de transición en un paso
* `Qbar_t`: distribución acumulada desde `t=0`

## 12.4. Loss

```text
loss_total = vb_loss + lambda_aux * aux_loss    (lambda_aux = 0.1)
```

* **VB loss**: KL entre el posterior verdadero y el predicho por el modelo
* **Aux loss**: cross entropy para reconstruir `z0` — estabiliza entrenamiento

## 12.5. Máscara de PAD

Se construye `pad_mask` para que la loss se calcule **solo sobre tokens válidos**. Si PAD entra en la loss, el modelo aprende una distribución errónea.

## 12.6. Unconditional sampling

En inferencia:

1. iniciar todo en `[MASK]`
2. correr reverse diffusion de `T` hasta `1`
3. en cada paso el modelo predice distribuciones sobre tokens
4. samplear categóricamente
5. obtener `z0`

---

# 13. Explicación separada: Preprocesamiento completo

## Objetivo

Transformar cada pantalla de RICO en una secuencia discreta utilizable por LayoutDM.

## Flujo paso a paso

### Paso 1. Leer cada JSON

Orden alfabético — determinista y reproducible.

### Paso 2. Recorrer el árbol

Pre-order, root + todos los `children` recursivamente.

### Paso 3. Extraer y normalizar bounds

```text
x = (x0 + x1) / 2 / screen_w
y = (y0 + y1) / 2 / screen_h
w = (x1 - x0) / screen_w
h = (y1 - y0) / screen_h
```

La resolución de pantalla se infiere haciendo snapping al candidato RICO más cercano: `720×1280`, `1080×1920`, `1440×2560`.

### Paso 4. Determinar la categoría

Prioridad: `componentLabel` > `class` > `"UNKNOWN"`

### Paso 5. Calcular estadísticas y elegir `M`

`M = ceil(percentil_95)` — resultado real: `M = 55`.

### Paso 6. Construir `good_ids` / `bad_ids`

Filtrar a solo pantallas parseables antes del split. El split opera sobre esta lista.

### Paso 7. Dividir train/val/test

Shuffle reproducible con `SEED = 42`, proporción 80/10/10.

### Paso 8. Construir `cat2id`

Solo desde train. Resultado: 25 categorías.

### Paso 9. Ajustar KMeans para x/y/w/h

Solo desde train, `BINS = 64`. Subsample hasta 2M valores por modalidad para evitar OOM.

### Paso 10. Tokenizar

```python
[c_id, x_id, y_id, w_id, h_id]
```

### Paso 11. Aplicar padding hasta `M`

Posiciones vacías → `pad_id` en todas las modalidades.

### Paso 12. Exportar artefactos

Tokens, centroides, `cat2id.json`, `vocab_meta.json`, `split_ids.json`.

## Resultado

```python
tokens_train: LongTensor [52956, 55, 5]
```

## Señal de que el preprocesamiento está bien

Si renderizas `tokens_val.pt` reales y los layouts se ven plausibles, el preprocessing está bien encaminado.

---

# 14. Implementación del entrenador: `layoutdm_trainer.py`

Esta sección documenta el código del entrenador, explica cada componente, las decisiones de diseño tomadas y los puntos que aún requieren corrección.

---

## 14.0. Iter1 → Iter2 → Iter3: evolución del entrenador

El entrenador tiene tres iteraciones. Cada una resolvió un conjunto distinto de problemas:

| | Iter1 (blueprint inicial) | Iter2 | Iter3 | Iter4 (versión actual) |
|---|---|---|---|---|
| **Datos** | Dataset dummy — tokens aleatorios en memoria | Datos reales de RICO (`tokens_*.pt`) | Sin cambios | Sin cambios |
| **`pad_mask`** | Un único `pad_id` global | `pad_id` por modalidad con broadcast | Sin cambios | Sin cambios |
| **`M`** | Hardcodeado a `25` | Leído desde `vocab_meta.json` | Sin cambios | Sin cambios |
| **Shuffle** | No implementado | `_shuffle_valid_elements()` | Sin cambios | Sin cambios |
| **Decode / render** | No incluido | `decode_layout()` + `render_layout()` + grid de validación | Sin cambios | Sin cambios |
| **Checkpoint** | Solo `model_state` | `model_state` + `cfg.__dict__` + `vocab_meta` | Sin cambios | Sin cambios |
| **`forward()` flatten** | Por bloques `[c1..cM, x1..xM,...]` (bug) | Por bloques (bug persistía) | **Intercalado** `[c1,x1,y1,w1,h1,...]` ✅ | Sin cambios |
| **`categorical_sample`** | Stub vacío | Stub vacío | **Implementada** ✅ | Sin cambios |
| **`compute_theta_posterior`** | No existía | No existía | No existía | **Implementada** (Eq. 3 LayoutDM) ✅ |
| **`compute_losses` — p_model** | Stub | Stub | `F.softmax(logits)` directo (bug: distribución sobre `z_0`, no sobre `z_{t-1}`) | `compute_theta_posterior(...)` — distribución sobre `z_{t-1}` ✅ |
| **`train_one_epoch`** | Stub vacío | Stub vacío | **Implementada** con logging cada 50 pasos ✅ | Sin cambios |
| **`unconditional_sample`** — sampling | Stub vacío | Stub vacío | `F.softmax` directo (muestreo desde `p(z_0\|z_t)`, incorrecto) | `compute_theta_posterior` → `torch.multinomial` ✅ |
| **Caso borde `t=1`** | No manejado | No manejado | `Qbars_prev = torch.eye(V)` explícito ✅ | Sin cambios |
| **`precompute_Q_mats`** | Dicts `{t: Qt}` | Dicts `{t: Qt}` | **Listas** `[Qt_1..Qt_T]` con offset `[t-1]` ✅ | Sin cambios |
| **`main()` retorno** | — | `(model, cfg, train_ds, val_ds, vocab_meta)` | `(model, cfg, train_ds, val_ds, vocab_meta)` | **Añade `Qts_all, Qbars_all`** para reusar en celdas interactivas ✅ |

**Iter1**: válido solo para comprobar que las matemáticas del proceso de difusión no crashearan. Cualquier generación era ruido sobre ruido — no tenía ninguna conexión con RICO.

**Iter2**: conecta el entrenador con artefactos reales. El pipeline completo ya corría, pero las funciones críticas (`forward`, `compute_losses`, `train_one_epoch`, `unconditional_sample`) seguían siendo stubs vacíos.

**Iter3**: implementa todas las funciones pendientes. Es la primera versión del entrenador que entrena y genera layouts de forma funcional. Sin embargo, tanto `compute_losses` como `unconditional_sample` cometían el mismo error: usaban `F.softmax(logits)` directamente como modelo de `p_theta(z_{t-1}|z_t)`, cuando en realidad eso es `p_theta(z_0|z_t)` — distribuciones sobre vocabularios distintos en el KL y sampling incorrecto en la cadena inversa.

**Iter4**: corrige el error de distribución en ambas funciones añadiendo `compute_theta_posterior`, que implementa la marginalización VQ-Diffusion (LayoutDM Eq. 3). Es la primera versión con el sampling inverso matemáticamente correcto.

---

## 14.1. Filosofía general del diseño

El entrenador está pensado para **claridad sobre velocidad**. Cada componente es independiente y razonable por sí solo — fue diseñado para poder depurar cada pieza en aislamiento y verificar que las matemáticas fueran correctas antes de preocuparse por eficiencia.

Las restricciones que guiaron el diseño:

* seguir lo más fielmente posible las ecuaciones del paper de LayoutDM
* no usar ninguna abstracción que oscureciera la matemática del proceso de difusión
* que cualquier fallo fuera observable directamente con un `print` o una inspección visual

---

## 14.2. `TrainConfig` — configuración global

```python
@dataclass
class TrainConfig:
    T: int = 100           # pasos de difusión
    lambda_aux: float = 0.1
    lr: float = 5e-4
    n_layers: int = 4
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    M: int = 25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

**Por qué `@dataclass`**: permite construir el config como objeto tipado, pasarlo a funciones y serializarlo en el checkpoint sin fricción (`cfg.__dict__`).

**`T = 100`**: valor exacto del paper. El schedule va de `t=1` (corrupción mínima) a `t=T` (casi todo enmascarado).

**`lambda_aux = 0.1`**: peso de la pérdida auxiliar, tomado directamente de la Ecuación 4 del paper. Sin esta pérdida el entrenamiento es inestable.

**`M = 25`**: valor de inicio para depuración. El valor real calculado desde RICO es `M = 55` (p95). Se usó `25` para reducir el espacio de error durante la fase de depuración inicial.

---

## 14.3. `LayoutTokenDataset` — acceso a los datos

```python
class LayoutTokenDataset(Dataset):
    def __init__(self, token_tensors, vocab_meta, shuffle_elements=False):
        self.pad_ids = torch.tensor(
            [vocab_meta[m]["pad_id"] for m in self.MODALITIES],
            dtype=torch.long
        )  # [5]

    def _build_pad_mask(self, tokens):
        return tokens.eq(self.pad_ids)   # broadcast [M,5] == [5] → [M,5]

    def __getitem__(self, idx):
        tokens = self.x[idx].clone()
        if self.shuffle_elements:
            tokens = self._shuffle_valid_elements(tokens)
        pad_mask = self._build_pad_mask(tokens)
        return tokens, pad_mask
```

**Por qué `pad_mask` se calcula aquí**: se necesita saber qué posiciones son PAD en todas las funciones de pérdida. Si se calculara en otro sitio habría riesgo de inconsistencia. Al generarlo en el Dataset, siempre está sincronizado con los tokens.

**`pad_ids` por modalidad**: el preprocesamiento usa `pad_id` distinto por modalidad (`C+1` para categoría, `BINS+1` para geometría). Por eso `pad_ids` es un tensor `[5]` construido desde `vocab_meta` — al hacer `tokens.eq(self.pad_ids)`, PyTorch hace broadcast `[M,5] == [5]` y produce el `pad_mask [M,5]` correcto por columna.

---

## 14.4. `make_transition_params` y `build_Qt` — el schedule de corrupción

### `make_transition_params`

```python
def make_transition_params(t: int, T: int):
    s = t / T
    gamma_t = math.sin(s * math.pi / 2) ** 2   # 0..1 creciente
    beta_t  = 0.01 * (1.0 - gamma_t)           # pequeño
    alpha_t = 1.0 - gamma_t
    return alpha_t, beta_t, gamma_t
```

Define tres probabilidades para el proceso de corrupción de un token en el paso `t`:

| Parámetro | Significado |
|---|---|
| `alpha_t` | probabilidad de **conservar** el token original |
| `beta_t` | probabilidad de **reemplazar** por otra clase normal (uniforme) |
| `gamma_t` | probabilidad de **enmascarar** con `[MASK]` |

`gamma_t` crece suavemente de 0 a 1 usando una curva seno: a `t=0` nada se enmascara; a `t=T` casi todo está en `[MASK]`. Esta forma viene de los schedules cosine / sine usados en VQ-Diffusion y LayoutDM.

**Nota**: esta implementación es una aproximación razonablemente fiel pero no idéntica al paper. El schedule exacto requiere precalcular los arrays `alpha_bar_t` acumulados de forma distinta. Esto es uno de los puntos pendientes de validación.

### `build_Qt`

```python
def build_Qt(V, alpha_t, beta_t, gamma_t, mask_id, device):
    Qt = torch.zeros((V, V), ...)
    for i in normal_ids:
        Qt[i, i]       = alpha_t + beta_t        # prob de quedarse
        Qt[i, j!=i]    = beta_t                  # prob de ir a otra clase
        Qt[i, mask_id] = gamma_t
        Qt[i] = Qt[i] / Qt[i].sum()             # renormalizar por fila
    Qt[mask_id, mask_id] = 1.0                   # MASK es absorbente
    return Qt
```

Construye la matriz de transición `Q_t[V, V]` para una modalidad. Cada fila es una distribución categórica que dice: "si estoy en el token `i`, ¿con qué probabilidad voy a cada token en el siguiente paso de corrupción?".

**Por qué `MASK` es absorbente**: una vez un token cae en `[MASK]`, no vuelve a ser normal. Esto permite el proceso de revelación gradual en el sampling reverso — el modelo aprende a revelar tokens de `[MASK]` hacia tokens válidos.

**Por qué se renormaliza por fila**: los parámetros `alpha + beta + gamma` no siempre suman exactamente 1 por restricciones numéricas; la renormalización garantiza que cada fila sea una distribución válida.

**Por qué una `Qt` por modalidad y no una sola global**: cada modalidad tiene un vocabulario distinto (categoría tiene `C+2` tokens; geometría tiene `BINS+2`). Mezclar vocabularios produciría índices sin sentido.

---

## 14.5. `precompute_Q_mats` — precómputo de las matrices de transición

```python
def precompute_Q_mats(cfg, vocab_meta, device):
    Qts_all   = {}   # Qts_all[m]   = list [Qt_1, Qt_2, ..., Qt_T]
    Qbars_all = {}   # Qbars_all[m] = list [Qbar_1, Qbar_2, ..., Qbar_T]

    for m in ["c", "x", "y", "w", "h"]:
        Qts, Qbars = [], []
        Qbar = torch.eye(V, device=device)     # acumulador, empieza en I

        for t in range(1, cfg.T + 1):
            alpha_t, beta_t, gamma_t = make_transition_params(t, cfg.T)
            Qt = build_Qt(V=V, ..., mask_id=mask_id, device=device)
            Qbar = Qt @ Qbar                  # acumulación izquierda
            Qts.append(Qt)
            Qbars.append(Qbar.clone())

        Qts_all[m]   = Qts    # índice con [t-1] para acceder a Q_t
        Qbars_all[m] = Qbars  # índice con [t-1] para Qbar_t, [t-2] para Qbar_{t-1}
```

**Por qué precalcular**: `build_Qt` y la multiplicación matricial son costosas. Si se calcularan en cada paso del training loop, el cuello de botella estaría en el CPU/setup, no en el modelo. Precalcular una sola vez al inicio tiene un costo de memoria aceptable (T × 5 matrices).

**Listas en lugar de dicts (iter3)**: la versión anterior usaba `Qts_all[m][t]` con clave entera de diccionario. Iter3 usa listas con offset: `Qts_all[m][t-1]` para `Q_t`, `Qbars_all[m][t-1]` para `Qbar_t`, `Qbars_all[m][t-2]` para `Qbar_{t-1}`. El caso borde `t=1` (que daría índice `[-1]`) se maneja explícitamente con `torch.eye(V)` en `train_one_epoch`.

**`Qbar_t = Q_t @ Qbar_{t-1}`**: la acumulación izquierda es la forma correcta de obtener la distribución marginal `q(z_t | z_0)` en un solo paso desde `z_0`:

$$\bar{Q}_t = Q_t \cdot Q_{t-1} \cdot \ldots \cdot Q_1$$

---

## 14.6. `q_sample_from_Qbar` — corrupción forward

```python
def q_sample_from_Qbar(z0, Qbar_t):
    probs = Qbar_t[z0]          # [B, L, V]  — fila de Qbar correspondiente a cada token
    return categorical_sample(probs)
```

Dado `z_0` (tokens limpios) y `Qbar_t` (distribución acumulada hasta `t`), muestrea `z_t ~ q(z_t | z_0)`.

**Por qué `Qbar_t[z0]`**: indexar una matriz con un tensor de índices es un lookup eficiente en PyTorch. Cada posición de `z0` selecciona la fila de `Qbar_t` correspondiente a ese token, que es exactamente su distribución de transición acumulada.

**Esta función es el punto pendiente más importante de validar**: si `Qbar_t` no está bien calculada, toda la corrupción forward es incorrecta y el modelo aprende una tarea que no corresponde al paper.

---

## 14.7. `q_posterior_true` — el posterior verdadero

```python
def q_posterior_true(z0, zt, Qt, Qbar_t_1):
    probs_t1_given_z0 = Qbar_t_1[z0]                       # [B,L,V]
    Qt_col = Qt[:, zt].permute(1, 2, 0)                    # [B,L,V]
    unnorm = Qt_col * probs_t1_given_z0
    return unnorm / (unnorm.sum(dim=-1, keepdim=True) + 1e-12)
```

Calcula la distribución verdadera:

$$q(z_{t-1} \mid z_t, z_0) \propto q(z_t \mid z_{t-1}) \cdot q(z_{t-1} \mid z_0)$$

**`Qt_col = Qt[:, zt]`**: extrae la columna `zt` de `Qt`, es decir, la probabilidad `q(z_t | z_{t-1})` para cada candidato `z_{t-1}`. Se hace indexando por columna y luego permutando a `[B, L, V]`.

**Por qué esta fórmula**: en difusión discreta el posterior verdadero es tratable analíticamente gracias a la estructura markoviana del proceso forward. Esta fórmula proviene directamente de la regla de Bayes y es la que el modelo debe aprender a aproximar.

**`+ 1e-12`**: previene división por cero en posiciones donde todos los candidatos tienen probabilidad cero (raro pero posible con tokens fuera de distribución).

---

## 14.7b. `compute_theta_posterior` — la marginalización VQ-Diffusion (añadida en iter4)

```python
def compute_theta_posterior(
    p_theta_z0: torch.Tensor,  # [B, M, V]  p(z_0 | z_t) del modelo
    zt_m: torch.Tensor,        # [B, M]     tokens ruidosos en el paso t
    Qt: torch.Tensor,          # [V, V]     matriz de transición Qt
    Qbar_t1: torch.Tensor,     # [V, V]     Qbar_{t-1} acumulada
) -> torch.Tensor:             # [B, M, V]  p_theta(z_{t-1} | z_t)
    p_z1_marginal = torch.matmul(p_theta_z0, Qbar_t1)       # [B, M, V]
    Qt_col = Qt[:, zt_m].permute(1, 2, 0).contiguous()      # [B, M, V]
    unnorm = Qt_col * p_z1_marginal                          # [B, M, V]
    return unnorm / (unnorm.sum(dim=-1, keepdim=True) + 1e-12)
```

Implementa la fórmula central de VQ-Diffusion / LayoutDM (Eq. 3):

$$p_\theta(z_{t-1}=v \mid z_t=s) \propto Q_t[v, s] \cdot \sum_u \left[ p_\theta(z_0=u \mid z_t) \cdot \bar{Q}_{t-1}[u, v] \right]$$

**Por qué esta función era el bug principal de iter3**: tanto `compute_losses` como `unconditional_sample` usaban `F.softmax(logits[m])` directamente como representación del modelo. Eso es `p_theta(z_0 | z_t)` — una distribución sobre el vocabulario de `z_0`. Al calcular KL contra `q_posterior_true(z0, zt)` (que es una distribución sobre `z_{t-1}`), los vectores tenían la misma dimensión de vocabulario pero semántica completamente distinta. Formalmente, el KL no tenía sentido matemático.

**`p_z1_marginal = p_theta_z0 @ Qbar_t1`**: marginaliza sobre `z_0`:  
$$\sum_u p_\theta(z_0=u \mid z_t) \cdot \bar{Q}_{t-1}[u, v]$$  
Convierte la distribución sobre `z_0` en una distribución sobre `z_{t-1}` ponderada por la probabilidad de transición acumulada.

**`Qt_col = Qt[:, zt_m].permute(1,2,0)`**: para cada posición, extrae la columna `zt_m` de `Qt`, es decir `Q_t[v, z_t]` — la probabilidad de que el token en `t` fuera `z_t` viniendo de `v` en `t-1`. Idéntico al indexado de `q_posterior_true`, pero aquí sobre la distribución del modelo.

**Usada en dos lugares**:
- `compute_losses`: como `p_model` en el KL — ahora comparamos `q(z_{t-1}|z_t,z_0)` vs `p_theta(z_{t-1}|z_t)`, ambas sobre `z_{t-1}`. ✅
- `unconditional_sample`: como distribución de muestreo en cada paso de denoising — el modelo ahora recorre correctamente `p_theta(z_{t-1}|z_t)`. ✅

---

## 14.8. `LayoutDMDenoiser` — el modelo Transformer

### Por qué Transformer encoder (no decoder)

LayoutDM modela el layout completo en paralelo, no elemento a elemento. El Transformer encoder aplica self-attention sobre toda la secuencia simultáneamente, sin máscara causal. Esto permite que la categoría de un elemento influya en la geometría de otro elemento en la misma pasada.

Un decoder autoregresivo generaría un token de cada vez (izquierda a derecha), lo cual introduciría un orden artificial en un problema donde el orden de los elementos no debería importar.

### Embeddings por modalidad

```python
self.emb = nn.ModuleDict({
    m: nn.Embedding(vocab_sizes[m], cfg.d_model) for m in self.modalities
})
```

Cada modalidad tiene su propio embedding porque los vocabularios son distintos e incompatibles. Un vocabulario compartido mezclaría `c_id=3` (categoría "Button") con `x_id=3` (bin de posición horizontal cercano al margen izquierdo), que no tienen ninguna relación semántica.

### Positional encodings desacoplados

```python
self.elem_pos = nn.Embedding(cfg.M, cfg.d_model)   # posición del elemento
self.attr_pos = nn.Embedding(5, cfg.d_model)        # posición del atributo
```

Se usan dos tipos de embeddings posicionales sumados:

* `elem_pos`: indica qué elemento es (primero, segundo, ..., M-ésimo)
* `attr_pos`: indica qué atributo es (c=0, x=1, y=2, w=3, h=4)

Sin positional encoding, el Transformer no puede distinguir si un token pertenece al elemento 5 o al elemento 12. Sin `attr_pos`, no sabría si está procesando una coordenada `x` o un ancho `w`.

### El `forward()` — flatten intercalado (corregido en iter3)

En iter2 el `forward()` era un stub vacío. En iter3 está completamente implementado con el flatten correcto:

```python
reps = []
for a, m in enumerate(self.modalities):
    tok = zt[:, :, a]                                    # [B, M]
    e = self.emb[m](tok)                                 # [B, M, D]
    attr_ids = torch.full((B, M), a, device=device)
    e = e + self.elem_pos(elem_ids) + self.attr_pos(attr_ids)  # [B, M, D]
    reps.append(e)

# FIX: intercalado por elemento, no por bloques de modalidad
mods = torch.stack(reps, dim=2)          # [B, M, 5, D]
seq  = mods.reshape(B, M * 5, cfg.d_model)  # [B, 5M, D]

h = self.encoder(seq)                    # [B, 5M, D]
h = h.reshape(B, M, 5, cfg.d_model)     # [B, M, 5, D]

out = {
    "c": self.head["c"](h[:, :, 0, :]),  # [B, M, Vc]
    "x": self.head["x"](h[:, :, 1, :]),
    "y": self.head["y"](h[:, :, 2, :]),
    "w": self.head["w"](h[:, :, 3, :]),
    "h": self.head["h"](h[:, :, 4, :]),
}
```

**Por qué `torch.stack(..., dim=2).reshape`** y no `torch.cat(..., dim=1)`: la versión `cat` habría producido la secuencia por bloques `[c1..cM, x1..xM, ...]`. El Transformer habría podido aprender relaciones entre la categoría del elemento 3 y la del elemento 7, pero **no** entre la categoría del elemento 3 y su propia geometría. Con `stack + reshape`, la secuencia queda intercalada:

```
[c1, x1, y1, w1, h1, c2, x2, y2, w2, h2, ...]
```

Así el Transformer puede atender a todos los atributos del mismo elemento en la misma ventana de atención, modelando coherencia interna (`Button` → rango de anchos plausible).

**Extracción de outputs**: `h.reshape(B, M, 5, D)` deshace el intercalado. Luego `h[:, :, a, :]` extrae la representación del atributo `a` para todos los elementos y pasa por el head de esa modalidad.

### Por qué `norm_first=True` en el TransformerEncoderLayer

La variante `norm_first` (Pre-LN) aplica LayerNorm antes de cada sublayer en lugar de después. Es más estable durante el entrenamiento — los gradientes fluyen mejor hacia las capas inferiores. LayoutDM y la mayoría de Transformers modernos lo usan por defecto.

---

## 14.9. `compute_losses` — la función de pérdida

```python
def compute_losses(cfg, logits, z0, zt, t, Qts_t, Qbars_prev, pad_mask):
    for a, m in enumerate(modalities):
        valid = (~pad_mask[:, :, a]).float()       # 1.0 donde no hay PAD
        denom = valid.sum().clamp_min(1.0)         # evitar div/0

        q_true = q_posterior_true(z0, zt, Qts_t[m], Qbars_prev[m])
        p_model = F.softmax(logits[m], dim=-1)

        kl = kl_categorical(q_true, p_model)       # [B,M]
        vb = (kl * valid).sum() / denom

        ce = F.cross_entropy(logits[m].reshape(-1, V), z0_m.reshape(-1), reduction="none")
        aux = (ce * valid).sum() / denom

    total = vb_loss + cfg.lambda_aux * aux_loss
```

### VB loss (variational bound)

Minimiza el KL entre el posterior verdadero `q(z_{t-1} | z_t, z_0)` y la distribución predicha por el modelo `p_θ(z_{t-1} | z_t)`. Este es el objetivo principal del paper — el modelo aprende a aproximar la reversión del proceso forward.

**Corrección iter4**: en iter3, `p_model = F.softmax(logits[m])` era `p_theta(z_0 | z_t)`, no `p_theta(z_{t-1} | z_t)`. El KL estaba calculando divergencia entre distribuciones que se referían a variables distintas (`z_{t-1}` vs `z_0`), aunque compartieran dimensión de vocabulario. Desde iter4, `p_model = compute_theta_posterior(p_theta_z0, zt_m, Qt, Qbar_prev)`, que produce correctamente `p_theta(z_{t-1} | z_t)`. 

### Aux loss (pérdida auxiliar)

Cross-entropy para reconstruir directamente `z_0` desde los logits. No está fundamentada en la formulación probabilística estricta, pero aporta una señal de entrenamiento más directa que estabiliza el aprendizaje, especialmente en los primeros pasos cuando el modelo no ha aprendido nada todavía.

### Por qué dividir por `valid.sum()` y no por `B*M`

La pérdida debe calcularse **solo sobre tokens que no son PAD**. Si un layout tiene 5 elementos reales y `M-5 = 20` posiciones de PAD, los 20 PADs no deberían contribuir. Dividir por `B*M` daría una señal diluida que favorece a los layouts cortos. Dividir por `valid.sum()` normaliza correctamente.

### Por qué `clamp_min(1.0)` en `denom`

Si por alguna razón todos los tokens de un batch son PAD (no debería ocurrir, pero es defensa), `denom = 0` produciría NaN. El `clamp_min(1.0)` evita esa situación.

---

## 14.10. `train_one_epoch` — el loop de entrenamiento

```python
for tokens, pad_mask in loader:
    tokens   = tokens.to(device)
    pad_mask = pad_mask.to(device)

    t = torch.randint(1, cfg.T + 1, (1,), device=device).item()
    if not (1 <= t <= cfg.T):
        raise ValueError(f"Sampled invalid timestep t={t}")

    zt = tokens.clone()
    Qts_t, Qbars_prev = {}, {}

    for a, m in enumerate(["c", "x", "y", "w", "h"]):
        Qbar_t           = Qbars_all[m][t - 1]          # Qbar_t
        zt[:, :, a]      = q_sample_from_Qbar(tokens[:, :, a], Qbar_t)
        Qts_t[m]         = Qts_all[m][t - 1]            # Qt
        if t == 1:
            Qbars_prev[m] = torch.eye(V, device=device) # Qbar_{t-1} = I
        else:
            Qbars_prev[m] = Qbars_all[m][t - 2]         # Qbar_{t-1}

    logits = model(zt)
    loss, metrics = compute_losses(cfg, logits, tokens, zt, t, Qts_t, Qbars_prev, pad_mask)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    step += 1
    if step % 50 == 0:
        print(f"step={step:04d} t={t:03d} loss={metrics['loss_total']:.4f} "
              f"vb={metrics['loss_vb']:.4f} aux={metrics['loss_aux']:.4f}")
```

**Un único `t` por batch**: samplear un solo timestep para todo el batch es la práctica estándar en difusión. Equivale a hacer un estimador Monte Carlo del objetivo ELBO — en expectativa sobre todos los batches se cubren todos los timesteps.

**Caso borde `t=1`**: cuando `t==1`, `Qbar_{t-1}` debería ser la identidad (`q(z_0|z_0)=I`). Acceder a `Qbars_all[m][-1]` con `t-2=-1` en Python devolvería el último elemento de la lista silenciosamente. El manejo explícito con `torch.eye` evita este bug.

**Validación del timestep**: el `raise ValueError` detecta off-by-one antes de que propaguen errores oscuros en las matrices de transición.

**`set_to_none=True` en `zero_grad`**: más eficiente — libera memoria en lugar de sobreescribir con ceros.

**Logging cada 50 pasos**: `loss_total`, `loss_vb` y `loss_aux` desagregados permiten diagnosticar si el KL o la auxiliary dominan la señal.

---

## 14.11. `unconditional_sample` — el sampling reverso

```python
# Inicializar z_T con [MASK] para todas las modalidades
zt = torch.zeros((B, M, 5), dtype=torch.long, device=device)
for a, m in enumerate(modalities):
    zt[:, :, a] = mask_ids[m]

for t in range(cfg.T, 0, -1):
    logits = model(zt)   # m -> [B, M, V_m]  predice p(z_0 | z_t)
    z_prev = zt.clone()

    for a, m in enumerate(modalities):
        p_theta_z0 = F.softmax(logits[m], dim=-1)          # [B, M, V]  p(z_0 | z_t)

        Qt = Qts_all[m][t - 1]
        if t == 1:
            Qbar_prev = torch.eye(vocab_meta[m]["vocab_size"], device=device)
        else:
            Qbar_prev = Qbars_all[m][t - 2]

        # Posterior p_theta(z_{t-1}|z_t) via VQ-Diffusion Eq. 3
        probs = compute_theta_posterior(p_theta_z0, zt[:, :, a], Qt, Qbar_prev)  # [B, M, V]

        flat = probs.reshape(-1, probs.size(-1))
        sampled = torch.multinomial(flat, 1).squeeze(-1).view(B, M)
        z_prev[:, :, a] = sampled

    # Regla de coherencia PAD: si c==PAD, forzar x/y/w/h a PAD
    c_pad = (z_prev[:, :, 0] == pad_ids["c"])
    if c_pad.any():
        z_prev[:, :, 1] = torch.where(c_pad, torch.full_like(z_prev[:, :, 1], pad_ids["x"]), z_prev[:, :, 1])
        z_prev[:, :, 2] = torch.where(c_pad, torch.full_like(z_prev[:, :, 2], pad_ids["y"]), z_prev[:, :, 2])
        z_prev[:, :, 3] = torch.where(c_pad, torch.full_like(z_prev[:, :, 3], pad_ids["w"]), z_prev[:, :, 3])
        z_prev[:, :, 4] = torch.where(c_pad, torch.full_like(z_prev[:, :, 4], pad_ids["h"]), z_prev[:, :, 4])

    zt = z_prev

return zt  # z_0
```

**Por qué iniciar en `[MASK]`**: el proceso forward lleva `z_0 → z_T` donde `z_T` es casi todo `[MASK]`. El sampling reverso debe partir de ese mismo estado.

**Corrección iter4 — sampling via `compute_theta_posterior`**: en iter3 se samplea directamente desde `F.softmax(logits[m])`, que es `p_theta(z_0 | z_t)`. En la cadena de denoising el modelo genera `z_{t-1}`, no `z_0` directamente. Usar `p(z_0|z_t)` como distribución de muestreo equivale a saltar directamente del paso `t` al paso `0` ignorando la estructura de la cadena markoviana. Desde iter4 se usa `compute_theta_posterior`, que marginaliza correctamente para obtener `p_theta(z_{t-1} | z_t)` antes de muestrear.

**`torch.where` por atributo** (no `z_prev[:, :, 1:]` en bloque): la regla de coherencia se aplica atributo a atributo porque cada modalidad tiene su propio `pad_id`. Un `torch.where` sobre un slice `[:, :, 1:]` mezclaría los `pad_id` de `x`, `y`, `w` y `h` con el mismo valor, lo que sería incorrecto si `BINS` cambia entre modalidades.

---

## 14.12. Resumen de decisiones de diseño

| Decisión | Razón |
|---|---|
| Transformer encoder (no decoder) | El layout es un conjunto no ordenado — no existe un orden izquierda-derecha natural |
| Difusión por modalidad (no conjunta) | Cada modalidad tiene vocabulario propio — mezclarlos produciría índices sin sentido |
| `Qbar_t` precalculada | Costosa de calcular en cada step; estable en memoria para T=100 |
| Listas con offset `[t-1]` en lugar de dict `{t: Qt}` | Más idiomático en Python; el caso borde `t=1` se maneja explícitamente con `torch.eye` |
| Flatten `stack + reshape` intercalado | Permite al Transformer atender a categoría y geometría del mismo elemento |
| `compute_theta_posterior` para `p_model` y sampling | Transforma `p_theta(z_0\|z_t)` en `p_theta(z_{t-1}\|z_t)` — distribución correcta para KL y denoising |
| Aux loss con CE | Señal más directa que estabiliza el entrenamiento, peso pequeño `lambda=0.1` |
| `valid` mask en la pérdida | PAD no debe contribuir a la señal de gradiente |
| Regla de coherencia PAD en sampling | Evita generar geometría en posiciones vacías |
| `torch.where` por atributo en coherencia PAD | Cada modalidad tiene su propio `pad_id` — un slice en bloque usaría el valor equivocado |
| `main()` retorna `Qts_all, Qbars_all` | Permite reusar las matrices precalculadas en celdas Colab posteriores sin recomputar |

## 14.13. Qué ya funciona (iter4)

* carga `tokens_*.pt` reales desde disco ✅
* lee `vocab_meta.json` y usa `M` real desde artefactos ✅
* `pad_mask` correcto por modalidad ✅
* `_shuffle_valid_elements`: shuffle solo de elementos reales, PAD tail intacto ✅
* positional encodings desacoplados (`elem_pos` + `attr_pos`) ✅
* `forward()` con flatten intercalado `stack([c,x,y,w,h], dim=2).reshape(B, 5M, D)` ✅
* `compute_theta_posterior`: marginalización VQ-Diffusion/LayoutDM Eq. 3 ✅
* `compute_losses` completa con VB (KL) correcto entre `q(z_{t-1}|z_t,z_0)` y `p_theta(z_{t-1}|z_t)`, aux (CE), máscara `valid`, métricas desagregadas ✅
* `categorical_sample` implementada ✅
* `unconditional_sample` implementada con posterior correcto `compute_theta_posterior → torch.multinomial` ✅
* regla de coherencia PAD en sampling (`torch.where` por atributo) ✅
* `train_one_epoch` completa con logging cada 50 pasos ✅
* caso borde `t=1` manejado con `torch.eye(V)` ✅
* schedule matemáticamente consistente ✅
* `Qbar_t` precalculada con listas e indexación con offset ✅
* ejecuta entrenamiento end-to-end sin errores ✅
* guarda checkpoint con `cfg`, `model_state` y `vocab_meta` ✅
* genera muestras renderizables ✅
* `main()` retorna `Qts_all, Qbars_all` para uso interactivo en Colab ✅

## 14.14. Qué sigue pendiente de validar / mejorar

| Componente | Estado |
|---|---|
| `q_sample_from_Qbar` | No validada visualmente — comprobar que la degradación sea gradual: casi intacta para `t` pequeño, casi todo `MASK` para `t=T` |
| Schedule forward | La curva seno es una aproximación razonablemente fiel pero no idéntica al paper — pendiente validar contra implementación oficial de `CyberAgentAILab/layout-dm` |
| Filtrado de RICO | Sin whitelist de categorías ni NMS, el dataset incluye elementos ruidosos que pueden perjudicar la calidad estructural del layout generado |
| Epochs de entrenamiento | Con todas las correcciones activas (iter4), reentrenar una corrida más larga y evaluar visualmente si los layouts mejoran respecto a iter3 |

## Cómo se prueba el resultado

No con accuracy tradicional. Se genera y renderiza. Las preguntas clave:

* ¿las cajas están dentro del canvas?
* ¿los tamaños son plausibles?
* ¿las clases son coherentes?
* ¿hay demasiados overlaps?
* ¿el layout se parece a una UI real?

---

# 15. Estado actual del proyecto


## Ya resuelto

* parser funcional (66 195 / 66 261 pantallas)
* manejo de archivos dañados con warning, sin crash
* `M = 55` calculado desde datos reales (p95); `M = 25` para depuración
* split reproducible (52 956 / 6 619 / 6 620)
* vocabulario de 25 categorías desde train
* KMeans train-only con BINS = 64
* exportación de tokens, centroides, `split_ids.json`
* trazabilidad `screen_id` ↔ token row resuelta
* training loop con datos reales (sin dummy)
* `pad_mask` correcto por modalidad
* flatten intercalado por elemento ✅
* shuffle de elementos validado como correcto ✅
* schedule forward exacto implementado ✅
* entrenamiento e2e funcional
* generación y render básicos

## Pendiente para mayor fidelidad al paper

* validar `q_sample()` visualmente
* auditoría completa del sampling reverso
* reentrenar corrida corta con todas las correcciones activas
* validación cruzada real vs generado sobre `tokens_val.pt`
* comparar métricas con el paper

---

# 16. Próximos pasos recomendados

## Prioridad alta

1. Validar `q_sample()` visualmente — degradar muestras reales de `tokens_val.pt` y confirmar comportamiento esperado: casi intacta para `t` pequeño, parcialmente enmascarada a `t` medio, casi toda en `MASK` para `t = T`
2. Auditar el bloque completo de sampling reverso: `q_sample_from_Qbar()`, `compute_losses()`, `unconditional_sample()`
3. Reentrenar una corrida corta controlada con todas las correcciones activas y comparar visualmente contra resultados anteriores

## Prioridad media

4. Evaluar filtrado estructural mínimo de RICO (eliminar elementos con categoría `UNKNOWN` o de muy baja frecuencia)
5. Comparar layouts generados vs muestras reales de `tokens_val.pt`
6. Ajustar `M` al valor óptimo según calidad visual observada

## Prioridad posterior

7. Entrenar más epochs con el pipeline completamente corregido
8. Conectar LayoutDM con UI-Diffuser para generación visual final

---

# 17. Conclusión

El trabajo logró pasar de una implementación blueprint parcialmente dummy a un **pipeline real completo** con RICO, tokenización discreta, entrenamiento funcional y generación renderizable.

Después de la primera generación se realizó una ronda de correcciones estructurales: se redujo `M` para depurar en un espacio más controlado, se corrigió el flatten intercalado del denoiser, se validó el shuffle de elementos, y se implementó el schedule forward exacto del paper.

Lo más importante que aprendimos:

* el problema no era una sola línea de código sino una combinación de factores: representación de la secuencia, dinámica de corrupción, y posiblemente el sampling reverso
* la compatibilidad entre preprocessing y training **no es automática** — hubo que alinear `M`, `pad_mask`, `vocab_meta`, indexación temporal, flatten y schedule
* una loss que baja **no garantiza buenos layouts** — los sanity checks visuales son indispensables
* la trazabilidad `screen_id` ↔ `token_row` es **obligatoria** para debug correcto
* antes de evaluar calidad del muestreo hay que verificar integridad estructural en toda la cadena

La parte más valiosa es que el sistema ahora opera con un pipeline matemáticamente más sólido. Los siguientes pasos apuntan directamente al reverse sampling, que es el último eslabón aún sin validar a fondo.

---

# 18. Resumen final simple

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
  → tokens.pt [N, 55, 5]  +  split_ids.json
```

### Entrenamiento

```
tokens.pt → pad_mask → diffusion corruption (modality-wise) → Transformer denoiser → VB + aux → reverse sampling
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
# Exportado automáticamente por main() como split_ids.json:
split_screen_ids = {
    "train": [screens[i]["id"] for i in train_idx],
    "val":   [screens[i]["id"] for i in val_idx],
    "test":  [screens[i]["id"] for i in test_idx],
}
with open(os.path.join(OUT_DIR, "split_ids.json"), "w", encoding="utf-8") as f:
    json.dump(split_screen_ids, f, ensure_ascii=False, indent=2)
```

**Por qué esto es crítico**: `load_all_screens()` filtra los JSONs que fallan al parsear. Cualquier código externo que reconstruya el split usando `len(all_json_files)` obtendrá un `n` distinto → shuffle distinto → desalineamiento índice/token.

---

# Anexo C: Correcciones de integración aplicadas

| Problema | Corrección |
|---|---|
| `pad_mask` con un único pad_id | Iterar sobre `["c","x","y","w","h"]` con pad_id por modalidad |
| `M=25` fijo | Leer `M` desde `vocab_meta["M"]` |
| Dataset dummy | Cargar `tokens_train.pt` / `tokens_val.pt` desde disco |
| Flatten por bloques de modalidad | Flatten intercalado por elemento (`[B, 5M, D]`) |
| `Qts_all[m][t]` fuera de rango | Usar `t-1` como índice; para `t==1`, `Qbar_0 = I` |
| Argumento `mask_id` como float | Usar argumentos nombrados en llamada a `build_Qt()` |
| Schedule placeholder no fiel al paper | Implementar mask-and-replace exacto con `alpha_t`, `beta_t`, `gamma_t` y validar filas de `Q_t` |
| Shuffle posiblemente roto | Validado: solo mezcla elementos reales, PAD queda al final |
