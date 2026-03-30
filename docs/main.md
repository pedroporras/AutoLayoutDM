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
* blueprint del modelo
* pipeline inicial de preprocesamiento de RICO

Había una base funcional para arrancar.

## 3.2. Qué faltaba o estaba incompleto

También había brechas importantes:

* el dataset real de RICO no estaba conectado al blueprint (usaba datos dummy)
* no estaba validado que el preprocesamiento fuera compatible con el blueprint
* faltaba el shuffle por elemento
* faltaban sanity checks visuales sólidos
* el schedule del paper no era exacto
* el flatten del modelo no seguía la estructura intercalada del paper
* `M` estaba fijo en `25` en lugar de calcularse desde los datos reales

La conclusión fue clara: antes de optimizar o comparar métricas, había que cerrar bien el pipeline real de datos y entrenamiento.

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

## 8.3. Dataset dummy en el blueprint

El modelo creaba un dataset falso en memoria. Se cambió para leer `tokens_train.pt` y `tokens_val.pt` desde disco.

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

### a. Reverse sampling posiblemente mal calibrado

Es ahora el sospechoso más fuerte. Muchos blueprints simplificados fallan aquí porque:

* usan una aproximación ingenua del posterior `q(z_{t-1} | z_t, z_0)`,
* no reconstruyen bien `z_{t-1}` paso a paso,
* o tratan incorrectamente `PAD` y `MASK` durante el sampling.

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

# 14. Explicación separada: Entrenamiento del modelo

## Arquitectura del denoiser

### 1. Embeddings por modalidad

Cada token discreto se convierte en embedding: `c_embed`, `x_embed`, `y_embed`, `w_embed`, `h_embed`, produciendo tensores `[B, M, D]` por modalidad.

### 2. Positional encodings desacoplados

Se usan dos tipos de embeddings posicionales:

* **posición de elemento**: qué elemento es dentro del layout
* **posición de atributo**: qué modalidad es dentro del elemento

Esto permite que el Transformer represente mejor la estructura bidimensional `(elemento, atributo)`.

### 3. Flatten intercalado

Los embeddings se reorganizan en secuencia intercalada `[B, 5M, D]` antes de entrar al Transformer. Ver sección 8.4 para detalle.

### 4. Transformer encoder

Procesa la secuencia completa sin autoregresión. LayoutDM modela el layout completo de forma paralela.

### 5. Heads por modalidad

Para cada modalidad hay una proyección lineal al tamaño de su vocabulario: `[B, M, V_m]`.

## Qué ya funciona

* carga `tokens_*.pt` reales desde disco ✅
* lee `vocab_meta.json` y usa `M` real ✅
* construye `pad_mask` correcto por modalidad ✅
* flatten intercalado por elemento ✅
* shuffle de elementos validado ✅
* schedule forward exacto implementado ✅
* ejecuta entrenamiento sin errores ✅
* guarda checkpoint ✅
* genera muestras renderizables ✅

## Qué sigue faltando

* validar `q_sample()` visualmente — comprobar degradación gradual de muestras reales
* revisión del sampling reverso (`q_sample_from_Qbar`, `compute_losses`, `unconditional_sample`)
* reentrenar corrida corta con todas las correcciones activas
* mejor validación cualitativa comparando real vs generado

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
