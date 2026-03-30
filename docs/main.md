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

## Paso 1. Entender qué trae RICO

RICO no viene listo para LayoutDM.

RICO trae principalmente:

* screenshots
* archivos JSON por pantalla
* jerarquía de componentes UI

Un JSON típico de `semantic_annotations` tiene esta pinta:

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

Esto representa una pantalla con:

* un nodo raíz
* hijos
* bounds en píxeles
* clase del componente
* a veces `componentLabel`

---

## Paso 2. Entender qué necesita LayoutDM

LayoutDM no consume JSON ni imágenes.

Consume una secuencia de elementos con esta estructura:

`(c, x, y, w, h)`

Pero no en continuo, sino en **discreto**.

Es decir, al final lo que necesita es algo como:

```python
[c_id, x_id, y_id, w_id, h_id]
```

y agrupado en un tensor:

```python
[N, M, 5]
```

donde:

* `N` = número de pantallas
* `M` = máximo número de elementos por pantalla
* `5` = `(c, x, y, w, h)`

---

## Paso 3. Convertir el árbol JSON a lista plana de elementos

El JSON de RICO viene como árbol.

Pero LayoutDM no usa la jerarquía.

Por eso el primer paso fue recorrer recursivamente el JSON y extraer todos los nodos con:

* categoría
* bounds válidos

Eso se hace con un parser que:

* recorre `children`
* lee `bounds`
* usa `componentLabel` si existe
* si no existe, usa `class`

El resultado es una lista plana como esta:

```python
[
  {
    "category": "Web View",
    "x": 0.5,
    "y": 0.467,
    "w": 1.0,
    "h": 0.934
  }
]
```

---

## Paso 4. Normalizar coordenadas

RICO trae los bounds en píxeles:

`[x0, y0, x1, y1]`

LayoutDM necesita:

* `x, y` = centro
* `w, h` = tamaño
* todo normalizado entre 0 y 1

La conversión es:

```text
width_px  = x1 - x0
height_px = y1 - y0

x_center = (x0 + x1) / 2
y_center = (y0 + y1) / 2

x = x_center / screen_width
y = y_center / screen_height
w = width_px / screen_width
h = height_px / screen_height
```

Esto fue clave porque el modelo no puede trabajar directamente con coordenadas absolutas de píxeles.

---

## Paso 5. Entender el formato intermedio

Antes de discretizar, obtuvimos un formato intermedio:

```python
[
  ("Web View", 0.50, 0.467, 1.00, 0.934),
  ("[PAD]", 0.0, 0.0, 0.0, 0.0),
  ...
]
```

Eso aún **no es LayoutDM final**, pero ya tiene la estructura correcta.

Todavía faltaba:

* convertir categoría a id
* cuantizar `x,y,w,h` con KMeans

---

## Paso 6. Elegir `M`

Como cada pantalla tiene distinto número de elementos, necesitábamos fijar un máximo `M`.

Eso significa:

* si una pantalla tiene menos de `M` elementos → se rellena con `PAD`
* si tiene más de `M` → se recorta

La forma correcta de elegir `M` fue calcular estadísticas del dataset:

* mínimo
* p50
* p90
* p95
* p99
* máximo

Y usar:

```text
M = percentil 95
```

Esto es una decisión práctica razonable porque:

* evita que unos pocos casos extremos hagan crecer demasiado la secuencia
* conserva casi todas las pantallas sin truncar demasiado

---

## Paso 7. Hacer split train/val/test

Antes de cuantizar, había que separar los datos:

* train
* val
* test

Esto es importante porque el paper y una implementación correcta asumen que todo lo aprendido del dataset se ajusta **solo con train**.

Eso aplica especialmente a KMeans.

---

## Paso 8. Cuantización con KMeans

Este fue uno de los puntos más importantes.

LayoutDM no usa coordenadas continuas directamente.
Usa coordenadas **discretizadas**.

Para eso entrenamos 4 KMeans independientes sobre el split de train:

* uno para `x`
* uno para `y`
* uno para `w`
* uno para `h`

Elegimos por defecto:

```text
BINS = 64
```

### Por qué 64 bins

Porque es un balance razonable entre:

* suficiente resolución espacial
* vocabulario no muy grande
* entrenamiento estable
* complejidad manejable

Con 64 bins:

* cada modalidad espacial tiene vocabulario `64 + 2`
* los `+2` son `[MASK]` y `[PAD]`

Entonces:

* `x_id, y_id, w_id, h_id` quedan entre `0..63`
* `mask_id = 64`
* `pad_id = 65`

---

## Paso 9. Crear vocabulario de categorías

También necesitábamos discretizar la categoría.

Se construyó un diccionario `cat2id` usando **solo train**.

Por ejemplo:

```json
{
  "Button": 0,
  "Web View": 1,
  "Text": 2
}
```

Y igual que en las modalidades espaciales:

* `mask_id = C`
* `pad_id = C + 1`

donde `C` es el número de categorías reales.

---

## Paso 10. Exportar tokens listos para entrenar

Después de eso, cada pantalla quedó representada como:

```python
[
  [c_id, x_id, y_id, w_id, h_id],
  [c_id, x_id, y_id, w_id, h_id],
  ...
  [pad, pad, pad, pad, pad]
]
```

Y agrupado en tensores:

* `tokens_train.pt`
* `tokens_val.pt`
* `tokens_test.pt`

Además se exportaron:

* `cat2id.json`
* `vocab_meta.json`
* `centroids_x.pt`
* `centroids_y.pt`
* `centroids_w.pt`
* `centroids_h.pt`

Estos archivos son los artefactos necesarios para conectar el dataset con el training loop.

---

# 5. Aspectos importantes que no se deben olvidar

## 5.1 El dataset dummy no sirve para resultados reales

Sirve solo para validar que el pipeline corre.

## 5.2 El schedule de diffusion debe ser el exacto del paper

En el notebook quedó como placeholder:

`make_transition_params()`

Eso significa que la arquitectura estaba bien, pero no era una réplica experimental exacta hasta reemplazar esa función.

## 5.3 El shuffle de elementos por sample todavía faltaba

El paper busca que el modelo no dependa del orden de los elementos.

Por eso, en una implementación más fiel, el Dataset debería hacer shuffle de los slots válidos antes de devolver cada muestra.

## 5.4 LayoutDM no entiende semántica profunda

No sabe qué es “header” o “CTA” como concepto de producto.
Solo aprende patrones geométricos y categóricos.

---

# 6. Qué implementamos para el entrenamiento

Después de preparar el dataset, vimos cómo implementar el loop mínimo de LayoutDM.

La idea fue construir un sistema claro y correcto antes que rápido.

---

## 6.1 Entrada del modelo

El modelo recibe:

```python
tokens: [B, M, 5]
```

donde cada token representa:

* categoría
* x
* y
* w
* h

Todos discretos.

---

## 6.2 Qué arquitectura usamos

Usamos un **Transformer encoder** como denoiser.

No es autoregresivo.
No es decoder-only.
No genera token por token como un modelo de lenguaje.

Eso es importante porque LayoutDM quiere modelar el layout completo y no depender de un orden fijo.

---

## 6.3 Qué hace el proceso de diffusion

Durante entrenamiento:

1. tomamos un layout limpio `z0`
2. elegimos un timestep `t`
3. corrompemos `z0` hasta obtener `zt`
4. el modelo intenta predecir una versión más limpia

Ese proceso se hace con **diffusion discreto**.

---

## 6.4 Qué significa modality-wise diffusion

No usamos una sola difusión para todo.

Hay una difusión separada por modalidad:

* categoría
* x
* y
* w
* h

Eso evita mezclar vocabularios incompatibles.

Por ejemplo:

* una categoría no debería “transformarse” en un ancho
* un token de `x` no debería caer en el espacio de `c`

---

## 6.5 Qué losses usamos

El entrenamiento usa dos partes:

### A. VB loss

Es una KL entre:

* el posterior verdadero del diffusion
* el posterior predicho por el modelo

Esto es el corazón teórico del modelo.

### B. Aux loss

Una cross entropy para ayudar al modelo a reconstruir `z0`.

Esto estabiliza el entrenamiento.

La loss total fue:

```text
loss_total = vb_loss + lambda_aux * aux_loss
```

con:

```text
lambda_aux = 0.1
```

---

## 6.6 Cómo se ignora PAD

Los slots PAD no representan elementos reales.

Por eso se construyó `pad_mask` y la loss se calcula solo sobre tokens válidos.

Si PAD entra en la loss, el modelo aprende una distribución errónea.

---

## 6.7 Qué hace el sampling unconditional

En inferencia unconditional:

1. se inicia todo en `[MASK]`
2. se corre el reverse diffusion desde `T` hasta `1`
3. en cada paso el modelo predice distribuciones sobre tokens
4. se samplea categóricamente
5. al final se obtiene `z0`

Esto produce un layout nuevo desde cero.

---

# 7. Explicación separada: Preprocesamiento

## Objetivo

Transformar los JSON de RICO en tensores discretos listos para LayoutDM.

## Flujo

### Paso 1

Leer todos los JSON de `semantic_annotations/`.

### Paso 2

Recorrer cada árbol y extraer nodos con:

* categoría
* bounds válidos

### Paso 3

Convertir `bounds` a:

* `x`
* `y`
* `w`
* `h`

normalizados entre 0 y 1.

### Paso 4

Calcular estadísticas de número de elementos por pantalla y elegir `M`.

### Paso 5

Hacer split train/val/test.

### Paso 6

Construir `cat2id` usando solo train.

### Paso 7

Entrenar 4 KMeans en train:

* x
* y
* w
* h

### Paso 8

Asignar cada valor continuo al centroide más cercano.

### Paso 9

Construir tensores `[N, M, 5]` con PAD.

### Paso 10

Guardar artefactos:

* tokens
* centroides
* vocab_meta
* cat2id

## Resultado

El resultado del preprocesamiento es un dataset listo para el training loop.

---

# 8. Explicación separada: Entrenamiento del modelo

## Objetivo

Entrenar un LayoutDM mínimo que aprenda a generar layouts discretos.

## Flujo

### Paso 1

Cargar:

* `tokens_train.pt`
* `tokens_val.pt`
* `vocab_meta.json`

### Paso 2

Construir el `LayoutTokenDataset`.

### Paso 3

Construir el modelo:

* embeddings por modalidad
* positional encoding
* Transformer encoder
* head por modalidad

### Paso 4

Precomputar matrices de transición `Qt` y acumuladas `Qbar`.

### Paso 5

Durante cada batch:

* samplear `t`
* corromper `z0` → `zt`
* pasar `zt` por el modelo
* calcular `VB + aux`
* hacer backpropagation

### Paso 6

Guardar checkpoint.

### Paso 7

Probar unconditional sampling.

## Resultado

El resultado del entrenamiento es un modelo que puede generar secuencias discretas de layouts.

---

# 9. Estado en que quedó el sistema

## Ya resuelto

* entendimos la estructura de RICO
* implementamos el parser
* entendimos el formato requerido por LayoutDM
* definimos el pipeline de preprocesamiento
* generamos el script de exportación de tokens
* implementamos el training loop mínimo
* implementamos unconditional sampling

## Pendiente para una réplica más fiel

* usar el schedule exacto del paper
* añadir shuffle por sample
* validar visualmente el decoding de layouts
* comparar métricas con el paper

---

# 10. Resumen final simple

Si lo resumimos en una sola cadena de pasos:

### Preprocesamiento

`RICO JSON -> parse tree -> flatten elements -> normalize xywh -> split -> KMeans train-only -> discretize -> pad -> tokens.pt`

### Entrenamiento

`tokens.pt -> diffusion corruption -> Transformer denoiser -> VB + aux -> reverse sampling`

### Resultado final

Un modelo que aprende a generar layouts en formato:

```python
(c_id, x_id, y_id, w_id, h_id)
```

y que luego puede decodificarse a cajas geométricas reales.

---

# 11. Recomendación práctica de continuación

El siguiente paso más razonable es este:

1. ejecutar el script de preprocesamiento completo sobre RICO
2. revisar estadísticas reales de `M`
3. entrenar un modelo pequeño
4. decodificar y renderizar algunas muestras
5. validar si la geometría se ve coherente antes de seguir con métricas o constraints

---

Si quieres, en el siguiente mensaje te lo convierto en una versión más “documento técnico” con secciones tipo:

* Introducción
* Arquitectura
* Pipeline de datos
* Entrenamiento
* Riesgos y validaciones
* Próximos pasos

o te lo separo en dos archivos `.md`: uno para **preprocesamiento** y otro para **entrenamiento**.
