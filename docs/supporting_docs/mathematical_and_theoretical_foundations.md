# Fundamentos matemáticos y teóricos

Este documento sostiene el indicador **PI3** (solución de problemas con matemáticas). Cada ecuación corresponde a código del repositorio; la columna *Implementación* de cada tabla indica dónde verificarla. Las cifras de §8.1, §8.3 y §8.5 son del reparto final (val, y test y celular evaluados una sola vez); las de §8.2 que se marcan como del segundo reparto ([preparación](../03_data_preparation.md#cómo-llegamos-al-reparto-final)) explican cómo se llegó a esta configuración.

| Evidencia exigida por la consigna | Sección |
| --- | --- |
| Formulación del problema y variables | §1 |
| Por qué es un problema complejo | §1.4 |
| Justificación de la estrategia y alternativas descartadas | §2, §6.6, §9 |
| Operaciones, cálculos y métricas | §3 a §7 |
| Interpretación matemática de resultados | §8 |
| Conclusiones técnicas y mejoras | §10 |

---

## 1. Formulación del problema

### 1.1 Datos y variables

El conjunto de datos es

$$
\mathcal{D}=\{(x_i,\,y_i,\,s_i,\,p_i,\,g_i)\}_{i=1}^{N},
$$

donde:

| Símbolo | Significado | Dominio |
| --- | --- | --- |
| $x_i$ | fotografía RGB de alto $H_i$ y ancho $W_i$ | $\{0,\dots,255\}^{H_i\times W_i\times 3}$ |
| $y_i$ | calidad visible de la pieza | $\mathcal{Y}=\{\text{good},\text{regular},\text{bad}\}$, $K=3$ |
| $s_i$ | fuente | $\{\text{ryandpark},\text{agrifreshnet},\text{afruitdb},\text{own},\text{celular}\}$ |
| $p_i$ | producto | 13 productos canónicos (`config.PRODUCTS`) |
| $g_i$ | grupo de captura (ráfaga, original + aumentos, sesión propia) | identificador |

$s_i$, $p_i$ y $g_i$ **no son entradas del modelo**. Se usan para particionar, auditar y medir sesgos.

### 1.2 Sistema como composición de funciones

El sistema entrega dos salidas a partir de la misma segmentación $S$:

$$
\hat y = h_\theta\big(\phi(S(x))\big),\qquad
\hat t = \tau\big(d_n(S(x))\big).
$$

- $S(x)=(c,\,m)$: recorte cuadrado $c\in\mathbb{R}^{128\times128\times3}$ y máscara binaria $m\in\{0,1\}^{128\times128}$ de la pieza (§3).
- $\phi$: extractor de atributos manuales, $\phi(c,m)\in\mathbb{R}^{359}$ (§5). Para la CNN, $\phi(c,m)=c\odot m$ (el fondo se anula).
- $h_\theta$: clasificador con parámetros $\theta$ (§6).
- $d_n$: diámetro normalizado de la pieza (§4); $\tau$: umbrales que lo convierten en *small/medium/large*.

### 1.3 Objetivo de aprendizaje

Los errores no cuestan lo mismo: vender una pieza `bad` como `good` afecta al consumidor; descartar una `good` es desperdicio. Con una matriz de costos $C_{jk}$ (costo de predecir $k$ cuando la clase real es $j$), el objetivo ideal es minimizar el riesgo esperado

$$
R(h)=\mathbb{E}_{(x,y)}\big[C_{y,\,h(x)}\big].
$$

Como $C$ no se conoce con precisión, se optimiza un sustituto diferenciable, la **entropía cruzada ponderada por clase** (§6.5), y se reporta aparte la tasa del error caro `bad→good` (§7.3).

El objetivo de generalización no es la distribución de los datasets sino la de **capturas nuevas**. Por eso los grupos $g_i$ deben quedar completos en una sola partición (§7.1): de lo contrario el modelo se evaluaría sobre casi la misma foto que vio en entrenamiento.

### 1.4 Por qué es un problema complejo de ingeniería

1. **Entrada no estructurada y de alta dimensión:** una foto de 1200×1600 px tiene $5{,}8\times10^6$ valores; la señal útil (una mancha) ocupa una fracción pequeña.
2. **Etiquetas heterogéneas:** Kaggle solo trae `good/bad`; AgriFreshNET trae `Fresh/Semi-Fresh/Rotten`. El mapeo a $\mathcal{Y}$ es una decisión de modelado, no un dato.
3. **Confusor fuente–clase:** en el inventario, $P(s=\text{agrifreshnet}\mid y=\text{regular})\approx0{,}82$ (era 1 antes de agregar AFruitDB). Un modelo puede minimizar la pérdida aprendiendo $s$ en lugar de $y$ (§8.3).
4. **Dependencia entre observaciones:** ráfagas y aumentos rompen el supuesto i.i.d.; val del segundo reparto tiene 3.795 recortes pero solo 852 grupos independientes. Definir mal el grupo produjo una fuga real (§7.1).
5. **Variable física no identificable:** el tamaño real no se puede recuperar de una foto sin escala (§4.2).
6. **Desbalance:** en el conjunto con que se ajusta la CNN, `regular` es el 22,2% de los recortes (§6.5).

---

## 2. Estrategia de solución y justificación

| Decisión | Justificación teórica | Alternativa descartada y motivo |
| --- | --- | --- |
| Segmentar antes de clasificar | Elimina el fondo, la variable que más difiere entre fuentes; reduce la varianza de entrada y habilita $d_n$. | Clasificar la foto entera: el modelo podría aprender el fondo de cada dataset. |
| Atributos de color, textura y forma | El deterioro visible se manifiesta como cambio de color (pardeamiento) y de textura (arrugas, moho). | Píxeles crudos para ML clásico: 49.152 dimensiones, sin invarianzas. |
| Regresión logística como línea base | Frontera lineal convexa con solución única; mide cuánto aporta la no linealidad. | Clasificador mayoritario: no aporta información sobre las features. |
| SVM lineal | Maximiza el margen; buena generalización en alta dimensión. | SVM RBF: costo $O(n^2)$ a $O(n^3)$ con $n=16.272$. |
| Random Forest | Capta interacciones no lineales entre atributos (p. ej. tono × saturación) y reduce varianza promediando árboles. | Árbol único: varianza alta. XGBoost: posible mejora, más hiperparámetros. |
| CNN de 4 bloques desde cero | Aprende filtros directamente de los píxeles; con 4 bloques el campo receptivo cubre una lesión completa (§6.4). | Transfer learning: mejor F1 esperado, pero la consigna lo recomienda menos. |
| Random Forest como modelo final | Empata en F1 con la CNN y el ensamble, y es el que menos frutas malas deja pasar como buenas (§8.1). | Ensamble RF + CNN: F1 igual, pero el doble de errores `bad→good` en el reparto final. |
| Split por ráfaga y CV agrupada | Estima el error sobre capturas nuevas (§7.1). | Split por imagen o por archivo: fuga entre fotos de una misma ráfaga, que se midió en el primer reparto. |
| F1 macro como métrica de selección | Pesa igual las 3 clases (§7.2). | Accuracy: premia ignorar la clase minoritaria. |

---

## 3. Segmentación (`src/data/segmentation.py`)

### 3.1 Espacio de color

La imagen se reduce a lado mayor 320 px, se suaviza con un filtro gaussiano 5×5 y se pasa a CIE Lab. $L$ codifica luminosidad y $(a,b)$ cromaticidad. Las sombras cambian sobre todo $L$, no $(a,b)$, lo que justifica medir distancias en el plano $(a,b)$.

Una foto se descarta por escala de grises si

$$
\frac{1}{|I|}\sum_{u\in I}\big(|a_u-128|+|b_u-128|\big)<4 .
$$

### 3.2 Modelo del fondo y máscara inicial

Con k-means ($k=3$, inicialización k-means++) sobre una franja del 6% del borde se obtienen los colores de fondo $\{\mu_j\}$; se descartan grupos con menos del 15% del borde. Para cada píxel $u$:

$$
\text{score}(u)=\min_j\big\lVert (a_u,b_u)-(a_j,b_j)\big\rVert_2+\max\!\big(0,\ \min_j L_j-L_u-60\big).
$$

El primer término mide qué tan distinto es el color del fondo; el segundo recupera zonas muy oscuras (podredumbre) que casi no tienen cromaticidad.

El umbral se elige con **Otsu**, que maximiza la varianza entre clases del histograma de `score`:

$$
t^\*=\arg\max_t\ \omega_0(t)\,\omega_1(t)\,\big(\mu_0(t)-\mu_1(t)\big)^2 .
$$

Luego se aplican apertura y cierre morfológicos (elemento elíptico 5×5) para quitar ruido y cerrar huecos.

### 3.3 Refinamiento con GrabCut

GrabCut modela fruta y fondo con mezclas gaussianas en color y minimiza una energía de campo aleatorio de Markov mediante cortes de grafo:

$$
E(\alpha)=\sum_u U(\alpha_u;\,x_u)+\gamma\!\!\sum_{(u,v)\in\mathcal{N}}\![\alpha_u\neq\alpha_v]\,e^{-\beta\lVert x_u-x_v\rVert^2},
$$

con $\alpha_u\in\{0,1\}$ la etiqueta fondo/fruta. $U$ penaliza asignar un píxel a la clase cuyo modelo de color lo explica peor; el segundo término penaliza bordes donde no hay contraste. Se inicializa con la máscara de §3.2: núcleo erosionado = fruta segura, anillo dilatado = fondo probable. Se hacen 4 iteraciones.

### 3.4 Conteo de piezas y controles

- **Objeto principal:** el componente conexo de mayor área, con peso 0,25 si toca dos o más bordes (suele ser la mesa).
- **Piezas pegadas:** transformada de distancia $D(u)=\min_{v\notin m}\lVert u-v\rVert$; sus máximos locales con $D>0{,}35\max D$ son centros candidatos. Dos picos son piezas distintas si el segmento que los une pasa por un "cuello" con $D<0{,}75\min(D_i,D_j)$; luego se separan con *watershed*.
- **Máscara dudosa:** área relativa fuera de $[0{,}02,\ 0{,}90]$ o máscara que toca 3 o 4 bordes.

Solo las fotos con estado `ok` (una pieza, máscara plausible) se usan para entrenar; la app aplica las mismas reglas para pedir otra captura.

---

## 4. Tamaño

### 4.1 Diámetro equivalente normalizado

Con $A$ el área de la máscara en píxeles de la imagen original:

$$
d=2\sqrt{\frac{A}{\pi}},\qquad d_n=\frac{d}{\max(H,W)}\in[0,\,\approx1].
$$

$d$ es el diámetro del círculo de igual área; dividir por el lado mayor hace la medida independiente de la resolución.

### 4.2 Límite teórico: el tamaño real no es identificable

Con el modelo de cámara estenopeica, un objeto de diámetro real $D$ a distancia $Z$ de una cámara con focal $f$ (en píxeles) aparece con diámetro $d\approx fD/Z$. Por tanto,

$$
d_n\approx\frac{f}{\max(H,W)}\cdot\frac{D}{Z}.
$$

$d_n$ depende del cociente $D/Z$: una mandarina cerca y un banano lejos pueden dar el mismo valor. **Sin conocer $Z$ (o una referencia de escala) $D$ no es identificable.** Por eso el sistema reporta *ocupación relativa del encuadre*, no centímetros.

### 4.3 Evidencia en las fotos propias

Medido con el mismo pipeline sobre las 35 fotos propias de una sola pieza con estado `ok` y tamaño anotado (se excluyen el racimo de banano y los mangos sin tamaño):

| Producto (fotos) | Etiqueta de tamaño | Rango de $d_n$ |
| --- | --- | --- |
| Banano (15) | medium | 0,411 – 0,488 |
| Banano (1) | small | 0,335 |
| Mandarina (15) | medium | 0,337 – 0,401 |
| Mandarina (1) | small | 0,252 |
| Manzana y papaya (3) | medium | 0,358 – 0,477 |

Lecturas:

1. Con el umbral provisional anterior (`small` < 0,45), 23 de 33 piezas `medium` quedaban como `small`: la regla contradecía la anotación en el 70% de los casos.
2. Los rangos dependen del producto: el `small` de banano (0,335) es prácticamente igual al `medium` mínimo de mandarina (0,337). Ningún umbral global los separa con margen, lo que confirma §4.2.
3. Los umbrales vigentes en `config.SIZE_THRESHOLDS` (`small` < 0,33 ≤ `medium` < 0,55 ≤ `large`) coinciden con 34 de las 35 anotaciones (97%). El corte inferior se apoya en solo dos ejemplos `small` y no hay ejemplos `large`, así que el superior es una extrapolación. Son **umbrales de un protocolo de captura** (pieza centrada, celular a una distancia similar), no una medida física.

### 4.4 Mejora propuesta: escala con objeto de referencia

Si en la foto aparece un objeto de diámetro conocido $D_\text{ref}$ (moneda de 500 COP, 23,5 mm) en el mismo plano, la escala cancela $f$ y $Z$:

$$
\hat D=D_\text{ref}\cdot\frac{d}{d_\text{ref}} .
$$

Esto convierte $\hat t$ en una medida en milímetros con error dominado por la segmentación y la perspectiva.

---

## 5. Atributos para los modelos clásicos (`src/data/features.py`)

Sea $\Omega=\{u: m_u=1\}$ el conjunto de píxeles de la pieza en el recorte de 128×128.

### 5.1 Color (96 atributos)

Para cada canal $c\in\{R,G,B,H,S,V\}$ y cada bin $b\in\{0,\dots,15\}$ (ancho 16 niveles):

$$
h_c(b)=\frac{1}{|\Omega|}\sum_{u\in\Omega}\mathbb{1}\!\left[\left\lfloor \tfrac{c_u}{16}\right\rfloor=b\right],\qquad \sum_b h_c(b)=1 .
$$

Normalizar por $|\Omega|$ hace el histograma invariante al tamaño de la pieza. HSV separa el tono (pardeamiento) de la saturación y el brillo.

### 5.2 Textura LBP (256 atributos)

Para cada píxel interior $u$ con vecinos $v_0,\dots,v_7$ (todos dentro de $\Omega$):

$$
\text{LBP}(u)=\sum_{k=0}^{7}\mathbb{1}[I(v_k)\ge I(u)]\;2^k\in\{0,\dots,255\},
$$

y el atributo es el histograma normalizado de esos códigos. LBP compara intensidades relativas, por lo que es invariante a cambios monótonos de brillo; arrugas y moho producen patrones de código distintos a la piel lisa.

### 5.3 Forma (7 atributos)

Con $A=|\Omega|$, perímetro $P$ (número de transiciones de borde) y $\lambda_1\ge\lambda_2$ los valores propios de la covarianza de las coordenadas de $\Omega$:

| Atributo | Fórmula | Interpretación |
| --- | --- | --- |
| Circularidad | $4\pi A/P^2$ | 1 en un círculo; baja con contornos irregulares o mordidos. |
| Elongación | $\sqrt{\lambda_2/\lambda_1}$ | 1 = redondo, cerca de 0 = alargado (banano). |
| Relación de aspecto | $w_\text{box}/h_\text{box}$ | Orientación y forma gruesa. |
| Llenado de la caja | $A/(w_\text{box}h_\text{box})$ | Concavidades. |
| Área relativa | $A/128^2$ | Ocupación del recorte. |
| Perímetro normalizado | $P/\sqrt{A}$ | Rugosidad del borde, invariante a escala. |
| Densidad de borde | $|\partial\Omega|/A$ | Fragmentación de la máscara. |

Total: $96+256+7=359$ atributos.

### 5.4 Estandarización

Para regresión logística y SVM: $z_j=(x_j-\hat\mu_j)/\hat\sigma_j$ con $\hat\mu_j,\hat\sigma_j$ estimados **solo en train** (dentro del `Pipeline`, también por fold). Sin esto, el término $\lVert w\rVert^2$ penalizaría distinto a atributos con escalas distintas. Random Forest no lo necesita porque sus cortes son invariantes a transformaciones monótonas.

---

## 6. Modelos

### 6.1 Regresión logística multinomial (línea base)

$$
P(y=k\mid z)=\frac{e^{w_k^\top z+b_k}}{\sum_{j=1}^{K}e^{w_j^\top z+b_j}},\qquad
\min_{W,b}\ \frac{1}{2}\lVert W\rVert_F^2-C\sum_{i}\omega_{y_i}\log P(y_i\mid z_i),
$$

con $C=0{,}1$ (elegido por la CV agrupada entre $\{0{,}1;\,1;\,10\}$), pesos de clase $\omega_k$ (§6.5) y solver L-BFGS. El problema es convexo: tiene un único óptimo global.

### 6.2 SVM lineal (`LinearSVC`)

Estrategia uno contra el resto: para cada clase $k$, con $t_i=+1$ si $y_i=k$ y $-1$ si no,

$$
\min_{w_k,b_k}\ \frac12\lVert w_k\rVert^2+C\sum_i\omega_{y_i}\,\max\!\big(0,\ 1-t_i(w_k^\top z_i+b_k)\big)^2 ,
\qquad \hat y=\arg\max_k\,(w_k^\top z+b_k).
$$

El margen geométrico es $2/\lVert w_k\rVert$: minimizar la norma lo ensancha, y $C$ regula cuánto se tolera violarlo. $C\in\{0{,}1;\,1;\,10\}$ se eligió por CV agrupada: en el reparto final, $C=0{,}1$, F1 = 0,573 ± 0,023 (en el segundo reparto había ganado $C=10$). Que ahora gane la regularización más fuerte indica que, con grupos más gruesos y más fuentes, la frontera lineal se ajusta a particularidades de los grupos de entrenamiento que no se repiten en otros pliegues (§8.2).

### 6.3 Random Forest

Cada nodo elige el corte $(j,\,t)$ que más reduce la impureza de Gini

$$
G(\mathcal{S})=1-\sum_{k=1}^{K}\hat p_k^2,\qquad
\Delta G=G(\mathcal{S})-\frac{|\mathcal{S}_L|}{|\mathcal{S}|}G(\mathcal{S}_L)-\frac{|\mathcal{S}_R|}{|\mathcal{S}|}G(\mathcal{S}_R).
$$

En cada nodo solo se evalúan $m$ atributos elegidos al azar, sobre una muestra bootstrap de train, y la predicción es el promedio de las probabilidades de $T$ árboles. Si cada árbol tiene varianza $\sigma^2$ y correlación $\rho$ con los demás,

$$
\operatorname{Var}\big(\bar h\big)=\rho\,\sigma^2+\frac{1-\rho}{T}\,\sigma^2 .
$$

El bootstrap y el submuestreo de atributos bajan $\rho$; promediar $T$ árboles elimina el segundo término. Por eso árboles profundos (sesgo bajo, varianza alta) funcionan bien en conjunto.

**Configuración elegida** (`src/models/model_definitions.py`). `GridSearchCV` prueba `min_samples_leaf` ∈ {1, 5, 20} × `max_depth` ∈ {sin límite, 20} × $m$ ∈ {$\lfloor\sqrt{359}\rfloor=18$, $\lfloor0{,}1\cdot359\rfloor=35$}, con $T=200$, y se queda con la de mayor F1 macro promedio en la CV por grupos (§7.2). En el reparto final eligió `min_samples_leaf=20`, sin límite de profundidad y $m=35$ (F1 de CV 0,602 ± 0,022); en el segundo reparto, una búsqueda equivalente había elegido `min_samples_leaf=5`, `max_depth=20` y $m=35$.

- Con $m=35$, la CV supera a $m=18$ en las 15 combinaciones de hoja y profundidad: cada corte ve más atributos útiles, a costa de árboles algo más correlacionados.
- Usar 400 árboles no cambia la CV (0,8186 frente a 0,8185): con $T=160$ el término $(1-\rho)\sigma^2/T$ ya es despreciable.
- Hojas de al menos 5 muestras y profundidad 20 reducen la mediana de hojas por árbol de 1.432 a 818.

### 6.4 Red convolucional (4 bloques)

Entrada $c\odot m$ normalizada a $[0,1]$. Cada bloque $\ell=1,\dots,4$, con $F_\ell\in\{32,64,128,256\}$ filtros, calcula

$$
a^{(\ell)}=\text{MaxPool}_{2\times2}\Big(\text{ReLU}\big(\text{BN}(W^{(\ell)}*a^{(\ell-1)})\big)\Big),\qquad
\text{BN}(u)=\gamma\,\frac{u-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}+\beta ,
$$

con convoluciones 3×3. Le siguen *Global Average Pooling* sobre el mapa de 8×8, $\bar a_f=\frac{1}{64}\sum_{u}a^{(4)}_{u,f}$, *dropout* con $p=0{,}35$ y una capa softmax de 3 salidas.

| Capa | Parámetros entrenables |
| --- | ---: |
| Conv 3×3×3×32 | 864 |
| Conv 3×3×32×64 | 18.432 |
| Conv 3×3×64×128 | 73.728 |
| Conv 3×3×128×256 | 294.912 |
| BatchNorm ($\gamma,\beta$) | 960 |
| Densa 256×3 + 3 | 771 |
| **Total** | **389.667** |

**Campo receptivo.** Con $r_\ell=r_{\ell-1}+(k-1)\,j_{\ell-1}$, y un salto $j$ que se duplica en cada pooling, cada neurona del último bloque ve un parche de **46×46 px** (36% del lado del recorte). Con 3 bloques eran 22×22 px (17%). Ese aumento explica la mejora de 3 a 4 bloques (§8.2): con 3 bloques, la red solo veía texturas locales y GAP diluía las manchas grandes; con 4 bloques, una neurona abarca una lesión completa y su contexto.

**Optimización.** Adam con $\eta_0=10^{-3}$:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad
\theta_t=\theta_{t-1}-\eta_t\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

*ReduceLROnPlateau* hace $\eta_t\leftarrow\max(\eta_t/2,\ 10^{-5})$ tras 4 épocas sin mejora de la pérdida en un holdout del 10% de los grupos de *train*. *EarlyStopping*, sobre ese mismo holdout y con paciencia 12, restaura los pesos de la mejor época. Validación no interviene.

### 6.5 Pérdida y pesos de clase

$$
L(\theta)=-\frac1N\sum_{i=1}^{N}\omega_{y_i}\log p_\theta(y_i\mid x_i),\qquad
\omega_k=\frac{N}{K\,N_k}.
$$

Con los conteos del conjunto con que se ajusta la CNN en el reparto final ($N=16.873$, *train* sin el holdout de parada temprana; `results/entrenamiento_cnn.json`):

| Clase | $N_k$ | $\omega_k$ |
| --- | ---: | ---: |
| good | 6.808 | 0,826 |
| regular | 3.744 | 1,502 |
| bad | 6.321 | 0,890 |

Con estos pesos, cada clase aporta lo mismo a la pérdida total ($\omega_kN_k=N/K$). Los pesos se calculan solo con train.

### 6.6 Ensamble por voto suave

$$
\hat p(k\mid x)=(1-w)\,p_\text{RF}(k\mid x)+w\,p_\text{CNN}(k\mid x),\qquad \hat y=\arg\max_k\hat p(k\mid x),\qquad w=0{,}5 .
$$

$w$ se fijó antes de evaluar y no se ajustó con validation. Promediar ayuda cuando los errores están poco correlacionados. Para la pérdida cuadrática, por la convexidad,

$$
\big(\bar p-y\big)^2\le\tfrac12(p_\text{RF}-y)^2+\tfrac12(p_\text{CNN}-y)^2 ,
$$

y la ganancia crece cuanto más difieren los dos modelos. RF y CNN fallan en conjuntos de filas que se solapan poco (Jaccard 0,25), lo que justifica combinarlos.

Dos precauciones:

1. `predict_proba` de scikit-learn ordena las clases alfabéticamente (`bad`, `good`, `regular`). Hay que reordenarlas a $\mathcal{Y}$ antes de sumar, como hacen la app (`src/prediction/predictor.py`) y la evaluación (`src/evaluation/evaluate_models.py`).
2. Las probabilidades de la CNN son más extremas que las de RF, así que $w$ nominal no es la influencia real de cada modelo. Calibrar ambas, por ejemplo con *temperature scaling*, queda como mejora.

---

## 7. Diseño experimental

### 7.1 Partición por grupos (`src/data/split.py`)

Cada grupo $g$ se asigna completo a $\pi(g)\in\{\text{train},\text{val},\text{test}\}$, con proporciones objetivo $q=(0{,}70;\,0{,}15;\,0{,}15)$. Sea $n_{p,s}$ el número de archivos del estrato $s$ (total, calidad, fuente × calidad, producto) en la partición $p$ y $N_s$ su total. Entre 256 permutaciones aleatorias (semilla 42) se elige la que minimiza, en orden lexicográfico:

1. clases ausentes en train;
2. productos ausentes en train;
3. clases ausentes en val y test;
4. el error de proporciones

$$
E=\sum_{s}w_s\left(\frac{n_{p,s}-q_pN_s}{\max(N_s,20)}\right)^2,\qquad w_s=\begin{cases}3&\text{total y calidad}\\1&\text{resto.}\end{cases}
$$

**La fuga del primer reparto y su corrección.** La garantía "cada grupo en una sola partición" solo sirve si el grupo representa de verdad una unidad independiente. En el primer reparto, las fotos de Kaggle tenían un grupo por archivo, en lugar de uno por ráfaga de captura: 645 de 746 ráfagas quedaron repartidas, y el 99% de los recortes de Kaggle en validation compartía ráfaga con train. Medido con CV agrupada, el F1 de Kaggle pasaba de 0,932 (agrupando por ráfaga) a 0,968 (por archivo): unos 3,5 puntos inflados. El segundo reparto reasigna los mismos recortes con los grupos por ráfaga:

| | Reparto 1 | Reparto 2 |
| --- | ---: | ---: |
| Recortes | 23.687 | 23.543 |
| Grupos | 13.192 | 5.677 |
| Ráfagas de Kaggle repartidas | 645 | 0 |
| Validation: recortes / grupos | 3.557 / 1.979 | 3.795 / 852 |
| Grupos de Kaggle en validation | 1.231 | 125 |

En el segundo reparto, el tamaño efectivo de val es de 852 grupos, no de 3.795 filas, y las métricas de Kaggle se apoyan en solo 125 grupos. Por eso sus intervalos son unas tres veces más anchos (§8.1).

### 7.2 Selección de hiperparámetros

Con `StratifiedGroupKFold` ($F=3$, grupos por ráfaga) sobre train:

$$
\lambda^\*=\arg\max_{\lambda\in\Lambda}\ \frac{1}{F}\sum_{f=1}^{F}\text{F1}_\text{macro}\big(h_{\lambda}^{(-f)},\ \text{fold}_f\big),
$$

implementado con `GridSearchCV(scoring="f1_macro", cv=StratifiedGroupKFold(3))`, que además reentrena el mejor modelo con todo *train*. Lo mismo se hace para el Random Forest de fruta, con el producto como etiqueta. Para la CNN no hay CV, porque cada entrenamiento tarda de 30 a 60 minutos (el final, 62 minutos en CPU): su arquitectura viene de los experimentos de §8.2 y su parada temprana usa un holdout de *train*. Así, validación y prueba no eligen nada y se miden una sola vez. En el segundo reparto, la CNN se había elegido con val, y por eso sus cifras de §8 son algo optimistas.

### 7.3 Métricas

Para la clase $k$:

$$
P_k=\frac{TP_k}{TP_k+FP_k},\quad R_k=\frac{TP_k}{TP_k+FN_k},\quad F1_k=\frac{2P_kR_k}{P_k+R_k},
$$

$$
\text{F1}_\text{macro}=\frac1K\sum_kF1_k,\qquad \text{BalAcc}=\frac1K\sum_kR_k,\qquad
\text{tasa}_{\text{bad}\to\text{good}}=\frac{\#\{y=\text{bad},\,\hat y=\text{good}\}}{\#\{y=\text{bad}\}} .
$$

F1 macro pesa igual a `regular` aunque sea la clase minoritaria. BalAcc es el promedio de los recalls y vale $1/K$ para un clasificador constante. La tasa `bad→good` mide el error de mayor costo de §1.3.

### 7.4 Incertidumbre de las métricas

Como los recortes de un grupo están correlacionados, la incertidumbre se estima con **bootstrap por grupos**: se remuestrean con reemplazo los grupos de validation ($B=2000$) y se recalcula la métrica. El intervalo del 95% va de los percentiles 2,5 a 97,5. Para comparar dos modelos se usa el **bootstrap pareado**: en cada réplica se calcula $\Delta=\text{F1}_A-\text{F1}_B$ sobre las mismas filas, y la diferencia es significativa si el intervalo de $\Delta$ excluye el 0.

Para dimensionar el conjunto externo de fotos propias, el semiancho del intervalo de una proporción $\hat p$ con $n$ piezas independientes es $1{,}96\sqrt{\hat p(1-\hat p)/n}$. Con $\hat p=0{,}5$ y un semiancho de 0,10, hacen falta $n\approx96$ piezas; para un recall por clase con semiancho de 0,15, unas 40 piezas por clase.

---

## 8. Resultados e interpretación matemática

§8.1 y §8.3 usan val del **reparto final**: 3.605 recortes en 129 grupos. Este reparto ya incluye AFruitDB, mango y la mitad "entrenar" de las fotos de celular. Test no se ha usado. Los experimentos de arquitectura de §8.2 son del segundo reparto, que tenía solo Kaggle y AgriFreshNET; se citan como historia del diseño.

### 8.1 Comparación con incertidumbre y elección del modelo final

| Modelo | F1 macro [IC 95% por grupos] | Recall `bad` | `bad→good` | Recall `regular` | `regular→good` |
| --- | --- | ---: | ---: | ---: | ---: |
| Regresión logística | 0,706 [0,654 – 0,762] | 0,824 | 124 | 0,603 | 339 |
| SVM lineal | 0,699 [0,639 – 0,764] | 0,858 | 122 | 0,510 | 413 |
| CNN de 4 bloques | 0,738 [0,677 – 0,801] | 0,878 | 120 | 0,436 | 448 |
| Ensamble 0,5/0,5 | 0,769 [0,705 – 0,839] | 0,919 | 82 | 0,496 | 425 |
| **Random Forest (modelo final)** | **0,762 [0,697 – 0,850]** | **0,948** | **41** | **0,643** | **322** |

Diferencias pareadas (bootstrap por grupos, 1.000 réplicas, IC 95%):

| Comparación | $\Delta$ F1 macro | $\Delta$ errores `bad→good` |
| --- | --- | --- |
| Ensamble − RF | +0,007 [−0,042; +0,045] | +41 [+8; +79] |
| CNN − RF | −0,025 [−0,090; +0,027] | +79 [+25; +138] |
| Ensamble − CNN | +0,032 [+0,015; +0,053] | −38 [−71; −14] |

Lecturas:

1. **Los no lineales superan a los lineales**, aunque la brecha es menor que en el segundo reparto (≈ 0,06).
2. **RF, la CNN y el ensamble empatan en F1**: los intervalos de $\Delta$ frente a RF incluyen el 0.
3. **RF comete significativamente menos errores caros.** El ensamble deja pasar 41 frutas malas más como buenas, y la CNN 79 más. Ninguno de los dos intervalos incluye el 0. RF también encuentra más regulares: 0,64, frente a 0,50 del ensamble.
4. **El ensamble sigue ayudando a la CNN**, como predice §6.6: corrige parte de sus errores. RF y la CNN coinciden en el 74,2% de las fotos. En las 930 donde discrepan, el ensamble acierta el 57,1%, RF el 47,4% y la CNN el 47,7%. Pero promediar con una CNN que tiende a decir `good` arrastra a RF hacia `good`.
5. **Decisión (tomada en val, antes de abrir test):** `APP_MODEL = "random_forest"`. Con F1 empatado decide el costo asimétrico de §1.3, y ahí el mejor es RF. Además, la app ya no necesita TensorFlow para decidir.

### 8.2 Sesgo, varianza y sobreajuste

| Modelo | Train | Validation | Diagnóstico |
| --- | --- | --- | --- |
| Logística | F1 0,740 | F1 0,714 | Sesgo alto: brecha de 0,026, el modelo no tiene capacidad. |
| SVM lineal | F1 0,737 | F1 0,717 | Sesgo alto; la CV elige el $C$ más grande. |
| Random Forest | F1 0,993 | F1 0,857 (CV 0,816) | Varianza alta por árbol, controlada por el promedio (§6.3). La CV agrupada es el estimador honesto. |
| CNN de 3 bloques | acc 0,846, pérdida 0,380 | acc 0,817, pérdida 0,412 | Sin sobreajuste, con un techo en 0,41: límite de capacidad, no de optimización. |
| CNN de 3 bloques + giros | pérdida 0,554 | pérdida 0,529 | Más regularización sobre un modelo que no sobreajustaba: empeora el ajuste y el F1. |
| CNN de 4 bloques | acc 0,968, pérdida 0,095 | acc 0,878, pérdida 0,334 | Sobreajuste desde la época 30, contenido por EarlyStopping (mejor época 38 de 50). |

La tabla anterior es del segundo reparto. **En el reparto final:**
- Regresión logística: F1 0,773 en train, 0,586 en la CV agrupada y 0,706 en val; SVM lineal: 0,774, 0,573 y 0,699. Las dos eligen $C=0{,}1$.
- RF tiene F1 0,930 en train, 0,602 en la CV agrupada y 0,762 en val. La brecha train–CV es grande: los árboles memorizan, y la elección de `min_samples_leaf=20` (las hojas más grandes de la rejilla) confirma que hace falta regularizar.
- La CNN de 4 bloques termina con accuracy 0,903 y pérdida 0,255 en train, y 0,883 y 0,355 en su holdout de parada temprana. Su mejor época fue la 18 de 30.
- Que val (0,76) supere a la CV (0,60) indica que los pliegues de la CV son más difíciles que val. Por ejemplo, un pliegue puede quedarse sin las sesiones de una fruta. La CV sirve para comparar configuraciones, no como estimación del error final.

La secuencia 3 bloques → 3 bloques con giros → 4 bloques funciona como un experimento del dilema sesgo–varianza:

- Con 3 bloques había sesgo (train y validation juntos y con pérdida alta).
- Los giros aumentaron la regularización y empeoró, lo que confirma que el problema no era la varianza.
- El cuarto bloque aumentó la capacidad (campo receptivo de 22 a 46 px) y redujo el sesgo, a cambio de varianza.

Con 4 bloques, la regularización vuelve a ser una mejora razonable, pero probar más variantes con validación aumentaría el sesgo de selección; se deja como trabajo futuro.

### 8.3 Descomposición por fuente: el confusor

| Modelo | F1 macro Kaggle (solo good/bad) | F1 macro AgriFreshNET (3 clases) | F1 macro AFruitDB (3 clases) | Recall `regular` AgriFreshNET / AFruitDB |
| --- | ---: | ---: | ---: | ---: |
| Regresión logística | 0,872 | 0,613 | 0,683 | 0,57 / 0,69 |
| SVM lineal | 0,886 | 0,604 | 0,681 | 0,47 / 0,60 |
| Random Forest | 0,966 | 0,661 | 0,655 | 0,58 / 0,80 |
| CNN | 0,950 | 0,661 | 0,641 | 0,46 / 0,38 |
| Ensamble | 0,964 | 0,699 | 0,695 | 0,51 / 0,46 |

Kaggle no tiene `regular`, así que ahí el problema es binario. **Las cifras representativas del problema de tres clases son las de AgriFreshNET y AFruitDB (≈ 0,66 para RF).** Con AFruitDB, `regular` ya no viene de una sola fuente, y RF reconoce el `regular` de AFruitDB (0,80) mejor que el de AgriFreshNET (0,58).

En el segundo reparto, sin AFruitDB, se cumplía $P(s=\text{agrifreshnet}\mid y=\text{regular})=1$, y la evidencia mostraba que los atributos codifican la fuente:

- Un RF que predice la *fuente* con los mismos 359 atributos alcanza 0,976 de balanced accuracy.
- Entrenado solo con una fuente y evaluado en la otra, el AUC de `good` frente a `bad` cae de 0,98 a 0,72 (AgriFreshNET → Kaggle) y de 0,99 a 0,75 (Kaggle → AgriFreshNET).
- El modelo entrenado solo con AgriFreshNET predice `regular` en 305 de 1.553 fotos de Kaggle; el modelo combinado, en 1.

La conclusión defendible es que hay un confusor y un cambio de dominio fuerte. Que la sonda acierte la fuente no prueba que el clasificador use la fuente como atajo, pero la señal aprendida no transfiere bien entre dominios. En val del segundo reparto había 4 fotos propias, y RF y la CNN fallaban 3. El tamaño de muestra no permite estimar nada, pero el resultado es coherente con lo anterior y motiva el conjunto externo de §7.4.

### 8.4 Dónde fallan

- **`regular→good`** es el error más frecuente en los tres candidatos (115 a 128 de 709). El deterioro inicial desplaza poco el histograma de color global, y la CNN, aun con 46 px de campo receptivo, promedia la mancha con piel sana en el GAP.
- **Los productos exclusivos de AgriFreshNET** (pepino, papaya, berenjena, tomate, piña) concentran los errores. Son también los que más empeoraron con la augmentation geométrica, lo que sugiere que la orientación o la iluminación de esa captura son constantes y el modelo las usa.

### 8.5 Prueba final: test y fotos de celular (evaluadas una sola vez)

| Modelo | F1 test [IC 95%] | Kaggle | AgriFreshNET | AFruitDB | `bad→good` |
| --- | --- | ---: | ---: | ---: | ---: |
| Regresión logística | 0,560 | 0,865 | 0,433 | 0,373 | 86 |
| SVM lineal | 0,566 | 0,880 | 0,460 | 0,348 | 80 |
| CNN | 0,603 | 0,903 | 0,499 | 0,472 | 190 |
| Ensamble | 0,621 | 0,929 | 0,513 | 0,539 | 101 |
| **Random Forest (final)** | **0,596 [0,522 – 0,677]** | **0,932** | **0,431** | **0,613** | **40** |

Test tiene 4.169 recortes en 184 grupos. La fruta se acierta en el 79,5 % (val: 86,9 %).

**Lecturas:**

1. **El criterio de costo se sostiene fuera de muestra.** RF vuelve a ser el que menos `bad→good` comete (40, frente a 101 y 190). El ensamble le gana en F1 por 0,025, del mismo orden que la diferencia no significativa de val.
2. **La brecha val → test (0,76 → 0,60) es de `regular` en AgriFreshNET.** RF llama `bad` a 570 de los 786 `regular` de esa fuente en test. La razón es el tamaño efectivo: casi todos los productos tienen solo de 2 a 5 lotes de `regular` (la excepción es la berenjena, con 21), y cada lote va entero a una partición. Val recibió lotes de naranja, papaya, tomate y berenjena, donde RF llama `bad` a menos del 5 % de los `regular`. Test recibió lotes de banano, pepino, piña y berenjena, con 87–93 %. Con tan pocas unidades independientes, $\hat F_1$ de val tiene una varianza que el bootstrap por grupos solo capta en parte: los intervalos [0,697; 0,850] y [0,522; 0,677] casi no se solapan.
3. **La frontera `regular`/`bad` depende del lote.** Para algunos productos, el `regular` de un lote se parece al `bad` de otro. Es una consecuencia del confusor de §8.3: lo que "regular" significa en AgriFreshNET varía entre lotes, y el modelo no puede aprenderlo con 2 a 5 ejemplos independientes.

**Fotos de celular** (302 fotos de 6 sesiones, etiquetadas a ciegas):

- **Fruta: accuracy 0,318 [0,053; 0,764].** El intervalo es enorme porque hay solo 6 sesiones. Por fruta: banano 1,00 y manzana 0,89 (con fotos de celular en train); naranja 0,94; guayaba 0,01, limón 0,00 y mango 0,00 (sin fotos de celular en train). Limones y guayabas amarillo-verdosos se predicen como naranja (probabilidad 0,3–0,5), y el mango como papaya. Los recortes y máscaras se revisaron visualmente y son correctos: no hay un error de segmentación, sino un cambio de dominio entre los datasets y la fruta local.
- **Estado: F1 macro 0,295 [0,227; 0,334].** Hay 250 buenas, 48 regulares y 4 malas. RF tiende a acertar las buenas, pero no detecta `regular`, y con 4 malas el recall de `bad` (0 de 4) no es estimable.
- **Conclusión:** entrenar con fotos de celular de una fruta es lo que más ayuda. Es la evidencia más directa a favor de ampliar la mitad "entrenar" a todas las frutas.

---

## 9. Alternativas consideradas y descartadas

| Alternativa | Motivo para no usarla |
| --- | --- |
| KNN | La distancia euclídea pierde significado en 359 dimensiones (concentración de distancias); predecir cuesta $O(n)$ por imagen. |
| Árbol de decisión único | Varianza alta; RF es su versión con varianza reducida (§6.3). |
| SVM con kernel RBF | El costo de entrenamiento crece de forma superlineal con $n=16.272$; la SVM lineal ya muestra que el problema no es la regularización. |
| XGBoost | Probable mejora sobre RF, pero agrega muchos hiperparámetros y más selección sobre los mismos datos. |
| Transfer learning (MobileNet, ResNet) | La consigna recomienda priorizar modelos entrenados desde cero; sirve como cota superior en trabajo futuro. |
| Augmentation geométrica (giros) | **Probada y descartada:** −0,051 de F1 [−0,073; −0,031] sobre un modelo sin sobreajuste. |
| CNN de 3 bloques con 64-128-256 filtros | Detenida por costo (197 s por época en una RTX 3050). La de 4 bloques logra más capacidad con un bloque adicional y menos cómputo. |
| Ajustar el peso del ensamble con validation | El máximo de la curva ($w=0{,}2$, F1 0,885) sería una selección sobre validation. Para ajustarlo habría que usar predicciones fuera de fold en train. |
| Regresión del tamaño en centímetros | Imposible sin escala (§4.2). |

---

## 10. Conclusiones técnicas

1. **La unidad estadística es el grupo, no la foto.** La fuga del primer reparto infló el F1 de Kaggle en unos 3,5 puntos aunque los grupos estuvieran bien separados: lo que fallaba era la definición del grupo. Val del segundo reparto tiene 852 unidades independientes, y sus intervalos se calculan sobre ellas.
2. **La formulación de costo asimétrico decide el modelo.** RF, la CNN y el ensamble empatan en F1. En el segundo reparto el ensamble cometía menos errores `bad→good`, pero en el reparto final RF comete la mitad (41 frente a 82, diferencia significativa). Ese es el criterio que §1.3 hace prioritario, así que el modelo final es RF.
3. **El diagnóstico sesgo–varianza guió los experimentos** y explica por qué la augmentation empeoró y la capacidad mejoró (§8.2).
4. **Fuente y clase están confundidas** para `regular`, y la transferencia entre dominios es débil. En test, F1 0,596 [0,522; 0,677], la caída frente a val viene de `regular` en lotes de AgriFreshNET que no se parecen a los de train (§8.5). Con fotos de celular, el modelo reconoce las frutas que vio en celular al entrenar y falla en las demás.
5. **El tamaño es geometría, no aprendizaje.** $d_n$ depende de $D/Z$; para obtener centímetros hace falta una referencia de escala (§4.4).
6. **Mejoras justificadas por la teoría:**
   - Comprobar si AgriFreshNET también tiene ráfagas por fecha y hora, con la misma prueba de CV de §7.1.
   - Fotos de celular de **todas** las frutas en la mitad de entrenamiento, con al menos 40 piezas por clase (§7.4, §8.5).
   - Más lotes independientes de `regular` por producto: con 2 a 5, la cifra de val no es estable (§8.5).
   - Calibrar las probabilidades antes de combinarlas (§6.6).
   - Regularizar la CNN eligiendo con validación cruzada, no con validación.

---

## Referencias

[1] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, 1979.
[2] C. Rother, V. Kolmogorov y A. Blake, "GrabCut: Interactive foreground extraction using iterated graph cuts," *ACM Trans. Graph.*, vol. 23, no. 3, pp. 309–314, 2004.
[3] D. Arthur y S. Vassilvitskii, "k-means++: The advantages of careful seeding," en *Proc. ACM-SIAM SODA*, 2007, pp. 1027–1035.
[4] T. Ojala, M. Pietikäinen y T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 7, pp. 971–987, 2002.
[5] C. Cortes y V. Vapnik, "Support-vector networks," *Mach. Learn.*, vol. 20, pp. 273–297, 1995.
[6] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, pp. 5–32, 2001.
[7] S. Ioffe y C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," en *Proc. ICML*, 2015.
[8] D. P. Kingma y J. Ba, "Adam: A method for stochastic optimization," en *Proc. ICLR*, 2015.
[9] B. Efron y R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
[10] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, pp. 153–157, 1947.
[11] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
[12] V. Meshram y K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications," *Data in Brief*, vol. 40, art. 107686, 2022.
[13] Universidad Icesi, *Competencias que se deben evaluar en el curso*, indicador PI3, 2026.
