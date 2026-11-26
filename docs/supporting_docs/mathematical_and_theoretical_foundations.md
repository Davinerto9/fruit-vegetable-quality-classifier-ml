# Fundamentos matemáticos y teóricos

Este documento sostiene el indicador **PI3** (solución de problemas con matemáticas). Cada ecuación corresponde a código del repositorio; la columna *Implementación* de cada tabla indica dónde verificarla. Las cifras de §8 son de validación del reparto v2 y explican por qué se eligió esta configuración; se reemplazan por las de v3 al reentrenar con el [plan simple](../plan_simple.md). Test sigue reservado.

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
| $s_i$ | fuente | $\{\text{ryandpark},\text{agrifreshnet},\text{own}\}$ |
| $p_i$ | producto | 12 productos canónicos (`config.PRODUCTS`) |
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
3. **Confusor fuente–clase:** en validation, $P(s=\text{agrifreshnet}\mid y=\text{regular})=1$. Un modelo puede minimizar la pérdida aprendiendo $s$ en lugar de $y$ (§8.3).
4. **Dependencia entre observaciones:** ráfagas y aumentos rompen el supuesto i.i.d.; validation v2 tiene 3.795 recortes pero solo 852 grupos independientes. Definir mal el grupo produjo una fuga real (§7.1).
5. **Variable física no identificable:** el tamaño real no se puede recuperar de una foto sin escala (§4.2).
6. **Desbalance:** en train, `regular` es el 18,7% de los recortes.

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
| Ensamble RF + CNN con peso fijo | Errores poco correlacionados; el promedio reduce el riesgo (§6.6). | Elegir un solo modelo: empatan en F1, pero el ensamble comete menos errores caros. |
| Split por ráfaga y CV agrupada | Estima el error sobre capturas nuevas (§7.1). | Split por imagen o por archivo: fuga entre fotos de una misma ráfaga, que se midió en v1. |
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

## 5. Atributos para los modelos clásicos (`src/features/classic.py`)

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

con $C=1$, pesos de clase $\omega_k$ (§6.5) y solver L-BFGS. El problema es convexo: tiene un único óptimo global.

### 6.2 SVM lineal (`LinearSVC`)

Estrategia uno contra el resto: para cada clase $k$, con $t_i=+1$ si $y_i=k$ y $-1$ si no,

$$
\min_{w_k,b_k}\ \frac12\lVert w_k\rVert^2+C\sum_i\omega_{y_i}\,\max\!\big(0,\ 1-t_i(w_k^\top z_i+b_k)\big)^2 ,
\qquad \hat y=\arg\max_k\,(w_k^\top z+b_k).
$$

El margen geométrico es $2/\lVert w_k\rVert$: minimizar la norma lo ensancha, y $C$ regula cuánto se tolera violarlo. $C\in\{0{,}1;\,1;\,10\}$ se eligió por CV agrupada por ráfaga en el split v2: $C=10$, F1 = 0,676 ± 0,010. Que gane el $C$ más grande (menos regularización) es una señal de que el modelo está limitado por su frontera lineal y no por sobreajuste (§8.2).

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

**Configuración elegida** (`src/training/train_classic.py`). `GridSearchCV` prueba `min_samples_leaf` ∈ {1, 5, 20} × `max_depth` ∈ {sin límite, 20} × $m$ ∈ {$\lfloor\sqrt{359}\rfloor=18$, $\lfloor0{,}1\cdot359\rfloor=35$}, con $T=200$, y se queda con la de mayor F1 macro promedio en la CV por grupos (§7.2). En v2, una búsqueda equivalente eligió `min_samples_leaf=5`, `max_depth=20` y $m=35$.

- Con $m=35$, la CV supera a $m=18$ en las 15 combinaciones de hoja y profundidad: cada corte ve más atributos útiles, a costa de árboles algo más correlacionados.
- Usar 400 árboles no cambia la CV (0,8186 frente a 0,8185): con $T=160$ el término $(1-\rho)\sigma^2/T$ ya es despreciable.
- Hojas de al menos 5 muestras y profundidad 20 reducen la mediana de hojas por árbol de 1.432 a 818.

### 6.4 Red convolucional (candidata E3b)

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

**Campo receptivo.** Con $r_\ell=r_{\ell-1}+(k-1)\,j_{\ell-1}$, y un salto $j$ que se duplica en cada pooling, cada neurona del último bloque ve un parche de **46×46 px** (36% del lado del recorte). Con 3 bloques eran 22×22 px (17%). Ese aumento explica la mejora de E1 a E3b (§8.2): con 3 bloques, la red solo veía texturas locales y GAP diluía las manchas grandes; con 4 bloques, una neurona abarca una lesión completa y su contexto.

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

Con los conteos de train del split v2 ($N=16.272$):

| Clase | $N_k$ | $\omega_k$ |
| --- | ---: | ---: |
| good | 6.845 | 0,792 |
| regular | 3.049 | 1,779 |
| bad | 6.378 | 0,850 |

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

1. `predict_proba` de scikit-learn ordena las clases alfabéticamente (`bad`, `good`, `regular`). Hay que reordenarlas a $\mathcal{Y}$ antes de sumar, como hacen la app (`src/app/inference.py`) y el auditor de test (`src/evaluation/model_audit.py`).
2. Las probabilidades de la CNN son más extremas que las de RF, así que $w$ nominal no es la influencia real de cada modelo. Calibrar ambas, por ejemplo con *temperature scaling*, queda como mejora.

---

## 7. Diseño experimental

### 7.1 Partición por grupos (`src/data/split.py`, `src/data/split_v2.py`)

Cada grupo $g$ se asigna completo a $\pi(g)\in\{\text{train},\text{val},\text{test}\}$, con proporciones objetivo $q=(0{,}70;\,0{,}15;\,0{,}15)$. Sea $n_{p,s}$ el número de archivos del estrato $s$ (total, calidad, fuente × calidad, producto) en la partición $p$ y $N_s$ su total. Entre 256 permutaciones aleatorias (semilla 42) se elige la que minimiza, en orden lexicográfico:

1. clases ausentes en train;
2. productos ausentes en train;
3. clases ausentes en val y test;
4. el error de proporciones

$$
E=\sum_{s}w_s\left(\frac{n_{p,s}-q_pN_s}{\max(N_s,20)}\right)^2,\qquad w_s=\begin{cases}3&\text{total y calidad}\\1&\text{resto.}\end{cases}
$$

**La fuga del split v1 y su corrección.** La garantía "cada grupo en una sola partición" solo sirve si el grupo representa de verdad una unidad independiente. En v1, las fotos de Kaggle tenían un grupo por archivo, en lugar de uno por ráfaga de captura: 645 de 746 ráfagas quedaron repartidas, y el 99% de los recortes de Kaggle en validation compartía ráfaga con train. Medido con CV agrupada, el F1 de Kaggle pasaba de 0,932 (agrupando por ráfaga) a 0,968 (por archivo): unos 3,5 puntos inflados. El split v2 (`splits_v2_burst.csv`) reasigna los mismos recortes con los grupos por ráfaga:

| | v1 | v2 |
| --- | ---: | ---: |
| Recortes | 23.687 | 23.543 |
| Grupos | 13.192 | 5.677 |
| Ráfagas de Kaggle repartidas | 645 | 0 |
| Validation: recortes / grupos | 3.557 / 1.979 | 3.795 / 852 |
| Grupos de Kaggle en validation | 1.231 | 125 |

En v2, el tamaño efectivo de validation es de 852 grupos, no de 3.795 filas, y las métricas de Kaggle se apoyan en solo 125 grupos. Por eso sus intervalos son unas tres veces más anchos (§8.1).

### 7.2 Selección de hiperparámetros

Con `StratifiedGroupKFold` ($F=3$, grupos por ráfaga) sobre train:

$$
\lambda^\*=\arg\max_{\lambda\in\Lambda}\ \frac{1}{F}\sum_{f=1}^{F}\text{F1}_\text{macro}\big(h_{\lambda}^{(-f)},\ \text{fold}_f\big),
$$

implementado con `GridSearchCV(scoring="f1_macro", cv=StratifiedGroupKFold(3))`, que además reentrena el mejor modelo con todo *train*. Lo mismo se hace para el Random Forest de fruta, con el producto como etiqueta. Para la CNN no hay CV, porque cada entrenamiento tarda de 30 a 55 minutos en GPU: su arquitectura viene de los experimentos de §8.2 y su parada temprana usa un holdout de *train*. Así, validación y prueba no eligen nada y se miden una sola vez. En v2, la CNN se había elegido con validación, y por eso sus cifras de §8 son algo optimistas.

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

## 8. Resultados e interpretación matemática (validation v2)

Todas las cifras son de validation v2 (3.795 recortes, 852 grupos). Test no se ha usado.

### 8.1 Comparación con incertidumbre

| Modelo | F1 macro [IC 95% por grupos] | Recall `bad` | `bad→good` (tasa) | `regular→good` |
| --- | --- | ---: | ---: | ---: |
| Regresión logística | 0,714 | 0,789 | 165 (9,7%) | — |
| SVM lineal ($C=10$) | 0,717 | 0,820 | 173 (10,2%) | — |
| CNN E1 (3 bloques) | 0,797 | 0,870 | 135 (8,0%) | 97 |
| Random Forest (rejilla) | 0,857 [0,834 – 0,878] | 0,897 | 121 (7,1%) | 122 |
| CNN E3b (4 bloques) | 0,858 [0,831 – 0,881] | 0,916 | 90 (5,3%) | 128 |
| **Ensamble 0,5/0,5** | **0,875** | **0,926** | **79 (4,7%)** | **115** |

Diferencias pareadas (bootstrap por grupos, IC 95%):

| Comparación | $\Delta$ F1 global | Kaggle | AgriFreshNET |
| --- | --- | --- | --- |
| E3b − E1 (capacidad) | +0,060 [+0,036; +0,084] | +0,030 | +0,094 |
| E2 − E1 (augmentation) | −0,051 [−0,073; −0,031] | −0,004 | −0,084 |
| CNN E3b − RF | +0,001 [−0,027; +0,028] | −0,016 | +0,023 |
| Ensamble − RF | +0,018 [−0,004; +0,042] | −0,005 | +0,044 [+0,012; +0,077] |
| Ensamble − CNN E3b | +0,018 [+0,008; +0,028] | +0,011 | +0,020 |

Lecturas:

1. **Los no lineales superan a los lineales** por unos 0,14 de F1: la frontera lineal no alcanza.
2. **RF y la CNN E3b empatan en F1.** El intervalo de $\Delta$ incluye el 0. La CNN, en cambio, comete un 26% menos de errores `bad→good`, el más caro.
3. **El ensamble mejora todos los indicadores de riesgo a la vez.** Frente a RF, la mejora global no es significativa, pero sí lo es en AgriFreshNET, la parte con tres clases. RF y CNN coinciden en el 84,3% de las fotos, con accuracy 0,951; en las 594 donde discrepan, el ensamble acierta el 58,4%, frente al 46,5% de RF y el 48,1% de la CNN. Esto concuerda con §6.6.
4. **La decisión se justifica por costo, no por F1.** Si el error `bad→good` pesa más que los demás (§1.3), el ensamble minimiza el riesgo esperado entre los candidatos: 79 frente a 121 errores caros.

### 8.2 Sesgo, varianza y sobreajuste

| Modelo | Train | Validation | Diagnóstico |
| --- | --- | --- | --- |
| Logística | F1 0,740 | F1 0,714 | Sesgo alto: brecha de 0,026, el modelo no tiene capacidad. |
| SVM lineal | F1 0,737 | F1 0,717 | Sesgo alto; la CV elige el $C$ más grande. |
| Random Forest | F1 0,993 | F1 0,857 (CV 0,816) | Varianza alta por árbol, controlada por el promedio (§6.3). La CV agrupada es el estimador honesto. |
| CNN E1 (3 bloques) | acc 0,846, pérdida 0,380 | acc 0,817, pérdida 0,412 | Sin sobreajuste, con un techo en 0,41: límite de capacidad, no de optimización. |
| CNN E2 (E1 + simetrías) | pérdida 0,554 | pérdida 0,529 | Más regularización sobre un modelo que no sobreajustaba: empeora el ajuste y el F1. |
| CNN E3b (4 bloques) | acc 0,968, pérdida 0,095 | acc 0,878, pérdida 0,334 | Sobreajuste desde la época 30, contenido por EarlyStopping (mejor época 38 de 50). |

La secuencia E1 → E2 → E3b funciona como un experimento del dilema sesgo–varianza:

- E1 tenía sesgo (train y validation juntos y con pérdida alta).
- E2 aumentó la regularización y empeoró, lo que confirma que el problema no era la varianza.
- E3b aumentó la capacidad (campo receptivo de 22 a 46 px) y redujo el sesgo, a cambio de varianza.

Con E3b, la regularización vuelve a ser una mejora razonable, pero probar más variantes con validación aumentaría el sesgo de selección; se deja como trabajo futuro.

### 8.3 Descomposición por fuente: el confusor

| Modelo | F1 macro Kaggle (solo good/bad) | F1 macro AgriFreshNET (3 clases) |
| --- | ---: | ---: |
| Regresión logística | 0,871 | 0,616 |
| SVM lineal | 0,881 | 0,628 |
| Random Forest | 0,954 | 0,799 |
| CNN E3b | 0,938 | 0,823 |
| Ensamble | 0,949 | 0,843 |

Kaggle no tiene `regular`, así que ahí el problema es binario. **La cifra representativa del problema de tres clases es la de AgriFreshNET (0,84 para el ensamble).**

Como $P(s=\text{agrifreshnet}\mid y=\text{regular})=1$, la información mutua $I(y;s)>0$, y la evidencia muestra que los atributos codifican la fuente:

- Un RF que predice la *fuente* con los mismos 359 atributos alcanza 0,976 de balanced accuracy.
- Entrenado solo con una fuente y evaluado en la otra, el AUC de `good` frente a `bad` cae de 0,98 a 0,72 (AgriFreshNET → Kaggle) y de 0,99 a 0,75 (Kaggle → AgriFreshNET).
- El modelo entrenado solo con AgriFreshNET predice `regular` en 305 de 1.553 fotos de Kaggle; el modelo combinado, en 1.

La conclusión defendible es que hay un confusor y un cambio de dominio fuerte. Que la sonda acierte la fuente no prueba que el clasificador use la fuente como atajo, pero la señal aprendida no transfiere bien entre dominios. En validation v2 hay 4 fotos propias, y RF y la CNN fallan 3. El tamaño de muestra no permite estimar nada, pero el resultado es coherente con lo anterior y motiva el conjunto externo de §7.4.

### 8.4 Dónde fallan

- **`regular→good`** es el error más frecuente en los tres candidatos (115 a 128 de 709). El deterioro inicial desplaza poco el histograma de color global, y la CNN, aun con 46 px de campo receptivo, promedia la mancha con piel sana en el GAP.
- **Los productos exclusivos de AgriFreshNET** (pepino, papaya, berenjena, tomate, piña) concentran los errores. Son también los que más empeoraron con la augmentation geométrica, lo que sugiere que la orientación o la iluminación de esa captura son constantes y el modelo las usa.

---

## 9. Alternativas consideradas y descartadas

| Alternativa | Motivo para no usarla |
| --- | --- |
| KNN | La distancia euclídea pierde significado en 359 dimensiones (concentración de distancias); predecir cuesta $O(n)$ por imagen. |
| Árbol de decisión único | Varianza alta; RF es su versión con varianza reducida (§6.3). |
| SVM con kernel RBF | El costo de entrenamiento crece de forma superlineal con $n=16.272$; la SVM lineal ya muestra que el problema no es la regularización. |
| XGBoost | Probable mejora sobre RF, pero agrega muchos hiperparámetros y más selección sobre los mismos datos. |
| Transfer learning (MobileNet, ResNet) | La consigna recomienda priorizar modelos entrenados desde cero; sirve como cota superior en trabajo futuro. |
| Augmentation geométrica (E2) | **Probada y descartada:** −0,051 de F1 [−0,073; −0,031] sobre un modelo sin sobreajuste. |
| CNN de 3 bloques con 64-128-256 filtros (E3a) | Detenida por costo (197 s por época en una RTX 3050). E3b logra más capacidad con un bloque adicional y menos cómputo. |
| Ajustar el peso del ensamble con validation | El máximo de la curva ($w=0{,}2$, F1 0,885) sería una selección sobre validation. Para ajustarlo habría que usar predicciones fuera de fold en train. |
| Regresión del tamaño en centímetros | Imposible sin escala (§4.2). |

---

## 10. Conclusiones técnicas

1. **La unidad estadística es el grupo, no la foto.** La fuga del split v1 infló el F1 de Kaggle en unos 3,5 puntos aunque los grupos estuvieran bien separados: lo que fallaba era la definición del grupo. Validation v2 tiene 852 unidades independientes, y sus intervalos se calculan sobre ellas.
2. **La formulación de costo asimétrico decide el modelo.** RF y la CNN empatan en F1. El ensamble no gana a RF de forma significativa en F1, pero reduce los errores `bad→good` de 121 a 79, y ese es el criterio que la formulación de §1.3 hace prioritario.
3. **El diagnóstico sesgo–varianza guió los experimentos** y explica por qué la augmentation empeoró y la capacidad mejoró (§8.2).
4. **Fuente y clase están confundidas** para `regular`, y la transferencia entre dominios es débil. Las conclusiones sobre `regular` se limitan a AgriFreshNET, y el desempeño en fotos de celular no está medido.
5. **El tamaño es geometría, no aprendizaje.** $d_n$ depende de $D/Z$; para obtener centímetros hace falta una referencia de escala (§4.4).
6. **Mejoras justificadas por la teoría:**
   - Comprobar si AgriFreshNET también tiene ráfagas por fecha y hora, con la misma prueba de CV de §7.1.
   - Conjunto externo de fotos propias con al menos 40 piezas por clase (§7.4).
   - Calibrar las probabilidades antes de combinarlas (§6.6).
   - Regularizar E3b eligiendo con validación cruzada, no con validación.

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
