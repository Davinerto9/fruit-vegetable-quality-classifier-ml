# Fundamentos matemáticos y teóricos

Este documento sostiene el indicador **PI3** (solución de problemas con matemáticas). Cada ecuación corresponde a código del repositorio; la columna *Implementación* de cada tabla indica dónde verificarla. Las cifras salen de `experiments/results/` (validation; test sigue reservado).

| Evidencia exigida por la consigna | Sección |
| --- | --- |
| Formulación del problema y variables | §1 |
| Por qué es un problema complejo | §1.4 |
| Justificación de la estrategia y alternativas descartadas | §2, §9 |
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
4. **Dependencia entre observaciones:** ráfagas y aumentos rompen el supuesto i.i.d.; hay 3.557 recortes de validation pero solo 1.979 grupos independientes.
5. **Variable física no identificable:** el tamaño real no se puede recuperar de una foto sin escala (§4.2).
6. **Desbalance:** en train, `regular` es el 19% de los recortes.

---

## 2. Estrategia de solución y justificación

| Decisión | Justificación teórica | Alternativa descartada y motivo |
| --- | --- | --- |
| Segmentar antes de clasificar | Elimina el fondo, la variable que más difiere entre fuentes; reduce la varianza de entrada y habilita $d_n$. | Clasificar la foto entera: el modelo podría aprender el fondo de cada dataset. |
| Atributos de color, textura y forma | El deterioro visible se manifiesta como cambio de color (pardeamiento) y de textura (arrugas, moho). | Píxeles crudos para ML clásico: 49.152 dimensiones, sin invarianzas. |
| Regresión logística como línea base | Frontera lineal convexa con solución única; mide cuánto aporta la no linealidad. | Clasificador mayoritario: no aporta información sobre las features. |
| SVM lineal | Maximiza el margen; buena generalización en alta dimensión. | SVM RBF: costo $O(n^2)$ a $O(n^3)$ con $n=16.456$; queda como mejora con submuestreo. |
| Random Forest | Capta interacciones no lineales entre atributos (p. ej. tono × saturación) y reduce varianza promediando árboles. | Árbol único: varianza alta. XGBoost: posible mejora, más hiperparámetros. |
| CNN pequeña desde cero | Aprende filtros locales directamente de los píxeles; la consigna prioriza modelos propios. | Transfer learning: mejor F1 esperado, pero la consigna lo recomienda menos. |
| Split por grupos y CV agrupada | Estima el error sobre capturas nuevas (§7.1). | Split aleatorio por imagen: fuga entre ráfagas y aumentos, métricas optimistas. |
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

El margen geométrico es $2/\lVert w_k\rVert$: minimizar la norma lo ensancha y $C$ regula cuánto se tolera violarlo. $C\in\{0{,}1;\,1;\,10\}$ se eligió por CV agrupada: $C=1$, F1 = 0,704 ± 0,009.

### 6.3 Random Forest

Cada nodo elige el corte $(j,\,t)$ que más reduce la impureza de Gini

$$
G(\mathcal{S})=1-\sum_{k=1}^{K}\hat p_k^2,\qquad
\Delta G=G(\mathcal{S})-\frac{|\mathcal{S}_L|}{|\mathcal{S}|}G(\mathcal{S}_L)-\frac{|\mathcal{S}_R|}{|\mathcal{S}|}G(\mathcal{S}_R),
$$

evaluando solo $m=\lfloor\sqrt{359}\rfloor=18$ atributos al azar por nodo, sobre una muestra bootstrap de train. La predicción es el voto promedio de $T=160$ árboles. Si cada árbol tiene varianza $\sigma^2$ y correlación $\rho$ entre árboles,

$$
\operatorname{Var}\big(\bar h\big)=\rho\,\sigma^2+\frac{1-\rho}{T}\,\sigma^2 .
$$

El bootstrap y el submuestreo de atributos bajan $\rho$; promediar $T$ árboles elimina el segundo término. Por eso árboles profundos (sesgo bajo, varianza alta) funcionan bien en conjunto. Configuración seleccionada por CV: `max_depth=None`, `min_samples_leaf=2`, F1 CV = 0,820 ± 0,0002.

### 6.4 Red convolucional

Entrada $c\odot m$ normalizada a $[0,1]$. Cada bloque $\ell=1,2,3$ con $F_\ell\in\{32,64,128\}$ filtros:

$$
a^{(\ell)}=\text{MaxPool}_{2\times2}\Big(\text{ReLU}\big(\text{BN}(W^{(\ell)}*a^{(\ell-1)})\big)\Big),\qquad
\text{BN}(u)=\gamma\,\frac{u-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}+\beta ,
$$

con convoluciones 3×3. Luego *Global Average Pooling* $\bar a_f=\frac{1}{16\cdot16}\sum_{u}a^{(3)}_{u,f}$, *dropout* con $p=0{,}35$ y una capa softmax de 3 salidas.

| Capa | Parámetros entrenables |
| --- | ---: |
| Conv 3×3×3×32 | 864 |
| Conv 3×3×32×64 | 18.432 |
| Conv 3×3×64×128 | 73.728 |
| BatchNorm ($\gamma,\beta$) | 448 |
| Densa 128×3 + 3 | 387 |
| **Total** | **93.859** |

**Campo receptivo.** Con $r_\ell=r_{\ell-1}+(k-1)\,j_{\ell-1}$ y salto $j$ que se duplica en cada pooling, cada neurona del último bloque ve un parche de **22×22 px** (17% del lado del recorte). La red es, en la práctica, un detector de texturas locales cuyas respuestas promedia GAP. Una mancha pequeña activa pocas posiciones y su señal se diluye en el promedio: esto ayuda a explicar los errores `regular→good` (§8.4).

Optimización con Adam ($\eta=10^{-3}$):

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Se guarda el checkpoint de menor pérdida en validation y se detiene si no mejora en 6 épocas.

### 6.5 Pérdida y pesos de clase

$$
L(\theta)=-\frac1N\sum_{i=1}^{N}\omega_{y_i}\log p_\theta(y_i\mid x_i),\qquad
\omega_k=\frac{N}{K\,N_k}.
$$

Con los conteos de train ($N=16.456$):

| Clase | $N_k$ | $\omega_k$ |
| --- | ---: | ---: |
| good | 6.970 | 0,787 |
| regular | 3.133 | 1,751 |
| bad | 6.353 | 0,863 |

Con estos pesos, cada clase aporta lo mismo a la pérdida total ($\omega_kN_k=N/K$). Los pesos se calculan solo con train.

---

## 7. Diseño experimental

### 7.1 Partición por grupos (`src/data/split.py`)

Cada grupo $g$ se asigna completo a $\pi(g)\in\{\text{train},\text{val},\text{test}\}$ con proporciones objetivo $q=(0{,}70;\,0{,}15;\,0{,}15)$. Sea $n_{p,s}$ el número de archivos del estrato $s$ (total, calidad, fuente × calidad, producto) en la partición $p$ y $N_s$ su total. Entre 256 permutaciones aleatorias (semilla 42) se elige la que minimiza, en orden lexicográfico:

1. clases ausentes en train;
2. productos ausentes en train;
3. clases ausentes en val y test;
4. el error de proporciones

$$
E=\sum_{s}w_s\left(\frac{n_{p,s}-q_pN_s}{\max(N_s,20)}\right)^2,\qquad w_s=\begin{cases}3&\text{total y calidad}\\1&\text{resto.}\end{cases}
$$

Resultado: 16.456 train, 3.557 val y 3.674 test, sin grupos compartidos (el código lanza error si aparece uno).

### 7.2 Selección de hiperparámetros

Con `StratifiedGroupKFold` ($F=3$) sobre train:

$$
\lambda^\*=\arg\max_{\lambda\in\Lambda}\ \frac{1}{F}\sum_{f=1}^{F}\text{F1}_\text{macro}\big(h_{\lambda}^{(-f)},\ \text{fold}_f\big).
$$

Los folds también respetan grupos. Validation se usa para comparar modelos; test se abre una sola vez, protegido por un archivo de bloqueo y un *freeze* de hashes.

### 7.3 Métricas

Para la clase $k$:

$$
P_k=\frac{TP_k}{TP_k+FP_k},\quad R_k=\frac{TP_k}{TP_k+FN_k},\quad F1_k=\frac{2P_kR_k}{P_k+R_k},
$$

$$
\text{F1}_\text{macro}=\frac1K\sum_kF1_k,\qquad \text{BalAcc}=\frac1K\sum_kR_k,\qquad
\text{tasa}_{\text{bad}\to\text{good}}=\frac{\#\{y=\text{bad},\,\hat y=\text{good}\}}{\#\{y=\text{bad}\}} .
$$

F1 macro pesa igual a `regular` aunque sea minoritaria; BalAcc es el promedio de recalls y vale $1/K$ para un clasificador constante. La tasa `bad→good` mide el error de mayor costo de §1.3.

### 7.4 Incertidumbre de las métricas

Como los recortes de un grupo están correlacionados, la incertidumbre se estima con **bootstrap por grupos**: se remuestrean con reemplazo los 1.979 grupos de validation ($B=1000$) y se recalcula la métrica. El intervalo del 95% son los percentiles 2,5 y 97,5.

Para comparar dos modelos sobre las mismas filas se usa además la prueba de McNemar, con $b$ = filas donde solo el modelo A falla y $c$ = filas donde solo falla B:

$$
\chi^2=\frac{(|b-c|-1)^2}{b+c}\ \sim\ \chi^2_1 .
$$

---

## 8. Resultados e interpretación matemática (validation)

### 8.1 Comparación con incertidumbre

| Modelo | F1 macro [IC 95% por grupos] | BalAcc | Recall `bad` | Tasa `bad→good` |
| --- | --- | ---: | ---: | ---: |
| Regresión logística | 0,685 [0,661 – 0,708] | 0,701 | 0,689 | 10,3% |
| SVM lineal | 0,698 [0,674 – 0,722] | 0,702 | 0,750 | 10,7% |
| Random Forest | **0,842** [0,822 – 0,864] | **0,836** | **0,868** | 7,9% |
| CNN RGB | 0,799 [0,774 – 0,822] | 0,814 | 0,837 | **6,2%** |

- La diferencia RF − CNN tiene IC 95% por grupos de **[0,019 ; 0,068]**: no incluye 0, así que la ventaja de RF no se explica por azar de muestreo.
- McNemar: $b=210$ (solo RF falla), $c=378$ (solo CNN falla), $\chi^2=47{,}4$, $p<10^{-10}$. El valor de $p$ es optimista porque supone filas independientes; el bootstrap por grupos es la evidencia principal.
- Ambos no lineales superan a la línea base por más de 0,11 de F1, con intervalos sin solapamiento: la no linealidad aporta.
- **Tensión con el costo:** RF gana en F1, pero la CNN comete menos errores caros (89 vs 114 `bad→good`). Si el negocio pondera fuerte ese error, la selección cambia. Esto justifica estudiar un ensamble o un umbral asimétrico.

### 8.2 Sesgo, varianza y sobreajuste

| Modelo | F1 train | F1 validation | Brecha | Diagnóstico |
| --- | ---: | ---: | ---: | --- |
| Regresión logística | 0,749 | 0,685 | 0,064 | Sesgo alto: la frontera lineal no alcanza. |
| SVM lineal | 0,747 | 0,698 | 0,049 | Sesgo alto; mismo límite lineal. |
| Random Forest | 1,000 | 0,842 | 0,158 | Varianza alta por árbol, controlada por el promedio (§6.3). La CV agrupada (0,820) es el estimador honesto; el error de train no informa. |
| CNN RGB | acc 0,821 | acc 0,814 | 0,007 | Subajuste: la mejor época fue la última (40/40) y la pérdida de validation seguía bajando (0,515 → 0,450 en las dos últimas). |

Lectura: los modelos lineales están limitados por la capacidad; RF por los datos; la CNN por el entrenamiento. La mejora esperada para la CNN es entrenar más (más épocas, reducción del learning rate, aumento de datos), no regularizar.

### 8.3 Descomposición por fuente: el confusor

| Modelo | F1 macro Kaggle (solo good/bad) | F1 macro AgriFreshNET (3 clases) |
| --- | ---: | ---: |
| Regresión logística | 0,878 | 0,570 |
| SVM lineal | 0,895 | 0,600 |
| Random Forest | 0,968 | 0,779 |
| CNN RGB | 0,944 | 0,713 |

Kaggle no tiene `regular` en validation: ahí el problema es efectivamente binario y por eso las cifras son altas. **La cifra representativa del problema de 3 clases es la de AgriFreshNET (0,78 para RF).**

Como $P(s=\text{agrifreshnet}\mid y=\text{regular})=1$, la información mutua $I(y;s)>0$ y un clasificador puede reducir la pérdida aprendiendo rasgos de la fuente. Hay un indicio: sobre 1.553 imágenes de Kaggle, RF predijo `regular` 0 veces. Este dato es compatible tanto con un modelo correcto como con un atajo; para separarlos hay que evaluar `regular` de otra fuente (fotos propias, carpeta Mixed) y entrenar un clasificador de fuente con los mismos atributos.

### 8.4 Dónde fallan

- El error dominante de RF es `regular→good` (145 de 705, 20,6%): el deterioro inicial es una diferencia pequeña en el histograma, y una mancha localizada mueve poco la distribución global de color.
- La CNN confunde más `good→regular` (210) y `bad→regular` (147): su campo receptivo de 22 px con GAP responde a texturas locales, y una textura irregular sana puede parecer deterioro.
- Los errores de RF y CNN coinciden poco: Jaccard 0,325. La complementariedad sugiere que un ensamble podría mejorar, siempre que se valide sin tocar test.

---

## 9. Alternativas consideradas y descartadas

| Alternativa | Motivo para no usarla (por ahora) |
| --- | --- |
| KNN | Distancia euclídea poco significativa en 359 dimensiones (concentración de distancias); predicción $O(n)$ por imagen. |
| Árbol de decisión único | Varianza alta; RF es su versión con varianza reducida (§6.3). |
| SVM con kernel RBF | Costo de entrenamiento superlineal en $n$; se puede probar con una submuestra. |
| XGBoost | Probable mejora sobre RF, pero agrega muchos hiperparámetros; queda como trabajo futuro. |
| Transfer learning (MobileNet, ResNet) | La consigna recomienda priorizar modelos desde cero; sirve como cota superior en trabajo futuro. |
| MLP sobre los 359 atributos | Sería redundante con RF y SVM sobre las mismas entradas. |
| Regresión del tamaño en cm | Imposible sin escala (§4.2). |

---

## 10. Conclusiones técnicas

1. **Formulación ↔ resultado.** La formulación como clasificación de costo asimétrico se refleja en la evaluación: el mejor modelo por F1 (RF) no es el de menor error caro (CNN).
2. **La estructura de grupos importa.** Hay 3.557 filas pero 1.979 unidades independientes; los intervalos se calculan sobre grupos. Un split por imagen habría inflado las métricas.
3. **Fuente y clase están confundidas** para `regular`. La cifra global (0,842) mezcla un problema binario fácil con uno de 3 clases (0,779). Las conclusiones sobre `regular` se limitan a AgriFreshNET mientras no haya otra fuente.
4. **El tamaño es geometría, no aprendizaje.** $d_n$ es una medida relativa determinada por $D/Z$. Los umbrales funcionan bajo un protocolo de captura; para centímetros se necesita una referencia de escala (§4.4).
5. **Mejoras justificadas por la teoría:**
   - CNN: más épocas, reducción del learning rate y aumento geométrico (rotación, volteo), sin alterar el tono, porque el color es la señal (§8.2).
   - Detectar defectos locales: atributos por región (cuadrantes) o *max pooling* global en lugar de GAP (§8.4).
   - Ensamble RF + CNN o umbral asimétrico, con su costo evaluado en validation (§8.1).
   - Evaluar `regular` en fotos propias y en Mixed para medir el atajo de fuente (§8.3).

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
