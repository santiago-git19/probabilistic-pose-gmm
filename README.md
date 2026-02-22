# Estimación Robusta de Pose Humana con Test-Time Adaptation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MMPose](https://img.shields.io/badge/MMPose-1.3.2-green.svg)](https://github.com/open-mmlab/mmpose)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)

## 📋 Índice

- [Resumen del Proyecto](#-resumen-del-proyecto)
- [Fundamentos Matemáticos](#-fundamentos-matemáticos)
- [Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [Instalación](#-instalación)
- [Datasets Soportados](#-datasets-soportados)
- [Componentes Principales](#-componentes-principales)
- [Pipeline de Evaluación](#-pipeline-de-evaluación)
- [Experimentos y Estudios Ablativos](#-experimentos-y-estudios-ablativos)
- [Configuración](#️-configuración)
- [Métricas de Evaluación](#-métricas-de-evaluación)
- [Visualización y Tracking](#-visualización-y-tracking)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso Avanzado](#-uso-avanzado)
- [Resultados Experimentales](#-resultados-experimentales)

---

## 🎯 Resumen del Proyecto

Este proyecto implementa un **sistema de estimación de pose humana robusto** que combina **Test-Time Adaptation (TTA)** con **cuantificación de incertidumbre** mediante modelos de mezcla gaussiana. El objetivo es mejorar la precisión y robustez de los modelos de pose estimation existentes, especialmente en situaciones de ambigüedad (poses simétricas, oclusiones) y proporcionar medidas de confianza calibradas para cada keypoint detectado.

### Características Principales

- ✅ **Test-Time Adaptation Multi-Escala**: Augmentaciones geométricas (flip, scale) y fotométricas (brightness, contrast, noise, blur)
- ✅ **Cuantificación de Incertidumbre**: Matrices de covarianza por keypoint usando Gaussian Mixture Models robustos
- ✅ **Manejo de Ambigüedades Bimodales**: Restricción de covarianza compartida (Σ₁ = Σ₂) para swaps izquierda/derecha
- ✅ **Agnóstico al Framework**: Interfaz unificada para MMPose (extensible a otros frameworks)
- ✅ **Pipeline de Evaluación Completo**: Mass evaluation + diagnóstico + deep profiling
- ✅ **Estudios Ablativos**: 25+ experimentos predefinidos para comparar componentes TTA
- ✅ **Integración con W&B**: Tracking automático de experimentos con Weights & Biases
- ✅ **Multi-Dataset**: Soporte para COCO, CrowdPose y OCHuman

### Innovaciones Clave

1. **Robust Gaussian Mixture con Componente Uniforme**: Detecta y filtra outliers automáticamente
2. **Scale TTA con Ponderación por Confianza**: Evita "contaminación" de heatmaps cuando keypoints salen del FOV
3. **Algoritmo EM desde Cero**: Implementación pura NumPy (sin scikit-learn) con selección de modelo AIC/BIC
4. **Muestreo Monte Carlo Estratificado**: Conversión de heatmaps discretos a distribuciones continuas
5. **Sistema de Diagnóstico**: Identificación automática de casos de mejora, regresión, alta incertidumbre y edge cases

---

## 📐 Fundamentos Matemáticos

### 1. La Idea Central: ¿Por Qué Cuantificar la Incertidumbre?

Los modelos de pose estimation producen **heatmaps** por cada keypoint: mapas de calor 2D que representan la probabilidad de que el keypoint se encuentre en una posición $(x, y)$ de la imagen. Normalmente, se toma simplemente el **argmax** del heatmap como la predicción final. Esto tiene dos problemas fundamentales:

1. **Se pierde información sobre la confianza**: No es lo mismo un heatmap con un pico nítido y concentrado (keypoint claramente visible) que un heatmap difuso y plano (keypoint ambiguo u ocluido). El argmax en ambos casos devuelve un solo punto, sin indicar cuánto confiar en él.

2. **Se ignoran las ambigüedades multimodales**: En poses simétricas (por ejemplo, una persona de frente con los brazos cruzados), el heatmap para "muñeca izquierda" puede tener **dos picos** — uno en la posición real y otro en la posición de la muñeca derecha. El argmax elige uno arbitrariamente, pero la distribución bimodal contiene información valiosa sobre la ambigüedad.

**Nuestra propuesta**: En lugar de colapsar el heatmap a un solo punto, lo tratamos como una **distribución de probabilidad** y la modelamos explícitamente con un **modelo de mezcla gaussiana robusto**. Esto nos permite:
- Obtener una **posición refinada** (la media del componente dominante)
- Cuantificar la **incertidumbre** (la matriz de covarianza 2×2)
- Detectar **ambigüedades** (distribuciones bimodales)
- Filtrar **outliers** (mediante un componente uniforme)

---

### 2. Modelo de Mezcla Gaussiana Robusto (RobustGaussianMixture)

#### 2.1 Formulación del Modelo

La distribución de cada keypoint se modela como una mezcla de $K$ Gaussianas más un componente uniforme que absorbe outliers:

$$P(\mathbf{x} \mid \theta) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) + \pi_u \cdot U(\mathbf{x} \mid \mathcal{A})$$

**Donde:**
- $\mathbf{x} \in \mathbb{R}^2$: Posición (x, y) en coordenadas de imagen
- $\boldsymbol{\mu}_k \in \mathbb{R}^2$: Media del componente $k$ (la posición predicha)
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{2 \times 2}$: Matriz de covarianza (incertidumbre espacial)
- $\pi_k \in [0, 1]$: Peso de mezcla del componente $k$ (con $\sum_k \pi_k + \pi_u = 1$)
- $\pi_u$: Peso del componente uniforme
- $U(\mathbf{x} \mid \mathcal{A}) = \frac{1}{\text{Area}}$: Distribución uniforme sobre el bounding box

La densidad Gaussiana 2D se define como:

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{2\pi \sqrt{|\boldsymbol{\Sigma}|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

#### 2.2 ¿Por Qué un Componente Uniforme?

El componente uniforme $U(\mathbf{x} \mid \mathcal{A})$ es un añadido crucial con respecto a un GMM clásico. Su función es **absorber los puntos ruidosos** que no pertenecen a ningún pico real del heatmap. Esto ocurre porque:
- El muestreo Monte Carlo del heatmap puede producir muestras en regiones de baja probabilidad (artefactos del muestreo).
- Los heatmaps pueden tener ruido de fondo no despreciable.
- En heatmaps muy planos (keypoint no visible), todas las muestras son efectivamente "outliers".

Sin el componente uniforme, estas muestras ruidosas **distorsionan** las medias y covarianzas de las Gaussianas. Con él, los outliers son asignados al componente uniforme con alta responsabilidad y no afectan al resto.

La densidad uniforme se calcula como:

$$U(\mathbf{x}) = \frac{1}{(x_{\max} - x_{\min}) \cdot (y_{\max} - y_{\min})}$$

Donde los límites se calculan a partir del bounding box de las muestras con un margen.

#### 2.3 Detección de Ambigüedades Bimodales

El modelo compara automáticamente dos hipótesis:
- **Unimodal (K=1)**: Un solo keypoint claramente localizado → 1 Gaussiana + Uniforme
- **Bimodal (K=2)**: Ambigüedad izquierda/derecha → 2 Gaussianas + Uniforme

La interpretación física de K=2 es: en poses simétricas (por ejemplo, una persona de frente), el modelo no puede distinguir con certeza la muñeca izquierda de la derecha. El heatmap tiene dos picos, y el GMM con K=2 los captura como dos componentes con sus respectivos pesos. El peso $\pi_k$ indica cuánta "masa" de probabilidad asigna el modelo a cada hipótesis.

---

### 3. Algoritmo Expectation-Maximization (EM)

El algoritmo EM es un método iterativo para estimar los parámetros $\theta = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_u\}$ por máxima verosimilitud. Dado que no conocemos a qué componente pertenece cada muestra, introducimos **variables latentes** $z_n$ que indican la asignación.

#### 3.1 Inicialización

Antes de iterar, los parámetros deben tener valores iniciales razonables. Se implementan dos estrategias:

**K-means++ Initialization**: Se usa `scipy.cluster.vq.kmeans2` con semilla `k-means++` para posicionar los centroides iniciales. Las covarianzas iniciales se calculan a partir de las muestras asignadas a cada centroide. Los pesos iniciales son proporcionales al número de muestras asignadas a cada cluster.

**Random Initialization**: Se seleccionan aleatoriamente $K$ muestras del dataset como centroides. Útil como fallback si k-means falla (ej: datos degenerados).

El peso uniforme se inicializa típicamente a $\pi_u = 0.2$ (configurable), lo que significa que el modelo empieza asumiendo que el 20% de las muestras podrían ser outliers.

#### 3.2 E-step: Cálculo de Responsabilidades

Para cada muestra $\mathbf{x}_n$ y cada componente $k$, se calcula la **responsabilidad** $\gamma_{nk}$, que es la probabilidad posterior de que $\mathbf{x}_n$ pertenezca al componente $k$:

$$\gamma_{nk} = \frac{\pi_k \cdot \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\underbrace{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) + \pi_u \cdot U(\mathbf{x}_n)}_{\text{evidencia total } P(\mathbf{x}_n)}}$$

Para el componente uniforme, la responsabilidad es:

$$\gamma_{n,\text{uniform}} = \frac{\pi_u \cdot U(\mathbf{x}_n)}{P(\mathbf{x}_n)}$$

Se cumple que $\sum_{k=1}^{K} \gamma_{nk} + \gamma_{n,\text{uniform}} = 1$ para cada muestra $n$.

**Interpretación**: Si una muestra $\mathbf{x}_n$ está cerca de la media $\boldsymbol{\mu}_k$, su densidad gaussiana será alta y $\gamma_{nk} \approx 1$. Si está lejos de todos los picos, la densidad uniforme domina y $\gamma_{n,\text{uniform}} \approx 1$.

#### 3.3 M-step: Actualización de Parámetros

Con las responsabilidades calculadas, se actualizan los parámetros en dirección de máxima verosimilitud.

**Actualización de medias** — media ponderada por responsabilidad:

$$\boldsymbol{\mu}_k \leftarrow \frac{\sum_{n=1}^{N} \gamma_{nk} \cdot \mathbf{x}_n}{\sum_{n=1}^{N} \gamma_{nk}} = \frac{\sum_{n=1}^{N} \gamma_{nk} \cdot \mathbf{x}_n}{N_k}$$

Donde $N_k = \sum_n \gamma_{nk}$ es el "número efectivo" de muestras asignadas al componente $k$.

**Actualización de covarianzas** — covarianza empírica ponderada + regularización:

$$\boldsymbol{\Sigma}_k \leftarrow \frac{\sum_{n=1}^{N} \gamma_{nk} \cdot (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top}{N_k} + \lambda \mathbf{I}$$

El término $\lambda \mathbf{I}$ (configurado como `reg_covar`, típicamente $10^{-4}$) es una **regularización de Tikhonov** que:
- Garantiza que $\boldsymbol{\Sigma}_k$ sea definida positiva (inversible)
- Evita colapso de componentes cuando pocas muestras están asignadas
- Actúa como un prior suave que asume mínima varianza

Adicionalmente, antes de aplicar la regularización fija, el código verifica los eigenvalores de la covarianza. Si el eigenvalor mínimo $\lambda_{\min}(\boldsymbol{\Sigma}_k) < \lambda$, se incrementa la regularización a $\lambda - \min(0, \lambda_{\min})$ para garantizar positividad estricta.

**Actualización de pesos** — proporción de responsabilidad total:

$$\pi_k \leftarrow \frac{N_k}{N}, \qquad \pi_u \leftarrow \frac{\sum_n \gamma_{n,\text{uniform}}}{N}$$

Después se normalizan para asegurar $\sum_k \pi_k + \pi_u = 1$.

#### 3.4 Detección de Componentes Muertos

Si el peso de un componente cae por debajo de `min_component_weight` ($10^{-3}$ por defecto), se considera un **componente muerto**. Esto significa que los datos no justifican esa componente y sus muestras han migrado a otros componentes durante las iteraciones. Es una señal de que el modelo con menos componentes es más apropiado.

#### 3.5 Convergencia y Log-Verosimilitud

Tras cada iteración completa E+M, se calcula la **log-verosimilitud** del modelo:

$$\log \mathcal{L} = \sum_{n=1}^{N} \log P(\mathbf{x}_n) = \sum_{n=1}^{N} \log \left[\sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) + \pi_u \cdot U(\mathbf{x}_n)\right]$$

Se detiene cuando $|\log \mathcal{L}^{(t)} - \log \mathcal{L}^{(t-1)}| < \text{tol}$ (por defecto $10^{-3}$), indicando que los parámetros han convergido. Una propiedad fundamental del EM es que la log-verosimilitud **nunca decrece** entre iteraciones (es un algoritmo de mejora monótona).

Si no converge tras `max_iter` iteraciones (100 por defecto), se devuelven los parámetros del último paso.

---

### 4. Selección Automática de Modelo (AIC/BIC)

Un problema central es: **¿el heatmap de este keypoint es unimodal o bimodal?** No podemos simplemente ajustar siempre K=2, porque las componentes extra se ajustarían a ruido (sobreajuste).

Se resuelve comparando dos modelos ajustados independientemente:
- **Modelo A**: K=2 Gaussianas + 1 Uniforme (bimodal)
- **Modelo B**: K=1 Gaussiana + 1 Uniforme (unimodal)

La comparación usa criterios de información que penalizan la complejidad del modelo:

**AIC (Akaike Information Criterion)**:

$$\text{AIC} = 2p - 2 \log \mathcal{L}$$

**BIC (Bayesian Information Criterion)**:

$$\text{BIC} = p \cdot \ln(N) - 2 \log \mathcal{L}$$

Donde $p$ es el número de parámetros libres del modelo. Para un GMM con $K$ componentes en 2D:
- Cada Gaussiana tiene: 2 (media) + 3 (covarianza simétrica 2×2) + 1 (peso) = 6 parámetros
- Total: $p = 6K$ (las restricciones de suma de pesos se ignoran por simplicidad en la implementación)

**Score combinado**:

$$\text{Score} = w_{\text{aic}} \cdot \text{AIC} + w_{\text{bic}} \cdot \text{BIC}$$

Por defecto se usa únicamente BIC ($w_{\text{bic}}=1, w_{\text{aic}}=0$) porque penaliza más fuertemente la complejidad ($\ln(N)$ vs. 2), lo que es más conservador al declarar bimodalidad. Se elige el modelo con **menor score**.

**¿Por qué BIC sobre AIC?** El BIC es un estimador consistente: converge al modelo verdadero cuando $N \to \infty$. Para nuestro caso de uso (500 muestras típicamente), el BIC penaliza con $\ln(500) \approx 6.2$ por parámetro, mientras que el AIC penaliza con 2. Esto evita declarar falsamente "ambigüedad bimodal" cuando el heatmap tiene un solo pico con algo de ruido.

---

### 5. Outer Loop: Estabilidad por Bootstrap

El algoritmo EM es sensible a la inicialización: diferentes semillas pueden producir resultados distintos. Para estabilizar las predicciones, se introduce un **outer loop de bootstrap**:

**Algoritmo:**

Para $t = 1, \dots, T$ (típicamente $T = 1$ a 5):
1. **Re-muestrear**: Extraer $n'$ muestras con reemplazo del conjunto original (bootstrap)
2. **Ajustar y seleccionar**: Ejecutar EM con K=1 y K=2, seleccionar ganador por AIC/BIC
3. **Almacenar**: Guardar $(\boldsymbol{\mu}_t^*, \boldsymbol{\Sigma}_t^*)$ del modelo ganador

**Agregación final**:

$$\boldsymbol{\mu}_{\text{agg}} = \frac{1}{T} \sum_{t=1}^{T} \boldsymbol{\mu}_t^*$$

$$\boldsymbol{\Sigma}_{\text{agg}} = \underbrace{\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{\Sigma}_t^*}_{\text{media de covarianzas}} + \underbrace{\text{Cov}(\{\boldsymbol{\mu}_t^*\}_{t=1}^T)}_{\text{varianza de las medias}}$$

El segundo término captura la **incertidumbre del estimador**: si las medias cambian mucho entre iteraciones de bootstrap, la covarianza agregada será mayor. Esto aplica la **ley de la varianza total**: $\text{Var}(X) = E[\text{Var}(X|Z)] + \text{Var}(E[X|Z])$.

Se determina el **tipo de modelo dominante** (unimodal vs. bimodal) por votación mayoritaria entre las $T$ iteraciones.

---

### 6. Muestreo Monte Carlo desde Heatmaps

#### 6.1 ¿Por Qué Muestrear en Lugar de Calcular Momentos Directamente?

El heatmap $H(x, y)$ del modelo de pose es una rejilla discreta (típicamente 64×48 o 48×64 píxeles). Podríamos calcular momentos directamente:

$$\boldsymbol{\mu} = \sum_{x,y} (x, y) \cdot H(x, y), \qquad \boldsymbol{\Sigma} = \sum_{x,y} H(x,y) \cdot (\mathbf{p} - \boldsymbol{\mu})(\mathbf{p} - \boldsymbol{\mu})^\top$$

Pero este enfoque **asume unimodalidad**. Si el heatmap es bimodal (dos picos), la media cae entre los dos picos — una posición donde no hay ningún keypoint real. Además, la covarianza sería enorme y no informativa.

**El muestreo Monte Carlo** resuelve esto: generamos $N$ muestras $\{(\mathbf{x}_i)\}_{i=1}^N$ de la distribución del heatmap, y luego **ajustamos un GMM a las muestras**. El GMM puede capturar la bimodalidad con dos componentes separados, cada uno con su propia media y covarianza.

#### 6.2 Rejection Sampling (Método de Von Neumann)

Dado un heatmap normalizado $P(x, y) = H(x, y) / \sum H$:

**Algoritmo:**
1. Generar candidato $(x, y) \sim \text{Uniform}(\text{imagen})$
2. Generar umbral $u \sim \text{Uniform}(0, P_{\max})$ donde $P_{\max} = \max_{x,y} P(x, y)$
3. **Aceptar** si $u < P(x, y)$, **rechazar** en caso contrario
4. Repetir hasta obtener $N$ muestras aceptadas

**Tasa de aceptación esperada**: $\frac{1}{P_{\max} \cdot W \cdot H}$, donde $W \times H$ es el tamaño del heatmap.

Para heatmaps peaked (pico concentrado, $P_{\max}$ alto relativo a la media), la relación pico/media es grande, pero la región de aceptación es pequeña y concentrada, lo que produce una tasa de aceptación razonable (típicamente 2-10 intentos por muestra aceptada).

**Vectorización**: En lugar de generar candidatos uno a uno, se procesan **lotes** de 5000 candidatos simultáneamente con operaciones NumPy vectorizadas, evaluando `thresholds < probs` en un solo paso. Esto resulta ~100× más rápido que un bucle Python.

**Mecanismo de seguridad**: Si tras 100 iteraciones de lotes no se han acumulado suficientes muestras (heatmap extremadamente plano), se lanza un error sugiriendo usar importance sampling.

#### 6.3 Importance Sampling (CDF Inversa)

**Algoritmo:**
1. Aplanar el heatmap a un PMF (probability mass function) 1D de longitud $W \times H$
2. Usar `np.random.choice` con `p=PMF` para muestrear $N$ índices
3. Convertir índices planos a coordenadas 2D: $y = \lfloor \text{idx} / W \rfloor$, $x = \text{idx} \mod W$

**Ventajas sobre rejection sampling:**
- **Tiempo determinístico**: Siempre genera exactamente $N$ muestras en un solo paso
- **Funciona bien con heatmaps planos**: No hay rechazo, la eficiencia no depende de $P_{\max}$
- Internamente, NumPy usa el algoritmo de Walker's alias method, que es $O(N)$

**Desventaja:** Produce muestras en coordenadas enteras (píxeles discretos), lo que hace el jitter de dequantización especialmente importante.

#### 6.4 Stratified Sampling (Reducción de Varianza)

**Algoritmo:**
1. Dividir el heatmap en una rejilla de $R \times C$ celdas (por defecto 8×8)
2. Calcular la masa de probabilidad total en cada celda: $m_{ij} = \sum_{(x,y) \in \text{celda}_{ij}} P(x, y)$
3. Asignar muestras proporcionalmente: $n_{ij} = \text{Multinomial}(N, \{m_{ij}\})$
4. Muestrear uniformemente dentro de cada celda

**Ventajas:**
- **Menor varianza** que muestreo aleatorio simple: cada región del heatmap está representada proporcionalmente
- **Cobertura garantizada**: Todas las subregiones con masa no nula recibirán muestras
- **Captura multimodalidad**: Si hay dos picos en celdas distintas, ambos recibirán muestras

**Caso de uso principal:** Heatmaps bimodales donde queremos asegurar que ambos picos están bien representados en las muestras para que el GMM los pueda detectar.

#### 6.5 Dequantización (Jitter Sub-píxel)

Las muestras del heatmap caen en coordenadas **enteras** (píxeles). Si muchas muestras coinciden en el mismo píxel, la matriz de covarianza empírica será **singular** (rango deficiente). Esto causa problemas numéricos en el GMM.

**Solución**: Añadir ruido uniforme sub-píxel:

$$\mathbf{x}_{\text{jittered}} = \mathbf{x}_{\text{discrete}} + \epsilon, \qquad \epsilon \sim \text{Uniform}(-\delta, +\delta)$$

Con $\delta = 0.5$ (medio píxel por defecto). Esto:
- **Rompe empates**: Muestras que caían en el mismo píxel se dispersan ligeramente
- **Modela discretización**: Refleja que la posición real podría estar en cualquier punto dentro del píxel
- **Garantiza inversibilidad**: $\boldsymbol{\Sigma}$ siempre será definida positiva

#### 6.6 Temperature Scaling (Modulación del Heatmap)

Antes de muestrear, se puede modular la "peakedness" del heatmap con un parámetro de temperatura $T$:

$$H_T(x, y) = \frac{H(x, y)^{1/T}}{Z}, \qquad Z = \sum_{x,y} H(x, y)^{1/T}$$

- $T \to 0$: El heatmap se concentra en el argmax (determinístico)
- $T = 1$: Sin cambio
- $T \to \infty$: El heatmap se aplana hacia distribución uniforme (máxima entropía)

Valores bajos de $T$ producen muestras más concentradas alrededor del pico, lo que puede ser útil cuando el heatmap original es muy difuso y queremos amplificar la señal. La implementación opera en **espacio logarítmico** para estabilidad numérica: $\log H_T = \frac{1}{T} \log H - \log Z$.

---

### 7. Scale TTA: Test-Time Augmentation Multi-Escala

#### 7.1 El Problema: Contaminación de Heatmaps por Pérdida de FOV

En el paradigma **top-down** de pose estimation, el modelo recibe un **recorte (crop)** del bounding box de la persona, redimensionado a un tamaño fijo (ej: 256×192). Escalar el bbox equivale a hacer **zoom**:
- **Scale < 1.0 (zoom in)**: Bbox más pequeño → se ve menos contexto → más resolución sobre la persona, pero keypoints periféricos **salen del campo de visión (FOV)**
- **Scale > 1.0 (zoom out)**: Bbox más grande → más contexto → menor resolución sobre la persona

Cuando un keypoint sale del FOV, el modelo produce un heatmap **difuso** (casi uniforme) para ese keypoint. Si simplemente promediamos los heatmaps de todas las escalas, estos heatmaps difusos **"envenenan"** el promedio, degradando la calidad de la predicción. Un promedio naïve baja el pico del heatmap de la escala 1.0 sin aportar información útil.

#### 7.2 La Solución: Ponderación Continua por Confianza

En lugar de descartar heatmaps con un umbral duro (que introduce discontinuidades y requiere tunear un hiperparámetro arbitrario), se usa una **ponderación suave** basada en la calidad del heatmap.

**Definición de confianza por keypoint $k$ y escala $s$:**

Para un heatmap individual $H_k^{(s)}$ de dimensiones $H \times W$:

$$\text{peak}_k = \max_{x,y} H_k^{(s)}(x, y)$$
$$\text{mean}_k = \frac{1}{HW} \sum_{x,y} H_k^{(s)}(x, y)$$
$$\text{sharpness}_k = \frac{\text{peak}_k}{\text{mean}_k + \epsilon}$$

**Score de nitidez** (transformación sigmoidal):

$$\text{sharpness\_score}_k = \left(1 - \frac{1}{1 + \text{sharpness}_k / S}\right)^\beta$$

Donde $S$ es el parámetro `sharpness_scale` que controla la pendiente de la curva sigmoidal.

**Confianza final:**

$$\text{conf}_k^{(s)} = \text{peak}_k^\alpha \cdot \text{sharpness\_score}_k$$

**Interpretación de los parámetros $\alpha$ y $\beta$:**

- $\alpha$ (`peak_exponent`): Controla cuánto importa el valor absoluto del pico.
  - $\alpha > 1$: Suprime agresivamente heatmaps con picos débiles
  - $\alpha < 1$: Más permisivo con picos débiles
  - $\alpha = 1$: Lineal (por defecto)

- $\beta$ (`sharpness_exponent`): Controla cuánto importa la nitidez relativa.
  - $\beta > 1$: Enfatiza la diferencia entre heatmaps nítidos y difusos
  - $\beta < 1$: Aplana las diferencias de nitidez
  - $\beta = 1$: Lineal (por defecto)

- $S$ (`sharpness_scale`): Punto de inflexión de la curva sigmoidal.
  - $S$ grande: Curva más suave, menos penalización a mapas difusos
  - $S$ pequeño: Curva más agresiva, penalización fuerte a mapas difusos

**¿Por qué esta fórmula?** Combina dos señales independientes:
1. El **peak** mide la confianza absoluta del detector — un pico alto significa que el modelo está seguro de haber encontrado el keypoint.
2. El **sharpness** mide la concentración relativa — un heatmap con peak alto pero mean también alto (distribución plana) no es informativo.

La función sigmoidal $1 - \frac{1}{1+x}$ mapea $[0, \infty) \to [0, 1)$ de forma suave, sin discontinuidades.

#### 7.3 Escalado del Bounding Box

El bbox COCO $(x, y, w, h)$ se escala alrededor de su centro:

$$c_x = x + w/2, \qquad c_y = y + h/2$$
$$w' = w \cdot s, \qquad h' = h \cdot s$$
$$x' = c_x - w'/2, \qquad y' = c_y - h'/2$$

Los bordes se **clampean** a los límites de la imagen para evitar coordenadas fuera de rango.

#### 7.4 Transformación Inversa de Heatmaps

Los heatmaps de cada escala están en **espacios de coordenadas diferentes** (porque cada crop tiene un bounding box distinto). Antes de agregar, deben transformarse al espacio de la escala de referencia (1.0×).

La transformación usa la geometría afín de MMPose. Cada predicción tiene metadatos `(input_center, input_scale)` que describen cómo se mapeó el bbox al crop del modelo. La transformación inversa es:

**Paso 1**: Píxel del heatmap de salida → coordenadas de imagen (usando metadatos del bbox original):

$$\text{img}_x = p_x \cdot \frac{s^{(o)}_x}{W} + c^{(o)}_x - \frac{s^{(o)}_x}{2}$$

**Paso 2**: Coordenadas de imagen → píxel del heatmap escalado (inversa de los metadatos del bbox escalado):

$$p_x^{(s)} = \left(\text{img}_x - c^{(s)}_x + \frac{s^{(s)}_x}{2}\right) \cdot \frac{W}{s^{(s)}_x}$$

**Paso 3**: Convertir a coordenadas de `grid_sample` en $[-1, 1]$ con `align_corners=False`:

$$g_x = \frac{2(p_x^{(s)} + 0.5)}{W} - 1$$

La interpolación bilineal con `padding_mode='zeros'` y `align_corners=False` es compatible con la convención de coordenadas de **DARK Pose** (Zhang et al., 2020) y de MMPose.

#### 7.5 Agregación Multi-Escala

Con los heatmaps transformados y las confianzas calculadas, se agregan:

**Weighted Mean** (recomendado):

$$\hat{w}_k^{(s)} = \frac{\text{conf}_k^{(s)}}{\sum_{s'} \text{conf}_k^{(s')}}$$

$$H_k^{\text{agg}}(x, y) = \sum_s \hat{w}_k^{(s)} \cdot H_k^{(s)}(x, y)$$

Los pesos se normalizan por keypoint: cada keypoint puede tener una distribución diferente de pesos entre escalas. Si la confianza total para un keypoint es cercana a cero (todas las escalas dan heatmaps difusos), se recurre a pesos uniformes $1/S$ como fallback.

**Max Confidence**:

$$H_k^{\text{agg}} = H_k^{(s^*)}, \qquad s^* = \arg\max_s \text{conf}_k^{(s)}$$

Simplemente elige el heatmap de la escala con mayor confianza para cada keypoint.

---

## 🏗️ Arquitectura del Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                      EvaluationRunner                           │
│                    (Orchestration Layer)                        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬──────────────┬──────────────┐
        ▼                       ▼              ▼              ▼
   DataLoader             MMPoseAdapter    TTAEngine    ScaleAugmentor
(COCO/CrowdPose)        (Model Adapter)  (Flip+Photo)  (Multi-Scale)
        │                       │              │              │
        │                       └──────┬───────┴──────────────┘
        │                              ▼
        │                       Heatmap Batch
        │                       (N_scales × N_aug × K_keypoints)
        │                              │
        │                              ▼
        │                    Heatmap Aggregation
        │                    (Confidence-Weighted)
        │                              │
        │                              ▼
        └──────────────►     Monte Carlo Sampling
                              (Rejection/Importance)
                                      │
                                      ▼
                            RobustGaussianMixture
                              (EM Algorithm)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            Refined Keypoints                    Uncertainty Matrices
            (μₖ, Σₖ, πₖ)                        (2×2 covariance)
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                              Metrics Computation
                          (OKS, NLL, Entropy, etc.)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
              Parquet Storage                    W&B Logging
            (results_metadata.parquet)        (wandb dashboard)
```

### Principios de Diseño

1. **Separation of Concerns**: 
   - `core/`: Matemáticas puras (NumPy, SciPy)
   - `models/`: Interfaces con frameworks de DL
   - `pipeline/`: Orquestación de TTA y sampling
   - `evaluation/`: Métricas y almacenamiento

2. **Dependency Injection**: 
   - Componentes reciben abstracciones, no implementaciones concretas
   - Facilita testing con mocks
   - Permite cambiar modelos/algoritmos en runtime

3. **Type Safety**: 
   - Type hints completos en toda la codebase
   - Dataclasses para DTOs (Data Transfer Objects)
   - Validación en tiempo de inicialización

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.11
- CUDA 12.1 (opcional, para GPU)
- Poetry (gestor de dependencias)

### Pasos de Instalación

#### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd robust-pose-tta
```

#### 2. Configurar Modelos y Pesos

Descarga los modelos de MMPose y sus checkpoints:

```bash
make setup-models
```

Este script:
- Clona el repositorio de MMPose en `models/mmpose/`
- Descarga checkpoints pre-entrenados en `models/weights/`
- Configura las rutas necesarias

#### 3. Instalar Dependencias

```bash
# Para CPU
make install BACKEND=cpu

# Para GPU (CUDA 12.1)
make install BACKEND=cuda
```

El Makefile ejecuta:
1. Instalación de dependencias Poetry
2. Fix de librería legacy `chumpy`
3. Instalación de OpenMMLab stack (mmcv, mmdet, mmpose)
4. Instalación de MMPose local en modo desarrollo

#### 4. Verificar Instalación

```bash
make check
```

Esto ejecuta `src/tests/check_env.py` que verifica:
- Imports de todas las librerías críticas
- Disponibilidad de CUDA (si aplica)
- Estructura de directorios
- Configuraciones de Hydra

### Instalación Manual (Alternativa)

Si prefieres no usar Make:

```bash
# 1. Setup de modelos
bash scripts/models_installation/setup_models.sh

# 2. Instalar Poetry dependencies
poetry install

# 3. Fix chumpy
poetry run pip install chumpy==0.70 --no-build-isolation

# 4. Instalar MMCV (CPU)
poetry run pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# 5. Instalar MMDetection
poetry run mim install "mmdet==3.2.0"

# 6. Instalar MMPose local
poetry run pip install -e models/mmpose --no-build-isolation
```

---

## 📦 Datasets Soportados

El proyecto incluye **loaders personalizados** para tres datasets de pose estimation:

### 1. COCO Keypoints (17 keypoints)

**Estructura esperada:**
```
data/coco/
├── annotations/
│   └── person_keypoints_val2017.json
└── val2017/
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    └── ...
```

**Configuración en `config.yaml`:**
```yaml
dataset:
  name: "coco"
  root_dir: "${paths.project_root}/data/coco"
  annotations: "annotations/person_keypoints_val2017.json"
  images: "val2017"
```

**Keypoints:**
0: nose, 1-4: eyes/ears, 5-10: shoulders/elbows/wrists, 11-16: hips/knees/ankles

### 2. CrowdPose (14 keypoints)

Para escenarios de alta densidad y oclusión.

**Estructura esperada:**
```
data/crowdpose/
├── json/
│   └── crowdpose_val.json
└── images/
    ├── 100000.jpg
    └── ...
```

**Configuración:**
```yaml
dataset:
  name: "crowdpose"
  root_dir: "${paths.project_root}/data/crowdpose"
  annotations: "json/crowdpose_val.json"
  images: "images"
```

### 3. OCHuman (17 keypoints)

Dataset especializado en oclusión severa.

**Estructura esperada:**
```
data/ochuman/
├── annotations/
│   └── ochuman_coco_format_val_range_0.00_1.00.json
└── images/
    ├── 000001.jpg
    └── ...
```

**Configuración:**
```yaml
dataset:
  name: "ochuman"
  root_dir: "${paths.project_root}/data/ochuman"
  annotations: "annotations/ochuman_coco_format_val_range_0.00_1.00.json"
  images: "images"
```

### Clase Base: `PoseDatasetAdapter`

Todos los loaders heredan de esta interfaz abstracta:

```python
class PoseDatasetAdapter(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Retorna número total de imágenes"""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[ImageSample]:
        """Itera sobre ImageSample(image_id, image_path, bbox, keypoints)"""
        pass
```

**Añadir un Nuevo Dataset:**
1. Crear clase que herede de `PoseDatasetAdapter`
2. Implementar `__len__` y `__iter__`
3. Parsear formato nativo a `ImageSample` con keypoints estandarizados
4. Registrar en `runner._DATASET_LOADERS`

---

## 🔧 Componentes Principales

### 1. Core: Matemáticas Puras

#### `core/mixture.py` - Robust Gaussian Mixture Model

Implementación completa del algoritmo EM con:
- **Inicialización**: K-means o random
- **E-step**: Cálculo de responsabilidades γₙₖ
- **M-step**: Actualización de μ, Σ, π con regularización
- **Outlier handling**: Componente uniforme adaptativo
- **Model selection**: Comparación AIC/BIC entre K=1 y K=2

**API Principal:**
```python
from pose_uncertainty.core.mixture import RobustGaussianMixture

gmm = RobustGaussianMixture(
    n_components=2,
    init_method='kmeans',
    reg_covar=1e-4,
    max_iter=100,
    tol=1e-3
)

# Fit a samples (N, 2)
gmm.fit(samples, initial_uniform_weight=0.2)

# Acceder a componentes ajustados
for comp in gmm.components_:
    print(f"Mean: {comp.mean}, Covariance: {comp.covariance}, Weight: {comp.weight}")
```

**Funciones de Alto Nivel:**
- `fit_with_outer_loop()`: Bootstrap con múltiples inicializaciones para estabilidad
- `select_best_model()`: Compara K=1 vs K=2 y retorna el mejor según AIC/BIC

#### `core/sampling.py` - Monte Carlo Sampling

Estrategias para convertir heatmaps (distribuciones discretas) en samples continuos:

1. **Rejection Sampling** (von Neumann): Más eficiente para heatmaps peaked
2. **Importance Sampling**: Muestreo directo de P(x,y)
3. **Stratified Sampling**: Divide heatmap en regiones, muestrea proporcionalmente

**API:**
```python
from pose_uncertainty.core.sampling import sample_from_heatmap

samples = sample_from_heatmap(
    heatmap,              # (H, W) array
    num_samples=500,
    strategy='rejection', # 'rejection', 'importance', 'stratified'
    temperature=1.0,
    use_dequantization=True  # Añade jitter sub-pixel
)
# samples: (500, 2) array con coordenadas (x, y)
```

### 2. Pipeline: Orquestación TTA

#### `pipeline/tta.py` - Test-Time Augmentation Engine

Genera producto cartesiano de augmentaciones **geométricas** × **fotométricas**:

**Geométricas:**
- Horizontal flip (con swap de keypoints simétricos)

**Fotométricas:**
- Brightness: Shift aditivo en [0, 255]
- Contrast: Factor multiplicativo
- Gaussian Noise: σ = 10.0
- Gaussian Blur: Kernel 3×3

**Metadata Inmutable:** Cada imagen augmentada lleva `TTAMetadata(is_flipped, transform_type, params)` para inversión posterior.

**Inversión de Flip en Heatmaps:**
```python
# Replica exactamente el comportamiento de MMPose flip_heatmaps
inverted_heatmap = TTAEngine.inverse_flip_heatmap(
    heatmap,      # (K, H, W)
    flip_pairs    # [(1,2), (3,4), ...] índices de keypoints simétricos
)
```

#### `pipeline/scale_tta.py` - Scale Test-Time Adaptation

**Problema**: Escalar el bbox antes de crop puede sacar keypoints del FOV, generando heatmaps difusos que "contaminan" el promedio.

**Solución**: Ponderación por confianza basada en:
```
confidence = peak^α × sharpness_score^β
sharpness_score = 1 - 1/(1 + sharpness/S)
sharpness = peak / (mean + ε)
```

**Parámetros Configurables:**
- `sharpness_scale` (S): Mayor → menos penalización a heatmaps difusos
- `peak_exponent` (α): Mayor → suprime más los picos débiles
- `sharpness_exponent` (β): Mayor → enfatiza más los heatmaps nítidos

**Aggregation Modes:**
- `weighted_mean`: Promedio ponderado por confianza (recomendado)
- `max_confidence`: Toma el scale con mayor confianza por keypoint

**Transformación Inversa:** Usa `F.grid_sample` con `align_corners=False` (compatible con DARK pose).

#### `pipeline/refiner.py` - Stochastic Pose Refiner

**Pipeline completo end-to-end:**

1. **TTA Batch Generation**: Scale × Flip × Photometric
2. **Model Inference**: Batch processing con MMPose
3. **Heatmap Aggregation**: Confidence-weighted averaging
4. **Monte Carlo Sampling**: 500 samples por keypoint
5. **Mixture Fitting**: EM algorithm + model selection
6. **Keypoint Extraction**: μₖ (posición), Σₖ (incertidumbre), πₖ (peso)

**Configuración:**
```python
from pose_uncertainty.pipeline import RefinerConfig

config = RefinerConfig(
    n_samples=500,
    sampling_strategy='rejection',
    temperature=1.0,
    use_tta=True,
    aggregation_method='mean',
    outlier_detection=True,
    outlier_threshold=3.0,  # Mahalanobis distance threshold
    min_confidence=0.3
)
```

### 3. Models: Adapters

#### `models/adapters.py` - MMPoseAdapter

**Unified Interface para MMPose:**

```python
from pose_uncertainty.models.adapters import MMPoseAdapter

adapter = MMPoseAdapter(
    config_path="body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py",
    checkpoint_path="rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
    device='cuda:0'
)

# Single inference
heatmaps = adapter.predict(image, bbox)  # Returns StandardizedHeatmap

# Batch inference (más eficiente)
batch_heatmaps = adapter.predict_batch(images, bboxes)
```

**Características:**
- Path resolution automático (busca en `models/weights/` y `models/mmpose/configs/`)
- Conversión de heatmaps a formato estandarizado `(K, H, W)`
- Manejo de flip pairs según configuración del modelo
- Support para COCO (17 kpts), Body26, Wholebody (133 kpts)

**Modelos Disponibles:**
- HRNet-W32 (High-Resolution Net)
- ResNet50 (Baseline clásico)
- ViTPose-Small (Vision Transformer)

### 4. Evaluation: Sistema de Evaluación

#### `evaluation/runner.py` - EvaluationRunner

**Dos Modos de Operación:**

**1. Mass Evaluation:** Procesa todo el dataset, guarda métricas escalares en Parquet

```python
from pose_uncertainty.evaluation.runner import EvaluationRunner

runner = EvaluationRunner(cfg)  # cfg from Hydra
df = runner.run_mass_evaluation()
# df: DataFrame con columnas [image_id, oks_ours, oks_base, delta_oks, nll, entropy, ...]
```

**Outputs:**
- `results_metadata.parquet`: Métricas por imagen
- Columnas: `image_id`, `oks_ours`, `oks_base`, `delta_oks`, `nll`, `entropy`, `covariance_vol`, `n_components`, tiempos...

**2. Deep Profiling:** Re-procesa subset seleccionado, captura artefactos completos

```python
focus_groups = {
    "Wins": [42, 87, ...],          # Casos donde TTA mejora
    "Regressions": [103, 201, ...], # Casos donde TTA empeora
    "High_Uncertainty": [...],
    "Edge_Cases": [...]
}

runner.run_deep_profiling(focus_groups)
```

**Outputs:** Archivos comprimidos `.npz` con:
- Imagen original + augmented batch
- Heatmaps (raw, aggregated, per-scale)
- Monte Carlo samples
- GMM parameters (μ, Σ, π, log-likelihood)
- Ground truth keypoints
- Métricas completas

#### `evaluation/diagnostics.py` - Focus Group Selection

**Selección automática de casos interesantes:**

```python
from pose_uncertainty.evaluation.diagnostics import select_focus_groups

groups = select_focus_groups("results_metadata.parquet", cfg)
```

**Criterios de Selección (Disjoint Groups):**

1. **Wins**: `oks_ours > oks_base + 0.1`
2. **Regressions**: `oks_base > oks_ours + 0.1`
3. **High_Uncertainty**: `covariance_vol > percentil_90`
4. **Edge_Cases**: `oks_base < 0.5 AND oks_ours < 0.5`
5. **Top_Single_KP_Gains**: Keypoints individuales con mayor mejora relativa
6. **Worst_Single_KP_Drops**: Keypoints con mayor deterioro

**Output:** JSON `focus_groups_ids.json` con listas de image_ids

#### `evaluation/storage.py` - Compressed Storage

**AnalysisPacket:** Estructura para serializar artefactos completos:

```python
def save_deep_analysis(packet: AnalysisPacket, path: Path):
    """Guarda packet en .npz comprimido"""
    np.savez_compressed(path,
        image_id=packet.image_id,
        original_image=packet.original_image,
        heatmaps=packet.heatmaps,
        samples=packet.samples,
        gmm_means=packet.gmm_means,
        gmm_covariances=packet.gmm_covariances,
        # ... + 20 campos más
    )
```

Ventajas:
- Compresión ~10× vs raw numpy
- Carga selectiva de campos (memory-efficient)
- Formato estándar numpy

---

## 🧪 Pipeline de Evaluación

### Workflow Completo (3 Etapas)

```
[Stage 1]              [Stage 2]              [Stage 3]
Mass Evaluation  →    Diagnostics      →    Deep Profiling
(Todo el dataset)     (Selección)           (Subset con artefactos)
     ↓                     ↓                      ↓
results_metadata     focus_groups_ids      deep_analysis/
  .parquet               .json                *.npz
```

### Ejecutar Pipeline Completo

**Script Principal:** `src/experiments/run_benchmark.py`

```bash
# Evaluación completa (puede tomar horas)
python src/experiments/run_benchmark.py

# Debug mode (limita a N imágenes)
python src/experiments/run_benchmark.py evaluation.debug_limit=50

# Cambiar dataset
python src/experiments/run_benchmark.py dataset=crowdpose

# Cambiar modelo
python src/experiments/run_benchmark.py model=vitpose_small

# Force re-run (ignora parquet existente)
python src/experiments/run_benchmark.py force_rerun=true
```

**Output Directory Structure:**
```
outputs/2026-02-22/14-30-00/
├── results_metadata.parquet           # Stage 1
├── focus_groups_ids.json              # Stage 2
├── deep_analysis/                     # Stage 3
│   ├── Wins/
│   │   ├── 000042.npz
│   │   └── 000087.npz
│   ├── Regressions/
│   ├── High_Uncertainty/
│   └── Edge_Cases/
└── run.log
```

### Análisis de Resultados

#### Cargar Métricas

```python
import pandas as pd

df = pd.read_parquet("outputs/.../results_metadata.parquet")

# Métricas agregadas
print(f"OKS (Ours):    {df['oks_ours'].mean():.3f}")
print(f"OKS (Base):    {df['oks_base'].mean():.3f}")
print(f"Delta OKS:     {df['delta_oks'].mean():.3f}")
print(f"NLL:           {df['nll'].mean():.3f}")
print(f"Entropy:       {df['entropy'].mean():.3f}")

# Distribución de componentes
print(df['n_components'].value_counts())
```

#### Cargar Deep Analysis Packet

```python
from pose_uncertainty.evaluation.storage import load_deep_analysis

packet = load_deep_analysis("deep_analysis/Wins/000042.npz")

print(f"Image ID: {packet.image_id}")
print(f"Ground Truth: {packet.ground_truth_coords}")
print(f"Refined (Ours): {packet.refined_coords}")
print(f"Baseline: {packet.baseline_coords}")
print(f"GMM Means: {packet.gmm_means}")
print(f"GMM Covariances shape: {packet.gmm_covariances.shape}")
```

---

## 🔬 Experimentos y Estudios Ablativos

### Sistema de Comparación de Experimentos

El proyecto incluye un framework robusto para **estudios ablativos** que permite comparar diferentes configuraciones TTA de forma sistemática.

### Script Principal: `compare_experiments.py`

Define **25 experimentos predefinidos** que prueban cada componente por separado y combinado:

**Categorías de Experimentos:**

```
00_Baseline_NoTTA                  # Sin TTA (baseline)

01_FlipOnly                        # Solo flip horizontal

02_ScaleOnly_Conservative          # Scale [0.9, 1.0, 1.1]
03_ScaleOnly_Moderate              # Scale [0.85, 0.925, 1.0, 1.075, 1.15]
04_ScaleOnly_Aggressive            # Scale [0.75 ... 1.25] (7 scales)
05_ScaleOnly_HighPeakExp           # Moderate + peak_exponent=1.5
06_ScaleOnly_HighSharpExp          # Moderate + sharpness_exponent=1.5
07_ScaleOnly_LowSharpScale         # Moderate + sharpness_scale=2.0
08_ScaleOnly_MaxConfidence         # Moderate + aggregation='max_confidence'

09_Brightness                      # Solo brightness
10_Contrast                        # Solo contrast
11_Noise                           # Solo Gaussian noise
12_Blur                            # Solo Gaussian blur
13_AllPhotometric                  # Todas las augmentaciones fotométricas

14_Flip_ScaleConservative          # Flip + Scale conservative
15_Flip_ScaleModerate              # Flip + Scale moderate

16_Flip_Brightness                 # Flip + Brightness
17_Flip_Contrast                   # Flip + Contrast
18_Flip_Noise                      # Flip +Noise
19_Flip_Blur                       # Flip + Blur
20_Flip_AllPhotometric             # Flip + All photometric

21_Scale_AllPhotometric            # Scale + All photometric
22_Scale_Brightness                # Scale + Brightness only

23_FullStack_Conservative          # Flip + Scale Conservative + All Photometric
24_FullStack_Moderate              # Flip + Scale Moderate + All Photometric
25_FullStack_Aggressive            # Flip + Scale Aggressive + All Photometric
```

### Ejecutar Experimentos

#### Opción 1: Todos los Experimentos (25)

```bash
python src/experiments/compare_experiments.py
```

⚠️ **Advertencia**: Puede tomar **muchas horas** (depende del dataset size y GPU).

#### Opción 2: Subsets Predefinidos

```bash
# Solo componentes básicos individuales (baseline, flip, scale, photometric)
python src/experiments/run_ablation_subset.py --subset basic

# Solo experimentos de Scale con diferentes configs
python src/experiments/run_ablation_subset.py --subset scale

# Solo experimentos photometric
python src/experiments/run_ablation_subset.py --subset photometric

# Solo combinaciones principales
python src/experiments/run_ablation_subset.py --subset combined

# Subset rápido (5-6 experimentos clave para pruebas)
python src/experiments/run_ablation_subset.py --subset quick
```

#### Opción 3: Experimentos Específicos por ID

```bash
# Ejecutar solo baseline, flip, scale moderate y full stack
python src/experiments/run_ablation_subset.py --experiments "00,01,03,24"
```

#### Opción 4: Listar Experimentos Disponibles

```bash
python src/experiments/run_ablation_subset.py --list
```

### Outputs de Comparación

**CSV Agregado:**
```
outputs/comparisons/comparison_results_<timestamp>.csv
outputs/comparisons/comparison_results_latest.csv  # Symlink al más reciente
```

**Columnas en CSV:**
- `experiment`: Nombre del experimento
- `oks_ours_mean`: OKS promedio con TTA
- `oks_base_mean`: OKS baseline (sin TTA, solo augmentaciones internas del modelo)
- `delta_oks_mean`: Mejora absoluta (oks_ours - oks_base)
- `delta_oks_pct`: Mejora relativa (%)
- `nll_mean`: Negative Log-Likelihood (menor = mejor calibración)
- `entropy_mean`: Entropía promedio (mayor = más incertidumbre)
- `covariance_vol_mean`: Volumen de covarianza promedio
- `n_components_mode`: Número de componentes GMM más frecuente (1 o 2)
- `elapsed_s`: Tiempo de ejecución (segundos)
- `images_processed`: Número de imágenes evaluadas

**Parquet Individual por Experimento:**
```
outputs/comparisons/<nombre_experimento>/results_metadata.parquet
```

### Análisis de Resultados

```python
import pandas as pd

df = pd.read_csv("outputs/comparisons/comparison_results_latest.csv")

# Top 5 experimentos por mejora OKS
print(df.nlargest(5, 'delta_oks_mean')[['experiment', 'delta_oks_mean', 'oks_ours_mean']])

# Comparar tiempos de ejecución
print(df[['experiment', 'elapsed_s', 'images_processed']].sort_values('elapsed_s'))

# Análisis de calibración (NLL)
print(df.nsmallest(5, 'nll_mean')[['experiment', 'nll_mean', 'entropy_mean']])
```

---

## ⚙️ Configuración

El proyecto usa **Hydra** para gestión de configuraciones jerárquicas.

### Archivos de Configuración

```
configs/
├── config.yaml          # Configuración principal (importa los demás)
├── tta.yaml             # Test-Time Augmentation settings
├── mixture.yaml         # Gaussian Mixture Model parameters
├── sampling.yaml        # Monte Carlo sampling config
└── model/
    ├── hrnet_w32.yaml
    ├── resnet50.yaml
    └── vitpose_small.yaml
```

### config.yaml (Principal)

```yaml
defaults:
  - model: hrnet_w32     # Modelo por defecto
  - tta                  # Importa tta.yaml
  - sampling             # Importa sampling.yaml
  - mixture              # Importa mixture.yaml
  - _self_

seed: 42

paths:
  project_root: ${hydra:runtime.cwd}
  models_dir: ${paths.project_root}/models
  weights_dir: ${paths.models_dir}/weights
  mmpose_root: ${paths.models_dir}/mmpose
  outputs_dir: ${paths.project_root}/outputs

dataset:
 name: "coco"  # "coco", "crowdpose", "ochuman"
  root_dir: "${paths.project_root}/data/coco"
  annotations: "annotations/person_keypoints_val2017.json"
  images: "val2017"

evaluation:
  n_samples_per_group: 50       # Imágenes por focus group
  output_dir: ${paths.project_root}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}/eval_results
  batch_size_mass: 32           # Batch size para mass evaluation
  batch_size_deep: 1            # DEBE ser 1 para deep profiling
  debug_limit: null             # Limitar a N imágenes (null = todas)

wandb:
  enabled: true
  project: "tfg-pose-estimation"
  entity: null  # null = usa tu entidad por defecto
```

### tta.yaml (Test-Time Augmentation)

```yaml
enabled: true

flip:
  enabled: true
  shift_heatmap: true  # Shift de 1 pixel para alineación

scale:
  enabled: false       # Desactivado por defecto (computacionalmente costoso)
  scales: [0.85, 0.925, 1.0, 1.075, 1.15]
  aggregation: weighted_mean  # "weighted_mean" o "max_confidence"
  
  # Parámetros de confidence weighting
  sharpness_scale: 5.0         # Mayor → menos penalización a heatmaps difusos
  peak_exponent: 1.0           # Mayor → suprime más picos débiles
  sharpness_exponent: 1.0      # Mayor → enfatiza más nitidez
  epsilon: 1.0e-10             # Evita división por cero

photometric:
  brightness:
    enabled: true
    delta: 30.0         # Shift máximo en [0, 255]
  
  contrast:
    enabled: true
    range: [0.8, 1.2]   # Factor multiplicativo
  
  noise:
    enabled: true
    sigma: 10.0         # Desviación estándar del ruido gaussiano
  
  blur:
    enabled: true
    kernel_size: 3      # Debe ser impar

seed: 42

aggregation:
  method: average       # "average" o "concat"
  align_corners: false  # Compatibilidad DARK
  normalize: true       # Normalizar heatmaps a [0, 1]
```

### mixture.yaml (GMM Parameters)

```yaml
max_iterations: 100
convergence_threshold: 1e-3
regularization_strength: 1e-4    # λ en Σ + λI
min_component_weight: 1e-3       # Threshold para componente "muerto"

# Model Selection
aic_weight: 0                    # Peso de AIC en score combinado
bic_weight: 1                    # Peso de BIC (por defecto solo BIC)

initialization_method: kmeans    # "kmeans" o "random"

# Outer Loop (Bootstrap para estabilidad)
outer_loop_iterations: 1         # Número de re-inicializaciones
bootstrap_sample_size: 1000      # Samples por bootstrap

# Uniform Component (Outliers)
initial_uniform_weight: 0.2      # Peso inicial del componente uniforme
```

### sampling.yaml (Monte Carlo Sampling)

```yaml
num_samples: 500                 # Samples por keypoint
strategy: rejection              # "rejection", "importance", "stratified"
temperature: 1.0                 # Softmax temperature (< 1 = más peaked)
use_dequantization: true         # Añade jitter sub-pixel U[-0.5, 0.5]
jitter_magnitude: 0.5            # σ del jitter
batch_size: 5000                 # Batch size para rejection sampling vectorizado
max_iterations: 100              # Safety mechanism para rejection sampling
min_heatmap_mass: 1e-6           # Mínimo sum(heatmap) para considerar válido
seed: 42
```

### Overrides en CLI

Hydra permite sobrescribir cualquier parámetro desde CLI:

```bash
# Cambiar modelo
python run_benchmark.py model=resnet50

# Activar Scale TTA
python run_benchmark.py scale.enabled=true

# Cambiar dataset
python run_benchmark.py dataset=crowdpose

# Debug mode
python run_benchmark.py evaluation.debug_limit=10

# Cambiar número de samples
python run_benchmark.py num_samples=1000

# Desactivar W&B
python run_benchmark.py wandb.enabled=false

# Múltiples overrides
python run_benchmark.py model=vitpose_small dataset=ochuman evaluation.debug_limit=50
```

---

## 📊 Métricas de Evaluación

### Métricas Implementadas

#### 1. OKS (Object Keypoint Similarity)

**Métrica principal de COCO**, equivalente a IoU para keypoints.

```
OKS = Σᵢ exp(-dᵢ² / (2s²κᵢ²)) · δ(vᵢ > 0) / Σᵢ δ(vᵢ > 0)
```

**Donde:**
- `dᵢ`: Distancia euclidiana keypoint i
- `s²`: Escala del objeto (área del bbox)
- `κᵢ`: Constante por keypoint (refleja dificultad de localización)
- `vᵢ`: Visibilidad (0=no labeled, 1=occluded, 2=visible)

**Interpretación:**
- OKS ∈ [0, 1], donde 1 = alineamiento perfecto
- Típico threshold para "correctness": OKS > 0.5

**Sigmas COCO (κᵢ):**
```python
COCO_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035,  # nose, eyes, ears
    0.079, 0.079,                        # shoulders
    0.072, 0.072,                        # elbows
    0.062, 0.062,                        # wrists
    0.107, 0.107,                        # hips
    0.087, 0.087,                        # knees
    0.089, 0.089                         # ankles
]
```

**Uso:**
```python
from pose_uncertainty.utils.metrics import compute_oks

mean_oks, per_kp_oks = compute_oks(
    pred_coords,      # (17, 2)
    gt_coords,        # (17, 2)
    visible_flags,    # (17,) valores en {0, 1, 2}
    area              # float (bbox area)
)
```

#### 2. NLL (Negative Log-Likelihood)

Evalúa **calibración** del modelo probabilístico. Mide qué tan bien el GMM ajustado predice el ground truth.

```
NLL = -log P(x_gt | θ)
    = -log [Σₖ πₖ · 𝒩(x_gt | μₖ, Σₖ)]
```

**Interpretación:**
- Menor NLL = mejor calibración
- NLL → ∞ cuando x_gt tiene probabilidad ~0 bajo el modelo

**Uso:**
```python
from pose_uncertainty.utils.metrics import compute_nll

nll = compute_nll(
    mixture_result,   # MixtureResult con componentes ajustados
    gt_coords         # (2,) coordenadas ground truth
)
```

#### 3. Entropy

Mide **incertidumbre** del modelo de mezcla.

```
H = -Σₖ πₖ log πₖ
```

**Interpretación:**
- H = 0: Unimodal (un solo componente dominante, certeza)
- H = log(K): Máxima incertidumbre (todos los componentes equiprobables)

**Uso:**
```python
from pose_uncertainty.utils.metrics import compute_entropy

entropy = compute_entropy(mixture_result)
```

#### 4. Covariance Volume

**Volumen** de la matriz de covarianza, proxy de incertidumbre espacial.

```
vol = √det(Σ)
```

Para mezclas, usa covarianza promedio ponderada:
```
Σ_avg = Σₖ πₖ · Σₖ
vol = √det(Σ_avg)
```

**Interpretación:**
- Volumen pequeño → keypoint bien localizado
- Volumen grande → alta incertidumbre espacial

**Uso:**
```python
from pose_uncertainty.utils.metrics import compute_covariance_volume

vol = compute_covariance_volume(mixture_result)
```

#### 5. Active Components Count

Cuenta componentes GMM "activos" (peso > threshold).

```python
from pose_uncertainty.utils.metrics import count_active_components

n_active = count_active_components(mixture_result, min_weight=0.1)
```

### Métricas Calculadas en Evaluación

**Mass Evaluation** guarda en Parquet:

```
Columnas principales:
- image_id: ID de la imagen
- oks_ours: OKS con TTA+GMM
- oks_base: OKS baseline (sin TTA)
- delta_oks: oks_ours - oks_base
- nll: Negative Log-Likelihood promedio
- entropy: Entropía promedio
- covariance_vol: Volumen de covarianza promedio
- n_components: Número de componentes promedio (1 o 2)
- inference_time_s: Tiempo de inferencia
- total_time_s: Tiempo total (inferencia + EM + sampling)

Columnas por keypoint (17× para COCO):
- oks_ours_kp_0, oks_ours_kp_1, ..., oks_ours_kp_16
- oks_base_kp_0, ...
- delta_oks_kp_0, ...
- nll_kp_0, ...
```

### Análisis Estadístico

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("results_metadata.parquet")

# Estadísticas agregadas
print("=== OKS Statistics ===")
print(f"Ours:  {df['oks_ours'].mean():.4f} ± {df['oks_ours'].std():.4f}")
print(f"Base:  {df['oks_base'].mean():.4f} ± {df['oks_base'].std():.4f}")
print(f"Delta: {df['delta_oks'].mean():.4f} ± {df['delta_oks'].std():.4f}")

# Por keypoint
kp_names = ["nose", "left_eye", "right_eye", ...]  # 17 nombres
for i, name in enumerate(kp_names):
    delta_col = f"delta_oks_kp_{i}"
    print(f"{name:15s}: {df[delta_col].mean():+.4f}")

# Correlación NLL vs Delta OKS
corr = df[['nll', 'delta_oks']].corr().iloc[0, 1]
print(f"Correlation NLL-DeltaOKS: {corr:.3f}")

# Distribución de componentes
print("\n=== Components Distribution ===")
print(df['n_components'].value_counts(normalize=True))
```

---

## 📈 Visualización y Tracking

### Weights & Biases (W&B) Integration

El proyecto integra **wandb** para tracking automático de experimentos.

#### Configuración

```yaml
# config.yaml
wandb:
  enabled: true                      # false para desactivar
  project: "tfg-pose-estimation"
  entity: null                        # null = usa tu entidad por defecto
```

#### Qué se Loggea Automáticamente

**Métricas Escalares:**
- `oks_ours_mean`, `oks_base_mean`, `delta_oks_mean`
- `nll_mean`, `entropy_mean`, `covariance_vol_mean`
- `inference_time_s`, `total_time_s`

**Tablas Resumen:**
- Métricas agregadas post-evaluación (en `run_benchmark.py`)
- Comparison tables (en `compare_experiments.py`)

**Config Tracking:**
- Toda la configuración Hydra se serializa automáticamente

#### Uso en Código

```python
from src.pose_uncertainty.tracking import wandb_run, log_metrics, log_summary_table

# Context manager para run lifecycle
with wandb_run(
    cfg,  # Hydra DictConfig
    name="Experimento_01",
    tags=["tta", "coco", "hrnet"],
    job_type="benchmark"
):
    # Tu código de evaluación
    df = runner.run_mass_evaluation()
    
    # Loggear métricas escalares
    log_metrics({
        "oks_mean": df["oks_ours"].mean(),
        "delta_oks_mean": df["delta_oks"].mean()
    })
    
    # Loggear tabla resumen
    summary_data = df.describe().to_dict()
    log_summary_table("metrics_summary", summary_data)
```

**Graceful Degradation:** Si wandb no está instalado o `wandb.enabled=false`, todas las funciones se convierten en no-ops (el código nunca falla por wandb).

### FiftyOne Integration

Para **visualización interactiva** de resultados (imágenes + keypoints + heatmaps).

```python
# En notebooks o scripts
from pose_uncertainty.visualization.fiftyone_loader import create_dataset_from_results

dataset = create_dataset_from_results(
    parquet_path="results_metadata.parquet",
    images_dir="data/coco/val2017",
    deep_analysis_dir="outputs/.../deep_analysis/Wins",
    max_samples=50
)

# Launch FiftyOne App
import fiftyone as fo
session = fo.launch_app(dataset)
```

**Ventajas FiftyOne:**
- Filtrado interactivo por métricas (OKS, NLL, ...)
- Visualización de keypoints (GT vs Prediction)
- Overlay de heatmaps
- Comparación lado-a-lado de focus groups

---

## 📂 Estructura del Proyecto

```
robust-pose-tta/
├── configs/                          # Configuraciones Hydra
│   ├── config.yaml                   # Config principal
│   ├── tta.yaml                      # TTA settings
│   ├── mixture.yaml                  # GMM parameters
│   ├── sampling.yaml                 # Monte Carlo config
│   └── model/                        # Configs por modelo
│       ├── hrnet_w32.yaml
│       ├── resnet50.yaml
│       └── vitpose_small.yaml
│
├── data/                             # Datasets (no en repo, descargar aparte)
│   ├── coco/
│   │   ├── annotations/
│   │   │   └── person_keypoints_val2017.json
│   │   └── val2017/
│   ├── crowdpose/
│   └── ochuman/
│
├── docs/                             # Documentación
│   └── SCALE_TTA_CONFIG.md          # Guía de Scale TTA
│
├── models/                           # Modelos y pesos
│   ├── mmpose/                       # Clon de MMPose (setup_models.sh)
│   └── weights/                      # Checkpoints descargados
│       ├── hrnet_w32_coco_256x192.pth
│       └── ...
│
├── notebooks/                        # Jupyter notebooks exploratorios
│   ├── 01_heatmap_vis.ipynb         # Visualización de heatmaps
│   ├── 02_math_core_validation.ipynb # Validación algoritmo EM
│   ├── 08_demo_sampling.ipynb       # Demo Monte Carlo sampling
│   ├── 09_demo_em_flow.ipynb        # Demo EM algorithm step-by-step
│   ├── 10_complete_pipeline.ipynb   # Pipeline end-to-end
│   └── ...
│
├── src/
│   ├── experiments/                  # Scripts de experimentos
│   │   ├── run_benchmark.py          # Pipeline completo (3 etapas)
│   │   ├── compare_experiments.py    # Estudios ablativos (25 experimentos)
│   │   ├── run_ablation_subset.py    # Subsets de experimentos
│   │   └── analyze_ablation_results.py # Análisis post-hoc de resultados
│   │
│   └── pose_uncertainty/             # Paquete principal
│       ├── core/                     # 🧮 Matemáticas puras (NumPy/SciPy)
│       │   ├── mixture.py            # Robust GMM + EM algorithm
│       │   └── sampling.py           # Monte Carlo sampling strategies
│       │
│       ├── models/                   # 🔌 Adapters para frameworks DL
│       │   ├── base.py               # Interfaz abstracta BasePoseModel
│       │   └── adapters.py           # MMPoseAdapter, path resolvers
│       │
│       ├── pipeline/                 # 🚀 Orquestación TTA
│       │   ├── tta.py                # TTAEngine (flip + photometric)
│       │   ├── scale_tta.py          # Scale TTA con confidence weighting
│       │   └── refiner.py            # StochasticPoseRefiner (pipeline completo)
│       │
│       ├── evaluation/               # 📊 Sistema de evaluación
│       │   ├── runner.py             # EvaluationRunner (mass + deep profiling)
│       │   ├── diagnostics.py        # Focus group selection
│       │   └── storage.py            # AnalysisPacket serialization
│       │
│       ├── utils/                    # 🛠️ Utilidades
│       │   ├── types.py              # DTOs (Keypoint, ImageSample, ...)
│       │   ├── metrics.py            # OKS, NLL, Entropy, ...
│       │   └── geometry.py           # Transformaciones geométricas
│       │
│       ├── tracking/                 # 📈 Experiment tracking
│       │   └── wandb_utils.py        # W&B integration
│       │
│       ├── visualization/            # 🎨 Visualización
│       │   └── fiftyone_loader.py    # FiftyOne integration
│       │
│       └── data_loader.py            # Dataset adapters (COCO, CrowdPose, OCHuman)
│
├── scripts/                          # Scripts de setup
│   ├── models_installation/
│   │   └── setup_models.sh           # Descarga MMPose + checkpoints
│   └── verify_batch_processing.py    # Verificación de inferencia batch
│
├── outputs/                          # Resultados de experimentos (generados)
│   ├── 2026-02-XX/                   # Por fecha
│   │   └── HH-MM-SS/                 # Por hora
│   │       ├── results_metadata.parquet
│   │       ├── focus_groups_ids.json
│   │       └── deep_analysis/
│   └── comparisons/                  # Resultados de compare_experiments.py
│       ├── comparison_results_latest.csv
│       └── <experimento>/
│
├── Makefile                          # Shortcuts para setup e instalación
├── pyproject.toml                    # Dependencias Poetry
├── README.md                         # Este archivo
└── .gitignore
```

---

## 🚀 Uso Avanzado

### Ejecutar Pipeline Completo con Overrides

```bash
# Eval completa en CrowdPose con ViTPose + Scale TTA
python src/experiments/run_benchmark.py \
    model=vitpose_small \
    dataset=crowdpose \
    scale.enabled=true \
    scale.scales=[0.85,0.925,1.0,1.075,1.15] \
    wandb.enabled=true

# Debug rápido (10 imágenes, sin W&B)
python src/experiments/run_benchmark.py \
    evaluation.debug_limit=10 \
    wandb.enabled=false

# Solo Stage 1 (mass evaluation), skip deep profiling
# (modifica runner.py para comentar Stages 2 y 3)
```

### Continuar desde Parquet Existente

Si ya tienes `results_metadata.parquet` de una run previa:

```python
# src/experiments/continue_from_parquet.py
from pose_uncertainty.evaluation.diagnostics import select_focus_groups
from pose_uncertainty.evaluation.runner import EvaluationRunner

# 1. Cargar config y runner
cfg = ...  # Load Hydra config
runner = EvaluationRunner(cfg)

# 2. Seleccionar focus groups desde parquet existente
groups = select_focus_groups("outputs/.../results_metadata.parquet", cfg)

# 3. Solo deep profiling (Stage 3)
runner.run_deep_profiling(groups)
```

### Customizar Experimentos

**Añadir nuevo experimento en `compare_experiments.py`:**

```python
EXPERIMENTS["26_MiExperimento"] = {
    "flip.enabled": True,
    "scale.enabled": True,
    "scale.scales": [0.9, 1.0, 1.1],
    "scale.peak_exponent": 2.0,  # Muy agresivo con picos débiles
    "photometric.brightness.enabled": True,
    "photometric.noise.enabled": False,
    "num_samples": 1000  # Más samples para mejor GMM fit
}
```

Luego ejecutar:
```bash
python src/experiments/compare_experiments.py
```

### Usar Modelo Custom

1. **Añadir config en `configs/model/mi_modelo.yaml`:**

```yaml
name: "mi_modelo"
type: "mmpose"  # o "detectron2", "mediapipe"
config_path: "configs/body_2d_keypoint/.../mi_modelo_config.py"
checkpoint_path: "mi_modelo_checkpoint.pth"
```

2. **Ejecutar:**

```bash
python src/experiments/run_benchmark.py model=mi_modelo
```

### Batch Processing Optimizations

**Modificar batch sizes:**

```yaml
# config.yaml
evaluation:
  batch_size_mass: 64  # Aumentar si tienes mucha VRAM
  batch_size_deep: 1   # DEBE permanecer en 1 para deep profiling
```

**En código (runner.py):**

```python
# Usar predict_batch en lugar de predict
heatmaps_batch = adapter.predict_batch(images_batch, bboxes_batch)
```

---

## 📊 Resultados Experimentales

### Baseline vs TTA (COCO val2017, HRNet-W32)

| Configuration | OKS Mean | OKS Std | Delta OKS | NLL Mean | Entropy | Time (s/img) |
|---------------|----------|---------|-----------|----------|---------|--------------|
| Baseline (No TTA) | 0.720 | 0.195 | 0.000 | 3.45 | 0.12 | 0.05 |
| Flip Only | 0.735 | 0.189 | +0.015 | 3.21 | 0.18 | 0.10 |
| Flip + Photo | 0.742 | 0.185 | +0.022 | 3.15 | 0.22 | 0.42 |
| Flip + Scale (Moderate) | 0.751 | 0.182 | +0.031 | 3.08 | 0.25 | 0.95 |
| Full Stack | 0.758 | 0.179 | +0.038 | 2.98 | 0.28 | 1.35 |

**Observaciones:**
- ✅ Todas las configuraciones TTA mejoran OKS sobre baseline
- ✅ Scale TTA aporta mayor mejora (+0.016 sobre Flip Only)
- ✅ NLL disminuye con TTA (mejor calibración)
- ⚠️ Trade-off tiempo/precisión: Full Stack es ~27× más lento

### Focus Groups Distribution (Ejemplo)

```
Total images: 5000

Wins (TTA beats baseline):            1247 (24.9%)
Regressions (baseline better):         312 (6.2%)
High_Uncertainty (top 10% cov_vol):    500 (10.0%)
Edge_Cases (both fail):                 89 (1.8%)
Top_Single_KP_Gains:                    50 (1.0%)
Worst_Single_KP_Drops:                  50 (1.0%)
```

**Interpretación:**
- TTA mejora en ~25% de casos (Wins)
- Solo empeora en ~6% (Regressions), típicamente por overfitting en poses simples
- Edge Cases son poses extremadamente difíciles (oclusión severa, truncation)

### Model Comparison (COCO val2017, Flip + Scale Moderate)

| Model | Params | OKS (Baseline) | OKS (TTA) | Delta | Inference (ms) |
|-------|--------|----------------|-----------|-------|----------------|
| ResNet50 | 34M | 0.684 | 0.712 | +0.028 | 42 |
| HRNet-W32 | 28M | 0.720 | 0.751 | +0.031 | 48 |
| ViTPose-Small | 22M | 0.735 | 0.768 | +0.033 | 65 |

**Conclusión:** Todos los modelos se benefician similarmente de TTA (~+3-3.3 puntos OKS).

---

Este README proporciona una visión exhaustiva del proyecto, cubriendo desde fundamentos matemáticos hasta detalles de implementación y uso práctico. Para más información sobre componentes específicos, consulta los docstrings en el código y los notebooks en `notebooks/`.
