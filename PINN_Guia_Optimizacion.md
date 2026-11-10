# Guía de Implementación: Optimización Analítica de PINNs de Segundo Orden

Este documento describe detalladamente:
1. Los cambios arquitectónicos y matemáticos incorporados en el código C++ para habilitar el entrenamiento por **Descenso de Gradiente sin diferencias finitas**.
2. La metodología universal para adaptar este código a **cualquier ecuación diferencial de segundo o primer orden** (lineal o no lineal).

---

## 1. Resumen de Cambios Realizados en el Código

### El Desafío Previo
Tu implementación ya lograba propagar hacia adelante (*forward pass*) los Jacobianos $\nabla_x u$ y el tensor de matrices Hessianas $\nabla_x^2 u$ con exactitud analítica. Sin embargo, para optimizar los parámetros de la red $\theta = \{W, b\}$ con respecto a la pérdida del residuo físico:
$$\mathcal{L}_{res} = \frac{1}{N_{eq}} \sum_{n=1}^{N_{eq}} (residual_n)^2$$
se requería calcular el gradiente respecto a los pesos:
$$\frac{\partial \mathcal{L}_{res}}{\partial \theta} = \frac{2}{N_{eq}} \cdot residual_n \cdot \frac{\partial (residual_n)}{\partial \theta}$$
Como el residuo depende de segundas derivadas espaciales/temporales ($u_{xx}, u_{yy}, u_{tt}$), se necesitaba calcular analíticamente la derivada de un Hessiano respecto a los pesos y sesgos (**Double Backpropagation** o método adjunto de segundo orden).

---

### Archivos Modificados

#### 1. `2_activations.hpp`
Para calcular el gradiente adjunto que atraviesa una función de activación no lineal $\sigma(z)$ cuando en la salida hay segundas derivadas, la regla de la cadena exige hasta la **tercera derivada** de la activación:
* Se añadió `tanh3Derivative(double x)`:
  $$\sigma'''(z) = (1 - \tanh^2 z)(-2 + 6\tanh^2 z)$$
* Se añadió `sigmoid3Derivative(double x)`:
  $$\sigma'''(z) = \sigma'(z)(1 - 6\sigma(z) + 6\sigma^2(z))$$

#### 2. `4_layers.hpp`
* **Estructura `LossGradients`**: Se introdujo un contenedor unificado para transportar los gradientes que la pérdida impone sobre las salidas de una capa:
  ```cpp
  struct LossGradients {
      Eigen::VectorXd d_out;              // ∂L/∂(u)          [n_out]
      Eigen::MatrixXd d_J;                // ∂L/∂(Jacobiano)  [n_out x d]
      std::vector<Eigen::MatrixXd> d_H;   // ∂L/∂(Hessiano)   n_out matrices de [d x d]
  };
  ```
* **Método `forward_derivatives()` en `Layer`**: Unifica la propagación hacia adelante del Jacobiano y del Hessiano, almacenando en cada capa sus tensores de entrada y salida (`input_Jacobian`, `layerJacobian`, `input_Hessian`, `layerHessian`).
* **Método `backward(const LossGradients &grad)`**:
  * **En `Linear`**:
    * Acumula el gradiente del sesgo: $\nabla_b \mathcal{L} = \bar{z}$.
    * Acumula el gradiente de pesos combinando activación, Jacobiano y Hessiano:
      $$\nabla_W \mathcal{L} = \bar{z} a^T + \bar{J}_z J_a^T + \sum_{\alpha, \beta} (\bar{H}_z)_{\alpha\beta} (H_a)_{\alpha\beta}$$
    * Propaga hacia la capa anterior: $\bar{a} = W^T \bar{z}$, $\bar{J}_a = W^T \bar{J}_z$, $\bar{H}_a = W^T \bar{H}_z$.
  * **En `Tanh` y `Sigmoid`**:
    * Propaga el Hessiano adjunto: $\bar{H}_{z, i} = \sigma'(z_i) \bar{H}_{a, i}$.
    * Propaga el Jacobiano adjunto: $\bar{J}_{z, i} = \sigma'(z_i) \bar{J}_{a, i} + \sigma''(z_i) J_{z, i} (\bar{H}_{a, i} + \bar{H}_{a, i}^T)$.
    * Propaga la activación adjunta: $\bar{z}_i = \bar{a}_i \sigma'(z_i) + \sigma''(z_i) \left[ \bar{J}_{a, i} J_{z, i}^T + \text{Tr}(\bar{H}_{a, i}^T H_{z, i}) \right] + \sigma'''(z_i) [J_{z, i} \bar{H}_{a, i} J_{z, i}^T]$.
* **Métodos `zero_grad()` y `step(double learning_rate)`**:
  * Desacoplan la acumulación de gradientes de la actualización de parámetros, permitiendo acumular los gradientes de todos los puntos ($IC$, $BC$ y $EQ$) antes de hacer el paso de optimización por época.

#### 3. `5_network.hpp`
* **`forward_propagation`**: Inicializa las coordenadas de entrada con Jacobiano identidad $I_{d \times d}$ y Hessiano cero, propagándolos capa por capa de forma ordenada.
* **`backward_propagation(LossGradients)`**: Recorre las capas en orden inverso (`rbegin()` a `rend()`) propagando las derivadas adjuntas.
* **`NN::Train`**: Estructura el ciclo de Descenso de Gradiente:
  1. `zero_grad()` al inicio de cada época.
  2. Acumula gradientes de Condiciones Iniciales ($IC$).
  3. Acumula gradientes de Condiciones de Frontera ($BC$).
  4. Acumula gradientes del Residuo Físico ($EQ$).
  5. `step(learning_rate)` para actualizar todos los $W$ y $b$.

---

## 2. Metodología Universal para Cualquier Ecuación Diferencial

Cualquier EDP o EDO se puede resolver siguiendo **4 pasos estándar**.

### Paso 1: Configurar las dimensiones y el mapeo de variables
Define las variables de entrada según el dominio físico:
* Si la entrada es $(x, t) \implies d = 2$:
  * Índice `0` $\to x$
  * Índice `1` $\to t$
* Si la entrada es $(x, y, t) \implies d = 3$:
  * Índice `0` $\to x$
  * Índice `1` $\to y$
  * Índice `2` $\to t$

Para la salida $u$ (índice de salida `0` si es escalar):
* **Valor de salida:** `u = output(0);`
* **Primeras derivadas (Jacobiano):**
  * $u_x$ $\to$ `NetworkJacobian_f(0, 0)`
  * $u_y$ $\to$ `NetworkJacobian_f(0, 1)`
  * $u_t$ $\to$ `NetworkJacobian_f(0, 2)`
* **Segundas derivadas (Hessiano):**
  * $u_{xx}$ $\to$ `NetworkHessianTensor[0](0, 0)`
  * $u_{yy}$ $\to$ `NetworkHessianTensor[0](1, 1)`
  * $u_{tt}$ $\to$ `NetworkHessianTensor[0](2, 2)`
  * $u_{xy}$ $\to$ `NetworkHessianTensor[0](0, 1)`
  * $u_{xt}$ $\to$ `NetworkHessianTensor[0](0, 2)`

---

### Paso 2: Calcular el Residuo
Escribe la ecuación igualada a cero: $\mathcal{R}(u, \nabla u, \nabla^2 u) = 0$.
Evalúa numéricamente el residuo para el punto $n$:
```cpp
double residual_n = /* tu fórmula de la EDP */;
Residual += pow(residual_n, 2);
```

---

### Paso 3: Derivada de la Pérdida respecto al Residuo
Como $\mathcal{L}_{res} = \frac{1}{N_{eq}} \sum (residual_n)^2$, el factor multiplicativo común por la regla de la cadena es:
```cpp
double dL_dres = (2.0 * residual_n) / n_eq;
```

---

### Paso 4: Asignar las Sensibilidades en `LossGradients`
Crea el objeto:
```cpp
LossGradients eq_grads(n_out, d);
```
Y asigna cada término según la derivada parcial de $\mathcal{R}$ respecto a esa componente:

$$\text{Factor en } \texttt{eq\_grads} = dL\_dres \cdot \frac{\partial \mathcal{R}}{\partial (\text{componente})}$$

---

## 3. Catálogo de Ejemplos Prácticos

### Ejemplo A: Ecuación de Onda 2D (Caso Actual)
$$\mathcal{R} = (u_{xx} + u_{yy}) - u_{tt} = 0$$
Entrada: $(x, y, t) \implies d = 3$.

```cpp
// 1. Extraer derivadas
double u_xx = NetworkHessianTensor[0](0, 0);
double u_yy = NetworkHessianTensor[0](1, 1);
double u_tt = NetworkHessianTensor[0](2, 2);

// 2. Residuo
double residual_n = (u_xx + u_yy) - u_tt;
Residual += pow(residual_n, 2);

// 3. Gradiente
double dL_dres = (2.0 * residual_n) / n_eq;
LossGradients eq_grads(n_out, d);
eq_grads.d_H[0](0, 0) = dL_dres * (1.0);  // ∂R/∂u_xx = +1
eq_grads.d_H[0](1, 1) = dL_dres * (1.0);  // ∂R/∂u_yy = +1
eq_grads.d_H[0](2, 2) = dL_dres * (-1.0); // ∂R/∂u_tt = -1

backward_propagation(eq_grads);
```

---

### Ejemplo B: Ecuación de Calor 1D
$$\mathcal{R} = u_t - \alpha u_{xx} = 0$$
Entrada: $(x, t) \implies d = 2$ (`0`: $x$, `1`: $t$). Constante $\alpha = 0.01$.

```cpp
// 1. Extraer derivadas
double u_t  = NetworkJacobian_f(0, 1);
double u_xx = NetworkHessianTensor[0](0, 0);

// 2. Residuo
double residual_n = u_t - alpha * u_xx;
Residual += pow(residual_n, 2);

// 3. Gradiente
double dL_dres = (2.0 * residual_n) / n_eq;
LossGradients eq_grads(n_out, d);
eq_grads.d_J(0, 1)    = dL_dres * (1.0);     // ∂R/∂u_t = +1
eq_grads.d_H[0](0, 0) = dL_dres * (-alpha);  // ∂R/∂u_xx = -alpha

backward_propagation(eq_grads);
```

---

### Ejemplo C: Ecuación de Poisson 2D con Fuente
$$\mathcal{R} = u_{xx} + u_{yy} - f(x, y) = 0$$
Entrada: $(x, y) \implies d = 2$ (`0`: $x$, `1`: $y$).

```cpp
double x = EQ_networkInput(n, 0);
double y = EQ_networkInput(n, 1);
double f = -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);

// 1. Extraer derivadas
double u_xx = NetworkHessianTensor[0](0, 0);
double u_yy = NetworkHessianTensor[0](1, 1);

// 2. Residuo
double residual_n = (u_xx + u_yy) - f;
Residual += pow(residual_n, 2);

// 3. Gradiente
double dL_dres = (2.0 * residual_n) / n_eq;
LossGradients eq_grads(n_out, d);
eq_grads.d_H[0](0, 0) = dL_dres * (1.0); // ∂R/∂u_xx = +1
eq_grads.d_H[0](1, 1) = dL_dres * (1.0); // ∂R/∂u_yy = +1

backward_propagation(eq_grads);
```

---

### Ejemplo D: Ecuación de Onda con Amortiguamiento
$$\mathcal{R} = u_{tt} - c^2 u_{xx} + \gamma u_t = 0$$
Entrada: $(x, t) \implies d = 2$ (`0`: $x$, `1`: $t$).

```cpp
double u_tt = NetworkHessianTensor[0](1, 1);
double u_xx = NetworkHessianTensor[0](0, 0);
double u_t  = NetworkJacobian_f(0, 1);

double residual_n = u_tt - c * c * u_xx + gamma * u_t;
Residual += pow(residual_n, 2);

double dL_dres = (2.0 * residual_n) / n_eq;
LossGradients eq_grads(n_out, d);
eq_grads.d_H[0](1, 1) = dL_dres * (1.0);         // ∂R/∂u_tt = +1
eq_grads.d_H[0](0, 0) = dL_dres * (-c * c);      // ∂R/∂u_xx = -c^2
eq_grads.d_J(0, 1)    = dL_dres * (gamma);       // ∂R/∂u_t  = +gamma

backward_propagation(eq_grads);
```

---

### Ejemplo E: Ecuación de Burgers 1D (EDP No Lineal)
$$\mathcal{R} = u_t + u \cdot u_x - \nu u_{xx} = 0$$
Entrada: $(x, t) \implies d = 2$ (`0`: $x$, `1`: $t$).

Observa que el término $u \cdot u_x$ es no lineal. Por la regla de la cadena:
* $\frac{\partial \mathcal{R}}{\partial u} = u_x$
* $\frac{\partial \mathcal{R}}{\partial u_x} = u$

```cpp
Eigen::VectorXd u_vec = forward_propagation(EQ_networkInput.row(n));
double u = u_vec(0);
double u_t  = NetworkJacobian_f(0, 1);
double u_x  = NetworkJacobian_f(0, 0);
double u_xx = NetworkHessianTensor[0](0, 0);

// 1. Residuo
double residual_n = u_t + u * u_x - nu * u_xx;
Residual += pow(residual_n, 2);

// 2. Gradiente
double dL_dres = (2.0 * residual_n) / n_eq;
LossGradients eq_grads(n_out, d);
eq_grads.d_J(0, 1)    = dL_dres * (1.0);  // ∂R/∂u_t  = 1
eq_grads.d_out(0)     = dL_dres * (u_x);  // ∂R/∂u    = u_x
eq_grads.d_J(0, 0)    = dL_dres * (u);    // ∂R/∂u_x  = u
eq_grads.d_H[0](0, 0) = dL_dres * (-nu);  // ∂R/∂u_xx = -nu

backward_propagation(eq_grads);
```

---

## 4. Resumen de Buenas Prácticas de Entrenamiento

1. **Tasa de Aprendizaje (`learning_rate`)**:
   * Para Descenso de Gradiente estándar con acumulación por época, valores entre $10^{-4}$ y $10^{-3}$ suelen ser un buen punto de partida.
   * Si la pérdida diverge a `NaN`, reduce el `learning_rate` un orden de magnitud (e.g. $10^{-5}$).
2. **Ponderación de Pérdidas**:
   Si notas que el residuo físico se reduce pero las condiciones de frontera se descuidan (o viceversa), puedes ponderar los gradientes multiplicándolos por hiperparámetros $\lambda_{ic}, \lambda_{bc}, \lambda_{eq}$:
   ```cpp
   ic_grads.d_out = (lambda_ic * MSElossDerivative(u_real, u_pred)) / n_ic;
   bc_grads.d_out = (lambda_bc * MSElossDerivative(u_real, u_pred)) / n_bc;
   double dL_dres = (lambda_eq * 2.0 * residual_n) / n_eq;
   ```
3. **Funciones de Activación**:
   * En PINNs de 2do orden se debe usar `Tanh` o `Sigmoid` (evitar `ReLU`, ya que su segunda derivada es idénticamente cero en casi todo el dominio).

