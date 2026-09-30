### Estructura de la Época de Entrenamiento (PINNs)

```text
epoch
│
├── Datos de entrenamiento
│   ├── forward
│   ├── calcular $\hat{u}$
│   └── calcular $L_{\text{data}}$
│
├── Puntos aleatorios del dominio
│   ├── forward
│   ├── calcular $J_{xU} = \frac{\partial \hat{u}}{\partial x}$
│   ├── calcular residual $r$
│   └── calcular $L_{\text{physics}}$
│
├── $L = L_{\text{data}} + \lambda L_{\text{physics}}$
│
├── calcular $\frac{\partial L}{\partial W}$ y $\frac{\partial L}{\partial b}$
│
└── actualizar $W$ y $b$