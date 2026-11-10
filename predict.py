#!/usr/bin/env python3
"""
predict.py
----------
Carga los parámetros de la red neuronal (PINN) desde parametros.txt,
predice el comportamiento del dominio:
    x_limits = (-1, 1)
    y_limits = (-1, 1)
    t_limits = (0, 10)
y genera una animación sincronizada en vista 3D isométrica y 2D superior (heatmap inferno).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class PINNModel:
    """Modelo de Red Neuronal implementado en NumPy a partir de parametros.txt."""

    def __init__(self, filepath="parametros.txt"):
        self.layers = []  # Lista de dicts: {'type': 'linear'/'act', ...}
        self.load_parameters(filepath)

    def load_parameters(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo de parámetros: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()
        i = 0
        current_layer_idx = None
        raw_layers = {}

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            if line.startswith("> Capa"):
                parts = line.split(":")
                capa_idx = int(parts[0].replace("> Capa", "").strip())
                act = parts[1].strip() if len(parts) > 1 else ""
                current_layer_idx = capa_idx
                raw_layers[capa_idx] = {"act": act, "W": None, "b": None}
                i += 1
                continue

            if line.startswith("W:"):
                w_rows = []
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if l.startswith("b:") or l.startswith("> Capa"):
                        break
                    if l:
                        # Extraer flotantes separados por coma
                        vals = [float(x) for x in l.strip(",").split(",") if x.strip()]
                        if vals:
                            w_rows.append(vals)
                    i += 1
                raw_layers[current_layer_idx]["W"] = np.array(w_rows, dtype=np.float32)
                continue

            if line.startswith("b:"):
                b_vals = []
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if l.startswith("> Capa") or l.startswith("W:"):
                        break
                    if l:
                        vals = [float(x) for x in l.strip(",").split(",") if x.strip()]
                        b_vals.extend(vals)
                    i += 1
                raw_layers[current_layer_idx]["b"] = np.array(b_vals, dtype=np.float32)
                continue

            i += 1

        # Construir la secuencia ordenada de capas
        sorted_indices = sorted(raw_layers.keys())
        self.layers = []
        for idx in sorted_indices:
            info = raw_layers[idx]
            if info["W"] is not None and info["b"] is not None:
                self.layers.append({
                    "id": idx,
                    "type": "linear",
                    "W": info["W"],
                    "b": info["b"]
                })
            if info["act"]:
                self.layers.append({
                    "id": idx,
                    "type": "activation",
                    "fn": info["act"]
                })

        print("=== Arquitectura de la Red Cargada ===")
        for idx, layer in enumerate(self.layers):
            if layer["type"] == "linear":
                print(f"[{idx}] Capa {layer['id']} Lineal: W={layer['W'].shape}, b={layer['b'].shape}")
            else:
                print(f"[{idx}] Capa {layer['id']} Activación: {layer['fn']}")
        print("======================================\n")

    def forward(self, X):
        """
        Propagación hacia adelante.
        X: array de dimensión (N, 3) con columnas [x, y, t]
        Retorna: array de dimensión (N,) con la altura predicha.
        """
        h = X
        for layer in self.layers:
            if layer["type"] == "linear":
                h = h @ layer["W"].T + layer["b"]
            elif layer["type"] == "activation":
                if layer["fn"] == "tanh":
                    h = np.tanh(h)
                elif layer["fn"] == "relu":
                    h = np.maximum(0, h)
                elif layer["fn"] == "sigmoid":
                    h = 1.0 / (1.0 + np.exp(-h))
                else:
                    raise ValueError(f"Activación no soportada: {layer['fn']}")
        return h.squeeze(-1)

    def predict_grid(self, xx, yy, t):
        """
        Evalúa la red en una malla 2D para un instante de tiempo t fijo.
        xx, yy: arrays 2D de meshgrid (H, W)
        t: escalar float
        Retorna: array 2D de dimensiones (H, W) con los valores de altura.
        """
        pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, t, dtype=np.float32)], axis=-1)
        z = self.forward(pts)
        return z.reshape(xx.shape)


def generate_animation(model,
                       x_limits=(-10, 10),
                       y_limits=(-10, 10),
                       t_limits=(0, 10),
                       grid_size=60,
                       n_frames=100,
                       fps=20,
                       output_gif="animacion_pinn.gif",
                       preview_png="prediccion_pinn_preview.png",
                       show_gui=False):
    """
    Genera y guarda la animación comparativa en 3D isométrica y 2D superior.
    """
    print(f"Calculando predicciones para el dominio:")
    print(f"  x in {x_limits}")
    print(f"  y in {y_limits}")
    print(f"  t in {t_limits}")
    print(f"  Resolución espacial: {grid_size}x{grid_size}")
    print(f"  Fotogramas: {n_frames} ({fps} FPS, duración aprox. {n_frames/fps:.1f}s)")

    x = np.linspace(x_limits[0], x_limits[1], grid_size, dtype=np.float32)
    y = np.linspace(y_limits[0], y_limits[1], grid_size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    t_vals = np.linspace(t_limits[0], t_limits[1], n_frames, dtype=np.float32)

    # Precomputar todas las matrices Z(x, y, t) para una animación fluida
    print("Precomputando estados temporales...")
    z_frames = []
    for t in t_vals:
        z = model.predict_grid(X, Y, t)
        z_frames.append(z)
    z_frames = np.array(z_frames)  # shape: (n_frames, grid_size, grid_size)

    # Escala global consistente para el colormap inferno y el eje Z
    vmin = float(z_frames.min())
    vmax = float(z_frames.max())
    # Margen para el eje Z
    z_margin = (vmax - vmin) * 0.1 if vmax > vmin else 0.1
    z_min_plot = vmin - z_margin
    z_max_plot = vmax + z_margin

    print(f"Rango de altura predicha: [{vmin:.4f}, {vmax:.4f}]")

    # Guardar snapshot estático con 4 instantes representativos (preview)
    fig_snap, axes_snap = plt.subplots(2, 4, figsize=(12, 8))
    sample_indices = np.linspace(0, n_frames - 1, 4, dtype=int)
    for col_idx, f_idx in enumerate(sample_indices):
        t_curr = t_vals[f_idx]
        z_curr = z_frames[f_idx]

        # Fila 0: 2D superior
        im = axes_snap[0, col_idx].imshow(
            z_curr, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
            origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bicubic"
        )
        axes_snap[0, col_idx].set_title(f"t = {t_curr:.2f} s", fontsize=11)
        axes_snap[0, col_idx].set_xlabel("x")
        axes_snap[0, col_idx].set_ylabel("y")

        # Fila 1: 3D vista isométrica aproximada con contornos o proyección
        ax3 = fig_snap.add_subplot(2, 4, 5 + col_idx, projection="3d")
        axes_snap[1, col_idx].remove()  # reemplazar el eje 2d por 3d
        ax3.plot_surface(
            X, Y, z_curr, cmap="inferno", vmin=vmin, vmax=vmax,
            rstride=1, cstride=1, edgecolor="none", antialiased=True
        )
        ax3.view_init(elev=35.264, azim=45)
        ax3.set_zlim(z_min_plot, z_max_plot)
        ax3.set_title(f"t = {t_curr:.2f} s", fontsize=11)
        ax3.set_xlabel("x", labelpad=2)
        ax3.set_ylabel("y", labelpad=2)
        ax3.set_zlabel("z", labelpad=2)

    fig_snap.suptitle("Wave Evolution prediction in domain [-1, 1] × [0, 10]", fontsize=14)
    plt.tight_layout()
    fig_snap.savefig(preview_png, dpi=150)
    plt.close(fig_snap)
    print(f"Resumen estático guardado en: {preview_png}")

    # Configuración de la figura principal de la animación
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 8))

    # Subplot 1: Vista 3D Isométrica
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # Proyección isométrica estándar: elevación = arctan(1/sqrt(2)) ≈ 35.264°, azimut = 45°
    ax3d.view_init(elev=35.264, azim=45)
    ax3d.set_xlim(x_limits)
    ax3d.set_ylim(y_limits)
    ax3d.set_zlim(z_min_plot, z_max_plot)
    ax3d.set_xlabel("X", fontsize=11, labelpad=8)
    ax3d.set_ylabel("Y", fontsize=11, labelpad=8)
    ax3d.set_zlabel("Z", fontsize=11, labelpad=8)
    title3d = ax3d.set_title("t = 0.00 s", fontsize=13, pad=12)

    # Subplot 2: Vista 2D Superior
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_xlim(x_limits)
    ax2d.set_ylim(y_limits)
    ax2d.set_xlabel("X", fontsize=11)
    ax2d.set_ylabel("Y", fontsize=11)
    ax2d.set_aspect("equal", "box")
    title2d = ax2d.set_title("t = 0.00 s", fontsize=13)

    # Elementos iniciales
    surf = [ax3d.plot_surface(
        X, Y, z_frames[0], cmap="inferno", vmin=vmin, vmax=vmax,
        rstride=1, cstride=1, edgecolor="none", antialiased=True, alpha=0.95
    )]

    im = ax2d.imshow(
        z_frames[0],
        extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic"
    )

    # Barra de color compartida
    cbar = fig.colorbar(im, ax=[ax3d, ax2d], orientation="horizontal", fraction=0.046, pad=0.12, shrink=0.7)
    cbar.set_label("predicted height $z = u(x, y, t)$", fontsize=11)

    fig.suptitle("Wave evolution prediction", fontsize=15, fontweight="bold", y=0.98)

    def update_frame(frame_idx):
        t_curr = t_vals[frame_idx]
        z_curr = z_frames[frame_idx]

        # Actualizar superficie 3D
        surf[0].remove()
        surf[0] = ax3d.plot_surface(
            X, Y, z_curr, cmap="inferno", vmin=vmin, vmax=vmax,
            rstride=1, cstride=1, edgecolor="none", antialiased=True, alpha=0.95
        )
        title3d.set_text(f"t = {t_curr:.2f} s")

        # Actualizar heatmap 2D
        im.set_data(z_curr)
        title2d.set_text(f"t = {t_curr:.2f} s")

        if frame_idx % 10 == 0 or frame_idx == n_frames - 1:
            print(f"  Renderizando fotograma {frame_idx + 1}/{n_frames} (t = {t_curr:.2f}s)...")

        return surf[0], im

    print("\nGenerando animación...")
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=int(1000 / fps),
        blit=False
    )

    print(f"Guardando animación en: {output_gif} (esto puede tardar unos segundos)...")
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_gif, writer=writer)
    print(f"¡Animación guardada exitosamente en {output_gif}!")

    if show_gui:
        print("Mostrando visualización interactiva...")
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PINN Predictor y Animación 3D/2D.")
    parser.add_argument("--params", type=str, default="parametros.txt", help="Ruta al archivo parametros.txt")
    parser.add_argument("--output", type=str, default="animacion_pinn.gif", help="Archivo GIF de salida")
    parser.add_argument("--preview", type=str, default="prediccion_pinn_preview.png", help="Archivo PNG de previsualización")
    parser.add_argument("--frames", type=int, default=100, help="Número de fotogramas temporales")
    parser.add_argument("--fps", type=int, default=20, help="Fotogramas por segundo")
    parser.add_argument("--grid-size", type=int, default=60, help="Resolución de la cuadrícula espacial (NxN)")
    parser.add_argument("--show", action="store_true", help="Mostrar ventana gráfica en vivo con plt.show()")

    args = parser.parse_args()

    # Cargar red neuronal
    model = PINNModel(args.params)

    # Generar predicción y animación
    generate_animation(
        model=model,
        x_limits=(-10.0, 10.0),
        y_limits=(-10.0, 10.0),
        t_limits=(0.0, 10.0),
        grid_size=args.grid_size,
        n_frames=args.frames,
        fps=args.fps,
        output_gif=args.output,
        preview_png=args.preview,
        show_gui=args.show
    )


if __name__ == "__main__":
    main()

