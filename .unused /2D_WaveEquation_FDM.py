# Optimized version of burner.py for maximum performance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Simulation parameters ---
c = 100.0               # Wave speed
sigma = 0.05            # Initial Gaussian pulse width
L = 5.0                 # Spatial domain limit [-L, L]
N = 600                 # Grid resolution (higher = finer)
gamma = 50        # Damping coefficient

# Spatial grid (float32 for speed & lower memory)
x = np.linspace(-L, L, N, dtype=np.float32)
y = np.linspace(-L, L, N, dtype=np.float32)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

# Time step respecting CFL condition
dt = 0.5 * min(dx, dy) / (c * np.sqrt(2))
frames = 200

# --- Build cubic obstacle with double slits (vectorized) ---
wall_dist = 1.5
slit_width = 0.3
slit_spacing = 0.6

# Boolean masks for vertical and horizontal walls
vertical = (np.abs(X) >= wall_dist - dx/2) & (np.abs(X) <= wall_dist + dx/2) & (np.abs(Y) <= wall_dist)
horizontal = (np.abs(Y) >= wall_dist - dy/2) & (np.abs(Y) <= wall_dist + dy/2) & (np.abs(X) <= wall_dist)

# Helper mask: inside a slit (either side of the wall)
in_slit_y = (np.abs(Y - slit_spacing) < slit_width/2) | (np.abs(Y + slit_spacing) < slit_width/2)
in_slit_x = (np.abs(X - slit_spacing) < slit_width/2) | (np.abs(X + slit_spacing) < slit_width/2)

# Combine: obstacle where wall exists BUT not inside a slit
obstacle_mask = (vertical & ~in_slit_y) | (horizontal & ~in_slit_x)

# --- Initial conditions ---
u_prev = 10.0 * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2)).astype(np.float32)
u_prev[obstacle_mask] = 0.0

# Pre‑compute the Laplacian kernel via rolls (still vectorized)
def laplacian(u):
    return (
        (np.roll(u, 1, axis=1) - 2.0 * u + np.roll(u, -1, axis=1)) / dx**2 +
        (np.roll(u, 1, axis=0) - 2.0 * u + np.roll(u, -1, axis=0)) / dy**2
    )

# First time step (leap‑frog scheme)
lap = laplacian(u_prev)
u_curr = u_prev + 0.5 * (c * dt) ** 2 * lap
u_curr[obstacle_mask] = 0.0

# --- Plot setup (unchanged visual style) ---
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(u_curr, extent=[-L, L, -L, L], origin='lower',
               cmap='inferno', vmin=0.0, vmax=0.4)
# Static obstacle overlay (gray mask)
ax.imshow(np.ma.masked_where(~obstacle_mask, obstacle_mask),
          extent=[-L, L, -L, L], origin='lower', cmap='gray_r', alpha=0.7)
ax.set_title("Difracción en Caja con Doble Rendija por Lado", fontsize=12)
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig.colorbar(im, label="Amplitud ($u$)")

# Reuse buffers to avoid repeated allocations
u_old = u_prev.copy()
u_now = u_curr.copy()

# --- Time‑step update (in‑place where possible) ---
def update(frame):
    global u_old, u_now
    lap = laplacian(u_now)
    denom = 1.0 + (gamma * dt / 2.0)
    u_next = (2.0 * u_now - u_old * (1.0 - (gamma * dt / 2.0)) + (c * dt) ** 2 * lap) / denom
    u_next[obstacle_mask] = 0.0
    im.set_array(u_next)
    ax.set_title(rf"Onda Amortiguada (\gamma={gamma}) | Tiempo $t$ = {frame * dt:.2f}", fontsize=12)
    # Swap references instead of copying whole arrays
    u_old, u_now = u_now, u_next
    return [im]

ani = FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
plt.show()
