import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Simulation parameters
N = 10000       # Number of particles
L = 10.0        # Box size
dt = 0.01       # Time step
T = 300         # Temperature
kB = 1.0
m = 1.0
cell_size = 1.0
dim = 2
n_cells = int(L / cell_size)
positions = np.random.rand(N, dim) * L
# Initialization
velocities = np.random.normal(0, np.sqrt(kB*T/m), size=(N, dim))
velocities -= np.mean(velocities, axis=0)  # Remove net momentum
# Rotation matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

theta = np.pi / 2  # 90-degree rotation
rot = rotation_matrix(theta)
position_history = []  # stores positions at each time step
n_save_steps = 10
# Main MPCD loop
n_steps = 1000
for step in range(n_steps):
    # Streaming step
    positions += velocities * dt
    positions %= L  # Periodic boundaries

    # Random grid shift (optional, improves isotropy)
    shift = np.random.uniform(0, cell_size, size=2)
    shifted_positions = (positions + shift) % L

    # Assign to cells
    cell_indices = (shifted_positions // cell_size).astype(int)
    cell_indices = cell_indices[:, 0] + n_cells * cell_indices[:, 1]

    # Collision step
    for cell in np.unique(cell_indices):
        idx = np.where(cell_indices == cell)[0]
        if len(idx) > 1:
            v_cm = np.mean(velocities[idx], axis=0)
            v_rel = velocities[idx] - v_cm
            velocities[idx] = v_cm + v_rel @ rot.T

    if step % n_save_steps == 0:
        position_history.append(positions.copy())

    # Optional: print progress
    if step % 100 == 0:
        print(f"Step {step} completed.")
# Plot final speed distribution
speeds = np.linalg.norm(velocities, axis=1)
v = np.linspace(0, np.max(speeds), 300)
f_MB = (m / (kB*T)) * v * np.exp(-m * v**2 / (2 * kB * T))
plt.hist(speeds, bins=100, density=True, alpha=0.6, label="Simulated")
plt.plot(v, f_MB, 'r--', label="Maxwell-Boltzmann (2D)")
plt.xlabel("Speed")
plt.ylabel("Probability density")
plt.title("MPCD Speed Distribution")
plt.legend()
plt.grid()
plt.show()
position_history=np.array(position_history)



# Assuming position_history has shape (10, 10000, 2)
# You MUST have position_history already defined

fig, ax = plt.subplots(figsize=(6, 6))

# Initialize with first time step (avoid empty scatter!)
scat = ax.scatter(position_history[0][:, 0], position_history[0][:, 1], s=0.1)

# Set fixed axis limits based on all positions
ax.set_xlim(np.min(position_history[..., 0]), np.max(position_history[..., 0]))
ax.set_ylim(np.min(position_history[..., 1]), np.max(position_history[..., 1]))

def update(frame):
    scat.set_offsets(position_history[frame])
    ax.set_title(f"Time Step: {frame} at T={T}")
    return scat,

# IMPORTANT: remove init_func to avoid blank frame
ani = FuncAnimation(fig, update, frames=range(100), interval=500, blit=True)

plt.show()
ani.save("particle_animation.mp4", fps=10, dpi=300)

