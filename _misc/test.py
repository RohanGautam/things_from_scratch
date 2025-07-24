import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Grid parameters
nx, ny = 100, 100  # number of points in x and y
dx = dy = 1.0  # grid spacing
dt = 0.5 * dx  # time step (CFL condition)

# Create grid
x = np.linspace(-50, 50, nx)
y = np.linspace(-50, 50, ny)
X, Y = np.meshgrid(x, y)


# Initialize phi (signed distance function)
# Negative inside, positive outside
def initialize_phi(X, Y, center_x=0, center_y=0, radius=10):
    return np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2) - radius


# Calculate spatial derivatives using central differences
def compute_derivatives(phi, dx, dy):
    # Forward differences
    phi_x_forward = np.roll(phi, -1, axis=1) - phi
    phi_y_forward = np.roll(phi, -1, axis=0) - phi

    # Backward differences
    phi_x_backward = phi - np.roll(phi, 1, axis=1)
    phi_y_backward = phi - np.roll(phi, 1, axis=0)

    return phi_x_forward, phi_y_forward, phi_x_backward, phi_y_backward


# Time evolution following Godunov's scheme
def evolve(phi, dt, dx, dy):
    # Compute derivatives
    phi_x_p, phi_y_p, phi_x_n, phi_y_n = compute_derivatives(phi, dx, dy)

    # Unit speed propagation (F = 1)
    F = 1.0

    # Godunov's scheme for Hamilton-Jacobi equation
    if F > 0:
        phi_x = np.maximum(np.maximum(phi_x_n, 0) ** 2, np.minimum(phi_x_p, 0) ** 2)
        phi_y = np.maximum(np.maximum(phi_y_n, 0) ** 2, np.minimum(phi_y_p, 0) ** 2)
    else:
        phi_x = np.maximum(np.minimum(phi_x_n, 0) ** 2, np.maximum(phi_x_p, 0) ** 2)
        phi_y = np.maximum(np.minimum(phi_y_n, 0) ** 2, np.maximum(phi_y_p, 0) ** 2)

    # Update phi
    phi_new = phi - dt * F * np.sqrt(phi_x + phi_y)

    return phi_new


# Initialize
phi = initialize_phi(X, Y)

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(8, 8))
plt.close()


def animate(frame):
    global phi

    # Update phi
    phi = evolve(phi, dt, dx, dy)

    # Clear previous plot
    ax.clear()

    # Plot the zero level set
    ax.contour(X, Y, phi, levels=[0], colors="b")
    ax.contour(X, Y, phi, levels=np.linspace(-50, 50, 20), colors="gray", alpha=0.5)

    # Set plot limits and labels
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title(f"Time step: {frame}")
    ax.set_aspect("equal")
    ax.grid(True)


# Create animation
anim = animation.FuncAnimation(fig, animate, frames=100, interval=50)
anim.save("out.gif")
plt.show()
