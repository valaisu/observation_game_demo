import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# THE ORIGINAL LORENZ PARAMETERS (guaranteed chaos)
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def derivatives(state, t):
    x, y, z = state
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    return np.array([dx, dy, dz])

def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Initial conditions - tiny perturbation matters!
x = 0.0
y = 1.0
z = 1.05

state = np.array([x, y, z])
dt = 0.01
t = 0

history = [state.copy()]
max_history = 10000

# Set up the plot
fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Lorenz Attractor')
ax1.set_xlim(-30, 30)
ax1.set_ylim(-30, 30)
ax1.set_zlim(-10, 50)

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlabel('Time')
ax2.set_ylabel('X', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_title('X Over Time')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlabel('Time')
ax3.set_ylabel('Y', color='tab:blue')
ax3.tick_params(axis='y', labelcolor='tab:blue')
ax3.set_title('Y Over Time')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_xlabel('Time')
ax4.set_ylabel('Z', color='tab:purple')
ax4.tick_params(axis='y', labelcolor='tab:purple')
ax4.set_title('Z Over Time')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

line_3d, = ax1.plot([], [], [], 'c-', linewidth=0.5, alpha=0.7)
point_3d, = ax1.plot([], [], [], 'ro', markersize=6)

line_x, = ax2.plot([], [], 'r-', linewidth=1)
line_y, = ax3.plot([], [], 'b-', linewidth=1)
line_z, = ax4.plot([], [], color='purple', linewidth=1)

title_text = fig.suptitle('', fontsize=12)

def update(frame):
    global state, t, history
    
    for _ in range(10):
        state = rk4_step(state, t, dt)
        t += dt
        history.append(state.copy())
        
        if len(history) > max_history:
            history.pop(0)
    
    hist_array = np.array(history)
    time_array = np.arange(len(history)) * dt
    
    line_3d.set_data(hist_array[:, 0], hist_array[:, 1])
    line_3d.set_3d_properties(hist_array[:, 2])
    
    point_3d.set_data([state[0]], [state[1]])
    point_3d.set_3d_properties([state[2]])
    
    line_x.set_data(time_array, hist_array[:, 0])
    line_y.set_data(time_array, hist_array[:, 1])
    line_z.set_data(time_array, hist_array[:, 2])
    
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    
    title_text.set_text(f'x = {state[0]:.2f}  |  y = {state[1]:.2f}  |  z = {state[2]:.2f}')
    
    return line_3d, point_3d, line_x, line_y, line_z

anim = FuncAnimation(fig, update, frames=None, interval=20, blit=False)

plt.show()