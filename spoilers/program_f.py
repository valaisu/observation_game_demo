"""

This is actually quite good system for chaos I think. The Z dimension 
experiences high and sudden peaks, while the other two variables dance 
in a circle. 

For example, I could say that the two dancers are temperature and
wind speed, while the last one is cloud coverage.

This is good for a few reasons: I could make a system where the 
rains come when cloud coverage is high enough. For a naive solution,
the player might be able to prepare for rain like one day, before it
starts. (not actually sure what the utility is here)

Also, I don't know if it is good that the system is chaotic, because
It might actually be a bit hard to create an accurate RL system for them.
But maybe this could also be a good thing, don't know.

33*49 = 1617


"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Rössler system parameters (known to be chaotic)
a = 0.2
b = 0.2
c = 5.7

# Scale to observable weather units
scale_T = 3.0
scale_W = 2.0
scale_C = 15.0
offset_T = 20.0

def derivatives(state, t):
    T, W, C = state
    
    # Convert to Rössler variables
    x = (T - offset_T) / scale_T
    y = W / scale_W
    z = C / scale_C
    
    # Rössler equations
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    
    # Convert back to weather units
    dT = dx * scale_T
    dW = dy * scale_W
    dC = dz * scale_C
    
    return np.array([dT, dW, dC])

def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Initial conditions
T = 20.0
W = -10.0
C = 0.5

state = np.array([T, W, C])
dt = 0.02
t = 0

history = [state.copy()]
max_history = 10000

# Set up the plot
fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Wind Speed (m/s)')
ax1.set_zlabel('Cloud Cover')
ax1.set_title('Phase Space Trajectory')
ax1.set_xlim(-10, 60)
ax1.set_ylim(-25, 20)
ax1.set_zlim(-5, 350)

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Temperature (°C)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_title('Temperature Over Time')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlabel('Time')
ax3.set_ylabel('Wind Speed (m/s)', color='tab:blue')
ax3.tick_params(axis='y', labelcolor='tab:blue')
ax3.set_title('Wind Speed Over Time')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_xlabel('Time')
ax4.set_ylabel('Cloud Cover', color='tab:purple')
ax4.tick_params(axis='y', labelcolor='tab:purple')
ax4.set_title('Cloud Cover Over Time')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

line_3d, = ax1.plot([], [], [], 'c-', linewidth=0.5, alpha=0.6)
point_3d, = ax1.plot([], [], [], 'yo', markersize=8)

line_T, = ax2.plot([], [], 'r-', linewidth=1)
line_W, = ax3.plot([], [], 'b-', linewidth=1)
line_C, = ax4.plot([], [], color='purple', linewidth=1)

title_text = fig.suptitle('', fontsize=12)

def update(frame):
    global state, t, history
    
    for _ in range(15):
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
    
    line_T.set_data(time_array, hist_array[:, 0])
    line_W.set_data(time_array, hist_array[:, 1])
    line_C.set_data(time_array, hist_array[:, 2])
    
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    
    title_text.set_text(f'T = {state[0]:.1f}°C  |  W = {state[1]:.1f} m/s  |  C = {state[2]:.1f}')
    
    return line_3d, point_3d, line_T, line_W, line_C

anim = FuncAnimation(fig, update, frames=None, interval=20, blit=False)

plt.show()