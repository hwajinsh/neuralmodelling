import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Define parameters for the TD model
num_trials = 100
time_steps = 300
reward_time = 200
stimulus_time = 100
learning_rate = 0.1

# Initialize variables
u = np.zeros(time_steps)  # Stimulus
r = np.zeros(time_steps)  # Reward
v = np.zeros(time_steps)  # Prediction
delta = np.zeros(time_steps)  # Prediction error (TD error)
w = np.zeros(time_steps)  # Weights

# Set stimulus and reward signals
u[stimulus_time] = 1  # Stimulus presented at time step 100
r[reward_time] = 1    # Reward presented at time step 200

# Reward should be a Gaussian with integral set to 2 (multiply Gaussian by 2)
sigma = 5  # Standard deviation for Gaussian
r = gaussian_filter1d(r, sigma)

# Normalize to sum to 2
r *= 2 / np.sum(r)

# Initialize arrays to store v and delta for all trials
v_all_trials = np.zeros((num_trials, time_steps))  # Predictions for all trials
delta_all_trials = np.zeros((num_trials, time_steps))  # Prediction error for all trials

# Temporal difference learning: update predictions for each trial
for trial in range(num_trials):
    for t in range(1, time_steps):  # Start from t=1 to match the inner logic
        # Compute current prediction as weighted sum of past stimuli
        v[t] = np.sum(w[:t] * u[t-1::-1])  # Equivalent to the "value_helper" logic
        
        # Compute delta_v (change in prediction)
        delta_v = v[t] - v[t-1]
        
        # Compute TD error (delta)
        delta[t] = r[t] + delta_v  # Same as: δ[t] = r[t] + V[t] - V[t-1]
        
        # Update weights for all past time steps τ
        for tau in range(t):
            w[tau] += learning_rate * delta[t] * u[t - tau - 1]

    # Store predictions and TD error for this trial
    v_all_trials[trial] = v.copy()
    delta_all_trials[trial] = delta.copy()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Create a mesh grid for time steps and trials
x, y = np.meshgrid(np.arange(time_steps), np.arange(num_trials))

# 3D Surface Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface of delta (TD error)
ax.plot_surface(x, y, delta_all_trials, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('Time Steps')
ax.set_ylabel('Trials')
ax.set_zlabel('Prediction Error (δ)')
ax.set_title('3D Plot of Time Steps, Trials, and Prediction Error (δ)')

plt.show()


# Plotting 2D
fig, ax = plt.subplots(5, 2, figsize=(12, 10))

# "Before Training" (Early Trial) - trial 0
ax[0, 0].plot(u, label="u (Stimulus)", color="black")
ax[1, 0].plot(r, label="r (Reward)", color="blue")
ax[2, 0].plot(v_all_trials[0], label="v (Prediction)", color="orange")  # Early Trial
ax[3, 0].plot(np.diff(v_all_trials[0], prepend=0), label="Δv", color="purple")  # Early Δv
ax[4, 0].plot(delta_all_trials[0], label="δ (TD Error)", color="red")  # Early δ

# Titles and labels for before training
ax[0, 0].set_title("Before Training")
for i in range(5):
    ax[i, 0].legend(loc="upper right")
    ax[i, 0].set_xlim([0, time_steps])

# "After Training" (Late Trial) - last trial
ax[0, 1].plot(u, label="u (Stimulus)", color="black")
ax[1, 1].plot(r, label="r (Reward)", color="blue")
ax[2, 1].plot(v_all_trials[-1], label="v (Prediction)", color="orange")  # Late Trial
ax[3, 1].plot(np.diff(v_all_trials[-1], prepend=0), label="Δv", color="purple")  # Late Δv
ax[4, 1].plot(delta_all_trials[-1], label="δ (TD Error)", color="red")  # Late δ

# Titles and labels for after training
ax[0, 1].set_title("After Training")
for i in range(5):
    ax[i, 1].legend(loc="upper right")
    ax[i, 1].set_xlim([0, time_steps])

# Display plot
fig.tight_layout()
plt.show()



