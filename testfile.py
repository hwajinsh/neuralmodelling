import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

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

# Reward should be a gaussian with integral set to 2 (multiply gaussian by 2)
# Define Gaussian filter and apply it
sigma = 5  # Standard deviation for Gaussian
r = gaussian_filter1d(r, sigma)

# Normalize to sum to 2
r *= 2 / np.sum(r)

# Initialize arrays to store v and delta for all trials
v_all_trials = np.zeros((num_trials, time_steps))  # Predictions for all trials
delta_all_trials = np.zeros((num_trials, time_steps))  # Prediction error for all trials

# Temporal difference learning: update predictions for each trial
for trial in range(num_trials):
    # Temporal difference learning: Update predictions
    for t in range(time_steps - 1):  # Avoid out of range error
        # Calculate prediction error (TD error)
        delta[t] = r[t] + v[t + 1] - v[t]

        # Update weights (w(tau)) for each time step using the learning rule
        for tau in range(t + 1):  # Loop over all previous time steps (stimulus influence)
            w[tau] += learning_rate * delta[t] * u[t - tau]
        
        # Update prediction using weighted sum of stimuli
        v[t + 1] = np.sum(w[:t + 1] * u[:t + 1])  # Weighted sum at step t+1

    
    # Store v and delta for this trial
    v_all_trials[trial] = v.copy()
    delta_all_trials[trial] = delta.copy()

# Prepare 3D meshgrid for plotting
X, Y = np.meshgrid(np.arange(time_steps), np.arange(num_trials))  # X: Time steps, Y: Trials

# Use the already computed delta values (prediction error, stored in delta_all_trials)
# delta_all_trials already contains the TD error (delta)

# Plotting 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(X, Y, delta_all_trials, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('Time Steps')
ax.set_ylabel('Trials')
ax.set_zlabel('Prediction Error (δ)')
ax.set_title('3D Plot of Time Steps, Trials, and Prediction Error (δ)')

# Show the plot
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



