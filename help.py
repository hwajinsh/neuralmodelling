import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Updated Parameters
num_trials = 100  # Total number of trials
tau_max = 250     # Duration of each trial
stimulus_time = 100  # Time of stimulus (τ = 100)
reward_time = 200    # Time of reward (τ = 200)
epsilon = 1        # Learning rate

# Initialize weights and prediction errors
w = np.zeros(tau_max)  # Delayed expectation weights
delta_history = np.zeros((num_trials, tau_max))  # Store prediction errors per trial

# Simulate trials
for trial in range(num_trials):
    v = np.convolve(w, np.eye(1, tau_max)[0], mode='full')[:tau_max]  # Total future reward expectation
    for tau in range(tau_max):
        # Define stimulus and reward signals
        u_tau = 1 if tau == stimulus_time else 0
        r_tau = 1 if tau == reward_time else 0

        # Prediction error
        delta_tau = r_tau + v[tau + 1]  - v[tau]

        # Update weights only if stimulus is present or prediction error occurs
        if tau <= reward_time:  # Only past and current influence learning
            w += epsilon * delta_tau * np.roll(np.eye(1, tau_max)[0], tau)

        # Store prediction error
        delta_history[trial, tau] = delta_tau

# Prepare data for the refined 3D plot
X = np.arange(num_trials)  # Trials
Y = np.arange(tau_max)     # Time within each trial (τ)
X, Y = np.meshgrid(X, Y)
Z = delta_history.T        # Prediction errors as a function of trials and τ

# Create 3D plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot for prediction error
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('Trials')
ax.set_ylabel('Time within Trial (τ)')
ax.set_zlabel('Prediction Error (δ)')
ax.set_title('3D Plot of Prediction Error Over Trials (Refined Behavior)')

# Add color bar for reference
fig.colorbar(surf, shrink=0.5, aspect=10, label='Prediction Error')

plt.show()
