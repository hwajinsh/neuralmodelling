import numpy as np
import matplotlib.pyplot as plt

# Option 1

# Parameters
num_trials = 300
time_steps = 300
stimulus_time = 100
reward_time = 200
learning_rate = 0.1

# Initialize arrays
u = np.zeros((num_trials, time_steps))  # Stimulus
r = np.zeros((num_trials, time_steps))  # Reward
v = np.zeros((num_trials, time_steps))  # Prediction
delta = np.zeros((num_trials, time_steps))  # Prediction error

# Define stimulus and reward for each trial
for trial in range(num_trials):
    u[trial, stimulus_time] = 1  # Stimulus at fixed time
    r[trial, reward_time] = 1 if trial < num_trials // 2 else 0  # Reward early in first half

# TD learning loop
for trial in range(1, num_trials):
    for t in range(time_steps - 1):
        delta[trial, t] = r[trial, t] + v[trial, t + 1] - v[trial, t]  # TD error
        v[trial, t] += learning_rate * delta[trial, t]  # Update prediction

# Plot the results for "Before" and "After" training
fig, axes = plt.subplots(5, 2, figsize=(10, 12), sharex=True)

# Helper function to plot a trial
def plot_trial(axes, trial_idx, title):
    axes[0].plot(u[trial_idx], label="u (Stimulus)", color="black")
    axes[1].plot(r[trial_idx], label="r (Reward)", color="blue")
    axes[2].plot(v[trial_idx], label="v (Prediction)", color="orange")
    axes[3].plot(np.diff(v[trial_idx], prepend=0), label="Δv", color="purple")
    axes[4].plot(delta[trial_idx], label="δ (TD Error)", color="red")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.set_xlim(0, time_steps)
    axes[0].set_title(title)

# Plot before and after training
plot_trial([axes[i, 0] for i in range(5)], 0, "Before Training")
plot_trial([axes[i, 1] for i in range(5)], num_trials - 1, "After Training")

plt.tight_layout()
plt.show()