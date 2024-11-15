import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the TD model
num_trials = 100
time_steps = 250
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

# Run TD learning for one early trial and one late trial
v_early = np.zeros(time_steps)  # Prediction before learning
v_late = np.zeros(time_steps)   # Prediction after learning

# Run learning for multiple trials to simulate learning effects
for trial in range(num_trials):
    # Temporal difference learning: Update predictions
    for t in range(time_steps - 1):  # Avoid out of range error
        # Calculate prediction error (TD error)
        delta[t] = r[t] + v[t + 1] - v[t] if t < time_steps - 1 else r[t] - v[t]
        
        # Update weights (w(tau)) for each time step using the learning rule
        for tau in range(t + 1):  # Loop over all previous time steps (stimulus influence)
            w[tau] += learning_rate * delta[t] * u[t - tau] if t - tau >= 0 else 0
        
        # Update prediction using weights (for future rewards)
        # Manually accumulate the weighted sum of all past stimuli up to the current time step
        if t < reward_time:  # Only predict before the reward is delivered
            v[t + 1] = np.sum(w[:t + 1] * u[:t + 1])  # Prediction based on all previous stimuli
        else:
            v[t + 1] = 0  # No prediction after reward is delivered (t >= reward_time)

    # Store early and late values for plotting
    if trial == 0:
        v_early = v.copy()
    elif trial == num_trials - 1:
        v_late = v.copy()

# Calculate Δv for early and late trials
delta_v_early = np.diff(v_early, prepend=0)
delta_v_late = np.diff(v_late, prepend=0)
delta_early = r + delta_v_early
delta_late = r + delta_v_late

# Plotting
fig, ax = plt.subplots(5, 2, figsize=(12, 10))

# Plot "Before Training" (Early Trial)
ax[0, 0].plot(u, label="u (Stimulus)", color="black")
ax[1, 0].plot(r, label="r (Reward)", color="blue")
ax[2, 0].plot(v_early, label="v (Prediction)", color="orange")
ax[3, 0].plot(delta_v_early, label="Δv", color="purple")
ax[4, 0].plot(delta_early, label="δ (TD Error)", color="red")

# Titles and labels for before training
ax[0, 0].set_title("Before Training")
for i in range(5):
    ax[i, 0].legend(loc="upper right")
    ax[i, 0].set_xlim([0, time_steps])

# Plot "After Training" (Late Trial)
ax[0, 1].plot(u, label="u (Stimulus)", color="black")
ax[1, 1].plot(r, label="r (Reward)", color="blue")
ax[2, 1].plot(v_late, label="v (Prediction)", color="orange")
ax[3, 1].plot(delta_v_late, label="Δv", color="purple")
ax[4, 1].plot(delta_late, label="δ (TD Error)", color="red")

# Titles and labels for after training
ax[0, 1].set_title("After Training")
for i in range(5):
    ax[i, 1].legend(loc="upper right")
    ax[i, 1].set_xlim([0, time_steps])

# Display plot
fig.tight_layout()
plt.show()
