import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Define parameters to implement the TD learning function

trials = 1000
num_trials = np.arange(trials)
steps = 300
time_steps = np.arange(steps)
reward_time = 200
stimulus_time = 100
reward_probability = 0.2  # Probability of reward presence at each time step
sigma = 5  # Standard deviation for Gaussian filter
alpha = 0.2  # Learning rate

# Initialise stimulus array and set it to 1 at time step 100
u = np.zeros((len(time_steps), 1)) 
u[stimulus_time] = 1

# Randomly generate rewards (2D array: trials x steps)
np.random.seed(42)  # For reproducibility
reward_matrix = np.random.rand(trials, steps) < reward_probability  # Bernoulli distribution
reward_matrix = reward_matrix.astype(float)  # Convert boolean to float

# Apply Gaussian smoothing and normalization to each trial
for n in range(trials):
    reward_matrix[n, :] = gaussian_filter1d(reward_matrix[n, :], sigma)
    reward_matrix[n, :] *= 2 / np.sum(reward_matrix[n, :])  # Normalize to sum to 2


def td_leaning_stochastic(alpha, stimuli, rewards, trials, steps):
    """
    Temporal Difference learning 

    Args:
        alpha (float): Learning rate (0 <= alpha <= 1).
        stimuli (array): Input stimuli signal
        rewards (2D array): Reward signals (trials x time steps)
        trials (int): Number of trials
        steps (int): Number of time points
    
    Returns:
        w (array): Weights (associations between stimuli and predictions) after learning.
        v (array): Prediction of reward
        delta_v (array): Change in predictions between consecutive time steps.
        delta (2D array): Temporal Difference (TD) errors across trials and time steps.
    """
    # Initialize arrays
    w = np.zeros(steps)  # Weights
    v = np.zeros(steps)  # Prediction
    delta = np.zeros((trials, steps))  # TD errors (trials x time steps)
    delta_v = np.zeros(steps)  # Change in predictions

    # TD learning: update predictions for each trial
    for n in range(trials):  # Iterate over trials
        for t in range(1, steps):  # Iterate over time steps
            # Compute prediction v(t) as the weighted sum of past stimuli
            v[t] = (w[0:t] @ stimuli[t:0:-1])[0]

            # Compute prediction difference
            delta_v[t - 1] = v[t] - v[t - 1]

            # Compute TD error for this trial and time step
            delta[n, t] = rewards[n, t] + delta_v[t]

            # Update weights
            for tau in range(t):
                w[tau] += alpha * delta[n, t] * stimuli[t - tau]

    return w, v, delta_v, delta


# Run TD learning with stochastic rewards
w, v, delta_v, delta_after = td_leaning_stochastic(learning_rate, u, rewards, trials, steps)

# Plot as before
fig = plt.figure(figsize=(12, 8))
x, y = np.meshgrid(np.arange(delta_after.shape[1]), np.arange(delta_after.shape[0]))

# Plot the surface
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, delta_after, cmap='viridis', edgecolor='none', alpha=0.9)

# Format plot
ax.set_xlabel('Time Steps (t)', fontsize=12)
ax.set_ylabel('Trials', fontsize=12)
ax.set_zlabel('Weight Change (δ)', fontsize=12)
plt.title('TD Error with Stochastic Rewards')
plt.show()


# Run TD learning with randomized rewards
w, v, delta_v, delta = td_learning_stochastic(alpha, stimuli, reward_matrix, trials, steps)

# Visualize the results
# Plot average Temporal Difference (TD) error over trials
avg_delta = np.mean(delta, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_delta, label='Average TD Error', color='blue')
plt.title('Average TD Error Over Time (Randomized Rewards)')
plt.xlabel('Time Steps')
plt.ylabel('TD Error (δ)')
plt.legend()
plt.grid()
plt.show()

# 3D visualization of TD errors
fig = plt.figure(figsize=(10, 8))
x, y = np.meshgrid(np.arange(steps), np.arange(trials))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, delta, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_title('TD Errors Across Trials and Time Steps')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Trials')
ax.set_zlabel('TD Error (δ)')
fig.colorbar(surf, shrink=0.5, aspect=10, label='TD Error (δ)')
plt.show()
