import numpy as np      

def rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon):
    """
    Implement the Rescorla-Wagner rule for associative learning.
    
    Parameters:
    stimuli_1 (array-like): Array of stimuli values for stimulus 1 across trials.
    stimuli_2 (array-like): Array of stimuli values for stimulus 2 across trials.
    rewards (array-like): Array of actual rewards received across trials.
    alpha (float): Learning rate

    Returns:
    tuple: predictions_v, weights_1, weights_2
        - predictions_v (numpy array): Predicted rewards for each trial.
        - weights_1 (numpy array): Associative weights for stimulus 1 across trials.
        - weights_2 (numpy array): Associative weights for stimulus 2 across trials.
    """
    
    # Number of trials
    num_trials = len(rewards)
    
    # Initialize weights and predictions arrays with zeros
    weights_1 = np.zeros(num_trials)
    weights_2 = np.zeros(num_trials)
    predictions_v = np.zeros(num_trials)
    
    # Loop through each trial to calculate predictions and update weights
    for i in range(1, num_trials):
        # Calculate the predicted reward for the current trial
        predictions_v[i] = weights_1[i-1] * stimuli_1[i] + weights_2[i-1] * stimuli_2[i]

        # Compute the prediction error as the difference between actual and predicted rewards
        error = rewards[i] - predictions_v[i]
        
        # Update associative weights for each stimulus based on the error
        weights_1[i] = weights_1[i-1] + epsilon * error * stimuli_1[i]
        weights_2[i] = weights_2[i-1] + epsilon * error * stimuli_2[i]
    return predictions_v, weights_1, weights_2
