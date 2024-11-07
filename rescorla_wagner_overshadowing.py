import numpy as np

def rescorla_wagner_overshadowing(stimuli_1, stimuli_2, rewards, epsilon1, epsilon2):
    """
    Implement the Rescorla-Wagner rule for associative learning.
    
    Parameters:
    stimuli_1 (array-like): Array of stimuli values for stimulus 1 across trials.
    stimuli_2 (array-like): Array of stimuli values for stimulus 2 across trials.
    rewards (array-like): Array of actual rewards received across trials.
    epsilon (float): Learning rate.

    Returns:
    tuple: predictions_v, weights_1, weights_2
        - predictions_v (numpy array): Predicted rewards for each trial.
        - weights_1 (numpy array): Associative weights for stimulus 1 across trials.
        - weights_2 (numpy array): Associative weights for stimulus 2 across trials.
    """
    
    # Number of trials
    num_trials = len(rewards)
    
    # Initialize weights and predictions arrays
    weights_1 = np.zeros(num_trials + 1)  # +1 to handle initial 0 weight properly
    weights_2 = np.zeros(num_trials + 1)
    predictions_v = np.zeros(num_trials)
    
    # Loop through each trial
    for i in range(num_trials):
        # Calculate the predicted reward for the current trial
        predictions_v[i] = weights_1[i] * stimuli_1[i] + weights_2[i] * stimuli_2[i]
        
        # Compute the prediction error
        error = rewards[i] - predictions_v[i]
        
        # Update associative weights for the next trial
        weights_1[i+1] = weights_1[i] + epsilon1 * error * stimuli_1[i]
        weights_2[i+1] = weights_2[i] + epsilon2 * error * stimuli_2[i]
    
    # Return weights excluding the extra initial value
    return predictions_v, weights_1[:-1], weights_2[:-1]
