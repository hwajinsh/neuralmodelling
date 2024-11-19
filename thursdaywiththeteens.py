n_steps = 300
n_trials = 100

def td_learning(n_steps, n_trials, gamma=0.98, alpha=0.001):
  """ Temporal Difference learning

  Args:
    env (object): the environment to be learned
    n_trials (int): the number of trials to run
    alpha (float): learning rate

  Returns:
    ndarray, ndarray: the value function and temporal difference error arrays
  """
  V = np.zeros(n_steps) # Array to store values over states (time)
  TDE = np.zeros((n_steps, n_trials)) # Array to store TD errors

  for n in range(n_trials):

    state = 0 # Initial state
    for t in range(n_steps):

      # Get next state and next reward
      next_state, reward = get_outcome(state)

      # Is the current state in the delay period (after CS)?
      is_delay = state_dict[state][0]

      # Write an expression to compute the TD-error
      TDE[state, n] = (reward + V[next_state] - V[state])

      # Write an expression to update the value function
      V[state] += alpha * TDE[state, n] * is_delay

      # Update state
      state = next_state

  return V, TDE
