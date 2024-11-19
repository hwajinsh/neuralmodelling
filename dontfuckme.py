# S: 1
# R: Gaussian with integral equal to 2 (within a set timeframe) 
# -> scipy.nd.image import gaussian filter
# -> V should also converge to 2
# Stochastic R: Stochasticity around the time / presence of reward 
# -> we can play around with it and explain each stochasticity as needed
# Maze: predition error should be represented in the maze in the same way as TD learning 

class TraceConditioning:
    def __init__(self, n_trials, trial_length, cs_time, reward_time, reward_magnitude):
        # Task variables
        self.n_trials = n_trials  # Total number of trials
        self.trial_length = trial_length  # Timesteps within each trial
        self.cs_time = cs_time  # CS presentation time (tau = 100)
        self.reward_time = reward_time  # Reward delivery time (tau = 200)
        self.current_trial = 0  # Track current trial
        self.current_step = 0  # Track current timestep within trial

        # Reward variables
        self.reward_state = [1, reward_time]  # Delay state and timing
        self.reward_magnitude = reward_magnitude  # Magnitude of the reward

        # Create state dictionary for a single trial
        self._create_state_dictionary()

    def reset(self):
        """
        Reset for a new experiment or after all trials are complete.
        """
        self.current_trial = 0
        self.current_step = 0

    def _create_state_dictionary(self):
        """
        Map timesteps (tau) in each trial to state attributes.
        """
        self.state_dict = {}
        for tau in range(self.trial_length):
            if tau < self.cs_time:
                self.state_dict[tau] = [0, 0]  # Before CS
            elif tau < self.reward_time:
                self.state_dict[tau] = [1, tau - self.cs_time]  # Delay after CS
            else:
                self.state_dict[tau] = [1, self.reward_time - self.cs_time]  # At/after reward

    def _create_state_dictionary(self):
        """
        Map timesteps (tau) in each trial to state attributes.
        """
        self.state_dict = {}
        for tau in range(self.trial_length):
            if tau < self.cs_time:
                self.state_dict[tau] = [0, 0]  # Before CS
            elif tau < self.reward_time:
                self.state_dict[tau] = [1, tau - self.cs_time]  # Delay after CS
            else:
                self.state_dict[tau] = [1, self.reward_time - self.cs_time]  # At/after reward

    def get_outcome(self):
        """
        Determine next timestep and reward, considering trial boundaries.
        """
        # Transition to next timestep
        if self.current_step < self.trial_length - 1:
            self.current_step += 1
        else:
            # End of trial, reset timestep and increment trial
            self.current_step = 0
            self.current_trial += 1

        # Check for reward at the current timestep
        if self.current_trial < self.n_trials and [1, self.current_step] == self.reward_state:
            reward = self.reward_magnitude
        else:
            reward = 0

        return self.current_step, reward

    def run_trials(self):
        """
        Simulate all trials.
        """
        for trial in range(self.n_trials):
            print(f"Trial {trial + 1}/{self.n_trials}")
            for tau in range(self.trial_length):
                state, reward = self.get_outcome()
                print(f"Timestep {tau}: State={self.state_dict[state]}, Reward={reward}")
            print("---")
        self.reset()  # Reset after running all trials

# Parameters for trace conditioning
n_trials = 100
trial_length = 300
cs_time = 100
reward_time = 200
reward_magnitude = # R: Gaussian with integral equal to 2 (within a set timeframe) -> scipy.nd.image import gaussian filter

# Create the environment
trace_env = TraceConditioning(n_trials, trial_length, cs_time, reward_time, reward_magnitude)

# Run the simulation
trace_env.run_trials()