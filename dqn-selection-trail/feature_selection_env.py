import numpy as np
import gym
from gym import spaces

class FeatureSelectionEnv(gym.Env):
    """
    RL Environment for Feature Selection using DQN.
    State: Binary vector of selected features.
    Action: Index of feature to add.
    Reward: Performance (AUC/accuracy) after selection.
    Episode ends after 6 features are selected.
    """
    def __init__(self, num_features, eval_func, max_selected=6):
        super(FeatureSelectionEnv, self).__init__()
        self.num_features = num_features
        self.max_selected = max_selected
        self.eval_func = eval_func  # Should accept a list of indices and return a scalar reward

        self.action_space = spaces.Discrete(self.num_features)
        self.observation_space = spaces.MultiBinary(self.num_features)
        self.reset()

    def reset(self):
        self.selected = []
        self.state = np.zeros(self.num_features, dtype=np.int8)
        self.done = False
        return self.state.copy()

    def step(self, action):
        if self.done or self.state[action] == 1:
            # Invalid action or episode over
            return self.state.copy(), 0.0, True, {}

        self.selected.append(action)
        self.state[action] = 1

        if len(self.selected) >= self.max_selected:
            reward = self.eval_func(self.selected)
            self.done = True
        else:
            reward = 0.0

        return self.state.copy(), reward, self.done, {}

    def render(self, mode='human'):
        print(f"Selected features: {self.selected}")

    def get_selected_features(self):
        return self.selected