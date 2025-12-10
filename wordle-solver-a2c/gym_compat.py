import gymnasium as gym

class GymCompatWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Proxy these directly for compatibility
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        # Gymnasium returns (obs, info) -> We return just obs
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        # Gymnasium returns 5 values -> We return 4
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    # --- THE FIX IS HERE ---
    def __getattr__(self, name):
        # usage of .unwrapped cuts through all wrappers (like TimeLimit)
        # to find attributes like 'words' or 'goal_word' in the real env
        return getattr(self.env.unwrapped, name)
