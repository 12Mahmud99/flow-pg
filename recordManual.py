import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DDPG

model = DDPG.load("./model.zip")
env = gym.make("Reacher-v4", render_mode="rgb_array", max_episode_steps=300)  # longer episodes
env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)

obs, _ = env.reset()
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()