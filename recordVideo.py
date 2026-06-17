import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DDPG

model = DDPG.load("outputs/train_rl/2026-06-16/23-43-38/model.zip")
env = gym.make("Reacher-v2", render_mode="rgb_array")
env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
print("Video saved in videos/")