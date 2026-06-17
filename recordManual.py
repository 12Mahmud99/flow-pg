import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DDPG

model = DDPG.load("./model.zip")
env = gym.make("Reacher-v2")
env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)  # record every episode

obs, _ = env.reset()
for _ in range(200):  # this will run ~4 episodes (200 steps)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
print("Video(s) saved in videos/ folder")