from stable_baselines3 import PPO
from ppo.ppo_ import RobotGraspEnv

env = RobotGraspEnv()
model = PPO.load("robot_grasp_ppo")

observation, _ = env.reset()
for _ in range(30000):
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Grasp successful!")
        break