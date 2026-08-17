from stable_baselines3 import PPO
from ppo.ppo_ import RobotGraspEnv

env = RobotGraspEnv()
model = PPO("MlpPolicy", env, verbose=1, device="cpu", learning_rate=0.001)
model.learn(total_timesteps=500000)
model.save("robot_grasp_ppo")
print("Training complete. Model saved.")






