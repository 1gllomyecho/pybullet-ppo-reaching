# PyBullet Robotic Grasping with PPO

A Gymnasium-compatible PyBullet environment for robotic arm grasping, trained with PPO from stable-baselines3. The KUKA arm learns to grasp a randomly placed cube using reinforcement learning.

## Demo

![Grasping Demo](demo.gif)

## Environment

- Python 3.12
- PyBullet
- stable-baselines3
- gymnasium

## How to Run

1. Install dependencies:
   pip install pybullet stable-baselines3 gymnasium
2. Train the model:
   python train_ppo.py
3. Evaluate the trained model:
   python test_ppo.py

## How It Works

- State: 7 joint angles + cube position + end-effector position
- Action: 7 joint target angles
- Reward: distance reduction + success bonus
- Algorithm: PPO with MlpPolicy

## Result

The robotic arm learns to grasp a randomly placed cube after 500k training steps.