# PyBullet Robotic Arm End-Effector Positioning with PPO

A Gymnasium-compatible PyBullet environment for robotic arm end-effector positioning, trained with PPO from stable-baselines3. The KUKA arm learns to reach a randomly placed cube using reinforcement learning.

## Demo

![Positioning Demo](demo.gif)

## Environment

- Python 3.12
- PyBullet
- stable-baselines3
- gymnasium

## How to Run

1. Install dependencies:
   pip install pybullet stable-baselines3 gymnasium
2. Train the model:
   python ppo_train.py
3. Evaluate the trained model:
   python test_ppo.py

## How It Works

- State: 7 joint angles + cube position + end-effector position
- Action: 7 joint target angles
- Reward: distance reduction + success bonus
- Algorithm: PPO with MlpPolicy

## Result

The robotic arm learns to reach a randomly placed cube after 500k training steps.