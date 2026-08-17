import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import random


class RobotGraspEnv(gym.Env):
    def __init__(self, render_mode="gui"):
        super().__init__()
        if p.isConnected():
            p.disconnect()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        if self.render_mode == "gui":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.cube = p.loadURDF("cube.urdf", [0.3, 0.0, 0.3], globalScaling=0.2)

        self.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.last_distance = None
        self.max_episode_steps = 200
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        for j in self.joint_indices:
            p.resetJointState(self.robot, j, 0.0)
        p.resetJointState(self.robot, 1, -0.3)

        cube_x = random.uniform(0.2, 0.7)
        cube_y = random.uniform(-0.4, 0.4)
        cube_z = random.uniform(0.25, 0.45)
        p.resetBasePositionAndOrientation(self.cube, [cube_x, cube_y, cube_z], [0, 0, 0, 1])


        self.last_distance = None
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        for i, joint in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot, joint, p.POSITION_CONTROL,
                targetPosition=action[i],
                force=80,
                maxVelocity=1.2
            )
        for _ in range(5):
            p.stepSimulation()

        obs = self._get_observation()
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        end_pos = p.getLinkState(self.robot, 6)[0]
        distance = np.linalg.norm(np.array(end_pos) - np.array(cube_pos))

        if self.last_distance is None:
            reward = 0.0

        else:
            reward = (self.last_distance - distance) * 30
        self.last_distance = distance
        reward -= 0.01

        if distance < 0.3:
            reward += 5
        if distance < 0.1:
            reward += 10

        terminated = False
        if distance < 0.05:
            reward += 100
            terminated = True

        self.step_count += 1
        truncated = bool(self.step_count >= self.max_episode_steps)

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        joints = [p.getJointState(self.robot, j)[0] for j in self.joint_indices]
        cube_pos = list(p.getBasePositionAndOrientation(self.cube)[0])
        end_pos = list(p.getLinkState(self.robot, 6)[0])
        return np.array(joints + cube_pos + end_pos, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        if p.isConnected():
            p.disconnect()