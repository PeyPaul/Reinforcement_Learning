# import the game
import gym_super_mario_bros
# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import Frame Stacker Wrapper and Grayscaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib
from matplotlib import pyplot as plt

# import os for file path management
import os
# import PPO
from stable_baselines3 import PPO
# import base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


# create the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human" )
# simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT) #to drop possible action from 256 to 7
# grayscale the environment
env = GrayScaleObservation(env, keep_dim = True)
# wrap inside the dummy environment
env = DummyVecEnv([lambda: env])
# stack the frames
env = VecFrameStack(env, 4, channels_order = 'last')


# load model
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
model = PPO.load('./train/best_model_1000')


# start the game
state = env.reset()
# loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()