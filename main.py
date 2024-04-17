### IMPORT DEPENDENCIES

# import the game
import gym_super_mario_bros
# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#done = True
#for step in range(100000):
#    if done:
#        env.reset()
#    state, reward, done, truncated, info = env.step(env.action_space.sample())
#    env.render()
#env.close()

### Preprocessing environement

# import Frame Stacker Wrapper and Grayscaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib
from matplotlib import pyplot as plt

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

### Train the RL model

# import os for file path management
import os
# import PPO
from stable_baselines3 import PPO
# import base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)
            
    def _on_step(self):
        if self.n_calls & self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True
    
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# setup model saving callback
callback = TrainAndLoggingCallback(check_freq = 100, save_path = CHECKPOINT_DIR) #by increasing check_freq we reduce the amount of memoryused by the model

# AI model started
model = PPO('CnnPolicy', env, verbose = 1, tensorboard_log = LOG_DIR, learning_rate = 0.000001, n_steps = 512)

# train the model
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
model.learn(total_timesteps = 1000, callback = callback)

### Test it out

