### IMPORT DEPENDENCIES

# import the game
import gym_super_mario_bros
# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

### Setup the game

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT) #to drop possible action from 256 to 7