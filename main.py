### IMPORT DEPENDENCIES

# import the game
import gym_super_mario_bros
# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

### Setup the game

env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human" )
env = JoypadSpace(env, SIMPLE_MOVEMENT) #to drop possible action from 256 to 7

done = True
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, truncated, info = env.step(env.action_space.sample())
    env.render()
env.close()