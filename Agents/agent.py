import logging, os, time, multiprocessing 

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import gym
import pybullet, pybullet_envs

import numpy as np
import pandas as pd

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.clac.policies import MlpPolicy as CLAC_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC

""" 
**** GYM ENVIRONMENT TEST SUITE ****
InvertedPendulumBulletEnv-v0
InvertedPendulumSwingupBulletEnv-v0
InvertedDoublePendulumBulletEnv-v0
"""

#p.connect(p.GUI)
#p.loadURDF("quadruped/quadruped.urdf") 

def test_agent(agent_step):
    for coef in COEFS:
        clac_env = gym.make(ENVIRONMENT_NAME)
        clac_env = DummyVecEnv([lambda: clac_env])

        #model = SAC(MlpPolicy, env, verbose=1)
        clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=coef, verbose=0)

        sac_env = gym.make(ENVIRONMENT_NAME)
        sac_env = DummyVecEnv([lambda: sac_env])

        #model = SAC(MlpPolicy, env, verbose=1)
        sac_model = SAC(MlpPolicy, sac_env, ent_coef=coef, verbose=0)

        # Set both environments to the same resampled values
        clac_env.env_method("randomize") 
        torqueForce, gravity = clac_env.env_method("get_features")[0]
        sac_env.env_method("set_features", torqueForce, gravity) 

        (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

        learning_results.to_pickle(FOLDER + "/results/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + ".pkl")
        clac_model.save(FOLDER +  "/models/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step))

        (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

        learning_results.to_pickle(FOLDER +  "/results/SAC_"+ str(coef).replace(".", "p") + "_" + str(agent_step) + ".pkl")
        sac_model.save(FOLDER + "/models/SAC_" + str(coef).replace(".", "p") + "_" + str(agent_step))
        
#agents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

def mp_handler():
    p = multiprocessing.Pool(len(AGENTS))
    p.map(test_agent, AGENTS)

if __name__ == '__main__':
    
    """ GLOBAL VARIABLES """
    COEFS = [0.04]
    AGENTS = np.linspace(0,1,1, dtype=int)

    ENVIRONMENT_NAME = "InvertedPendulumBulletEnv-v0"
    NUM_TRAINING_STEPS = 1000
    FOLDER = "Pendulum"
	
    # Create target Directory if don't exist
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    if not os.path.exists(FOLDER + "/models"):
        os.mkdir(FOLDER + "/models")
    if not os.path.exists(FOLDER + "/results"):
        os.mkdir(FOLDER + "/results")

    mp_handler()


