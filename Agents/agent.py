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

HopperBulletEnv-v0
HalfCheetahBulletEnv-v0
WalkerBaseBulletEnv-v0
"""

#p.connect(p.GUI)
#p.loadURDF("quadruped/quadruped.urdf") 

def test_agent(agent_step):
    for coef in COEFS:
        features = pd.DataFrame()

        clac_env = gym.make(ENVIRONMENT_NAME)
        clac_env = DummyVecEnv([lambda: clac_env])
        clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=coef, verbose=0)

        sac_env = gym.make(ENVIRONMENT_NAME)
        sac_env = DummyVecEnv([lambda: sac_env])

        sac_model = SAC(MlpPolicy, sac_env, ent_coef=coef, verbose=0)

        (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

        learning_results.to_pickle(FOLDER + "/results/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")
        clac_model.save(FOLDER +  "/models/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_0")

        (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

        learning_results.to_pickle(FOLDER +  "/results/SAC_"+ str(coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")
        sac_model.save(FOLDER + "/models/SAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_0")

        for resample_step in range(1, NUM_RESAMPLES + 1):
            # Set both environments to the same resampled values
            clac_env.env_method("randomize")
            env_features = clac_env.env_method("get_features")[0]
            sac_env.env_method("set_features", env_features) 

            torqueForce = env_features[0]
            gravity = env_features[1]
            pole1Mass = env_features[2]

            if(len(env_features) > 3): # Used in the double pendulum
                pole2Mass = env_features[3]
                d = {"Coefficient": coef, "Resample Step":resample_step, "TorqueForce": torqueForce, "gravity": gravity, "pole1Mass": pole1Mass, "pole2Mass": pole2Mass}
            else:
                d = {"Coefficient": coef, "Resample Step":resample_step, "TorqueForce": torqueForce, "gravity": gravity, "pole1Mass": pole1Mass}
            
            features = features.append(d, ignore_index = True)

            (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

            learning_results.to_pickle(FOLDER + "/results/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
            clac_model.save(FOLDER +  "/models/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

            (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

            learning_results.to_pickle(FOLDER +  "/results/SAC_"+ str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
            sac_model.save(FOLDER + "/models/SAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

        features.to_pickle(FOLDER +  "/features/features_" + str(agent_step) + "_" + str(coef) + ".pkl")

        del sac_model
        del clac_model
        del clac_env
        del sac_env

def mp_handler():
    p = multiprocessing.Pool(len(AGENTS))
    p.map(test_agent, AGENTS)

if __name__ == '__main__':
    
    """ GLOBAL VARIABLES """
    COEFS = [0.5, 1.0] 
    AGENTS = np.linspace(1,16,16, dtype=int)
    NUM_RESAMPLES = 9

    ENVIRONMENT_NAME = "InvertedPendulumSwingupBulletEnv-v0"
    NUM_TRAINING_STEPS = 5000
    FOLDER = "InvertedPendulumSwingupBulletEnv"
	
    # Create target Directory if don't exist
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    if not os.path.exists(FOLDER + "/models"):
        os.mkdir(FOLDER + "/models")
    if not os.path.exists(FOLDER + "/results"):
        os.mkdir(FOLDER + "/results")
    if not os.path.exists(FOLDER + "/features"):
        os.mkdir(FOLDER + "/features")

    mp_handler()


