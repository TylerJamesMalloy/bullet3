import logging, os, time, multiprocessing, sys, signal

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import gym
import pybullet, pybullet_envs, pybullet_data

import numpy as np
import pandas as pd

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.clac.policies import MlpPolicy as CLAC_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC
 
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#ENVIRONMENT_NAMES = [Walker2DBulletEnv-v0, Robots/AntBulletEnv-v0  , "HopperBulletEnv-v0" , "HumanoidBulletEnv-v0", "HalfCheetahBulletEnv-v0"]
#FOLDERS = [  "Robots/AntBulletEnv" , "Robots/HopperBulletEnv" , "Robots/HumanoidBulletEnv", "Robots/HumanoidFlagrunBulletEnv"]     

#physicsClient = pybullet.connect(pybullet.GUI) #or p.DIRECT for non-graphical version
#pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

# Robots
# RobotsGen
# RobotsExtremeGen

FOLDER = "Robots/AntBulletEnv" 

# Create target Directory if don't exist
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)
if not os.path.exists(FOLDER + "/models"):
    os.mkdir(FOLDER + "/models")
if not os.path.exists(FOLDER + "/results"):
    os.mkdir(FOLDER + "/results")
if not os.path.exists(FOLDER + "/features"):
    os.mkdir(FOLDER + "/features")

NUM_RESAMPLES = 0
NUM_TRAINING_STEPS = 1000000
ENVIRONMENT_NAME = "AntBulletEnv-v0"

RANDOMIZATION_LEVEL = "None"
#RANDOMIZATION_LEVEL = "Test" 
#RANDOMIZATION_LEVEL = "Normal" 
#RANDOMIZATION_LEVEL = "Extreme"
CLAC_COEFS = [2.0]  

agent_step = 9

for coef_index in range(len(CLAC_COEFS)):
    mut_coef = CLAC_COEFS[coef_index]

    if(agent_step == 1):
        print(mut_coef,  "  ", NUM_TRAINING_STEPS, "  ",  ENVIRONMENT_NAME, "  ", FOLDER)
    
    features = pd.DataFrame()
    
    mirl_env = gym.make(ENVIRONMENT_NAME)
    mirl_env = DummyVecEnv([lambda: mirl_env])

    print("here 3")

    mirl_model = CLAC(CLAC_MlpPolicy, mirl_env, mut_inf_coef=mut_coef, coef_schedule=0.99, verbose=1)

    (mirl_model, learning_results) = mirl_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=10)
    learning_results['AgentID'] = agent_step
    learning_results.to_pickle(FOLDER + "/results/MIRL_TEST_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

    print("here 3")

    print("here 5")
    mirl_model.save(FOLDER + "/models/MIRL_TEST_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
    
    del mirl_model
    del mirl_env



