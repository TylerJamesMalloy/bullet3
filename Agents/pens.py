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

# sac cheetah 0.1
# sac hopper  0.15
# clac cheetah 0.01
# clac hopper 0.01

FOLDER = "OldResults/Robots/HalfCheetahBulletEnv" 

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
NUM_TRAINING_STEPS = 5000000
ENVIRONMENT_NAME = "HalfCheetahBulletEnv-v0"

RANDOMIZATION_LEVEL = "None"
#RANDOMIZATION_LEVEL = "Normal" # Same as none
#RANDOMIZATION_LEVEL = "Random"
#RANDOMIZATION_LEVEL = "Extreme"

CLAC_COEFS = [0.008, 0.01, 0.012]  
SAC_COEFS = [0.08, 0.1, 0.12]

def test_agent(agent_step):
    for coef_index in range(len(CLAC_COEFS)):
        mut_coef = CLAC_COEFS[coef_index]
        ent_coef = SAC_COEFS[coef_index]

        if(agent_step == 1):
            print(mut_coef,  "  ",  ent_coef, "  ", NUM_TRAINING_STEPS, "  ",  ENVIRONMENT_NAME, "  ", FOLDER)
        
        features = pd.DataFrame()
        
        clac_env = gym.make(ENVIRONMENT_NAME)
        clac_env = DummyVecEnv([lambda: clac_env])
        clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=mut_coef, verbose=1)

        sac_env = gym.make(ENVIRONMENT_NAME)
        sac_env = DummyVecEnv([lambda: sac_env])

        sac_model = SAC(MlpPolicy, sac_env, ent_coef=ent_coef, verbose=1)

        mirl_env = gym.make(ENVIRONMENT_NAME)
        mirl_env = DummyVecEnv([lambda: mirl_env])

        mirl_model = CLAC(CLAC_MlpPolicy, mirl_env, mut_inf_coef=mut_coef, coef_schedule=3.3e-4, verbose=1)

        (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
        learning_results['AgentID'] = agent_step
        learning_results.to_pickle(FOLDER + "/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
        learning_results['AgentID'] = agent_step
        learning_results.to_pickle(FOLDER +  "/results/SAC_"+ str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        (mirl_model, learning_results) = mirl_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
        learning_results['AgentID'] = agent_step
        learning_results.to_pickle(FOLDER + "/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")
        
        for resample_step in range(1, NUM_RESAMPLES):
            # Set both environments to the same resampled values
            if(RANDOMIZATION_LEVEL == "Normal"):
                clac_env.env_method("randomize", 0)
            elif(RANDOMIZATION_LEVEL == "Random"):
                clac_env.env_method("randomize", 1)
            elif(RANDOMIZATION_LEVEL == "Extreme"):
                clac_env.env_method("randomize", 2)
            elif(RANDOMIZATION_LEVEL == "Test"):
                clac_env.env_method("randomize", -1)
            else:
                print("Error resampling unknown value: ", RANDOMIZATION_LEVEL)
                continue

            env_features = clac_env.env_method("get_features")[0]
            sac_env.env_method("set_features", env_features) 
            mirl_env.env_method("set_features", env_features) 

            if(agent_step == 1):
                print(env_features)

            Power = env_features[0]
            Density = env_features[1]
            Friction = env_features[2]
            Gravity = env_features[3]

            d = {"Mut Coefficient":  mut_coef, "Ent Coefficient":  ent_coef, "Resample Step":resample_step, "Power": Power, "Density": Density, "Friction": Friction, "Gravity": Gravity}
            features = features.append(d, ignore_index = True)

            (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS, reset_num_timesteps=False,  log_interval=1000)
            learning_results.to_pickle(FOLDER + "/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS, reset_num_timesteps=False,  log_interval=1000)
            learning_results.to_pickle(FOLDER +  "/results/SAC_"+ str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            (mirl_model, learning_results) = mirl_model.learn(total_timesteps=NUM_TRAINING_STEPS, reset_num_timesteps=False,  log_interval=1000)
            learning_results.to_pickle(FOLDER + "/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

        
        clac_model.save(FOLDER +  "/models/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
        sac_model.save(FOLDER + "/models/SAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
        mirl_model.save(FOLDER + "/models/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
        features.to_pickle(FOLDER +  "/features/features_" + str(agent_step) + "_" + str(mut_coef)  + "_" + str(ent_coef) + ".pkl")

        #print(features)

        del sac_model
        del sac_env

        del clac_model
        del clac_env
        
        del mirl_model
        del mirl_env

def main():
    Agents = [1,2,3,4,5,6,7,8,9,10]
    print("Initializng workers: ", Agents)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(processes=len(Agents))
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        print("Starting jobs")
        res = pool.map_async(test_agent, Agents)
        print("Waiting for results")
        #res.get(1000000) # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        print("Caught Keyboard Interrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        print("Normal termination")
        pool.close()
    pool.join()

if __name__ == "__main__":
    main()

