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

FOLDER = "Results/Walker2DBulletEnv" 

# Create target Directory if don't exist
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)
if not os.path.exists(FOLDER + "/models"):
    os.mkdir(FOLDER + "/models")
if not os.path.exists(FOLDER + "/results"):
    os.mkdir(FOLDER + "/results")
if not os.path.exists(FOLDER + "/features"):
    os.mkdir(FOLDER + "/features")

NUM_RESAMPLES = 100
NUM_TRAINING_STEPS = 100
NUM_TESTING_STEPS = 100
ENVIRONMENT_NAME = "Walker2DBulletEnv-v0"

RANDOMIZATION_LEVEL = "None"
#RANDOMIZATION_LEVEL = "Test" 
#RANDOMIZATION_LEVEL = "Normal" 
#RANDOMIZATION_LEVEL = "Extreme"
CLAC_COEFS = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025]  
SAC_COEFS = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025]


def eval_models(model, env, features, model_name, coef, testing_timesteps, training_timestep, randomization):
    obs = env.reset()
    states = None
    reward_sum = 0
    Data = pd.DataFrame()
    all_rewards = []
    allPlayedCards = []

    if(RANDOMIZATION_LEVEL == "Normal"):
        env.env_method("randomize", 0)
    elif(RANDOMIZATION_LEVEL == "Extreme"):
        env.env_method("randomize", 1)
    elif(RANDOMIZATION_LEVEL == "Test"):
        env.env_method("randomize", -1)
    env_features = clac_env.env_method("get_features")[0]

    for test_time in range(testing_timesteps):
        action, states = model.predict(obs, states)
        obs, rewards, dones, infos = env.step(action)
        reward_sum += rewards

        print(dones)

        if(dones):
            d = {"Model": model_name, "Reward": reward_sum, "Timestep": training_timestep, "Coef": coef, "Randomization", randomization}
            Data = Data.append(d, ignore_index=True)
            all_rewards.append(reward_sum)
            reward_sum = 0
        
        action_masks.clear()
        for info in infos:
            action_masks.append(info.get('action_mask'))
    
    env.env_method("reset_features")
    Avg = np.mean(all_rewards)
    return Data


def test_agent(agent_step):
    now = time.time()
    for coef_index in range(len(CLAC_COEFS)):
        mut_coef = CLAC_COEFS[coef_index]
        ent_coef = SAC_COEFS[coef_index]
        training_timestep = 0

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

        mirl_model = CLAC(CLAC_MlpPolicy, mirl_env, mut_inf_coef=mut_coef, coef_schedule=3.3e-3, verbose=1)

        (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
        learning_results['AgentID'] = agent_step
        learning_results.to_pickle(FOLDER + "/Training/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
        learning_results['AgentID'] = agent_step
        learning_results.to_pickle(FOLDER +  "/Training/results/SAC_"+ str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        (mirl_model, learning_results) = mirl_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
        learning_results['AgentID'] = agent_step
        learning_results.to_pickle(FOLDER + "/Training/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        training_timestep += NUM_TRAINING_STEPS

        sac_env.env_method("set_features", env_features) 
        mirl_env.env_method("set_features", env_features) 

        #if(agent_step == 0):
        #    print(env_features)

        Power = env_features[0]
        Density = env_features[1]
        Friction = env_features[2]
        Gravity = env_features[3]

        d = {"Mut Coefficient":  mut_coef, "Ent Coefficient":  ent_coef, "Resample Step":resample_step, "Power": Power, "Density": Density, "Friction": Friction, "Gravity": Gravity}
        #d = {"Mut Coefficient":  mut_coef, "Resample Step":resample_step, "Power": Power, "Density": Density, "Friction": Friction, "Gravity": Gravity}
        features = features.append(d, ignore_index = True)

        # Train generalization 
        eval_results = eval_model(clac_model, env, "CLAC", mut_coef, NUM_TESTING_STEPS, training_timestep, 0)
        eval_results['AgentID'] = agent_step
        eval_results.to_pickle(FOLDER + "/Generalization/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        eval_results = eval_model(sac_model, env, "SAC", ent_coef, NUM_TESTING_STEPS, training_timestep, 0)
        eval_results['AgentID'] = agent_step
        eval_results.to_pickle(FOLDER + "/Generalization/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        eval_results = eval_model(mirl_model, env, "MIRL", mut_coef, NUM_TESTING_STEPS, training_timestep, 0)
        eval_results['AgentID'] = agent_step
        eval_results.to_pickle(FOLDER + "/Generalization/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0.pkl")

        clac_model.save(FOLDER +  "/Training/models/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
        sac_model.save(FOLDER + "/Training/models/SAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
        mirl_model.save(FOLDER + "/Training/models/MIRL_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_0")
        #features.to_pickle(FOLDER +  "/features/features_" + str(agent_step) + "_" + str(mut_coef)  + "_" + str(ent_coef) + ".pkl")
        
        for resample_step in range(1, NUM_RESAMPLES):
            if(agent_step == 1):
                print(mut_coef,  "  ",  ent_coef, "  ", NUM_TRAINING_STEPS, "  ",  ENVIRONMENT_NAME, "  ", FOLDER, " resample step ", resample_step)
            
            (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS, reset_num_timesteps=False,  log_interval=1000)
            learning_results.to_pickle(FOLDER + "/Training/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS, reset_num_timesteps=False,  log_interval=1000)
            learning_results.to_pickle(FOLDER +  "/Training/results/SAC_"+ str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            (mirl_model, learning_results) = mirl_model.learn(total_timesteps=NUM_TRAINING_STEPS, reset_num_timesteps=False,  log_interval=1000)
            learning_results.to_pickle(FOLDER + "/Training/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            training_timestep += NUM_TRAINING_STEPS

            clac_model.save(FOLDER +  "/Training/models/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))
            sac_model.save(FOLDER + "/Training/models/SAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))
            mirl_model.save(FOLDER + "/Training/models/MIRL_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

        #print(features)

        del sac_model
        del sac_env

        del clac_model
        del clac_env
        
        del mirl_model
        del mirl_env

    later = time.time()
    difference = int(later - now)
    print("Tested Agent Time: ", difference)

def main():
    Agents = [1, 2, 3, 4] 
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

