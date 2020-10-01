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

#     ENVIRONMENT_NAMES Walker2DBulletEnv-v0, Robots/AntBulletEnv-v0  , HopperBulletEnv-v0 , HumanoidBulletEnv-v0, HalfCheetahBulletEnv-v0

FOLDER = "Results/InvertedDoublePendulumBulletEnv" 

NUM_RESAMPLES = 100
NUM_TRAINING_STEPS = 1000
NUM_TESTING_STEPS = 1000
ENVIRONMENT_NAME = "InvertedDoublePendulumBulletEnv-v0"

if(not os.path.exists(FOLDER + '/Extreme/results')):
    os.mkdir(FOLDER + '/Extreme/results')
if(not os.path.exists(FOLDER + '/Generalization/results')):
    os.mkdir(FOLDER + '/Generalization/results')
if(not os.path.exists(FOLDER + '/Training/results')):
    os.mkdir(FOLDER + '/Training/results')
if(not os.path.exists(FOLDER + '/Training/models')):
    os.mkdir(FOLDER + '/Training/models')

CLAC_COEFS  = [2.0]  
SAC_COEFS   = [2.0]

def eval_model(model, env, model_name, coef, testing_timesteps, training_timestep, agent_step, resample_step, randomization):
    obs = env.reset()
    states = None
    reward_sum = 0
    Data = pd.DataFrame()
    all_rewards = []
    allPlayedCards = []

    if(randomization > 0):
        env.env_method("randomize", randomization) 

    for test_time in range(testing_timesteps):
        action, states = model.predict(obs, states)
        obs, rewards, dones, infos = env.step(action)
        reward_sum += rewards[0]

        if(dones[0]):
            d = {"Model": model_name, "Reward": reward_sum, "Timestep": training_timestep, "Coef": coef, "Randomization": randomization, "AgentID": agent_step, "Resample": resample_step}
            Data = Data.append(d, ignore_index=True)
            all_rewards.append(reward_sum)
            reward_sum = 0
            
            if(randomization > 0):
                env.env_method("randomize", randomization) 
    
    Avg = np.mean(all_rewards)
    return Data

def test_agent(agent_step):
    now = time.time()
    for coef_index in range(len(CLAC_COEFS)):

        mut_coef = CLAC_COEFS[coef_index]
        ent_coef = SAC_COEFS[coef_index]
        training_timestep = 0

        clac_env = gym.make(ENVIRONMENT_NAME)
        clac_env = DummyVecEnv([lambda: clac_env])
        clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=mut_coef, verbose=1)

        sac_env = gym.make(ENVIRONMENT_NAME)
        sac_env = DummyVecEnv([lambda: sac_env])

        sac_model = SAC(MlpPolicy, sac_env, ent_coef=ent_coef, verbose=1)

        mirl_env = gym.make(ENVIRONMENT_NAME)
        mirl_env = DummyVecEnv([lambda: mirl_env])

        mirl_model = CLAC(CLAC_MlpPolicy, mirl_env, mut_inf_coef=mut_coef, coef_schedule=3.3e-3, verbose=1)
        
        for resample_step in range(0, NUM_RESAMPLES):
            features = pd.DataFrame()

            if(agent_step == 1):
            print(mut_coef,  "  ",  ent_coef, "  ", NUM_TRAINING_STEPS, "  ",  ENVIRONMENT_NAME, "  ", FOLDER, " ", resample_step)

            (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
            (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)
            (mirl_model, learning_results) = mirl_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=1000)

            # Save models 
            clac_model.save(FOLDER + "/Training/models/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))
            sac_model.save(FOLDER + "/Training/models/CLAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))
            mirl_model.save(FOLDER + "/Training/models/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

            training_timestep += NUM_TRAINING_STEPS

            # Test Normal 
            eval_results = eval_model(clac_model, clac_env, "CLAC", mut_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 0)
            eval_results.to_pickle(FOLDER + "/Training/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            eval_results = eval_model(sac_model, sac_env, "SAC", ent_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 0)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Training/results/SAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            eval_results = eval_model(mirl_model, mirl_env, "MIRL", mut_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 0)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Training/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            # Test generalization 
            eval_results = eval_model(clac_model, clac_env, "CLAC", mut_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 1)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Generalization/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            eval_results = eval_model(sac_model, sac_env, "SAC", ent_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 1)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Generalization/results/SAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            eval_results = eval_model(mirl_model, mirl_env, "MIRL", mut_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 1)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Generalization/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            # Test generalization Extreme
            eval_results = eval_model(clac_model, clac_env, "CLAC", mut_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 2)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Extreme/results/CLAC_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            eval_results = eval_model(sac_model, sac_env, "SAC", ent_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 2)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Extreme/results/SAC_" + str(ent_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            eval_results = eval_model(mirl_model, mirl_env, "MIRL", mut_coef, NUM_TESTING_STEPS, training_timestep, agent_step, resample_step, 2)
            eval_results['AgentID'] = agent_step
            eval_results.to_pickle(FOLDER + "/Extreme/results/MIRL_" + str(mut_coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

            clac_env.env_method("reset_features")
            sac_env.env_method("reset_features")
            mirl_env.env_method("reset_features")
        
        del sac_model
        del sac_env

        del clac_model
        del clac_env
        
        del mirl_model
        del mirl_env

    later = time.time()
    difference = int(later - now)
    print("Tested Agent Time: ", difference)


test_agent(1)
assert(False)
def main():
    Agents = [1, 2, 3, 4, 5, 6, 7, 8] 
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

