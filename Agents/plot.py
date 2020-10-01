import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


nchain_filenames = [    "Results/AntBulletEnv/Training",
                        "Results/AntBulletEnv/Generalization", 
                        "Results/AntBulletEnv/Extreme",
                    ]

"""nchain_filenames = [    "Results/InvertedDoublePendulumBulletEnv/Training",
                        "Results/InvertedDoublePendulumBulletEnv/Generalization", 
                        "Results/InvertedDoublePendulumBulletEnv/Extreme",
                    ]"""

NUM_RESAMPLES = 100

coefs = [0.01]
#clac_tag_strings = ["2p0"]
#sac_tag_strings = ["2p0"]
#mirl_tag_strings = ["2p0"]
clac_tag_strings = ["0p01"]
sac_tag_strings = ["0p01"]
mirl_tag_strings = ["0p01"]

TrainingData = pd.DataFrame()
GenData = pd.DataFrame()
ExtremeData = pd.DataFrame()

ROUNDING_VALUE = -3

agents = [1,2,3,5,6,7,8] 
print(agents)

#clac_data = pd.read_pickle("Results/AntBulletEnv/Training/results/CLAC_0p01_1_0.pkl")
#print(clac_data)

for index, nchain_filename in enumerate(nchain_filenames):
    for tag in clac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                if(agent_id == 2 and resample_num == 21):
                    continue

                clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                clac_model_file = nchain_filename + "/results/" + clac_model_name + ".pkl"

                if(not path.exists(clac_model_file)):
                    continue

                clac_data = pd.read_pickle(clac_model_file)
                
                if(clac_data.empty):
                    print("Empty: ", clac_model_file)
                    continue

                #clac_data = clac_data.iloc[-1:]
                clac_data["Resample"] = resample_num
                clac_data["Resample"] = math.ceil(resample_num / 2.) * 2  # resample_num
                clac_data["Environment"] = nchain_filename
                clac_data["Timestep"] = clac_data["Timestep"].astype(float).round(ROUNDING_VALUE)
                clac_data.loc[clac_data['Model'].str.contains('CLAC'), 'Model'] = 'CLAC'
                if(index == 0):
                    TrainingData = TrainingData.append(clac_data, sort="full")
                elif(index == 1):
                    GenData = GenData.append(clac_data, sort="full")
                else:
                    ExtremeData = ExtremeData.append(clac_data, sort="full")

    for tag in sac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                sac_model_file = nchain_filename +  "/results/" + sac_model_name + ".pkl"

                if(not path.exists(sac_model_file)):
                    continue

                sac_data = pd.read_pickle(sac_model_file)

                if(sac_data.empty):
                    continue

                #sac_data = sac_data.iloc[-1:]
                sac_data["Resample"] = resample_num
                sac_data["Resample"] = math.ceil(resample_num / 2.) * 2 # resample_num
                sac_data["Environment"] = nchain_filename
                sac_data["Timestep"] = sac_data["Timestep"].astype(float).round(ROUNDING_VALUE)
                sac_data.loc[sac_data['Model'].str.contains('SAC'), 'Model'] = 'SAC'
                

                if(index == 0):
                    TrainingData = TrainingData.append(sac_data, sort="full")
                elif(index == 1):
                    GenData = GenData.append(sac_data, sort="full")
                else:
                    ExtremeData = ExtremeData.append(sac_data, sort="full")

    for tag in mirl_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                mirl_model_name = "MIRL" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                mirl_model_file = nchain_filename + "/results/" + mirl_model_name + ".pkl"

                if(not path.exists(mirl_model_file)):
                    continue

                mirl_data = pd.read_pickle(mirl_model_file)
                
                if(mirl_data.empty):
                    continue

                #clac_data = clac_data.iloc[-1:]
                mirl_data["Resample"] = resample_num
                mirl_data["Resample"] = math.ceil(resample_num / 2.) * 2  # resample_num
                mirl_data["Environment"] = nchain_filename
                mirl_data["Timestep"] = mirl_data["Timestep"].astype(float).round(ROUNDING_VALUE)
                mirl_data.loc[mirl_data['Model'].str.contains('CLAC'), 'Model'] = 'MIRL'

                if(index == 0):
                    TrainingData = TrainingData.append(mirl_data, sort="full")
                elif(index == 1):
                    GenData = GenData.append(mirl_data, sort="full")
                else:
                    ExtremeData = ExtremeData.append(mirl_data, sort="full")

#print("Done Loading Results")
#print("CLAC Training Max: ", TrainingData.loc[TrainingData['Model'] == "CLAC"]["Reward"].max())
#print("CLAC Training Max: ", TrainingData.loc[TrainingData['Model'] == "CLAC"]["Reward"].max())
#print("CLAC Training Max: ", TrainingData.loc[TrainingData['Model'] == "CLAC"]["Reward"].max())
#assert(False)

print(TrainingData)

fig, axes = plt.subplots(2, 1, sharey='col')

ax0 = sns.lineplot(x="Resample", y="Reward", hue="Model", legend="full", ci=99, data=TrainingData, ax=axes[0])
ax0 = sns.lineplot(x="Resample", y="Reward", hue="Model", legend="full", ci=99, data=ExtremeData, ax=axes[1]) 


axes[0].set_title('Ant Walker', fontsize=48)
axes[1].set_title('', fontsize=48)

axes[0].set_ylabel('Training \n Reward', fontsize=32)
axes[1].set_ylabel('Testing  \n Reward', fontsize=32)

axes[0].set_xlabel('', fontsize=48)
axes[1].set_xlabel('Timestep 10M', fontsize=48)

plt.subplots_adjust(wspace=0.05, hspace=0.2)
fig.suptitle('Robot Generalization Results', fontsize=64)
plt.show()
