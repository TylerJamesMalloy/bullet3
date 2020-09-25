import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


nchain_filenames = [    "Results/Walker2DBulletEnv/Training",
                        "Results/Walker2DBulletEnv/Generalization", 
                        "Results/Walker2DBulletEnv/Extreme",
                    ]

NUM_RESAMPLES = 100

coefs = [0.01]
clac_tag_strings = ["0p01"]
sac_tag_strings = ["0p01"]
mirl_tag_strings = ["0p01"]

TrainingData = pd.DataFrame()
GenData = pd.DataFrame()
ExtremeData = pd.DataFrame()

ROUNDING_VALUE = -3

agents = [1,2,3,4] 
print(agents)

for index, nchain_filename in enumerate(nchain_filenames):
    for tag in clac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

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

print("Done Loading Results")
print(ExtremeData)


fig, axes = plt.subplots(1, 3 , sharey='row')

ax0 = sns.lineplot(x="Resample", y="Reward", hue="Model", legend="full", ci=99, data=TrainingData, ax=axes[0])
ax0 = sns.lineplot(x="Resample", y="Reward", hue="Model", legend="full", ci=99, data=GenData, ax=axes[1]) 
ax0 = sns.lineplot(x="Resample", y="Reward", hue="Model", legend="full", ci=99, data=ExtremeData, ax=axes[2])  


axes[0].set_title('Training Results', fontsize=48)
axes[1].set_title('Randomized Testing', fontsize=48)
axes[2].set_title('Extreme Testing', fontsize=48)

axes[0].set_xlabel('Resample Step', fontsize=48)
axes[1].set_xlabel('Resample Step', fontsize=48)
axes[2].set_xlabel('Resample Step', fontsize=48)

plt.subplots_adjust(wspace=0.05, hspace=0.2)
fig.suptitle('Robot Generalization Results', fontsize=64)
plt.show()
