import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


nchain_filenames = [    "Robots/AntBulletEnv",
                        "RobotsGen/AntBulletEnv", 
                        "RobotsExtremeGen/AntBulletEnv",
                        "Robots/InvertedDoublePendulumBulletEnv-v0",
                        "RobotsGen/InvertedDoublePendulumBulletEnv-v0", 
                        "RobotsExtremeGen/InvertedDoublePendulumBulletEnv-v0"
                    ]

NUM_RESAMPLES = 100

mirl_tags = [   [2.0],
                [2.0],
                [2.0],
                [2.0],
                [2.0],
                [2.0]]
mirl_env_strings = []
for env in mirl_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    mirl_env_strings.append(env_tags)



clac_tags = [   [0.025],
                [0.025],
                [0.025],
                [2.0],
                [2.0],
                [2.0]]
clac_env_strings = []
for env in clac_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    clac_env_strings.append(env_tags)

sac_tags = [    [0.04],
                [0.04],
                [0.04],
                [2.0],
                [2.0],
                [2.0]]
sac_env_strings = []
for env in sac_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    sac_env_strings.append(env_tags)

All_Data = pd.DataFrame()
ROUNDING_VALUE = -3

agents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16] 
print(agents)

for index, nchain_filename in enumerate(nchain_filenames):
    clac_tag_strings = clac_env_strings[index]
    for tag in clac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                clac_model_file = nchain_filename + "/results/" + clac_model_name + ".pkl"

                if(not path.exists(clac_model_file)):
                    continue

                clac_data = pd.read_pickle(clac_model_file)
                
                if(clac_data.empty):
                    continue

                #clac_data = clac_data.iloc[-1:]
                clac_data["Resample"] = resample_num
                clac_data["Resample"] = math.ceil(resample_num / 2.) * 2  # resample_num
                clac_data["Environment"] = nchain_filename
                clac_data["Timestep"] = clac_data["Timestep"].astype(float).round(ROUNDING_VALUE)
                clac_data.loc[clac_data['Model'].str.contains('CLAC'), 'Model'] = 'CLAC'

                All_Data = All_Data.append(clac_data, sort="full")
    
    mirl_tag_strings = mirl_env_strings[index]
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

                All_Data = All_Data.append(mirl_data, sort="full")

    sac_tag_strings = sac_env_strings[index]
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
                

                All_Data = All_Data.append(sac_data, sort="full")

#All_Data = All_Data.loc[All_Data["Timestep"] < 50000]

print("Done Loading Results")
print(All_Data)
#print(clac_data.loc[clac_data['Model'].str.contains('MIRL'), 'Model'])

Normal_Ant = All_Data.loc[All_Data["Environment"] == nchain_filenames[0]]
Gen_Ant = All_Data.loc[All_Data["Environment"] == nchain_filenames[1]]
Ext_Ant = All_Data.loc[All_Data["Environment"] == nchain_filenames[2]]

print(Gen_Ant.loc[Gen_Ant["Model"] == "MIRL"])

Normal_Walker = All_Data.loc[All_Data["Environment"] == nchain_filenames[3]]
Gen_Walker = All_Data.loc[All_Data["Environment"] == nchain_filenames[4]]
Ext_Walker = All_Data.loc[All_Data["Environment"] == nchain_filenames[5]]

fig, axes = plt.subplots(2, 3 , sharey='row')
Ant_axes, Walker_axes = axes

#InvertedPendulumBulletEnv_Data_0 = All_Data.loc[All_Data["Resample"] == 0]
#InvertedPendulumBulletEnv_Data_1 = All_Data.loc[All_Data["Resample"] == NUM_RESAMPLES - 1]

sac_pal = ["blue"]
clac_pal = ["orange"]

ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci="sd", data=Normal_Ant, ax=Ant_axes[0])
ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", ci="sd", data=Gen_Ant, ax=Ant_axes[1]) 
ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", ci="sd", data=Ext_Ant, ax=Ant_axes[2])  

ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci="sd", data=Normal_Walker, ax=Walker_axes[0])
ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", ci="sd", data=Gen_Walker, ax=Walker_axes[1]) 
ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", ci="sd", data=Ext_Walker, ax=Walker_axes[2])  


Ant_axes[0].set_title('Normal', fontsize=48)
Ant_axes[1].set_title('Randomization', fontsize=48)
Ant_axes[2].set_title('Extreme', fontsize=48)

Ant_axes[0].set_ylabel('', fontsize=48)
Walker_axes[0].set_ylabel('', fontsize=48)

#Ant_axes[0].get_xaxis().set_ticks([])
#Ant_axes[1].get_xaxis().set_ticks([])
#Ant_axes[2].get_xaxis().set_ticks([])

Walker_axes[0].set_xlabel('Time Step', fontsize=48)
Walker_axes[1].set_xlabel('Resample Step', fontsize=48)
Walker_axes[2].set_xlabel('Resample Step', fontsize=48)

Ant_axes[0].set_xlabel('')
Ant_axes[1].set_xlabel('')
Ant_axes[2].set_xlabel('')

Ant_axes[1].set_ylabel('', fontsize=48)
Ant_axes[2].set_ylabel('', fontsize=48)

Walker_axes[1].set_ylabel('', fontsize=48)
Walker_axes[2].set_ylabel('', fontsize=48)

Ant_axes[0].set_ylabel('Walker Reward', fontsize=32)
Walker_axes[0].set_ylabel('Pendulum Reward', fontsize=32)

Walker_axes[0].get_legend().remove()
Walker_axes[1].get_legend().remove()
Walker_axes[2].get_legend().remove()

Ant_axes[1].get_legend().remove()
Ant_axes[2].get_legend().remove()


#fig.text(0.01, 0.5, 'Average Reward', va='center', rotation='vertical' , fontsize=48)

plt.subplots_adjust(wspace=0.1, hspace=0.25)

fig.suptitle('Robot Generalization Results', fontsize=64)
plt.show()
