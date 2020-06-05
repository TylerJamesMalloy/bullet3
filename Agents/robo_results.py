import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from os import path

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


nchain_filenames = [    "Robots/Walker2DBulletEnv",
                        "RobotsGen/Walker2DBulletEnv",
                        "RobotsExtremeGen/Walker2DBulletEnv"
                    ]

NUM_RESAMPLES = 100

clac_tags = [   [0.01], 
                [0.01],
                [0.01]]
clac_env_strings = []
for env in clac_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    clac_env_strings.append(env_tags)

sac_tags = [    [0.05],
                [0.05],
                [0.05]]
sac_env_strings = []
for env in sac_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    sac_env_strings.append(env_tags)

All_Data = pd.DataFrame()
ROUNDING_VALUE = -4

agents = np.linspace(0, 16, 16, dtype="int")
#agents = np.delete(agents, 6)
#agents = np.delete(agents, 42)
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
                #clac_data = clac_data.iloc[-1:]
                clac_data["Resample"] = resample_num
                clac_data["Environment"] = nchain_filename
                clac_data["Timestep"] = clac_data["Timestep"].astype(float).round(ROUNDING_VALUE)
                clac_data.loc[clac_data['Model'].str.contains('CLAC'), 'Model'] = 'CLAC'

                All_Data = All_Data.append(clac_data, sort="full")

    sac_tag_strings = sac_env_strings[index]
    for tag in sac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                sac_model_file = nchain_filename +  "/results/" + sac_model_name + ".pkl"

                if(not path.exists(sac_model_file)):
                    continue

                sac_data = pd.read_pickle(sac_model_file)
                #sac_data = sac_data.iloc[-1:]
                sac_data["Resample"] = resample_num
                sac_data["Environment"] = nchain_filename
                sac_data["Timestep"] = sac_data["Timestep"].astype(float).round(ROUNDING_VALUE)
                sac_data.loc[sac_data['Model'].str.contains('SAC'), 'Model'] = 'SAC'
                

                All_Data = All_Data.append(sac_data, sort="full")

#All_Data = All_Data.loc[All_Data["Timestep"] < 50000]

print("Done Loading Results")
print(All_Data)

Walker2DBulletEnv_Data = All_Data.loc[All_Data["Environment"] == nchain_filenames[0]]
HopperBulletEnv_Data = All_Data.loc[All_Data["Environment"] == nchain_filenames[1]]
AntBulletEnv_Data = All_Data.loc[All_Data["Environment"] == nchain_filenames[2]]

fig, axes = plt.subplots(1, 3)

#InvertedPendulumBulletEnv_Data_0 = All_Data.loc[All_Data["Resample"] == 0]
#InvertedPendulumBulletEnv_Data_1 = All_Data.loc[All_Data["Resample"] == NUM_RESAMPLES - 1]

SAC_DATA_Walker = Walker2DBulletEnv_Data.loc[Walker2DBulletEnv_Data["Model"] == "SAC"]
CLAC_DATA_Walker  = Walker2DBulletEnv_Data.loc[Walker2DBulletEnv_Data["Model"] == "CLAC"]

SAC_DATA_Hopper = HopperBulletEnv_Data.loc[HopperBulletEnv_Data["Model"] == "SAC"]
CLAC_DATA_Hopper = HopperBulletEnv_Data.loc[HopperBulletEnv_Data["Model"] == "CLAC"]

SAC_DATA_Ant = AntBulletEnv_Data.loc[AntBulletEnv_Data["Model"] == "SAC"]
CLAC_DATA_Ant  = AntBulletEnv_Data.loc[AntBulletEnv_Data["Model"] == "CLAC"]

sac_pal = ["blue"]
clac_pal = ["orange"]

ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci=99, data=Walker2DBulletEnv_Data, ax=axes[0])
ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", ci=99, data=HopperBulletEnv_Data, ax=axes[1])
ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", ci=99, data=AntBulletEnv_Data, ax=axes[2])

#ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Coefficient", legend="full", ci=99, data=SAC_DATA_Walker, ax=axes[0])
#ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Coefficient", legend="full", ci=99, data=CLAC_DATA_Walker, ax=axes[1])


axes[0].set_title('Normal', fontsize=48)
axes[1].set_title('Random', fontsize=48)
axes[2].set_title('Extreme', fontsize=48)

axes[0].set_ylabel('Average Reward', fontsize=48)
axes[1].set_ylabel('', fontsize=48)
axes[2].set_ylabel('', fontsize=48)

axes[0].set_xlabel('Time Step', fontsize=48)
axes[1].set_xlabel('Time Step', fontsize=48)
#axes[2].set_xlabel('Time Step', fontsize=48)

fig.suptitle('Robot Training Results', fontsize=64)
plt.show()
