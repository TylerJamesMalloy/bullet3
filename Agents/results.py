import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)

nchain_filenames = [    "Pendulum", # InvertedPendulumBulletEnv
                        "InvertedPendulumSwingupBulletEnv",
                        "InvertedDoublePendulumBulletEnv"
                    ]

NUM_RESAMPLES = 1
NUM_RESAMPLES += 1 # 1 more than number of resamples
# NUM_GENERALIZATION_EPISODES = 100 # Unused

clac_tags = [[0.3], [], []] 
clac_env_strings = []
for env in clac_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    clac_env_strings.append(env_tags)

sac_tags = [[0.3], [], []]
sac_env_strings = []
for env in sac_tags:
    env_tags = []
    for tag in env:
        env_tags.append(str(tag).replace(".", "p"))
    sac_env_strings.append(env_tags)

All_Data = pd.DataFrame()

agents = np.linspace(1, 16, 16, dtype="int")

for index, nchain_filename in enumerate(nchain_filenames):
    clac_tag_strings = clac_env_strings[index]
    for tag in clac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                clac_model_file = nchain_filename + "/results/" + clac_model_name + ".pkl"

                clac_data = pd.read_pickle(clac_model_file)
                clac_data["Resample"] = resample_num
                clac_data["Environment"] = nchain_filename
                clac_data["Timestep"] = clac_data["Timestep"].astype(float).round(-3)
                clac_data["Mean Reward"] = clac_data["Episode Reward"].rolling(20).mean()
                clac_data.loc[clac_data['Model'].str.contains('CLAC'), 'Model'] = 'CLAC'

                All_Data = All_Data.append(clac_data, sort=False)

    sac_tag_strings = sac_env_strings[index]
    for tag in sac_tag_strings:
        for agent_id in agents:
            for resample_num in range(NUM_RESAMPLES):

                sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
                sac_model_file = nchain_filename +  "/results/" + sac_model_name + ".pkl"
                sac_data = pd.read_pickle(sac_model_file)
                sac_data["Resample"] = resample_num
                sac_data["Environment"] = nchain_filename
                sac_data["Timestep"] = sac_data["Timestep"].astype(float).round(-3)
                sac_data["Mean Reward"] = sac_data["Episode Reward"].rolling(20).mean()
                sac_data.loc[sac_data['Model'].str.contains('SAC'), 'Model'] = 'SAC'

                All_Data = All_Data.append(sac_data, sort=False)

#All_Data = All_Data.loc[All_Data["Resample"] == 1]

InvertedPendulumBulletEnv_Data = All_Data.loc[All_Data["Environment"] == nchain_filenames[0]]
InvertedPendulumSwingupBulletEnv_Data = All_Data.loc[All_Data["Environment"] == nchain_filenames[1]]
InvertedDoublePendulumBulletEnv_Data = All_Data.loc[All_Data["Environment"] == nchain_filenames[2]]

fig, axes = plt.subplots(1, 3)

InvertedPendulumBulletEnv_Data_0 = All_Data.loc[All_Data["Resample"] == 0]
InvertedPendulumBulletEnv_Data_1 = All_Data.loc[All_Data["Resample"] == 1]

ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci="sd", data=InvertedPendulumBulletEnv_Data_0, ax=axes[0]) 
ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci="sd", data=InvertedPendulumBulletEnv_Data_1, ax=axes[1]) 


#ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci=68, data=InvertedPendulumBulletEnv_Data, ax=axes[0]) 
#ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci=68, data=InvertedPendulumSwingupBulletEnv_Data, ax=axes[1]) 
#ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", legend="full", ci=68, data=InvertedDoublePendulumBulletEnv_Data, ax=axes[2]) 

#ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", data=InvertedPendulumBulletEnv_Data, ax=axes[0]) 
#ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", data=InvertedPendulumSwingupBulletEnv_Data, ax=axes[1]) 
#ax0 = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", legend="full", data=InvertedDoublePendulumBulletEnv_Data, ax=axes[2]) 

#ax0 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Coefficient", legend="full", data=CLAC_DATA, ax=axes[0]) 
#ax1 = sns.lineplot(x="Timestep", y="Episode Reward", hue="Coefficient", legend="full", data=SAC_DATA, ax=axes[1]) 
axes[0].set_title('Inverted Pendulum', fontsize=48)
axes[1].set_title('SwingUp Pendulum', fontsize=48)
axes[2].set_title('Double Pendulum', fontsize=48)

axes[0].set_ylabel('Average Reward', fontsize=48)
axes[1].set_ylabel('', fontsize=48)
axes[2].set_ylabel('', fontsize=48)

axes[0].set_xlabel('Resample Step', fontsize=48)
axes[1].set_xlabel('Resample Step', fontsize=48)
axes[2].set_xlabel('Resample Step', fontsize=48)

fig.suptitle('Pendulum Generalization Results', fontsize=64)

plt.show()
