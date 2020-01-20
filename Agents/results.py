import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)

nchain_filename = "Pendulum"

NUM_AGENTS = 32
NUM_RESAMPLES = 5
# NUM_GENERALIZATION_EPISODES = 100 # Unused

tags = [0.04] # 0.08
tag_strings = []
for tag in tags:
    tag_strings.append(str(tag).replace(".", "p"))

All_Data = pd.DataFrame()

agents = np.linspace(1, 32, 32, dtype="int")

for tag in tag_strings:
    for agent_id in agents:
        for resample_num in range(NUM_RESAMPLES):

            clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            clac_model_file = nchain_filename + "/results/" + clac_model_name + ".pkl"

            sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            sac_model_file = nchain_filename +  "/results/" + sac_model_name + ".pkl"
            
            #print(pd.read_pickle(sac_model_file))

            clac_data = pd.read_pickle(clac_model_file)
            sac_data = pd.read_pickle(sac_model_file)

            #clac_data["Coefficient"] = clac_data["mut_inf_coef"]
            #sac_data["Coefficient"] = sac_data["ent_coef"]

            #clac_data = clac_data.iloc[-10:]
            #sac_data = sac_data.iloc[-10:]

            clac_data.loc[clac_data['Model'].str.contains('CLAC'), 'Model'] = 'CLAC'
            sac_data.loc[sac_data['Model'].str.contains('SAC'), 'Model'] = 'SAC'

            All_Data = All_Data.append(clac_data, sort=False)
            All_Data = All_Data.append(sac_data, sort=False)

print(All_Data)

ax = sns.lineplot(x="Resample", y="Episode Reward", hue="Model", data=All_Data, ci="sd")  
plt.title('40K Time Step NChain Learning Results')
plt.show()

assert(False)

# df.loc[df['column_name'] == some_value]
print(All_Data["Resample"].unique())
Zero_Resample = All_Data.loc[All_Data["Resample"] == "0"]
First_Resample = All_Data.loc[All_Data["Resample"] == "1"]
Last_Resample = All_Data.loc[All_Data["Resample"] == "9"]

Zero_Resample_CLAC = All_Data.loc[All_Data["Model"] == 'CLAC']
Zero_Resample_CLAC = All_Data.loc[All_Data["Model"] == 'CLAC']

Zero_Resample_SAC = All_Data.loc[All_Data["Model"] == 'SAC']
Zero_Resample_SAC = All_Data.loc[All_Data["Model"] == 'SAC']

print(Zero_Resample)

g=sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", data=Zero_Resample, ci="sd", alpha=0.1) 
g.set(ylim=(-60, 0))

plt.title('25K Time Step NChain Learning Results')
plt.show()

g=sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", data=First_Resample, ci="sd", alpha=0.1) 
g.set(ylim=(-60, 0))

plt.title('25K Time Step NChain First Resample Results')
plt.show()

g=sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", data=Last_Resample, ci="sd", alpha=0.1) 
g.set(ylim=(-60, 0))
plt.title('25K Time Step NChain Last Resample Results')
plt.show()

All_Data = pd.DataFrame()


"""
#All_Data = pd.read_pickle(nchain_filename + "/Generalization_Results_10K_5N.pkl")
All_Data = pd.read_pickle(nchain_filename + "/Randomization_Results_10K_5N.pkl")

print(All_Data)

#g = sns.FacetGrid(All_Data, col="Model", sharex=False)
#g.map(sns.boxplot, 'Coefficient', 'Reward')

ax = sns.boxplot(x="Coefficient", y="Reward", hue="Model", data=All_Data)  # RUN PLOT   
#ax.set_ylim(-150,-100)
plt.title('250K Time Step NChain Randomization Results')
plt.show()
"""
