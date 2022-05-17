# from sre_parse import State
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

NUM_ADR_STATES = 5 # Does not include UNKNOWN
ADR_STATES = ["VALID", "RESET", "CYCLE_SLIP", "HALF_CYCLE_RESOLVED", "HALF_CYCLE_REPORTED"]

def gnss_log_to_dataframes(path):
    print('Loading ' + path, flush=True)
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            gnss_map[dataline[0]] = dataline[1:]
        elif not is_header:
            if dataline[0] not in datas:
                print("Unable to parse line", dataline)
            else:
                datas[dataline[0]].append(dataline[1:])

    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            results[k][col] = pd.to_numeric(results[k][col])

    return results

def extract_state(df, state_idx):
    '''
    df is the dataframe from which to extract the state
    j is the index of the state
    '''
    return np.bitwise_and(df['AccumulatedDeltaRangeState'], 1 << state_idx)

def plot_adr_state(df):
    constellation_types = df["ConstellationType"].unique()
    # print(constellation_types) # Should be the list of all satellites

    fig, axs = plt.subplots(len(constellation_types))
    fig.suptitle("ADR states by constellation type")

    for i in range(len(constellation_types)):
        measurements = df[df["ConstellationType"] == constellation_types[i]]
        # print(measurements["ConstellationType"].unique()) # Should be a single satellite
        counts = []
        for j in range(NUM_ADR_STATES):
            extracted_state = extract_state(measurements, j)
            counts.append(np.count_nonzero(extracted_state))
        ind = [j for j in range(NUM_ADR_STATES)]
        axs[i].bar(ind, counts)
        axs[i].set_ylabel("Count")
        axs[i].set_xlabel("State")
        # axs[i].set_xticks(ADR_STATES)

def plot_adr_state_vs_time(df):
    svids_constellations = list(set(zip(k["ConstellationType"], k["Svid"])))
    svids_constellations.sort()

    ncols = int(np.sqrt(len(svids_constellations)) + 1)
    fig, axs = plt.subplots(nrows = ncols * 2, ncols=ncols)
    for ax in axs.reshape(-1): # gets rid of ticks
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle("ADR States Over Time by Constellation and Satellite (500 datapoints sampled for each)")

    all_uncertainties = []
    all_states = [[] for i in range(NUM_ADR_STATES)]

    for i, (constellation, svid) in enumerate(svids_constellations):
        measurements = df[(df[["ConstellationType", "Svid"]] == (constellation, svid)).all(1)]
        # print(measurements["ConstellationType"].unique(), measurements["Svid"].unique()) # Should be a single constellation, a single satellite
        sampled = measurements.sample(min(len(measurements), 500))
        # print(sampled["ConstellationType"].unique(), sampled["Svid"].unique()) # Should be a single constellation, a single satellite
        y = i % ncols
        x = (i // ncols) * 2
        axs[x, y].title.set_fontsize(8)
        axs[x, y].title.set_text("Const " + str(constellation) + ", Svid " + str(svid))
        for j in range(NUM_ADR_STATES):
            sampled[str(j)] = extract_state(sampled, j)
            all_states[j].extend(sampled[str(j)])
            nonzero_j = sampled[sampled[str(j)] == 1 << j]
            axs[x, y].scatter(nonzero_j['utcTimeMillis'], len(nonzero_j) * [j])
        axs[x + 1, y].scatter(sampled['utcTimeMillis'], sampled['AccumulatedDeltaRangeUncertaintyMeters'])
        all_uncertainties.extend(sampled['AccumulatedDeltaRangeUncertaintyMeters'])

    print("Correlation between AccumulatedDeltaRangeUncertaintyMeters and Various States")
    for j in range(NUM_ADR_STATES):
        print("Corr for", ADR_STATES[j], ":", stats.pearsonr(all_uncertainties, all_states[j]))

dfs = gnss_log_to_dataframes('gnss_log.txt')

# extract the dataframe
k = dfs['Raw']

## TASKS 
""""
For code phase, look at fields: MultipathIndicator, PseudorangeRateUncertaintyMetersPerSecond 
"""
plot_adr_state(k)
plot_adr_state_vs_time(k)
plt.show()


