from operator import truth
import os
import csv
from traceback import walk_stack

import numpy as np
import pandas as pd

from android import AndroidDerived
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, "../../../")
import gnss_lib_py.core.coordinates as coord

# from numpy.random import default_rng

path = 'Pixel4_derived.csv'
baseline_path = "baseline_locations_train.csv"

derived = AndroidDerived(path)
df = derived.pandas_df()
baseline_df = pd.read_csv(baseline_path)

#arr = df.to_numpy()
# print (df)
# print(df.keys())
# print(df["b_sat_m"])


#### TASKS

# compute correctedPrM = rawPrM + satClkBiasM - isrbM - ionoDelayM - tropoDelayM
df['corrected_pr_m'] = df['raw_pr_m'] + df['b_sat_m'] - df['intersignal_bias_m'] - df['iono_delay_m'] - df['tropo_delay_m']
print(df['corrected_pr_m'] - df['pseudo']) # Note that we can just use the pseudo field of the df instead
sample = df.sample(500)
# plt.scatter(sample['raw_pr_m'], sample['corrected_pr_m'])
# plt.show()
# print(df)
# print(baseline_df)

# compute residuals for each satellite measurement.. group them by constellation
# 
# 
# feasture extraction: start here: line 163 onwards
# https://github.com/Stanford-NavLab/deep_gnss/blob/main/src/correction_network/android_dataset.py

# entire sequence of tasks

"""
get ground truth, WLS, - DONE
sync derived times, GT and WLS (check for units) - DONE
get initial posn estimates by adding noise to WLS in each dirn ## def add_guess_noise(self, true_XYZb):
transform initial posn estimate from ecef XYZ to NED frame
perform primary feature extraction.. (lines 163 onwards)
create the concatenated feature vector (line 175)
spend most time here.. how to improve features..etc

group them by different satellite constellations (nodes in graph)

how to do this grouping?


then we will take a look at the training part..
 https://github.com/Stanford-NavLab/deep_gnss/blob/3f3b9099235dcbe972b5b52c5523d0bb8daa6da9/py_scripts/train_android.py#L100
here true loss.. is going to be the true residual.. can be GT - WLS, or GT - all the random initializations..

need to get the raw dataset working too.

"""
def get_meas_and_base_df(df, baseline_df):
	# Sync derived times for GT and WLS
	meas_and_base_df = df.merge(baseline_df.rename(columns = {"collectionName": "trace_name"}), on=["millisSinceGpsEpoch", "trace_name"], how="left")
	assert(not meas_and_base_df['latDeg'].isnull().values.any()) # If any are null, consider switching to an inner join instead
	return meas_and_base_df

def get_timesteps(meas_and_base_df):
	return meas_and_base_df['millisSinceGpsEpoch'].unique()

# Copied wholsale from https://github.com/Stanford-NavLab/deep_gnss/blob/main/src/correction_network/android_dataset.py,
# but harcoded in the random number generator and the guess range
# Also, removed the b term
def add_guess_noise(XYZb):
	# rng = default_rng()
	# guess_range taken from https://github.com/Stanford-NavLab/deep_gnss/blob/main/py_scripts/eval_android.py
	# Is this reasonable?
	# This is a hyperparameter that we can test
	guess_range = [15, 15, 15]
	guess_noise = np.array([
		np.random.uniform(-guess_range[0], guess_range[0]),
		np.random.uniform(-guess_range[1], guess_range[1]),
		np.random.uniform(-guess_range[2], guess_range[2]),
		])
	return XYZb + guess_noise

# Copied wholesale from https://github.com/Stanford-NavLab/deep_gnss/blob/main/src/correction_network/android_dataset.py
def expected_measurements(dframe, guess_XYZb):
    satX = dframe.loc[:, "x_sat_m"].to_numpy()
    satY = dframe.loc[:, "y_sat_m"].to_numpy()
    satZ = dframe.loc[:, "z_sat_m"].to_numpy()
    satvX = dframe.loc[:, "vx_sat_mps"].to_numpy()
    satvY = dframe.loc[:, "vy_sat_mps"].to_numpy()
    satvZ = dframe.loc[:, "vz_sat_mps"].to_numpy()
    expected_ranges = np.sqrt((satX-guess_XYZb[0])**2 \
                       +(satY-guess_XYZb[1])**2 \
                       +(satZ-guess_XYZb[2])**2) 
#     gt_ranges = gt_ranges.values
    # expected_rho = gt_ranges + guess_XYZb[3]
    satXYZV = pd.DataFrame()
    satXYZV['x'] = satX
    satXYZV['y'] = satY
    satXYZV['z'] = satZ
    satXYZV['vx'] = satvX
    satXYZV['vy'] = satvY
    satXYZV['vz'] = satvZ
    return expected_ranges, satXYZV

# Idea: This will become get_item function in dataset
def get_item(idx, timesteps, meas_and_base_df):
	# Questions: We use corrected pseudorange for this, correct?
	timestep = timesteps[idx]
	# print(timestep)
	data = meas_and_base_df.loc[meas_and_base_df['millisSinceGpsEpoch'] == timestep]
	# print(data)

	# Convert LLA to ECEF
	lla = data.head(1)[['latDeg', 'lngDeg', 'heightAboveWgs84EllipsoidM']].to_numpy(np.float64)[0]
	ecef = coord.geodetic2ecef(lla)
	# What was the b term that they had in here?
	noised_guess = add_guess_noise(ecef)
	# print("ECEF", ecef)
	# print("Noised guess", noised_guess)

	# Below is copied with minimal modifications
	ref_local = coord.LocalCoord.from_ecef(noised_guess)
	# Note: guess_NED should be 0, since we're treating p_init as the origin of
	# the NED frame.
	print(ref_local)
	guess_NED = ref_local.ecef2ned(noised_guess[:, None])[:, 0]   # position
	# guess_NED = ref_local.ecef2ned(noised_guess[:, None])   # position
	# print("NED", guess_NED)

	# print("Other guess", ref_local.ecef2ned(ecef[:, None]))

	# Primary feature extraction
	expected_pseudo, satXYZV = expected_measurements(data, ecef)
	residuals = (data['corrected_pr_m'] - expected_pseudo).to_numpy()
	data['residuals'] = residuals
	# What's reasonable for the residuals?
	print(residuals)

	# Grouping by the constellations
	# Is this the correct way to group?
	# for idx, dframe in data.groupby("gnss_id"):
		# print(dframe)

	# Are worrying about self.bias_pd?
	# if self.bias_pd is not None:
	#     bias_slice = self.bias_pd[self.bias_pd['tracePath']==key[0]]
	#     for idx in range(len(residuals)):
	#         svid = data["SvName"].values[idx]
	#         residuals[idx] = residuals[idx] - bias_slice.loc[bias_slice['SvName']==svid, 'bias'].to_numpy()[0]
	los_vector = (satXYZV[['x', 'y', 'z']] - noised_guess)
	los_vector = los_vector.div(np.sqrt(np.square(los_vector).sum(axis=1)), axis='rows').to_numpy()
	los_vector = ref_local.ecef2nedv(los_vector)
	
	features = np.concatenate((np.reshape(residuals, [-1, 1]), los_vector), axis=1)
	
	# sample = {
	#     'features': torch.Tensor(features),
	#     'true_correction': (true_NEDb-guess_NEDb)[:3],
	#     'guess': guess_XYZb
	# }

meas_and_base_df = get_meas_and_base_df(df, baseline_df)
timesteps = get_timesteps(meas_and_base_df)
for i in range(len(timesteps)):
	get_item(i, timesteps, meas_and_base_df)
