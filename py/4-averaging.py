# %% [markdown]
# # Averaging
#
# The EEG is a very noisy signal.
# Therefore, if we are interest in the brain's response to a certain event (e.g., seeing a face versus a car), we need to average the EEG response over many trials (repetitions) of that event (i.e., over many different face trials and many different car trials).
# With this **averaging** procedure, unsystematic noise will get canceled out, while the systematic brain response will remain.
#
# Averaging is performed separately for each EEG channel and timepoint, reducing the dimensionality of the data from trials × channels × timepoints (epochs) to channels × timepoints (average).
#
# :::{figure-md}
# <img src="https://files.mtstatic.com/site_7339/94057/0?Expires=1700853581&Signature=UeBaEMEJOBCxd9GX-iXbaSsW6XPM4iX2uABmaWUjOB1~qIaRQf-m4158~2wbQeVqmaaLPnX6o04fkunrzmP88Gr-zGgPhrMBes0MEkLXLy7B43XgdPwUPpO3tUVhOjvHrAuQthoD9lv9b2IYJNsFZ1hcLWeW4ZsezFs7~iSTMFM_&Key-Pair-Id=APKAJ5Y6AV4GI7A555NA" alt="Continuous, epoched, and averaged EEG" width=500>
#
# Continuous, epoched, and averaged EEG.
# Source: {cite:t}`luck2022a`
# :::
#
# In "traditional" ERP analysis, these averages (separately for each participant and condition) are also used for statistical testing (using $t$-tests or ANOVAs), but modern approaches (e.g., linear mixed models) are typically performed on the single trials data (before averaging).
# Averaging remains useful for visualization purposes.
#
# ```{admonition} Goals
# :class: note
#
# * Creating an averaged ERP ("evoked")
# * Plotting evokeds as time courses and topographies
# ```
#
# %% [markdown]
# ## Load Python packages
#
# As always, all the functions we'll need are provided by the [MNE-Python](https://mne.tools/stable/index.html) package {cite:p}`gramfort2013`.
#
# %%
# # %pip install mne hu-neuro-pipeline

# %%
import pandas as pd
from mne import Epochs, combine_evoked, merge_events, set_bipolar_reference
from mne.io import read_raw
from mne.preprocessing import ICA
from mne.viz import plot_compare_evokeds
from pipeline.datasets import get_erpcore

# %% [markdown]
# ## Recreate epochs
#
# We repeat the preprocessing and epoching steps from the previous chapters in a condensed form (without any intermediate plots).
#
# %% tags=["hide-input", "hide-output"]
# Download data
files_dict = get_erpcore('N170', participants='sub-004', path='data')
raw_file = files_dict['raw_files'][0]
log_file = files_dict['log_files'][0]

# Preprocessing
raw = read_raw(raw_file, preload=True)
raw = set_bipolar_reference(raw, anode='FP1', cathode='VEOG_lower',
                            ch_name='VEOG', drop_refs=False)
raw = set_bipolar_reference(raw, anode='HEOG_right', cathode='HEOG_left',
                            ch_name='HEOG', drop_refs=False)
raw = raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
raw = raw.drop_channels(['VEOG_lower', 'HEOG_right', 'HEOG_left'])
raw = raw.set_montage('biosemi64', match_case=False)
raw = raw.filter(l_freq=0.1, h_freq=30.0)
raw_copy = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
ica = ICA(n_components=15)
ica = ica.fit(raw_copy)
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['VEOG', 'HEOG'],
                                            verbose=False)
ica.exclude = eog_indices
raw = ica.apply(raw)
raw = raw.set_eeg_reference('average')

# Epoching
log = pd.read_csv(log_file, sep='\t')
events = log[['sample', 'duration', 'value']].values.astype(int)
events = merge_events(events, ids=range(1, 41), new_id=1)
events = merge_events(events, ids=range(41, 81), new_id=2)
event_id = {'face': 1, 'car': 2}
epochs = Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8,
                baseline=(-0.2, 0.0), preload=True)
epochs = epochs.drop_bad({'eeg': 200e-6})

# %% [markdown]
# This gives us the cleaned, epoched data as an `Epochs` object.
#
# %%
epochs

# %% [markdown]
# ## Averaging
#
# The actual averaging step is simple: We just need to use the `average()` method on an `Epochs` object.
# This will return an `Evoked` object, which contains the averaged data (channels × timepoints).
#
# %%
evokeds = epochs.average()
evokeds

# %%
evokeds.get_data().shape

# %% [markdown]
# Like all MNE objects, evokeds have a `plot()` method to visualize the averaged time course at a single EEG channel:

# %%
_ = evokeds.plot(picks='PO8')

# %% [markdown]
# Or at all channels (also called a **butterfly plot**):
#
# %%
_ = evokeds.plot()

# %% [markdown]
# But wait!
# Now we have averaged across *all* trials, but we are interested in the *difference* between face and car trials.
# To do so, we can perform the averaging twice, each time selecting only the epochs of one of the two conditions.
# We add a string as a `comment` to each evoked object to keep track of which is which.
#
# %%
evokeds_face = epochs['face'].average()
evokeds_face.comment = 'face'
evokeds_face

# %%
evokeds_car = epochs['car'].average()
evokeds_car.comment = 'car'
evokeds_car

# %% [markdown]
# That way, we can now compare the two conditions by showing both of them in the same plot:
#
# %%
evokeds_list = [evokeds_face, evokeds_car]
_ = plot_compare_evokeds(evokeds_list, picks='PO8')

# %% [markdown]
# ## Difference wave
#
# From the two separate evokeds for the two conditions, we can also compute a **difference wave,** which is simply the difference (subtraction) between the two evokeds at each time point.
#
# There are multiple ways to implement this subtraction in code.
# One would be to extract the two Numpy arras (using the `get_data()` method) and then subtract them from one another.
# Or we could use MNE's `combine_evoked()` function with a specific set of weights that will perform the subtraction:
#
# %%
evokeds_diff = combine_evoked(evokeds_list, weights=[1, -1])
evokeds_diff.comment = 'face - car'
_ = evokeds_diff.plot()

# %% [markdown]
# The **time course plots** that we have created thus far are showing the data for a single EEG channel and all time points (i.e., one *row* of the data matrix).
# Alternatively, we could plot the data for all EEG channels at a single time point (i.e., one *column* of the data matrix).
# This is called a **scalp topography**.
#
# %%
_ = evokeds_diff.plot_topomap(times=[0.17])

# %%
# The `Evoked` object even has a method to plot the time course (butterfly) plot and a few scalp topographies at the same time:
#
# %%
_ = evokeds_diff.plot_joint(times=[0.0, 0.15, 0.17])

# %% [markdown]
# ## Exercises
#
# 1. Repeat the preprocessing and epoching (first code cell) for a different ERP CORE experiment (valid experiment names are `'N170'`, `'MMN'`, `'N2pc'`, `'N400'`, `'P3'`, or `'ERN'`).
#    Create evokeds for the two conditions in your experiment and visualize them using time course and scalp topography plots.
#
# %%  tags=["skip-execution"]
# Your code goes here
...

# %% [markdown]
# ## Further reading
#
# * Tutorial on evokeds on the [MNE-Python website](https://mne.tools/stable/auto_tutorials/evoked/10_evoked_overview.html)
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
#
