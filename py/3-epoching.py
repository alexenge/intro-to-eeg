# %% [markdown]
# # Epoching
#
# We are typically not interested in the full continuous EEG recording, but only in the EEG activity that happens around certain **events** of interest.
# Such events could be the onset of a stimulus or the onset of a response.
#
# The events are usually stored in the EEG data as "triggers" (also called "markers" or "annotations").
# These are numerical codes at certain timepoints that indicate the onset of an event.
# We can use them to "cut" the continuous EEG recording into smaller segments (called **epochs**) that begin a few hundred milliseconds before the event (to have a neutral "baseline" period) and end a few hundred milliseconds after the event (to capture brain activity related to the event).
#
# :::{figure-md}
# <img src="https://files.mtstatic.com/site_7339/94057/0?Expires=1700853581&Signature=UeBaEMEJOBCxd9GX-iXbaSsW6XPM4iX2uABmaWUjOB1~qIaRQf-m4158~2wbQeVqmaaLPnX6o04fkunrzmP88Gr-zGgPhrMBes0MEkLXLy7B43XgdPwUPpO3tUVhOjvHrAuQthoD9lv9b2IYJNsFZ1hcLWeW4ZsezFs7~iSTMFM_&Key-Pair-Id=APKAJ5Y6AV4GI7A555NA" alt="Continuous, epoched, and averaged EEG" width=500>
#
# Continuous, epoched, and averaged EEG.
# Source: {cite:t}`luck2022a`
# :::
#
# ```{admonition} Goals
# :class: note
#
# * Extracting event information from the data
# * Segmenting the data into epochs
# * Applying baseline correction
# * Rejecting high-amplitude epochs
# * Visualizing the epoched data
# ```
#
# %% [markdown]
# ## Load Python modules
#
# As before, we'll make extensive use of the [MNE-Python](https://mne.tools/stable/index.html) package{cite:p}`gramfort2013`.
# We'll also use a new package called [pandas](https://pandas.pydata.org) (more on that below).
#
# %%
# # %pip install mne hu-neuro-pipeline pandas

# %%
import pandas as pd
from mne import (Epochs, events_from_annotations, merge_events,
                 set_bipolar_reference)
from mne.io import read_raw
from mne.preprocessing import ICA
from pipeline.datasets import get_erpcore

# %% [markdown]
# ## Recreate preprocessing
#
# We repeat the preprocessing steps from the previous chapter in condensed form (without any intermediate plots).
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

# %% [markdown]
# This gives us the cleaned, continuous EEG data as a `Raw` object.
#
# %%
raw

# %% [markdown]
# (events)=
# ## Extract events
#
# We can extract the event codes that are stored within the raw data using the `events_from_annotations()` function.
#
# %%
events, event_id = events_from_annotations(raw)

# %% [markdown]
# This functions returns two outputs:
#
# * `events` is a Numpy array with three columns: the sample number (timepoint) of each event, the duration of the event, and the event code (here indicating the specific face or car stimulus)
# * `event_id`: a dictionary mapping the event codes to human-readable names
#
# %%
events

# %% tags=["output_scroll"]
event_id

# %% [markdown]
# An alternative way to get the events is to use an explicit events file that accompanies the raw data.
# This is useful if there are no events stored in the raw data or if they are erroneous (as is actually the case here for the N170 dataset!).
#
# For instance, the EEG data from each participant in the ERP CORE datasets (e.g., `sub-004_task-N170_eeg.set`) is accompanied by a tab-separated events file (e.g., `sub-004_task-N170_events.tsv`):
#
# %% tags=["output_scroll"]
# %cat data/erpcore/N170/sub-004/eeg/sub-004_task-N170_events.tsv

# %% [markdown]
# We can read this file using the [pandas](https://pandas.pydata.org) package, which offers a `DataFrame` type for working tabular data (similar to a `data.frame` or `tibble` in R).
#
# %%
log_file = files_dict['log_files'][0]
log = pd.read_csv(log_file, sep='\t')
log

# %% [markdown]
# From this data frame, we can re-create the 3-column `events` array (see above), now with the correct timings and event codes.
# Note that by converting all columns from floats to integers, we are rounding the duration of the events from `0.3` (the actual length of the stimulus) to `0`, as is common practice in ERP analysis.
#
# %%
events = log[['sample', 'duration', 'value']].values.astype(int)
events

# %% [markdown]
# However, these many different numerical event codes still have no obvious meaning to us.
# We can check what they stand for by looking at the `task-N170_events.json` file that accompanies the dataset:
#
# %%
# %cat data/erpcore/N170/task-N170_events.json

# %% [markdown]
# We see that triggers 1--40 correspond to face stimuli, and triggers 41--80 correspond to car stimuli.
# To make our epoching job easier, we'll collapse these 80 original event codes into just two event codes: `1` for faces and `2` for cars, using MNE's `merge_events()` function.
#
# %%
events = merge_events(events, ids=range(1, 41), new_id=1)
events = merge_events(events, ids=range(41, 81), new_id=2)
events

# %% [markdown]
# Note that `range()` is a built-in Python function that generates a sequence of numbers (integers) between a start and end value.
# For example, `range(1, 41)` generates the sequence 1, 2, ..., 40 (the end value is not included).
#
# We also need to update the `event_id` dictionary accordingly, mapping the new event codes to human-readable condition labels.
#
# %%
event_id = {'face': 1, 'car': 2}

# %% [markdown]
# ## Epoching
#
# No we're ready to segment our raw data into epochs, using the `events` and `event_id` that we just created.
# We'll use a time window of 1 s, starting 200 ms before the event onset.
#
# For now, we'll skip baseline correction (this would be enabled by default) because we want to show the effect of baseline correction later.
#
# %%
epochs = Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=None)

# %%
epochs.get_data().shape

# %% [markdown]
# We see that the dimensions have changed, from EEG channels × timepoints (`raw`) to epochs × EEG channels × timepoints (`epochs`).
#
# Just as the raw data, the epochs also have a `plot()` method to visualize the data.
#
# %%
_ = epochs.plot(events=True)

# %% [markdown]
# Another effective way of visualizing *all* epochs in the dataset is the so-called **ERP image** plot.
# This is a heatmap with time on the x-axis, epochs on the y-axis, and the EEG amplitude represented by color.
#
# We'll create this plot for a single EEG channel (`'PO8'`) that is typically sensitive to the N170 face effect.
#
# %%
_ = epochs.plot_image(picks='PO8')

# %% [markdown]
# This type of plot could also be used to check for interesting patterns in the data, e.g., by sorting the epochs on the y-axis by some variables of interest (e.g., stimulus condition or reaction time).
#
# %% [markdown]
# ## Baseline correction
#
# Some of the epochs have a large offset (vertical shift) compared to the other epochs, which can happen due to technical or physiological drifts at some channels.
#
# We can remove these offsets by applying a baseline correction.
# This works by, separately for each channel, subtracting the mean of the baseline activity from each timepoint in the epoch.
# The baseline activity is typically defined as the 200 ms prior to the event onset.
#
# %%
epochs = epochs.apply_baseline((-0.2, 0.0))
_ = epochs.plot(events=True)

# %% [markdown]
# ## Rejecting bad epochs
#
# Despite all our data cleaning efforts (filtering, ICA, referencing, baseline correction), some epochs will still contain large-amplitude artifacts, e.g., due to movements or technical glitches.
#
# We can "reject" (delete) these epochs from the data by using the `drop_bad()` method (or, alternatively, the `reject=...` argument of the `Epochs` constructor).
# This allows us to specify a peak-to-peak-threshold (in volts).
# If, at any channel, the difference between the minimum and maximum amplitude in an epoch exceeds this threshold, the epoch will be rejected (deleted).
#
# Depending on how clean the dataset is, this threshold will typically be between 50 and 200 µV.
# Note that the lower the threshold (that is, the more epochs we're rejecting), the cleaner the remaining epochs will be, but the fewer epochs we'll have left, potentially reducing our statistical power.
#
# %%
epochs = epochs.drop_bad({'eeg': 100e-6})

# %% [markdown]
# We see that the ERP image now has a reduced number of epochs (rows), but the color patterns are much clearer (as some large-amplitude epochs have been removed):

# %%
_ = epochs.plot_image(picks='PO8')

# %% [markdown]
# ## Exercises
#
# 1. Repeat the preprocessing (first code cell) for a different ERP CORE experiment (valid experiment names are `'N170'`, `'MMN'`, `'N2pc'`, `'N400'`, `'P3'`, or `'ERN'`).
#    Then check the `task-<experiment>_events.json` file and construct the correct `events` and `event_id` variables.
#    Finally, create epochs from the data and plot the ERP image for a channel that is typically sensitive to the effect of interest (check {cite:t}`kappenman2021`, Table 1, for a suggestion).
#
# %%  tags=["skip-execution"]
# %cat data/erpcore/.../task-..._events.json

# %%  tags=["skip-execution"]
# Your code goes here
...

# %% [markdown]
# ## Further reading
#
# * Online chapter [*Segmentation into ERP epochs*](https://neuraldatascience.io/7-eeg/erp_segmentation.html) in {cite:t}`newman2020`
# * Tutorial on epochs on the [MNE-Python website](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html)
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
#
