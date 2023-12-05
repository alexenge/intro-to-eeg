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
#
# %%
# # %pip install mne hu-neuro-pipeline

# %%
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
eog_names = ['VEOG', 'HEOG']
eog_anodes = ['FP1', 'HEOG_right']
eog_cathodes = ['VEOG_lower', 'HEOG_left']
eog_drop = ['VEOG_lower', 'HEOG_left', 'HEOG_right']
montage = 'biosemi64'
l_freq = 0.1
h_freq = 30.0
n_components = 15

files_dict = get_erpcore('N170', participants='sub-004', path='data')
raw_file = files_dict['raw_files'][0]

raw = read_raw(raw_file, preload=True)

for eog_name, anode, cathode in zip(eog_names, eog_anodes, eog_cathodes):
    raw = set_bipolar_reference(raw, anode, cathode, eog_name, drop_refs=False)
    raw = raw.set_channel_types({eog_name: 'eog'})
raw = raw.drop_channels(eog_drop)
raw = raw.set_montage(montage, match_case=False)

raw = raw.filter(l_freq, h_freq)

raw_copy = raw.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=n_components)
ica = ica.fit(raw_copy)
eog_indices, eog_scores = ica.find_bads_eog(raw, eog_names, verbose=False)
ica.exclude = eog_indices
raw = ica.apply(raw)

raw = raw.set_eeg_reference('average')

# %% [markdown]
# This gives us the cleaned, continuous EEG data as a `Raw` object.
#
# %%
raw

# %% [markdown]
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
# We can see that there are many different numerical event codes without any obvious meaning.
# We can check what these codes stand for by looking at the `task-N170_events.json` file that accompanies the dataset:
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
