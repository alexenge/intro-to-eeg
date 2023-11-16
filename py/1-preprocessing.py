# %% [markdown]
# # 1. Preprocessing
#
# **Preprocessing** is the first step in EEG data analysis.
# It usually involves a series of steps aimed at removing non-brain-related noise and artifacts from the data.
# Unlike the following steps (e.g., epoching and averaging), it leaves the data in a continuous format (EEG channels × timepoints).
#
# ```{admonition} Learning goals
# :class: note
# * Load raw EEG data from a single participant
# * Plot the raw data
# * Filter the data to remove low and high frequency noise
# * Correct eye artifacts using independent component analysis (ICA)
# * Re-reference the data to the average of all electrodes
# ```
#
# %% [markdown]
# ## 1.1. Load Python modules
#
# We will use the following Python modules:
# * [MNE](https://mne.tools/stable/index.html) for EEG data analysis {cite:p}`gramfort2013`
# * [hu-neuro-pipeline](https://github.com/alexenge/hu-neuro-pipeline) for downloading example data
#
# Note that on Google Colab, you will need to install these modules first.
# You can uncomment and run the following cell to do so.
#
# %%
# # %pip install mne hu-neuro-pipeline

# %%
from mne import set_bipolar_reference
from mne.io import read_raw
from mne.preprocessing import ICA
from mne.viz import set_browser_backend
from pipeline.datasets import get_erpcore

# %% [markdown]
# ## 1.2 Download example data
#
# We'll use data from the ERP CORE dataset {cite:p}`kappenman2021`.
# This dataset contains EEG data from 40 participants who completed 6 different experiments.
# Each experiment was designed to elicit one or two commonly studied ERP components.
#
# <img src="https://ars.els-cdn.com/content/image/1-s2.0-S1053811920309502-gr1.jpg" width="500">
# <br><br>
#
# In this example, we'll use the data from one participant from the face percpetion (N170) experiment.
#
# %%
files_dict = get_erpcore('N170', participants='sub-004')
files_dict

# %% [markdown]
# ## 1.3 Load raw data
#
# We read the actual EEG data files (`eeg.set`/`eeg.fdt`) into MNE-Python.
# The result is a `Raw` object, which contains the continuous EEG data and some metadata.
#
# %%
raw_file = files_dict['raw_files'][0]
raw = read_raw(raw_file, preload=True)
raw

# %% [markdown]
#
# We can access the actual data array using the `get_data()` method.
# Let's check the size (number of dimensions and their length) of this array:
#
# %%
raw.get_data().shape

# %% [markdown]
# ## 1.4. Plot raw data
#
# We can plot the raw data using the `plot()` method.
# In notebook-like environments (such as Google Colab), we need to use the `'matplotlib'` backend, which will create a static image.
# On a local machine, we could use the default `'browser'` backend, which will create an interactive plot in a new window.
#
# We specifiy which time segment of the data to plot using the `start` and `duration` arguments.
# Here we plot 5 seconds of data, starting at 60 seconds.
#
# %%
set_browser_backend('matplotlib')
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# ## 1.5 Add channel information
#
# %%
raw = set_bipolar_reference(raw, anode='FP1', cathode='VEOG_lower',
                            ch_name='VEOG', drop_refs=False)
raw = set_bipolar_reference(raw, anode='HEOG_right', cathode='HEOG_left',
                            ch_name='HEOG', drop_refs=False)
raw = raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
raw = raw.drop_channels(['VEOG_lower', 'HEOG_right', 'HEOG_left'])

# %%
raw = raw.set_montage('biosemi64', match_case=False)

# %% [markdown]
# ## 1.6 Filter data
#
# Filtering is a common preprocessing step that is used to remove parts of the EEG signal that are unlikely to contain brain activity of interest.
# There are four different types of filters:
#
# * A **high-pass filter** removes low-frequency noise (e.g., slow drifts due to sweat or breathing)
# * A **low-pass filter** removes high-frequency noise (e.g., muscle activity)
# * A **band-pass filter** combines a high-pass and a low-pass filter in one step
# * A **band-stop filter** removes a narrow band of frequencies (e.g., 50 Hz line noise)
#
# We first apply a high-pass filter at 0.1 Hz to remove slow drifts and plot the filtered data.
#
# %%
raw = raw.filter(l_freq=0.1, h_freq=None)
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
#
# Next, we apply a low-pass filter at 30 Hz to remove high-frequency noise and plot the data again.
#
# Note that we've performed these two filters separately for demonstration purposes, but we could have also applied a single band-pass filter at 0.1--30 Hz.
#
# %%
raw = raw.filter(l_freq=None, h_freq=30.0)
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# ## 1.7 Correct eye artifacts
#
# Eye blinks and eye movements are the most prominent source of artifacts in EEG data.
# They are approximately 10 times larger than the brain signals we are interested in and affect especially the frontal electrodes.
#
# There are multiple ways to remove eye artifacts from EEG data.
# The most common one is a machine learning technique called **independent component analysis (ICA)**.
# ICA decomposes the EEG data into a set of independent components, each of which represents a different source of EEG activity.
#
# Each component is characterized by a topography (i.e., a spatial pattern of activity across electrodes) and a time course (i.e., a pattern of activity over time).
# We can then identify those components that we think reflect eye artifacts and remove them from the data.
#
# ICA is typically applied on a high-pass filtered copy of the data (cutoff = 1 Hz).
# We ask the algorithm to identify 15 components and plot their scalp topographies.
#
# %%
raw_copy = raw.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15)
ica = ica.fit(raw_copy)
_ = ica.plot_components()

# %%
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['HEOG', 'VEOG'],
                                            verbose=False)
ica.exclude = eog_indices
_ = ica.plot_scores(eog_scores)

# %%
raw = ica.apply(raw)
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# ## 1.8 Re-reference data
#
# %%
raw = raw.set_eeg_reference('average')
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# ## Exercises
#
# 1. Re-run the above analysis for a different experiment.
#    For this, you can simply reuse the code cells above, changing only the second cell.
#    Valid experiment names are `'N170'`, `'MMN'`, `'N2pc'`, `'N400'`, `'P3'`, or `'ERN'`.
# 2. Below, try out the effect of different filter settings such as a higher high-pass cutoff or a lower low-pass cutoff.
#    For this, write your own code that achieves the following:
#    (a) read the raw data from one participant,
#    (b) apply your own custom high-pass, low-pass, or band-pass filter,
#    (c) plot the filtered data, and
#    (d) repeat for different filter settings.
#
# %%  tags=["skip-execution"]
# Your code goes here
...

# %% [markdown]
# ## References
#
# ```{bibliography}
# ```
#
