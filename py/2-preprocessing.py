# %% [markdown]
# # Preprocessing
#
# **Preprocessing** is the first step in EEG data analysis.
# It usually involves a series of steps aimed at removing non-brain-related noise and artifacts from the data.
# Unlike the following steps (e.g., epoching and averaging), it leaves the data in a continuous format (EEG channels × timepoints).
#
# ```{admonition} Goals
# :class: note
#
# * Loading raw EEG data
# * Plotting the raw data
# * Filtering the data to remove low and high frequency noise
# * Correcting eye artifacts using independent component analysis (ICA)
# * Re-referencing the data to an average reference
# ```
#
# %% [markdown]
# ## Load Python modules
#
# We will use the following Python modules:
# * [MNE-Python](https://mne.tools/stable/index.html) for EEG data analysis {cite:p}`gramfort2013`
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
# ## Download example data
#
# We'll use data from the ERP CORE dataset {cite:p}`kappenman2021`.
# This dataset contains EEG data from 40 participants and 6 different experiments.
# Each experiment was designed to elicit one or two commonly studied ERP components.
#
# :::{figure-md}
# <img src="https://ars.els-cdn.com/content/image/1-s2.0-S1053811920309502-gr1.jpg" width="500">
#
# The six different ERP CORE experiments.
# Source: {cite:t}`kappenman2021`
# :::
#
# In this example, we'll use the data from the fourth participant in the face perception (N170) experiment.
#
# %%
files_dict = get_erpcore('N170', participants='sub-004', path='data')
files_dict

# %% [markdown]
# ## Load raw data
#
# We read the actual EEG data files (`eeg.set`/`eeg.fdt`) into MNE-Python.
# The result is a `Raw` object, which contains the continuous EEG data and some metadata.
#
# %%
raw_file = files_dict['raw_files'][0]
raw = read_raw(raw_file, preload=True)
raw

# %% [markdown]
# We can access the actual data array (a Numpy array) using the `get_data()` method.
#
# %%
raw.get_data()

# %% [markdown]
# Let's check the size (number of dimensions and their length) of this array:
#
# %%
raw.get_data().shape

# %% [markdown]
# We see that it has two dimensions (EEG channels × timepoints).
#
# %% [markdown]
# ## Plot raw data
#
# We can plot the raw data using the `plot()` method.
# We specify which time segment of the data to plot using the `start` and `duration` arguments.
# Here we plot 5 seconds of data, starting at 60 seconds.
#
# %%
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# ## Add channel information
#
# Right now, MNE thinks that all channels are EEG channels.
# However, we know that some of them are actually EOG channels that record eye movements and blinks.
# We'll use these to create new "virtual" EOG channels that pick up strong eye signals (vertical EOG [VEOG] = difference between above and below the eyes; horizontal EOG [HEOG] = difference between left and right side of the eyes).
# We explicitly set their channel type to `'eog'` and drop the original channels, so that we are left with 30 EEG channels and 2 EOG channels.
#
# %%
raw = set_bipolar_reference(raw, anode='FP1', cathode='VEOG_lower',
                            ch_name='VEOG', drop_refs=False)
raw = set_bipolar_reference(raw, anode='HEOG_right', cathode='HEOG_left',
                            ch_name='HEOG', drop_refs=False)
raw = raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
raw = raw.drop_channels(['VEOG_lower', 'HEOG_right', 'HEOG_left'])

# %% [markdown]
# Then we load the locations of the EEG electrodes as provided by the manufacturer of the EEG system.
# Many of these standard EEG montages are shipped with MNE-Python.
#
# %%
raw = raw.set_montage('biosemi64', match_case=False)
_ = raw.plot_sensors(show_names=True)

# %% [markdown]
# ## Filter data
#
# Filtering is a common preprocessing step that is used to remove parts of the EEG signal that are unlikely to contain brain activity of interest.
# There are four different types of filters:
#
# * A **high-pass filter** removes low-frequency noise (e.g., slow drifts due to sweat or breathing)
# * A **low-pass filter** removes high-frequency noise (e.g., muscle activity)
# * A **band-pass filter** combines a high-pass and a low-pass filter
# * A **band-stop filter** removes a narrow band of frequencies (e.g., 50 Hz line noise)
#
# We first apply a high-pass filter at 0.1 Hz to remove slow drifts and plot the filtered data.
#
# %%
raw = raw.filter(l_freq=0.1, h_freq=None)
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# Next, we apply a low-pass filter at 30 Hz to remove high-frequency noise and plot the data again.
#
# %%
raw = raw.filter(l_freq=None, h_freq=30.0)
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# Note that we've performed these two filters separately for demonstration purposes, but we could have also applied a single band-pass filter.
#
# %% [markdown]
# ## Correct eye artifacts
#
# Eye blinks and eye movements are the most prominent source of artifacts in EEG data.
# They are approximately 10 times larger than the brain signals we are interested in and affect especially the frontal electrodes.
#
# There are multiple ways to remove eye artifacts from EEG data.
# The most common one is a machine learning technique called **independent component analysis (ICA)**.
# ICA decomposes the EEG data into a set of independent components, each of which represents a different source of EEG activity.
#
# Each component is characterized by a *topography* (i.e., a spatial pattern of activity across electrodes) and a *time course* (i.e., a pattern of activity over time).
# We can use these to identify components that we think reflect eye artifacts, and remove them from the data.
#
# ICA is typically computed based on a high-pass filtered copy of the data (cutoff = 1 Hz).
# We ask the algorithm to identify 15 components and plot their scalp topographies.
#
# %%
raw_copy = raw.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15)
ica = ica.fit(raw_copy)
_ = ica.plot_components()

# %% [markdown]
# Then we can use a clever method that automatically identifies components that are likely to reflect eye artifacts (based on the correlation of the component's time course with our two VEOG and HEOG channels).
#
# %%
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['HEOG', 'VEOG'],
                                            verbose=False)
ica.exclude = eog_indices
_ = ica.plot_scores(eog_scores)

# %% [markdown]
# Finally, by "applying" the ICA to the data (formally, back-projecting the non-artifact components from component space to channel space), we can remove the eye artifacts from the data.
#
# %%
raw = ica.apply(raw)
_ = raw.plot(start=60.0, duration=5.0)

# %% [markdown]
# ## Re-reference data
#
# **Re-referencing** is our final preprocessing step.
# Since the EEG signal is measured as the difference in voltage between two electrodes, the signal at any given electrode depends strongly on the "online" reference electrode (typically placed on the mastoid bone behind the ear or on the forehead).
#
# During preprocessing ("offline"), we typically want to re-reference the data to a more neutral (and less noisy) reference, such as the average of all channels.
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
# ## Further reading
#
# * Tutorials on preprocessing on the [MNE-Python website](https://mne.tools/stable/auto_tutorials/preprocessing/index.html)
# * Blog post on [*Pitfalls of filtering the EEG signal*](https://sapienlabs.org/lab-talk/pitfalls-of-filtering-the-eeg-signal/) by Narayan P. Subramaniyam
# * Paper *EEG is better left alone* {cite:p}`delorme2023`
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
#
