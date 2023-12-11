# %% [markdown]
# # Pipeline
#
# Thus far, we've applied the preprocessing, epoching, and averaging steps to a single participant only.
# However, we typically study a larger group of participants, since (a) data from a single participant is typically still quite noisy, and (b) we want to generalize our findings to the population from which our participants were drawn.
#
# Therefore, we need to repeat the analysis for all participants by creating an **analysis pipeline** that takes each participant's raw data as an input and performs the same set of processing steps on them.
#
# We will first do this manually (using a loop) and then with a pre-packaged automatic pipeline function.
#
# ```{admonition} Goals
# :class: note
#
# * Repeating the processing steps for all participants
# * Using a fully automated pipeline
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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne import Epochs, combine_evoked, merge_events, set_bipolar_reference
from mne.io import read_raw
from mne.preprocessing import ICA
from mne.viz import plot_compare_evokeds
from pipeline import group_pipeline
from pipeline.datasets import get_erpcore

# %% [markdown]
# ## Custom pipeline
#
# We'll start by creating a custom pipeline that repeats the preprocessing, epoching, and averaging steps for all participants.
# This is as easy as taking all the code from the previous chapters and, instead of reading only a single raw EEG file, putting it inside a `for` loop that iterates over all raw EEG files.
#
# Let's first download some more datasets (for the sake of time, we're only using 10 participants instead of all 40):
#
# %% tags=["output_scroll"]
files_dict = get_erpcore('N170', participants=10, path='data')

# %%
files_dict

# %% [markdown]
# Then we'll run a `for` loop with all the processing steps inside it.
# Before the loop, we create empty lists where we will store the processed outputs (in this case, the evokeds) for all participants.
#
# %% tags=["output_scroll"]
evokeds_face = []
evokeds_car = []
evokeds_diff = []

for raw_file, log_file in zip(files_dict['raw_files'],
                              files_dict['log_files']):

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

    # Averaging
    evoked_face = epochs['face'].average()
    evokeds_face.append(evoked_face)
    evoked_car = epochs['car'].average()
    evokeds_car.append(evoked_car)
    evoked_list = [evoked_face, evoked_car]
    evoked_diff = combine_evoked(evoked_list, weights=[1, -1])
    evokeds_diff.append(evoked_diff)

# %% [markdown]
# We have collected the averaged ERP (evoked) for both conditions (face and car) from all participants.
# Averaging one more time, this time not across trials but across participants, gives us the **grand average**.
# Let's do this separately for both condition and display the grand averages as a time course plot:
#
# %%
grand_evoked_face = combine_evoked(evokeds_face, weights='equal')
grand_evoked_face.comment = 'face'

grand_evoked_car = combine_evoked(evokeds_car, weights='equal')
grand_evoked_car.comment = 'car'

grand_evoked_list = [grand_evoked_face, grand_evoked_car]
_ = plot_compare_evokeds(grand_evoked_list, picks='PO8')

# %% [markdown]
# We see that the first negative peak in the ERP (N1/N170 component) is earlier and (and maybe also larger) for faces than for cars.
#
# This becomes even more apparent when grand-averaging and plotting the difference waves:
#
# %%
grand_evoked_diff = combine_evoked(evokeds_diff, weights='equal')
grand_evoked_diff.comment = 'face - car'
_ = grand_evoked_diff.plot(picks='PO8')

# %% [markdown]
# The corresponding butterfly and scalp topography plot looks like this:
#
# %%
_ = grand_evoked_diff.plot_joint(times=[0.0, 0.15, 0.17])

# %% [markdown]
# ## Neuro Lab pipeline
#
# At the [Abdel Rahman Lab for Neurocognitive Psychology](https://abdelrahmanlab.com) at HU Berlin (the "Neuro Lab," for short), we've developed a Python package that provides a fully automated EEG processing pipeline.
# The pipeline was originally developed and published in the MATLAB language by {cite:t}`fromer2018`.
# The more recent Python version is available at [https://hu-neuro-pipeline.readthedocs.io](https://hu-neuro-pipeline.readthedocs.io).
#
# The pipeline is fully automated in the sense that it takes raw EEG data from multiple participants as an input and performs a number of standardized processing step at the participant and group level with minimal user input.
# The typical steps are visualized in this flowchart:
#
# :::{figure-md}
# <img src="https://hu-neuro-pipeline.readthedocs.io/en/stable/_images/flowchart.svg" alt="Flowchart of the processing steps in the `hu-neuro-pipeline package`" width=400>
#
# Processing steps in the `hu-neuro-pipeline` package.
# Source: [Docs](https://hu-neuro-pipeline.readthedocs.io/en/stable/processing_overview.html).
# :::
#
# The pipeline package only has one main function, called `group_pipeline()`.
# Let's use it to process the same example data as before:
#
# %% tags=["output_scroll"]
trials, evokeds, config = group_pipeline(raw_files=files_dict['raw_files'],
                                         log_files=files_dict['log_files'],
                                         output_dir='output',
                                         montage='biosemi64',
                                         ica_method='fastica',
                                         ica_n_components=15,
                                         triggers=range(1, 81),
                                         skip_log_conditions={'value': range(81, 203)},
                                         components={'name': 'N170',
                                                     'tmin': 0.15,
                                                     'tmax': 0.2,
                                                     'roi': ['PO8']},
                                         average_by={'face': 'value <= 40',
                                                     'car': 'value > 40'})

# %% [markdown]
# In this example we've specified:
#
# * The input and output file paths (`raw_files`, `log_files`, `output_dir`)
# * Some preprocessing options (`montage`, `ica_method`, `ica_n_components`; note that the pipeline also applies a 0.1--40 Hz band-pass filter by default)
# * The event codes of interest for epoching (`triggers`; note that the pipeline also applies a default peak-to-peak rejection at 200 µV)
# * Some event codes to skip (`skip_log_conditions`; anything other than faces and houses)
# * The definition of our ERP component(s) of interest (`components`; here only the N170)
# * Rules to create by-participant condition averages/evokeds (`average_by`; here for faces and cars)
#
# A (long) list of these and all other input options is available on the [pipeline documentation website](https://hu-neuro-pipeline.readthedocs.io/en/stable/usage_inputs.html).
#
# The pipeline returns three objects (which also get written as text files into the `output_dir`):
#
# * `trials`: A data frame with the **single trial data** for all participants, also containing the single trial ERP amplitudes (averaged across the time window and channels of interest)
# * `evokeds`: A data frame with the **averaged (evoked) ERP amplitudes** for all time points, channels, and participants
# * `config`: A dictionary with the **pipeline configuration**
#
# Let's look of each of these in turn:
#
# %% [markdown]
# ### Single trial data
#
# %%
trials

# %% [markdown]
# We could (and will!) use this for statistically analyzing the ERP component(s) of interest.
# Specifically, we can fit a mixed-effects model that tests if the single trial N170 amplitudes differs as a function of the condition of the trial (face vs. car).
#
# %% [markdown]
# ### Evokeds
#
# %%
evokeds

# %% [markdown]
# The evokeds can be used for plotting the ERP time courses for both conditions.
# For this, we will use the [seaborn package](https://seaborn.pydata.org), which is great for visualizing tabular data (similar to the [ggplot2 package](https://ggplot2.tidyverse.org) in R).

# %%
_ = sns.lineplot(evokeds, x='time', y='PO8', hue='label',
                 estimator='mean', errorbar='se')

# %% [markdown]
# We can make the plot yet a bit prettier by adding custom x-axis limits, vertical and horizontal lines at zero, and more informative axis labels:
#
# %%
_ = sns.lineplot(evokeds, x='time', y='PO8', hue='label',
                 estimator='mean', errorbar='se')
_ = plt.margins(x=0.0, y=0.1)
_ = plt.xlim(-0.2, 0.8)
_ = plt.axvline(0.0, color='black', linestyle='--')
_ = plt.axhline(0.0, color='black', linestyle='--')
_ = plt.xlabel('Time (s)')
_ = plt.ylabel('PO8 amplitude (µV)')

# %% [markdown]
# ### Pipeline configuration
#
# %% [markdown]
# Finally, the `config` output is a dictionary with information about the pipeline run.
# It contains all the user-specified and default input arguments plus some new information that the pipeline has computed along the way (e.g., `'auto_rejected_epochs'`, the rejected epochs for each participant):
#
# %% tags=["output_scroll"]
config

# %% [markdown]
# ## Exercises
#
# 1. Run the EEG analysis pipeline (using a custom `for` loop or the `group_pipeline()` function) for 10 participants from a different ERP CORE experiment (valid experiment names are `'N170'`, `'MMN'`, `'N2pc'`, `'N400'`, `'P3'`, or `'ERN'`).
#    Create evokeds for the two conditions of interest and visualize their time course.
#
# %%  tags=["skip-execution"]
# Your code goes here
...

# %% [markdown]
# ## Further reading
#
# * Paper *Group-level EEG-processing pipeline for flexible single trial-based analyses including linear mixed models* {cite:p}`fromer2018`
# * hu-neuro-pipeline package [documentation](https://hu-neuro-pipeline.readthedocs.io)
# * [Slides](https://github.com/alexenge/hu-neuro-pipeline-workshop) on the hu-neuro-pipeline package
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
#
