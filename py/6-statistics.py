# %% [markdown]
# # Statistics
#
# In the previous chapters, we already got an idea about the differences in EEG activity between our conditions of interest (e.g., faces versus cars), as shown in the time course and scalp topography plots.
# However, we have to model the data statistically to be able to quantify the size of this difference and if it is statistically significant (i.e., not due to chance).
#
# In this chapter, we will encounter different statistical tests that can be used to test hypotheses about the data.
# While the EEG processing is done in Python as before, we will use the R programming language for statistical modeling because it has a larger number of statistical functions and packages, and is widely used in the psychological science community.
#
# ```{admonition} Goals
# :class: note
#
# * Apply "classical" models based on averaged data (e.g., $t$-tests, ANOVA)
# * Apply linear mixed-effects models to the single trial data
# ```
#
# %% [markdown]
# ## Load Python packages
#
# We'll use the hu-neuro-pipeline package introduced in Chapter 5 (Pipeline) for EEG processing, and Numpy and seaborn for post-processing and plotting.
# As mentioned before, the actual statistical modeling will be done in R, but there are also Python packages for this (e.g., [statsmodels](https://www.statsmodels.org/stable/index.html)).
#
# %%
import numpy as np
import seaborn as sns
from pipeline import group_pipeline
from pipeline.datasets import get_erpcore

# %% [markdown]
# ## EEG processing pipeline
#
# We use the same processing pipeline as introduced in Chapter 5 (Pipeline), giving us the single trial data and the average time courses.
#
# %% tags=["output_scroll"]
files_dict = get_erpcore('N170', participants=10, path='data')

trials, evokeds, config = group_pipeline(raw_files=files_dict['raw_files'],
                                         log_files=files_dict['log_files'],
                                         output_dir='output',
                                         montage='biosemi64',
                                         ica_method='fastica',
                                         ica_n_components=15,
                                         triggers=range(1, 81),
                                         skip_log_conditions={'value': range(81, 203)},
                                         components={'name': 'N170',
                                                     'tmin': 0.110,
                                                     'tmax': 0.150,
                                                     'roi': ['PO8']},
                                         average_by={'face': 'value <= 40',
                                                     'car': 'value > 40'})

# %% [markdown]
# ## Single trial data
#
# The main output of the hu-neuro-pipeline package is the single trial data frame, which contains the EEG data for each trial, averaged across an *a priori* hypothesized time window and electrode(s) of interest (see the `components` argument above).
#
# %%
trials

# %% [markdown]
# Using a combination of pandas and Numpy, we'll create a new column in the data frame with verbal labels for our two conditions of interest (faces and cars).
# This is based on the numerical event codes (stored in the `value` column), the meaning of which was described in Chapter 3 (Epoching).
#
# %%
trials['condition'] = np.where(trials['value'] <= 40, 'face', 'car')
trials

# %% [markdown]
# Using seaborn, we can plot the distribution of the single trial N170 amplitudes, separately for the two conditions.
# Note that this plot does not take into account the repeated measurements of the same participant, which we will address later.
#
# %%
_ = sns.violinplot(data=trials, y='N170', hue='condition',
                   inner='quart', split=True, fill=False)

# %% [markdown]
# ## Linear models
#
# The "traditional" way for statistical analysis of ERPs is to (a) average the data across trials for each participant and condition, and (b) apply a statistical test to the averaged data.
# Let's start with the first step.
# The pandas package has the necessary methods to group the data by participant and condition (`groupby()` method), and compute the average N170 amplitude across trials for each grouping (`mean()` method).
#
# %%
trials_ave = trials[['participant_id', 'condition', 'N170']].\
    groupby(['participant_id', 'condition']).\
    mean().\
    reset_index()

# %% [markdown]
# Now we can pass the data frame to R and apply an appropriate statistical test.
# Using the `rpy2` package, we can run R code directly in the Jupyter notebook, using the `%%R` magic command at the beginning of a code cell.
#
# %%
# %load_ext rpy2.ipython

# %% [markdown]
# An appropriate statistical test needs to take into account that our two conditions are manipulated *within* participants, that is, we have repeated measures that are likely correlated with one another (violating the independence assumption of many statistical tests, e.g., linear regression).
#
# Luckily, there are statistical tests that can handle repeated measures data, such as the paired $t$-test or repeated measures ANOVA.
#
# Let's start with the paired $t$-test:
#
# %%
# %%R -i trials_ave
t.test(N170 ~ condition, data = trials_ave, paired=TRUE)

# %% [markdown]
# We see that in this sample, the amplitude in response to faces is approximately 2.5 µV lower (more negative) than in response to cars, as would be expected for the N170 component.
# This difference is statistically significant with $t(9) = -6.6$, $p \approx .0001$.
#
# Note that we could have gotten the same result by applying a one sample $t$-test to the difference scores:
#
# %%
trials_ave_wide = trials_ave.pivot(index='participant_id', columns='condition', 
                                   values='N170')
trials_ave_wide['diff'] = trials_ave_wide['car'] - trials_ave_wide['face']
trials_ave_wide

# %%
# %%R -i trials_ave_wide
t.test(trials_ave_wide$diff)

# %% [markdown]
# Or by running a repeated measures ANOVA with a single (two-level) factor:

# %%
# %%R -i trials_ave

# install.packages("ez")

ez::ezANOVA(
  data = trials_ave,
  dv = N170,
  wid = participant_id,
  within = condition
)

# %% [markdown]
# ## Linear mixed-effects models
#
# %%
# %%R -i trials

# install.packages("lme4")

mod <- lme4::lmer(N170 ~ 1 + condition + (1 | participant_id), trials)
summary(mod)

# %%
# %%R -i trials

# install.packages("lmerTest")

mod <- lmerTest::lmer(N170 ~ 1 + condition + (1 | participant_id), trials)
summary(mod)
