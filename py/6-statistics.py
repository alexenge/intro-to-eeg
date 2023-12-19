# %% [markdown]
# # Statistics
#
# In the previous chapters, we already got an idea about the differences in EEG activity between our conditions of interest (e.g., faces versus cars), as shown in the time course and scalp topography plots.
# However, we have to model the data statistically to be able to quantify the size of this difference and test if it is statistically significant (i.e., not due to chance).
# In this chapter, we will encounter different **statistical tests** that can be used to do this.
#
# While the EEG processing is done in Python as before, we will use the **[R](https://www.r-project.org) programming language** for statistical modeling because it has a larger number of statistical functions and packages, and because it is widely used in the psychological science community.
#
# ```{admonition} Goals
# :class: note
#
# * Test for condition differences using "classical" models based on averaged data (e.g., $t$-tests, ANOVA)
# * Do the same by applying linear mixed models to the single trial data
# ```
#
# %% [markdown]
# ## Load Python packages
#
# We'll use the hu-neuro-pipeline package (introduced in Chapter 5) for EEG processing, and Numpy and seaborn for post-processing and plotting.
# As mentioned before, the actual statistical modeling will be done in R, but there are also Python packages for this (e.g., [statsmodels](https://www.statsmodels.org/stable/index.html)).
#
# %%
# # %pip install numpy seaborn hu-neuro-pipeline

# %%
import numpy as np
import seaborn as sns
from pipeline import group_pipeline
from pipeline.datasets import get_erpcore

# %% [markdown]
# ## Re-run the pipeline
#
# We use the same processing pipeline as introduced in the {ref}`pipeline` section of the previous chapter, giving us the single trial data and the average time courses.
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
# The main output of the hu-neuro-pipeline package is the **single trial data frame**, which contains the EEG data for each trial (column `N170`), averaged across an *a priori* hypothesized time window and electrode(s) of interest (see the `components` argument above).
#
# %%
trials

# %% [markdown]
# Using a combination of pandas and Numpy, we'll create a new column in the data frame with verbal labels for our two conditions of interest (faces and cars).
# This is based on the numerical event codes (stored in the `value` column), the meaning of which was described in the {ref}`events` section of the Epoching chapter.
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
#
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
# Let's start with the **paired $t$-test**:
#
# %%
# %%R -i trials_ave

t.test(N170 ~ condition, data = trials_ave, paired = TRUE)

# %% [markdown]
# We see that in this sample, the amplitude in response to faces is approximately 2.51 µV lower (more negative) than in response to cars, as would be expected for the N170 component.
# This difference is statistically significant with $t(9) = -6.56$, $p \approx .0001$.
#
# Note that we could have gotten the same result by applying a **one sample $t$-test** to the difference scores:
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
# Or by running a **repeated measures ANOVA** with a single (two-level, within-participant) factor:

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
# ## Linear mixed models
#
# The above approach of running a repeated measures linear model (ANOVA or $t$-test) on the averaged data is the "traditional" way of analyzing ERPs and still widely used.
# However, it comes with a number of drawbacks:
#
# * The averaging step discards a lot of information
# * We often don't just have repeated measures of the same participants, but also of the same items (again violating the independence assumption) {cite:p}`judd2012,burki2018`
# * We cannot include any information about specific trials or stimuli in the model
# * We cannot include any continuous predictor variables in the model
# * These models assume the same noise level (i.e., number of averaged trials) for all participants and conditions
#
# A more flexible approach that can solve all of these problems (and more!) is to use a **linear-mixed effects model (LMM)** {cite:p}`fromer2018,volpert-esmond2021`.
# This model predicts the single trial amplitudes directly and accounts for repeated measures of participants and/or items by including random effects for these factors.
# They can also include continuous predictor variables at the participant and trial level, and they do not required a balanced design (i.e., the same number of trials for each participant and condition).
#
# In R, we can use the `lmer()` function from the [lme4](https://github.com/lme4/lme4) package to fit LMMs {cite:p}`bates2015`:
#
# %%
# %%R -i trials

# install.packages("lme4")

mod <- lme4::lmer(N170 ~ 1 + condition + (1 | participant_id), trials)
summary(mod)

# %% [markdown]
# In this model, we're predicting the single trial N170 amplitude (`N170`) from a categorical predictor variable (`condition`) and an intercept (`1`) as fixed effects.
# We also specify a random intercept for the participant factor (`(1 | participant_id)`), which accounts for differences in the (average) voltage level between participants.
# Note that we could (and should) also include a random slope for the condition factor (`(1 + condition | participant_id)`), as well as a random intercept for the item factor (`(1 | value)`).
# However, in this example case with only 10 participants, this model would be overly complex and likely fail to converge.
#
# In the above output, you will not any find any $p$-values to decide if the fixed effects are statistically significant.
# If and how best to compute $p$-values for LMMs is still a matter of debate, but one common solution is a method called the Satterthwaite approximation.
# This is implemented in the [lmerTest](https://github.com/runehaubo/lmerTestR) package {cite:p}`kuznetsova2017`, which has a drop-in replacement for the `lmer()` function but with $p$-values:

# %%
# %%R -i trials

# install.packages("lmerTest")

mod <- lmerTest::lmer(N170 ~ 1 + condition + (1 | participant_id), trials)
summary(mod)

# %% [markdown]
# We see that there is a highly (statistically) significant reduction of N170 voltages for faces compared to cars, but also that the estimate and $p$-value are slightly different compared to the previous models (based on the averaged data, with all the drawbacks mentioned above).
#
# Note that there are other methods for statistical analysis of ERP data, such as **cluster-based permutation tests** (CBPTs) {cite:p}`maris2007,sassenhagen2019`.
# These do not require a strict *a priori* hypothesis about the time window and channel(s) of interest, and are therefore especially useful for exploratory analyses.
#
# A tutorial for how to compute CBPTs in the MNE-Python and hu-neuro-pipeline packages will be included as a bonus chapter in the future.
#
# %% [markdown]
# ## Exercises
#
# 1. Re-run the analysis pipeline for 10 participants from a different ERP CORE experiment (valid experiment names are `'N170'`, `'MMN'`, `'N2pc'`, `'N400'`, `'P3'`, or `'ERN'`), average the single trial amplitudes of the component of interest for each participant and condition, and fit a $t$-test to these averaged amplitudes.
# 2. Repeat the same analysis but with the single trial data and a linear mixed model.
#    Try to see if you can include random intercepts and slopes for participants, and random intercepts or slopes for items (if appropriate).
#    Simplify the random effects in case the model fails to converge, and try to interpret the fixed effect estimates and $p$-values.
#
# %%  tags=["skip-execution"]
# Your code goes here
...

# %% [markdown]
# ## Further reading
#
# * Paper *Group-level EEG-processing pipeline for flexible single trial-based analyses including linear mixed models* {cite:p}`fromer2018`
# * Chapter *Principles of statistical analyses: Old and new tools* {cite:p}`kretzschmar2023`
# * [Blog post](https://benediktehinger.de/blog/science/lmm-type-1-error-for-1condition1subject/) by Benedikt Ehinger on why to (almost) always include random slopes in LMMs
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
#
