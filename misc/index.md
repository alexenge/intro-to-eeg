# Introduction to EEG analysis

[![Python](https://img.shields.io/badge/🐍-Python-blue)](<https://www.python.org/>)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<https://alexenge.github.io/intro-to-eeg/>)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/alexenge/intro-to-eeg/deploy.yml)](https://github.com/alexenge/intro-to-eeg/actions/workflows/deploy.yml)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This course provides a very brief introduction into analyzing electroencephalography (EEG) data.

It uses the [Python](https://www.python.org/) programming language and the [MNE-Python](https://mne.tools/stable/index.html) package for EEG analysis.
Python and MNE-Python are open source software, which means that they are free to use and co-created and constantly improved by a large community of users.
While prior programming experience is useful to follow the course materials, no specific knowledge of the Python language or MNE-Python is required.

## Content

```{tableofcontents}
```

## How to

|                            |                                                                                                                                                                                                                                                                                                |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🌍 **Online** (recommended) | At the top of each chapter, hit the "Interactive" button (🚀) and "Colab" to open an interactive notebook in your browser. This allows you to run and/or modify the code without having to install anything. You can also save your changes to your own Google Drive.                           |
| 💻 **Offline**              | [Download the course materials from GitHub](https://github.com/alexenge/intro-to-eeg/archive/refs/heads/main.zip), unzip them, and open them in a local Python IDE. This is a more advanced option, as it requires you to setup your own Python environment and install the required packages. |
| 💤 **Static**               | Browse through the non-interactive version of the notebooks as listed above and in the left sidebar.                                                                                                                                                                                           |

## Feedback

If you encounter any typos, factual errors, or problems when running the code, or if you would like to contribute additional content, please [open an issue on GitHub](https://github.com/alexenge/intro-to-eeg/issues).
Thanks!

## Learn more

Luckily, this course isn't the only place to learn about EEG data analysis!
Here are some further resources that you may want to look into instead of or in addition to this course, if you want to learn more:

* Aaron J. Newman's free online book [*Neural Data Science in Python*](https://neuraldatascience.io) has multiple chapters on Python and EEG analysis, very similar to the material covered in this course.
* The [MNE-Python documentation](https://mne.tools/stable/index.html) has many great tutorials and examples for EEG and MEG analysis, including more advanced topics such as source localization and machine learning.
* The [hu-neuro-pipeline documentation](https://hu-neuro-pipeline.readthedocs.io/en/stable/index.html) and [this slide deck](https://github.com/alexenge/hu-neuro-pipeline-workshop/blob/main/slides.pdf) contain more information and examples for our lab's EEG processing pipeline (introduced in the {ref}`pipeline` section of the course).
* Steve Luck's book [*An Introduction to the Event-Related Potential Technique*](https://mitpress.mit.edu/books/introduction-event-related-potential-technique-second-edition) is the "bible" for EEG acquisition and analysis.
  You may be able to find a free PDF version online.
* Steve Luck also has a free online book on [*Applied Event-Related Potential Data Analysis*](https://socialsci.libretexts.org/Bookshelves/Psychology/Biological_Psychology/Applied_Event-Related_Potential_Data_Analysis_(Luck)) in MATLAB.
* The [EEGLAB](https://sccn.ucsd.edu/eeglab/index.php) MATLAB toolbox is a popular alternative to MNE-Python.
* [eegUtils](https://craddm.github.io/eegUtils/) and [eeguana](https://bruno.nicenboim.me/eeguana/) are two (rather experimental) R packages for EEG analysis.
* The paper [*Good scientific practice in EEG and MEG research: Progress and perspectives*](https://doi.org/10.1016/j.neuroimage.2022.119056) by Niso, Krol, et al. (2022) has many great suggestions for reproducible EEG research.
