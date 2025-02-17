# EEG-Based Dementia Subtype Classification

This repository contains the data and model information necessary to reproduce the results presented in the manuscript **"Translational Approach for Dementia Subtype Classification Using EEG Connectome Profile-Based Convolutional Neural Network"** published in _Scientific Reports_.  The original dataset before preprocessing can be gound at [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/). Here, we are providing pre-processed connectivity maps, model architecture definitions, and trained model weights to facilitate reproducibility.

## Contents

* **`connectivity_maps/`:** This directory contains the instance-wise connectivity maps used in the study, both with and without outlier rejection.  The files are organized by subject and condition (A, C, F).
* **`model_architecture/`:** This directory contains the PyTorch definition of the convolutional neural network (CNN) architecture used for dementia subtype classification.  The file `model.py` defines the model structure.
* **`model_checkpoints/`:** This directory contains the state dictionary of the best-performing CNN models for each classification task (multiclass, pairwise). The files are named according to the task they were trained for.

## Data Description

The `connectivity_maps/` directory contains the pre-processed connectivity matrices for each subject and condition. There are two types of data: `full/` with all data included and `rejected/` with outlier rejected. These matrices were derived from resting-state EEG recordings in the matrix format as demonstrated in the main manuscript.  The files are in .mat format.  The naming convention is as follows: 
[condition][subject_id]\_I20\_[instance_id]

Where:

* `condition`:  The dementia subtype (A, F, C according to AD, FTD, HC).
* `subject_id`:  A unique identifier for each subject.
* `instance_id`:  A unique identifier for each instance from a subject.

## Model Description

The `model_architecture/model.py` file defines the CNN architecture used in this study. The specific hyperparameters used for training are described in the manuscript's Methods section.

## Citation

* If you use the original data, please refer to doi:10.18112/openneuro.ds004504.v1.0.8
* If you use these preprocessed data or models in your research, please cite the original manuscript: [Insert my manuscript citation here]

## Contact

For any questions or inquiries, please contact the corresponding author as mentioned in the original manuscript.
