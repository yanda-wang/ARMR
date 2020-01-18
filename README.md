# ARMR

# Overview
This repository consists of the implement of the ARMR model. ARMR is an end-to-end model for medication recommendation, and it is formed by a encoder-decoder framework for recommendation and a GAN framework for distribution regularization. The encoder-decoder framework takes diagnoses and procedures of patients as inputs, after generating patients' representations, the framework carries out multi-hop reading on a key-value memory network, whose keys are representations of historical admissions while the values are corresponding medications, to capture historical information of patients for medication recommendation. To avoid DDI, a GAN model is built to shape the distribution of patients' represnetations so that the it follows the distribution of representations related to patients with low DDI rate.

# Requirement
Pytorch 1.1

Python 3.7

# Data
Experiments are carried out based on [MIMIC-III](https://mimic.physionet.org)， which is a real-world Electoric Healthcare Records (EHRs) dataset, and it collects clinical information related to over 45,000 patients. The diagnoses and procedures are used as inputs of ARMR, and the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

Patient records are firstly selected out from the raw data into a file, and each line contains the information for a single admission in the form of \[subject_id, hadm_id, admittime, medications, diagnoses, procedures\].You could find an example below.

\[17, 194023, 2134-12-27 07:15:00, A12A; C01C; B05C; N07A; A12C; A07A; N01A; C02D; M01A; A10A, 2724; 45829; 7455; V1259, 3571; 3961; 8872\]

After constructing the vocabulary for medical concepts, i.e., assigning a identical integer to each medical concepts, medications, diagnoses, and procedures are represented by corresponding integers, and patient records are transformed into a np.array, while each element in the array represents information for a single patient in the form \[adm_1, adm_2, ..., adm_n\]. For each adm_i, the form is \[\[med_1, med_2, ..., med_m\],\[diag_1, diag_2, ..., dig_d\],\[pro_1, pro_2, ..., pro_p\],\[ddi rate\]]. For instance, we could find an example below, and there are two admissions in the example. For the first admission, the medications are 3, 4, and 5, the diagnoses are 6 and 7, the procedures are 8 and 9, and the ddi rate is 0.3.

\[\[\[3, 4, 5\], \[6, 7\], \[8, 9\], \[0.3\]\], \[\[10 ,11\], \[12, 13, 14\], \[15, 16\], \[0.2\]\]\]

# Code

Auxiliary.py: data loader and data preprocessing.

Networks.py: encoder(generator), decoder, and discriminator.

Optimization.py: basic modules that warp encoder and decoder for hyper-parameter tuning

MedRecOptimization.py: hyper-parameter tuning for MedRec, i.e. medication recommendation without GAN regularization.

DiscriminatorOptimization.py: hyper-parameter tuning for the discriminator.

Training.py: model training.

Evaluation.py: model evaluation.

Parameters.py: global parameters for model.








