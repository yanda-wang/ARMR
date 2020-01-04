# GRMR

# Overview
This repository consists of the implement of the GRMR model. GRMR is an end-to-end model for medication recommendation, and it is formed by a encoder-decoder framework for recommendation and a GAN framework for distribution regularization. The encoder-decoder framework takes diagnoses and procedures of patients as inputs, after generating patients' representations, the framework carries out multi-hop reading on a key-value memory network, whose keys are representations of historical admissions while the values are corresponding medications, to capture historical information of patients for medication recommendation. To avoid DDI, a GAN model is built to shape the distribution of patients' represnetations so that the it follows the distribution of representations related to patients with low DDI rate.

# Requirement
Pytorch 1.1

Python 3.7

# Data
Experiments are carried out based on [MIMIC-III](https://mimic.physionet.org)， which is a real-world Electoric Healthcare Records (EHRs) dataset, and it collects clinical information related to over 45,000 patients. The diagnoses and procedures are used as inputs of GRMR, and the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

# Code
