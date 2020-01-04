import pickle
import numpy as np
import csv
import math
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split


class DataLoaderMedRec:
    """
    dataloader for training medication recommendation model
    """

    def __init__(self, patient_records_file_name, ddi_rate_threshold, data_mode='train', data_split=False):
        """
        :param patient_records_file_name: patient records file
        :param ddi_rate_threshold: only patient records with a ddi rate less than or equal to ddi_rate_threshold will be sampled
        :param data_mode: choose from train, test, and validation, sample data from corresponding dataset
        :param data_split: True or False, True=patient records are stored separately according to their ddi rate
        """
        self.patient_records_file_name = patient_records_file_name
        self.ddi_rate_threshold = ddi_rate_threshold
        self.data_mode = data_mode
        self.data_split = data_split
        self.patient_records = None
        self.patient_count = 0
        self.patient_count_split = {}
        self.read_index = None

        if self.data_split:
            patient_records = np.load(self.patient_records_file_name)[data_mode]
            self.patient_records = {}
            self.read_index = {}
            self.patient_count = 0
            for rate in np.arange(0, self.ddi_rate_threshold + 0.1, 0.1):
                self.patient_records[round(rate, 1)] = patient_records[round(rate, 1)]
                self.read_index[round(rate, 1)] = 0
                self.patient_count_split[round(rate, 1)] = len(self.patient_records[round(rate, 1)])
                self.patient_count = self.patient_count + self.patient_count_split[round(rate, 1)]
        else:
            self.patient_records = np.load(self.patient_records_file_name)[data_mode][ddi_rate_threshold]
            self.patient_count = len(self.patient_records)
            self.read_index = 0

    # shullef the patient records
    def shuffle(self, ddi_rate=-0.1):
        """
        :param ddi_rate: for data_split=False, ddi_rate should be smaller than 0 and all data will be shuffled
                         for data_split=Ture, patient records related to ddi_rate will be shuffled
        """
        if ddi_rate < 0:
            np.random.shuffle(self.patient_records)
            self.read_index = 0
        else:
            np.random.shuffle(self.patient_records[ddi_rate])
            self.read_index[ddi_rate] = 0

    def load_patient_record(self, ddi_rate=-0.1):
        """
        :param ddi_rate: for data_split=False, ddi_rate should be smaller than 0 and a patient record will be read and returned
                         for data_split=False, a patient record related to ddi_rate will be read and returned
        :return: a patient record
        """
        if ddi_rate < 0:
            return self.load_patient_record_split_false()
        else:
            return self.load_patient_record_split_true(ddi_rate)

    def load_patient_record_split_false(self):
        """
        :return: a patient record
        """
        if self.read_index >= self.patient_count:
            # print('index out of range, shuffle patient records')
            self.shuffle()
        picked_patient = self.patient_records[self.read_index]  # pick a patient
        medications = [admission[0] for admission in picked_patient]
        diagnoses = [admission[1] for admission in picked_patient]
        procedures = [admission[2] for admission in picked_patient]
        self.read_index += 1
        return medications, diagnoses, procedures

    def load_patient_record_split_true(self, ddi_rate):
        """
        :param ddi_rate: load a patient record whose ddi rate equals ddi_rate
        :return: a patient record whose ddi rate equals ddi_rate
        """
        if self.read_index[ddi_rate] >= self.patient_count_split[ddi_rate]:
            # print('index out of range, shuffle patient records')
            self.shuffle(ddi_rate)
        picked_patient = self.patient_records[ddi_rate][self.read_index[ddi_rate]]
        medications = [admission[0] for admission in picked_patient]
        diagnoses = [admission[1] for admission in picked_patient]
        procedures = [admission[2] for admission in picked_patient]
        self.read_index[ddi_rate] += 1
        return medications, diagnoses, procedures


class DataLoaderGANSingleDistribution:
    """
    dataloader for training GRMR
    """

    def __init__(self, fitted_distribution_file_name, patient_records_file_name, ddi_rate_threshold, data_mode='train'):
        """
        :param fitted_distribution_file_name: file that stores real data for GAN model
        :param patient_records_file_name: patient records file
        :param ddi_rate_threshold: patient records with ddi rate smaller than ddi_rate will be used for training
        :param data_mode: choose from train, test, and validation, sample data from corresponding dataset
        """
        self.fitted_distribution_file_name = fitted_distribution_file_name
        self.patient_records_file_name = patient_records_file_name
        self.ddi_rate_threshold = ddi_rate_threshold
        self.data_mode = data_mode

        self.distribution_data = np.load(
            self.fitted_distribution_file_name)['real_data']  # np.array,dim=(#data point, data dimension)
        self.dataloader_medrec = DataLoaderMedRec(self.patient_records_file_name, self.ddi_rate_threshold,
                                                  self.data_mode, True)

        self.patient_count = self.dataloader_medrec.patient_count
        self.patient_count_split = self.dataloader_medrec.patient_count_split  # dict, key: ddi rate, value: #patients with the corresponding ddi rate
        self.distribution_n = len(self.distribution_data)
        self.read_index = 0

    def shullfe_all(self):
        """
        shuffle real data and patient records
        """
        for ddi_rate in np.arange(0, self.ddi_rate_threshold + 0.1, 0.1):
            self.dataloader_medrec.shuffle(round(ddi_rate, 1))
        self.shuffle_distribution()

    def shuffle_distribution(self):
        """
        shuffle real data
        """
        np.random.shuffle(self.distribution_data)
        self.read_index = 0

    def load_data(self, ddi_rate):
        """
        read a patient record and a single instance from real data
        :param ddi_rate: a patient record whose ddi rate equals ddi_rate will be read
        :return: medication, diagnoses, procedures related to a patient, and a single instance from real data
        """
        if self.read_index >= self.distribution_n:
            self.shuffle_distribution()
        medication, diagnoses, procedures = self.dataloader_medrec.load_patient_record(ddi_rate)
        sampled_distribution = self.distribution_data[self.read_index]
        self.read_index += 1
        return medication, diagnoses, procedures, sampled_distribution


class Concept2Id(object):
    """
    class for storing medical concepts and corresponding id
    """

    def __init__(self):
        self.concept2id = {}
        self.id2concept = {}

    def add_concepts(self, concepts):
        """
        given a sequence of medical concepts, obtain their ids and store the mapping
        :param concepts: a medical concept
        """
        for item in concepts:
            if item not in self.concept2id.keys():
                # self.id2concept[len(self.concept2id)] = item
                self.concept2id[item] = len(self.concept2id)
                self.id2concept[self.concept2id.get(item)] = item

    def get_concept_count(self):
        return len(self.concept2id)


def map_concepts2id(patient_info_file_path, concept2id_output_file_path):
    """
    read patient records and build the Concept2Id object
    :param patient_info_file_path: patient information, each line for a admission, file format:subject_id,hadm_id,admittime,prescriptions_set(ATC3 code seperated by ;), diagnoses_set, procedures_set
    :param concept2id_output_file_path: output file for results,
    """
    concept2id_prescriptions = Concept2Id()
    concept2id_diagnoses = Concept2Id()
    concept2id_procedures = Concept2Id()

    patient_info_file = open(patient_info_file_path, 'r')
    for line in patient_info_file:
        line = line.rstrip('\n')
        patient_info = line.split(',')
        prescriptions = patient_info[3].split(';')
        diagnoses = patient_info[4].split(';')
        procedures = patient_info[5].split(';')
        concept2id_prescriptions.add_concepts(prescriptions)
        concept2id_diagnoses.add_concepts(diagnoses)
        concept2id_procedures.add_concepts(procedures)
    patient_info_file.close()
    dump_objects = {'concept2id_prescriptions': concept2id_prescriptions, 'concept2id_diagnoses': concept2id_diagnoses,
                    'concept2id_procedures': concept2id_procedures}
    pickle.dump(dump_objects, open(concept2id_output_file_path, 'wb'))


def get_ddi_information(drug_ddi_file, top_N, ddi_info_output_file_pre):
    """
    :param drug_ddi_file: ddi type source file
    :param top_N: the number of ddi types that will be considered
    :param ddi_info_output_file_pre: output file
    :return:
    """
    drug_ddi_df = pd.read_csv(drug_ddi_file)
    ddi_most_pd = drug_ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)

    ddi_most_pd = ddi_most_pd.iloc[-top_N:, :]
    fliter_ddi_df = drug_ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)
    ddi_info_output_file = ddi_info_output_file_pre + "_top" + str(top_N)
    pickle.dump({'ddi_info': ddi_df}, open(ddi_info_output_file, 'wb'))


def construct_ddi_matrix(ddi_file_path, stitch2atc_file_path, concept2id_file_path, ddi_matrix_output_file):
    """
    construct the ddi matrix:two stitch_ids from ddi_file_path -> get the atcs of these two stitch_ids from stitch2atc_file_path
                             -> get the id of these atcs as coordinates in the matrix
    :param ddi_file_path: ddi information from the TWOSIDES dataset
    :param stitch2atc_file_path: map between stitch_id and atc
    :param concept2id_file_path: map between atc to id
    :param ddi_matrix_output_file: output file of the ddi matrix
    """
    concept2id_prescriptions = np.load(concept2id_file_path).get('concept2id_prescriptions')

    stitch2atc_dict = {}
    with open(stitch2atc_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            stitch_id = line[0]
            atc_set = line[1:]
            stitch2atc_dict[stitch_id] = atc_set

    prescriptions_size = concept2id_prescriptions.get_concept_count()
    ddi_matrix = np.zeros((prescriptions_size, prescriptions_size))
    ddi_info = np.load(ddi_file_path)['ddi_info']

    for index, row in ddi_info.iterrows():
        stitch_id1 = row['STITCH 1']
        stitch_id2 = row['STITCH 2']

        if stitch_id1 in stitch2atc_dict.keys() and stitch_id2 in stitch2atc_dict.keys():
            for atc_i in stitch2atc_dict[stitch_id1]:
                for atc_j in stitch2atc_dict[stitch_id2]:
                    atc_i = atc_i[:4]
                    atc_j = atc_j[:4]
                    if atc_i in concept2id_prescriptions.concept2id.keys() and atc_j in concept2id_prescriptions.concept2id.keys() and atc_i != atc_j:
                        ddi_matrix[
                            concept2id_prescriptions.concept2id.get(atc_i), concept2id_prescriptions.concept2id.get(
                                atc_j)] = 1
                        ddi_matrix[
                            concept2id_prescriptions.concept2id.get(atc_j), concept2id_prescriptions.concept2id.get(
                                atc_i)] = 1

    ddi_matrix_object = {'ddi_matrix': ddi_matrix}
    ddi_matrix_output_file = ddi_matrix_output_file + '_' + ddi_file_path.split('_')[-2] + '_' + \
                             ddi_file_path.split('_')[-1]
    pickle.dump(ddi_matrix_object, open(ddi_matrix_output_file, 'wb'))


def construct_cooccurance_matrix(patient_records_file, concept2id_file, matrix_output_file, patient_ddi_rate):
    """
    construct co-occurrence matrix
    :param patient_records_file: patient records file
    :param concept2id_file: the file for mapping between medical concepts and their id
    :param matrix_output_file: output file for co-occurrence matrix
    :param patient_ddi_rate: patient records with ddi rate is smaller than patient_ddi_rate will be used
    :return:
    """
    concetp2id_medications = np.load(concept2id_file).get('concept2id_prescriptions')
    medication_count = concetp2id_medications.get_concept_count()
    matrix = np.zeros((medication_count, medication_count))
    patient_records = np.load(patient_records_file)['train'][1.0]
    count = 0
    for patient in patient_records:
        for admission in patient:
            if admission[-1][0] <= patient_ddi_rate:
                medications = admission[0]
                for med_i, med_j in combinations(medications, 2):
                    count += 1
                    matrix[med_i][med_j] = 1
                    matrix[med_j][med_i] = 1
    pickle.dump(matrix, open(matrix_output_file + '_' + str(patient_ddi_rate), 'wb'))

    unique, counts = np.unique(matrix, return_counts=True)
    print(dict(zip(unique, counts)))


def construct_patient_records(patient_info_file_path, concept2id_mapping_file_path, ddi_matrix_file_path,
                              patient_records_output_file_path):
    """
    transform patient records that consist of medical concepts to records that consist of corresponding ids
    :param patient_info_file_path: the file that contains patient information
                                   file format: subject_id,hadm_id,admittime,prescriptions_set(ATC3 code seperated by ;), diagnoses_set, procedures_set
    :param concept2id_mapping_file_path: map between concepts and ids
    :param ddi_matrix_file_path: ddi matrix, for computing ddi rate for each admission
    :param patient_records_output_file_path: output file for the result
    """
    ddi_matrix = np.load(ddi_matrix_file_path)['ddi_matrix']

    def get_ddi_rate(medications):
        med_pair_count = 0.0
        ddi_count = 0.0
        ddi_rate = 0
        for med_i, med_j in combinations(medications, 2):
            med_pair_count += 1
            if ddi_matrix[med_i][med_j] == 1:
                ddi_count += 1
        if med_pair_count != 0:
            ddi_rate = ddi_count / med_pair_count
        return ddi_rate

    concept2id_object = np.load(concept2id_mapping_file_path)
    concept2id_prescriptions = concept2id_object.get('concept2id_prescriptions')
    concept2id_diagnoses = concept2id_object.get('concept2id_diagnoses')
    concept2id_procedures = concept2id_object.get('concept2id_procedures')

    tmp_ddi_rate = []

    patient_records = []
    patient = []
    last_subject_id = ''
    patient_info_file = open(patient_info_file_path)
    for line in patient_info_file:
        admission = []
        line = line.rstrip('\n').split(',')
        current_subject_id = line[0]
        prescriptions = line[3].split(';')
        diagnoses = line[4].split(';')
        procedures = line[5].split(';')
        admission.append([concept2id_prescriptions.concept2id.get(item) for item in prescriptions])
        admission.append([concept2id_diagnoses.concept2id.get(item) for item in diagnoses])
        admission.append([concept2id_procedures.concept2id.get(item) for item in procedures])
        ddi_rate = get_ddi_rate(admission[0])
        # admission.append([get_ddi_rate(admission[0])])
        admission.append([ddi_rate])
        tmp_ddi_rate.append(round(ddi_rate, 1))
        if current_subject_id == last_subject_id:
            patient.append(admission)
        else:
            # if len(patient) != 0 and filter_patient_records(patient):
            if len(patient) != 0:
                patient_records.append(patient)
            patient = []
            patient.append(admission)
        last_subject_id = current_subject_id
    patient_records.append(patient)
    patient_info_file.close()
    dump_object = {'patient_records': patient_records}
    pickle.dump(dump_object, open(patient_records_output_file_path, 'wb'))


def data_sampling(patient_records_file_path, sampling_data_seperate_output_file_path,
                  sampling_data_accumulate_output_file_path):
    """
    split data into traning set, test set, and validation set
    :param patient_records_file_path: patient records file, built by function construct_patient_records
    :param sampling_data_seperate_output_file_path: output file for the results, contains a dic with keys 'trian','test', and 'validation'
                                                    for each ddi rate, patients with ddi rate belongs (ddi rate-0.1,ddi rate] are stored
    :param sampling_data_accumulate_output_file_path: utput file for the results, contains a dic with keys 'trian','test', and 'validation'
                                                    for each ddi rate, patients with ddi rate smaller than ddi rate are stored
    """
    ddi_rate_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    patient_records_split_by_ddi_rate = {}
    for ddi_rate in ddi_rate_bins:
        patient_records_split_by_ddi_rate[ddi_rate] = []
    patient_records = np.load(patient_records_file_path)['patient_records']
    for patient in patient_records:
        for idx, admission in enumerate(patient):
            ddi_rate = admission[3][0]
            current_patient_record = patient[:idx + 1]
            patient_records_split_by_ddi_rate[math.ceil(ddi_rate * 10.0) / 10].append(current_patient_record)

    train, test, validation = {}, {}, {}
    for ddi_rate, patients in patient_records_split_by_ddi_rate.items():
        train_patients, test_patients = train_test_split(patients, test_size=0.1)
        train_patients, validation_patients = train_test_split(train_patients, test_size=0.1)
        train[ddi_rate], test[ddi_rate], validation[ddi_rate] = train_patients, test_patients, validation_patients
    pickle.dump({'train': train, 'test': test, 'validation': validation},
                open(sampling_data_seperate_output_file_path, 'wb'))

    print('patient records information stored seperately by ddi rate')
    print('training dataset:')
    for key, value in train.items():
        print(key, len(value), end=';')
    print()
    print('test dataset')
    for key, value in test.items():
        print(key, len(value), end=';')
    print()
    print('validation dataset')
    for key, value in validation.items():
        print(key, len(value), end=';')
    print()

    for ddi_rate in ddi_rate_bins[1:]:
        train[ddi_rate] = train[ddi_rate] + train[round(ddi_rate - 0.1, 1)]
        test[ddi_rate] = test[ddi_rate] + test[round(ddi_rate - 0.1, 1)]
        validation[ddi_rate] = validation[ddi_rate] + validation[round(ddi_rate - 0.1, 1)]
    pickle.dump({'train': train, 'test': test, 'validation': validation},
                open(sampling_data_accumulate_output_file_path, 'wb'))

    print('patient records information stored accumulately by ddi rate')
    print('training dataset:')
    for key, value in train.items():
        print(key, len(value), end=';')
    print()
    print('test dataset')
    for key, value in test.items():
        print(key, len(value), end=';')
    print()
    print('validation dataset')
    for key, value in validation.items():
        print(key, len(value), end=';')
    print()
