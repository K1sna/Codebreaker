import json
from tqdm import tqdm
import tensorflow as tf
from scipy.stats import mode
import numpy as np
from functools import partial


class BlindMI:
    
    @staticmethod
    def threshold_Divide(mix, lo=0.5, hi=1):
        if mix.size == 0:
            print("Warning: Input array 'mix' is empty. Returning empty prediction.")
            return np.array([]) 
    
        max_val, min_val, mean_val, median_val = np.max(mix), np.min(mix), np.mean(mix), np.median(mix)
        print(f"max_val = {max_val}, min_val = {min_val}, mean_val = {mean_val}, median_val = {median_val}")
    
        threshold = median_val * 2 * hi
        threshold1 = min_val * (1 + lo)
    
        m_pred = np.where(mix < threshold, 1, 0) & np.where(mix > threshold1, 1, 0)
        np.set_printoptions(threshold=np.inf)
        print(f"pred_ori: {m_pred}")
        return m_pred

    @staticmethod
    def compute_pairwise_distances(x, y):
        if not len(x.get_shape()) == len(y.get_shape()): # == 2
            raise ValueError("Both inputs should be matrices.")
        norm = lambda x: tf.reduce_sum(tf.square(x), 1)

        return tf.transpose(norm(tf.expand_dims(x, 1) - tf.transpose(y)))
    
    @staticmethod
    def gaussian_kernel_matrix(x, y, sigmas):
        beta = 1.0 / (2.0 * (tf.expand_dims(sigmas, 1)))
        beta = tf.cast(beta, dtype=tf.float32)  

        dist = BlindMI.compute_pairwise_distances(x, y)
        dist = tf.cast(dist, dtype=tf.float32)  

        s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
        return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

    
    @staticmethod
    def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
        with tf.name_scope("MaximumMeanDiscrepancy"):
            # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
            cost = tf.reduce_mean(kernel(x, x))
            cost += tf.reduce_mean(kernel(y, y))
            cost -= 2 * tf.reduce_mean(kernel(x, y))

            # We do not allow the loss to become negative.
            cost = tf.where(cost > 0, cost, 0, name="value")
        return cost
        
    @staticmethod
    def mmd_loss(source_samples, target_samples, weight, scope=None):
        sigmas = [
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            100,
            1e3,
            1e4,
            1e5,
            1e6,
        ]
        gaussian_kernel = partial(BlindMI.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

        loss_value = BlindMI.maximum_mean_discrepancy(
            source_samples, target_samples, kernel=gaussian_kernel
        )
        loss_value = tf.maximum(1e-4, loss_value) * weight

        return loss_value

    @staticmethod
    def blind(features, lo, hi, ids, to_0=0, to_1=1):
        if features.size == 0:
            print("Warning: 'features' array is empty. Skipping processing and returning empty results.")
            return [], [], [] 

        pred = BlindMI.threshold_Divide(features, lo, hi)
        print(f"start of prediction = {pred}")

        if pred.size == 0:
            print("Warning: Prediction is empty. Returning empty results.")
            return [], [], []

        data = (
            tf.data.Dataset.from_tensor_slices((features, pred, ids))
            .shuffle(buffer_size=features.shape[0])
            .batch(4000)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        pred_list, id_list = [], []
        last_non_empty_pred = None
        last_non_empty_id_list = None
        last_non_empty_selected_ids = None
        last_non_empty_s_features = None
        last_non_empty_us_features = None

        for (features_batch, pred_batch, id_batch) in data:
            pred_batch = pred_batch.numpy()
            Flag = True
            while Flag:
                dis_ori = BlindMI.mmd_loss(
                    features_batch[pred_batch.astype(bool) == False],
                    features_batch[pred_batch.astype(bool)],
                    weight=1,
                )
                print(f"mmd: {dis_ori}")
                Flag = False

                if to_1:
                    for index, item in tqdm(enumerate(features_batch)):
                        if pred_batch[index] == 0:
                            pred_batch[index] = 1
                            mix_1 = features_batch[pred_batch.astype(bool)]
                            mix_2 = features_batch[pred_batch.astype(bool) == False]
                            dis_new = BlindMI.mmd_loss(mix_2, mix_1, weight=1)
                            if dis_new < dis_ori:
                                pred_batch[index] = 0
                            else:
                                Flag = True
                                dis_ori = tf.identity(dis_new)
                if to_0 or (to_0 == 0 and to_1 == 0):
                    for index, item in tqdm(enumerate(features_batch)):
                        if pred_batch[index] == 1:
                            pred_batch[index] = 0
                            mix_1 = features_batch[pred_batch.astype(bool)]
                            mix_2 = features_batch[pred_batch.astype(bool) == False]
                            dis_new = BlindMI.mmd_loss(mix_2, mix_1, weight=1)
                            if dis_new < dis_ori:
                                pred_batch[index] = 1
                            else:
                                Flag = True
                                dis_ori = tf.identity(dis_new)

            print("Loop finished")
            pred_list.append(pred_batch)
            id_list.append(id_batch)

            pred_list_flat = np.concatenate(pred_list)
            if np.any(pred_list_flat == 1):
                last_non_empty_pred = pred_list.copy()
                last_non_empty_id_list = id_list.copy()
                last_non_empty_selected_ids = [id_batch[i] for i, pred in enumerate(pred_batch) if pred == 1]
                last_non_empty_s_features = [features[i].tolist() for i, pred in enumerate(pred_list_flat) if pred == 1]
                last_non_empty_us_features = [features[i].tolist() for i, pred in enumerate(pred_list_flat) if pred == 0]

        if last_non_empty_selected_ids is None:
            print("No selected IDs found, returning last non-empty state.")
            return last_non_empty_selected_ids, last_non_empty_s_features, last_non_empty_us_features

        pred_list = np.concatenate(pred_list)
        id_list = np.concatenate(id_list)
        selected_ids = [id_list[i] for i, pred in enumerate(pred_list) if pred == 1]
        selected_features = [features[i].tolist() for i, pred in enumerate(pred_list) if pred == 1]
        unselected_features = [features[i].tolist() for i, pred in enumerate(pred_list) if pred == 0]

        return selected_ids, selected_features, unselected_features



    @staticmethod
    def select(algo, wppls, mppls, lmppls, gppls):
        sl = []
        ori_ids = [i for i in range(len(wppls))] # ori_id: id in jsonl file

        # 0: whole sentence's perplexity #
        if algo == 0: features = np.array(wppls)
        # 1: multi-ppls(0.1-0.2) #
        if algo == 1: features = np.array(mppls)
        # 2: large multi-ppls(0.5-0.75-0.9) #
        if algo == 2: features = np.array(lmppls)
        # 3: 3-5 gram ppls #
        if algo == 3: features = np.array(gppls)
        # 4: log_probs #
        # if algo == 4: features = np.array(log_probs)

        arg = {
        "ids": ori_ids,
        "features": features,
        "lo": 0.25,
        "hi": 0.75,
        "to_0": 0,
        "to_1": 1,
        }

        #print(f"ori_ids = {ori_ids}")
        #print(f"features:{features}")
        selected_id = BlindMI.blind(**arg)

        return selected_id
    
    '''NEW UPDATE in BMI as below 24.6'''

    @staticmethod
    def read_feature4BMI(file_path, feature):
        wppl_values = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue 
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        wppl_row = [value for key, value in entry.items() if feature in key]
                        wppl_values.append(wppl_row)
                    else:
                        print(f"Skipping non-dict entry: {entry}")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                    continue 
    
        return wppl_values 
    
    @staticmethod
    def flatten_2d_array(array_2d):
        flattened_list = []
        original_positions = []
    
        for row_index, row in enumerate(array_2d):
            flattened_list.extend(row)
            original_positions.extend([row_index] * len(row))
    
        return flattened_list, original_positions
    
    @staticmethod
    def select_new(file, feature):
        ori_features = BlindMI.read_feature4BMI(file, feature)
        sl = []
        features, ori_ids = BlindMI.flatten_2d_array(ori_features)
        features = np.array(features)

        arg = {
        "ids": ori_ids,
        "features": features,
        "lo": 0.25,
        "hi": 0.75,
        "to_0": 0,
        "to_1": 1,
        }

        print(f"ori_ids = {ori_ids}")
        print(f"features:{features}")
        selected_id, s_features, us_features = BlindMI.blind(**arg)

        return selected_id, s_features, us_features
    
    @staticmethod
    def sortid_calweights(index_list):
        index_count = {}
        for index in index_list:
            if index in index_count:
                index_count[index] += 1
            else:
                index_count[index] = 1
        sorted_indices = sorted(index_count.keys())

        weights = [index_count[index] for index in sorted_indices]

        return sorted_indices, weights
    
    '''NEW UPDATE in BMI as below 24.7.1'''
    @staticmethod
    def read_feature4BMI_24_7(file_path, feature, index):
        wppl_values = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue 
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        wppl_row = [value for key, value in entry.items() if f"{feature}{index}" in key]
                        wppl_values.append(wppl_row)
                    else:
                        print(f"Skipping non-dict entry: {entry}")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                    continue 
    
        return wppl_values 
    
    @staticmethod
    def select_new_24_7(file, feature, index):
        ori_features = BlindMI.read_feature4BMI_24_7(file, feature, index)
        sl = []
        features, ori_ids = BlindMI.flatten_2d_array(ori_features) 
        features = np.array(features)

        arg = {
        "ids": ori_ids,
        "features": features,
        "lo": 0.15,
        "hi": 0.8,
        "to_0": 1,
        "to_1": 0,
        }

        print(f"ori_ids = {ori_ids}")
        print(f"features:{features}")
        selected_id, s_features, us_features = BlindMI.blind(**arg)

        return selected_id, s_features, us_features
