import json
import os
import numpy as np
import jsonlines
from PSelection.BlindMI import BlindMI
import tensorflow as tf
from functools import partial
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

"------ Draw the figures ------"
def read_all_features_4figs_GA(n, m, feature, mode):
    path1 = f"./data/{mode}/evol{n}.jsonl"
    path2 = f"./data/{mode}/evol{m}.jsonl"
    features1 = []
    features2 = [] 
    print(f"reading from {path1} and {path2}")
    with open(path1, 'r') as file:
        for line in file:
            data = json.loads(line)
            for key in data:
                if key.startswith(feature):
                    features1.append(data[key])

    with open(path2, 'r') as file:
        for line in file:
            data = json.loads(line)
            for key in data:
                if key.startswith(feature):
                    features2.append(data[key])
    
    return features1, features2

def draw_figs(n, m, mode, feature = "wppl"):

    features1, features2 = read_all_features_4figs_GA(n, m, feature, mode)
    print(f"round{n} feature: {features1}")
    print(f"round{m} feature: {features2}")
    features1 = np.array(features1)
    features2 = np.array(features2)

    save_path = f'./plots/{mode}/evol{n}to{m}/'
    os.makedirs(save_path, exist_ok=True)

    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 6))
    sns.histplot(features1, bins=30, kde=False, color='blue', label='feature 1', stat='density')
    sns.histplot(features2, bins=30, kde=False, color='red', label='feature 2', stat='density')
    plt.title('Histogram')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'histogram_{feature}.png'))
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.kdeplot(features1, color='blue', fill=True, label='feature 1')
    sns.kdeplot(features2, color='red', fill=True, label='feature 2')
    plt.title('Density Plot')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'density_plot_{feature}.png'))
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.boxplot(data=[features1, features2])
    plt.xticks([0, 1], ['features 1', 'features 2'])
    plt.title('Box Plot')
    plt.savefig(os.path.join(save_path, f'box_plot_{feature}.png'))
    plt.close()

    print(f"Plots saved to {save_path}")


"----##### Cal MMD between GAs #####----"
def read_jsonl_4calmmd(file_path):
    features = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'wppl' in data:
                features.append(data['wppl'])
    return features

def cal_all_mmd(num_files):
    weight = 1.0 
    for i in range(1, num_files):
        file_path1 = f'./data/evol{i}.jsonl'
        file_path2 = f'./data/evol{i+1}.jsonl'
        
        features1 = read_jsonl_4calmmd(file_path1)
        features2 = read_jsonl_4calmmd(file_path2)

        features1_tensor = tf.convert_to_tensor(features1, dtype=tf.float32)
        features2_tensor = tf.convert_to_tensor(features2, dtype=tf.float32)

        mmd_value = BlindMI.mmd_loss(features1_tensor, features2_tensor, weight)
        
        tf.print(f'MMD value between evol{i}.jsonl and evol{i+1}.jsonl:', mmd_value)

def cal_GA_mmd(i, mode):
    weight = 1.0 
    file_path1 = f'./data/{mode}/evol{i-1}.jsonl'
    file_path2 = f'./data/{mode}/evol{i}.jsonl'
        
    features1 = read_jsonl_4calmmd(file_path1)
    features2 = read_jsonl_4calmmd(file_path2)

    features1_tensor = tf.convert_to_tensor(features1, dtype=tf.float32)
    features2_tensor = tf.convert_to_tensor(features2, dtype=tf.float32)

    mmd_value = BlindMI.mmd_loss(features1_tensor, features2_tensor, weight).numpy().item()

    print(f'MMD value between evol{i-1}.jsonl and evol{i}.jsonl:', mmd_value)

    return mmd_value


class DataHandler:

    '''seeds.jsonl: '''
    def read_seeds(seeds_path):
        texts = []
        with open(seeds_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                text = data.get('txt', '')
                texts.append(text)
        print(f"seeds read from {seeds_path}")
        return texts

    '''evol.jsonl: {prompt:, response: , feature: , ...}'''
    def write_prompts(prompts, evol_path):
        with open(evol_path, 'w', encoding='utf-8') as file:
            for prompt in prompts:
                entry = {"prompt": prompt} 
                json.dump(entry, file) 
                file.write('\n')
        print(f"write 'prompts' to {evol_path}")

    def write_responses_features(generated_texts, probs_list, log_probs_list, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, (generated_text, probs, log_probs) in zip(lines, zip(generated_texts, probs_list, log_probs_list)):
                entry = json.loads(line.strip())
                entry["response"] = generated_text
                entry["probs"] = probs
                entry["log_probs"] = log_probs
                json.dump(entry, file, ensure_ascii=False)
                file.write('\n')
        print(f"got responses & features, write into {evol_path}")
    
    def write_multi_responses_features(results, n, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, result in zip(lines, results):
                entry = json.loads(line.strip())
                for i in range(n):
                    entry[f"response{i}"] = result[f'response{i+1}']
                    entry[f"probs{i}"] = result[f'probs{i+1}']
                    entry[f"log_probs{i}"] = result[f'log_probs{i+1}']
                json.dump(entry, file, ensure_ascii=False)
                file.write('\n')
        print(f"got responses & features, write into {evol_path}")

    def write_multi_responses_features2(results, n, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, result in zip(lines, results):
                print(f"Type of result: {type(result)}, result content: {result}")
                entry = json.loads(line.strip())
                for i in range(n):
                    entry[f"response{i}"] = result[f'response{i+1}']
                    entry[f"log_probs{i}"] = result[f'log_probs{i+1}']
                json.dump(entry, file, ensure_ascii=False)
                file.write('\n')
        print(f"got responses & features, write into {evol_path}")

    def add_value_to_jsonl(file_path, n, value, value_to_add):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if n < 0 or n >= len(lines):
            print(f"Index {n} is out of range for the JSONL file with {len(lines)} lines.")
            return

        with open(file_path, 'w', encoding='utf-8') as file:
            for index, line in enumerate(lines):
                entry = json.loads(line.strip())
                if index == n:
                    if value in entry:
                        if isinstance(entry[value], list):
                            entry[value].append(value_to_add)
                        else:
                            entry[value] = [entry[value], value_to_add]
                    else:
                        entry[value] = value_to_add
                json.dump(entry, file, ensure_ascii=False)
                file.write('\n')

    def read_prompts(path):
        texts = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                text = data.get('prompt', '')
                texts.append(text)
        return texts
    
    def read_log_probs(path):
        texts = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                text = data.get('log_probs', '')
                texts.append(text)
        return texts
    
    def write_wppls(wppls, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, wppl in zip(lines, wppls):
                entry = json.loads(line.strip())
                entry['wppl'] = wppl
                json.dump(entry, file)
                file.write('\n')
        print(f"write 'whole ppls' to file: {evol_path}")

    def write_mppls(mppls, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, mppl in zip(lines, mppls):
                entry = json.loads(line.strip())
                entry['mppl'] = mppl
                json.dump(entry, file)
                file.write('\n')
        print(f"write 'multi ppls' to file: {evol_path}")

    def write_lmppls(lmppls, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, lmppl in zip(lines, lmppls):
                entry = json.loads(line.strip())
                entry['lmppls'] = lmppl
                json.dump(entry, file)
                file.write('\n')
        print(f"write 'large multi ppls' to file: {evol_path}")

    def write_gppls(gppls, evol_path):
        with open(evol_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(evol_path, 'w', encoding='utf-8') as file:
            for line, gppl in zip(lines, gppls):
                entry = json.loads(line.strip())
                entry['gppl'] = gppl
                json.dump(entry, file)
                file.write('\n')
        print(f"write 'gram ppls' to file: {evol_path}")

    def save_sel_prompts(id_list, input_file, output_file):
        prompts = DataHandler.read_prompts(input_file)
        selected_prompts = [prompts[id] for id in id_list]
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f: 
                pass
        DataHandler.write_prompts(selected_prompts, output_file)

    def add_prompts_to_jsonl(prompts, jsonl_file):
        with jsonlines.open(jsonl_file, mode='a') as writer:
            for prompt in prompts:
                writer.write({'prompt': prompt})

    #--- Update write wppl to jsonl # 24.6.24 # ---#
    def add_wppl(results, path):
        with open(path, 'r') as f:
            lines = f.readlines()
    
        # Parse the JSONL file into a list of dicts
        existing_data = [json.loads(line) for line in lines]
    
        # Add PPL values to the corresponding lines
        for i, result in enumerate(results):
            for key, value in result.items():
                if 'wppl' in key:
                    existing_data[i][key] = value
    
        # Write the updated data back to the JSONL file
        with open(path, 'w') as f:
            for entry in existing_data:
                json.dump(entry, f)
                f.write('\n')


class multi_features:

    @staticmethod
    def cal_ppl(log_probs): # cal sequence ppl from log_probs
        total_log_prob = sum(log_probs)
        num_tokens = len(log_probs)
        perplexity = 2 ** (-total_log_prob / num_tokens)
        return perplexity

    @staticmethod
    def cal_whole_ppls(log_probs_list): #
        whole_ppls = []
        for log_probs in log_probs_list:
            perplexity = multi_features.cal_ppl(log_probs)
            whole_ppls.append(perplexity)
        return whole_ppls
    
    @staticmethod
    def cal_multippls(log_probs_list):
        multippls = []
        for log_probs in log_probs_list:
            text_length = len(log_probs)

            all_ppls = []
            for percent in [0.1, 0.2]:
                window_size = int(percent * text_length)
                for start in range(0, text_length - window_size + 1, 1):
                    end = start + window_size
                    sub_log_probs = log_probs[start:end]
                    ppl = multi_features.cal_ppl(sub_log_probs)
                    all_ppls.append(ppl)

            min_ppl = min(all_ppls)
            multippls.append(min_ppl)

        return multippls

    @staticmethod
    def cal_gramppls(log_probs_list):
        multippls = []

        for log_probs in log_probs_list:
            text_length = len(log_probs)
            all_ppls = []
            for window_size in [3, 5]:
                for start in range(0, text_length - window_size + 1, 1):
                    end = start + window_size
                    sub_log_probs = log_probs[start:end]
                    ppl = multi_features.cal_ppl(sub_log_probs)
                    all_ppls.append(ppl)

            min_ppl = min(all_ppls)
            multippls.append(min_ppl)

        return multippls
    
    @staticmethod
    def cal_large_multippls(log_probs_list):
        multippls = []
        for log_probs in log_probs_list:
            text_length = len(log_probs)

            all_ppls = []
            for percent in [0.5, 0.75, 0.9]:
                window_size = int(percent * text_length)
                for start in range(0, text_length - window_size + 1, 1):
                    end = start + window_size
                    sub_log_probs = log_probs[start:end]
                    ppl = multi_features.cal_ppl(sub_log_probs)
                    all_ppls.append(ppl)

            min_ppl = min(all_ppls)
            multippls.append(min_ppl)

        return multippls