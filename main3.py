import json
import random
import os
import sys
import time
import datetime
import pprint
import traceback
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import string
import heapq
import random

import numpy as np
from PSelection.BlindMI import BlindMI
from DMutation.crossover import WordCrossover
from DMutation.mutation import TextMutator
from PSelection.cal_sentropy import *
import utils
from utils import DataHandler, multi_features, cal_GA_mmd


def mutation_crossover(prompts):
    
    n = len(prompts)
    half_n = n / 2
    rounded_half_n = round(half_n)
    if rounded_half_n % 2 == 0:
        even_half = rounded_half_n
    else:
        if half_n > rounded_half_n:
            even_half = rounded_half_n + 1
        else:
            even_half = rounded_half_n - 1
    print(f"Prompts of crossover number will be {even_half}")
    n_mu = n - even_half
    print(f"Prompts of mutation number will be {n_mu}")

    #crossover#
    corpus_file = "./data/corpus.txt"
    children = WordCrossover.word_crossover_all(prompts, even_half)
    #mutation#
    mutator = TextMutator(corpus_file)
    random_items = random.sample(prompts, n_mu)
    texts = mutator.load_data(random_items)
    mutated_texts = mutator.mutate_text_list(texts)

    return children + mutated_texts

def expand_prompts(existing_prompts, existing_weights, target_size, expand_algorithm):

    current_size = len(existing_prompts)
    if current_size >= target_size:
        return existing_prompts[:target_size]  
    
    expanded_prompts = existing_prompts[:]
    
    while len(expanded_prompts) < target_size:
        selected_prompts = random.choices(
            existing_prompts,
            weights=existing_weights,
            k=min(target_size - len(expanded_prompts), current_size)
        )

        selected_prompts = list(set(selected_prompts))

        new_prompts = expand_algorithm(selected_prompts)

        expanded_prompts.extend(new_prompts)
        expanded_prompts = list(set(expanded_prompts)) 

    return expanded_prompts[:target_size]

def save_state(round_count, mode):
    state = {
        "round_count": round_count
    }
    with open(f"./data/{mode}/state.json", 'w') as f:
        json.dump(state, f)

def load_state(mode):
    state_path = f"./data/{mode}/state.json"
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
            return state["round_count"]
    else:
        return 0

def _main_(mode, model):
    seeds = DataHandler.read_seeds("./data/seeds.jsonl")
    if not os.path.exists(f"./data/{mode}/"):
        os.mkdir(f"./data/{mode}/")
    DataHandler.write_prompts(seeds, f"./data/{mode}/evol1.jsonl")
    
    round_count = load_state(mode)
    
    while True:
        save_state(round_count, mode)
        round_count += 1

        evol_path = f"./data/{mode}/evol{round_count}.jsonl"
        next_path = f"./data/{mode}/evol{round_count+1}.jsonl"
        print(f"---- ### Round {round_count} ### ----")
            
        ''' Query'''
        if round_count == 1: resp_n = 4
        else: resp_n = 2
        prompts = DataHandler.read_prompts(evol_path) 
        results = model.batch_query(prompts, 1, resp_n) 
        DataHandler.write_multi_responses_features2(results, resp_n, evol_path)
        DataHandler.add_wppl(results, evol_path)
        print(f"wppl has write into jsonl ")

        ''' Feature Calculation and Figure Drawing '''
        if round_count > 1:
            print(f"Draw figures for the round {round_count-1} & {round_count}")
            utils.draw_figs(round_count-1, round_count, mode, "wppl")

        ''' STOP or NOT?'''
        if round_count > 1:
            mmd_ga = cal_GA_mmd(round_count, mode)
            if mmd_ga > λ or round_count > 20: break

        ''' SE & BlindMI Selection '''
        cal_sentropy(evol_path) 
        torch.cuda.empty_cache()

        # Can choose whether use SE or not
        if round_count == 1:
            selected_ids0, _, _ = BlindMI.select_new_24_7(evol_path, "wppl", 0)
            selected_ids1, _, _ = BlindMI.select_new_24_7(evol_path, "wppl", 1)
            selected_ids2, _, _ = BlindMI.select_new_24_7(evol_path, "wppl", 2)
            selected_ids3, _, _ = BlindMI.select_new_24_7(evol_path, "wppl", 3)
            selected_ids = selected_ids0 +selected_ids1 +selected_ids2 +selected_ids3
            sids, sweights = BlindMI.sortid_calweights(selected_ids)
            print(f"select {len(sids)} prompts in round {round_count}")
            if not os.path.exists(next_path):
                os.system(r"touch {}".format(next_path))      
            DataHandler.save_sel_prompts(sids, evol_path, next_path)

        if round_count != 1:
            selected_ids0, _, _ = BlindMI.select_new_24_7(evol_path, "wppl", 0)
            selected_ids1, _, _ = BlindMI.select_new_24_7(evol_path, "wppl", 1)
            selected_ids = selected_ids0 +selected_ids1
            sids, sweights = BlindMI.sortid_calweights(selected_ids)
            print(f"select {len(sids)} prompts in round {round_count}")
            if not os.path.exists(next_path):
                os.system(r"touch {}".format(next_path))
            DataHandler.save_sel_prompts(sids, evol_path, next_path) 

        ''' Mutation & Crossover '''
        next_prompts = DataHandler.read_prompts(next_path)
        extended_prompts = expand_prompts(next_prompts, sweights, 1024, mutation_crossover)
        target_length = 1024
        corpus_file = "./data/corpus.txt"
        mutator = TextMutator(corpus_file)
        while True:
            total_length = len(extended_prompts) + len(next_prompts)

            if total_length == target_length:
                DataHandler.add_prompts_to_jsonl(extended_prompts, next_path)
                print(f"add new prompts DONE!")
                break

            elif total_length < target_length:
                mmm = target_length - total_length
                if len(extended_prompts) >= mmm:
                    random_items = random.sample(extended_prompts, mmm)
                else:
                    random_items = extended_prompts
                    missing_count = mmm - len(extended_prompts)

                    texts = mutator.load_data(next_prompts[:missing_count])
                    mutated_texts = mutator.mutate_text_list(texts)
                    random_items += mutated_texts[:missing_count] 

                extended_prompts += random_items
                print(f"SELF-Warning: total length less than {target_length}, sampling from extended_prompts and mutating...")

            elif total_length > target_length:

                excess_length = total_length - target_length

                extended_prompts = random.sample(extended_prompts, len(extended_prompts) - excess_length)
                print(f"SELF-Warning: total length exceeds {target_length}, adjusting extended_prompts by random sampling...")

        print(f"Final result length: {len(extended_prompts) + len(next_prompts)}")

    print(f"ALL GAs have done {round_count} rounds!")
    print("="*40)

    return round_count

if __name__ == "__main__":

    λ = 0.5
    round_count = _main_(mode, model)

