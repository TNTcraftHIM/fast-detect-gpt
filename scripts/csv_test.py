# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
import pandas as pd
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from sklearn.metrics import roc_auc_score, roc_curve


# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

def process_p_values_and_labels_odd(answer_labels, results_list):
    # 计算 AUROC
    all_auroc = roc_auc_score(answer_labels, results_list)
    fpr, tpr, thresholds = roc_curve(answer_labels, results_list)
    all_avg_fpr = np.mean(fpr)
    all_avg_tpr = np.mean(tpr)
    all_std_fpr = np.std(fpr)
    all_std_tpr = np.std(tpr)

    # 打印结果
    print(
        "ALL_AUROC: {:.4f}; ALL_AVG_FPR: {:.4f}; ALL_STD_FPR: {:.4f}; ALL_AVG_TPR: {:.4f}; ALL_STD_TPR: {:.4f}".format(
            all_auroc, all_avg_fpr, all_std_fpr, all_avg_tpr, all_std_tpr))

    return all_auroc, all_avg_fpr, all_std_fpr, all_avg_tpr, all_std_tpr


# run interactive local inference
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('Reading csv from data/test_data.csv...')
    csv = pd.read_csv('data/test_data.csv', encoding='utf-8')
    crits = []
    true_labels = [1 if 'MGT' in label else 0 for label in csv['label']]
    print('Calculating the AUROC...')
    for text in csv['text']:
        # evaluate text
        tokenized = scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated text
        crits.append(crit)
    # calculate the AUROC
    process_p_values_and_labels_odd(true_labels, crits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)



