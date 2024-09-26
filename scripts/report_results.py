# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import argparse
import json
import numpy as np


def save_lines(lines, file):
    with open(file, 'w') as fout:
        fout.write('\n'.join(lines))

def get_auroc(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['roc_auc']

def get_fpr_tpr(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['fpr'], res['metrics']['tpr']

def report_chatgpt_gpt4_results(args):
    datasets = {'xsum': 'XSum',
                'writing': 'Writing',
                'pubmed': 'PubMed'}
    source_models = {'gpt-3.5-turbo': 'ChatGPT',
                     'gpt-4': 'GPT-4'}
    score_models = { 't5-3b': 'T5-3B',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large',
                'Hello-SimpleAI_chatgpt-detector-roberta': 'ChatGPT-detector',
                'trained_trained_model':'ReMoDetect'}
    methods2 = {'likelihood': 'Likelihood', 'entropy': 'Entropy', 'logrank': 'LogRank'}
    methods3 = {'lrr': 'LRR', 'npr': 'NPR', 'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast'}

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    headers2 = ['Method'] + [datasets[dataset] for dataset in datasets] + ['Avg.'] \
               + [datasets[dataset] for dataset in datasets] + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods

    filters2 = {'likelihood': ['.gpt-neo-2.7B'],
               'entropy': ['.gpt-neo-2.7B'],
               'logrank': ['.gpt-neo-2.7B']}
    filters3 = {'lrr': ['.t5-3b_gpt-neo-2.7B'],
               'npr': ['.t5-3b_gpt-neo-2.7B'],
               'perturbation_100': ['.t5-3b_gpt-neo-2.7B'],
               'sampling_discrepancy_analytic': ['.gpt-j-6B_gpt-neo-2.7B']}
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp/results/")
    args = parser.parse_args()
    report_chatgpt_gpt4_results(args)

