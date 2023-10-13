import re
import os
import argparse
import pandas as pd
import matplotlib.pylab as plt
from nltk.translate.bleu_score import sentence_bleu

# This script calculates the accuracy of one task at a time, change it here before running
evaluate_task = 'task4'

def evaluate():
    parser = argparse.ArgumentParser()


    parser.add_argument('--in_file', default="result_file/predictions1-4_test_result.tsv", type=str)
    parser.add_argument('--out_file', default="result_file/evaluate_task4.tsv", type=str)


    args = parser.parse_args()

    results_file_path = args.in_file
    structured_results_path = args.out_file

    evaluation_greedy(results_file_path, structured_results_path)



def evaluation_greedy(input_file_path, output_file_path):
    true_count = 0
    label_list = []
    pred_list = []

    # uses pandas to read the file, not json
    df_raw = pd.read_csv(input_file_path, sep='\t')
    data_num = 0
    for i, row in df_raw.iterrows(): 
        if row['task'] != evaluate_task: # skip the non-logging tasks
            continue

        data_num += 1

        '''Get ground truth'''
        label = row["label"]
        label_list.append(label)


        '''Get predictions'''
        predict = row['predict']
        pred_list.append(predict)

        # If there are single quotes before and after predict, remove them, and comment them out if not
        # predict = predict[1:-1]
        # print(predict)

        if label == predict:
            true_count += 1

    print(evaluate_task + "\'s accuracy:", round(true_count/data_num, 3))

    df = pd.DataFrame({"gth": label_list, "prediction": pred_list})
    df["prompt"] = df_raw['prompt']
    df.to_csv(output_file_path, sep='\t')


if __name__ == '__main__':

    evaluate()



