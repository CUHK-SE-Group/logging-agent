import re
import os
import argparse
import pandas as pd
import matplotlib.pylab as plt
from nltk.translate.bleu_score import sentence_bleu

logging_task = 'task5'

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default="result_file/predictions5_2step.tsv", type=str)
    parser.add_argument('--out_file', default="result_file/results5_2step.tsv", type=str)
    args = parser.parse_args()

    results_file_path = args.in_file
    structured_results_path = args.out_file

    # evaluation_sampling(results_file_path, structured_results_path)
    evaluation_greedy(results_file_path, structured_results_path)
    # cal_metrics(structured_results_path)

def get_logging_greedy(sample):
    data = sample.split(">")
    pos, log = data[0], ">".join(data[1:])
    pos = pos + ">"
    pos, log = pos.strip(), log.strip()
    res = re.search(r'[.](off)?(fatal)?(error)?(warn)?(info)?(debug)?(trace)?(all)?[(]', log)
    if res is not None:
        level = log[res.span()[0] + 1: res.span()[1] - 1].strip()
        pattern = re.compile('%s\((.+)\)' % level)
        result = pattern.findall(log)
        if result == []:
            message = ""
        else:
            message = result[0]
    else:
        level, message = "", ""
    return pos.lower(), level.lower(), message.lower()


def get_logging_sampling(samples):
    datas = samples.split("<line")
    datas = [each.strip() for each in datas if each.strip() != ""]

    k = 0
    pos_preds, level_preds, message_preds = [], [], []
    while k < len(datas):
        data = datas[k]
        data = "<line" + data
        data = data.split(">")
        if len(data) < 2:
            pos_preds.append("")
            level_preds.append("")
            message_preds.append("")
            k += 1
            continue
        pos_pred, log = data[0], ">".join(data[1:])
        pos_pred, log = pos_pred.strip(), log.strip()
        pos_pred = pos_pred + ">"
        res = re.search(r'[.](off)?(fatal)?(error)?(warn)?(info)?(debug)?(trace)?(all)?[(]', log)
        if res is not None:
            level_pred = log[res.span()[0] + 1: res.span()[1] - 1].strip()
            pattern = re.compile('%s\((.+)\)' % level_pred)
            result = pattern.findall(log)
            if result == []:
                message_pred = ""
            else:
                message_pred = result[0]
        else:
            level_pred, message_pred = "", ""
        pos_preds.append(pos_pred.lower())
        level_preds.append(level_pred.lower())
        message_preds.append(message_pred.lower())
        k += 1
    return pos_preds, level_preds, message_preds


def evaluation_sampling(input_file_path, output_file_path):
    pos_count, level_count, message_count, bleu_score = 0, 0, 0, 0
    pos_level_count, pos_message_count = 0, 0
    pos_gth_list, level_gth_list, message_gth_list = [], [], []
    pos_pred_list, level_pred_list, message_pred_list = [], [], []

    # uses pandas to read the file, not json
    df_raw = pd.read_csv(input_file_path, sep='\t')
    data_num = 0
    for i, row in df_raw.iterrows(): 

        if row['task'] != logging_task: # skip the non-logging tasks
            continue

        data_num += 1
        flag = False

        '''Get ground truth'''
        label = row["label"]
        pos_gth, level_gth, message_gth = get_logging_greedy(label)

        pos_gth_list.append(pos_gth)
        level_gth_list.append(level_gth)
        message_gth_list.append(message_gth)


        '''Get predictions'''
        predict = row['predict']
        pos_preds, level_preds, message_preds = get_logging_sampling(predict)

        pos_level_mess_dict = {} # use position as the key, to find the nearest correct prediction.
        for pos, level, message in zip(pos_preds, level_preds, message_preds):
            if pos not in pos_level_mess_dict \
                    or (pos_level_mess_dict[pos][0] == "" and level != ""):
                pos_level_mess_dict[pos] = (level, message)

        if pos_gth in pos_level_mess_dict: # only if the position is true, the level and message prediction are valid.
            pos_pred, level_pred, message_pred = pos_gth, pos_level_mess_dict[pos_gth][0], \
                                                    pos_level_mess_dict[pos_gth][1]
            # set default level
            if level_pred == "":
                level_pred = "info"

            if pos_gth == pos_pred:
                pos_count += 1
            if level_gth == level_pred:
                level_count += 1
                pos_level_count += 1
            if message_gth == message_pred:
                message_count += 1
                pos_message_count += 1

            if message_gth == message_pred:
                bleu_score += 1
            else:
                bleu_score += sentence_bleu([message_gth.split()], message_pred.split())

            flag = True

        if flag is False:
            # if all pred positions does not hit the target，pick the nearest predicted <line#> to the target one
            count += 1

            pos_tag, min_dis = "", 100000000
            pos_tar_idx = int(pos_gth.split("<line")[-1][:-1])
            for key in pos_level_mess_dict:
                if key == "":
                    continue
                cur = key.split("<line")[-1][:-1]  # <line1> -> 1> -> 1
                if cur.isdigit():
                    if abs(int(cur) - pos_tar_idx) < min_dis:
                        pos_tag = key

            pos_pred, level_pred, message_pred = pos_tag, pos_level_mess_dict[pos_tag][0], pos_level_mess_dict[pos_tag][1]
            # set default level
            if level_pred == "":
                level_pred = "info"

            if pos_gth == pos_pred:
                pos_count += 1
            if level_gth == level_pred:
                level_count += 1
            if message_gth == message_pred:
                message_count += 1
            if message_gth == message_pred:
                bleu_score += 1
            else:
                bleu_score += sentence_bleu([message_gth.split()], message_pred.split())

        if pos_pred == "":
            print("pos_level_mess_dict:", pos_level_mess_dict, row["samples"])

        pos_pred_list.append(pos_pred)
        level_pred_list.append(level_pred)
        message_pred_list.append(message_pred)

    print("position accuracy (PA): ", round(pos_count/data_num, 3))
    print("level accuracy (LA): ", round(level_count/data_num, 3))
    print("message accuracy (MA): ", round(message_count/data_num, 3))
    print("conditional level accuracy (CLA): ", round(pos_level_count/pos_count, 3))
    print("conditional message accuracy (CMA): ", round(pos_message_count/pos_count, 3))
    print("BLEU-DM: ", round(bleu_score/data_num, 3))

    df = pd.DataFrame({"pos_gth": pos_gth_list, "pos_pred": pos_pred_list})
    df["pos_res"] = df["pos_gth"] == df["pos_pred"]
    df["level_gth"] = level_gth_list
    df["level_pred"] = level_pred_list
    df["level_res"] = df["level_gth"] == df["level_pred"]
    df["message_gth"] = message_gth_list
    df["message_pred"] = message_pred_list
    df["message_res"] = df["message_gth"] == df["message_pred"]
    df["prompt"] = df_raw['prompt']
    df.to_csv(output_file_path, sep='\t')


def evaluation_greedy(input_file_path, output_file_path):
    pos_count, level_count, message_count, bleu_score = 0, 0, 0, 0
    pos_level_count, pos_message_count = 0, 0
    pos_gth_list, level_gth_list, message_gth_list = [], [], []
    pos_pred_list, level_pred_list, message_pred_list = [], [], []

    # uses pandas to read the file, not json
    df_raw = pd.read_csv(input_file_path, sep='\t')
    data_num = 0
    for i, row in df_raw.iterrows(): 

        if row['task'] != logging_task: # skip the non-logging tasks
            continue

        data_num += 1

        '''Get ground truth'''
        label = row["label"]
        pos_gth, level_gth, message_gth = get_logging_greedy(label)

        pos_gth_list.append(pos_gth)
        level_gth_list.append(level_gth)
        message_gth_list.append(message_gth)


        '''Get predictions'''
        predict = row['predict']
        pos_pred, level_pred, message_pred = get_logging_greedy(predict)

        pos_pred_list.append(pos_pred)
        level_pred_list.append(level_pred)
        message_pred_list.append(message_pred)

        if pos_gth == pos_pred:
            pos_count += 1
        if level_gth == level_pred:
            level_count += 1
        if pos_gth == pos_pred and level_gth == level_pred:
            pos_level_count += 1
        if message_gth == message_pred:
            message_count += 1
        if pos_gth == pos_pred and message_gth == message_pred:
            pos_message_count += 1

        if message_gth == message_pred:
            bleu_score += 1
        else:
            bleu_score += sentence_bleu([message_gth.split()], message_pred.split())

    print("position accuracy (PA): ", round(pos_count/data_num, 3))
    print("level accuracy (LA): ", round(level_count/data_num, 3))
    print("message accuracy (MA): ", round(message_count/data_num, 3))
    print("conditional level accuracy (CLA): ", round(pos_level_count/pos_count, 3))
    print("conditional message accuracy (CMA): ", round(pos_message_count/pos_count, 3))
    print("BLEU-DM: ", round(bleu_score/data_num, 3))

    df = pd.DataFrame({"pos_gth": pos_gth_list, "pos_pred": pos_pred_list})
    df["pos_res"] = df["pos_gth"] == df["pos_pred"]
    df["level_gth"] = level_gth_list
    df["level_pred"] = level_pred_list
    df["level_res"] = df["level_gth"] == df["level_pred"]
    df["message_gth"] = message_gth_list
    df["message_pred"] = message_pred_list
    df["message_res"] = df["message_gth"] == df["message_pred"]
    # df["prompt"] = df_raw['prompt']
    df.to_csv(output_file_path, sep='\t')


def plot(distance, file_name):
    plt.figure()
    distance = sorted(distance.items(), key=lambda d: d[0])
    x = [each[0] for each in distance]
    height = [each[1] for each in distance]
    plt.bar(x=x, height=height)
    plt.title(file_name)
    plt.xlabel("distance")
    plt.ylabel("nums")
    plt.savefig("figures/" +file_name + ".pdf")


def line2idx(x):
    if pd.notna(x):
        x = x.split("<line")[-1][:-1]
    else:
        return -100

    return int(x)


def cal_metrics(data_dir):
    df = pd.read_csv(data_dir, sep='\t')
    pos_true = len(df[df["pos_res"] == True])
    pos_level_true = len(df[df["pos_res"] == True][df["level_res"] == True])
    pos_message_true = len(df[df["pos_res"] == True][df["message_res"] == True])
    pos_level_message_true = len(df[df["pos_res"] == True][df["level_res"] == True][df["message_res"] == True])

    print("position √ and level √ :", pos_level_true)
    print("the ratio is: ", pos_level_true / pos_true)
    print("#"*50)
    print("position √ and message √ :", pos_message_true)
    print("the ratio is: ", pos_message_true / pos_true)
    print("#"*50)
    print("position √ and level √ and message √ :", pos_level_message_true)
    print("the ratio is: ", pos_level_message_true / pos_true)

    position_distance = {}
    df["pos_gth_idx"] = df["pos_gth"].apply(line2idx)
    df["pos_pred_idx"] = df["pos_pred"].apply(line2idx)

    level_distance = {}
    level2idx = {"trace": 0, "debug": 1, "info": 2, "warn": 3, "error": 4, "fatal": 5, "": 20, "all": 0}
    df["level_gth_idx"] = df["level_gth"].apply(lambda x: level2idx[x])
    df["level_pred"].fillna("", inplace=True)
    df["level_pred_idx"] = df["level_pred"].apply(lambda x: level2idx[x])

    for i, row in df.iterrows():
            dis = abs(row["pos_gth_idx"] - row["pos_pred_idx"])
            if dis in position_distance:
                position_distance[dis] += 1
            else:
                position_distance[dis] = 1

    for i, row in df.iterrows():
            dis = min(abs(row["level_gth_idx"] - row["level_pred_idx"]), 6)  # if pred level in null， set distance as 6
            if dis in level_distance:
                level_distance[dis] += 1
            else:
                level_distance[dis] = 1

    if not os.path.exists("figures"):
        os.mkdir("figures")
    plot(position_distance, "position_distance")
    plt.show()
    plot(level_distance, "level_distance")
    plt.show()



if __name__ == '__main__':

    evaluate()