import re
import argparse
import pandas as pd
import matplotlib.pylab as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="instruct_logging/codellama7b", type=str, required=True)
    parser.add_argument('--in_file', default="predictions/raw.tsv", type=str, required=True)
    parser.add_argument('--out_file', default="predictions/structured.tsv", type=str, required=True)
    args = parser.parse_args()

    results_file_path = args.in_file
    structured_results_path = args.out_file

    evaluation(results_file_path, structured_results_path)
    cal_metrics(structured_results_path)

def get_structured_data(gth):
    data = gth.split(">")
    pos_gth, log = data[0], ">".join(data[1:])
    pos_gth = pos_gth + ">"
    pos_gth, log = pos_gth.strip(), log.strip()
    res = re.search(r'[.](off)?(fatal)?(error)?(warn)?(info)?(debug)?(trace)?(all)?[(]', log)
    level_gth = log[res.span()[0] + 1: res.span()[1] - 1].strip()
    pattern = re.compile('%s\((.+)\)' % level_gth)
    result = pattern.findall(log)

    if result == []:
        message_gth = ""
    else:
        message_gth = result[0]
    return pos_gth.lower(), level_gth.lower(), message_gth.lower()


def get_predict(pred):
    datas = pred.split("<line")
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
                # print(log)
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


def evaluation(input_file_path, output_file_path):
    pos_count, level_count, message_count, bleu_score = 0, 0, 0, 0
    pos_level_count, pos_message_count = 0, 0
    pos_gth_list, level_gth_list, message_gth_list = [], [], []
    pos_pred_list, level_pred_list, message_pred_list = [], [], []
    count = 0

    # uses pandas to read the file, not json
    df_raw = pd.read_csv(input_file_path, sep='\t')
    data_num = 0
    print("data num:", data_num)
    for i, row in df_raw.iterrows(): 

        if row['task'] != 'task9': # skip the non-logging tasks
            continue

        data_num += 1
        # pattern = r'(?=<line\d+>)'
        flag = False

        '''Get ground truth'''
        label = row["label"]
        pos_gth, level_gth, message_gth = get_structured_data(label)

        pos_gth_list.append(pos_gth)
        level_gth_list.append(level_gth)
        message_gth_list.append(message_gth)


        '''Get predictions'''
        predict = row['predict']
        pos_preds, level_preds, message_preds = get_predict(predict)

        pos_level_mess_dict = {}
        for pos, level, message in zip(pos_preds, level_preds, message_preds):
            if pos not in pos_level_mess_dict \
                    or (pos_level_mess_dict[pos][0] == "" and level != ""):
                pos_level_mess_dict[pos] = (level, message)

        if pos_gth in pos_level_mess_dict:
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
                # print("key: ", key)
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

    print("pos: ", pos_count)
    print("levels: ", level_count)
    print("message: ", message_count)

    print("pos acc: ", round(pos_count/data_num, 3))
    print("levels acc: ", round(level_count/data_num, 3))
    print("message acc: ", round(message_count/data_num, 3))
    print("levels acc under pos right: ", round(pos_level_count/pos_count, 3))
    print("message acc under pos right: ", round(pos_message_count/pos_count, 3))
    print("bleu_score: ", round(bleu_score/data_num, 3))

    df = pd.DataFrame({"pos_gth": pos_gth_list, "pos_pred": pos_pred_list})
    df["pos_res"] = df["pos_gth"] == df["pos_pred"]
    df["level_gth"] = level_gth_list
    df["level_pred"] = level_pred_list
    df["level_res"] = df["level_gth"] == df["level_pred"]
    df["message_gth"] = message_gth_list
    df["message_pred"] = message_pred_list
    df["message_res"] = df["message_gth"] == df["message_pred"]
    df["code"] = df_raw['code']
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
    plt.savefig(file_name)


def line2idx(x):
    if pd.notna(x):
        x = x.split("<line")[-1][:-1]
    else:
        return -100

    return int(x)


def cal_metrics(data_dir):
    df = pd.read_csv(data_dir)
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

    plot(position_distance, "results/position_distance")
    plot(level_distance, "results/level_distance")


if __name__ == '__main__':

    evaluate()