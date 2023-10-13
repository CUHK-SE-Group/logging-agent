import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
def line2idx(x):
    if pd.notna(x):
        x = x.split("<line")[-1][:-1]
    else:
        return -100

    return int(x)

def plot_paradigm(data1, data2, data3, file_name, xrange):
    plt.figure()
    d1 = sorted(data1.items(), key=lambda d: d[0])
    d2 = sorted(data2.items(), key=lambda d: d[0])
    d3 = sorted(data3.items(), key=lambda d: d[0])
    x1 = np.array([each[0] for each in d1])
    y1 = np.array([each[1] for each in d1])
    x2 = np.array([each[0] for each in d2])
    y2 = np.array([each[1] for each in d2])
    x3 = np.array([each[0] for each in d3])
    y3 = np.array([each[1] for each in d3])
    print(x1[-1])
    print(x2[-1])
    print(x3[-1])
    bar_width = 0.3
    plt.bar((x1 - bar_width)[:xrange], y1[:xrange], bar_width, label='In-Context Learning', color="#bdd7ee")
    plt.bar((x2)[:xrange], y2[:xrange], bar_width, label='Fine Tuning', color="#6baed6")
    plt.bar((x3 + bar_width)[:xrange], y3[:xrange], bar_width, label='Instruction Tuning', color="#2171b5")
    plt.title(file_name, fontsize=18)
    plt.xlabel("distance", fontsize=14, labelpad=0)
    plt.ylabel("nums", fontsize=14, labelpad=0)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig("figures/" +file_name + ".pdf")

def plot_mode(data1, data2, file_name, xrange):
    plt.figure()
    d1 = sorted(data1.items(), key=lambda d: d[0])
    d2 = sorted(data2.items(), key=lambda d: d[0])
    x1 = np.array([each[0] for each in d1])
    y1 = np.array([each[1] for each in d1])
    x2 = np.array([each[0] for each in d2])
    y2 = np.array([each[1] for each in d2])
    print(x1[-1])
    print(x2[-1])
    bar_width = 0.4
    plt.bar((x1 - bar_width/2)[:xrange], y1[:xrange], bar_width, label='InsLog-E', color="#2171b5")
    plt.bar((x2 + bar_width/2)[:xrange], y2[:xrange], bar_width, label='InsLog-S', color="#bdd7ee")
    plt.title(file_name, fontsize=18)
    plt.xlabel("distance", fontsize=14, labelpad=0)
    plt.ylabel("nums", fontsize=14, labelpad=0)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig("figures/" +file_name + ".pdf")

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

    return position_distance, level_distance


p1, l1 = cal_metrics("result_file/results5_base_icl.tsv")
p2, l2 = cal_metrics("result_file/results5_part.tsv")
p3, l3 = cal_metrics("result_file/results5.tsv")
p4, l4 = cal_metrics("result_file/results5_2step.tsv")



if not os.path.exists("figures"):
    os.mkdir("figures")
plot_paradigm(p1, p2, p3, "position_distance_paradigm", 21)
plt.show()
plot_paradigm(l1, l2, l3, "level_distance_paradigm", 7)
plt.show()
plot_mode(p3, p4, "position_distance_mode", 21)
plt.show()
plot_mode(l3, l4, "level_distance_mode", 7)
plt.show()