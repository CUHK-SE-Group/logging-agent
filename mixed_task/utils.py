import os
import pandas as pd
import random
import csv
import re
from collections import OrderedDict
import numpy as np
import pandas as pd

logging_levels = ['error', 'warn', 'info', 'debug', 'trace']






#============================================= task 1 =============================================#





def remove_random_logging_code(row):

    full_code = ""
    full_code = row['full_code']

    pattern = r'<line\d+>\s+(.+?)(?=(?:<line\d+>)|$)'  #分组 (?:<line\d+>)|$，用于匹配下一个 <line(\d+)> 或者文本结束
    logging_lines = re.findall(pattern, row['logging_code_labels'])
    
    #print("logging_lines:",logging_lines)
    num_logging_lines = len(logging_lines)
    #print("num:",num_logging_lines)
    num_to_remove = random.randint(0, num_logging_lines)
    #print("num_to_remove:",num_to_remove)
    label = 'No'

    if num_to_remove > 0:
        label = 'Yes'       
        lines_to_remove = random.sample(logging_lines, num_to_remove)
        #print("lines_to_remove:",lines_to_remove)
        
        for remove_code in lines_to_remove:
            #print("remove:",remove_code)
            full_code = full_code.replace(remove_code, "")

    return full_code,label



def task1_generate_dataset(df,new_file_path):
    # 对每一行应用函数来随机删除日志代码
    df["code"],df["label"] = zip(*df.apply(remove_random_logging_code, axis=1))
    df = df[['code','label']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)





#============================================= task 2 =============================================#


def extract_logging_line(row):
    logging_lines = re.findall(r"<line(\d+)>", row['logging_code_labels'])
    return [int(line) for line in logging_lines]


def extract_full_line(row):
    logging_lines = re.findall(r"<line(\d+)>", row['without_logging_code_index'])
    list1 = [int(line) for line in logging_lines]
    list1 = list(range(list1[-1]))
    return list1


def generate_line_index(df):
    # 提取logging_line的列表
    df['logging_line'] = df.apply(extract_logging_line, axis=1)
    #print(df['logging_line'])

    # 获取full_code_index的最后一个line数字
    df['all_line'] = df.apply(extract_full_line, axis=1)    

    return(df)



def generate_random_logging_line(row):
    line_index = random.choice(row['logging_line'])
    line_index =  " <line" + str(line_index) + ">"
    label = "Yes"
    return line_index, label


def generate_random_other_line(row):
    all_line_without_logging = [line_num for line_num in row['all_line'] if line_num not in row['logging_line']]
    line_index = random.choice(all_line_without_logging)
    line_index =  " <line" + str(line_index) + ">"
    label = "No"
    return line_index, label


def task2_generate_dataset(df,new_file):
    # 选择前50%的数据
    first_half_df = df.head(len(df)//2)
    first_half_df['line_index'], first_half_df['label'] = zip(*first_half_df.apply(generate_random_logging_line, axis=1))

    # 选择后50%的数据
    second_half_df = df.tail(len(df)//2)
    second_half_df['line_index'], second_half_df['label'] = zip(*second_half_df.apply(generate_random_other_line, axis=1))

    # 合并两部分的数据
    new_df = pd.concat([first_half_df, second_half_df])

    # 选择所需的列
    new_df = new_df[['without_logging_code_index','line_index', 'label']]

    # 保存新的DataFrame为tsv文件
    new_df = new_df.sample(frac=1, random_state=42)  # random_state 用于保证结果可复现，frac=1 表示抽取整个df
    new_df.to_csv(new_file, sep='\t', index=False)
    print(len(new_df))




#============================================= task 3 =============================================#


def get_logging_num(row):

    full_code = ""
    full_code = row['full_code']
    
    pattern = r'<line\d+>\s+(.+?)(?=(?:<line\d+>)|$)'  #分组 (?:<line\d+>)|$，用于匹配下一个 <line(\d+)> 或者文本结束
    logging_lines = re.findall(pattern, row['logging_code_labels'])

    label = len(logging_lines)

    return row['without_logging_code'],label



def task3_generate_dataset(df,new_file_path):
    df["code"],df["label"] = zip(*df.apply(get_logging_num, axis=1))
    df = df[['code','label']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)




#============================================= task 4 =============================================#


def get_logging_line(row):
    logging_lines = re.findall(r"<line\d+>", row['logging_code_labels'])
    #去重
    #unique_lines = list(set(logging_lines))
    unique_lines = list(OrderedDict.fromkeys(logging_lines))
    return unique_lines


def task4_generate_dataset(df,new_file_path):

    df["label"] = (df.apply(get_logging_line, axis=1))

    # 使用apply()和lambda函数将列表转化为字符串
    df['label'] = df['label'].apply(lambda x: ', '.join(x))
    new_df =  df[['without_logging_code_index','label']]

    print(new_df)
    new_df.to_csv(new_file_path, sep='\t', index=True)





# 处理test数据集
df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t')

# 将df平均分成4份
dfs = np.array_split(df, 4)

# 调用不同的任务处理每个数据集
for i, data in enumerate(dfs):
    output_file = f"task{i+1}_eval_small.tsv"
    if i == 0:
        task1_generate_dataset(data, output_file)
    elif i == 1:
        data = generate_line_index(data)
        task2_generate_dataset(data, output_file)
    elif i == 2:
        task3_generate_dataset(data, output_file)
    elif i == 3:
        task4_generate_dataset(data, output_file)