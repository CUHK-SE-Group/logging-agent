import os
import pandas as pd
import random
import csv
import re
from collections import OrderedDict

logging_levels = ['error', 'warn', 'info', 'debug', 'trace', 'fatal']


#============================================= source files preprocessing =============================================#
def drop_duplicates_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    print("len: ", len(df))
    df.drop_duplicates(subset=['full_code', 'logging_code_labels'], inplace=True)
    print("len: ", len(df))
    df.to_csv(file_path, sep='\t', index=False)



def delete_dirty_data(df,output_file):

    df = df[~df.apply(lambda row: row.astype(str).str.contains('STATS_LOG').any(), axis=1)]
    df = df[~df.apply(lambda row: row.astype(str).str.contains('FAKE_LOGGER').any(), axis=1)]

    df = df.applymap(lambda cell: re.sub(r'(|s)_logger\.', 'logger.', cell, flags=re.IGNORECASE) if isinstance(cell, str) else cell)
    df = df.applymap(lambda cell: re.sub(r'(|s)_log\.', 'log.', cell, flags=re.IGNORECASE) if isinstance(cell, str) else cell)

    pattern = r'\b(?:log|logger|LOG|LOGGER)\.(?:' + '|'.join(logging_levels) + r')\b'
    mask = df['logging_code_labels'].str.contains(pattern, case=False, flags=re.IGNORECASE)
    
    for index, row in df[~mask].iterrows():
        print(f"Row {index}: {row['logging_code_labels']}")
        df = df.drop(index)

    df.to_csv(output_file, sep='\t', index=False)
    new_df = df['logging_code_labels']
    new_df.to_csv('source_train_label.tsv', sep='\t', index=False)

"""
output_file = 'source_train.tsv'
df = pd.read_csv(output_file, sep='\t') 
delete_dirty_data(df,output_file)
"""


def filter_non_english_ascii_data(inputfile, outputfile):

    non_english_ascii_pattern = re.compile(r'[^\x00-\x7F]+')

    with open(inputfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    filtered_lines = []

    count = 0

    for line in lines:
        if not non_english_ascii_pattern.search(line):
            filtered_lines.append(line)
        else:
            count += 1
    print('filtered num:',count)

    with open(outputfile, 'w', encoding='utf-8') as file:
        file.writelines(filtered_lines)

# filter_non_english_ascii_data("source_eval.tsv","source_eval.tsv")
# filter_non_english_ascii_data("source_test.tsv","source_test.tsv")
# filter_non_english_ascii_data("source_train.tsv","source_train.tsv")



#============================================= task 1 =============================================#

def get_logging_line(row):
    logging_lines = re.findall(r"<line\d+>", row['logging_code_labels'])
    unique_lines = list(OrderedDict.fromkeys(logging_lines))
    return unique_lines


def task1_generate_dataset(df,new_file_path):

    df["label"] = (df.apply(get_logging_line, axis=1))

    df['label'] = df['label'].apply(lambda x: ','.join(x))
    df['task'] = 'task1'
    df["lineID"] = ""
    new_df =  df[['without_logging_code_index','label','task', 'lineID']].rename(columns={'without_logging_code_index': 'code'})

    print(new_df)
    new_df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task1_generate_dataset(df,"task1_test.tsv")

# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task1_generate_dataset(df,"task1_train.tsv")




#============================================= task 2 =============================================#
# task 2: predict logging level (masked) → predicted logging level

def mask_log_level(row, mask="UNKNOWN"):
    code = row['full_code_index']
    level_pattern = '|'.join(logging_levels)
    pattern = r'((?:logger|log)\.)(' + level_pattern + r')(\()'
    new_code = re.sub(pattern, r'\1' + mask + r'\3', code, flags=re.IGNORECASE, count=0)
    # Extract and filter log levels
    labels = re.findall(pattern, code, flags=re.IGNORECASE)
    if not labels:
        print("Logging statement not found.")
        print(row['full_code'])
        print(row['logging_code_labels'])
    else:
        labels = [label[1] for label in labels if label[1].lower() in (level.lower() for level in logging_levels)]
        labels_str = ','.join(labels)

    return new_code, labels_str


def task2_generate_dataset(df,new_file_path):
    df["masked_code"],df["label"] = zip(*df.apply(mask_log_level, axis=1))
    # df = df[['masked_code', 'full_code', 'label']]
    df['task'] = 'task2'
    df["lineID"] = (df.apply(get_logging_line, axis=1))
    df['lineID'] = df['lineID'].apply(lambda x: ','.join(x))
    df = df[['masked_code', 'lineID', 'label', 'task']].rename(columns={'masked_code': 'code'})
    print(df)
    df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task2_generate_dataset(df,"task2_test.tsv")

# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task2_generate_dataset(df,"task2_train.tsv")



#============================================= task 3 =============================================#
# task 3: predict logging message (masked) → predicted logging message

def extract_log_mes(row):

    pattern = r'\b(?:LOG|LOGGER)\.(?:debug|info|warn|error|trace|fatal)\((.*?)\);'
    matches = re.findall(pattern, row['logging_code_labels'], flags = re.MULTILINE | re.IGNORECASE)

    log_messages = [match for match in matches]
    log_messages_str = '<DIV>'.join(log_messages)
    return log_messages_str


def mask_log_mes(row):

    logging_code_labels = row['logging_code_labels']
    full_code = row['full_code_index']
    log_pattern = r'<line\d+>\s+(.*?);' 
    log_matches = re.findall(log_pattern, logging_code_labels)

    log_list = []

    for log_match in log_matches:
        log_list.append(log_match)
    
    for log in log_list:
        log_name_pattern = r'(\w+)\.' 
        log_name_match = re.search(log_name_pattern, log)
        if log_name_match:
            log_name = log_name_match.group(1)
        log_level_pattern = r'\.(\w+)\('
        log_level_match = re.search(log_level_pattern, log)
        if log_level_match:
            log_level = log_level_match.group(1)
        mask = log_name + '.' + log_level + '(UNKNOWN)'
        full_code = full_code.replace(log, mask)
    
    return full_code,log_list


def task3_generate_dataset(df,new_file_path):
    
    df['log_message'] = df.apply(extract_log_mes, axis=1)
    df["masked_code"],df["log_statement"] = zip(*df.apply(mask_log_mes, axis=1))

    df['task'] = 'task3'
    df["lineID"] = (df.apply(get_logging_line, axis=1))
    df['lineID'] = df['lineID'].apply(lambda x: ','.join(x))
    new_df = df[['masked_code', 'lineID', 'log_message', 'task']].rename(columns={'masked_code': 'code', 'log_message': 'label'})
    print(new_df)
    new_df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task3_generate_dataset(df,"task3_test.tsv")

# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task3_generate_dataset(df,"task3_train.tsv")




#============================================= task 4 =============================================#
# (new task), given pos, predict log statement

def get_log_statement(row):
    logging_code_labels = row['logging_code_labels']
    pattern = r'<line\d+>\s+(.*?);' 
    log_matches = re.findall(pattern, logging_code_labels)
    logs = [match for match in log_matches]
    log_str = '<DIV>'.join(logs)
    # print(log_str)
    return log_str

def task4_generate_dataset(df,new_file_path):
    #input
    df['task'] = 'task4'
    df["lineID"] = (df.apply(get_logging_line, axis=1))
    df['lineID'] = df['lineID'].apply(lambda x: ','.join(x))
    #label
    df["logging_statement"] = df.apply(get_log_statement, axis=1)
    new_df =  df[['without_logging_code_index','logging_statement','task', 'lineID']].rename(columns={'without_logging_code_index': 'code', 'logging_statement': 'label'})
    print(new_df)
    new_df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task4_generate_dataset(df,"task4_train.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task4_generate_dataset(df,"task4_test.tsv")




#============================================= task 5 =============================================#
# logging task

def get_task5_data(row):
    without_logging_code_index = row['without_logging_code_index']
    logging_code_labels = row['logging_code_labels']
    return without_logging_code_index,logging_code_labels


def task5_generate_dataset(df,new_file_path):
    df["without_logging_code_index"],df["logging_code_labels"] = zip(*df.apply(get_task5_data, axis=1))
    df['task'] = 'task5'
    df["lineID"] = ""
    new_df =  df[['without_logging_code_index','logging_code_labels','task', 'lineID']].rename(columns={'without_logging_code_index': 'code', 'logging_code_labels': 'label'})

    print(new_df)
    new_df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task5_generate_dataset(df,"task5_train.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task5_generate_dataset(df,"task5_test.tsv")


