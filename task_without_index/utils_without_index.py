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
    # 对 DataFrame 进行去重处理
    print("去重前的df数量:", len(df))
    df.drop_duplicates(subset=['full_code', 'logging_code_labels'], inplace=True)
    print("去重后的df数量:", len(df))
    df.to_csv(file_path, sep='\t', index=False)
    """
    去重前的df数量: 12014
    去重后的df数量: 7122
    6929
    去重前的df数量: 101400
    去重后的df数量: 58589
    57131
    去重前的df数量: 12645
    去重后的df数量: 7321
    7134
    """

### 对source数据去重
#drop_duplicates_data("source_test.tsv")
#drop_duplicates_data("source_train.tsv")
#drop_duplicates_data("source_eval.tsv")


def delete_dirty_data(df,output_file):
    #删除包含'STATS_LOG'的行
    df = df[~df.apply(lambda row: row.astype(str).str.contains('STATS_LOG').any(), axis=1)]
    df = df[~df.apply(lambda row: row.astype(str).str.contains('FAKE_LOGGER').any(), axis=1)]

    #替换
    df = df.applymap(lambda cell: re.sub(r'(|s)_logger\.', 'logger.', cell, flags=re.IGNORECASE) if isinstance(cell, str) else cell)
    df = df.applymap(lambda cell: re.sub(r'(|s)_log\.', 'log.', cell, flags=re.IGNORECASE) if isinstance(cell, str) else cell)


    # 使用正则表达式匹配包含log/logger和任何日志级别的行
    pattern = r'\b(?:log|logger|LOG|LOGGER)\.(?:' + '|'.join(logging_levels) + r')\b'
    mask = df['logging_code_labels'].str.contains(pattern, case=False, flags=re.IGNORECASE)
    
    # 打印并删除不匹配的行
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

    # 匹配既不是英文字符又不是ASCII字符的内容
    non_english_ascii_pattern = re.compile(r'[^\x00-\x7F]+')

    # 打开TSV文件以读取
    with open(inputfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 用于存储过滤后的行
    filtered_lines = []

    count = 0

    for line in lines:
        if not non_english_ascii_pattern.search(line):
            filtered_lines.append(line)
        else:
            count += 1
    print('filtered num:',count)

    # 打开一个新的TSV文件以写入
    with open(outputfile, 'w', encoding='utf-8') as file:
        file.writelines(filtered_lines)

# filter_non_english_ascii_data("source_eval.tsv","source_eval.tsv")
# filter_non_english_ascii_data("source_test.tsv","source_test.tsv")
# filter_non_english_ascii_data("source_train.tsv","source_train.tsv")


def identify_logging_statement_code_line(target_line, target_idx, lines):

    logging_code_regex = r'[.](off)?(fatal)?(error)?(warn)?(info)?(debug)?(trace)?(all)?[(]'

    logging_code_label = ""
    span = ()

    logging_code_match_res = re.search(logging_code_regex, target_line)

    if logging_code_match_res:

        # logging_code_label = "<line" + str(target_idx - 1) + "> " + target_line
        logging_code_label = target_line

        if target_line.endswith("("):  # consider cross lines logging statement code
            for j in range(target_idx + 1, len(lines)):
                logging_code_label += lines[j].strip()
                if lines[j].endswith(");"): span = (target_idx, j+1); break
        else:
            span = (target_idx, target_idx + 1)

    return logging_code_label, span



def save_as_csv_file(data,new_filename):
    with open(new_filename, "w", newline="", encoding="utf-8") as output:
        fieldnames = ["index", "function"]
        csv_writer = csv.DictWriter(output, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(data)
    print("new functions saved to", new_filename)



#计算logging_code_labels是否为空
def df_is_null(df):
    if df['logging_code_labels'].isnull().any():
        empty_logging_code_count = df[df['logging_code_labels'].isnull()].shape[0]
    else:
        empty_logging_code_count = 0

    print("空数据的count:", empty_logging_code_count)
    return empty_logging_code_count




# 1.打开所有7个文件，并存为存为7个df
# 2.获取一个df的行数，默认所有都一样
# 3.for row in len(df)
# 4.random选择df，并给他们设置概率
# 5.读取选择的df[row],写入新df中
# 6.新df保存为新的tsv文件

def generate_mix_file(inputfile, outputfile):
    # 存储7个文件的DataFrame的字典
    dfs = {}
    
    # 打开并读取所有7个文件，存为7个df
    for i in [1, 3, 5, 6, 7, 8]:
        file_name = f"task{i}_" + inputfile + "_without_index.tsv"
        file_path = os.path.join("./", file_name)
        
        # 读取当前文件并存储在相应的DataFrame中
        dfs[f'df{i}'] = pd.read_csv(file_path, sep='\t')

    # 获取一个df的行数，默认所有都一样
    num_rows = len(dfs['df1'])  # 默认每个文件的行数相同
    
    # 创建一个新的DataFrame来存储随机选择的行
    selected_rows = []
    
    # 为每个文件设置选择概率
    file_weights = [0.25 if i in [1, 3] else 0.125 for i in [1, 3, 5, 6, 7, 8]]
    print(file_weights)

    # 循环遍历每一行
    for row in range(num_rows):
              
        # 随机选择一个文件，并设置概率
        random_file_key = random.choices(list(dfs.keys()), weights=file_weights)[0]
        random_file = dfs[random_file_key]  # 获取选定的文件
          
        # 读取选择的df[row]，写入新df中
        selected_row = random_file.iloc[row:row+1][['code', 'label']]
        selected_row['task'] = 'task' + random_file_key[2:]  # 添加任务列
        selected_rows.append(selected_row)


    # 将选定的行合并为一个新的DataFrame
    new_df = pd.concat(selected_rows, ignore_index=True)
    
    # 将新的DataFrame写入到一个新的TSV文件中
    new_df.to_csv(outputfile, sep='\t', index=False)



# generate_mix_file('eval','mixed_task_eval_without_index.tsv')
# generate_mix_file('test','mixed_task_test_without_index.tsv')
# generate_mix_file('train','mixed_task_train_without_index.tsv')



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


# #生成task1的数据集
# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task1_generate_dataset(df,"task1_train_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task1_generate_dataset(df,"task1_test_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task1_generate_dataset(df,"task1_eval_without_index.tsv")


#测试
#df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
#task1_generate_dataset(df,"task1_just_test.tsv")






#============================================= task 3 =============================================#


def get_logging_num(row):

    pattern = r'<line\d+>\s+(.+?)(?=(?:<line\d+>)|$)'  #分组 (?:<line\d+>)|$，用于匹配下一个 <line(\d+)> 或者文本结束
    logging_lines = re.findall(pattern, row['logging_code_labels'])

    label = len(logging_lines)

    return row['without_logging_code'],label



def task3_generate_dataset(df,new_file_path):
    df["code"],df["label"] = zip(*df.apply(get_logging_num, axis=1))
    df = df[['code','label']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)


#生成task3的数据集
# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task3_generate_dataset(df,"task3_train_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task3_generate_dataset(df,"task3_test_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task3_generate_dataset(df,"task3_eval_without_index.tsv")


#测试
#df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
#task1_generate_dataset(df,"task1_just_test.tsv")






#============================================= task 5 =============================================#


def mask_log_level(row, mask="UNKNOWN"):
    code = row['full_code']
    level_pattern = '|'.join(logging_levels)
    #pattern = r'((?:logger|log)\.)([a-zA-Z]+)(\()' #分成三个组，每个（）为一组    
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
    return new_code, labels



def task5_generate_dataset(df,new_file_path):
    df["masked_code"],df["label"] = zip(*df.apply(mask_log_level, axis=1))
    df = df[['masked_code', 'full_code', 'label']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)




#生成task5的数据集
# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task5_generate_dataset(df,"task5_train_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task5_generate_dataset(df,"task5_test_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task5_generate_dataset(df,"task5_eval_without_index.tsv")

#df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
#task5_generate_dataset(df,"task5_test.tsv")




#============================================= task 6 =============================================#


def replace_log_level_2_random_level(row):
    # Join all logging levels into a regex pattern
    code = row['full_code']
    level_pattern = '|'.join(logging_levels)
    random_level = random.choice(logging_levels)

    # Include the logging level pattern into the whole pattern
    pattern = r'((?:logger|log)\.)(' + level_pattern + r')(\()'

    #re.sub函数的count参数来指定替换的次数为0，表示替换所有匹配项。
    new_code = re.sub(pattern, r'\1' + random_level + r'\3', code, flags=re.IGNORECASE, count=0)

    # Extract and filter log levels
    labels = re.findall(pattern, code, flags=re.IGNORECASE)
    if labels == None:
        print("Logging statement not found.")
    else:
        labels = [label[1] for label in labels if label[1].lower() in (level.lower() for level in logging_levels)]

    return new_code, labels


def task6_generate_dataset(df,new_file_path):
    df["code"],df["label"] = zip(*df.apply(replace_log_level_2_random_level, axis=1))
    df = df[['code', 'full_code', 'label']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)


#生成task6的数据集
# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task6_generate_dataset(df,"task6_train_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task6_generate_dataset(df,"task6_test_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task6_generate_dataset(df,"task6_eval_without_index.tsv")






#============================================= task 7 =============================================#

def extract_log_mes(row):
    #这个函数可以正确提取到log_messages,但是替换有问题，会漏和误替换。
    # 正则表达式模式，用于匹配log括号内的内容
    pattern = r'\b(?:LOG|LOGGER)\.(?:debug|info|warn|error|trace|fatal)\((.*?)\);'
    

    matches = re.findall(pattern, row['logging_code_labels'], flags = re.MULTILINE | re.IGNORECASE)

    # 将匹配结果存储在一个列表中
    log_messages = [match for match in matches]
    return log_messages


def mask_log_mes(row):
    #1.从label中提取出所有log statements为一个list
    #匹配到以line开头然后接n个空格，；结尾的句子，然后括号内的就是匹配的内容，即log statement
    logging_code_labels = row['logging_code_labels']
    full_code = row['full_code']
    log_pattern = r'<line\d+>\s+(.*?);' 
    log_matches = re.findall(log_pattern, logging_code_labels)

    log_list = []

    for log_match in log_matches:
        log_list.append(log_match)
    
    #2.循环处理每个code的所有logs
    for log in log_list:
        #2.1.提取出log头
        log_name_pattern = r'(\w+)\.' 
        log_name_match = re.search(log_name_pattern, log)
        if log_name_match:
            log_name = log_name_match.group(1)
        else:
            #说明这个样本有问题，生成文件后，手动删除该条数据
            log_name = 'DeleteMe!'
        #2.2.提取出log level
        log_level_pattern = r'\.(\w+)\('
        log_level_match = re.search(log_level_pattern, log)
        if log_level_match:
            log_level = log_level_match.group(1)
        else:
            log_level = 'DeleteMe!'
        #2.3.组合为mask = log.level(UNKNOWN)
        mask = log_name + '.' + log_level + '(UNKNOWN)'
        #2.4.在full_code中查找这个log statement，并替换为mask
        full_code = full_code.replace(log, mask)
    
    #3.return masked_full_code, log statements
    return full_code,log_list


def task7_generate_dataset(df,new_file_path):
    
    df["log_message"] = df.apply(extract_log_mes, axis=1)
    df["masked_code"],df["log_statement"] = zip(*df.apply(mask_log_mes, axis=1))

    print(df)
    new_df = df[['full_code','masked_code','log_statement','log_message']]
    new_df.to_csv(new_file_path, sep='\t', index=False)



# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task7_generate_dataset(df,"task7_eval_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task7_generate_dataset(df,"task7_test_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task7_generate_dataset(df,"task7_train_without_index.tsv")

#df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
#task7_generate_dataset(df,"task7_just_test.tsv")





#============================================= task 8 =============================================#


# 1.得到一个log的样本库，一条log为一列

def generate_log_lib(input_file,output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile, delimiter='\t')
        # 创建目标TSV文件的列名
        fieldnames = ['log_statements']
        
        # 创建CSV写入对象
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for row in reader:      

            logging_code_labels = row['logging_code_labels']
            log_pattern = r'<line\d+>\s+(.*?);' 
            log_matches = re.findall(log_pattern, logging_code_labels)

            log_list = []

            for log_match in log_matches:
                #log_list.append(log_match)
                writer.writerow({'log_statements': log_match})

# input_file = "source_eval.tsv"
# output_file = "log_library.tsv"
# generate_log_lib(input_file,output_file)


def get_code_line_num(row):
    # 比如line0-3，返回4
    logging_lines = re.findall(r"<line(\d+)>", row['full_code_index'])
    #print(len(logging_lines))
    return len(logging_lines)



def get_a_random_log(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random_line_number = random.randint(1, len(lines) - 1)
    random_line = lines[random_line_number]

    # print(random_line)
    return random_line

    

def insert_random_log(full_code, logging_lines_num, line_index):

    insert_log = '<line' + str(line_index-1) + '> ' + get_a_random_log("log_library.tsv").replace("\n", "") + '; '
    # print('insert_log:',insert_log)

    new_code = full_code.replace('<line' + str(line_index-1) + '> ', insert_log + '<line' + str(line_index) + '> ')

    # 4.循环i=len 到 x, 替换<line i> 为 <line i+1>, i--
    for i in range (logging_lines_num, line_index-1, -1):
        new_code = new_code.replace(f'<line{i}>', f'<line{i+1}>')

    # 5.最后会有两个<line x+1>,替换第一个为<line x>，指定只替换第一个
    new_code = new_code.replace('<line' + str(line_index+1) + '> ','<line' + str(line_index) + '> ',1)

    # print('new_code:',new_code)


    #删除行号
    pattern = r'<line\d+>'
    new_code = re.sub(pattern,'', new_code, count=0)
    insert_log = re.sub(pattern,'', insert_log, count=0)

    return insert_log, new_code




def task8_generate_dataset(df,new_file_path):
    for index,row in df.iterrows():
        random_num = random.randint(0, 3)
        # print(random_num)

        insert_log_list = []
        new_code = row['full_code_index']

        logging_lines_num = get_code_line_num(row)

        # 3.构造随机的 <line x>，从log样本库中随机拿一条log, 拼接为 <line x> log xxx; <line x> 并替换在code中的 <line x>
        line_index_list = set()
        while len(line_index_list) < random_num:
            line_index = random.randint(1, logging_lines_num - 1)
            line_index_list.add(line_index)

        # 将set转换回列表
        line_index_list = sorted(list(line_index_list))


        for i in range(random_num):
            insert_log, new_code = insert_random_log(new_code,logging_lines_num+i,line_index_list[i])
            # print(f'new_code{i}:', new_code)
            insert_log_list.append(insert_log)
        
        df.loc[index, "new_code"] = new_code
        df.loc[index, "insert_log"] = str(insert_log_list)

        if random_num == 0:
            df.loc[index, "insert_log"] = None


    new_df = df[['code','label','insert_log']]
    new_df.to_csv(new_file_path, sep='\t', index=False)
    print(new_df)



def delete_task8_index(file_name):
    df = pd.read_csv(os.path.join("./", file_name), sep='\t') 
    for index,row in df.iterrows():
        #删除行号
        pattern = r'<line\d+>'
        df.loc[index, "code"] = re.sub(pattern,'', row['code'], count=0)
        df.loc[index, "label"] = re.sub(pattern,'', row['label'], count=0)
    df.to_csv(file_name, sep='\t', index=False)

# delete_task8_index("task8_test_without_index.tsv")
# delete_task8_index("task8_train_without_index.tsv")
# delete_task8_index("task8_eval_without_index.tsv")


# df = pd.read_csv(os.path.join("./", "source_just_test.tsv"), sep='\t') 
# task8_generate_dataset(df,"task8_just_test.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task8_generate_dataset(df,"task8_test_without_index.tsv")


# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task8_generate_dataset(df,"task8_train_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task8_generate_dataset(df,"task8_eval_without_index.tsv")





#============================================= task 10 =============================================#

def task10_generate_dataset(df,new_file_path):
    df['task'] = 'task10'
    df = df[['without_logging_code', 'full_code', 'task']]
    df = df.rename(columns={'without_logging_code': 'code', 'full_code': 'label'})

    print(df)
    df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task10_generate_dataset(df,"task10_test_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task10_generate_dataset(df,"task10_train_without_index.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task10_generate_dataset(df,"task10_eval_without_index.tsv")


