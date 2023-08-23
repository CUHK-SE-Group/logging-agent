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


"""
#生成task1.1的数据集
df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task1_generate_dataset(df,"task1_train.tsv")

df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
task1_generate_dataset(df,"task1_test.tsv")

df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
task1_generate_dataset(df,"task1_eval.tsv")

"""
#测试
#df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
#task1_generate_dataset(df,"task1_just_test.tsv")





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
    """
                id                                          full_code  ... logging_line                                         all_line
0            0  ['public class A {', '  public int enable(GL g...  ...          [3]                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
1            1  ['public class A {', '  public static void gre...  ...     [30, 39]  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...
...        ...                                                ...  ...          ...                                                ...
101398  106380  ['public class A {', '  private boolean doChec...  ...          [5]  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...
101399  106381  ['public class A {', '  @Test', '  public void...  ...         [50]  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...
    """


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


def task2_generate_datasets(df,new_file):
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


"""
#生成task2的train数据集
df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
new_df = generate_line_index(df)
task2_generate_datasets(new_df,"task2_train.tsv")


#生成task2的test数据集
df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
new_df = generate_line_index(df)
task2_generate_datasets(new_df,"task2_test.tsv")

df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
new_df = generate_line_index(df)
task2_generate_datasets(new_df,"task2_eval.tsv")
"""




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

"""
#生成task3的数据集
df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task3_generate_dataset(df,"task3_train.tsv")

df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
task3_generate_dataset(df,"task3_test.tsv")
"""


#测试
#df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
#task1_generate_dataset(df,"task1_just_test.tsv")




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

"""
#生成task4的train数据集
df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task4_generate_dataset(df,"task4_train.tsv")


#生成task4的test数据集
df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
task4_generate_dataset(df,"task4_test.tsv")
"""

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



"""
#生成task5的数据集
df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task5_generate_dataset(df,"task5_train.tsv")

df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
task5_generate_dataset(df,"task5_eval.tsv")
"""
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

"""
#生成task6的数据集
df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task6_generate_dataset(df,"task6_train.tsv")

df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
task6_generate_dataset(df,"task6_test.tsv")
"""







#============================================= task 9(bak) =============================================#


def mask_message1(row):
    code = row["full_code"]
    pattern = r'(log\.[a-zA-Z_]+|logger\.[a-zA-Z_]+)\((.*)\)'
    match = re.search(pattern, row["full_code"], flags=re.IGNORECASE)

    if match:
        logging_statement = match.group(2)
        masked_code = re.sub(pattern, r'\1.("UNKNOWN")', code, flags=re.IGNORECASE,count=0)
        print(masked_code)

    # Extract and filter log levels
    labels = re.findall(pattern, code, flags=re.IGNORECASE)
    print(logging_statement)
    
    return masked_code, labels

#row = {"full_code": '123 hhha LOG.debug(""determine record resources for data model {}"", finalDataModel.getUuid()); 123 hhha LOG.info(""determine"", finalDataModel.getUuid());'}
#mask_message1(row)




def task9_generate_dataset(df,new_file_path):
    df["code"],df["label"] = zip(*df.apply(mask_message, axis=1))
    df = df[['code', 'full_code', 'label']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)



def mask_message(row):
    code = row["full_code"]
    #pattern = r'((log|logger)\.[a-zA-Z_]+\()(.*?)(\))'
    pattern = r'(log\.[a-zA-Z_]+|logger\.[a-zA-Z_]+)\((.*)\)'
    masked_code = code
    logging_statements = []

    for match in re.finditer(pattern, code, flags=re.IGNORECASE):
        logging_statement = match.group(3)
        logging_statements.append(logging_statement)
        masked_logging_statement = match.group(1) + '"UNKNOWN"' + match.group(4)
        masked_code = masked_code.replace(match.group(0), masked_logging_statement)
    
        print(masked_code)
    #print(masked_code)
    #print(logging_statement)

    return masked_code, logging_statements



#row = {"full_code": 'LOG.debug(""determine record resources for data model {}"", finalDataModel.getUuid());'}
#mask_message(row)



#============================================= task 11 =============================================#



#提取logging statement
def get_logging_statement(row):
    code = row['without_logging_code']
    pattern = r'\b(?:log|logger|LOGGER|LOG|log)\.[a-zA-Z]+\([^)]+\);'
    print(row['logging_code_labels'])
    #log_str = "<line5>    log.debug(""StartOf createProvider - REQUEST Insert /providers"");<line11>      logger.info(""createProvider exception"", e);<line13>    LOGGER.debug(""EndOf createProvider"");"
    log_statements = re.findall(pattern, row['logging_code_labels'])
    #log_statements = re.findall(pattern, log_str)
    print(str(log_statements))
    return code,log_statements

#row = {"without_logging_code":'',"logging_code_labels": "<line5>    log.debug(""StartOf createProvider - REQUEST Insert /providers"");<line11>      logger.info(""createProvider exception"", e);<line13>    LOGGER.debug(""EndOf createProvider"");"}
#get_logging_statement(row)


def task11_generate_dataset(df,new_file_path):
    df= df[['without_logging_code_index', 'logging_code_labels']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)

"""

df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
task11_generate_dataset(df,"task11_test.tsv")


df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task11_generate_dataset(df,"task11_train.tsv")

df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
task11_generate_dataset(df,"task11_eval.tsv")

"""




#============================================= task 12 =============================================#

def task12_generate_dataset(df,new_file_path):
    df= df[['without_logging_code', 'full_code']]
    print(df)
    df.to_csv(new_file_path, sep='\t', index=True)


"""
df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
task12_generate_dataset(df,"task12_test.tsv")


df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
task12_generate_dataset(df,"task12_train.tsv")

df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
task12_generate_dataset(df,"task12_eval.tsv")
"""







#============================================= task 8 (doing) =============================================#


#data = ['"".sendEvent value="" + value', '""Unable to added handler for {} due to missing classes on the classpath"",opClass.getSimpleName(),e', '""Failed to extract tags due to : {} Total Parser Errors : {}"",e.getMessage(),parserErrors', '""new ExtensionException(prototype.getClass())""', '"".sendEvent value="" + value', '""Unable to added handler for {} due to missing classes on the classpath"",opClass.getSimpleName(),e']

#result_list = ['value', 'opClass.getSimpleName();e', 'e.getMessage();parserErrors', None, 'value', 'opClass.getSimpleName();e']

def extract_variable1(row):
    # 匹配并删除 "" "" 及其之间的内容
    pattern = r'""(.*?)""'
    result_list = []

    for item in list(row['messages']):
        result = re.sub(pattern, '', item)
        result = re.sub(r'\s+', ' ', result)  # 删除多余的空格

        result = result.replace('+', '')  # 删除 "+"
        result = result.replace(',', ';')  # 将逗号替换为分号

        if result and result[0] == ';':  # 如果结果不为空且第一个字符是分号
            result = result[1:]  # 删除第一个字符（分号）        

        result = result.strip() if result else None  # 去除首尾空格，如果结果为空则设置为 None
        result_list.append(result)


    return row['masked_code'],row['full_code'],result_list


def mask_log_var(row):

    # 正则表达式模式，用于匹配log括号内的内容
    pattern = r'\b(?:LOG|LOGGER)\.(?:debug|info|warn|error|trace|fatal)\((.*?)\);'

    matches = re.findall(pattern, row['logging_code_labels'], flags = re.MULTILINE | re.IGNORECASE)

    # 将匹配结果存储在一个列表中
    log_messages = [match.strip() for match in matches]

    # 将匹配到的内容替换为 "UNKNOW"
    # for match in matches:
    #     print(match)
    #     row['masked_code'] = re.sub(re.escape(match), "UNKNOWN", row['full_code'], flags=re.MULTILINE | re.IGNORECASE)
    
    pattern_1 = r'"(.*?)"'
    for full_mes in log_messages:
        print(full_mes)
        match = re.search(pattern_1,full_mes)
        mes = '\"\"' + match.group(1) + '\"\"'

        print(mes)
        extracted_mes = full_mes.replace('\"' + match.group(1) + '\"', "")
        print(extracted_mes)

    
    row['masked_code'] = row['full_code']
    return row['masked_code'],row['full_code'],log_messages



def extract_variable(row):
    # 匹配并删除 "" "" 及其之间的内容
    pattern = r'""(.*?)""'
    full_code = row['full_code']
    result_list = []

    for item in list(row['messages']):
        mes = re.search(pattern,item)
 
        result = re.sub(pattern, '', item)

        result = re.sub(r'\s+', ' ', result)  # 删除多余的空格

        result = result.replace('+', '')  # 删除 "+"
        result = result.replace(',', ';')  # 将逗号替换为分号

        if result and result[0] == ';':  # 如果结果不为空且第一个字符是分号
            result = result[1:]  # 删除第一个字符（分号）        

        result = result.strip() if result else None  # 去除首尾空格，如果结果为空则设置为 None
        result_list.append(result)

    return row['masked_code'], full_code, result_list



def task8_generate_dataset(df,new_file_path):
    df["masked_code"],df["full_code"],df["var"] = zip(*df.apply(mask_log_var, axis=1))
    #print(df)
    new_df = df[['masked_code','full_code','var']]
    new_df.to_csv(new_file_path, sep='\t', index=False)


#df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
#task8_generate_dataset(df,"taskx_just_test.tsv")




#============================================= task 9 =============================================#

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




def task9_generate_dataset(df,new_file_path):
    
    df["log_message"] = df.apply(extract_log_mes, axis=1)
    df["masked_code"],df["log_statement"] = zip(*df.apply(mask_log_mes, axis=1))

    print(df)
    new_df = df[['full_code','masked_code','log_statement','log_message']]
    new_df.to_csv(new_file_path, sep='\t', index=False)


# df = pd.read_csv(os.path.join("./reformatted_data", "source_just_test.tsv"), sep='\t') 
# task9_generate_dataset(df,"task9_just_test.tsv")

# df = pd.read_csv(os.path.join("./", "source_eval.tsv"), sep='\t') 
# task9_generate_dataset(df,"task9_eval.tsv")

# df = pd.read_csv(os.path.join("./", "source_test.tsv"), sep='\t') 
# task9_generate_dataset(df,"task9_test.tsv")

# df = pd.read_csv(os.path.join("./", "source_train.tsv"), sep='\t') 
# task9_generate_dataset(df,"task9_train.tsv")




