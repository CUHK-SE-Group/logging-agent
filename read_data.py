import os
import pandas as pd
import random

def get_prompt(instruction, input):
    str0 = "### System: You are a logging agent designed to identify or generate log statements for a given code snippet." + "\n"
    str1 = "### Instruction: " + instruction + "\n"
    str2 = "### Input: " + input + "\n"
    str3 = "### Output:" 

    prompt = str1 + str2 + str3
    return prompt

def get_instrucion(task_name):
    instruction = ""
    # need logging for the function or not ?
    if task_name == "task1":
        prefix = "The input consists of the code with each line separated by line ID '<line#>'. "
        instructions = [
            "Please assess this code and determine whether more logging is required.",
            "Please ascertain if additional logging is necessary for the given code.",
            "Identify if further logging is needed for the provided code snippet.",
            "Examine whether this code snippet requires supplemental logging.",
            "Please decide if extra logging would be beneficial for the following code.",
            "Confirm if more log statements should be added in the given function.",
            "Determine if the addition of more log statements is necessary for this code snippet.",
            "Please decide if additional logging is required for this code snippet.",
            "Check this code and confirm whether it needs more logging.",
            "Specify if this piece of code still need additional logging."
        ]
        suffix = " The expected output should be 'Yes' or 'No', indicating whether the code needs logging."
        instruction = prefix + random.choice(instructions) + suffix
    # need logging for the specified line or not ?
    if task_name == "task2":
        prefix = "The input consists of the code with each line separated by line ID '<line#>' (after '<CODE>') and a specified line ID to be inferred (after '<LINE>'). "
        instructions = [
            "Please determine if this code requires logging for the specified line.",
            "State whether logging is necessary for the line.",
            "Identify if additional logging is needed for the specified line.",
            "Decide if the code needs log statements for the given line.",
            "Confirm if the code snippet could benefit from logging for the specified line.",
            "Determine if more log statements should be added for the specified line.",
            "Please assess the code and decide if the function requires additional log statements for the given line.",
            "Check whether the provided function needs logging for the line.",
            "Specify if the given code could use more log statements for the given line."
        ]
        suffix = " The expected output should be 'Yes' or 'No', indicating whether the specified line needs logging."
        instruction = prefix + random.choice(instructions) + suffix
    # predict number of logging statement needed
    if task_name == "task3":
        prefix = "The input consists of the code with each line separated by line ID '<line#>'. "
        instructions = [
            "Review this function and indicate the necessary number of log statements.",
            "Provide the count of log statements that should be created in this code snippet.",
            "Identify the needed amount of log statements for this function.",
            "Estimate the number of log statements essential for this code.",
            "For this function, specify the count of the log statements required.",
            "Determine the necessary number of logging statements needed for this code snippet.",
            "Suggest how many log statements are needed for this function.",
            "Specify the number of log statements to be added in the function.",
            "Identify the number of log statements required for this function.",
            "Estimate the needed count of log statements within the code snippet."
        ]
        suffix = " The expected output should be an integer, indicating the number of new log statements needed."
        instruction = prefix + random.choice(instructions) + suffix
    # predict line ID of logging statement needed
    if task_name == "task4":
        prefix = "The input consists of the code with each line separated by line ID '<line#>'. "
        instructions = [
            "Identify prospective line IDs suitable for logging in the given code.",
            "Examine the provided code and record the line IDs where additional logging could be necessary.",
            "Determine the appropriate line IDs where additional log statements can be beneficial.",
            "Mark the prospective locations in the code where you perceive that log statements could be inserted.",
            "Determine which line IDs in the provided code will have an advantage with the addition of logging statements.",
            "Note down the possible line IDs in the given code which necessitate new log statements.",
            "Describe where in the given snippet of code new log statements should be inserted.",
            "Analyze the code snippet and decide where additional log statements can be incorporated.",
            "Study the code snippet provided, and suggest where additional log statements could make the code more informative.",
            "Record the line IDs in the code snippet where the insertion of logging statements would make it more debug-friendly."
        ]
        suffix = " The expected output should be the line IDs '<line#>' separated by commas, indicating the positions where log statements should be added."
        instruction = prefix + random.choice(instructions) + suffix
    # predict log level
    if task_name == "task5":
        prefix = "The input consists of the code with each line separated by line ID '<line#>', and the log levels in this code include 'fatal','error', 'warn', 'info', 'debug', and 'trace'. "
        instructions = [
            "Please replace each 'UNKNOWN' log level with the corresponding log level based on its severity.",
            "Allocate correct log levels to replace the 'UNKNOWN' log levels for all log statements.",
            "Please assign the appropriate log levels to replace the 'UNKNOWN' log levels for each log statement.",
            "Replace all log statements labeled as 'UNKNOWN' with correct log levels.",
            "Please replace each 'UNKNOWN' log level in the function with suitable log levels based on their severity.",
            "Update 'UNKNOWN' log levels with the correct ones for each log statement.",
            "Please make all the log statements with 'UNKNOWN' level assigned with the correct log levels.",
            "Assign appropriate log levels to each log statement in the provided function, replacing 'UNKNOWN' log levels.",
            "Adjust 'UNKNOWN' log labels, substituting them with the correct log levels.",
            "For each log statement in the function, replace the 'UNKNOWN' log level with the correct severity level."
        ]
        suffix = " The expected output should be the log levels separated by commas, indicating the appropriate log levels for each log statement."
        instruction = prefix + random.choice(instructions) + suffix
    # correct log level
    if task_name == "task6":
        prefix = "The input consists of the code with each line separated by line ID '<line#>', and all the log levels in this code, including 'fatal', 'error', 'warn', 'info', 'debug', and 'trace', are incorrect."
        instructions = [
            "Please revise and correct all incorrect log levels in the given function.",
            "Please rectify every wrong level of log statements in the provided code.",
            "Correct all the log levels used in the function.",
            "Rectify every inaccuracy present in log levels in the function.",
            "Investigate the function and modify all flawed log levels accordingly.",
            "Please adjust every misused log level to the appropriate ones in the code.",
            "Ensure that the function uses the correct log levels by correcting all the existing levels.",
            "Analyze the function and revise each one of the incorrect log levels.",
            "Thoroughly review and amend all incorrect log levels within the function.",
            "Inspect all lines of the function, spot all false log levels, and correct each of them."
        ]
        suffix = " The expected output should be the corrected log levels separated by commas."
        instruction = prefix + random.choice(instructions) + suffix
    # predict log message
    if task_name == "task7":
        prefix = "The input consists of the code with each line separated by line ID '<line#>'. "
        instructions = [
            "Please interpret the 'UNKNOWN' log messages in the provided function.",
            "Based on the context within the function, infer the hidden log message masked as 'UNKNOWN'.",
            "Please estimate the concealed log message originally labeled as 'UNKNOWN'.",
            "From the given function, please deduce the actual content of the log message that has been masked as 'UNKNOWN'.",
            "Please make an appropriate estimate of the hidden log message.",
            "Based on your understanding of the given code, determine what the hidden 'UNKNOWN' log message could be.",
            "Suppose that the 'UNKNOWN' represents a log message in the function, please predict its actual content.",
            "Please fill in the hidden information from the 'UNKNOWN' log message in the function.",
            "Try to suggest possible content for the log messages that are masked as 'UNKNOWN'.",
            "Determinate the appropriate message for the concealed log message labeled as 'UNKNOWN' in the function."
        ]
        suffix = " The expected output should be the log messages separated by the separator '<DIV>', indicating the appropriate log messages for each log statement."
        instruction = prefix + random.choice(instructions) + suffix
    # remove logging statement
    if task_name == "task8":
        prefix = "The input consists of the code with each line separated by line ID '<line#>'. "
        instructions = [
            "Scrutinize the provided code snippet to identify and eliminate any redundant log statements.",
            "Remove any log statements that do not add value or relate to the code's purpose.",
            "Remove all unessential log statements in the code snippet.",
            "Remove any log statements that unessential or unrelated to the function's main task.",
            "Delete any log statements that may be redundant or don't provide meaningful information in the given code.",
            "Eliminate any log statements that are not directly relevant to the function's operation.",
            "Go through the function and remove any irrelevant or unnecessary log statement.",
            "Scan the function and remove any extraneous or unrelated log statement.",
            "Analyze the code, delete any irrelevant log statements.",
            "Delete any pointless or unrelated log messages in the provided code snippet."
        ]
        suffix = " The expected output should be the redundant log statements to be removed with their line ID '<line#>' ahead."
        instruction = prefix + random.choice(instructions) + suffix
    # generate logging statement
    if task_name == "task9":
        prefix = "The input consists of the code with each line separated by line ID '<line#>'. "
        instructions = [
            "Please generate the log statements with the appropriate line IDs for the given code.",
        ]
        suffix = " The expected output should be the complete log statements needed with their line ID '<line#>' ahead."
        instruction = prefix + random.choice(instructions) + suffix
    return instruction


def read_data(row):
    task_name = row['task']
    input = row['code']
    output = row['label']

    prompt = []
    for i in range(len(input)):
        prompt.append(get_prompt(get_instrucion(task_name[i]), input[i]))
    return prompt, output


def get_one_prompt(row):
    task_name = row['task']
    input = row['code']

    prompt = get_prompt(get_instrucion(task_name), input)
    return prompt


# unit test
if __name__ == "__main__":
    datafile_path = "task_data/subtasks_test_without_index.tsv"
    df = pd.read_csv(os.path.join(datafile_path), sep='\t')
    row = df.iloc[0]
    data, label = read_data(row)
    print(data, label)