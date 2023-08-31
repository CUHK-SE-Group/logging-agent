import os
import pandas as pd
import random

def get_instruction_format(instruction,input):
    str0 = "### System: You are a logging agent designed to identify or generate logging statements for a given code snippet." + "\n"
    str1 = "### Instruction: " + instruction + "\n"
    str2 = "### Input: " + input + "\n"
    str3 = "### Output: " 

    instruction_format = str1 + str2 + str3
    # print(instruction_format)
    return instruction_format



def get_one_instrucion(task_name):
    if task_name == "task1":
        a = "The input is a Java code function, which may or may not already have logging. "
        instructions = [
            "Please assess this Java code and determine whether more logging is required.",
            "Evaluate this piece of code to ascertain if additional logging is necessary.",
            "Analyze this code snippet and identify if further logging is needed.",
            "Examine this Java function and state whether it requires supplemental logging.",
            "Review this Java code and decide if extra logging would be beneficial.",
            "Inspect this function and confirm if more logging statements should be added.",
            "Scrutinize this code to determine if the addition of more log statements is necessary.",
            "Please decide if additional logging is required for this Java code snippet.",
            "Check this Java code and confirm whether it needs more logging.",
            "Study this Java code and specify if it could use additional logging."
        ]
        b = "If it does, output ""Yes""; otherwise, output ""No."""
        instruction = a + random.choice(instructions) + b
    if task_name == "task3":
        a = "The input is a Java code function, and all the logging statement has been removed from it."
        instructions = [
            "Review this Java code function and indicate the necessary number of logging statements.",
            "Considering this Java code segment, provide the count of log statements that should be created.",
            "Analyze this Python function and identify the needed amount of logging statements.",
            "Estimate the number of log statements essential for this Java code.",
            "For this Java function, specify the count of the logging statements required.",
            "Study this code segment and determine the necessary number of logging instances.",
            "Scrutinize this Java function and suggest how many logging statements are needed.",
            "Inspect this piece of Java code and specify the number of logs to be added.",
            "Identify the number of logging statements required for this Java function.",
            "Given this Java code snippet, estimate the needed count of log instances."
        ]
        b = "Please output the number."
        instruction = a + random.choice(instructions) + b
    if task_name == "task5":
        a = "The input is a Java code function, and the level of all the logging statement has been masked as 'UNKNOWN'. Log levels typically include the following levels, from highest to lowest severity: 'fatal','error', 'warn', 'info', 'debug', 'trace'."
        instructions = [
            "In this Java code function, please replace each 'UNKNOWN' log level with the corresponding log level based on its severity.",
            "Assess the Java code function and allocate correct log levels to the logging statements.",
            "Please review the Java function, then determine and assign the appropriate log levels to each logging statement.",
            "Analyze the Java code function, find all logging statements labeled as 'UNKNOWN' and replace them with correct log levels.",
            "Identify each 'UNKNOWN' log level in the Java code function and replace with suitable log levels based on their severity.",
            "Examine the Java code function and update 'UNKNOWN' log levels with the correct ones for each logging statement.",
            "Carefully review the Java function and ensure all the logging statements are assigned with the correct log levels.",
            "Assign appropriate log levels to each logging statement in the provided Java code function, replacing 'UNKNOWN' log levels.",
            "Inspect the given Java code function and adjust 'UNKNOWN' log labels, substituting them with the determined log levels.",
            "For each logging statement in the Java code function, replace the 'UNKNOWN' log level with the correct severity level."
        ]
        b = "Please output the list of predict log levels in order, such as: ['info', 'warn']."
        instruction = a + random.choice(instructions) + b
    if task_name == "task6":
        a = "The input is a Java code function, wherein certain log levels may be inaccurate. This encompasses the possibility of all log levels being correct, all being incorrect, or a combination of both scenarios. Log levels in logging typically include the following levels, from highest to lowest severity: 'fatal','error', 'warn', 'info', 'debug', 'trace'."
        instructions = [
            "Rectify any log level inaccuracies in the given Java code function, and provide the corrected version.",
            "Please scrutinize the log levels in the Java code function and ensure their appropriateness; make adjustments as necessary.",
            "Evaluate the log levels used in the Java code function and correct them as required.",
            "Assess and rectify any inconsistencies in log levels present in the Java code function.",
            "Investigate the Java code function for any inaccurate log levels and remedy them accordingly.",
            "Examine the Java code function and modify any misused log levels you find.",
            "Ensure the Java code function uses the correct log levels; rectify any errors you encounter.",
            "Analyze and correct any inaccurate log levels in the Java code function.",
            "Review and amend any incorrect log levels within the Java function.",
            "Look through the Java function, spot any false log levels, and correct them."
        ]
        b = "Please output the updated code function with corrected log levels."
        instruction = a + random.choice(instructions) + b  
    if task_name == "task7":
        a = "The input is a Java code function, and the message of all the logging statement has been masked as 'UNKNOWN'. "
        instructions = [
            "Please interpret the 'UNKNOWN' log messages in the provided Java code function.",
            "Based on the context within the Java code function, infer the hidden log message masked as 'UNKNOWN'.",
            "Analyze the Java code and estimate the concealed log message originally labeled as 'UNKNOWN'.",
            "From the given Java code function, please deduce the actual content of the log message that has been masked as 'UNKNOWN'.",
            "Evaluate the Java code, and make an appropriate estimate of the hidden log message.",
            "Based on your understanding of the given Java code, determine what the hidden log message could be.",
            "Suppose that the 'UNKNOWN' represents a log message in the Java code function. Predict its actual content.",
            "Please fill in the hidden information from the 'UNKNOWN' log message in the Java code function.",
            "Try to comprehend the given Java code function, and suggest possible content for the masked log messages.",
            "Analyze and determinate the appropriate message for the concealed log statements labeled as 'UNKNOWN' in the Java code function."
        ]
        b = "Please output the modified full code function."
        instruction = a + random.choice(instructions) + b
    if task_name == "task8":
        a = "The input is a Java code function, which may contain some irrelevant or unrelated logs to this function, or may be all the logs within the function are appropriate."
        instructions = [
            "Scrutinize the provided code snippet to identify and eliminate any redundant or irrelevant log statements."
            "Examine the Java function for any log messages that do not add value or relate to its purpose and remove them."
            "Assess whether the code snippet contains any unnecessary or unrelated logs. If so, remove them."
            "Inspect the code and identify any log messages that seem unessential or unrelated to the function's main task. Proceed to remove those."
            "Look through the code to find any log entries that may be redundant or don't provide meaningful information, and remove them."
            "Study the code carefully and eliminate any logging statements that are not directly relevant to the function's operation."
            "Go through the Java code function, spot any irrelevant or unnecessary log entries, and remove them."
            "Scan the Java code function for any extraneous or unrelated log messages, and if found, remove them."
            "Analyze the code, identify any irrelevant or superfluous log statements, and ensure they are deleted."
            "Peruse the provided Java code function, find any pointless or unrelated log messages, and take steps to remove them."
        ]
        b = "Remove the identified incorrect logs and output the full clean Java function."
        instruction = a + random.choice(instructions) + b
    if task_name == "task10":
        a = "The input is a Java code function, and all the logging statement has been removed from it. "
        instructions = [
            "Incorporate important log entries into this Java function to keep track of its operation.",
            "Execute logging statements at critical points in the function to keep track of its progress.",
            "Add meaningful logs in this Java function for potential debugging or execution tracing.",
            "Mark significant actions in the given Java function with logging statements.",
            "Add log statements at crucial locations of the Java function to aid in tracking and debugging.",
            "Generate logs for this Java function to record important actions or variable values during its execution.",
            "Please embed appropriate logging language within this Java function to aid in monitoring its running state.",
            "Ensure the function's significant operation points are annotated with logging statements.",
            "Improvise this Java function by adding logging statements which provide insights about its inner workings.",
            "In this Java function, please incorporate logging statements to capture significant events during the execution."
        ]
        b = "Add newly created logging statements into the appropriate sections of the function and output the revised complete function."
        instruction = a + random.choice(instructions) + b
    
    return instruction


def get_one_data(row):
    task_name = row['task']
    input = row['code']
    output = row['label']

    instruction = get_one_instrucion(task_name)
    data = get_instruction_format(instruction, input)
    return data,output




datafile_path = "mixed_task_test_without_index.tsv"
df = pd.read_csv(os.path.join(datafile_path), sep='\t')

for index, row in df.iterrows():
    data = get_one_data(row)