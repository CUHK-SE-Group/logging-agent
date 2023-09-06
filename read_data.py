import os
import pandas as pd
import random

def get_instruction_format(instruction,input):
    str0 = "### System: You are a logging agent designed to identify or generate log statements for a given code snippet." + "\n"
    str1 = "### Instruction: " + instruction + "\n"
    str2 = "### Input: " + input + "\n"
    str3 = "### Output:" 

    instruction_format = str1 + str2 + str3
    # print(instruction_format)
    return instruction_format



def get_one_instrucion(task_name):
    if task_name == "task1":
        prefix = ""
        instructions = [
            "Please assess this code and determine whether more logging is required.",
            "Evaluate this piece of code to ascertain if additional logging is necessary.",
            "Analyze this code snippet and identify if further logging is needed.",
            "Examine this function and state whether it requires supplemental logging.",
            "Review this code and decide if extra logging would be beneficial.",
            "Inspect this function and confirm if more log statements should be added.",
            "Scrutinize this code to determine if the addition of more log statements is necessary.",
            "Please decide if additional logging is required for this code snippet.",
            "Check this code and confirm whether it needs more logging.",
            "Study this code and specify if it could use additional logging."
        ]
        suffix = " The expected output should be 'Yes' or 'No'."
        instruction = prefix + random.choice(instructions) + suffix
    if task_name == "task3":
        prefix = ""
        instructions = [
            "Review this function and indicate the necessary number of log statements.",
            "Considering this code segment, provide the count of log statements that should be created.",
            "Analyze this function and identify the needed amount of log statements.",
            "Estimate the number of log statements essential for this code.",
            "For this function, specify the count of the log statements required.",
            "Study this code segment and determine the necessary number of logging instances.",
            "Scrutinize this function and suggest how many log statements are needed.",
            "Inspect this piece of code and specify the number of logs to be added.",
            "Identify the number of log statements required for this function.",
            "Given this code snippet, estimate the needed count of log instances."
        ]
        suffix = " The expected output should be an integer."
        instruction = prefix + random.choice(instructions) + suffix
    if task_name == "task5":
        prefix = "Log levels typically include 'fatal','error', 'warn', 'info', 'debug', and 'trace'. "
        instructions = [
            "In this function, please replace each 'UNKNOWN' log level with the corresponding log level based on its severity.",
            "Assess the function and allocate correct log levels to the log statements.",
            "Please review the function, then determine and assign the appropriate log levels to each log statement.",
            "Analyze the function, find all log statements labeled as 'UNKNOWN' and replace them with correct log levels.",
            "Identify each 'UNKNOWN' log level in the function and replace with suitable log levels based on their severity.",
            "Examine the function and update 'UNKNOWN' log levels with the correct ones for each log statement.",
            "Carefully review the function and ensure all the log statements are assigned with the correct log levels.",
            "Assign appropriate log levels to each log statement in the provided function, replacing 'UNKNOWN' log levels.",
            "Inspect the given function and adjust 'UNKNOWN' log labels, substituting them with the determined log levels.",
            "For each log statement in the function, replace the 'UNKNOWN' log level with the correct severity level."
        ]
        suffix = " The expected output should be denoised log levels separated by commas."
        instruction = prefix + random.choice(instructions) + suffix
    if task_name == "task6":
        prefix = "Log levels typically include 'fatal','error', 'warn', 'info', 'debug', and 'trace'. "
        instructions = [
            "Rectify any log level inaccuracies in the given function, and provide the corrected version.",
            "Please scrutinize the log levels in the function and ensure their appropriateness; make adjustments as necessary.",
            "Evaluate the log levels used in the function and correct them as required.",
            "Assess and rectify any inaccuracy in log levels present in the function.",
            "Investigate the function for any inaccurate log levels and remedy them accordingly.",
            "Examine the function and modify any misused log levels you find.",
            "Ensure the function uses the correct log levels; rectify any errors you encounter.",
            "Analyze and correct any inaccurate log levels in the function.",
            "Review and amend any incorrect log levels within the function.",
            "Look through the function, spot any false log levels, and correct them."
        ]
        suffix = " The expected output should be the revised log levels separated by commas."
        instruction = prefix + random.choice(instructions) + suffix  
    if task_name == "task7":
        prefix = ""
        instructions = [
            "Please interpret the 'UNKNOWN' log messages in the provided function.",
            "Based on the context within the function, infer the hidden log message masked as 'UNKNOWN'.",
            "Analyze the code and estimate the concealed log message originally labeled as 'UNKNOWN'.",
            "From the given function, please deduce the actual content of the log message that has been masked as 'UNKNOWN'.",
            "Evaluate the code, and make an appropriate estimate of the hidden log message.",
            "Based on your understanding of the given code, determine what the hidden log message could be.",
            "Suppose that the 'UNKNOWN' represents a log message in the function. Predict its actual content.",
            "Please fill in the hidden information from the 'UNKNOWN' log message in the function.",
            "Try to comprehend the given function, and suggest possible content for the masked log messages.",
            "Analyze and determinate the appropriate message for the concealed log statements labeled as 'UNKNOWN' in the function."
        ]
        suffix = " The expected output should be the denoised log messages separated by '<DIV>'."
        instruction = prefix + random.choice(instructions) + suffix
    if task_name == "task8":
        prefix = ""
        instructions = [
            "Scrutinize the provided code snippet to identify and eliminate any redundant or irrelevant log statements.",
            "Examine the function for any log messages that do not add value or relate to its purpose and remove them.",
            "Assess whether the code snippet contains any unnecessary or unrelated logs. If so, remove them.",
            "Inspect the code and identify any log messages that seem unessential or unrelated to the function's main task. Proceed to remove those.",
            "Look through the code to find any log entries that may be redundant or don't provide meaningful information, and remove them.",
            "Study the code carefully and eliminate any log statements that are not directly relevant to the function's operation.",
            "Go through the function, spot any irrelevant or unnecessary log entries, and remove them.",
            "Scan the function for any extraneous or unrelated log messages, and if found, remove them.",
            "Analyze the code, identify any irrelevant or superfluous log statements, and ensure they are deleted.",
            "Peruse the provided function, find any pointless or unrelated log messages, and take steps to remove them.",
        ]
        suffix = " The expected output should be the revised function with redundant log statements removed."
        instruction = prefix + random.choice(instructions) + suffix
    if task_name == "task10":
        prefix = ""
        instructions = [
            "Incorporate important log entries into this function to keep track of its operation.",
            "Execute log statements at critical points in the function to keep track of its progress.",
            "Add meaningful logs in this function for potential debugging or execution tracing.",
            "Mark significant actions in the given function with log statements.",
            "Add log statements at crucial locations of the function to aid in tracking and debugging.",
            "Generate logs for this function to record important actions or variable values during its execution.",
            "Please embed appropriate logging language within this function to aid in monitoring its running state.",
            "Ensure the function's significant operation points are annotated with log statements.",
            "Improvise this function by adding log statements which provide insights about its inner workings.",
            "In this function, please incorporate log statements to capture significant events during the execution."
        ]
        suffix = " The expected output should be the revised function with log statements added."
        instruction = prefix + random.choice(instructions) + suffix
    if task_name == "task12":
        prefix = "The input are several log statements (post '<LOG>', split by '<DIV>') and a function (post '<CODE>'). "
        instructions = [
            "Incorporate the given log statements into the function at appropriate positions for best readability and debugging effectiveness.",
            "Adjust the structure of the function by positioning the provided log statements at their appropriate locations.",
            "Based on the given log data, identify the best locations within the function to incorporate each log statement.",
            "Fill the missing log statements back into the function where they are most logical and make the most sense.",
            "Insert the log statements back into the correct positions of the provided function to ensure the function can be properly traced and debugged.",
            "Evaluate the function and insert the given log statements in the most suitable positions.",
            "Place each log statement back into the function where it is most relevant for monitoring the function's activities.",
            "Analyze both the function and log statements, and position each log accordingly to improve the overall quality of the function.",
            "Decipher the function and place the log lines back into their most fitting positions.",
            "Restore the log entries in the provided function at positions where they are most informative and useful."
        ]
        suffix = " The expected output should be the revised function with log statements added."
        instruction = prefix + random.choice(instructions) + suffix
    
    return instruction


def get_one_data(row):
    task_name = row['task']
    input = row['code']
    output = row['label']

    instruction = get_one_instrucion(task_name)
    data = get_instruction_format(instruction, input)
    return data, output

# unit test
if __name__ == "__main__":
    datafile_path = "task_data/subtasks_test_without_index.tsv"
    df = pd.read_csv(os.path.join(datafile_path), sep='\t')
    row = df.iloc[0]
    data, label = get_one_data(row)
    print(data, label)