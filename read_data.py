import os
import pandas as pd
import random

def get_instruction_format(instruction,input,output):
    str1 = "### Instruction: " + instruction + "\n"
    str2 = "### Input: " + input + "\n"
    str3 = "### Output: " + output + "\n"

    instruction_format = str1 + str2 + str3
    print(instruction_format)
    return instruction_format



def get_one_instrucion(task_name):
    if task_name == "task1":
        instructions = [
            "Assess if any additional logging is needed for this code snippet.",
            "Should more logging be added to this code?",
            "Is further logging necessary for this code?",
            "Analyze if there's a requirement for additional logging in this code.",
            "Would this code benefit from extra logging?",
            "Determine whether this code would benefit from supplementary logging.",
            "Is it necessary to include more logging in this code?",
            "Evaluate whether additional logging would be valuable for this code.",
            "Could this code be enhanced by extra logging?",
            "Scrutinize if there's a need to augment the logging in this code."
        ]
        instruction = random.choice(instructions)
    if task_name == "task2":
        instructions = [
            "Does this line require logging?",
            "Should logging be done at this line?",
            "Is this line a potential spot for a log entry?",
            "Should we consider adding a log here?",
            "Analyze if logging would be beneficial at this line.",
            "Is it appropriate to put log entries here?",
            "Do we require any log entries at this line of code?",
            "Identify if logging is necessary at this specific code line.",
            "Determine if a logging is required at this point of code.",
            "If a logging is needed, would this be the line to add it?"
        ]
        instruction = random.choice(instructions)
    if task_name == "task3":
        instructions = [
            "Determine the number of loggings this code should have.",
            "Calculate the quantity of logging operations necessary in this piece of coding.",
            "Establish how many loggings are needed for this piece of code.",
            "State the number of logging statements this code requires.",
            "Generate an instruction list specifying the number of logging statements needed for this code.",
            "Figure out the amount of logging necessary for this code.",
            "Discover how many times we need to log in this section of code.",
            "Determine the number of logging points that should be inserted into this code."
        ]
        instruction = random.choice(instructions)
    if task_name == "task4":
        instructions = [
            "Identify the specific line(s) in this code that necessitate logging.",
            "Determine which line(s) in this code should be logged.",
            "Spot the line(s) in this code that call for logging.",
            "Can you pinpoint the line(s) in this code that require logging?",
            "Highlight the line(s) in this code where logging is necessary.",
            "Which line(s) in this code should be specified for logging?",
            "Point out the line(s) in this code where logging is advisable.",
            "Pinpoint the line(s) in this code that require logging.",
            "In this code, which specific line(s) would necessitate logging?",
            "Can you determine which line(s) in this code should be logged?"
        ]
        instruction = random.choice(instructions)   
    if task_name == "task11":
        instructions = [
            "What logging statement(s) need to be added to this code snippet?",
            "Identify the particular logging statement(s) necessary for this code.",
            "Please provide the logging statement(s) that should be added to this code snippet.",
            "Decide on the particular logging statement(s) that should be incorporated into this piece of code.",
            "Point out the necessary logging statement(s) to enhance this code's function.",
            "What logging statement(s) should be inserted in this code?",
            "Can you identify the logging statement(s) needed in this code?",
            "Any specific logging needed for this code?",
            "What logging statement(s) do you suggest to be added to this section of code?",
            "What specific logging statement(s) do you think this code requires?"
        ]
        instruction = random.choice(instructions)
    if task_name == "task12":
        instructions = [
            "Provide the complete code added with the logging statement(s).",
            "Display the finalized version of the code with embedded logging operations.",
            "Could you provide the full code with incorporated logging?",
            "Generate the entirety of the code that includes the newly added logging statement(s).",
            "What does the completed code look like with the necessary logging statement(s) embedded?",
            "Will you be able to present the final code with the appropriate logging operations added?",
            "Show the completed code that integrates the necessary logging elements.",
            "Present the code in its full form, inclusive of the logging operations.",
            "Can you show the entirety of the code after adding in the log operations?",
            "Are you able to provide the completed code with the logging statement(s)?"
        ]
        instruction = random.choice(instructions)
    
    return instruction


def get_one_data(row, task_name):
    if task_name == "task1":
        input = row['code']
        output = row['label']
    if task_name == "task2":
        input = row['without_logging_code_index'] + "\t" + row['line_index']
        output = row['label']
    if task_name == "task3":
        input = row['code']
        output = row['label']
    if task_name == "task4":
        input = row['without_logging_code_index']
        output = row['label']

    if task_name == "task11":
        input = row['without_logging_code_index']
        output = row['logging_code_labels']
    if task_name == "task12":
        input = row['without_logging_code']
        output = row['full_code']
    instruction = get_one_instrucion(task_name)

    data = get_instruction_format(instruction, input, output)
    return data



datafile_path = "task1_train.tsv"
df = pd.read_csv(os.path.join(datafile_path), sep='\t')
for index, row in df.iterrows():
    data = get_one_data(row, "task1")