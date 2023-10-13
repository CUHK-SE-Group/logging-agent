def get_prompt(instruction, input):
    str1 = "Instruction: " + instruction + "\n"
    str2 = "Input: " + input + "\n"
    str3 = "Output:" 

    prompt = str1 + str2 + str3
    return prompt

def get_instrucion(task_name, lineID=None):
    instruction = ""
    if task_name == "task0":
        instruction = "Find the logging statement with its line index ahead in the given code."
    # predict line ID of logging statement needed
    if task_name == "task1":
        instruction = "Identify a prospective line index suitable for logging in the given code."
    # predict log level
    if task_name == "task2":
        instruction = "Assign an appropriate log level to replace the 'UNKNOWN' tag at {}.".format(lineID)
    # predict log message
    if task_name == "task3":
        instruction = "Infer the missing content of the log message that has been masked as 'UNKNOWN' at {}.".format(lineID)
    # given pos, generate logging statement (level,msg)
    if task_name == "task4":
        instruction = "Insert a log statement for the code snippet at {}.".format(lineID)
    # generate logging statement (pos,level,msg)
    if task_name == "task5":
        instruction = "Generate a complete log statement with an appropriate line index ahead for the given input code."
    return instruction


def read_data(row):
    task_name = row['task']
    input = row['code']
    output = row['label']
    lineID = row['lineID']

    prompt = []
    for i in range(len(input)):
        prompt.append(get_prompt(get_instrucion(task_name[i], lineID[i]), input[i]))
    
    return prompt, output


def get_one_prompt(row):
    task_name = row['task']
    input = row['code']
    lineID = row['lineID']

    prompt = get_prompt(get_instrucion(task_name, lineID), input)
    return prompt
