import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pandas as pd
from read_data import get_one_prompt



def get_task1_prompt(row):
    instruction = "Identify a prospective line index suitable for logging in the given code."
    input = row['code']
    str1 = "Instruction: " + instruction + "\n"
    str2 = "Input: " + input + "\n"
    str3 = "Output:" 

    prompt = str1 + str2 + str3
    return prompt



def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--model_name_or_path', default="instruct_logging/codellama7b_task1-5_7k/checkpoint-5555/", type=str)
    parser.add_argument('--in_file', default="task_data/task5_test.tsv", type=str)
    parser.add_argument('--out_file', default="result_file/predictions5_7k_2step.tsv", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.in_file, sep='\t')
    df = df.apply(
        lambda row: pd.Series(
            {'task': row['task'], 'prompt1': get_task1_prompt(row), 'code': row['code'], 'label': row['label'], 'predict1': "", 'predict2': "", 'predict': ""}
        ), axis=1
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    model.eval()

    print("Load model successfully")

    for i in tqdm(range(df.shape[0])):
        # substep1: infer line index and save in predict1
        prompt1 = df['prompt1'][i]
        input_ids = tokenizer(prompt1, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()
        pred = model.generate(input_ids, 
                              max_new_tokens=256, 
                              do_sample=False, 
                              eos_token_id=2, 
                              bos_token_id=1, 
                              pad_token_id=tokenizer.pad_token_id)
        rets = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets_list = ""    
        for j in range(args.num_samples):
            rets_list += rets[j].strip().replace(prompt1, "")
        df['predict1'][i] = rets_list

        # substep2: using predict1 to infer log statement and save in predict2
        instruction =  "Instruction: Insert a log statement for the code snippet at "+ df['predict1'][i] + "." + "\n"
        code = "Input: " + df['code'][i] + "\n"
        output = "Output:" 

        prompt2 = instruction + code + output
        input_ids = tokenizer(prompt2, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()
        pred = model.generate(input_ids, 
                              max_new_tokens=256, 
                              do_sample=False, 
                              eos_token_id=2, 
                              bos_token_id=1, 
                              pad_token_id=tokenizer.pad_token_id)
        rets = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets_list = ""    
        for j in range(args.num_samples):
            rets_list += rets[j].strip().replace(prompt2, "")
        df['predict2'][i] = rets_list

        # substep3: combine two results
        df['predict'][i] = df['predict1'][i] + " " + df['predict2'][i]

    
    if not os.path.exists("result_file"):
        os.mkdir("result_file")
    new_df =  df[['task','label','predict1', 'predict2', 'predict']]
    new_df.to_csv(args.out_file, sep='\t', index=False)

if __name__ == '__main__':
    
    infer()
