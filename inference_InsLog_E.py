import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pandas as pd
from read_data import get_one_prompt

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--model_name_or_path', default="instruct_logging/codellama7b_task1-5/", type=str)
    parser.add_argument('--in_file', default="task_data/task5_test.tsv", type=str)
    parser.add_argument('--out_file', default="result_file/predictions5_E.tsv", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.in_file, sep='\t')
    df = df.apply(
        lambda row: pd.Series(
            {'task': row['task'], 'prompt': get_one_prompt(row), 'label': row['label'], 'predict': ""}
        ), axis=1
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    model.eval()

    print("Load model successfully")

    for i in tqdm(range(df.shape[0])):
        prompt = df['prompt'][i]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
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
            rets_list += rets[j].strip().replace(prompt, "")
        df['predict'][i] = rets_list
    
    if not os.path.exists("result_file"):
        os.mkdir("result_file")
    df.to_csv(args.out_file, sep='\t', index=False)

if __name__ == '__main__':
    
    infer()
