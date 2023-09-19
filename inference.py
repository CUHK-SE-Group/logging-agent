import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import pandas as pd
from read_data import get_one_prompt

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="instruct_logging/codellama7b", type=str, required=True)
    parser.add_argument('--in_file', default="task_data/mixtasks_test.tsv", type=str, required=True)
    parser.add_argument('--out_file', default="predictions/raw.tsv", type=str, required=True)
    args = parser.parse_args()

    generation_config = dict(
        temperature=0.5,
        top_k=30,
        top_p=0.6,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=128,
        num_return_sequences=5,
    )

    df = pd.read_csv(args.in_file, sep='\t')
    df.apply(
        lambda row: pd.Series(
            {'prompt': get_one_prompt(row), 'predict': ""}
        ), axis=1
    )

    # load_type = torch.float16
    load_type = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "right"

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, 
                                                 torch_dtype=load_type, 
                                                 config=model_config, 
                                                 device_map='auto')
    model.eval()

    print("Load model successfully")

    for index, row in tqdm(df.iterrows()):
        prompt = row['prompt']
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        generation_output = model.generate(
            input_ids = inputs["input_ids"],
            **generation_config
        )[0]

        pred_list = tokenizer.batch_decode(generation_output, 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False,
                                                )
        pred_str = ""
        for pred in pred_list:
            pred_str += pred.strip()
        row['predict'] = pred_str
        
    df.to_csv(args.out_file, sep='\t', index=False)

if __name__ == '__main__':
    
    infer()