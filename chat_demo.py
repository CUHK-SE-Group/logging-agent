from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "tuned_model/codellama7b"
print("loading model, path:", model_path)

model =  AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("### Instruction: ")
Instruction = input()
while Instruction:
        prompt = '### Instruction: ' + Instruction.strip() + '\n'
        print("### Input: ")
        Input = input()
        prompt = prompt + '### Input: ' + Instruction.strip() + '\n### Output:'
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()
        Output = model.generate(input_ids, max_new_tokens=256, do_sample = True, top_k = 30, top_p = 0.6, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=32016)
        rets = tokenizer.batch_decode(Output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("### Output:\n" + rets[0].strip().replace(prompt, ""))
        print("### Instruction: ")
        Instruction = input()