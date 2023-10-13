from transformers import AutoModelForCausalLM, AutoTokenizer

num_samples = 5

model_path = "instruct_logging/codellama7b"
print("loading model, path:", model_path)

model =  AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Instruction: ")
Instruction = input()
while Instruction:
        prompt = 'Instruction: ' + Instruction.strip() + '\n'
        print("Input: ")
        Input = input()
        prompt = prompt + 'Input: ' + Input.strip() + '\nOutput:'
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()
        Output = model.generate(input_ids, 
                                max_new_tokens=256, 
                                do_sample = True, 
                                top_k = 70, 
                                top_p = 0.9, 
                                temperature = 0.9, 
                                repetition_penalty=1., 
                                eos_token_id=2, 
                                bos_token_id=1, 
                                pad_token_id=tokenizer.pad_token_id, 
                                num_return_sequences=num_samples)
        rets = tokenizer.batch_decode(Output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(num_samples):
                print("Output:\n" + rets[i].strip().replace(prompt, ""))
        # print("Output:\n" + rets[0])
        print("Instruction: ")
        Instruction = input()
