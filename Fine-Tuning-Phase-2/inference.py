import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM


# Generating Random Text from the Dataset
tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-hf-cric-IPL")
model = AutoModelForCausalLM.from_pretrained(
    "./Llama-2-7b-hf-cric-IPL",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

index = 10
dash_line = '-' * 100

prompt = '''Generate the combined playing-11 from the following squads:

Mumbai Indians Squad 2024:\nRohit Sharma, Jasprit Bumrah, Suryakumar Yadav, Ishan Kishan, Dewald Brevis, Tilak Varma, Hardik Pandya, Tim David, Arjun Tendulkar, Kumar Kartikeya, Akash Madhwal, Vishnu Vinod, Romario Shepherd, Shams Mulani, Nehal Wadhera, Piyush Chawla, Gerald Coetzee, Shreyas Gopal, Nuwan Thushara, Naman Dhir, Anshul Kamboj, Mohammad Nabi, Luke Wood

Royal Challengers Bengaluru Squad 2024:\nFaf du Plessis, Virat Kohli, Glenn Maxwell, Mohammed Siraj, Dinesh Karthik, Cameron Green, Vyshak Vijaykumar, Manoj Bhandage, Rajat Patidar, Anuj Rawat, Suyash Prabudessai, Akash Deep, Reece Topley, Rajan Kumar, Himanshu Sharma, Karn Sharma, Mahipal Lomror, Will Jacks, Alzarri Joseph, Yash Dayal, Tom Curran, Lockie Ferguson, Swapnil Singh, Saurav Chauhan

Match Information:
Venue: Wankhede Stadium, Mumbai
Toss: Mumbai Indians won the toss and elected to field'''

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

output = model.generate(inputs['input_ids'], max_new_tokens=128)[0]
original_model_summary = tokenizer.decode(output, skip_special_tokens=True)

print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{original_model_summary}\n')