import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset = load_dataset(dataset_path)

    def get_dataframe(self, split):
        return pd.DataFrame(self.dataset[split])

class ModelHandler:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def generate_summary(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=128)
        summaries = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return summaries

    def evaluate_rouge(self, dataset, num_shots=[0, 1, 2]):
        rouge_scores = {}
        for shots in num_shots:
            original_model_outputs = []
            human_baseline_outputs = []

            for index in range(50):
                prompt = dataset['input'][index]
                target = dataset['output'][index]

                if shots > 0:
                    prompt_subset = [prompt] * shots
                else:
                    prompt_subset = [""]

                generated_output = self.generate_summary(prompt_subset)
                original_model_outputs.append(generated_output[0])
                human_baseline_outputs.append(target)

            rouge = evaluate.load('rouge')
            rouge_results = rouge.compute(
                predictions=original_model_outputs,
                references=human_baseline_outputs
            )
            rouge_scores[f"{shots}-shot"] = rouge_results

        return rouge_scores

if __name__ == "__main__":
    data_loader = DataLoader("./combined_playing_11_dataset")
    test_data = data_loader.get_dataframe('eval')

    model_handler_1 = ModelHandler("./google-gemma-2b-it-IPL")
    model_handler_2 = ModelHandler("./Llama-2-7b-hf-cric-IPL")

    # ROUGE Evaluation
    rouge_scores_1 = model_handler_1.evaluate_rouge(test_data)
    for shots, scores in rouge_scores_1.items():
        print(f"Gemma {shots} Rogue:")
        print(scores)
    
    rouge_scores_2 = model_handler_2.evaluate_rouge(test_data)
    for shots, scores in rouge_scores_2.items():
        print(f"Llama {shots} Rogue:")
        print(scores)