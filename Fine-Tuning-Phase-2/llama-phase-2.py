import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate

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

    def generate_summary(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(inputs['input_ids'], max_new_tokens=128)[0]
        summary = self.tokenizer.decode(output, skip_special_tokens=True)
        return summary

    def evaluate_rouge(self, dataset, num_samples=500):
        rouge = evaluate.load('rouge')
        original_model_outputs = []
        human_baseline_outputs = []

        for index in range(num_samples):
            prompt = dataset['input'][index]
            target = dataset['output'][index]

            generated_output = self.generate_summary(prompt)
            original_model_outputs.append(generated_output)
            human_baseline_outputs.append(target)

        rouge_results = rouge.compute(
            predictions=original_model_outputs,
            references=human_baseline_outputs
        )
        return rouge_results

def tokenize(examples, tokenizer):
    inputs = examples["input"]
    labels = examples["output"]
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt", max_length=1024)
    tokenized_labels = tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt", max_length=1024)
    return {
        'input_ids': tokenized_inputs.input_ids,
        'attention_mask': tokenized_inputs.attention_mask,
        'labels': tokenized_labels.input_ids
    }

def train_model(model, tokenizer, train_data, eval_data):
    tokenized_train_data = train_data.map(lambda examples: tokenize(examples, tokenizer), batched=True)
    tokenized_eval_data = eval_data.map(lambda examples: tokenize(examples, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        max_steps=1000,
        learning_rate=3e-3
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenizer.padding_side = 'right'

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        packing=True,
        dataset_text_field="id",
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_seq_length=1024,
    )

    trainer.train()

    model.save_pretrained('./Llama-2-7b-hf-cric-IPL')


if __name__ == "__main__":
    data_loader = DataLoader("./combined_playing_11_dataset")
    train_data = data_loader.get_dataframe('train')
    eval_data = data_loader.get_dataframe('eval')

    model_handler = ModelHandler("./Llama-2-7b-hf-cric-SFTT")

    # Random Text Generation
    index = 10
    prompt = train_data['input'][index]
    team = train_data['output'][index]

    generated_summary = model_handler.generate_summary(prompt)
    print(f"Randomly Generated Summary:\n{generated_summary}\n")

    # ROUGE Evaluation
    rouge_results = model_handler.evaluate_rouge(train_data)
    print('ROUGE Evaluation Results:')
    print(rouge_results)

    # Fine-Tuning
    train_model(model_handler.model, model_handler.tokenizer, train_data, eval_data)
