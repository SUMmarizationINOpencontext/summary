import json
import torch
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# Load the tokenizer and model for LLaMA
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Adjust based on model size you're using
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Increase model token size
tokenizer.model_max_length = 4096  # Set to larger token size for longer documents

# Load pre-trained LLaMA model
model = LlamaForCausalLM.from_pretrained(model_name)

# Prepare model for LoRA fine-tuning with int8 support
model = prepare_model_for_int8_training(model)

# Configure LoRA settings (low-rank adapters)
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Alpha
    target_modules=["q_proj", "v_proj"],  # Target projection layers
    lora_dropout=0.1,  # Dropout to avoid overfitting
    bias="none",  # No additional biases
    task_type="CAUSAL_LM"  # Causal Language Modeling (for summarization)
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Load and preprocess datasets
def preprocess_text(entry):
    # Combine all the text blocks into a single text input
    text = " ".join([entry.get('introduction', ''),
                     entry.get('methods', ''),
                     entry.get('results', ''),
                     entry.get('discussion', ''),
                     entry.get('conclusion', '')])
    return text

def prepare_datasets(input_file, summary_file):
    # Load input dataset and summary dataset
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    with open(summary_file, 'r') as f:
        summaries = json.load(f)

    # Create the dataset for fine-tuning
    data = []
    for pmid, article in articles.items():
        if pmid in summaries:
            text = preprocess_text(article)
            summary = summaries[pmid]['summary']
            data.append({"text": text, "summary": summary})

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    return dataset

# Load dataset
input_json = 'processed_articles.json'
summary_json = 'summary_dataset.json'
dataset = prepare_datasets(input_json, summary_json)

# Tokenization
def preprocess_batch(batch):
    inputs = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=4096, return_tensors="pt")
    labels = tokenizer(batch['summary'], padding='max_length', truncation=True, max_length=1024, return_tensors="pt")
    inputs['labels'] = labels['input_ids']
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_batch, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./llama-lora-summaries",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,  # Use FP16 for memory efficiency
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Optionally split eval set
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./llama-lora-summarization-model")
tokenizer.save_pretrained("./llama-lora-summarization-model")