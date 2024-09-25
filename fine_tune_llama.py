import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model
import os

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device (optional)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Special Tokens
special_tokens = {
    'system': '<|system|>',
    'begin_context': '<|begincontext|>',
    'end_context': '<|endcontext|>',
    'begin_target': '<|begintarget|>',
    'end_target': '<|endtarget|>',
    'pad_token': '<|pad|>',
    'bos_token': '<|startoftext|>',
}

# List of additional special tokens (excluding pad and bos tokens)
additional_special_tokens = [v for k, v in special_tokens.items() if k not in ['pad_token', 'bos_token']]

# Step 1: Load the tokenizer and model with FP16 precision
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with the model you have access to

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add special tokens to the tokenizer
special_tokens_dict = {
    'additional_special_tokens': additional_special_tokens,
    'pad_token': special_tokens['pad_token'],
    'bos_token': special_tokens['bos_token'],
    'eos_token': special_tokens['end_target'],
}
tokenizer.add_special_tokens(special_tokens_dict)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # Automatically distribute the model across GPUs
    torch_dtype=torch.float16,
)

# Resize model embeddings to match the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Configure LoRA settings
lora_config = LoraConfig(
    r=16,                          # Rank for LoRA
    lora_alpha=32,                 # Alpha value
    target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"],  # Include embedding and LM head layers
    lora_dropout=0.1,              # Dropout rate
    bias="none",                   # No additional bias terms
    task_type="CAUSAL_LM"          # Task type: Causal Language Modeling
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Freeze base model parameters
for name, param in model.named_parameters():
    if 'lora' not in name and 'embed_tokens' not in name and 'lm_head' not in name:
        param.requires_grad = False

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Ensure model is in training mode
model.train()

# Print trainable parameters
model.print_trainable_parameters()

# Step 3: Preprocess the dataset
def preprocess_text(entry):
    # Combine text from different sections into one single input
    text = " ".join([
        entry.get('introduction', ''),
        entry.get('methods', ''),
        entry.get('results', ''),
        entry.get('discussion', ''),
        entry.get('conclusion', '')
    ])
    return text

def prepare_datasets(input_file, summary_file):
    # Load input articles
    with open(input_file, 'r') as f:
        articles = json.load(f)

    # Load summaries as a list
    with open(summary_file, 'r') as f:
        summaries_list = json.load(f)

    # Convert the summaries list into a dictionary with PMIDs as keys
    summaries = {str(item['pmid']): item for item in summaries_list}

    # Create lists to store the data
    texts = []
    summaries_data = []

    # Create a dataset for fine-tuning
    for pmid, article in articles.items():
        pmid_str = str(pmid)  # Ensure PMIDs are strings
        if pmid_str in summaries:
            text = preprocess_text(article)
            summary = summaries[pmid_str]['summary']
            texts.append(text)
            summaries_data.append(summary)
        else:
            print(f"PMID {pmid_str} not found in summaries.")

    # Check if data is empty
    if not texts:
        print("No matching PMIDs found between articles and summaries.")
        return None

    # Create a dictionary of lists
    data = {'text': texts, 'summary': summaries_data}

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data)
    return dataset

# Load your dataset
input_json = 'processed_articles.json'
summary_json = 'summary_output.json'
dataset = prepare_datasets(input_json, summary_json)

if dataset is None:
    print("Dataset is empty. Exiting.")
    exit()
else:
    print(f"Number of examples in the dataset: {len(dataset)}")
    print("First example:")
    print(dataset[0])

# Tokenization and preprocessing
def preprocess_batch(batch):
    # Define the system prompt
    system_prompt = "Find the most important sentences as if you were asked to find the summary of the text, find the most important sentences only."

    # Combine the system prompt, text, and summary into a single string
    combined_texts = [
        f"{special_tokens['system']}: {system_prompt}\n{special_tokens['begin_context']}{text}{special_tokens['end_context']}\n{special_tokens['begin_target']}{summary}{special_tokens['end_target']}"
        for text, summary in zip(batch['text'], batch['summary'])
    ]

    # Tokenize the combined texts
    tokenized_inputs = tokenizer(
        combined_texts,
        padding='max_length',
        truncation=True,
        max_length=2048,  # Adjust max_length based on your GPU memory
        return_tensors="pt",
    )

    # Prepare labels
    input_ids = tokenized_inputs["input_ids"]
    labels = input_ids.clone()

    # Replace padding token id's in labels with -100 to ignore in loss computation
    labels[labels == tokenizer.pad_token_id] = -100

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_batch, batched=True, remove_columns=['text', 'summary'])

# Verify the columns
print("Tokenized dataset columns:", tokenized_dataset.column_names)

# Step 4: Define the training arguments
training_args = TrainingArguments(
    output_dir="./llama-lora-summaries",
    per_device_train_batch_size=1,     # Adjust based on GPU memory
    gradient_accumulation_steps=16,    # Adjust to control effective batch size
    num_train_epochs=3,                # Set the number of training epochs
    logging_dir="./logs",
    logging_steps=500,
    save_strategy="epoch",             # Save at each epoch
    eval_strategy="epoch",             # Updated argument name
    fp16=True,                         # Use FP16 for memory efficiency
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_total_limit=2,                # Keep only the last 2 checkpoints
    report_to="none",                  # No reporting
    dataloader_num_workers=4,
    remove_unused_columns=False,       # Prevents the Trainer from dropping necessary columns
    ddp_find_unused_parameters=False,  # Important for LoRA with DDP
)

# Step 5: Set up the data collator
data_collator = default_data_collator

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # You can create a separate validation set if needed
    data_collator=data_collator
)

# Step 7: Fine-tune the model
trainer.train()

# Step 8: Save only LoRA weights
model.save_pretrained("./llama-lora-weights")

# Save tokenizer
tokenizer.save_pretrained("./llama-lora-weights")

print("LoRA weights and tokenizer saved successfully.")