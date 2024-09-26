import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
from datasets import Dataset

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Special Tokens (same as in training)
special_tokens = {
    'system': '<|system|>',
    'begin_context': '<|begincontext|>',
    'end_context': '<|endcontext|>',
    'begin_target': '<|begintarget|>',
    'end_target': '<|endtarget|>',
    'pad_token': '<|pad|>',
    'bos_token': '<|startoftext|>',
}

additional_special_tokens = [v for k, v in special_tokens.items() if k not in ['pad_token', 'bos_token']]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./llama-lora-weights", padding_side="left")

# Ensure special tokens are added
special_tokens_dict = {
    'additional_special_tokens': additional_special_tokens,
    'pad_token': special_tokens['pad_token'],
    'bos_token': special_tokens['bos_token'],
    'eos_token': special_tokens['end_target'],
}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the base model with device_map="auto"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    # max_memory can be specified if needed
)

# Resize model embeddings to match the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Load the LoRA weights
model = PeftModel.from_pretrained(model, "./llama-lora-weights")
model.eval()  # No need to call model.to(device)

# Disable caching to reduce memory usage (if needed)
model.config.use_cache = False

# Function to preprocess text (same as in your training script)
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

# Load your dataset (adjust paths as needed)
input_json = 'processed_articles.json'
summary_json = 'summary_output.json'

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

    # Create a dataset for inference
    for pmid, article in articles.items():
        pmid_str = str(pmid)  # Ensure PMIDs are strings
        if pmid_str in summaries:
            text = preprocess_text(article)
            summary = summaries[pmid_str]['summary']
            texts.append(text)
            summaries_data.append(summary)
        else:
            print(f"PMID {pmid_str} not found in summaries.")

    # Create a dictionary of lists
    data = {'text': texts, 'summary': summaries_data}

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data)
    return dataset

# Prepare the dataset
dataset = prepare_datasets(input_json, summary_json)

# Choose an example text from the dataset
example_index = 2 # Change the index to test different examples
example = dataset[example_index]
text_to_summarize = example['text']

# Define the system prompt
system_prompt = "What is the most important sentence here? One only:"

# Prepare the input text with special tokens
input_text = (
    f"{special_tokens['system']}: {system_prompt}\n"
    f"{special_tokens['begin_context']}{text_to_summarize}{special_tokens['end_context']}\n"
    f"{special_tokens['begin_target']}"
)

# Tokenize the input text
input_ids = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=1024,  # Adjust based on your GPU memory
).input_ids

# Move input_ids to the first device used by the model
# Get the model's device map
device_map = model.hf_device_map
first_device = list(device_map.values())[0]

# Tokenize the input text
encoding = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=1024,  # Adjust based on your GPU memory
    padding=True,     # Ensure padding is added if needed
)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

# Move input_ids and attention_mask to the first device used by the model
input_ids = input_ids.to(first_device)
attention_mask = attention_mask.to(first_device)

# Generate the summary
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass the attention mask here
        max_new_tokens=1048,             # Adjust based on desired summary length
        temperature=0.9,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.convert_tokens_to_ids(special_tokens['end_target']),
        pad_token_id=tokenizer.pad_token_id,
    )

# Decode the generated tokens
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

# Extract the summary from the generated text
# Find the position of begin_target and end_target in the generated text
begin_target_token = special_tokens['begin_target']
end_target_token = special_tokens['end_target']

# Find the starting index of the summary
summary_start = generated_text.find(begin_target_token) + len(begin_target_token)
# Find the ending index of the summary
summary_end = generated_text.find(end_target_token, summary_start)

# Extract the summary text
if summary_end != -1:
    summary_text = generated_text[summary_start:summary_end]
else:
    summary_text = generated_text[summary_start:]

# Clean up the summary text
summary_text = summary_text.strip()

print("Generated Summary:")
print(summary_text)

print("Original summary:")
print(example["summary"])