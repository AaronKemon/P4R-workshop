from transformers import AutoTokenizer
from transformers import pipeline
import transformers
import torch
from datasets import load_dataset
import json
import pandas as pd
import gzip
import re
from data_process_user import flatten_data, save_result, save
import pickle

model_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
#model_checkpoint="meta-llama/Meta-Llama-3-8B-Instruct"
delimiter = "####"

amazon_item_system_prompt = f"""You will serve as an assistant to help me identifying specific features, themes, or gameplay mechanics that resonate most with different user segments. \
You will be provided with video games information: \
The information will be delimited with \
{delimiter} characters. \
the input will have five dimensions: name: the name of the game. \ 
categories: categories of the game. \
sales type: sales type of the game. \
brand:the brand of the game. \
requirements: 1.provide your answer in JSON format, following this structure:

    "short summarization":<what is this game about> \
    "who will probably like this game": <should contain at least 3 words, either extracted or predicted according to the provided information> \
    "reasoning":give explanation and reasoning> \

2. do not provide any other text outside the JSON string. \
3. remember to use the'[]'to include the file."""

def prompt_format(user_message):
    message = f"""
    <s>[INST] <<SYS>>
    {amazon_item_system_prompt}
    <</SYS>>
    {delimiter}{user_message}{delimiter} 
    [/INST]
    """
    return message

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token = True)

pipeline = transformers.pipeline(
    "text-generation",
    model = model_checkpoint,
    torch_dtype = torch.float16,
    device_map = "auto",
)
generated_texts = {}

#读取txt格式的数据集
# 读取txt文件
with open('amazon_chosen_item.txt', 'r') as file:
    lines = file.readlines()

# 提取每一行的内容
for line in lines[1:]:  # 跳过第一行，因为第一行是列名
    items = line.strip().split('\t')
    if len(items)>=5:
        business_id = items[0]
        item_name = items[1]
        categories = items[2]
        brand = items[3]
        sales_type = items[4]
    elif len(items)==4:
        business_id = items[0]
        item_name = items[1]
        categories = items[2]
        brand = items[3]
        sales_type = "unknown"
    elif len(items)==3:
        business_id = items[0]
        item_name = items[1]
        categories = items[2]
        brand="unkonwn"
        sales_type="unknown"
    else:
        business_id = items[0]
        item_name = items[1]
        categories = "unknown"
        brand="unkonwn"
        sales_type="unknown"        
        
    # 输出所需的属性内容
    print(f"item_name: {item_name}, categories: {categories}, sales_type: {sales_type}, brand: {brand}")
    texts = f""" item_name: {item_name}, categories: {categories}, sales_type: {sales_type}, brand: {brand} \
            """
    input = prompt_format(texts)
    match = False
    not_matched_count = 0
    while not match:
        sequences = pipeline(
            input,
            truncation=True,
            do_sample=True,
            num_beams=1,
            top_k=30,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=512,
        )

        for seq_index, seq in enumerate(sequences):
            generated_text = seq['generated_text']
            match = re.search(r'{(.*?)}', generated_text, re.DOTALL)
            if match:
                generated_texts.setdefault(business_id, []).append(match.group(1))
                print(f"This is {business_id} Result {seq_index + 1}: {match.group(1)}")
                break  # 找到匹配的文本后跳出循环

        if not match:  # If no match found
            not_matched_count += 1
            if not_matched_count > 3:
                print("Exceeded maximum attempts. Breaking out of the loop.")
                generated_texts.setdefault(business_id, []).append(texts)
                break  # Break out of the loop if exceeded maximum attempts
            else:
                print("Generated text does not match the pattern. Regenerating...")
        else:
        # Reset the counter if match is found
            not_matched_count = 0

    # Break out of the outer loop if match found or exceeded maximum attempts
        if match or not_matched_count > 5:
            break

# 将生成的文本保存为 pickle 文件
with open("amazon_llama3_item_profile.pkl", "wb") as f:
    pickle.dump(generated_texts, f)
