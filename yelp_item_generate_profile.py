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

yelp_item_system_prompt2 = f"""You will serve as an assistant to help me analyze which types of users will like this business. \
You will be provided with USA business information: \
The information will be delimited with \
{delimiter} characters. \
the input will have five dimensions: name: the name of the business. \ 
categories: categories of the business.\
average_stars: average rating from users who have used this business.
city,and state: the business location \
requirements: 1.provide your answer in JSON format, following this structure:

    "short summarization":<what is this business for> \
    "who will probably like this business": <should contain at least 3 words, either extracted or predicted according to the provided information> \
    "reasoning":give explanation and reasoning> \

2. do not provide any other text outside the JSON string. \
3. remember to use the'[]'to include the file."""

def prompt_format(user_message):
    message = f"""
    <s>[INST] <<SYS>>
    {yelp_item_system_prompt2}
    <</SYS>>
    {delimiter}{user_message}{delimiter} 
    [/INST]
    """
    return message
def llama3prompt_format(user_message):
    message=f"""
    {yelp_item_system_prompt2}{delimiter}{user_message}{delimiter} 
    """


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
with open('yelp_chosen_item.txt', 'r') as file:
    lines = file.readlines()

# 提取每一行的内容
for line in lines[1:]:  # 跳过第一行，因为第一行是列名
    items = line.strip().split('\t')
    business_id = items[0]
    item_name = items[1]
    address = items[2]
    city = items[3]
    state = items[4]
    postal_code = items[5]
    latitude = float(items[6])
    longitude = float(items[7])
    item_stars = float(items[8])
    item_review_count = float(items[9])
    is_open = float(items[10])
    categories = items[11]
    # 输出所需的属性内容
    print(f"item_name: {item_name}, categories: {categories}, state: {state}, city: {city}")
    texts = f""" name:{item_name} average_stars:{item_stars} categories: {categories} city:{city} state:{state}\
            """
    input = prompt_format(texts)
    match = False
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

        if not match:  # 如果生成的文本没有匹配，重新生成
            print("Generated text does not match the pattern. Regenerating...")
            continue
        # 如果匹配到了文本，跳出内部循环并继续下一个行的处理
        break
# 将生成的文本保存为 pickle 文件
with open("yelp_llama3_item_profile.pkl", "wb") as f:
    pickle.dump(generated_texts, f)
