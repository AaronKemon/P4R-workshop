import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# 可选：如果您想了解正在发生的事情的更多信息，请点击以下方式激活记录器
import logging
import pandas as pd
import pickle
logging.basicConfig(level = logging.INFO)
# 定义一个自定义的全连接层来修改输出维度
class CustomFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomFC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

# 加载预训练的模型标记器（词汇表）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df=pd.read_pickle('.pkl')
# 加载预训练模型（权重）
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # 模型是否返回所有隐状态。
                                  )
encoded_texts = {}
desired_dim=64
custom_fc_layer = CustomFC(model.config.hidden_size, desired_dim)
#这部分用来将生成的itemprofile编码

for index in df.keys():
    #count+=1
    #print(f"Key: {index}, Value: {df[index]}")
    business_id=index
    text=str(df[index]).strip()
    #print(f"Key: {index}, Value: {text}")
    # 标记化文本并添加特殊标记
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # 将输入转换为 PyTorch 张量
    tokens_tensor = torch.tensor([indexed_tokens])

    # 运行 BERT 模型以获得隐藏状态
    with torch.no_grad():
        outputs = model(tokens_tensor)
        encoded_layers = outputs[0]  # 只获取最后一层的隐藏状态

    # 将隐藏状态进行汇总，例如通过平均池化或取CLS标记的向量
    # 这里我们简单地使用[CLS]标记的向量作为编码结果
    encoded_text = encoded_layers[:, 0, :]  # 获取[CLS]标记的向量
    # 使用自定义全连接层修改输出维度
    modified_embedding = custom_fc_layer(encoded_text)

    # 将修改后的嵌入保存到字典中
    encoded_texts[business_id] = modified_embedding
# 将编码结果保存为 pickle 文件
with open('yelp_chosen_item_emb_llama_13b_64.pkl', 'wb') as f:
    pickle.dump(encoded_texts, f)

