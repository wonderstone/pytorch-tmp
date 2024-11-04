import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)  # 768 是 BERT base 模型的隐藏层大小

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTClassifier(num_classes=2)  # 假设是二分类任务

# 准备输入数据
text = "这是一个 BERT 模型的示例。"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 前向传播
outputs = model(inputs['input_ids'], inputs['attention_mask'])

print(outputs.shape)  # 输出形状应该是 [1, 2]，因为是二分类任务