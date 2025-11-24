# categorizer.py

import torch
import joblib
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Load scaler and label encoder
scaler = joblib.load("categorization_model/amount_scaler.pkl")
le = joblib.load("categorization_model/label_encoder.pkl")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define model architecture
class TransactionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc_text = nn.Linear(self.bert.config.hidden_size, 128)
        self.fc_out = nn.Linear(128 + 1, num_classes)

    def forward(self, input_ids, attention_mask, amount):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = torch.relu(self.fc_text(pooled_output))
        x = self.dropout(x)
        if len(amount.shape) == 1:
            amount = amount.unsqueeze(1)
        x = torch.cat((x, amount), dim=1)
        logits = self.fc_out(x)
        return logits

# Initialize model and set eval mode
num_classes = len(le.classes_)
model = TransactionClassifier(num_classes)
model.load_state_dict(torch.load("categorization_model/transaction_category_model.pth"))
model.eval()
device = torch.device("cpu")
model.to(device)

def predict_category(merchant_name, context_feature, amount):
    text = (merchant_name or '') + ' ' + (context_feature or '')
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')

    amount_scaled = scaler.transform([[amount]])
    amount_scaled = torch.tensor(amount_scaled, dtype=torch.float)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    amount_scaled = amount_scaled.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, amount=amount_scaled)
    logits = outputs
    predicted_id = torch.argmax(logits, dim=1).item()

    return le.inverse_transform([predicted_id])[0]

def categorize_pandas_df(df, merchant_col='extracted_merchant_name', context_col='context_feature', amount_col='amount'):
    def apply_pred(row):
        return predict_category(row[merchant_col], row[context_col], row[amount_col])
    df['category'] = df.apply(apply_pred, axis=1)
    return df
