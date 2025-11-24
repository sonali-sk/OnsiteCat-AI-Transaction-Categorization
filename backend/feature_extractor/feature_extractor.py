import re
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
import pandas as pd
import os

# # base_path = os.path.dirname(os.path.abspath(__file__))  # path of your current script
# # model_dir = os.path.join(base_path, '..', 'checkpoint-774')  # adjust accordingly
# model_dir = '/Users/sonalisk/Projects/GHCI/hackathon/backend/feature_extractor/checkpoint-774'

# Initialize model and tokenizer once globally
label_list = ['O', 'B-MERCHANT', 'I-MERCHANT', 'B-CONTEXT', 'I-CONTEXT']
model_dir = './feature_extractor/checkpoint-774'  # Path to your saved model checkpoint

device = torch.device("cpu")

model = BertForTokenClassification.from_pretrained(model_dir, local_files_only=True)
tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
model.to(device)
model.eval()

def custom_tokenize(description):
    # Split by '/' for UPI-like strings, else split by whitespace
    return [t for t in re.split(r'[ /]+', description.strip()) if t]

def extract_entities_from_prediction(tokens, label_ids):
    def build_entity(prefixes):
        entity_tokens = []
        current_word = ''
        current_label_id = None
        i = 0
        while i < len(tokens):
            token = tokens[i]
            label_id = label_ids[i]
            label_name = label_list[label_id] if label_id != -100 else 'O'

            # Skip special tokens
            if token in tokenizer.all_special_tokens:
                if current_word:
                    entity_tokens.append(current_word)
                    current_word = ''
                current_label_id = None
                i += 1
                continue

            if not any(label_name.startswith(p) for p in prefixes):
                if current_word:
                    entity_tokens.append(current_word)
                    current_word = ''
                current_label_id = None
                i += 1
                continue

            # Start a new word
            current_word = token
            current_label_id = label_id
            current_label_name_prefix = label_name[:2]  # 'B-' or 'I-'
            i += 1

            # Merge consecutive B- tokens or following subwords starting with '##'
            while i < len(tokens):
                next_token = tokens[i]
                next_label_id = label_ids[i]
                next_label_name = label_list[next_label_id] if next_label_id != -100 else 'O'

                if next_label_name.startswith('B-') and next_label_id == current_label_id and current_label_name_prefix == 'B-':
                    current_word += next_token.lstrip('##')
                    i += 1
                elif next_token.startswith('##') and next_label_id == current_label_id and next_label_name.startswith(tuple(prefixes)):
                    current_word += next_token[2:]  # remove '##'
                    i += 1
                else:
                    break

            entity_tokens.append(current_word)
            current_word = ''

        return ' '.join(entity_tokens).strip()

    merchant = build_entity(['B-MERCHANT', 'I-MERCHANT'])
    context = build_entity(['B-CONTEXT', 'I-CONTEXT'])
    return merchant, context

def predict(text):
    pre_tokens = custom_tokenize(text)
    encodings = tokenizer(pre_tokens, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=512)

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_label_ids = torch.argmax(logits, axis=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    merchant, context = extract_entities_from_prediction(tokens, predicted_label_ids)

    return merchant, context


def predict_spark_df(df, input_column='description', merchant_column='extracted_merchant_name', context_column='context_feature'):
    df[[merchant_column, context_column]] = df[input_column].apply(
        lambda x: pd.Series(predict(x))
    )
    return df
