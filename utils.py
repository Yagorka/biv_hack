from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
from tqdm.auto import tqdm
import re 
import numpy as np
import pandas as pd
import torch

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

nltk.download('stopwords')

# Получение русских стоп-слов
russian_stopwords = set(stopwords.words('russian'))
# Добавим специфичные для платежей стоп-слова
payment_stopwords = {'оплата', 'по', 'от', 'за', 'сумма', 'без', 'ндс', 'договор', 'счет', 'счёт'}
russian_stopwords.update(payment_stopwords)

id2label = {'LOAN': 0,
    'REALE_STATE': 1,
    'LEASING': 2,
    'NOT_CLASSIFIED': 3,
    'NON_FOOD_GOODS': 4,
    'FOOD_GOODS': 5,
    'BANK_SERVICE': 6,
    'SERVICE': 7,
    'TAX': 8}
label2id = {v:k for k, v in id2label.items()}

def normalize(word):
    return morph.parse(word)[0].normal_form

def clean_text(text):
    """Базовая очистка текста"""
    if isinstance(text, str):
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление цифр и специальных символов, оставляя пробелы между словами
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
        text = " ".join([normalize(word) for word in text.split() if word not in russian_stopwords and len(word) > 2])
        # Удаление множественных пробелов
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def preprocess_text(data, column_with_text):
    data[f'{column_with_text}'] = data[column_with_text].apply(clean_text)

def pad_texts(texts_tokenized):
    max_len = 0
    for i in texts_tokenized:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in texts_tokenized])
    return padded

def attention_mask(padded_text):
    return np.where(padded_text != 0, 1, 0)

def create_features(data, tokenizer, model, column_with_text, device, batch_size=16):
    X = data[column_with_text].values
    X_tokenized = [tokenizer.encode(x, max_length=512, truncation=True, add_special_tokens=True) for x in X]
    padded_test = pad_texts(X_tokenized)
    attention_mask_test = attention_mask(padded_test)


    model.eval()
    model.to(device)

    output_test = []
    for inx in range(0, attention_mask_test.shape[0], batch_size):
        batch = torch.tensor(padded_test[inx:inx+batch_size]).to(device)
        local_attention_mask = torch.tensor(attention_mask_test[inx:inx+batch_size]).to(device)

        with torch.no_grad():
            last_hidden_states = model(batch, attention_mask=local_attention_mask)[0][:, 0, :].cpu().numpy()
            output_test.append(last_hidden_states)

    test_features = np.vstack(output_test)
    return test_features

def predict(model, test_features):

    y_pred = model.predict(test_features)
    return y_pred

def create_answer_format(y_pred, test_data):

    df = pd.DataFrame({'labels': y_pred}, index=test_data.index.values)
    df['labels'] = df['labels'].apply(lambda x: label2id[x])

    return df



