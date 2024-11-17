from transformers import AutoModel, AutoTokenizer

model_name = 'cointegrated/LaBSE-en-ru'

# Скачиваем модель и токенайзер
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# # Сохраняем их в локальной папке
# tokenizer.save_pretrained('./LaBSE-en-ru')
# model.save_pretrained('./LaBSE-en-ru')