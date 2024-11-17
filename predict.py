import os
import pickle
import glob
import pandas as pd
import torch
import transformers as ppb

from utils import create_features, preprocess_text, predict, create_answer_format
# from consumer_wrapper import track

# @track
def main():
    try:
        name_tsv_file = glob.glob('data/*.tsv')[0]
        print(f"Файл {name_tsv_file} будет загружен")
    except Exception as ex:
        print(ex)
        print('Загрузите Ваш .tsv файл в папку data')

    device = torch.device('cpu') #torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model_class, tokenizer_class, pretrained_weights = (ppb.AutoModel, ppb.AutoTokenizer, "cointegrated/LaBSE-en-ru")

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    column_with_text, columns_with_category = 'text', 'category'
    try:
        test_data = pd.read_csv(name_tsv_file, index_col=0, sep='\t', names = ['number',  'date',  'sum', column_with_text, columns_with_category])
        print(f"Файл {name_tsv_file} загружен")
    except Exception as ex:
        print(ex)
        print('Загрузите Ваш .tsv файл в правильном формате')
    

    preprocess_text(test_data, column_with_text)

    test_features = create_features(test_data, tokenizer, model, column_with_text, device, batch_size=16)

    # load the model 
    load_model = pickle.load(open(os.path.join('models', 'linear_model.sav'), 'rb')) 

    y_pred = predict(load_model, test_features)

    df = create_answer_format(y_pred, test_data)

    df.to_csv(os.path.join('data', 'result.tsv'), header=False, sep='\t')
if __name__ == "__main__":
    main()

   






