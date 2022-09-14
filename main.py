import argparse

import pandas as pd
import pickle

from train import train_lda
from predict import predict_by_lda
from pred_evaluate import main
from utils.data_loader import load_data_and_vocab

parser = argparse.ArgumentParser(description='train.py',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opt = parser.parse_args()

opt.custom_vocab_filename_suffix = False
opt.vocab_size = 50000
opt.only_train_lda = True
opt.vocab = 'processed_data/StackExchange_s150_t10/'
opt.one2many = False
opt.custom_data_filename_suffix = False
opt.data = 'processed_data/StackExchange_s150_t10/'
opt.remove_src_eos = False
opt.batch_workers = 4
opt.batch_size = 64
opt.evaluate_coherence = False
opt.skip_saving_lda_result = True
opt.pred_path = 'pred/predict__'
opt.src = 'data/StackExchange/test_src.txt'
opt.trg = 'data/StackExchange/test_trg.txt'
opt.pred = 'pred/predict__/predictions.txt'
opt.export_filtered_pred = False
opt.num_preds = 50
opt.disable_valid_filter = False
opt.disable_extra_one_word_filter = True
opt.match_by_str = False
opt.target_separated = False
opt.exp_path = 'pred\\predict__'
opt.invalidate_unk = True


train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=True)

df = pd.DataFrame()

topics = [10, 20, 30, 40, 50, 60, 70, 80]
for topic in topics:
    opt.topic_num = topic
    print('train lda start')
    train_lda(opt, bow_dictionary, 1)
    print('training done')

    print('\n\n\npredict start')
    predict_by_lda(opt, bow_dictionary)
    print('predict done')

    print('\n\n\nevaluate start')
    field_list_samsung, result_list_samsung = main(opt)
    print('evaluate done')

    pair_result = zip(field_list_samsung, result_list_samsung)
    pair_result_dict = dict()
    for i, j in pair_result:
        pair_result_dict[i] = j
    result_series = pd.Series(pair_result_dict, name=str(opt.topic_num))
    df = df.append(result_series)
    pickle.dump(df, open('result_samsung.pkl', 'wb'))
    print(df)


pickle.dump(df.T, open('result_samsung.pkl', 'wb'))
print(df.T)
