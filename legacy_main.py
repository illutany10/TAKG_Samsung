import argparse
import os
import pickle

import pandas as pd
import torch

import config
import predict
import train
import train_mixture
from pred_evaluate import main
from pykp.model import Seq2SeqModel, NTM
from utils.data_loader import load_data_and_vocab


def train_per_topic(tag, n_topic=50, timemark=None, model_path=None):
    parser = argparse.ArgumentParser(description='train.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.my_own_opts(parser)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()

    opt.gpuid = 0
    opt.device = torch.device("cuda:%d" % opt.gpuid)
    opt.topic_num = n_topic

    if tag == 'ntm':
        opt.only_train_ntm = True
        opt.ntm_warm_up_epochs = 1
    else:  # joint
        opt.copy_attention = True
        opt.use_topic_represent = True
        opt.load_pretrain_ntm = True
        opt.joint_train = True
        opt.topic_attn = True
        opt.only_train_ntm = False
        opt.check_pt_ntm_model_path = model_path
        if timemark:
            opt.timemark = timemark

    opt = train.process_opt(opt)
    train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
    word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=True)
    opt.bow_vocab_size = len(bow_dictionary)

    model = Seq2SeqModel(opt).to(opt.device)
    ntm_model = NTM(opt).to(opt.device)
    optimizer_seq2seq, optimizer_ntm, optimizer_whole = train.init_optimizers(model, ntm_model, opt)

    return train_mixture.train_model(model,
                                     ntm_model,
                                     optimizer_seq2seq,
                                     optimizer_ntm,
                                     optimizer_whole,
                                     train_data_loader,
                                     valid_data_loader,
                                     bow_dictionary,
                                     train_bow_loader,
                                     valid_bow_loader,
                                     opt)


def predict_per_topic(check_pt_model_path, check_pt_ntm_model_path):
    parser = argparse.ArgumentParser(description='predict.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.my_own_opts(parser)
    config.predict_opts(parser)
    config.vocab_opts(parser)

    opt = parser.parse_args()
    opt.model = check_pt_model_path
    opt.ntm_model = check_pt_ntm_model_path

    opt = predict.process_opt(opt)

    test_data_loader, word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=False)
    opt.bow_vocab_size = len(bow_dictionary)
    model, ntm_model = predict.init_pretrained_model(opt)
    predict.predict(test_data_loader, model, ntm_model, bow_dictionary, opt)


if __name__ == "__main__":
    df = pd.DataFrame()

    topics = [10, 20, 30, 40, 50, 60, 70, 80]

    for topic in topics:
        timemark, model_path = train_per_topic('ntm', n_topic=topic)
        check_pt_model_path, check_pt_ntm_model_path = train_per_topic('joint', n_topic=topic, timemark=timemark, model_path=model_path)
        # check_pt_model_path = 'model/StackExchange_s150_t10.joint_train.use_topic.topic_num50.topic_attn.no_topic_dec.copy.seed9527.emb150.vs50000.dec300.20220916-105751\\e3.val_loss=2.152.model-0h-06m'.replace('/', '\\')
        # check_pt_ntm_model_path = 'model/StackExchange_s150_t10.joint_train.use_topic.topic_num50.topic_attn.no_topic_dec.copy.seed9527.emb150.vs50000.dec300.20220916-105751\\e3.val_loss=2.152.model_ntm-0h-06m'.replace('/', '\\')
        predict_per_topic(check_pt_model_path, check_pt_ntm_model_path)

        # load settings for training
        parser = argparse.ArgumentParser(
            description='pred_evaluate.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        config.post_predict_opts(parser)
        opt = parser.parse_args()

        opt.src = 'data/StackExchange/test_src.txt'
        opt.trg = 'data/StackExchange/test_trg.txt'
        opt.pred = 'pred/predict__/predictions.txt'

        opt.exp_path = os.path.dirname(opt.pred)
        assert os.path.exists(opt.exp_path) is True

        opt.filtered_pred_path = opt.exp_path
        opt.export_filtered_pred = False
        opt.invalidate_unk = True
        opt.disable_extra_one_word_filter = True

        field_list_samsung, result_list_samsung = main(opt)

        pair_result = zip(field_list_samsung, result_list_samsung)
        pair_result_dict = dict()
        for i, j in pair_result:
            pair_result_dict[i] = j
        result_series = pd.Series(pair_result_dict, name=str(opt.topic_num))
        df = df.append(result_series)
        pickle.dump(df, open('result_legacy.pkl', 'wb'))
        print(df)

    pickle.dump(df.T, open('result_legacy.pkl', 'wb'))
    print(df.T)

