import logging
import math
import os
import sys
import time
import pickle

import gensim
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

from evaluate import evaluate_loss
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.time_log import time_since, convert_time2str

EPS = 1e-6


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))


def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)


def train_ntm_one_epoch(model, dataloader, optimizer, opt, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data_bow in enumerate(dataloader):
        data_bow = data_bow.to(opt.device)
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        _, _, recon_batch, mu, logvar = model(data_bow_norm)
        loss = loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_bow), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader),
                       loss.item() / len(data_bow)))

    logging.info('====>Train epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    sparsity = check_sparsity(model.fcd1.weight.data)
    logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    logging.info("Target sparsity = %.3f" % opt.target_sparsity)
    update_l1(model.l1_strength, sparsity, opt.target_sparsity)
    return sparsity


def test_ntm_one_epoch(model, dataloader, opt, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(opt.device)
            data_bow_norm = F.normalize(data_bow)

            _, _, recon_batch, mu, logvar = model(data_bow_norm)
            test_loss += loss_function(recon_batch, data_bow, mu, logvar).item()

    avg_loss = test_loss / len(dataloader.dataset)
    logging.info('====> Test epoch: {} Average loss:  {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True


def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc):  # 몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic, 4), doc]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return (topic_table)

def make_predict_word_top_j_k(opt, ldamodel, corpus):
    word_table = []

    tp = ldamodel.print_topics(num_words=10, num_topics=100)

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        words_for_doc=[]
        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc):  # 몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j < 3:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽 3개
                words_in_topic=tp[topic_num][1].split('+')
                for word in words_in_topic[:7]: #k=7, 7개까지 word
                    words_for_doc.append(word.split('*')[-1][1:-2])
            else:
                break
        word_table.append(';'.join(words_for_doc))

    pred_fn = os.path.join(opt.pred_path, 'predictions.txt')
    pred_output_file = open(pred_fn, "w", encoding='utf8')
    for pred_word in word_table:
        pred_output_file.write(pred_word + '\n')
    pred_output_file.close()

    return word_table


def train_model(model, ntm_model, optimizer_ml, optimizer_ntm, optimizer_whole, train_data_loader, valid_data_loader,
                bow_dictionary, train_bow_loader, valid_bow_loader, opt):
    logging.info('======================  Start Training  =========================')

    if opt.only_train_lda:
        if not opt.load_pretrain_lda:
            tokenized_doc = torch.load(opt.data + 'corpus.pt')
            corpus = [bow_dictionary.doc2bow(text) for text in tokenized_doc]
            # tokenized_doc_test = torch.load(opt.data + 'corpus_test.pt')
            # test_corpus = [bow_dictionary.doc2bow(text) for text in tokenized_doc_test]
            # corpus = corpus + test_corpus
            pickle.dump(corpus, open('gensim_corpus_corpus.pkl', 'wb'))
            n_topic = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            n_iter = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            max_topic_coherence = -100
            max_topic_coherence_cv = -100
            max_score_topic = -1
            max_score_topic_cv = -1
            model_per_topic = {}
            for n in n_topic:
                for iter in n_iter:
                    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=n, id2word=bow_dictionary, passes=15, iterations=iter)
                    cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
                    cm_c_v = CoherenceModel(model=lda_model, texts=tokenized_doc, coherence='c_v')
                    coherence = cm.get_coherence()
                    coherence_cv = cm_c_v.get_coherence()
                    perplexity = lda_model.log_perplexity(corpus)
                    print(f'# of Topic : {n}, Coherence : {coherence}, Perplexity : {perplexity}')
                    model_per_topic[str(n) + '_' + str(iter)] = (lda_model, coherence, coherence_cv, perplexity)
                    if max_topic_coherence < coherence:
                        max_topic_coherence = coherence
                        max_score_topic = n
                    if max_topic_coherence_cv < coherence_cv:
                        max_topic_coherence_cv = coherence_cv
                        max_score_topic_cv = n
                    pickle.dump(model_per_topic, open('model_per_topic.pkl', 'wb'))  # for Debug

            print(f'Max Score Topic (u_mass): {max_score_topic} ({max_topic_coherence}), Max Score Topic (CV): {max_score_topic_cv} ({max_topic_coherence_cv})')
            lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=max_score_topic, id2word=bow_dictionary, passes=15)
            lda_model.save('gensim_model.gensim')

            topic_table = make_topictable_per_doc(lda_model, corpus)
            topic_table = topic_table.reset_index()  # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
            topic_table.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토핑의 비중']
            pickle.dump(topic_table, open('gensim_topic_table.pkl', 'wb'))
        else:
            lda_model = gensim.models.ldamodel.LdaModel.load('gensim_model.gensim')
            corpus = pickle.load(open('gensim_corpus_corpus.pkl', 'rb'))
            topic_table = pickle.load(open('gensim_topic_table.pkl', 'rb'))

        print(topic_table[:10])
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, bow_dictionary)
        pyLDAvis.save_html(vis, './vis.html')
        return

    # lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)
    # lda_top = lda_model.fit_transform(X)
    # print(lda_model.components_)
    # print(lda_model.components_.shape)

    if opt.only_train_ntm or (opt.use_topic_represent and not opt.load_pretrain_ntm):
        print("\nWarming up ntm for %d epochs" % opt.ntm_warm_up_epochs)
        for epoch in range(1, opt.ntm_warm_up_epochs + 1):
            sparsity = train_ntm_one_epoch(ntm_model, train_bow_loader, optimizer_ntm, opt, epoch)
            val_loss = test_ntm_one_epoch(ntm_model, valid_bow_loader, opt, epoch)
            if epoch % 10 == 0:
                ntm_model.print_topic_words(bow_dictionary, os.path.join(opt.model_path, 'topwords_e%d.txt' % epoch))
                best_ntm_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntm_model' %
                                                   (epoch, val_loss, sparsity))
                logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
                torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))
    elif opt.use_topic_represent:
        print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))

    if opt.only_train_ntm:
        return

    total_batch = 0
    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    best_ntm_valid_loss = float('inf')
    joint_train_patience = 1
    ntm_train_patience = 1
    global_patience = 5
    num_stop_dropping = 0
    num_stop_dropping_ntm = 0
    num_stop_dropping_global = 0

    t0 = time.time()
    Train_Seq2seq = True
    begin_iterate_train_ntm = opt.iterate_train_ntm
    check_pt_model_path = ""
    print("\nEntering main training for %d epochs" % opt.epochs)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        if Train_Seq2seq:
            if epoch <= opt.p_seq2seq_e or not opt.joint_train:
                optimizer = optimizer_ml
                model.train()
                ntm_model.eval()
                logging.info("\nTraining seq2seq epoch: {}/{}".format(epoch, opt.epochs))
            elif begin_iterate_train_ntm:
                optimizer = optimizer_ntm
                model.train()
                ntm_model.train()
                fix_model(model)
                logging.info("\nTraining ntm epoch: {}/{}".format(epoch, opt.epochs))
                begin_iterate_train_ntm = False
            else:
                optimizer = optimizer_whole
                unfix_model(model)
                model.train()
                ntm_model.train()
                logging.info("\nTraining seq2seq+ntm epoch: {}/{}".format(epoch, opt.epochs))
                if opt.iterate_train_ntm:
                    begin_iterate_train_ntm = True

            logging.info("The total num of batches: %d, current learning rate:%.6f" %
                         (len(train_data_loader), optimizer.param_groups[0]['lr']))

            for batch_i, batch in enumerate(train_data_loader):
                total_batch += 1
                batch_loss_stat, _ = train_one_batch(batch, model, ntm_model, optimizer, opt, batch_i)
                report_train_loss_statistics.update(batch_loss_stat)
                total_train_loss_statistics.update(batch_loss_stat)

                if (batch_i + 1) % (len(train_data_loader) // 10) == 0:
                    print("Train: %d/%d batches, current avg loss: %.3f" %
                          ((batch_i + 1), len(train_data_loader), batch_loss_stat.xent()))

            current_train_ppl = report_train_loss_statistics.ppl()
            current_train_loss = report_train_loss_statistics.xent()

            # test the model on the validation dataset for one epoch
            model.eval()
            valid_loss_stat = evaluate_loss(valid_data_loader, model, ntm_model, opt)
            current_valid_loss = valid_loss_stat.xent()
            current_valid_ppl = valid_loss_stat.ppl()

            # debug
            if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                logging.info(
                    "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                exit()

            if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
                print("Valid loss drops")
                sys.stdout.flush()
                best_valid_loss = current_valid_loss
                best_valid_ppl = current_valid_ppl
                num_stop_dropping = 0
                num_stop_dropping_global = 0
                if epoch >= opt.start_checkpoint_at and epoch > opt.p_seq2seq_e and not opt.save_each_epoch:
                    check_pt_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.model-%s' %
                                                       (epoch, current_valid_loss, convert_time2str(time.time() - t0)))
                    # save model parameters
                    torch.save(
                        model.state_dict(),
                        open(check_pt_model_path, 'wb')
                    )
                    logging.info('Saving seq2seq checkpoints to %s' % check_pt_model_path)

                    if opt.joint_train:
                        check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                        # save model parameters
                        torch.save(
                            ntm_model.state_dict(),
                            open(check_pt_ntm_model_path, 'wb')
                        )
                        logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)
            else:
                print("Valid loss does not drop")
                sys.stdout.flush()
                num_stop_dropping += 1
                num_stop_dropping_global += 1
                # decay the learning rate by a factor
                for i, param_group in enumerate(optimizer.param_groups):
                    old_lr = float(param_group['lr'])
                    new_lr = old_lr * opt.learning_rate_decay
                    if old_lr - new_lr > EPS:
                        param_group['lr'] = new_lr
                        print("The new learning rate for seq2seq is decayed to %.6f" % new_lr)

            if opt.save_each_epoch:
                check_pt_model_path = os.path.join(opt.model_path, 'e%d.train_loss=%.3f.val_loss=%.3f.model-%s' %
                                                   (epoch, current_train_loss, current_valid_loss,
                                                    convert_time2str(time.time() - t0)))
                torch.save(  # save model parameters
                    model.state_dict(),
                    open(check_pt_model_path, 'wb')
                )
                logging.info('Saving seq2seq checkpoints to %s' % check_pt_model_path)

                if opt.joint_train:
                    check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                    torch.save(  # save model parameters
                        ntm_model.state_dict(),
                        open(check_pt_ntm_model_path, 'wb')
                    )
                    logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)

            # log loss, ppl, and time

            logging.info('Epoch: %d; Time spent: %.2f' % (epoch, time.time() - t0))
            logging.info(
                'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                    current_train_ppl, current_valid_ppl, best_valid_ppl))
            logging.info(
                'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                    current_train_loss, current_valid_loss, best_valid_loss))

            report_train_ppl.append(current_train_ppl)
            report_valid_ppl.append(current_valid_ppl)
            report_train_loss.append(current_train_loss)
            report_valid_loss.append(current_valid_loss)

            report_train_loss_statistics.clear()

            if not opt.save_each_epoch and num_stop_dropping >= opt.early_stop_tolerance:  # not opt.joint_train or
                logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)

                break

                # if num_stop_dropping_global >= global_patience and opt.joint_train:
                #     logging.info('Reach global stoping dropping patience: %d' % global_patience)
                #     break

                # if num_stop_dropping >= joint_train_patience and opt.joint_train:
                #     Train_Seq2seq = False
                # num_stop_dropping_ntm = 0
                # break

        # else:
        #     logging.info("\nTraining ntm epoch: {}/{}".format(epoch, opt.epochs))
        #     logging.info("The total num of batches: {}".format(len(train_bow_loader)))
        #     sparsity = train_ntm_one_epoch(ntm_model, train_bow_loader, optimizer_ntm, opt, epoch)
        #     val_loss = test_ntm_one_epoch(ntm_model, valid_bow_loader, opt, epoch)
        #     if val_loss < best_ntm_valid_loss:
        #         print('Ntm loss drops...')
        #         best_ntm_valid_loss = val_loss
        #         num_stop_dropping_ntm = 0
        #         num_stop_dropping_global = 0
        #     else:
        #         print('Ntm loss does not drop...')
        #         num_stop_dropping_ntm += 1
        #         num_stop_dropping_global += 1
        #
        #     if num_stop_dropping_global > global_patience:
        #         logging.info('Reach global stoping dropping patience: %d' % global_patience)
        #         break
        #
        #     if num_stop_dropping_ntm >= ntm_train_patience:
        #         Train_Seq2seq = True
        #         num_stop_dropping = 0
        #         # continue
        #
        #     if opt.joint_train:
        #         ntm_model.print_topic_words(bow_dictionary, os.path.join(opt.model_path, 'topwords_e%d.txt' % epoch))

    return check_pt_model_path


def train_one_batch(batch, model, ntm_model, optimizer, opt, batch_i):
    # train for one batch
    src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow = batch
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    # model.train()
    optimizer.zero_grad()

    if opt.use_topic_represent:
        src_bow = src_bow.to(opt.device)
        src_bow_norm = F.normalize(src_bow)
        if opt.topic_type == 'z':
            topic_represent, _, recon_batch, mu, logvar = ntm_model(src_bow_norm)
        else:
            _, topic_represent, recon_batch, mu, logvar = ntm_model(src_bow_norm)

        if opt.add_two_loss:
            ntm_loss = loss_function(recon_batch, src_bow, mu, logvar)
    else:
        topic_represent = None

    start_time = time.time()

    # for one2one setting
    decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _, _ \
        = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, topic_represent)

    forward_time = time_since(start_time)

    start_time = time.time()
    if opt.copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss)

    loss_compute_time = time_since(start_time)

    total_trg_tokens = sum(trg_lens)

    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print(src_oov)
        print(src_lens)
        print(src_mask)
        print("trg")
        print(trg)
        print(trg_oov)
        print(trg_lens)
        print(trg_mask)
        print("oov list")
        print(oov_lists)
        print("Decoder")
        print(decoder_dist)
        print(h_t)
        print(attention_dist)
        raise ValueError("Loss is NaN")

    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')

    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    if opt.add_two_loss:
        loss += ntm_loss
    # back propagation on the normalized loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()

    # construct a statistic object for the loss
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat, decoder_dist.detach()
