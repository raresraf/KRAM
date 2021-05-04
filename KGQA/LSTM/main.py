import argparse
import os
import time

import numpy as np
import torch
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word_to_ix:
                idx2word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)

        length = len(sent.split())
        if length > maxLength:
            maxLength = length

    return word_to_ix, idx2word, maxLength


def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}

    f = open(entity_dict, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()

    f = open(relation_dict, 'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e, r


def validate(data_path, device, model, word2idx, entity2idx, model_name):
    model.eval()
    data = process_text_file(data_path)
    answers = []
    data_gen = data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    for i in tqdm(range(len(data))):
        try:
            d = next(data_gen)
            head = d[0].to(device)
            question = d[1].to(device)
            ans = d[2]
            ques_len = d[3].unsqueeze(0)
            tail_test = torch.tensor(ans, dtype=torch.long).to(device)
            top_2 = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()
            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]
            if type(ans) is int:
                ans = [ans]
            is_correct = 0
            # print(question, pred_ans, ans)
            if pred_ans in ans:
                total_correct += 1
                is_correct = 1
            q_text = d[-1]
            answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
        except:
            error_count += 1

    print(error_count)
    accuracy = total_correct / len(data)
    return answers, accuracy


def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()




def test(data_path, entity_path, relation_path, entity_dict, relation_dict, neg_batch_size, batch_size, shuffle,
          num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda, patience, freeze,
          validate_every, num_hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, w_matrix, bn_list,
          valid_data_path=None):
    entities = np.load(entity_path)
    relations = np.load(relation_path)
    e, r = preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    data = process_text_file(data_path, split=False)
    # data = pickle.load(open(data_path, 'rb'))
    word2ix, idx2word, max_len = get_vocab(data)
    hops = str(num_hops)
    # print(idx2word)
    # aditay
    # print(idx2word.keys())
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data=data, word2ix=word2ix, relations=r, entities=e, entity2idx=entity2idx)
    data_loader = DataLoaderMetaQA(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # TODO(raresraf): Figure out vocab_size.
    model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=84,
                              num_entities=len(idx2entity), relation_dim=relation_dim,
                              pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop=entdrop,
                              reldrop=reldrop, scoredrop=scoredrop, l3_reg=l3_reg, model=model_name, ls=ls,
                              w_matrix=w_matrix, bn_list=bn_list)
    model.load_state_dict(torch.load('../../checkpoints/MetaQA/best_score_model.pt', map_location=torch.device('cpu')))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0


    model.eval()
    eps = 0.0001
    answers, score = validate(model=model, data_path=valid_data_path, word2idx=word2ix,
                              entity2idx=entity2idx, device=device, model_name=model_name)
    print(score)


def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r', encoding='utf-8')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1 + 'NE' + question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split == False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data


def data_generator(data, word2ix, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1].strip().split(' ')
        encoded_question = [word2ix[word.strip()] for word in question]
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question, dtype=torch.long), ans, torch.tensor(
            len(encoded_question), dtype=torch.long), data_sample[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hops', type=str, default='1')
    parser.add_argument('--ls', type=float, default=0.0)
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--model', type=str, default='Rotat3')
    parser.add_argument('--kg_type', type=str, default='half')

    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--entdrop', type=float, default=0.0)
    parser.add_argument('--reldrop', type=float, default=0.0)
    parser.add_argument('--scoredrop', type=float, default=0.0)
    parser.add_argument('--l3_reg', type=float, default=0.0)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--shuffle_data', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--nb_epochs', type=int, default=90)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--neg_batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--relation_dim', type=int, default=30)
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--freeze', type=str2bool, default=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = parser.parse_args()

    hops = args.hops
    if hops in ['1', '2', '3']:
        hops = hops + 'hop'
    if args.kg_type == 'half':
        data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '_half.txt'
    else:
        data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '.txt'
    print('Train file is ', data_path)

    hops_without_old = hops.replace('_old', '')
    valid_data_path = '../../data/QA_data/MetaQA/qa_dev_' + hops_without_old + '.txt'
    test_data_path = '../../data/QA_data/MetaQA/qa_test_' + hops_without_old + '.txt'

    model_name = args.model
    kg_type = args.kg_type
    print('KG type is', kg_type)
    embedding_folder = '../../pretrained_models/embeddings/' + model_name + '_MetaQA_' + kg_type

    entity_embedding_path = embedding_folder + '/E.npy'
    relation_embedding_path = embedding_folder + '/R.npy'
    entity_dict = embedding_folder + '/entities.dict'
    relation_dict = embedding_folder + '/relations.dict'
    w_matrix = embedding_folder + '/W.npy'

    bn_list = []

    for i in range(3):
        bn = np.load(embedding_folder + '/bn' + str(i) + '.npy', allow_pickle=True)
        bn_list.append(bn.item())


    if args.mode == 'test':
        test(data_path=test_data_path,
              entity_path=entity_embedding_path,
              relation_path=relation_embedding_path,
              entity_dict=entity_dict,
              relation_dict=relation_dict,
              neg_batch_size=args.neg_batch_size,
              batch_size=args.batch_size,
              shuffle=args.shuffle_data,
              num_workers=args.num_workers,
              nb_epochs=args.nb_epochs,
              embedding_dim=args.embedding_dim,
              hidden_dim=args.hidden_dim,
              relation_dim=args.relation_dim,
              gpu=args.gpu,
              use_cuda=args.use_cuda,
              valid_data_path=valid_data_path,
              patience=args.patience,
              validate_every=args.validate_every,
              freeze=args.freeze,
              num_hops=args.hops,
              lr=args.lr,
              entdrop=args.entdrop,
              reldrop=args.reldrop,
              scoredrop=args.scoredrop,
              l3_reg=args.l3_reg,
              model_name=args.model,
              decay=args.decay,
              ls=args.ls,
              w_matrix=w_matrix,
              bn_list=bn_list)

    elif args.mode == 'eval':
        eval(data_path=test_data_path,
             entity_path=entity_embedding_path,
             relation_path=relation_embedding_path,
             entity_dict=entity_dict,
             relation_dict=relation_dict,
             model_path='../../checkpoints/MetaQA/best_score_model.pt',
             train_data=data_path,
             gpu=args.gpu,
             hidden_dim=args.hidden_dim,
             relation_dim=args.relation_dim,
             embedding_dim=args.embedding_dim)
