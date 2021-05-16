import os

import numpy as np
import torch
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class Engine:
    def preprocess_entities_relations(self, entity_dict, relation_dict, entities, relations):
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

    def prepare_embeddings(self, embedding_dict):
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


    def process_text_file(self, text_file, split=False):
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


    def process_question(self, question):
        data_line = question
        data_array = []
        data_line = data_line.strip()
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1 + 'NE' + question_2
        ans = '1997'
        data_array.append([head, question.strip(), ans])
        
        return data_array


    def get_vocab(self,data):
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




    def __init__(self, embedding_dim=256, hidden_dim = 256, relation_dim = 200, 
           gpu = 0, use_cuda = False, freeze = 0,
           num_hops = 2, lr=0.0005, entdrop = 0.1, reldrop = 0.2, scoredrop = 0.2, l3_reg = 0.0,
           model_name = "ComplEx", decay = 1.0, ls = 0.0, vocab_size=84):

        hops = str(num_hops) + 'hop'
        data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '.txt'

        embedding_folder = '../../pretrained_models/embeddings/' + model_name + '_MetaQA_full'

        entity_path = embedding_folder + '/E.npy'
        relation_path = embedding_folder + '/R.npy'
        entity_dict = embedding_folder + '/entities.dict'
        relation_dict = embedding_folder + '/relations.dict'
        w_matrix = embedding_folder + '/W.npy'

        bn_list = []

        for i in range(3):
            bn = np.load(embedding_folder + '/bn' + str(i) + '.npy', allow_pickle=True)
            bn_list.append(bn.item())





        entities = np.load(entity_path)
        relations = np.load(relation_path)
        e, r = self.preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
        entity2idx, idx2entity, embedding_matrix = self.prepare_embeddings(e)
        data = self.process_text_file(data_path, split=False)
        word2idx, idx2word, max_len = self.get_vocab(data)
        device = torch.device(gpu if use_cuda else "cpu")
        # TODO(raresraf): Figure out vocab_size.
        model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size,
                                  num_entities=len(idx2entity), relation_dim=relation_dim,
                                  pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop=entdrop,
                                  reldrop=reldrop, scoredrop=scoredrop, l3_reg=l3_reg, model=model_name, ls=ls,
                                  w_matrix=w_matrix, bn_list=bn_list)
        model.load_state_dict(torch.load('../../checkpoints/MetaQA/2_hop/best_score_model_2.pt', map_location=torch.device('cpu')))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, decay)
        optimizer.zero_grad()
        model.eval()
        
        self.model = model
        self.word2idx = word2idx
        self.entity2idx = entity2idx
        self.idx2entity = idx2entity
        self.device = device
        self.model_name = model_name



    def data_generator(self,data, word2idx, entity2idx):
        for i in range(len(data)):
            data_sample = data[i]
            head = entity2idx[data_sample[0].strip()]
            question = data_sample[1].strip().split(' ')
            encoded_question = [word2idx[word.strip()] for word in question]
            if type(data_sample[2]) is str:
                ans = entity2idx[data_sample[2]]
            else:
                ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

            yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question, dtype=torch.long), ans, torch.tensor(
                len(encoded_question), dtype=torch.long), data_sample[1]

    def single_entry(self, question, device, model, word2idx, entity2idx, model_name):
        model.eval()
        # question is string
        data = self.process_question(question)
        answers = []
        data_gen = self.data_generator(data=data, word2idx=word2idx, entity2idx=entity2idx)
        total_correct = 0
        error_count = 0

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
        # print(question, pred_ans, ans)
        return pred_ans

    def answer(self, question):
        ans = self.single_entry(model=self.model, question=question, word2idx=self.word2idx,
           entity2idx=self.entity2idx, device=self.device, model_name=self.model_name)
        return self.idx2entity[ans]


engine = Engine()

print(engine.answer("which person directed the movies starred by [John Krasinski]"))


