import torch

import numpy as np

#from utils.data import *
from utils.load_embedding import *

import json

embed_dim = 100

def getIndex(document_list, true_pmid):
    i = 0
    flag = 0
    idxs = []
    for pmid, score in document_list:
        if pmid in true_pmid:
            idxs.append(i)
            flag = 1

        i += 1

    if flag == 0:
        return idxs.append(-1)

    return idxs



def evaluate(glove_model, query, document, query2document, model):

    average_rank = 0

    top1 = 0

    top10 = 0

    top50 = 0

    top100 = 0

    top200 = 0

    number = 0

    q2a_ranking = {}
    true_ranking = {}

    for query_id, true_document_ids in query2document.items():

        #query_id = random.choice(list(query.keys()))
        query_words = query[query_id]

        query_embed = getWeightMatrix(glove_model, query_words, embed_dim)

        document_candiates = {}

        for pmid, document_words in document.items():

            sim_matrix = torch.zeros(len(query_words), len(document_words), dtype=torch.float)
            document_embed = getWeightMatrix(glove_model, document_words, embed_dim)

            for i, query_word in enumerate(query_words):

                for j, document_word in enumerate(document_words, 0):
                    sim_matrix[i][j] = wordSimilarity(glove_model, query_word, document_word)

            if torch.cuda.is_available():
                sim_matrix = sim_matrix.cuda()

            score = model(query_embed, document_embed, sim_matrix, 0.5, 0.5).cpu()

            score_value = score.data.item()


            document_candiates[pmid] = score_value

        sorted_documents = sorted(document_candiates.items(), key=lambda x: x[1])

        q2a_ranking[query_id] = sorted_documents


        true_index = getIndex(sorted_documents, true_document_ids)

        true_ranking[query_id] = true_index

        print ("The true document is ranked as %dth in the list" % true_index[0])

        if true_index[0] < 1:
            top1 += 1
        if true_index[0] < 10:
            top10 += 1
        if true_index[0] < 50:
            top50 += 1
        if true_index[0] < 100:
            top100 += 1
        if true_index[0] < 200:
            top200 += 1

        average_rank += true_index[0]

        number += 1



    rr = []
    for true_index in true_ranking:
        tmp = 1.0/(true_index[0]+1)
        rr.append(tmp)
    MRR = np.mean(rr)

    pp = []
    for true_index in true_ranking:
        p = []
        for i, idx in enumerate(true_index):
            p.append((i+1)/float(idx+1))
        ap = np.mean(p)
        pp.append(ap)
    MAP = np.mean(pp)


    print("MAP: %f, MRR: %f"%(MAP,MRR))

    with open('../data/result_true_answer_ranking.json', 'w') as fwrite:
        json.dump(true_ranking, fwrite, indent=4)

    with open('../data/twitter_to_news_ranking.json', 'w') as fwrite:
        json.dump(q2a_ranking, fwrite, indent=4)



    average_rank = float(average_rank/number)


    print ("Top 1: %d, Top 10: %d, Top 50: %d, Top 100: %d, Top200: %d" %(top1, top10, top50, top100, top200))

    return average_rank











