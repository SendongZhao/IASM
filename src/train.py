import torch
import math

import random
from loss import *
#from utils.data import *
from utils.load_embedding import *


embed_dim = 100


def train(glove_model, query, document, query2document_true, query2document_false, model, optimizer, criterion, alpha, beta):
    """

    :type optimizer: object
    """
    model.train()

    optimizer.zero_grad()
    epoch_loss = 0

    model.train()

    #query2document = data.getQuery2Document()
    #query = data.getQuery()
    #document = data.getDocument()

    counter = 0

    for query_id, document_ids in query2document_true.items():

        if query_id not in query2document_false:
            fake_document_ids = random.choice(list(query2document_false.values()))
        else:
            fake_document_ids = query2document_false[query_id]


        query_words = query[query_id]

        for document_id in document_ids:

            document_words = document[document_id]
            fake_document_id = random.choice(fake_document_ids)
            fake_document_words = document[fake_document_id]

            document_embed = getWeightMatrix(glove_model, document_words, embed_dim)
            fake_document_embed = getWeightMatrix(glove_model, fake_document_words, embed_dim)

            query_embed = getWeightMatrix(glove_model, query_words, embed_dim)

            sim_matrix = torch.zeros(len(query_words), len(document_words), dtype=torch.float).cuda()
            fake_sim_matrix = torch.zeros(len(query_words), len(fake_document_words), dtype=torch.float).cuda()

            for i, query_word in enumerate(query_words):

                for j, document_word in enumerate(document_words, 0):
                    sim_matrix[i][j] = wordSimilarity(glove_model, query_word, document_word)

                for j, fake_word in enumerate(fake_document_words, 0):
                    fake_sim_matrix[i][j] = wordSimilarity(glove_model, query_word, fake_word)


            sim_matrix = sim_matrix.cuda()
            fake_sim_matrix = fake_sim_matrix.cuda()

            #print ("Distance between query and true document")

            dist_true = model(query_embed, document_embed, sim_matrix, alpha, beta)
            #print ("True Distance:")
            #print (dist_true)

            #print ("Distance between query and fake document")
            dist_fake = model(query_embed, fake_document_embed, fake_sim_matrix, alpha, beta)
            #print ("Fake Distance:")
            #print (dist_fake)

            #print ("train loss computation")
            train_loss = criterion(dist_true, dist_fake, 0)

            train_loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            counter += 1

            epoch_loss += train_loss.item()
        

    return epoch_loss/float(counter)
