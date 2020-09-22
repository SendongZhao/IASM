from utils.load_embedding import GloveVector
from train import *
import loss
from model import *

from evaluate import *

import torch
import torch.optim as optim

import time
import os

#from utils.data import *
from utils.load_embedding import *

from args import get_args

import json


#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def epoch_time(start_time, end_time):
    temp = end_time - start_time

    minutes = temp//60

    seconds = temp - 60*minutes

    return minutes, seconds



if __name__ == "__main__":

    print ("Start ...")

    args = get_args()

    IN_SIZE = args.input_size
    HIDDEN_SIZE = args.hidden_size
    OUT_SIZE = args.output_size
    DROPOUT = args.dropout
    N_EPOCHS = args.epochs
    Alpha = args.alpha
    Beta = args.beta

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print("Note: You are using GPU for training")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)


    model = Scoring(IN_SIZE, HIDDEN_SIZE, OUT_SIZE, DROPOUT)

    if torch.cuda.is_available():
        print ("There is cuda !")

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    #optimizer = optim.SGD(model.parameters(), lr = args.lr)

    criterion = loss.marginLoss()
    criterion = criterion.cuda()

    print ("Read GloVe Embedding ...")

    glove_model = GloveVector("../GloVe/glove.6B.100d.txt")
    glove_model.loadGloveModel()

    print ("Read Data ...")


    fread = open('../data/id2question.json')
    query_set = json.load(fread)
    fread.close()

    fread = open('../data/id2answer.json')
    document_set = json.load(fread)
    fread.close()

    fread = open('../data/q_a2label_train.json')
    train_pairs = json.load(fread)
    fread.close()

    fread = open('../data/q_a2label_test.json')
    test_pairs = json.load(fread)
    fread.close()

    train_true_samples = {}
    train_false_samples = {}

    test_true_samples = {}
    test_false_samples ={}

    test_document_set = {}

    for p, label in train_pairs.items():
        question_id, answer_id = p.split()

        if label == '0' or label == 0:
            if question_id not in train_false_samples:
                train_false_samples[question_id] = []

            train_false_samples[question_id].append(answer_id)


        elif label == '1' or label == 1:
            if question_id  not in train_true_samples:
                train_true_samples[question_id] = []

            train_true_samples[question_id].append(answer_id)

    for p, label in test_pairs.items():
        question_id, answer_id = p.split()

        if label == '0' or label == 0:
            if question_id not in test_false_samples:
                test_false_samples[question_id] = []

            test_false_samples[question_id].append(answer_id)
            #test_document_set[answer_id] = document_set[answer_id]

        elif label == '1' or label == 1:
            if question_id not in test_true_samples:
                test_true_samples[question_id] = []

            test_true_samples[question_id].append(answer_id)
            test_document_set[answer_id] = document_set[answer_id]


    print ("Start Evaluation ...")

    snapshot_path = os.path.join(args.save_path, args.dataset, args.mode + '_best_model.pt')

    if args.cuda:
        model = torch.load(snapshot_path, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(snapshot_path, map_location=lambda storage, location: storage)


    average_rank = evaluate(glove_model, query_set, test_document_set, test_true_samples, model)

    print ("Average Rank %f" % average_rank)

