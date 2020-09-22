import torch
import torch.nn as nn


class GloveVector:

    def __init__(self, glove_file):

        self.glove_file = glove_file
        self.model = {}

    def loadGloveModel(self):

        print ("Loading Glove Model ...")

        fglove = open(self.glove_file, 'r')

        for line in fglove:
            split_line = line.split()
            word = split_line[0]
            embedding = list([float(val) for val in split_line[1:]])
            self.model[word] = embedding
        print ("Done. %d words loaded!" % len(self.model))

    def getWordVector(self, word):
        # type: (object) -> object

        word = word.lower()

        if word not in self.model:
            #print ("No word %s, return <OOV>" % word)
            #return torch.LongTensor(self.model['<OOV>'])
            return torch.FloatTensor(list(float(0.00001) for i in range(100)))

        return torch.FloatTensor(self.model[word])


def getWeightMatrix(glove_model, target_vocab, embed_dim):
    matrix_len = len(target_vocab)
    weights_matrix = torch.zeros(matrix_len, embed_dim, dtype=torch.float)
    words_found = 0

    for i, word in enumerate(target_vocab):

        try:
            weights_matrix[i] = glove_model.getWordVector(word)
            words_found += 1
        except KeyError:
            weights_matrix[i] = torch.rand(embed_dim, )

    if torch.cuda.is_available():
        weights_matrix = weights_matrix.cuda()

    return weights_matrix


def wordSimilarity(glove_model, wordA, wordB):
    a_embed = glove_model.getWordVector(wordA)
    b_embed = glove_model.getWordVector(wordB)

    cos = nn.CosineSimilarity(dim=0)

    sim = cos(a_embed, b_embed)

    return sim
