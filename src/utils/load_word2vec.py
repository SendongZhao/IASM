import numpy as np
# loads 300x1 word vectors from file.
def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split()) # 3000000 300
        binary_len = np.dtype('float32').itemsize * layer1_size # 1200
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


# add random vectors of unknown words which are not in pre-trained vector file.
# if pre-trained vectors are not used, then initialize all words in vocab with random value.
def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


vectors_file = './GoogleNews-vectors-negative300.bin'
vocab = ['I', 'can', 'do']

vectors = load_bin_vec(vectors_file, vocab)  # pre-trained vectors
add_unknown_words(vectors, vocab)
print(vectors['I'])
print('*'*40)
print(vectors['can'])
print('*'*40)
print(vectors['do'])