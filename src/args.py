from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Graph Convolution")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')

    parser.add_argument('--word2vec', type=str, default='../GloVe/vectors.txt')
    parser.add_argument('--data_path', type=str, default='../data/pubmed19n0155.json')

    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=3435)

    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--output_size', type=int, default= 100)
    parser.add_argument('--hidden_size', type=int, default= 150)
    parser.add_argument('--dropout', type=float, default= 0.5)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)

    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--dataset', type=str, help='FQA', default='FQA')
    parser.add_argument('--mode', type=str, default='static')
    parser.add_argument('--train_sample_num', type=int, default=20000)

    args = parser.parse_args()
    return args