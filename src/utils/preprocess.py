import csv
import tokenization
tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

import json
import random


def train_test_split(dict_pairs, rate):

    train_files = {}
    test_files = {}
    for query_id in dict_pairs:
        rand = random.random()

        if rand <= rate:
            train_files[query_id] = dict_pairs[query_id]
        else:
            test_files[query_id] = dict_pairs[query_id]
    return train_files, test_files


def process(train_file, test_file):

    question2id = {}
    answer2id = {}

    questions = {}
    answers = {}
    q_a2label_train = {}
    question2answer = {}

    csvFile = open(train_file, "r")

    reader = csv.reader(csvFile)

    for item in reader:
        if reader.line_num == 1:
            continue

        if item[0] not in question2id:
            q_id = len(questions)
            questions[q_id] = tokenizer.tokenize(item[0])
            question2id[item[0]] = q_id

        if item[2] not in answer2id:
            a_id = len(answers)
            answers[a_id] = tokenizer.tokenize(item[2])
            answer2id[item[2]] = a_id

        if item[1] is '1':
            if question2id[item[0]] not in question2answer:
                question2answer[question2id[item[0]]] = []

            question2answer[question2id[item[0]]].append(answer2id[item[2]])

        q_a2label_train[str(question2id[item[0]])+' '+str(answer2id[item[2]])] = item[1]

    csvFile.close()

    csvFile = open(test_file, "r")

    reader = csv.reader(csvFile)

    q_a2label_test = {}

    for item in reader:
        if reader.line_num == 1:
            continue

        if item[0] not in question2id:
            q_id = len(questions)
            questions[q_id] = tokenizer.tokenize(item[0])
            question2id[item[0]] = q_id

        if item[2] not in answer2id:
            a_id = len(answers)
            answers[a_id] = tokenizer.tokenize(item[2])
            answer2id[item[2]] = a_id

        if item[1] is '1':
            if question2id[item[0]] not in question2answer:
                question2answer[question2id[item[0]]] = []

            question2answer[question2id[item[0]]].append(answer2id[item[2]])

        q_a2label_test[str(question2id[item[0]])+' '+str(answer2id[item[2]])] = item[1]

    csvFile.close()


    with open('../../data/id2question.json', 'w') as fwrite:
        json.dump(questions, fwrite, indent=4)

    with open('../../data/id2answer.json', 'w') as fwrite:
        json.dump(answers, fwrite, indent=4)

    with open('../../data/q_a2label_train.json', 'w') as fwrite:
        json.dump(q_a2label_train, fwrite, indent=4)

    with open('../../data/q_a2label_test.json', 'w') as fwrite:
        json.dump(q_a2label_test, fwrite, indent=4)

    with open('../../data/question2answer_true.json', 'w') as fwrite:
        json.dump(question2answer, fwrite, indent=4)


if __name__ == "__main__":

    process('../../data/curatedv2-training.csv', '../../data/curatedv2-test.csv')




