import json



def countData(train, test, id2document, id2query):

    fread = open(train)
    query2document_train = json.load(fread)
    fread.close()

    fread = open(test)
    query2document_test = json.load(fread)
    fread.close()

    fread = open(id2document)
    document = json.load(fread)
    fread.close()

    fread = open(id2query)
    query = json.load(fread)
    fread.close()


    i_split = 0

    num_train_query = 0
    num_train_document = 0
    num_train_query_word = 0
    train_query = {}
    train_document = {}
    train_query_words = {}

    num_valid_query = 0
    num_valid_document = 0
    num_valid_query_word = 0
    valid_query = {}
    valid_document = {}
    valid_query_words = {}

    for q2a in query2document_train.keys():

        query_id, document_id = q2a.split()

        i_split += 1

        if i_split % 9  == 0:
            query_words = query[query_id]
            for word in query_words:
                if word not in valid_query_words:
                    valid_query_words[word] = 1
                    num_valid_query_word += 1
            if query_id not in valid_query:
                valid_query[query_id] = 1
                num_valid_query += 1
            if document_id not in valid_document:
                valid_document[document_id] = 1
                num_valid_document += 1
        else:
            query_words = query[query_id]
            for word in query_words:
                if word not in train_query_words:
                    train_query_words[word] = 1
                    num_train_query_word += 1
            if query_id not in train_query:
                train_query[query_id] = 1
                num_train_query += 1
            if document_id not in train_document:
                train_document[document_id] = 1
                num_train_document += 1

    num_test_query = 0
    num_test_document = 0
    num_test_query_word = 0
    test_query = {}
    test_document = {}
    test_query_words = {}

    print("train and valid complete...")

    for q2a in query2document_test.keys():

        query_id, document_id = q2a.split()

        #print ('query id', query_id)
        query_words = query[query_id]
        for word in query_words:
            if word not in test_query_words:
                test_query_words[word] = 1
                num_test_query_word += 1
        if query_id not in test_query:
            test_query[query_id] = 1
            num_test_query += 1
        if document_id not in test_document:
            test_document[document_id] = 1
            num_test_document += 1

    print ("num_train_query: %d, num_train_document: %d, num_train_query_word: %d" %(num_train_query, num_train_document, num_train_query_word))
    print ("num_valid_query: %d, num_valid_document: %d, num_valid_query_word: %d" % (num_valid_query, num_valid_document, num_valid_query_word))
    print ("num_test_query: %d, num_test_document: %d, num_test_query_word: %d" % (num_test_query, num_test_document, num_test_query_word))




if __name__ == "__main__":
    train = "../../data/q_a2label_train.json"
    test = "../../data/q_a2label_test.json"
    id2document = "../../data/id2answer.json"
    id2query = "../../data/id2question.json"
    countData(train, test, id2document, id2query)








