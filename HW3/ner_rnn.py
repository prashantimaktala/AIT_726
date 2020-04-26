
"""
AIT726 HW 3 Due 4/16/2020
Name entity recognition using 6 neural network models (RNN, bi-RNN, LSTM, bi-LSTM, GRU, bi-GRU) on CONLL 2013 datset
Authors: Yasas, Prashanti, Ashwini
Command to run the file: python ner_rnn.py --rnn [simple_rnn/lstm/gru] --bidirectional [true/false] --trainable [true/false]
Select one option from options provided in [ ]
Flow:
i. main
    1. Use the arguments passed via the command prompt, if not default values are set
    2. Pre-processing - train, validate, and test data
        a. read the data and extract sentences and labels
        (Lower case capitalized words (i.e., starts with a capital letter) but not all capital words)
        b. calculate max lenght and add padding
    3. Build vocab
    4. Load word2vec embeddings and create word embeddings and indexes based on word2vec
    5. create neural network models (with one layer of 256 hidden units, and a fully connected output layer using softmax
    as activation function. We have used Adam optimizer, and cross-entropy for the loss function with
    learning rate 0.001 for all the models)
    6.Convert the input sentences to index sequence for both train and validation data using embedding index
    7.Fit and validate the model using training data and validation data
    8.Save all the models to .h5 file (select best model based on accuracy)
    9.Test all seven models using test data (for 7th model which is the best model(bi-GRU in our case) pass trainable = true )
    10.Based on predicted values generate a text file
    11.Evaluate the models using conlleval.py (we have imported the evaluate function from conlleval.py)

Note: We used 125 Epochs with 2000 batch size for Training models 1 - 6.
The learning rate was set to 0.001. We reduced the learning rate to decrease the number of epochs
required for the model to converge. For more details please check README file

"""
import argparse
import gensim
import numpy as np
from tensorflow.keras.layers import InputLayer, Embedding, Dense, GRU, SimpleRNN, LSTM, Bidirectional, TimeDistributed, \
    Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from conlleval import evaluate as _evaluate


def preprocess(sentence):
    """
    preprocess - preprocess sentences. Lower case capitalized words but not all capital words. Do not remove stopwords.
    :param sentence: input sentence
    :return: lower cased sentence tokens
    """
    return [word if word.isupper() else word.lower() for word in sentence]


def get_sentences(path):
    """
    get_sentences - helps to navigate through the files in the path, read the files and retrieve the sentences
    and labels. Words are extracted from the first column and labels (which are the gold standards)
    are extracted from the last column.
    """
    sentences, labels = [], []
    sentence, label = [], []
    for line in open(path, "r").read().split('\n'):
        if line.startswith('-DOCSTART-'):
            continue
        if len(line.strip()) == 0:
            if len(sentence) == 0:
                continue
            sentence = preprocess(sentence)
            sentences.append(sentence)
            labels.append(label)
            sentence, label = [], []
        else:
            sentence.append(line.split(" ", 1)[0])
            label.append(line.split(" ")[3])
    return sentences, labels


def pad_tag(dataset, max_length_sentence):
    """
    pad_tag - helps to append 0's at the end of shorter sentences . Tags for the 0's are set as <pad>.
    max_length_sentence is passed as an argument which is calculated as 513 for our sentences.
    """
    sentences, labels = dataset
    for idx in range(len(sentences)):
        # assert max_length_sentence - len(sentences[idx]) > 0
        sentences[idx] += ['0'] * (max_length_sentence - len(sentences[idx]))
        labels[idx] += ['<pad>'] * (max_length_sentence - len(labels[idx]))
        sentences[idx] = sentences[idx][:max_length_sentence]
        labels[idx] = labels[idx][:max_length_sentence]
    return sentences, labels


def get_vocab(train_sentences):
    """
    get_vocab - retrieves the list of unique vocabulary form our train_sentences.
    """
    vocab = set()
    for sent in train_sentences:
        vocab.update(sent)
    return list(vocab)


def load_word2vec(path='./GoogleNews-vectors-negative300.bin.gz'):
    """
    load_word2vec - create word2vec (  pre trained word embeddings ) from GoogleNews-vectors-negative300.bin.gz
    """
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vec


def get_embeddings(vocab):
    """
    get_embeddings - create word embeddings and embedding_index based on word2vec. Here we are assigning 0 index for '0'
    which we assigned during padding, also we are assigning index 1 for all the words missing in the word2vec.
    we are randomly assigning index for all the words in vocab based on word2vec. we are returning word embeddings
    and embedding_index from the function.
    """
    word2vec = load_word2vec()
    embeddings_index = {'0': 0, '<missing>': 1}
    embeddings = [np.random.random(300) for _ in range(len(vocab) + 2)]
    for word in vocab:
        if (word not in embeddings_index) and (word in word2vec):
            idx = len(embeddings_index)
            embeddings_index[word] = idx
            embeddings[idx] = word2vec[word]
    return np.array(embeddings), embeddings_index


def get_input_seq(sentences, embeddings_index):
    """
   get_input_seq - we are indexing each word in the sentences based on embeddings_index we have created
    which will be the input sequence to the models

    """
    input_seq = []
    for tokens in sentences:
        input_seq.append(
            [embeddings_index[token] if token in embeddings_index else embeddings_index['<missing>'] for token in
             tokens])
    return np.array(input_seq)


def create_rnn_model(embedding_matrix, input_length, rnn='simple_rnn', bidirectional=False, trainable_emb=False,
                     hidden_size=256, lr=0.001):
    """
    create_rnn_model - Depending on the rnn and bidirectional parameter passed ,
    we are creating all the required 6 models (RNN, bi-RNN, LSTM, bi-LSTM, GRU, bi-GRU)
    for all the models we have one layer of 256 hidden units, and a fully connected output layer using softmax
    as activation function. We have used Adam optimizer, and cross-entropy for the loss function with
    learning rate 0.001 for all the models.
     """
    vocabulary_size, embedding_dims = embedding_matrix.shape[0], 300
    model = Sequential()
    model.add(InputLayer(input_shape=(input_length,)))
    model.add(Embedding(vocabulary_size, embedding_dims, weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=trainable_emb))
    rnn = {
        'simple_rnn': SimpleRNN(hidden_size, return_sequences=True),
        'lstm': LSTM(hidden_size, return_sequences=True),
        'gru': GRU(hidden_size, return_sequences=True),
    }[rnn]
    if bidirectional:
        rnn = Bidirectional(rnn)
    model.add(rnn)
    model.add(TimeDistributed(Dense(10, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    return model


def encode_labels(labels, classes=None):
    """
    encode_labels - encode labels to either 0 or 1 for all the 10 classes we have identified depending on if it
    belongs to the class. The 10 classes we have identified are  <pad>, O, B-ORG, B-PER, B-LOC, B-MISC, I-ORG, I-PER,
    ILOC, I-MISC
    """
    if classes is None:
        classes = set()
        for sent in labels:
            classes.update(sent)
        classes = list(classes)
    output = [[[1 if c == lbl else 0 for c in classes] for lbl in sent] for sent in labels]
    return np.array(output), classes


def decode_labels(predictions, classes):
    """
    decode_labels - Decode labels back to the original format
    """
    return np.array([[classes[lbl.argmax()] for lbl in sent] for sent in predictions])


def write_to_file(filename, sentences, labels, preds):
    """
    write_to_file - writing the results to text file consisting of gold standard and predicted labels in the format required
    """
    with open('%s.txt' % filename, 'w', encoding='utf-8') as f:
        f.write('Word Gold_Standard Prediction\n')
        for sent_idx, words in enumerate(sentences):
            for word_idx, word in enumerate(words):
                if word != '0' and labels[sent_idx][word_idx] != '<pad>':
                    f.write('%s %s %s\n' % (words[word_idx], labels[sent_idx][word_idx], preds[sent_idx][word_idx]))
            f.write('\n')


def train_validate(model, x_train, y_train, x_valid, y_valid, num_epochs=125, batch_size=2000):
    """
    train_validate- validate the training data with the required number of epochs and batch_size
    """
    model.fit(x_train, y_train, epochs=num_epochs, verbose=1, batch_size=batch_size, validation_data=(x_valid, y_valid))
    return model


def evaluate(y_true, y_pred, classes):
    """
        evaluate - using get_result function from conllevel.py to evaluate accuracies which will be later written to a text file
    """
    true_seqs = []
    pred_seqs = []
    for true_val, pred_val in zip(decode_labels(y_true, classes).flatten(), decode_labels(y_pred, classes).flatten()):
        if true_val != '<pad>':
            true_seqs += [true_val]
            pred_seqs += [pred_val if pred_val != '<pad>' else 'O']
    return _evaluate(true_seqs, pred_seqs)


def main(**kwargs):
    """
    main - Execution of appropriate functions as per the required call for training, validation and test data.
    Execution of appropriate models built and evaluating the best results from the saved models and results
    """
    # default model will be unidirectional vanilla_rnn with trainable false
    # The trainable parameter for the Embedding layer was set to false to prevent updating the embeddings along with the rest of the network
    rnn, bidirectional = kwargs.get('rnn', 'simple_rnn'), kwargs.get('bidirectional', False)
    trainable_emb = kwargs.get('trainable', False)
    model_name = '%s%s' % ('bi-' if bidirectional else '', rnn)
    input_length = 513
    train_sentences, train_labels = pad_tag(get_sentences('./conll2003/train.txt'), input_length)
    vocab = get_vocab(train_sentences)
    print('Vocab Size: %d' % len(vocab))
    embedding_matrix, embeddings_index = get_embeddings(vocab)
    model = create_rnn_model(
        embedding_matrix,
        input_length,
        rnn=rnn,
        bidirectional=bidirectional,
        trainable_emb=trainable_emb,
        hidden_size=256,
        lr=0.001
    )
    # Train and validate the model
    # -- Convert the input text to (index) sequence
    x_train, (y_train, classes) = get_input_seq(train_sentences, embeddings_index), encode_labels(train_labels)
    # -- Load and Convert the validation text to (index) sequence
    valid_sentences, valid_labels = pad_tag(get_sentences('./conll2003/valid.txt'), input_length)
    x_valid, (y_valid, _) = get_input_seq(valid_sentences, embeddings_index), encode_labels(valid_labels, classes)
    train_validate(model, x_train, y_train, x_valid, y_valid)
    # Save the model
    model.save(model_name + '.h5')
    # Predict for the test data
    test_sentences, test_labels = pad_tag(get_sentences('./conll2003/test.txt'), input_length)
    x_test, (y_test, _) = get_input_seq(test_sentences, embeddings_index), encode_labels(test_labels, classes)
    y_pred = model.predict(x_test)
    #evaluate using conlleval.py
    _, result = evaluate(y_test, y_pred, classes)
    write_to_file(model_name, test_sentences, test_labels, decode_labels(y_pred, classes))
    #save the accuracy to the text file
    with open('results_%s.txt' % model_name, 'a', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    """
    main - Pass the required parameters needed to be passed to main
    """
    parser = argparse.ArgumentParser(description='Train NER Sequence Model.')
    parser.add_argument('--rnn', help='RNN layer type [simple_rnn]/lstm/gru', default='simple_rnn')
    parser.add_argument('--bidirectional', help='whether to use bidirectional true/[false]', default='false')
    parser.add_argument('--trainable', help='whether to use trainable embeddings true/[false]', default='false')
    args = parser.parse_args()
    main(rnn=args.rnn, bidirectional=args.bidirectional == 'true')
