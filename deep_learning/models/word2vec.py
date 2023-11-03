import numpy as np
from collections import defaultdict


class word2vec:
    """
    skip-gram version word to vector
    n: int
        Dimensions of embedding size, also refer to size of hidden layer
    learning_rate: float
        learning rate
    epochs: int
        number of training epochs
    window_size: int
        context window size +- center word
    """

    def __init__(self, n, learning_rate, epochs, window_size):
        self.n = n
        self.lr = learning_rate
        self.epochs = epochs
        self.window = window_size

    def generate_training_data(self, settings, corpus):
        # Find unique word counts using dictionary
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        ########################################################################
        # word_counts
        # {'natural': 1, 'language': 1, 'processing': 1, 'and': 2, 'machine': 1,
        # 'learning': 1, 'is': 1, 'fun': 1, 'exciting': 1})
        #########################################################################

        # Get number of unique word in corpus also as size of dictionary: 9
        self.v_count = len(word_counts.keys())

        # Generate Lookup Dictionaries (vocab)
        self.words_list = list(word_counts.keys())
        #################################################################################################
        # print(self.words_list)																		#
        # ['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'exciting']	#
        #################################################################################################

        # Generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        ##################################################################################################
        # print(self.word_index)
        # {'natural': 0, 'language': 1, 'processing': 2, 'and': 3, 'machine': 4, 'learning': 5,
        # 'is': 6, 'fun': 7, 'exciting': 8}
        ##################################################################################################

        # Generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        ##################################################################################################
        # print(self.index_word)
        # {0: 'natural', 1: 'language', 2: 'processing', 3: 'and', 4: 'machine', 5: 'learning',
        # 6: 'is', 7: 'fun', 8: 'exciting'}
        ##################################################################################################

        training_data = []

        # loop through each sentence in corpus
        for sentence in corpus:
            # get each sentence length
            sent_len = len(sentence)

            # loop through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot, ex: [1, 0, 0, 0, 0, 0, 0, 0, 0] word as 'natural'
                w_target = self.word2onehot(sentence[i])

                # loop through context window
                w_context = []

                # window_size 2 will have range of 5 values
                for j in range(i - self.window, i + self.window + 1):
                    # Criteria for context word
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0)
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1)
                    if j != i and 0 <= j <= sent_len:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                    # print(sentence[i], sentence[j])
                    #########################
                    # Example:				#
                    # natural language		#
                    # natural processing	#
                    # language natural		#
                    # language processing	#
                    # language append 		#
                    #########################

                # training_data contains a one-hot representation of the target word and context words
                #################################################################################################
                # Example:																						#
                # [Target] natural, [Context] language, [Context] processing									#
                # print(training_data)																			#
                # [[[1, 0, 0, 0, 0, 0, 0, 0, 0], [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]]]	#
                # each target will contain all context in one list target --> [context1, context2, ..]          #
                #################################################################################################
                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        # word_vec - initialise a blank vector, size as dictionary size
        word_vec = [0 for i in range(self.v_count)]  # Alternative - np.zeros(self.v_count)
        #################################
        # print(word_vec)			    #
        # [0, 0, 0, 0, 0, 0, 0, 0, 0]	#
        #################################

        # Get ID of word from word_index
        word_index = self.word_index[word]

        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):
        # Initialising weight matrices
        # np.random.uniform(HIGH, LOW, OUTPUT_SHAPE)
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.uniform.html
        # self.w1 = np.array(getW1)
        # self.w2 = np.array(getW2)
        # w1: [number of voc, embedding size], w2: [embedding size, number of voc]
        # w1 is final word to vector embedding, w2 is transfer matrix to convert to output size same as input
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        # loop through each epoch
        for i in range(self.epochs):
            # Intialise loss to 0
            self.loss = 0
            # loop through each training sample
            # w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
                # Forward pass
                # 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
                y_pred, h, u = self.forward_pass(w_t)
                #########################################
                # print("Vector for target word:", w_t)	#
                # print("W1-before backprop", self.w1)	#
                # print("W2-before backprop", self.w2)	#
                #########################################

                # Calculate error, ex: [1, 9]
                # formula: E/W = sum[1:c](y_pred - y_truth) * hidden_layer_output (h), first part is EI in here
                # 1. For a target word, calculate difference between y_pred and each of the context words
                # 2. Sum up the differences using np.sum to give us the error for this particular target word
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                #########################
                # print("Error", EI)	#
                # [0.2, 0.3, ..., 0.9]  #
                #########################

                # Backpropagation
                # We use SGD to backpropagate errors - calculate loss on the output layer
                self.backprop(EI, h, w_t)
                #########################################
                # print("W1-after backprop", self.w1)	#
                # print("W2-after backprop", self.w2)	#
                #########################################

                # Calculate loss
                # formula: -sum[1, c](Uc) + c * log(sum[1, v](exp(u)), c is number of context, v is size of dictionary
                # There are 2 parts to the loss function
                # Part 1: sum of all u, this is u is context corresponding index, ex: context1: 1, context2: 2, u1 + u2
                # Part 2: length of context words * log of sum for all elements (exponential) in the output layer before softmax (u)
                # Note: word.index(1) returns the index in the context word vector with value 1
                # Note: u[word.index(1)] returns the value of the output layer before softmax
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            #############################################################
            # Break if you want to see weights after first target word 	#
            # break 													#
            #############################################################
            print('Epoch:', i, "Loss:", self.loss)

    def forward_pass(self, x):
        # no activation function for each matrix weight
        # x is one-hot vector for target word, shape - [1, 9]
        # Run through first matrix (w1) to get hidden layer - [1, 9] * [9, 10] --> [1, 10]
        h = np.dot(x, self.w1)
        # Dot product hidden layer with second matrix (w2) - [1, 10] * [10, 9] --> [1, 9]
        u = np.dot(h, self.w2)
        # Run 1x9 through softmax to force each element to range of [0, 1] - 1x9, probability of each word
        y_c = self.softmax(u)
        return y_c, h, u

    def backprop(self, e, h, x):
        # formula: dw2: e * h(output of hidden layer); dw1: e * w2 * x(input x)
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current target
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 1x10, e - shape 1x9, dl_dw2 - shape 10x9 same as w2
        # x - shape 1x9, w2 - 10x9, e.T - 9x1, dl_dw1 - shape 9x10
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        ########################################
        # print('Delta for w2', dl_dw2)			#
        # print('Hidden layer', h)				#
        # print('np.dot', np.dot(self.w2, e.T))	#
        # print('Delta for w1', dl_dw1)			#
        #########################################

        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    def softmax(self, x):
        # e_x = np.exp(x - np.max(x))
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def word_vec(self, word):
        # Get vector from word
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def vec_sim(self, word, top_n):
        # Input vector, returns nearest word(s)
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            # Find the similarly score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
