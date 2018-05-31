from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Embedding, Dense, Input, RepeatVector, TimeDistributed, concatenate, Merge, add, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

HIDDEN_UNITS = 100
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10



class RecursiveRNN1(object):
    model_name = 'recursive-rnn-1'
    """
    A second alternative model is to develop a model that generates a single word forecast and call it recursively.
    
    That is, the decoder uses the context vector and the distributed representation of all words generated so far as 
    input in order to generate the next word. 
    
    A language model can be used to interpret the sequence of words generated so far to provide a second context vector 
    to combine with the representation of the source document in order to generate the next word in the sequence.
    
    The summary is built up by recursively calling the model with the previously generated word appended (or, more 
    specifically, the expected previous word during training).
    
    The context vectors could be concentrated or added together to provide a broader context for the decoder to 
    interpret and output the next word.
    """

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        if 'version' in config:
            self.version = config['version']
        else:
            self.version = 0
        self.config = config

        print('max_input_seq_length', self.max_input_seq_length)
        print('max_target_seq_length', self.max_target_seq_length)
        print('num_input_tokens', self.num_input_tokens)
        print('num_target_tokens', self.num_target_tokens)

        inputs1 = Input(shape=(self.max_input_seq_length,))
        am1 = Embedding(self.num_input_tokens, 128)(inputs1)
        am2 = LSTM(128)(am1)

        inputs2 = Input(shape=(self.max_target_seq_length,))
        sm1 = Embedding(self.num_target_tokens, 128)(inputs2)
        sm2 = LSTM(128)(sm1)

        decoder1 = concatenate([am2, sm2])
        outputs = Dense(self.num_target_tokens, activation='softmax')(decoder1)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def split_target_text(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x)+1 >= self.max_target_seq_length:
                    x.append('END')
                    break
            temp.append(x)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        encoder_input_data_batch = []
        decoder_input_data_batch = []
        decoder_target_data_batch = []
        line_idx = 0
        while True:
            for recordIdx in range(0, len(x_samples)):
                target_words = y_samples[recordIdx]
                x = x_samples[recordIdx]
                decoder_input_line = []

                for idx in range(0, len(target_words)-1):
                    w2idx = 0  # default [UNK]
                    w = target_words[idx]
                    if w in self.target_word2idx:
                        w2idx = self.target_word2idx[w]
                    decoder_input_line = decoder_input_line + [w2idx]
                    decoder_target_label = np.zeros(self.num_target_tokens)
                    w2idx_next = 0
                    if target_words[idx+1] in self.target_word2idx:
                        w2idx_next = self.target_word2idx[target_words[idx+1]]
                    if w2idx_next != 0:
                        decoder_target_label[w2idx_next] = 1
                    decoder_input_data_batch.append(decoder_input_line)
                    encoder_input_data_batch.append(x)
                    decoder_target_data_batch.append(decoder_target_label)

                    line_idx += 1
                    if line_idx >= batch_size:
                        yield [pad_sequences(encoder_input_data_batch, self.max_input_seq_length),
                               pad_sequences(decoder_input_data_batch,
                                             self.max_target_seq_length)], np.array(decoder_target_data_batch)
                        line_idx = 0
                        encoder_input_data_batch = []
                        decoder_input_data_batch = []
                        decoder_target_data_batch = []

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN1.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN1.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN1.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None, batch_size=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version

        config_file_path = RecursiveRNN1.get_config_file_path(model_dir_path)
        weight_file_path = RecursiveRNN1.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = RecursiveRNN1.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.split_target_text(Ytrain)
        Ytest = self.split_target_text(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        total_training_samples = sum([len(target_text)-1 for target_text in Ytrain])
        total_testing_samples = sum([len(target_text)-1 for target_text in Ytest])
        train_num_batches = total_training_samples // batch_size
        test_num_batches = total_testing_samples // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        start_token = self.target_word2idx['START']
        wid_list = [start_token]
        sum_input_seq = pad_sequences([wid_list], self.max_target_seq_length)
        terminated = False

        target_text = ''

        while not terminated:
            output_tokens = self.model.predict([input_seq, sum_input_seq])
            sample_token_idx = np.argmax(output_tokens[0, :])
            sample_word = self.target_idx2word[sample_token_idx]
            wid_list = wid_list + [sample_token_idx]

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or len(wid_list) >= self.max_target_seq_length:
                terminated = True
            else:
                sum_input_seq = pad_sequences([wid_list], self.max_target_seq_length)
        return target_text.strip()
