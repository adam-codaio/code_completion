'''
LSTM
'''
from __future__ import absolute_import
from __future__ import division

from lstm_util import Progbar, minibatches
from model import Model

from lstm_cell import LSTMCell

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from lstm_util import print_sentence, write_conll, read_conll
from lstm_data_util import load_embeddings, ModelHelper
from utils import code_comp_utils #import load_and_preprocess_data, CodeCompleter
#from defs import LBLS
from sequence_model import SequenceModel
from lstm_cell import LSTMCell

logger = logging.getLogger("vanilla_lstm")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_token_features = 1 # Number of features for every token in the input.
    max_length = 49 # longest sequence to parse
    non_terminal_vocab = 50200
    terminal_vocab = 50200
    dropout = 0.5
    embed_size = 1500
    hidden_size = embed_size
    batch_size = 80
    n_epochs = 1
    max_grad_norm = 5.
    lr = 0.001

    def __init__(self, args):
        self.cell = args.cell
        self.predict = args.non_terminal
	self.clip_gradients = args.clip

        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/"#/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"


def pad_sequences(data, max_length):
    """
    Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.
    """

    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_token_features
    zero_label = 4 # corresponds to the 'O' tag

    for code_snippet, labels in data:
        in_pad = max_length*2 - len(code_snippet)
	mask_pad = int(len(code_snippet)/2)
        if in_pad <= 0:
            ret.append((code_snippet[:max_length*2], labels, [False] * (max_length-1) + [True]))
        else:
	    mask = [False] * max_length
	    mask[mask_pad] = True
            ret.append((code_snippet + zero_vector * in_pad, labels, mask))
                   
    return ret

class LSTMModel(SequenceModel):

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors
        """
        #self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length, self.config.n_token_features))
	self.non_terminal_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length))#, self.config.n_token_features))
	self.terminal_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length))#, self.config.n_token_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=([None]))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        """
        feed_dict = {}
        if inputs_batch is not None:
	    #non_terminal, terminal = zip(*inputs_batch)
	    feed_dict[self.non_terminal_input_placeholder] = inputs_batch[:,::2]
	    feed_dict[self.terminal_input_placeholder] = inputs_batch[:,1::2]
            #feed_dict[self.input_placeholder] = inputs_batch
        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch
        if labels_batch is not None:
	    labels_batch = labels_batch.flatten()
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embedding_size).

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        #terminal_embed_tensor = tf.Variable(self.terminal_embeddings)
        #non_terminal_embed_tensor = tf.Variable(self.non_terminal_embeddings)
        embeddings = tf.nn.embedding_lookup(self.embeddings, self.terminal_input_placeholder) + tf.nn.embedding_lookup(self.embeddings, self.non_terminal_input_placeholder)
        #output = tf.nn.embedding_lookup(embed_tensor, self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.max_length, self.config.n_token_features * self.config.embed_size])
        return embeddings

    def add_prediction_op(self):
        """
        Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, non_terminal_vocab)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = [] # Predicted output at each timestep should go here!

        cell = LSTMCell(Config.n_token_features * Config.embed_size, Config.hidden_size)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        xinit = tf.contrib.layers.xavier_initializer()
        if self.config.predict == "non_terminal":
            output_size = self.config.non_terminal_vocab
        else:
            output_size = self.config.terminal_vocab
        U = tf.get_variable('U', shape=[self.config.hidden_size, output_size],
                            initializer=xinit)
        b2 = tf.Variable(tf.zeros([output_size]))
	c_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size])
        h_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size])
	state_tuple = (c_t, h_t)

        with tf.variable_scope("LSTM"):
            for time_step in range(self.max_length):
		if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t= cell(x[:,time_step,:], state_tuple)
        	o_drop_t = tf.nn.dropout(o_t, dropout_rate)
        	preds.append(tf.matmul(o_drop_t, U) + b2)
	preds = tf.stack(preds, 1)
	preds = tf.boolean_mask(preds, self.mask_placeholder)
	
        return preds

    def add_loss_op(self, preds):
        """
        Adds Ops for the loss function to the computational graph.

        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds,
                                                                       labels=self.labels_placeholder)) 
        return loss

    def add_training_op(self, loss):
        """
        Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = optimizer.compute_gradients(loss)
        gradients, values = zip(*gvs)
        if self.config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.config.max_grad_norm)
        self.grad_norm = tf.global_norm(gradients)
        gvs = zip(gradients, values)
        train_op = optimizer.apply_gradients(gvs)
        

        return train_op

    def preprocess_sequence_data(self, examples):
        return pad_sequences(examples, self.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (code_snippet, labels) in enumerate(examples_raw):
            labels_ = preds[i] # only select elements of mask.
            ret.append([code_snippet, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, embeddings, report=None):
        super(LSTMModel, self).__init__(helper, config, report)
        self.max_length = 49#min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length # Just in case people make a mistake.
        self.embeddings = embeddings
	self.grad_norm = None

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None


        self.build()

def do_train(args):
    # Set up some parameters.
    config = Config(args)
    code_comp, train, dev, test = code_comp_utils.load_and_preprocess_data()
    train = train[:320]
    dev = dev[:320]
    embeddings = np.array(np.random.randn(50200, 1500), dtype=np.float32)
    config.embed_size = embeddings.shape[1]
    helper = ModelHelper(code_comp.tok2id, 49)
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = LSTMModel(code_comp, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)

def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = LSTMModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                #predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)

def do_shell(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .
""")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ["O"] * len(tokens))]):
                        #predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
                except EOFError:
                    print("Closing session.")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    #command_parser = subparsers.add_parser('test1', help='')
    #command_parser.set_defaults(func=do_test1)

    #command_parser = subparsers.add_parser('test2', help='')
    #command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll", help="Training data")
    #command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    #command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    #command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    #command_parser.add_argument('-c', '--cell', choices=["lstm"], default="lstm", help="Type of RNN cell to use.")
    #command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    #command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll", help="Training data")
    #command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    #command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    #command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["lstm"], default="lstm", help="Type of RNN cell to use.")
    command_parser.add_argument('-nt', '--non_terminal', choices=["terminal", "non_terminal"], default="non_terminal", help="Predict terminal or non_terminal")
    command_parser.add_argument('-cp', '--clip', choices=["clip", "no_clip"], default="clip", help="clip gradients")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Training data")
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["lstm"], default="lstm", help="Type of RNN cell to use.")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.add_argument('-tnt', '--non-terminal', choices=["terminal", "non_terminal"], default="non_terminal", help="Predict terminal or non_terminal")
    command_parser.add_argument('-cp', '--clip', choices=["terminal", "non_terminal"], default="non_terminal", help="clip gradients")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["lstm"], default="lstm", help="Type of RNN cell to use.")
    command_parser.add_argument('-tnt', '--non-terminal', choices=["terminal", "non_terminal"], default="non_terminal", help="Predict terminal or non_terminal")
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

