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
    non_terminal_vocab = 176
    terminal_vocab = 50001
    dropout = 0.5
    embed_size = 50
    hidden_size = embed_size
    batch_size = 200
    n_epochs = 12
    max_grad_norm = 5.
    lr = 0.001
    unk_label = 50000
    train_nt = 'data/train_nt_vectorized.txt'
    train_t = 'data/train_t_vectorized.txt'
    dev_nt = 'data/dev_nt_vectorized.txt'
    dev_t = 'data/dev_t_vectorized.txt'
    test_nt = 'data/test_nt_vectorized.txt'
    test_t = 'data/test_t_vectorized.txt'

    def __init__(self, args):
        self.cell = args.cell
        self.terminal_pred = 0
	if args.non_terminal == "terminal": self.terminal_pred = 1
	self.clip_gradients = args.clip

        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"
        self.results = self.output_path + "real_results.txt"

def pad_sequences(data, max_length, terminal_pred):
    """
    Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.
    """

    ret = []
    attn_ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_token_features
    zero_label = 4 # corresponds to the 'O' tag

    for code_snippet, labels in data:
        in_pad = max_length*2 + terminal_pred - len(code_snippet)
        mask_pad = int((len(code_snippet)-terminal_pred)/2)
        if in_pad <= 0:
            ret.append((code_snippet[:max_length*2+terminal_pred], labels, [False] * (max_length-1) + [True], [1] * (max_length-1) + [0]))
        else:
            mask = [False] * max_length
	    mask[mask_pad] = True
            attn_mask = np.zeros(max_length)
            attn_mask[:mask_pad] = 1
            ret.append((code_snippet + zero_vector * in_pad, labels, mask, list(attn_mask)))
                   
    return ret

class LSTMModel(SequenceModel):

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors
        """
        #self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length, self.config.n_token_features))
	self.non_terminal_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length))#, self.config.n_token_features))
	self.terminal_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length))#, self.config.n_token_features))
	self.next_non_terminal_input_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=([None]))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_length))
        self.attn_mask_placeholder = tf.placeholder(tf.float64, shape=(None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float64)

    def create_feed_dict(self, inputs_batch, mask_batch, attn_mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        """
        feed_dict = {}
        if inputs_batch is not None:
	    feed_dict[self.non_terminal_input_placeholder] = inputs_batch[:,::2]
	    feed_dict[self.terminal_input_placeholder] = inputs_batch[:,1::2]
	    if self.config.terminal_pred:
		feed_dict[self.non_terminal_input_placeholder] = feed_dict[self.non_terminal_input_placeholder][:,:-1]
		feed_dict[self.next_non_terminal_input_placeholder]=inputs_batch[:,-1]
        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch
        if attn_mask_batch is not None:
            feed_dict[self.attn_mask_placeholder] = attn_mask_batch
        if labels_batch is not None:
	    labels_batch = labels_batch.flatten()
	    if not self.config.terminal_pred: labels_batch -= self.config.terminal_vocab
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
        hidden = []

        cell = LSTMCell(Config.n_token_features * Config.embed_size, Config.hidden_size)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        xinit = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
        if not self.config.terminal_pred:
            output_size = self.config.non_terminal_vocab
        else:
            output_size = self.config.terminal_vocab

        U = tf.get_variable('U', shape=[self.config.hidden_size, output_size],
                            initializer=xinit, dtype=tf.float64)
        b2 = tf.get_variable('b2', shape=[output_size], initializer = tf.constant_initializer(0.0), dtype=tf.float64)
	c_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size], dtype=tf.float64)
        h_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size], dtype=tf.float64)
	state_tuple = (c_t, h_t)

	scope = "LSTM_terminal" if self.config.terminal_pred else "LSTM_non_terminal"

    	with tf.variable_scope(scope):
            for time_step in range(self.max_length):
	        if time_step > 0:
		    tf.get_variable_scope().reuse_variables()
                o_t, h_t= cell(x[:,time_step,:], state_tuple)
        	o_drop_t = tf.nn.dropout(o_t, dropout_rate)
        	preds.append(tf.matmul(o_drop_t, U) + b2)
                hidden.append(h_t[1])

		if (self.config.cell == "lstmAcont"):
		    W_a = tf.get_variable('W_a', shape = [self.config.hidden_size, self.config.hidden_size], dtype = tf.float64, initializer = xinit)
                    W_o = tf.get_variable('W_o', shape = [2*self.config.hidden_size, output_size], dtype = tf.float64, initializer = xinit)
                    W_s = tf.get_variable('W_s', shape = [output_size, output_size], dtype = tf.float64, initializer = xinit)
                    b_o = tf.get_variable('b_o', shape = [output_size], dtype = tf.float64, initializer = tf.constant_initializer(0.0))
                    b_s = tf.get_variable('b_s', shape = [output_size], dtype = tf.float64, initializer = tf.constant_initializer(0.0))
		    
		    hidden_stack = tf.stack(hidden, 1)
                    ht = tf.reshape(tf.matmul(h_t[1], W_a), (tf.shape(x)[0], -1, self.config.hidden_size))
		    weights = tf.reduce_sum(ht * hidden_stack, axis=2) * tf.slice(self.attn_mask_placeholder, [0,0] , [-1,time_step + 1])
                    weights = tf.nn.softmax(weights)
		    context = tf.reduce_sum(tf.reshape(weights, (tf.shape(weights)[0], tf.shape(weights)[1], -1)) * hidden_stack, axis = 1)		    
		    hidden = hidden[:-1] + [context]

	    preds = tf.stack(preds, 1)
            hidden = tf.stack(hidden, 1)
	    final_preds = tf.boolean_mask(preds, self.mask_placeholder)
	    final_hidden = tf.boolean_mask(hidden, self.mask_placeholder)
        
    	if (self.config.cell == "lstmAend") or (self.config.cell == "lstmAcont"):
            W_a = tf.get_variable('W_a', shape = [self.config.hidden_size, self.config.hidden_size], dtype = tf.float64, initializer = xinit)
            W_o = tf.get_variable('W_o', shape = [2*self.config.hidden_size, output_size], dtype = tf.float64, initializer = xinit)
            W_s = tf.get_variable('W_s', shape = [output_size, output_size], dtype = tf.float64, initializer = xinit)
            b_o = tf.get_variable('b_o', shape = [output_size], dtype = tf.float64, initializer = tf.constant_initializer(0.0))
            b_s = tf.get_variable('b_s', shape = [output_size], dtype = tf.float64, initializer = tf.constant_initializer(0.0))
            ht = tf.reshape(tf.matmul(final_hidden, W_a), (tf.shape(x)[0], -1, self.config.hidden_size))
            weights = tf.reduce_sum(ht * hidden, axis=2) * self.attn_mask_placeholder
            weights = tf.nn.softmax(weights)

            context = tf.reduce_sum(tf.reshape(weights, (tf.shape(weights)[0], tf.shape(weights)[1], -1)) * hidden, axis = 1)
            final_preds = tf.tanh(tf.matmul(tf.concat(1, [context, final_hidden]), W_o) + b_o)
            final_preds = tf.matmul(final_preds, W_s) + b_s
        
    	if self.config.terminal_pred:
            nt = tf.nn.embedding_lookup(self.embeddings, self.next_non_terminal_input_placeholder)
            nt = tf.reshape(nt, [-1, self.config.n_token_features * self.config.embed_size])
            U_nt = tf.get_variable('U_nt', shape = [self.config.hidden_size, output_size], initializer = xinit,dtype=tf.float64)
            b_t = tf.get_variable('b_t', shape = [output_size], initializer = tf.constant_initializer(0.0), dtype=tf.float64)
            final_preds = final_preds + tf.matmul(nt, U_nt) + b_t
	
    	return final_preds

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

        return pad_sequences(examples, self.max_length, self.config.terminal_pred)

    def consolidate_predictions(self, examples_file, preds):
        """Batch the predictions into groups of sentence length.
        """

        ret = []
	with open(examples_file, 'r') as f:
	    i = 0
	    for line in f:
	        _, label = tuple(eval(line.strip()))
	        label_ = preds[i]
	        ret.append([label[0], label_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch, attn_mask):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch, attn_mask_batch=attn_mask)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, attn_mask):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch, attn_mask_batch=attn_mask,
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
    config.unk = True if args.unk == 'unk' else False
    config.lstm = args.cell

    code_comp = code_comp_utils.get_code_comp()
   
    embeddings = code_comp_utils.get_embeddings()
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
            if not config.terminal_pred:
                model.fit(session, saver, config.train_nt, config.test_nt)
            else:
                model.fit(session, saver, config.train_t, config.test_t)

def do_evaluate(args):
    '''I don't think this should be working yet'''
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
    command_parser.add_argument('-c', '--cell', choices=["lstm", "lstmAend", "lstmAcont"], default="lstm", help="Type of RNN cell to use.")
    command_parser.add_argument('-nt', '--non_terminal', choices=["terminal", "non_terminal"], default="non_terminal", help="Predict terminal or non_terminal")
    command_parser.add_argument('-cp', '--clip', choices=["clip", "no_clip"], default="clip", help="clip gradients")
    command_parser.add_argument('-unk', '--unk', choices=["unk", "no_unk"], default="unk", help="deny unk predictions")
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

