#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import pdb
import logging

import tensorflow as tf
from lstm_util import Progbar, get_minibatch
from model import Model
import numpy as np
#from defs import LBLS

logger = logging.getLogger("lstm-sequence")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SequenceModel(Model):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, helper, config, report=None):
        self.helper = helper
        self.config = config
        self.report = report
        self.debug = False
        self.train_size = 100000
	self.dev_size = 20000
        self.debug_size = 500
        self.eval_debug_size = 100
        self.eval_size = 30000

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    def evaluate(self, sess, data_file, size):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
	
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for labels, labels_ in self.output(sess, data_file, size):
            if not self.config.unk and not self.config.nt:
	        correct_preds += (labels == labels_ and labels != self.config.unk_label)
            else:
                correct_preds += (labels == labels_)
	    total_preds += 1
	    '''
            gold = set(labels)
            pred = set(labels_)
            correct_preds = len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return (p, r, f1)
	'''
	return correct_preds/total_preds	


    def run_epoch(self, sess, train_file, eval_file):
        num_train = self.debug_size if self.debug else self.train_size
        prog = Progbar(target=1 + num_train / self.config.batch_size)
        with open(train_file, 'r') as f:
            i = 0
            num_examples = num_train
            while num_examples > 0:
		if num_examples - self.config.batch_size > 0:
		    next_batch = self.config.batch_size
		else:
		    next_batch = num_examples
                b = get_minibatch(f, next_batch)
                i += 1
                b = self.preprocess_sequence_data(b)
		batch = []
		for col in zip(*b):
       	 	    batch = [np.array(col) for col in zip(*b)]
                num_examples -= next_batch
                loss = self.train_on_batch(sess, *batch)
                prog.update(i + 1, [("train loss", loss)])
                if self.report: self.report.log_train_loss(loss)
            print("")

        with open(self.config.results, 'a') as f:
            f.write("Loss: %.2f, " % loss)
        logger.info("Evaluating on data set")
	eval_size = self.eval_debug_size if self.debug else self.eval_size
        entity_scores = self.evaluate(sess, eval_file, eval_size)
        
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
	logger.info("Accuracy: %.2f", entity_scores)

        

        #f1 = entity_scores[-1]
        #return f1
	return entity_scores

    def output(self, sess, input_file, size):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """

        preds = []
        prog = Progbar(target=1 + int(size / self.config.batch_size))
	with open (input_file, 'r') as f:
	    i = 0
            num_examples = size
	    while num_examples > 0:
		if num_examples - self.config.batch_size > 0:
		    next_batch = self.config.batch_size
		else:
		    next_batch = num_examples
	        b = get_minibatch(f, next_batch)
	        i += 1
	        b = self.preprocess_sequence_data(b)
		batch = []
                for col in zip(*b):
                    batch = [np.array(col) for col in zip(*b)]
                num_examples -= next_batch
                # Ignore predict
                batch = batch[:1] + batch[2:]
            	preds_ = self.predict_on_batch(sess, *batch)
           	preds += list(preds_)
            	prog.update(i + 1, [])

        return self.consolidate_predictions(input_file, preds)

    def fit(self, sess, saver, train_file, eval_file):
        best_score = 0.
	
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            with open(self.config.results, 'a') as f:
                f.write("Epoch: %d, " % (epoch + 1))
            score = self.run_epoch(sess, train_file, eval_file)
            with open(self.config.results, 'a') as f:
                f.write("Accuracy: %.2f\n" % score)

            if score > best_score:
                best_score = score
                logger.info("New best score!")
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score
