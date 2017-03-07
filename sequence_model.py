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
        self.debug = True
        self.train_size = 100000
        self.debug_size = 500

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


    def evaluate(self, sess, examples, examples_raw):
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
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
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


    def run_epoch(self, sess, train_file, dev_file):
        num_train = config.train_size if config.debug else config.debug_size
        prog = Progbar(target=1 + num_train / self.config.batch_size))
        with open(train_file, 'r') as f:
            more = True
            i = 0
            num_examples = num_train
            while not more and num_examples > 0:
                batch, more = get_minibatch(f, self.config.batch_size)
                i += 1
                batch = self.preprocess_sequence_data(batch)
                num_examples -= len(batch)
                loss = self.train_on_batch(sess, *batch)
                prog.update(i + 1, [("train loss", loss)])
                if self.report: self.report.log_train_loss(loss)
            print("")

        logger.info("Evaluating on development data")
        #TODO: fix evaluate
        entity_scores = []
        #entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
	logger.info("Accuracy: %.2f", entity_scores)

        #f1 = entity_scores[-1]
        #return f1
	return entity_scores

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(inputs_raw)

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_file, dev_file):
        best_score = 0.
	
        #train_examples = self.preprocess_sequence_data(train_examples_raw)
        #dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_file, dev_file)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score
