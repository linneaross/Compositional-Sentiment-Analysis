from collections import defaultdict
import math
import numpy as np
import sys
import string
import re

class PureStatistical:

    def __init__(self, pos_dataset, neg_dataset, train_dataset, n=1, alpha=1):
        self.pos_data_file = pos_dataset
        self.neg_data_file = neg_dataset
        self.pos_data = []
        self.neg_data = []
        self.train_data = train_dataset
        self.n = n
        self.alpha = alpha
        self.vocab = {'total': defaultdict(lambda: []), 'pos': defaultdict(lambda: []), 'neg': defaultdict(lambda: [])}
        # check indices
        for i in range(1,n-1):
            gram = str(i+1) + '-gram'
            self.vocab['total'][gram] = []
            self.vocab['pos'][gram] = []
            self.vocab['neg'][gram] = []
        self.class_doc_counts = {'pos': 0.0, 'neg': 0.0}
        self.total_doc_count = 0.0
        self.class_token_cardinality = {'pos': 0.0, 'neg': 0.0}
        self.class_unit_counts = {'pos': defaultdict(lambda: defaultdict(lambda: 0.0)), 'neg': defaultdict(lambda: defaultdict(lambda: 0.0))}
        self.min_prob = 1.0
        self.max_prob = 0.0

    def train_model(self):
        for (item, label) in [(self.pos_data_file, 'pos'), (self.neg_data_file, 'neg')]:
            for data in item:
                self.class_doc_counts[label] += 1.0
                self.total_doc_count += 1.0

                cleaned = self.tokenize(data)

                # n-gram pre-padding
                n = self.n - 1
                n_temp = n
                data = ['#'] * n + cleaned

                # increment counts
                for i in range(len(data)):
                    if data[i] != '#':
                        self.class_token_cardinality[label] += 1.0
                        while(n_temp > -1):
                            if n_temp == 0:
                                self.class_unit_counts[label][''][data[i]] += 1.0
                                if data[i] not in self.vocab['total']['1-gram']:
                                    self.vocab['total']['1-gram'].append(data[i])
                                if data[i] not in self.vocab[label]['1-gram']:
                                    self.vocab[label]['1-gram'].append(data[i])
                            else:
                                history = data[(i-n_temp):i]
                                hist_string = self.to_string(history)
                                gram = str(n_temp+1) + '-gram'
                                if hist_string not in self.vocab['total'][gram]:
                                    self.vocab['total'][gram].append(hist_string)
                                if hist_string not in self.vocab[label][gram]:
                                    self.vocab[label][gram].append(hist_string)
                                self.class_unit_counts[label][hist_string][data[i]] += 1.0
                            n_temp -= 1
                        n_temp = n

    def tokenize(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.strip()
        sentence = sentence.translate(None, string.punctuation)
        tokens = sentence.split()
        return tokens

    def classify_word(self, word, alpha):
        best_tag = ''
        highest_prob = 0.0
        pos_prob = 0.0
        neg_prob = 0.0
        for label in self.class_unit_counts:
            # fix the prob calculation denominator
            prob_word_given_label = self.class_unit_counts[label][''][word]/self.class_token_cardinality[label]
            prob_label = (self.class_doc_counts[label]/self.total_doc_count)
            prob = (prob_word_given_label * prob_label) + alpha
            if label == 'pos':
                pos_prob = prob
            else:
                neg_prob = prob
            if prob > highest_prob:
                highest_prob = prob
                best_tag = label
        return best_tag, pos_prob, neg_prob

    def classify_datum(self, datum, alpha):
        storage = {}
        tokens = self.tokenize(datum)
        tokens = ['#'] * (self.n - 1) + tokens
        for label in self.class_unit_counts:
            prob = 0.0
            for word in range(len(tokens)):
                if tokens[word] != '#':
                    history = tokens[(word-(self.n-1)):word]
                    order = len(history)+1
                    hist_string = self.to_string(history)
                    while hist_string not in self.class_unit_counts[label]:
                        history = history[1:]
                        order = len(history)+1
                        hist_string = self.to_string(history)
                    word_prob = self.compute_probability(tokens[word], hist_string, order, label, alpha)
                    if word_prob < self.min_prob:
                        self.min_prob = word_prob
                    if word_prob > self.max_prob:
                        self.max_prob = word_prob
                    prob = prob + math.log(word_prob)
            storage[label] = prob
        if storage['pos'] > storage['neg']:
            return 'pos', storage['pos']
        elif storage['pos'] == storage['neg']:
            return 'tie', storage['pos']
        else:
            return 'neg', storage['neg']

    def to_string(self, tokens):
        string = ''
        for word in tokens:
            if len(string) == 0:
                string = string + word
            else:
                string = string + ' ' + word
        return string

    def compute_probability(self, word, history, order, label, alpha):
        gram = str(order) + '-gram'
        prob_sequence = (self.class_unit_counts[label][history][word] + alpha)/(len(self.class_unit_counts[label][history]) + (alpha * len(self.vocab[label][gram])))
        prob_label = self.class_doc_counts[label]/self.total_doc_count
        return prob_sequence * prob_label

    def test(self, test_data, alpha, verbose=False):
        result = {}
        for item in test_data:
            tag, prob = self.classify_datum(item, alpha)
            result[item] = tag
            if verbose:
                print "=== " + item + " ===", 'BEST TAG: ' + tag, '=== LOG PROB: ' + str(prob)
        return result
