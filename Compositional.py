from collections import defaultdict
import math
import numpy as np
import sys
import string
import re
# lexical sentiment resources
from senticnet5 import senticnet
#from pattern.en import sentiment
# part of speech tagger
import nltk
#nltk.download('averaged_perceptron_tagger')

class Compositional:

    def __init__(self, rules, test, hybrid_model=None, and_case=0.5, but_case=0.3, v=0.6, np=0.7, s=0.4):
        self.rule_file = rules
        self.test_data = test
        self.hybrid = hybrid_model
        self.and_left_weight = and_case
        self.but_left_weight = but_case
        self.v_weight = v
        self.np_weight = np
        self.s_weight = s
        self.rules = rules

    def tag_sentence(self, sentence):
        # tag the sentence with words and populate the parsing matrix
        sentence = self.tokenize(sentence)
        parts_of_speech = nltk.pos_tag(sentence)
        matrix = []
        for i in range(len(sentence)):
            matrix.append([])
            for j in range(len(sentence)):
                matrix[i].append([])
        for x in range(len(parts_of_speech)):
            #matrix[x][x] = [self.tag_word(sentence[x])]
            matrix[x][x].append(self.tag_word(parts_of_speech[x][0], parts_of_speech[x][1]))
        return matrix

    def tokenize(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.strip()
        tokens = sentence.split()
        return tokens
    '''
    def parse_rules(self):
        print 'parsing rules!'
        rules = {}
        for line in self.rule_file:
            print line
            line = line.strip()
            tokenized = line.split()
            if len(tokenized) > 0:
                left_side = tokenized[0]
                right_side = tokenized[1:]
                if left_side not in rules:
                    rules[left_side] = []
                rules[left_side].append(right_side)
        self.rules = rules
        print self.rules
    '''
    def tag_word(self, word, pos):
        # tag each word with part of speech and sentiment feature value
        tagged_pos = pos
        if word == 'not':
            # EXPAND THIS
            pos = 'Neg'
        else:
            if tagged_pos == 'CC':
                pos = 'Conj'
            elif tagged_pos == 'NN' or tagged_pos == 'NNS' or tagged_pos =='NNP' or tagged_pos == 'NNPS' or tagged_pos =='PRP' or tagged_pos == 'WP' or tagged_pos =='WP$':
                pos = 'N'
            elif tagged_pos == 'VB' or tagged_pos == 'VBD' or tagged_pos == 'VBG' or tagged_pos == 'VBN' or tagged_pos == 'TO' or tagged_pos == 'VBP' or tagged_pos =='VBZ':
                pos = 'V'
            elif tagged_pos == 'RB' or tagged_pos == 'RBR' or tagged_pos =='RBS' or tagged_pos == 'WRB':
                pos = 'Adv'
            elif tagged_pos == 'JJ' or tagged_pos == 'JJR' or tagged_pos == 'JJS' or tagged_pos =='PRP$':
                pos = 'Adj'
            elif tagged_pos == 'WDT' or tagged_pos == 'DT':
                pos = 'Det'
            elif tagged_pos == 'IN':
                pos = 'Prep'
        sentiment = self.word_sentiment(word)
        if pos == 'Det':
            sentiment = 0.0
        backpointer = word
        tag = Tag(pos, sentiment, backpointer)
        #if self.hybrid != None:
            #print word, pos, sentiment
        return tag

    def word_sentiment(self, word):
        if self.hybrid == None:
            if word not in sentiwords:
                sentiment = 0.0
            else:
                sentiment = sentiwords[word]
            # using senticnet
            '''
            if word not in senticnet:
                sentiment = 0.0
            else:
                sentiment = float(senticnet[word][7])
            '''
            # making all relatively neutral sentiment values true neutral
            if sentiment < 0.2 and sentiment > -0.2:
                sentiment = 0.0
        else:
            tag, pos_prob, neg_prob = self.hybrid.classify_word(word, .001)
            '''
            if pos_prob < neg_prob:
                sentiment = 2 * ((neg_prob-self.hybrid.min_prob)/(self.hybrid.max_prob-self.hybrid.min_prob)-1)
            else:
                sentiment = 2 * ((pos_prob-self.hybrid.min_prob)/(self.hybrid.max_prob-self.hybrid.min_prob)-1) +
            '''
            if pos_prob == neg_prob or (pos_prob - neg_prob) < 0.00001 and (pos_prob - neg_prob) > -0.00001:
                sentiment = 0.0
            elif tag == 'pos':
                sentiment = 0.8
            elif tag == 'neg':
                sentiment = -0.3
            #print word
            #print sentiment
        return sentiment

    def parse(self, matrix):
        for row in reversed(range(len(matrix))):
            for column in reversed(range(len(matrix))):
                if row != column and row<column:
                    #for k in range(row):
                    for k in range(row,column):
                        for left_tag in matrix[row][k]:
                            for down_tag in matrix[k+1][column]:
                                matrix[row][column] += (self.get_parent(left_tag, down_tag))
        #if self.hybrid != None:
            #print 'NEW MATRIX'
            #print matrix[0][len(matrix)-1]
        #print 'final parse found?'
        #print len(matrix[0][len(matrix)-1]) > 0
        return matrix[0][len(matrix)-1], matrix

    def get_parent(self, left, down):
        result = []
        right_side = [left.get_pos(), down.get_pos()]
        for left_side in self.rules:
            if right_side in self.rules[left_side]:
                pos = left_side
                sentiment = self.combine_sentiment(left, down, pos)
                tag = Tag(pos, sentiment, [left, down])
                result.append(tag)
        return result

    def combine_sentiment(self, left, down, parent):
        sentiment = 0.0
        # neg case:
        if parent == 'NegP':
            sentiment = self.neg(down)
        # conj case:
        elif left.get_pos() == 'Conj':
            sentiment = down.get_sentiment()
        # conjP case: CHECK THIS SLICING!
        elif down.get_pos()[:4] == 'Conj':
            sentiment = self.conj(left, down)
        # V case
        elif parent == 'VP':
            sentiment = self.general(left, down, self.v_weight)
        # np case
        elif parent == 'NP':
            sentiment = self.general(left, down, self.np_weight)
        # sentence case
        elif parent == 'S':
            sentiment = self.general(left, down, self.s_weight)
        # general case
        else:
            # equal weights to both sides
            sentiment = self.general(left, down, 0.5)
        #print 'combining sentiment of ' + left.get_pos() + ' ' + down.get_pos()
        #print parent, sentiment
        return sentiment

    def neg(self, down, hyperparameter=0.35):
        # fix this actual formula
        return (down.get_sentiment() - hyperparameter)

    def conj(self, left, down):
        # check about neutrality !!!!!
        # same sign on each side - 'AND' CASE
        if (left.get_sentiment() * down.get_sentiment()) >= 0:
            return ((left.get_sentiment() * self.and_left_weight) + (down.get_sentiment() * (1-self.and_left_weight)))
        # opposite sign on each side - 'BUT' CASE
        else:
            return ((left.get_sentiment() * self.but_left_weight) + (down.get_sentiment() * (1-self.but_left_weight)))

    def general(self, left, down, weight=0.5):
        if left.get_sentiment() == 0.0 and down.get_sentiment() == 0:
            return 0.0
        elif left.get_sentiment() == 0.0 and down.get_sentiment() != 0:
            return down.get_sentiment()
        elif left.get_sentiment() != 0.0 and down.get_sentiment() == 0:
            return left.get_sentiment()
        else:
            return (left.get_sentiment() * weight) + (down.get_sentiment() * (1-weight))

    def test(self, verbose=False):
        #self.parse_rules()
        tagged_sentences = {}
        for item in self.test_data:
            tagged = self.tag_sentence(item)
            tags, matrix = self.parse(tagged)
            for tag in tags:
                final_sentiment = tag.get_sentiment()
                if final_sentiment > 0:
                    cast_sentiment = 'pos'
                elif final_sentiment < 0:
                    cast_sentiment = 'neg'
                else:
                    cast_sentiment = 'neut'
                if verbose:
                    print 'SENTENCE: ' + item.strip()
                    print 'FINAL SENTIMENT: ' + str(final_sentiment)
                    print 'SENTIMENT TAG: ' + str(cast_sentiment)
                    print 'FINAL PARSE: ' + tag.get_pos()
                tagged_sentences[item] = cast_sentiment
            if len(tags) == 0 and verbose:
                print 'No parse found!'
        return tagged_sentences
