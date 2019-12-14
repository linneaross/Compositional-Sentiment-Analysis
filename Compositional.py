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

    def __init__(self, rules, test, hybrid_model=None, verbose=False,and_case=0.5, but_case=0.5, neg=1.3, v=1.0, np=0.7, s=0.75):
        self.rule_file = rules
        self.test_data = test
        self.hybrid = hybrid_model
        self.and_left_weight = and_case
        self.but_left_weight = but_case
        self.neg_param = neg
        self.v_weight = v
        self.np_weight = np
        self.s_weight = s
        self.rules = rules
        self.verbose = verbose

    def tag_sentence(self, sentence):
        # tag the sentence with words and populate the parsing matrix
        sentence = self.tokenize(sentence)
        #parts_of_speech = nltk.pos_tag(sentence)
        matrix = []
        for i in range(len(sentence)):
            matrix.append([])
            for j in range(len(sentence)):
                matrix[i].append([])
        for x in range(len(sentence)):
            #matrix[x][x] = [self.tag_word(sentence[x])]
            matrix[x][x].append(self.create_word_tag(sentence[x]))
        if self.verbose:
            print 'new matrix'
            print '=========='
            for row in range(len(matrix)):
                result = ''
                for col in range(len(matrix[row])):
                    if len(matrix[row][col]) > 0:
                        for item in matrix[row][col]:
                            result = result + ' ' + item.get_pos() + '\t'
                    else:
                        result += '[]\t'
                print result
            print '==========='
        return matrix

    def tokenize(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.strip()
        tokens = sentence.split()
        return tokens

    def create_word_tag(self, word):
        pos = word
        sentiment = self.word_sentiment(word)
        bp = None
        return Tag(pos, sentiment, bp)

    def tag_word(self, word):
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
            if word not in treebank:
                sentiment = 0.0
            else:
                prob = float(treebank[word])
                normalized = (2 * (prob-0)/(1-0)) - 1
                sentiment = normalized
        else:
            pos_percentage = self.hybrid.even_split(word)
            if pos_percentage > 0.5:
                sentiment = (2 * pos_percentage)-1
            elif pos_percentage < 0.5:
                sentiment = (2 * pos_percentage)-1
            else:
                sentiment = 0.0
        return sentiment

    def parse(self, matrix):
        unary_checked = []
        for row in reversed(range(len(matrix))):
            for column in (range(len(matrix))):
                if row != column and row<column:
                    for k in range(row,column):
                        for left_tag in matrix[row][k]:
                            if left_tag not in unary_checked:
                                matrix[row][k] += (self.get_parent_unary(left_tag))
                                unary_checked.append(left_tag)
                            for down_tag in matrix[k+1][column]:
                                if down_tag not in unary_checked:
                                    matrix[k+1][column] += (self.get_parent_unary(down_tag))
                                matrix[row][column] += (self.get_parent_binary(left_tag, down_tag))
                                #if len(self.get_parent_binary(left_tag, down_tag)) > 0:
                                    #print left_tag.get_pos(), down_tag.get_pos(), self.get_parent_binary(left_tag, down_tag)[0].get_pos()
                                #else:
                                    #print 'no results for ' + left_tag.get_pos(), down_tag.get_pos()
        if self.verbose:
            print 'parsed:'
            print '=========='
            for row in range(len(matrix)):
                result = ''
                for col in range(len(matrix[row])):
                    result = result + ' ST: '
                    if len(matrix[row][col]) > 0:
                        for item in matrix[row][col]:
                            result = result + ' ' + item.get_pos() + '\t\t\t'
                    else:
                        result += '[]\t\t\t'
                print result
            print '==========='
        return matrix[0][len(matrix)-1], matrix

    def get_parent_unary(self, child):
        result = []
        right_side = [child.get_pos()]
        for left_side in self.rules:
            if right_side in self.rules[left_side]:
                pos = left_side
                sentiment = child.get_sentiment()
                tag = Tag(pos, sentiment, child)
                result.append(tag)
        return result

    def get_parent_binary(self, left, down):
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
        if parent[:3] == 'Neg':
            sentiment = self.neg(down, self.neg_param)
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
            sentiment = self.general(left, down, 0.7)
        #print 'combining sentiment of ' + left.get_pos() + ' ' + down.get_pos()
        #print parent, sentiment
        return sentiment

    def neg(self, down, hyperparameter=1.0):
        # sliding weight depending on the value - higher sentiment score, higher weight
        # the value for change is the amount the original score will be changed
        # to find this, we square the sentiment of the negated nonterminal
        if down.get_sentiment() > 0:
            return down.get_sentiment()-hyperparameter
        elif down.get_sentiment() < 0:
            return down.get_sentiment()+hyperparameter
        else:
            return 0.0


    def conj(self, left, down):
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

    def test(self):
        #self.parse_rules()
        tagged_sentences = {}
        for item in self.test_data:
            tagged = self.tag_sentence(item)
            tags, matrix = self.parse(tagged)
            final_sentiment = tags[0].get_sentiment()
            if final_sentiment > 0:
                cast_sentiment = 'pos'
            elif final_sentiment < 0:
                cast_sentiment = 'neg'
            else:
                cast_sentiment = 'neut'
            if self.verbose:
                print 'SENTENCE: ' + item.strip()
                print 'FINAL SENTIMENT: ' + str(final_sentiment)
                print 'SENTIMENT TAG: ' + str(cast_sentiment)
                print 'FINAL PARSE: ' + tags[0].get_pos()
                print '====='
            tagged_sentences[item] = cast_sentiment
            if len(tags) == 0 and self.verbose:
                print 'No parse found!'
                print '====='
        return tagged_sentences
