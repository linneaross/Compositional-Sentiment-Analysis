from collections import defaultdict
import math
import numpy as np
import sys
import string
import re
# lexical sentiment resources
from senticnet5 import senticnet
#from pattern.en import sentiment

# NLTK part of speech tagger
import nltk
#nltk.download('averaged_perceptron_tagger')

# populating a dict that can be used for lexical sentiment values
# from Stanford Sentiment Treebank resource
treebank = {}
scores = {}
for line in open('dictionary.txt'):
    line_and_index = line.split('|')
    line = line_and_index[0].strip()
    line = line.lower()
    line = line.translate(None, string.punctuation)
    index = re.sub('\n','', line_and_index[1])
    treebank[line] = index
for line in open('sentiment_labels.txt'):
    index_and_score = line.split('|')
    score = re.sub('\n','', index_and_score[1])
    scores[index_and_score[0]] = score
for item in treebank:
    treebank[item] = scores[treebank[item]]

# populating a dict that can be used for lexical sentiment values
# from Sentiwords resource
sentiwords = {}
for line in open('sentiwords.txt'):
    word = line.split('#')
    word = word[0]
    score = re.split('\t|\n',line)
    score = score[1].strip()
    if re.match(r'(-*)(\d)(\.*)(\d*)', score):
        score = float(score)
        sentiwords[word] = score

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
        matrix = []
        for i in range(len(sentence)):
            matrix.append([])
            for j in range(len(sentence)):
                matrix[i].append([])
        for x in range(len(sentence)):
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
        # compositional tokenization function
        sentence = sentence.lower()
        sentence = sentence.strip()
        tokens = sentence.split()
        return tokens

    def create_word_tag(self, word):
        # generates a new tag for a word
        pos = word
        sentiment = self.word_sentiment(word)
        bp = None
        return Tag(pos, sentiment, bp)

    def tag_word(self, word):
        # DEPRECATED: was used before unary branching rules were added to ruleset
        # tag each word with part of speech and sentiment feature value
        tagged_pos = pos
        if word == 'not':
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
        # calculates sentiment for a word
        # hybrid model case:
        if self.hybrid == None:
            if word not in treebank:
                sentiment = 0.0
            else:
                prob = float(treebank[word])
                normalized = (2 * (prob-0)/(1-0)) - 1
                sentiment = normalized
        # pure lexical compositional case
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
        # CKY parsing algorithm
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
        # returns unary branching parents
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
        # returns possible parents for binary combinations
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
        # calculates binary combination sentiment
        sentiment = 0.0
        # neg case:
        if parent[:3] == 'Neg':
            sentiment = self.neg(down, self.neg_param)
        # conj case:
        elif left.get_pos() == 'Conj':
            sentiment = down.get_sentiment()
        # conjP case:
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
        return sentiment

    def neg(self, down, hyperparameter=1.0):
        # subtracting hyperparameter from sentiment score of the right-side argument
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
        # base case
        if left.get_sentiment() == 0.0 and down.get_sentiment() == 0:
            return 0.0
        elif left.get_sentiment() == 0.0 and down.get_sentiment() != 0:
            return down.get_sentiment()
        elif left.get_sentiment() != 0.0 and down.get_sentiment() == 0:
            return left.get_sentiment()
        else:
            return (left.get_sentiment() * weight) + (down.get_sentiment() * (1-weight))

    def test(self):
        # classify test data and return as a dict
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
