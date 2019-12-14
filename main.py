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

# sentiwords lexical sentiment resource - not currently in use
sentiwords = {}
for line in open('sentiwords.txt'):
    word = line.split('#')
    word = word[0]
    score = re.split('\t|\n',line)
    score = score[1].strip()
    if re.match(r'(-*)(\d)(\.*)(\d*)', score):
        score = float(score)
        sentiwords[word] = score

# statistical training data:
pos_data = open(sys.argv[1])
neg_data = open(sys.argv[2])
# compositional ruleset:
rules = open(sys.argv[3])
# testing data for every model
testing = open(sys.argv[4])
test_dict = {}
for line in testing:
    components = line.split('#')
    components[1] = re.sub('\n','', components[1])
    test_dict[components[1]] = components[0]

def parse_rules(rule_file):
    rules = {}
    for line in rule_file:
        line = line.strip()
        tokenized = line.split()
        if len(tokenized) > 0:
            left_side = tokenized[0]
            right_side = tokenized[1:]
            if left_side not in rules:
                rules[left_side] = []
            rules[left_side].append(right_side)
    return rules

rules = parse_rules(rules)

nb = PureStatistical(pos_data, neg_data, test_dict, 3, .001)
nb.train_model()
print 'NB'
print nb_results

comp = Compositional(rules, test_dict, None, False, 0.5, 0.0, 0.9, 0.9, 0.7, 0.75)
comp_results = comp.test()
print 'LEX-COMP'
print comp_results

hybrid = Compositional(rules, test_dict, nb, False, 0.5, 0.4, 0.6, 0.2, 0.7, 0.75)
hybrid_results = hybrid.test()
print 'STAT-COMP'
print hybrid_results

def test_models(test_dict, nb_output, comp_output, hybrid_output=None):
    if hybrid_output != None:
        result_dict = {'nb': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f-score': 0.0}, 'comp': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f-score': 0.0}, 'hybrid': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f-score': 0.0}}
        # test nb model
        result_count = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for item in nb_output:
            result_count += 1
            if nb_output[item] == test_dict[item] and nb_output[item] == 'pos':
                tp += 1.0
            elif nb_output[item] == test_dict[item]:
                tn += 1.0
            elif nb_output[item] != test_dict[item]:
                if test_dict[item] == 'pos':
                    fp += 1.0
                else:
                    fn += 1.0
        result_dict['nb']['accuracy'] = (tp+tn)/result_count
        result_dict['nb']['precision'] = tp/(tp+fp)
        result_dict['nb']['recall'] = tp/(tp+fn)
        result_dict['nb']['f-score'] = 2 * ((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))
        # test comp model
        result_count = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for item in comp_output:
            result_count += 1
            if comp_output[item] == test_dict[item] and comp_output[item] == 'pos':
                tp += 1.0
            elif comp_output[item] == test_dict[item]:
                tn += 1.0
            elif comp_output[item] != test_dict[item]:
                if test_dict[item] == 'neg':
                    fn += 1.0
                else:
                    fp += 1.0
        result_dict['comp']['accuracy'] = (tp+tn)/result_count
        result_dict['comp']['precision'] = tp/(tp+fp)
        result_dict['comp']['recall'] = tp/(tp+fn)
        result_dict['comp']['f-score'] = 2 * ((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))
        # test hybrid model
        result_count = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for item in comp_output:
            result_count += 1
            if hybrid_output[item] == test_dict[item] and hybrid_output[item] == 'pos':
                tp += 1.0
            elif hybrid_output[item] == test_dict[item]:
                tn += 1.0
            elif hybrid_output[item] != test_dict[item]:
                if test_dict[item] == 'pos':
                    fp += 1.0
                else:
                    fn += 1.0
        result_dict['hybrid']['accuracy'] = (tp+tn)/result_count
        result_dict['hybrid']['precision'] = tp/(tp+fp)
        result_dict['hybrid']['recall'] = tp/(tp+fn)
        result_dict['hybrid']['f-score'] = 2 * ((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))
    else:
        result_dict = {'nb': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f-score': 0.0}, 'comp': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f-score': 0.0}}
        # test nb model
        result_count = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for item in nb_output:
            result_count += 1
            if nb_output[item] == test_dict[item] and nb_output[item] == 'pos':
                tp += 1.0
            elif nb_output[item] == test_dict[item]:
                tn += 1.0
            elif nb_output[item] != test_dict[item]:
                if test_dict[item] == 'neg':
                    fn += 1.0
                else:
                    fp += 1.0
        result_dict['nb']['accuracy'] = (tp+tn)/result_count
        result_dict['nb']['precision'] = tp/(tp+fp)
        result_dict['nb']['recall'] = tp/(tp+fn)
        result_dict['nb']['f-score'] = 2 * ((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))
        # test comp model
        result_count = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for item in comp_output:
            result_count += 1
            if comp_output[item] == test_dict[item] and comp_output[item] == 'pos':
                tp += 1.0
            elif comp_output[item] == test_dict[item]:
                tn += 1.0
            elif comp_output[item] != test_dict[item]:
                if test_dict[item] == 'neg':
                    fn += 1.0
                else:
                    fp += 1.0
        result_dict['comp']['accuracy'] = (tp+tn)/result_count
        result_dict['comp']['precision'] = tp/(tp+fp)
        result_dict['comp']['recall'] = tp/(tp+fn)
        result_dict['comp']['f-score'] = 2 * ((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))

    return result_dict

def report_results(test_results):
    for model in test_results:
        print '==================='
        print 'MODEL: ' + model
        print 'ACCURACY: ' + str(test_results[model]['accuracy'])
        print 'PRECISION: ' + str(test_results[model]['precision'])
        print 'RECALL: ' + str(test_results[model]['recall'])
        print 'F-SCORE: ' + str(test_results[model]['f-score'])

print 'PURE-STAT length: ' + str(len(nb_results))
print 'LEX-COMP length: ' + str(len(comp_results))
print 'STAT-COMP length: ' + str(len(hybrid_results))
for item in nb_results:
    if item not in comp_results:
        print 'Item not parsed by LEX-COMP: ' + item
    elif item not in hybrid_results:
        print 'Item not parsed by STAT-COMP: ' + item
results = test_models(test_dict, nb_results, comp_results, hybrid_results)
report_results(results)
