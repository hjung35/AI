# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """

    # return variable 
    uni_pred_label = []

    # create probabilities from train set in order of positive to negative reviews 
    positive_prob_words, positive_prob_UNK = set_uni_probabilities(train_set, train_labels, 1, smoothing_parameter)    
    negative_prob_words, negative_prob_UNK = set_uni_probabilities(train_set, train_labels, 0, smoothing_parameter)

    # MAP values calculation
    for i in range(len(dev_set)):
        positive_prob = 0.0
        negative_prob = 0.0

        for word in dev_set[i]:
            if(word in positive_prob_words):
                positive_prob += positive_prob_words[word] 
            elif(word not in positive_prob_words): 
                positive_prob += positive_prob_UNK
            
            if(word in negative_prob_words):
                negative_prob += negative_prob_words[word] 
            elif(word not in negative_prob_words):
                negative_prob += negative_prob_UNK

        #posterior probabilities adding up 
        positive_prob += math.log(pos_prior)
        negative_prob += math.log(1-pos_prior)
        
        # setup return label 
        if positive_prob >= negative_prob:
            uni_pred_label.append(1)
        else:
            uni_pred_label.append(0)
 
    return uni_pred_label


# creates list of words with corresponding number = ('cat'in spam emails / all words in spam emails)
def set_uni_probabilities(train_set, train_labels, pos_neg, smoothing_param):
    # Add labeled words into wordList and count numbers
    words_list = {}
    total_num_words = 0

    # Count words into words_list
    for i in range(len(train_labels)):

        # if our train label is against our target label category whether they are positive or negative, move on
        if (train_labels[i] != pos_neg):
            continue

        # filling up words_list
        for word in train_set[i]:
            total_num_words += 1

            # if we have seen the word before, increment the count
            if word in words_list:
                words_list[word] += 1 
            # else we set them seen once
            else: 
                words_list[word] = 1   

    # calculate UNK probability from words_list
    prob_denom = total_num_words + (smoothing_param * (len(words_list) + 1))
    # using logs to prevent underflow 
    prob_UNK = math.log(smoothing_param / prob_denom)

    # given formula, setup the words_list to return 
    for word, w_count in words_list.items():
        words_list[word] = math.log((w_count + smoothing_param) / prob_denom)
    return words_list, prob_UNK



def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1, bigram_smoothing_parameter=0.007, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """

    # Unigram term temp variable in the formula 
    uni_pro_labels = []
    uni_neg_labels = []

    # BiGram term temp variablem in the formula 
    bi_pro_labels = []
    bi_neg_labels = []

    # return variable
    bi_pred_labels = []

    # create unigram probabilities from train set in order of positive to negative reviews 
    positive_prob_words, positive_prob_UNK = set_uni_probabilities(train_set, train_labels, 1, unigram_smoothing_parameter)    
    negative_prob_words, negative_prob_UNK = set_uni_probabilities(train_set, train_labels, 0, unigram_smoothing_parameter)

    # MAP values calculation for unigram term 
    for review in dev_set:
        positive_prob = 0.0
        negative_prob = 0.0

        for word in review:
            if(word in positive_prob_words):
                positive_prob += positive_prob_words[word] 
            elif(word not in positive_prob_words): 
                positive_prob += positive_prob_UNK
            
            if(word in negative_prob_words):
                negative_prob += negative_prob_words[word] 
            elif(word not in negative_prob_words):
                negative_prob += negative_prob_UNK

        # store negative and positive labels separately for later calculation
        # we dont add posterior probabilities here, we still have leftover calculation left
        uni_pro_labels.append(positive_prob)
        uni_neg_labels.append(negative_prob)


    # create Bigram probabilities from train set in order of positive to negative reviews 
    positive_prob_words_bi, positive_prob_UNK_bi = set_bi_probabilities(train_set, train_labels, 1, bigram_smoothing_parameter)    
    negative_prob_words_bi, negative_prob_UNK_bi = set_bi_probabilities(train_set, train_labels, 0, bigram_smoothing_parameter)

    for review in dev_set:
        positive_prob = 0.0
        negative_prob = 0.0

        for item in [review[i:i+2] for i in range(len(review)-1)]:
            # combine each adjacent words into a single word count here
            word = str(item)
            
            if(word in positive_prob_words_bi):
                positive_prob += positive_prob_words_bi[word] 
            elif(word not in positive_prob_words_bi): 
                positive_prob += positive_prob_UNK_bi
            
            if(word in negative_prob_words_bi):
                negative_prob += negative_prob_words_bi[word] 
            elif(word not in negative_prob_words_bi):
                negative_prob += negative_prob_UNK_bi
            
        bi_pro_labels.append(positive_prob)
        bi_neg_labels.append(negative_prob)


    # finally calculate what we wanted to output 
    for i in range(len(dev_set)):
        positive_prob = (1-bigram_lambda) * uni_pro_labels[i] + (bigram_lambda * bi_pro_labels[i])
        negative_prob = (1-bigram_lambda) * uni_neg_labels[i] + (bigram_lambda * bi_neg_labels[i])

        # finally adding up posterior probabilities
        positive_prob += math.log(pos_prior)
        negative_prob += math.log(1-pos_prior)

        # similar to unigram method, report 0 or 1 based on MAP positive and negative values
        if positive_prob >= negative_prob:
            bi_pred_labels.append(1)
        else:
            bi_pred_labels.append(0)

    return bi_pred_labels


# Very similar to set_uni_probabilities, but here we have to combine two words into a one word counting
# rest methods are very similar 
def set_bi_probabilities(train_set, train_labels, pos_neg, smoothing_param):
    # Add labeled words into wordList and count numbers
    words_bi_list = {}
    total_num_words = 0

    # Count words into words_list
    for i in range(len(train_labels)):

        # if our train label is against our target label category whether they are positive or negative, move on
        if (train_labels[i] != pos_neg):
            continue

        review = train_set[i]

        for item in [review[i: i+2] for i in range(len(review)-1)]:
            #Bigram, calculating total num words
            word = str(item)
            total_num_words += 1

            # if we have seen the word before, increment the count; here the word is bi-words not a singular uniword
            if word in words_bi_list:
                words_bi_list[word] += 1 
            # else we set them seen once
            else: 
                words_bi_list[word] = 1   

    # calculate Bigram UNK probability from words_list
    prob_denom = total_num_words + (smoothing_param * (len(words_bi_list) + 1))
    # using logs to prevent underflow 
    prob_bi_UNK = math.log(smoothing_param / prob_denom)

    # given formula, setup the words_list to return 
    for word, w_count in words_bi_list.items():
        words_bi_list[word] = math.log((w_count + smoothing_param) / prob_denom)
    return words_bi_list, prob_bi_UNK    