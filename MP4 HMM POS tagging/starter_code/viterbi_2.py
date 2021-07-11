"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import math
from collections import Counter

# Referrence algorithm : 1) https://web.stanford.edu/~jurafsky/slp3/8.pdf ; perusall reading we had
                #     2) https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # 1. INITIALIZATION
    # sf is a smoothing factor
    sf = 1e-7

    # return variable
    pred_list_sentences = [] 

    # Dictionaries to keep probabilities of each type:
    transmission_probability = {}     # Transmission Probability
    initial_probability = {}      # Initial Probability
    hapax_probability = {}     # Hapax word tag probabilities

    # Dictionary to keep count of occurrence
    first_word_sentence = Counter()            # (first_word_of_sentence, count)
    words = Counter()               # (word, count)
    tags = Counter()                # (tag, count)
    word_tag = Counter()         # ((word, tag), count)
    cur_prev = Counter()        # ((cur, prev), count)
    hapax_word_tags = Counter()     # (hapax-word-tag, count)

    # 2. TRAINING SIDE
    # TRAINING SESSION
    #print("======= Training Started =======")

    for sentence in train:

        max_sentence_length = len(sentence)-1
        pair_idx = 0

        # increment first word of sentence's tag / 0:word, 1:tag 
        first_word_tag = sentence[0][1]        
        first_word_sentence[first_word_tag] += 1        

        # filling up stage
        for word, tag in sentence:
            cur_word = word
            cur_tag = tag

            # Increment word, tag, (word, tag) count
            words[cur_word] += 1
            tags[cur_tag] += 1
            word_tag[(cur_word, cur_tag)] += 1

            # Increment (cur,next) count
            if pair_idx == max_sentence_length:
                continue

            else:
                pair_idx += 1
                cur_prev[(cur_tag, sentence[pair_idx][1])] += 1


    # 3. HAPAX EXCEPTION DEALING
    # Probabilities calculation; Hapax probability  
    
    # Simple count hapax words and its tags. First, create list of hapax words to calculate the probability
    hapax_words = []
    for word in words:
        if words[word] == 1:
            hapax_words.insert(0, word)

    # Find and incrememnt those hapax words' tags
    for tag in tags:
        for hapax_word in hapax_words:
            hapax_word_tags[tag] += word_tag[(hapax_word, tag)]

    # 4. PROBABILITIES
    # Probabilities calculation; initia, transmission, hapax probabilities 
    # logs are as purpose of avoiding underestimate as previous MPs
    # similar to viterbi_1, just a small modification

    tag_len = len(tags)
    for cur_tag in tags:
        for prev_tag in tags:
            numerator = cur_prev[(prev_tag, cur_tag)] + sf
            denominator = tags[prev_tag] + (sf*tag_len)
            transmission_probability[(cur_tag, prev_tag)] = math.log(numerator/denominator)

        numerator = first_word_sentence[cur_tag] + sf
        denominator = sum(first_word_sentence.values()) + (sf*tag_len)
        initial_probability[cur_tag] = math.log(numerator / denominator)

        numerator = hapax_word_tags[cur_tag] + sf
        denominator = len(hapax_words) + (sf*tag_len)
        hapax_probability[cur_tag] = numerator/denominator

    # 4. TEST SIDE
    # APPLYING SESSION
    #print("======= Applying Started =======")
    for sentence in test:

        #initialization
        viterbi = {}
        backpointer = {}
        best_path_max = {}

        first_word = sentence[0]
        max_time_step = range(len(sentence))
        max_sentence_length2 = len(sentence) - 1

        best_path = []
        predicted_sentence = []

        word_len = len(words)

        # probabilities calculation
        for cur_tag in tags:
            # Get hapax distribution factor
            hapax_sf = sf * hapax_probability[cur_tag]

            # finding new emission probabilities with new smoothing factor 
            numerator = word_tag[(first_word, cur_tag)] + hapax_sf
            denominator = tags[cur_tag] + hapax_sf * word_len
            emission_probability = math.log(numerator / denominator)

            viterbi[(cur_tag, 0)] = initial_probability[cur_tag] + emission_probability
            backpointer[(cur_tag, 0)] = 0

        # Recursion steps; calculation continues 
        for t in max_time_step:
            if t == 0:
                continue

            for cur_tag in tags:
                # Get hapax distribution factor
                hapax_sf = sf * hapax_probability[cur_tag]

                # get max probability of previous time step
                prev_max_prob = {}
                for prev_tag in tags:
                    numerator = word_tag[(sentence[t], cur_tag)] + hapax_sf
                    denominator = tags[cur_tag] + (hapax_sf*word_len)
                    emission_probability = math.log(numerator/denominator)
                    prev_max_prob[prev_tag] = transmission_probability[(cur_tag, prev_tag)] + emission_probability + viterbi[(prev_tag, t-1)]

                # important, dependancy#1; find the max probability from previous time step
                viterbi[(cur_tag, t)] = max(prev_max_prob.values())

                # find such key of max tag into back pointer     
                backpointer[(cur_tag, t)] = (max(prev_max_prob, key=prev_max_prob.get), t-1)

     # 5. WRAP-UP 
        # bring the whole list of most probabble probabilities for each tag at max_sentence_length
        for cur_tag in tags:
            best_path_max[(cur_tag, max_sentence_length2)] = viterbi[(cur_tag, max_sentence_length2)]

        # find the proper index of the best probability
        bestpathpointer = max(best_path_max, key=best_path_max.get)

        # follows backpointer[] for backtracking
        while bestpathpointer != 0:
            best_path.insert(0, bestpathpointer[0])
            bestpathpointer = backpointer[bestpathpointer]

        pair_idx = 0
        for pair in sentence:
            predicted_sentence.append((sentence[pair_idx], best_path[pair_idx]))
            pair_idx += 1

        pred_list_sentences.append(predicted_sentence)

    return pred_list_sentences 