"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import Counter

# Referrence algorithm : 1) https://web.stanford.edu/~jurafsky/slp3/8.pdf ; perusall reading we had
                #     2) https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    # 1. INITIALIZATION
    # sf is a smoothing factor
    sf = 0.000001

    # return variable 
    pred_list_sentences = []

    # Dictionaries to keep probabilities of each type:
    transmission_probability = {}  # Transmission Probability
    initial_probability = {}  # Initial Probability
    # emission_probability = {}

    # Dictionaries to keep track of occurrence counts using massive useful given library Counter
    first_word_sentence = Counter()  # (first_word_of_sentence, count)
    words = Counter()                # (word, count)
    tags = Counter()                 # (tag, count)
    word_tag = Counter()             # ((word, tag), count)
    cur_prev = Counter()             # ((cur, prev), count)


    # 2. TRAINING SIDE
    # TRAINING SESSION
    #print("======= Training Started =======")

    # very similar to the baseline algorithm
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

            # Increment word, tag, word_tag(a.k.a word, tag) count
            words[cur_word] += 1
            tags[cur_tag] += 1
            word_tag[(cur_word, cur_tag)] += 1

            # Increment (cur,next) count
            if pair_idx + 1 < max_sentence_length:
                cur_prev[(cur_tag, sentence[pair_idx+1][1])] += 1
            pair_idx += 1

    
    # 3. PROBABILITIES
    # Probabilities calculation; initial and transmission probabilities 
    # logs are as purpose of avoiding underestimate as previous MPs
    
    tag_len = len(tags)
    for cur_tag in tags:
        for prev_tag in tags:
            numerator = cur_prev[(prev_tag, cur_tag)] + sf
            denominator = tags[prev_tag] + (sf*tag_len)
            transmission_probability[(cur_tag, prev_tag)] = math.log(numerator/denominator)

        numerator = first_word_sentence[cur_tag] + sf
        denominator = sum(first_word_sentence.values()) + (sf*tag_len)
        initial_probability[cur_tag] = math.log(numerator/denominator)


    # 4. TEST SIDE
    # APPLYING SESSION
    #print("======= Applying Started =======")

    for sentence in test:

        # initialization
        viterbi = {}
        backpointer = {}
        best_path_max = {}

        first_word = sentence[0]
        max_time_step = range(len(sentence))
        max_sentence_length2 = len(sentence)-1

        best_path = []
        predicted_sentence = []

        word_len = len(words)

        # probabilities calculation 
        for cur_tag in tags:
            numerator = word_tag[(first_word, cur_tag)] + sf
            denominator = tags[cur_tag] + (sf*word_len)
            emission_probability = math.log(numerator/denominator)

            viterbi[(cur_tag, 0)] = initial_probability[cur_tag] + emission_probability
            backpointer[(cur_tag, 0)] = 0

        # Recursion steps; calculation continues 
        for t in max_time_step:
            if t == 0:
                continue

            for cur_tag in tags:
                # get max probability of previous time step
                temp_prob = {}
                for prev_tag in tags:
                    numerator = word_tag[(sentence[t], cur_tag)] + sf
                    denominator = tags[cur_tag] + (sf*word_len)
                    emission_probability = math.log(numerator/denominator)
                    temp_prob[prev_tag] = transmission_probability[(cur_tag, prev_tag)] + emission_probability + viterbi[(prev_tag, t-1)]

                # important, dependancy#1; find the max probability from previous time step
                viterbi[(cur_tag, t)] = max(temp_prob.values())

                # find such key of max tag into back pointer     
                backpointer[(cur_tag, t)] = (max(temp_prob, key=temp_prob.get), t-1)


     # 5. WRAP-UP 
        # bring the whole list of most probabble probabilities for each tag at max_sentence_length
        for cur_tag in tags:
            best_path_max[(cur_tag, max_sentence_length2)] = viterbi[(cur_tag, max_sentence_length2)]

        # find the proper index of the best probability
        best_path_index = max(best_path_max, key=best_path_max.get)

        # follows backpointer[] for backtracking
        while best_path_index != 0:
            best_path.insert(0, best_path_index[0])
            best_path_index = backpointer[best_path_index]

        # Last step, filling up for returning output
        for index in max_time_step:
            predicted_sentence.append((sentence[index], best_path[index]))

        pred_list_sentences.append(predicted_sentence)

    return pred_list_sentences
