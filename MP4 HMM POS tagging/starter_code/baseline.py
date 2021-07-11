"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
import time
import math


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    pred_list_sentences = []

    # Follow each Tag's frequency using Dictionary 
    tag_freq = {}
    # Follow up words count using Dictionary; word dependancy on each tag
    tag_counts = {}

    # train side of the whole function 
    # fillilng up dictionaries 
    for sentence in train:
        for word, tag in sentence:
            # add tag into the tag_freq and count
            # we can use the if/else though we can also use try/except which is 
            # actually for error handling instaed. 

            # if tag exists/ is seen already in tag_freq dictionary, we increment the frequency, otherwise set it eqaul to 1
            try:
                tag_freq[tag] += 1
            except:
                tag_freq[tag] = 1

            # add word into tag_counts and count
            if word not in tag_counts:
                tag_counts[word] = {}
            try:
                tag_counts[word][tag] += 1
            except:
                tag_counts[word][tag] = 1

    freq_tag = max(tag_freq.keys(), key=(lambda key: tag_freq[key]))
    #print(lambda key: tag_freq[key])

    #print("Train = %s sec" % (time.time() - start))

    #print("tag_freq:", tag_freq, '\n\n')
    #print("tag_counts", tag_counts, '\n\n')

    #print(tag_counts, "tag_counts printed")
    #print(tag_freq, "tag_freq printed")
    #print(tag_freq.keys(), "tag_freq.keys printed")
    #print(freq_tag)


    # test side of the whole fucntion 
    for sentence in test:

        # temporary holder to predict sentences
        predicting_sentence = []

        for word in sentence:
            
            # we assume all no word,tag combination to be best_ta, the most frequent tag in the train set 
            fit_tag = freq_tag

            # we there is word found in word dictionary based on tag(which is tag_counts), then we set new fit_tag
            if word in tag_counts:
                #print(fit_tag, "current tag")
                fit_tag = max(tag_counts[word].keys(), key=(lambda key: tag_counts[word][key]))
                #print(fit_tag,"new tag")

            #fill up list of sentences with our prediction with most suitable tag expected/predicted.     
            predicting_sentence.append((word, fit_tag))
        pred_list_sentences.append(predicting_sentence)

    return pred_list_sentences

