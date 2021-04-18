import nltk
from nltk.corpus import stopwords
import json
from collections import Counter
from math import floor
from random import shuffle
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.parse.stanford import StanfordDependencyParser
import os
from scipy import spatial
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import KFold
import numpy as np


def feature_split(featureSets):
    shuffle(featureSets)
    feature_list = []
    add = floor(len(featureSets)*(0.2))
    for count in range(5):
        if count == 0:
            temp = featureSets[0:add]
            feature_list.append(temp)
        elif count != 5:
            temp = featureSets[add:add+floor(len(featureSets)*(0.2))]
            feature_list.append(temp)
            add += floor(len(featureSets)*(0.2))
        else:
            temp = featureSets[add:]
            feature_list.append(temp)
    return feature_list


def k_split_10(feature_list):
    accuracy_list = []
    for count in range(5):
        training_set = []
        test_set = feature_list[count]
        for count2 in range(5):
            if count == count2:
                continue
            else:
                training_set.extend(feature_list[count2])
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        # classifier.show_most_informative_features(50)
        nltk.classify.accuracy(classifier, test_set)
        print(f'itr:{i}\n-------------------------------------------------------------------\nAccuracy:',
              nltk.classify.accuracy(classifier, test_set))
        accuracy_list.append(nltk.classify.accuracy(classifier, test_set))
    print(f'$$$   accuracy AVG: {sum(accuracy_list)/len(accuracy_list)}   $$$')

    return accuracy_list


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


tknzr = TweetTokenizer()
tknzr_punc = RegexpTokenizer(r'\w+')
new_stopwords = stopwords.words('english')
new_stopwords.append('the')


smsfilepath = os.getcwd()+'\customized_reviews.json'
data = []
with open('customized_reviews.json') as json_file:
    data = json.load(json_file)

subjective = []
objective = []
text = []
wordArray = []

for i in range(len(data)):
    subjective.append(data[i]['subjective'])
    objective.append(data[i]['objective'])
    text.append(data[i]['text'])


for sentence in text:
    wordArray.extend(tknzr_punc.tokenize(sentence))


wordArray = [word.lower()
             for word in wordArray if word.lower() not in new_stopwords]
wordArray = [word for word in wordArray if not any(c.isdigit() for c in word)]
wordArray = [t for t in wordArray if len(t) > 1]
pos_tags = pos_tag(wordArray)
wordArray = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1]))
             for t in pos_tags]
#wordArray = " ".join(wordArray)


wordFreq = nltk.FreqDist(w.lower() for w in wordArray)
# wordFreq.plot(50,cumulative=False)
wordFeatures = list(wordFreq)[:3420]


def dan_is_dingus(fdist):
    final_dan_list = []
    fdist = fdist.most_common(len(fdist))
    for i in range(len(fdist)):
        appendDANLUL = list(fdist[i])
        appendDANLUL[i] = f'{i}'
        final_dan_list.append(appendDANLUL)
    return final_dan_list


def document_features(document):
    features = {}
    # want to get all of the unique words in a given message
    documentWords = set(document)
    for word in wordFeatures:
        features['contains({})'.format(word)] = (word in documentWords)
    return features


featureSets = [(document_features(text[i]), objective[i])
               for i in range(0, len(text))]
# shuffle(featureSets)
#trainSet,testSet = featureSets[:floor(len(featureSets)*.9)], featureSets[floor(len(featureSets)*.1):]
#classifier = nltk.NaiveBayesClassifier.train(trainSet)
#print(f'itr:{i}\n-------------------------------------------------------------------\nAccuracy:', nltk.classify.accuracy(classifier, testSet))
# classifier.show_most_informative_features(50)
#nltk.classify.accuracy(classifier, testSet)


#feature_list = feature_split(featureSets)
#accuracy_list = k_split_10(feature_list)

tru_count = 0
false_count = 0

binary_list = []
binary_list_objective = []
binary_list_subjective = []
for indx, feature_set in enumerate(featureSets):
    temp_binary = []
    for key in feature_set[0]:
        val = feature_set[0][key]
        if val == False:
            temp_binary.append(0)
            false_count += 1
        elif val == True:
            temp_binary.append(1)
            tru_count += 1
        else:
            print("shit!")
            break
    if objective[indx] == 1:
        temp_tuple = (temp_binary, 'Objective Review')
        binary_list_objective.append(temp_binary)
        # binary_list_subjective.append('NaN')
    elif objective[indx] == 0:
        temp_tuple = (temp_binary, 'Subjective Review')
        binary_list_subjective.append(temp_binary)
        # binary_list_objective.append('NaN')
    binary_list.append(temp_tuple)
print(f'tru_count: {tru_count}\nfalse_count: {false_count}')


def tokenize_data(review_list):

    tknzr_punc = RegexpTokenizer(r'\w+')
    tokenized_list = []
    stop_words = set(stopwords.words('english'))
    for review in review_list:
        tokens = tknzr_punc.tokenize(review)
        filtered_tokens = [w for w in tokens if not w in stop_words]
        tokenized_list.append(filtered_tokens)
    return tokenized_list


def binary_checker(q1, q2):

    junk_list = []
    if len(q1) != len(q2):
        while len(q1) > len(q2):
            q2.append(junk_list)
        while len(q2) > len(q1):
            q1.append(junk_list)
    q1_binary = q1
    q2_binary = q2

    for question_number in range(len(q2)):

        if len(q1[question_number]) >= len(q2[question_number]):
            for index, word in enumerate(q2[question_number]):
                if word in q1[question_number]:
                    q2_binary[question_number][index] = 1
                else:
                    q2_binary[question_number][index] = 0
            # convert q1 to all 1s
            for i in range(len(q1[question_number])):
                q1_binary[question_number][i] = 1
                if i >= len(q2[question_number]):
                    q2[question_number].append(0)
        # else only happens if q1 < q2
        else:
            for index, word in enumerate(q1[question_number]):
                if word in q2[question_number]:
                    q1_binary[question_number][index] = 1
                else:
                    q1_binary[question_number][index] = 0
            # convert q1 to all 1s
            for i in range(len(q2[question_number])):
                q2_binary[question_number][i] = 1
                if i >= len(q1[question_number]):
                    q1[question_number].append(0)

    return q1_binary, q2_binary


def cosine_simularity(obj_list, sub_list):

    outer_obj = []
    # loop through entire obj list
    for outer_idx, obj_review_1 in enumerate(obj_list):

        # loop through entire obj list
        cos_sim_tracker_obj = []
        for inner_idx, obj_review_2 in enumerate(obj_list):
            # make sure obj1 != obj2
            if outer_idx == inner_idx:
                continue
            else:
                # make sure vectors are same legnth for cosine simularity
                if len(obj_review_1) != len(obj_review_2):
                    while len(obj_review_1) > len(obj_review_2):
                        obj_review_2.append(0)
                    while len(obj_review_2) > len(obj_review_1):
                        obj_review_1.append(0)

                for i in range(len(obj_review_1)):
                    obj_review_1[i] += 0.001
                    obj_review_2[i] += 0.001

                cos_sim = 1 - \
                    spatial.distance.cosine(obj_review_1, obj_review_2)
                cos_sim_tracker_obj.append(cos_sim)
                #print(f'\nouter_obj = {outer_idx}\ninner_obj = {inner_idx}\ncos_sim = {cos_sim}')

        # now calculate average cos_sim for obj_review_1 and each other objective review
        temp_average_obj = sum(cos_sim_tracker_obj) / len(cos_sim_tracker_obj)

        cos_sim_tracker_subj = []
        # loop through entire subj list
        for subj_review in sub_list:
            # make sure vectors are same legnth for cosine simularity
            if len(obj_review_1) != len(subj_review):
                while len(obj_review_1) > len(subj_review):
                    subj_review.append(0)
                while len(subj_review) > len(obj_review_1):
                    obj_review_1.append(0)

            for i in range(len(obj_review_1)):
                obj_review_1[i] += 0.001
                subj_review[i] += 0.001

            cos_sim = 1 - spatial.distance.cosine(obj_review_1, subj_review)
            cos_sim_tracker_subj.append(cos_sim)
            #print(f'cos_sim = {cos_sim}')

        # now calculate average cos_sim for obj_review_1 and each other subjective review
        temp_average_subj = sum(cos_sim_tracker_subj) / \
            len(cos_sim_tracker_subj)

        # store all information in temp tumple, then add temp_tuple to outer_obj list
        temp_tuple = (
            f'Objective Review: {outer_idx}', temp_average_obj, temp_average_subj)
        outer_obj.append(temp_tuple)

    outer_subj = []
    # loop through entire subjective list
    for outer_idx, subj_review_1 in enumerate(sub_list):

        # loop through entire subj list
        cos_sim_tracker_subj = []
        for inner_idx, subj_review_2 in enumerate(sub_list):
            # make sure subj1 != subj2
            if outer_idx == inner_idx:
                continue
            else:
                # make sure vectors are same legnth for cosine simularity
                if len(subj_review_1) != len(subj_review_2):
                    while len(subj_review_1) > len(subj_review_2):
                        subj_review_2.append(0)
                    while len(subj_review_2) > len(subj_review_1):
                        subj_review_1.append(0)

                for i in range(len(subj_review_1)):
                    subj_review_1[i] += 0.001
                    subj_review_2[i] += 0.001

                cos_sim = 1 - \
                    spatial.distance.cosine(subj_review_1, subj_review_2)
                cos_sim_tracker_subj.append(cos_sim)
                #print(f'cos_sim = {cos_sim}')

        # now calculate average cos_sim for subj_review_1 and each other subjective review
        temp_average_subj = sum(cos_sim_tracker_subj) / len(sub_list)

        cos_sim_tracker_obj = []
        # loop through entire subj list
        for obj_review in obj_list:
            # make sure vectors are same legnth for cosine simularity
            if len(subj_review_1) != len(obj_review):
                while len(subj_review_1) > len(obj_review):
                    obj_review.append(0)
                while len(obj_review) > len(subj_review_1):
                    subj_review_1.append(0)

            for i in range(len(subj_review_1)):
                subj_review_1[i] += 0.001
                obj_review[i] += 0.001
            cos_sim = 1 - spatial.distance.cosine(subj_review_1, obj_review)
            cos_sim_tracker_obj.append(cos_sim)
            #print(f'cos_sim = {cos_sim}')

        # now calculate average cos_sim for subj_review_1 and each other subjective review
        temp_average_obj = sum(cos_sim_tracker_obj) / len(cos_sim_tracker_obj)
        temp_tuple = (
            f'Subjective Review: {outer_idx}', temp_average_obj, temp_average_subj)
        outer_subj.append(temp_tuple)

    return outer_obj, outer_subj


def precision_recall(final_obj, final_subj):
    # true positive -> our model classifies something correctly (review that is truly objective)
    # false positive -> would be a subjective review our model classifies as objective
    # false negative -> would be an objective review our model classifies as subjective
    # Precision ->  number of true positives over the number of true positives plus the number of false positives
    # Recall -> number of true positives over the number of true positives plus the number of false negatives
    accuracy = 0
    count = 0
    true_positives = 0
    false_postives = 0
    false_negatives = 0

    for review in final_obj:
        count += 1
        if review[1] > review[2]:
            true_positives += 1
            accuracy += 1
        elif review[1] < review[2]:
            false_negatives += 1

    for review in final_subj:
        count += 1
        if review[2] > review[1]:
            true_positives += 1
            accuracy += 1
        elif review[2] < review[1]:
            false_postives += 1

    accuracy_final = accuracy / len(data)  # (length of data)
    precision = true_positives / (true_positives + false_postives)
    recall = true_positives / (true_positives + false_negatives)

    return accuracy_final, precision, recall


def print_list(list_input):
    for i in list_input:
        print(i)


obj_sentences_list = []
subj_sentences_list = []

for i in range(len(data)):
    if data[i]['objective'] == 1:
        obj_sentences_list.append(data[i]['text'])
    elif data[i]['objective'] == 0:
        subj_sentences_list.append(data[i]['text'])


obj_sentences_list_tokenized = tokenize_data(obj_sentences_list)
subj_sentences_list_tokenized = tokenize_data(subj_sentences_list)
obj_binary, subj_binary = binary_checker(
    obj_sentences_list_tokenized, subj_sentences_list_tokenized)
final_obj, final_subj = cosine_simularity(obj_binary, subj_binary)
accuracy, precision, recall = precision_recall(final_obj, final_subj)
print(
    f'-------------------------------\naccuracy: {accuracy}\n precision: {precision} \nrecall: {recall}')

print_list(final_obj)
print_list(final_subj)


# true positive -> our model classifies something correctly (review that is truly objective)
# false positive -> would be a subjective review our model classifies as objective
# false negative -> would be an objective review our model classifies as subjective
# Precision ->  number of true positives over the number of true positives plus the number of false positives
# Recall -> number of true positives over the number of true positives plus the number of false negatives
