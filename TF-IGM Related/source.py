import sys
import json
import re, string
import copy
import math
import random
from datetime import datetime
import inflect
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
import os


punctuation = list(string.punctuation)
cachedStopWords = stopwords.words('english')
total_clusters = 0
vocab = []
clusters = {}
cluster_text = {}
isClusterUpdated = {}
cluster_weights = {}
cluster_igm_frequencies = {}
# calculated emperically(0-1)
file_id = 0
threshhold = 0.09


current_cleanTweet = []
current_tweetCount = 0
current_tweetId = 0
cluster_freq = {}
cluster_weight_vector = {}

clusterWords = {}

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



def bagOfWords(str):
    # Returns a bag of words from a given string Space delimited, removes punctuation, lowercase, Cleans text from url, stop words, tweet @, and 'rt'
    cleanWords = []
    lemmatizer = WordNetLemmatizer()
    # print("Original Tweet Text :  ", str)
    str = str.replace('â€¦', '…')
    str = str.replace('â€"', '–')
    str = str.replace('â€™', "'")
    str = str.replace('â€œ', '"')
    str = str.replace('&amp;', '&')

    str = str.lower().strip()

    words = re.sub("(@[A-Za-z]+)|([^A-Za-z0-9 \t])|(\w+:\/\/\S+)", " ", str).split()
    for word in words:
        if word in cachedStopWords or re.match('\s', word) or word == 'rt' or word == 'http':
            continue
        if word.isdigit():
            continue

        cleanWords.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
    # print(words)
    # print(cleanWords)

    return cleanWords


def addToCluster(tweet_id, cluster_id):
    global clusters, total_clusters

    # assign a tweet to new cluster  or to an existing one
    if cluster_id not in clusters:
        clusters[cluster_id] = []
        cluster_text[cluster_id] = []
        total_clusters = total_clusters + 1
        clusterWords[cluster_id] = {}
    clusters[cluster_id].append(tweet_id)
    # mark the cluster as updated to recalculate avg weight
    isClusterUpdated[cluster_id] = 1
    updateClusterText(cluster_id)

    return cluster_id


def updateVocab():
    global current_cleanTweet
    for word in set(current_cleanTweet):
        if word not in cluster_freq:
            cluster_freq[word] = {}


# TO UPDATE TEXT OF TWEETS IN A GIVEN CLUSTER
def updateClusterText(cluster_id):
    global current_cleanTweet,cluster_text
    for term in set(current_cleanTweet):
        if term not in cluster_text[cluster_id]:
            cluster_freq[term][cluster_id] = 0

        cluster_freq[term][cluster_id] += current_cleanTweet.count(term)

    cluster_text[cluster_id].extend(current_cleanTweet)


def getSimilarity(tweet_id, cluster_id):
    global tweets, isClusterUpdated,cluster_text,tweet_weight_vector

    tweet_Normal_Sum = 0.0
    cluster_Normal_Sum = 0.0
    cluster_weight_vector = {}
    for term in tweet_weight_vector:
        tweet_Normal_Sum += tweet_weight_vector[term]*tweet_weight_vector[term]
        cluster_weight_vector[term] = 0.0


    for term in set(cluster_text[cluster_id]):
        cluster_weight_vector[term] = calc_tf(term, cluster_text[cluster_id])
        #print(cluster_weight_vector[term])
        cluster_Normal_Sum += cluster_weight_vector[term] * cluster_weight_vector[term]



    tweet_Nvalue = math.sqrt(tweet_Normal_Sum)
    cluster_Nvalue = math.sqrt(cluster_Normal_Sum)


    if tweet_Nvalue == 0.0 or cluster_Nvalue == 0.0:
        return 0.0
    else:
        similarity = 0.0
        for term in tweet_weight_vector:
            similarity += (tweet_weight_vector[term]/tweet_Nvalue)*(cluster_weight_vector[term]/cluster_Nvalue)
    return similarity





def calc_tf(term, tokenized_document):
    term_frequency = tokenized_document.count(term)
    if (term_frequency > 0):
        log_tf = 1.0 + math.log10(float(term_frequency))
        return log_tf
    else:
        return 0.0


def calc_igm(term):
    lamda = 7.0         #by default
    sorted_freq = sorted(cluster_freq[term].items(), key=lambda kv: (kv[1], kv[0]),reverse=True)


    sum = 0.0
    for i in range(0,len(sorted_freq)):
        sum += sorted_freq[i][1] * (i+1)



    if sum != 0.0:
        igm = 1 + lamda*(sorted_freq[0][1]/sum)
    else:
        igm = 0.0

    #print("igm = ",igm)
    return igm



# TO GET NORMALIZED TF-IDF WEIGHT VECTOR OF A TWEET
def tf_idf_tweet():
    global tweets, cluster_freq,total_clusters,current_cleanTweet

    tweet_vector = {}
    for term in set(current_cleanTweet):
        tf = calc_tf(term,current_cleanTweet)
        igm = calc_igm(term)
        #idf = math.log10(float(current_tweetCount)/vocab[term])

        tf_igm = tf * igm
        tweet_vector[term] = tf_igm
        #print(term,tf_igm)
    return tweet_vector



# FUNCTION TO PERFORM INCREMENTAL CLUSTERING
def incrementalClustering():
    global current_tweetId,clusters
    chooseCluster = 0  # cluster Id that is chosen
    tweetID = current_tweetId
    maxSimilarity = 0.0
    # get the cluster with max similarity
    for clusterID in clusters:
        sim = getSimilarity(tweetID, clusterID)
        if (maxSimilarity < sim):
            chooseCluster = clusterID
            maxSimilarity = sim
    # add only if similarity greater than threshold value
    #print("max similarity " ,maxSimilarity)
    if (maxSimilarity >= threshhold):
        cluster = addToCluster(tweetID, chooseCluster)
    else:
        cluster = addToCluster(tweetID, total_clusters)
    return cluster




directory_name = 'clustersTest' + str(file_id) + '_' + str(threshhold)
os.mkdir(directory_name)


inputfile = 'sortedTweets.json'
with open(inputfile, 'r') as tweet_file:
    json_data = json.load(tweet_file)
    for k in range(int(len(json_data))):
        # tweet is read
        try:

            current_tweetText = json_data[k]['full_text']
            current_tweetCount += 1
            current_tweetId = json_data[k]['id_str']

            # To convert back
            # dt_object = datetime.fromtimestamp(timestamp)

            timestamp = json_data[k]['timestamp']
            user_id = json_data[k]['user_id']
            reply_id = json_data[k]['in_reply_to_status_id']
            fav_count = json_data[k]['favorite_count']
            retweet_count = json_data[k]['retweet_count']
            in_reply_to = json_data[k]['in_reply_to_user_id_str']
            annotation = json_data[k]['annotation']
            current_cleanTweet = bagOfWords(current_tweetText)
        except MemoryError as error:
            print("Memory not sufficient")
        except:
            continue

        #try:
        updateVocab()
        # To get tfidf of current tweet
        tweet_weight_vector = tf_idf_tweet()

        my_cluster_id = incrementalClustering()

        #write output in file
        fp = open(directory_name+"/cluster"+str(my_cluster_id).zfill(6)+".txt","a",encoding='utf-8')
        current_tweetText = current_tweetText.replace('"', '\\\"')
        tweetText = current_tweetText.replace('\n',' ')

        fp.writelines('{"id_str":"' + current_tweetId + '","full_text":"' + tweetText + '","timestamp":"' + str(
            timestamp) + '","in_reply_to_status_id_str":"' + str(reply_id) + '","retweet_count":"' + str(
            retweet_count) + '","favorite_count":"' + str(
            fav_count) + '","user_id_str":"' + user_id + '","in_reply_to_user_id_str":"' + str(
            in_reply_to) + '","annotation":"' + annotation + '","predicted":"' + str(my_cluster_id).zfill(6) + '"}' + '\n')
        fp.close()
        """except MemoryError as error:
            print("Memory not sufficient")
        except:
            print("Tweet not valid! An exception occured.",current_tweetId)
        """
        print(k)
    tweet_file.close()
    print(inputfile, " Done")
del json_data

# printClusterText()
#for this maintain the tweets dictionary of form tweet[id] = text
#writeClusterToFile()


#writeClusterLengthToFile()
print("Clusters successfully written to file")
for cluster_id in clusters:
    for term in cluster_text[cluster_id]:
        clusterWords[cluster_id][term] = calc_igm(term) * calc_tf(term,cluster_text[cluster_id])

fp = open("words_significant.txt","wt")
for cluster_id in clusterWords:
    sortedList = sorted(clusterWords[cluster_id].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    fp.write(str(cluster_id) + "\n")
    for k in range(0, min(20, len(sortedList))):
        fp.write(str(sortedList[k][0]) + " : " + str(sortedList[k][1]) + "\n")
print("cluster most significant words found")
