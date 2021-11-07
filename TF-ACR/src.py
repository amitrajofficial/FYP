import sys
import json
import re, string
import copy
import math
import random
from datetime import datetime
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
import xxhash
import numpy as np
import itertools
import os

punctuation = list(string.punctuation)
cachedStopWords = stopwords.words('english')
total_clusters = 0
vocab = []
clusters = {}
cluster_text = {}
isClusterUpdated = {}
cluster_weights = {}

clusters_filled = {}
clusterWords = {}
# calculated emperically(0-1)
file_id = 111
threshhold = 0.12


current_cleanTweet = []
current_tweetCount = 0
current_tweetId = 0
cluster_freq = {}
cluster_weight_vector = {}

clusterNormalSums = {}

Capacity = 10000
bloomfilter = {}
filter_size = 64
no_hashfxn = 2

# testBit() returns a nonzero result, 2**offset, if the bit at 'offset' is one.
def testBit(int_type, offset):
    mask =  np.uint64(1) << np.uint64(offset)
    return(int_type & mask)


# setBit() returns an integer with the bit at 'offset' set to 1.
def setBit(int_type, offset):
    mask = np.uint64(1) << np.uint64(offset)
    return(int_type | mask)


def clearBit(int_type, offset):
  mask = ~(np.uint64(1) << offset)
  return(int_type & mask)



class BloomFilter:

    # BloomFilter(1000)
    # A bloom filter with 1000-element capacity
    def __init__(self, term,  cap=1000):
        self.capacity = cap

        self.number_of_insertions = 0

        self.weight = 0.0
        # Build the empty bloom filter
        self.filter = np.uint64(0)


    # Add an element to the bloom filter
    def add(self, element):
        # Make sure it's not full
        if self.number_of_insertions == self.capacity:
            raise Exception('Bloom Filter is full.')
            return

        # Run the hashes and add at the hashed index
        for seed in range(no_hashfxn):

            bitPosition = xxhash.xxh64(element, seed=seed).intdigest() % filter_size

            self.filter = setBit(self.filter,bitPosition)

        self.number_of_insertions += 1


    # Check the filter for an element
    # False means the element is definitely not in the filter
    # True means the element is PROBABLY in the filter
    def check(self, element):
        # Check each hash location. The seed for each hash is
        # just incremented from the previous seed starting at 0
        for seed in range(no_hashfxn):
            bit_pos_test = xxhash.xxh64(element, seed=seed).intdigest() % filter_size
            if not testBit(self.filter,bit_pos_test):
                return False
        # Probably in the filter if it was at each hashed location
        return True


    # For testing purposes
    def print_stats(self):
        print('Capacity ',self.capacity)
        #print('Expected Probability of False Positive '+self.false_positive_rate)
        print('Number of Elements ',self.number_of_insertions)


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

        #no word in cluster has its bloomfilter filled
        clusters_filled[cluster_id] = 0

        clusterWords[cluster_id] = {}


        bloomfilter[cluster_id] = {}

        clusterNormalSums[cluster_id] = 0.0

        total_clusters = total_clusters + 1

    clusters[cluster_id].append(tweet_id)
    # mark the cluster as updated to recalculate avg weight
    isClusterUpdated[cluster_id] = 1




    #we need to calculate our metric values before we have inserted our latest tweet for analysis
    #calculateBloomValues(cluster_id)



    updateClusterText(cluster_id)


    return cluster_id


def updateVocab():
    global current_cleanTweet
    for word in set(current_cleanTweet):
        if word not in cluster_freq:
            cluster_freq[word] = 0



def updateTermWeight(word,cluster_id):
    global bloomfilter,filter_size,no_hashfxn,clusterNormalSums

    total_insertions = bloomfilter[cluster_id][word].number_of_insertions
    #print(1 - (float(bin(bloomfilter[cluster_id][word].filter).count('1')) / filter_size))

    try:
        no_of_elements = -1 * (float(filter_size) / no_hashfxn) * math.log(1 - (float(bin(bloomfilter[cluster_id][word].filter).count('1')) / filter_size))  # return ln(x)

    except:

        print("no. of hash fxn : ",no_hashfxn)
        print("Bloomfilter filled for term : " + word + "  filled in cluster : " + str(cluster_id))
        print("Insertions done in the filter : " + str(total_insertions))
        print("Cluster length" + str(len(clusters[cluster_id])))
        writeClusterLengthToFile()
        sys.exit()



    if no_of_elements == 0.0:
        ratio = 0.0
    else:
        ratio = math.log10(float(total_insertions) / no_of_elements)

    temp = bloomfilter[cluster_id][word].weight

    term_weight = calc_tf(word, cluster_text[cluster_id]) * (1.0 + ratio)
    bloomfilter[cluster_id][word].weight = term_weight

    clusterNormalSums[cluster_id] = math.sqrt(math.pow(clusterNormalSums[cluster_id],2) - temp*temp + term_weight*term_weight)




# TO UPDATE TEXT OF TWEETS IN A GIVEN CLUSTER
def updateClusterText(cluster_id):
    global current_cleanTweet,cluster_text
    cluster_text[cluster_id].extend(current_cleanTweet)




def getSimilarity(cluster_id):
    global cluster_text,tweet_weight_vector

    tweet_Normal_Sum = 0.0

    for term in tweet_weight_vector:
        tweet_Normal_Sum += tweet_weight_vector[term]*tweet_weight_vector[term]

        if term in bloomfilter[cluster_id]:
            cluster_weight_vector[term] = bloomfilter[cluster_id][term].weight
        else:
            cluster_weight_vector[term] = 0.0


    tweet_Nvalue = math.sqrt(tweet_Normal_Sum)


    if tweet_Nvalue == 0.0 or clusterNormalSums[cluster_id] == 0.0:
        return 0.0
    else:
        similarity = 0.0
        for term in tweet_weight_vector:
            similarity += (tweet_weight_vector[term]/tweet_Nvalue)*(cluster_weight_vector[term]/clusterNormalSums[cluster_id])
    return similarity




def calc_tf(term, tokenized_document):
    term_frequency = tokenized_document.count(term)
    if (term_frequency > 0):
        log_tf = 1.0 + math.log10(float(term_frequency))
        return log_tf
    else:
        return 0.0


# TO GET NORMALIZED TF-IDF WEIGHT VECTOR OF A TWEET
def tf_idf_tweet():
    global tweets, cluster_freq,total_clusters,current_cleanTweet

    tweet_vector = {}
    for term in set(current_cleanTweet):
        tweet_vector[term] = calc_tf(term,current_cleanTweet)
    return tweet_vector



# FUNCTION TO PERFORM INCREMENTAL CLUSTERING
def incrementalClustering():
    global current_tweetId
    chooseCluster = 0  # cluster Id that is chosen
    tweetID = current_tweetId
    maxSimilarity = 0.0
    # get the cluster with max similarity
    for clusterID in clusters:
        sim = getSimilarity(clusterID)
        if (maxSimilarity < sim):
            chooseCluster = clusterID
            maxSimilarity = sim
    # add only if similarity greater than threshold value
    if (maxSimilarity >= threshhold):
        cluster = addToCluster(tweetID, chooseCluster)
    else:
        cluster = addToCluster(tweetID, total_clusters)
    return cluster




#This calculates values for analysis and writes in file.
def calculateBloomValues(cluster_id):
    global current_cleanTweet, bloomfilter, cluster_text, filter_size, no_hashfxn

    fp = open("BloomValues.txt", "a", encoding='utf-8')

    fp.write("Current tweet : " + (' ').join(current_cleanTweet) + "\n" + "Cluster id: " + str(cluster_id) + "\n")
    #fp.write("Cluster text : " + (' ').join(cluster_text[cluster_id]) + "\n")
    fp.write("Term".rjust(25) + "cluster_tf".rjust(25) + "Total insertions".rjust(25) + "Total elements".rjust(25) + "Total insertions/total elements".rjust(40) + "\n")
    for word in set(current_cleanTweet):
        flag = 0
        C_tf = cluster_text[cluster_id].count(word)

        if word in bloomfilter[cluster_id]:
            total_insertions = bloomfilter[cluster_id][word].number_of_insertions
            no_of_elements = -1 * (float(filter_size)/no_hashfxn) * math.log(1 - (float(bin(bloomfilter[cluster_id][word].filter).count('1')) / filter_size))      #return ln(x)
            print(1 - (float(bin(bloomfilter[cluster_id][word].filter).count('1')) / filter_size))
            if no_of_elements == 0.0:
                ratio = 0.0
            else:
                ratio = float(total_insertions)/no_of_elements


            fp.write(word.rjust(25) + str(C_tf).rjust(25) + str(total_insertions).rjust(25) + str("{:.5f}".format(no_of_elements)).rjust(25) + str("{:.5f}".format(ratio)).rjust(40) + "\n")
        else:
            #means that word not in cluster already, so no ass. filter, so all values are 0
            flag = 1
            fp.write(word.rjust(25) + str(C_tf).rjust(25) + "N.A.".rjust(25) + "N.A.".rjust(25) + "N.A.".rjust(40) + "\n")
    fp.write("\n\n")
    fp.close()






#update only on repitition of co-occurrence

def updateBloomfilters(cluster_id):
    global current_cleanTweet,bloomfilter



    #if any one bloomfilter in cluster is filled stop updating the insertions, but we need to create empty bloomfilters for new words
    if clusters_filled[cluster_id] == 0:
        for pair in itertools.combinations(set(current_cleanTweet), 2):
            word1 = pair[0]
            word2 = pair[1]

            if word1 in bloomfilter[cluster_id] and word2 in bloomfilter[cluster_id]:
                if bin(bloomfilter[cluster_id][word1].filter).count('1') >= 62 or bin(bloomfilter[cluster_id][word1].filter).count('1') >= 62:
                    clusters_filled[cluster_id] = 1
                    break
                bloomfilter[cluster_id][word1].add(word2)
                bloomfilter[cluster_id][word2].add(word1)


    for word in set(current_cleanTweet):
        if word not in bloomfilter[cluster_id]:
            bloomfilter[cluster_id][word] = BloomFilter(word, Capacity)

    for term in set(current_cleanTweet):
        updateTermWeight(term,cluster_id)



def writeClusterLengthToFile():
    global clusters
    with open("clusters"+str(file_id)+"/clustersInfo.txt","wt") as f:
        for id in clusters:
            try:
                f.write('{"cluster_id":"'+str(id).zfill(6) + '","cluster_length":' + str(len(clusters[id])) + '"}' + '\n')
            except:
                print("Can't write length for cluster",id)
        f.close()


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
        try:
            #updateVocab()
            # To get tfidf of current tweet
            tweet_weight_vector = tf_idf_tweet()

            my_cluster_id = incrementalClustering()


            updateBloomfilters(my_cluster_id)

            #write output in file
            fp = open(directory_name+"/cluster"+str(my_cluster_id).zfill(6)+".txt","a",encoding='utf-8')
            current_tweetText = current_tweetText.replace('"', '\\\"')
            tweetText = current_tweetText.replace('\n',' ')

            fp.writelines('{"timestamp":"'+ str(timestamp)+ '","full_text":"'+ tweetText+ '","id_str":"'+ current_tweetId + '","in_reply_to_status_id_str":"'+ str(reply_id)+ '","retweet_count":"'+ str(retweet_count)+ '","favorite_count":"'+ str(fav_count)+'","user_id_str":"'+ user_id+ '","in_reply_to_user_id_str":"' + str(in_reply_to) + '","annotation":"' + annotation + '","predicted":"' + str(my_cluster_id).zfill(6)+'"}' + '\n')
            fp.close()

        except MemoryError as error:
            print("Memory not sufficient")
        except:
            print("Tweet not valid! An exception occured.",current_tweetId)

        print(k)

    tweet_file.close()
    print(inputfile, " Done")


#writeClusterLengthToFile()
#print("Clusters successfully written to file")

for cluster_id in bloomfilter:
    for term in bloomfilter[cluster_id]:
        clusterWords[cluster_id][term] = bloomfilter[cluster_id][term].weight

fp = open("words_significant.txt","wt")
for cluster_id in clusterWords:
    sortedList = sorted(clusterWords[cluster_id].items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    fp.write(str(cluster_id) + "\n")
    for k in range(0,min(20,len(sortedList))):
        fp.write(str(sortedList[k][0]) + " : " + str(sortedList[k][1]) + "\n")
print("cluster most significant words found")

