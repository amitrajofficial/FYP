# COMBINED EVALUATION SCRIPT

# TO CONVERT A SERIES OF TEXT FILES TO CORRESPONDING JSON FILES

import sys
import os
import json
import shutil

input_directory = 'clusters11_0.11'

try:
    shutil.rmtree('clustersBLOOM')

    shutil.rmtree('clustersBLOOM_GT')
    shutil.rmtree('clustersBGT')
    os.mkdir('clustersBLOOM')
    os.mkdir('clustersBGT')
    os.mkdir('clustersBLOOM_GT')
except:
    print("directory problem")
    sys.exit()
directory = os.listdir(input_directory)

for file in directory:
    # Contains the output json file

    data = []
    outputfile = open('clustersBLOOM/' + file[:-4] + '.json', "wt", encoding='utf-8')
    with open(input_directory + '\\' + file, "r", errors='ignore') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as inst:
                continue
    outputfile.write(json.dumps(data))
    outputfile.close()
    f.close()

directory = os.listdir('clustersBLOOM')

for file in directory:
    annotation = []
    fp = open('clustersBLOOM/' + file, 'r')
    json_data = json.load(fp)
    length = len(json_data)
    for k in range(0, length):
        current_tweetId = json_data[k]['id_str']
        annotation = json_data[k]['annotation']
        predicted = json_data[k]['predicted']
        fp1 = open("clustersBLOOM_GT/cluster" + annotation + ".txt", "a", encoding='utf-8')
        fp1.writelines(
            '{"id_str":"' + current_tweetId + '","annotation":"' + annotation + '","predicted":"' + predicted + '"}' + '\n')
        fp1.close()
    fp.close()

directory = os.listdir('clustersBLOOM_GT')

for file in directory:
    # Contains the output json file

    data = []
    outputfile = open('clustersBGT/' + file[:-4] + '.json', "wt", encoding='utf-8')
    with open('clustersBLOOM_GT/' + file, "r", errors='ignore') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as inst:
                continue
    outputfile.write(json.dumps(data))
    outputfile.close()
    f.close()

import itertools

directory = os.listdir('clustersBLOOM')
SS = 0
SD = 0
# CALCULATING PRECISION
for file in directory:
    annotation = []
    # IF IN SAME FILE, ALREADY THE SAME PREDICTED
    fp = open('clustersBLOOM/' + file, 'r')
    json_data = json.load(fp)
    length = len(json_data)
    for k in range(0, length):
        # GROUND TRUTH ANNOTATION
        annotation.append(json_data[k]['annotation'])

    for pair in itertools.combinations(annotation, 2):
        ann1 = pair[0]
        ann2 = pair[1]
        if ann1 == ann2:
            SS += 1
        else:
            SD += 1
            # print(pair,SS,SD)
precision = float(SS) / (SS + SD)
print("\nPRECISION = ", precision)

# CALCULATING RECALL

DS = 0
directory = os.listdir('clustersBGT')

for file in directory:
    prediction = []
    # IF IN SAME FILE, ALREADY THE SAME PREDICTED
    fp = open('clustersBGT/' + file, 'r')
    json_data = json.load(fp)
    length = len(json_data)
    for k in range(0, length):
        # GROUND TRUTH ANNOTATION
        prediction.append(json_data[k]['predicted'])

    for pair in itertools.combinations(prediction, 2):
        ann1 = pair[0]
        ann2 = pair[1]
        if ann1 != ann2:
            DS += 1

recall = float(SS) / (SS + DS)
print("\nRECALL = ", recall)

fmeasure = (precision * recall * 2) / (precision + recall)
print("\nFMeasure = ", fmeasure)