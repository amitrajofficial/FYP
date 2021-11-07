#SCRIPT TO COMBINE ALL CLUSTER TEXT FILES, THEN SORT IT AND THEN CONVERT IT TO JSON FORMAT.


import os
import re
import shutil
import json
import datetime
#os.chdir('clusters_src')
# print(directory)


"""
directory=os.listdir('clusters_dest1')

fp = open("combined/clusters.txt","wt",encoding='utf-8')
for file in directory:
#Contains the output json file
    #inputfile = open('JsonClusters/clusterj' + file[8:-4] + '.json', 'wt')
    data = []
    with open('clusters_dest1/' + file ,"r",errors='ignore') as f:

        text = f.read()
        fp.write(text)

fp.close()
"""
fp = open("partial_gt.txt","r",encoding='utf-8')
output = open("sorted_partial_gt.txt","wt",encoding='utf-8')
lines = fp.readlines()
sorted_lines = sorted(lines)
for line in sorted_lines:
    output.write(line)
fp.close()
output.close()

print("Done sorting tweets")
"""
outputfile = open('GROUNDTRUTH/final_gt_txt_parts/.json', 'wt', encoding='utf-8')
data = []

with open('combined/sorted_manual_ann.txt' ,"r",errors='ignore') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as inst:
                continue
outputfile.write(json.dumps(data))
outputfile.close()
print("Done converting tweets")

"""









