#FOR ANNOTATION

import os
import re
import shutil

keyWords = {}

#directory=os.listdir('clusters_src')

regex=re.compile('"annotation": "[0-9]*"')

#for file in directory:
#    print(file)
fp = open("completeShortClusters.txt", "r", encoding= 'utf-8')

lines = fp.readlines()

#for line in lines:
#    print(line)

print("\n\n\n")


print("existing keywords")
print(keyWords)

option = input("Do you want to enter more keywords")

while option == 'y':
    # enter like dreamers,daca,fthjd         without space
    key = input("Enter keywords      ")
    ann = input("enter annotation for this keyword          ")
    ann = ann.zfill(6)
    keyWords[ann] = key.split(',')

    option = input("Still more keywords : 'y' or 'n'")



for line in lines:
    print(line)
    flag = 0
    fp1 = open("clusters_ann.txt", "a", encoding='utf-8')
    #fp2 = open("unknownCluster.txt","a",encoding='utf-8')
    fp2 = open("automatedCluster.txt","a",encoding='utf-8')
    for annotation in keyWords:
        for word in keyWords[annotation]:
            if word in line.lower():
                flag = 1
                new_line = regex.sub("\"annotation\": " + "\"" + annotation + "\"",line)
                fp1.write(new_line)
                break

        if flag:
            break

    if flag:
        continue
    else:
        fp2.write(line)
    """
    opt = input("Do you want to enter annotation manually: y or n    ")

    if opt == 'y':
        ann = input("ANN:")
        new_line = regex.sub("\"annotation\":" + "\"" + ann.zfill(6) + "\"", line)
        fp1.write(new_line)
    else:
        fp2.write(line)
    """
    fp1.close()
    fp2.close()

fp.close()
