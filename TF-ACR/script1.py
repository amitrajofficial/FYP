#TO FILTER CLUSTERS ON BASIS OF LENGTH AND AUTOMATICALLY ANNOTATE THEM

import os
import re
import shutil
directory=os.listdir('clusters_dest')
#os.chdir('clusters_src')
# print(directory)
for file in directory:
	#print("\n\n\n")
	#print("file=",file)

	fp = open("clusters_dest/"+file, "r", encoding= 'utf-8')
	total_text = fp.read()
	regex=re.compile('"annotation": " "')
	total_text = regex.sub("\"annotation\": " + "\"" + file[8:-4]+ "\"",total_text)
	fp1 = open("clusters_ann_dest/"+file, "wt", encoding = 'utf-8')
	fp1.write(total_text)
	fp.close()
	fp1.close()





