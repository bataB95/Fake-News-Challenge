import numpy as np
import csv
import sys

print("Start loading Word Vector Dictionary")
wordVec={}
with open('glove.txt', encoding="utf8") as glove:
    count=0
    for line in glove:
        temp=line.split()
        l=len(temp)
        wordVec[' '.join(temp[0:l-50])]=list(map(np.float,temp[l-50:l]))
        count=count+1
        print(count)
print("Finish loading Word Vector Dictionary")
print("Start loading training stances\n")
csv.field_size_limit(sys.maxsize)
with open('train_stances.csv', encoding="utf8") as csvfile_stance:
    stanceReader=csv.reader(csvfile_stance)
    stances={}
    for row in stanceReader:
        temp=[]
        for c,c_ in zip(row[0],row[0][1:]):
            if c.isalnum() or c.isspace():
                temp.append(c.lower())
            else:
                if not c_.isspace():
                    temp.append(" ")
        stances[row[1]]=[''.join(temp),row[2]]
print("Finish loading training stances\n")
print("Start loading training bodies\n")
with open('train_bodies.csv', encoding="utf8") as csvfile_body:
    bodyReader=csv.reader(csvfile_body)
    bodies={}
    for row in bodyReader:
        temp=[]
        for c,c_ in zip(row[1],row[1][1:]):
            if c.isalnum() or c.isspace():
                temp.append(c.lower())
            else:
                if not c_.isspace():
                    temp.append(" ")
        bodies[row[0]]=''.join(temp)
print("Finish loading training bodies")
print("Start merging training stances and training bodies\n")
raw_training_set={}
for key,value in stances.items():
    raw_training_set[value[0]]=[bodies[key],value[1]]
    #print(bodies[key])
print("Finish loading training stanes and training bodies\n")
stanceReader=None
bodyReader=None
stances=None
bodies=None
print("Start embedding\n")
with open('processed_data.csv', "w", encoding="utf8") as csvfile:
    vecWriter=csv.writer(csvfile)
    for key,value in raw_training_set.items():
        title=key.split()
        body=value[0].split()
        label=value[1]
        titleVec=np.zeros(50)
        bodyVec=np.zeros(50)
        for i in range(len(title)):
            #print(title[i] in wordVec)
            if title[i] in wordVec:
                word=wordVec[title[i]]
                #print(wordVec["the"])
                titleVec=np.add(titleVec,word)
        titleVec=np.divide(titleVec,np.sqrt(np.dot(titleVec,titleVec)))
        for i in range(len(body)):
            if body[i] in wordVec:
                word=wordVec[body[i]]
                bodyVec=np.add(bodyVec,word)
        bodyVec=np.divide(bodyVec,np.sqrt((np.dot(bodyVec,bodyVec))))
        vecWriter.writerow([titleVec,bodyVec,label])
print("Finish embedding\n")