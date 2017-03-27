from decision_making_network import decision_making_network
import scorer
import random
import csv
import _thread
import numpy as np
np.set_printoptions(threshold=np.nan)
with open('processed_data.csv', encoding="utf8") as csvfile:
    csv_reader=csv.reader(csvfile)
    X=[]
    for row in csv_reader:
        X.append([list(map(float,row[0][1:len(row[0])-1].split())),list(map(int,row[1][1:len(row[1])-1].split(',')))])
# five-fold cross-validation

five_folds=[[],[],[],[],[]]
size=len(X)
for i in range(5):
    if i!=4:
        five_folds[i]=five_folds[i]+X[i*size//5:(i+1)*size//5-1]
    else:
        five_folds[i]=five_folds[i]+X[i*size//5:size-1]

NN=[]
"""
for i in range(5):
    NN.append(decision_making_network(len(X[0][0]),2*len(X[0][0]),4))
    train_X=[]
    test_X=[]
    for j in range(5):
        if i!=j:
            train_X=train_X+five_folds[j]
        else:
            test_X=test_X+five_folds[j]
    NN[i].train(train_X)
    accuracy.append(NN[i].test(test_X))
for i in range(len(accuracy)):
    print(i+1,"th fold:",accuracy[i])
"""

trainDataEagle = [[], [], [], [], []]
testDataEagle = [[], [], [], [], []]
temp=open("weights.txt","w")
temp.close()
temp=open("prediction.txt","w")
temp.close()
temp=open("accuracy.txt","w")
temp.close()
for i in range(5):
    NN.append(decision_making_network(len(X[0][0]),2*len(X[0][0]),4))
    testDataEagle[i]=testDataEagle[i]+five_folds[i]
    for j in range(5):
        if i!=j:
            trainDataEagle[i]=trainDataEagle[i]+five_folds[j]

def startTraining(networkNum, trainSet, testSet):
    NN[networkNum].train(trainSet,100)
    weight_file=open("weights.txt","a")
    weight_file.write('\nwi\n')
    weight_file.write(np.array2string(NN[networkNum].wi))
    weight_file.write('\nwo\n ')
    weight_file.write(np.array2string(NN[networkNum].wo))
    weight_file.write('\n')
    weight_file.close()
    accuracy=NN[networkNum].test(testSet)
    print(str(networkNum) + "th: " + str(accuracy))
    accuracy_file=open('accuracy.txt','a')
    accuracy_file.write(str(networkNum)+"th:"+str(accuracy)+"\n")
    accuracy_file.close()

for i in range(5):
    _thread.start_new_thread(startTraining, (i, trainDataEagle[i], testDataEagle[i]))
    print("threading:",i)
                
while 1:
    pass
