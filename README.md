# Fake-News-Challenge
Machine Learning in fighting fake news  
http://www.fakenewschallenge.org/   
Project Proposal  


FEATURES:  
In order to reduce the dimension of feature space, we abandon the schema using the full dictionary as features set. Instead, we merge synonyms into disjoint sets(concept) and use them as features.  


ALGORITHM: 
DNN:  
Step 1: Embed all the words into vector representation and add them to, for each of the title text and the body text, hold a concept map(a vector of 50 dimension).Source:https://nlp.stanford.edu/projects/glove/   

Step 2: Feed the two vectors to a decision making network(classification) that has four output categories(unrelated,supportive , neutral, contradictory)  
  
LSTM:  
Step 1:Same as above  

Step 2:Encode two texts with lstm, feed the product of the encoded texts,which are two vectors of the same dimensions as the number of the hidden units of the lstm, to the soft-max classifier  

Step 3:Minimize the loss with optimizer  

FILES:  
embedding.py:Perform word embeddment  
decision_making_network.py: A standard feedforward neural network with 100 hidden nodes  
experiment.py:Training and testing(single thread, 5-fold cross validation)  
experimentThreaded.py:Multiple threading version of experiment(training 5 networks at the same time)  
experimentPool.py:Multiple processes version of experiment(training 5 networks at the same time)  
lstm_related.py:long short term memory network identifying whether the title is related to the body  
lstm_stance.py:long short term memory network identifying the stances  
Future research plan:  
Long short term memory network  
https://www.tensorflow.org/tutorials/seq2seq  
https://arxiv.org/abs/1406.1078

