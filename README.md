# Fake-News-Challenge
Machine Learning in fighting fake news  
http://www.fakenewschallenge.org/   
Project Proposal  


FEATURES:  
In order to reduce the dimension of feature space, we abandon the schema using the full dictionary as features set. Instead, we merge synonyms into disjoint sets(concept) and use them as features.  


ALGORITHM:  
Step 1: Embed all the words into vector representation and add them to, for each of the title text and the body text, hold a concept map(a vector of 50 dimension).   

Step 2: Feed the two vectors to a decision making network(classification) that has four output categories(unrelated,supportive , neutral, contradictory)  

Future research plan:  
Long short term memory network
