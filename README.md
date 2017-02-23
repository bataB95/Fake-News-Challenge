# Fake-News-Challenge
Machine Learning in fighting fake news  
http://www.fakenewschallenge.org/   
Project Proposal  

INTUITION:  
When we read texts with similar topic, the regions that represent the corresponding concepts light up. Presumably, the more related two texts are, the more similar the firing patterns are.  
When we read on, the level of activity of the previous concept decays unless something triggers it again.  


FEATURES:  
In order to reduce the dimension of feature space, we abandon the schema using the full dictionary as features set. Instead, we merge synonyms into disjoint sets(concept) and use them as features.  


ALGORITHM:  
Step 1: For each of the title text and the body text, hold a concept map(a vector with dimension of the total number of concepts in our semantic dictionary).Each column of the map records the activation of concepts seen in the paragraph, then decay the activations by their positions in the sentence with a proper decay function. By doing this, it encodes the order of the words into the concept map.   

Step 2: Feed the two vectors to a decision making network(classification) that has four output categories(unrelated,supportive , neutral, contradictory)  

POSSIBLE IMPROVEMENT:  
This is just simplified model of human reading.There are a lot of questions that have not been addressed here. But we will put them into the agenda.  
1)Similar concepts and even related concepts would stimulate the previous and keep it activated.  
