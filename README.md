# Moderator-Detection-in-a-Debate-uisng-Hidden-Markov-Model-Speech-Processing-
Moderator of a debate is detected using audio clip of the debate. Hidden markov model is trained to detect various parts and debate and infer the debate moderator from it.

We used component=5 and mix=2 and got an output accuray of around 80% for the taken test files.
A little error is caused because the total time we consider will be a sum of both single and multi state. Since we don't consider the multi state a small part of the error will be because of that as it will be used for calculating accuracy.  

