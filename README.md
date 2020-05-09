# NLP-Final-Project

nlp_final.py contains all the code we used for the final project.  The inital variation of the python file simply performed naive bayes on 
200 reviews in order to train a model to detect objective vs subjective reviews.  As we progressed in the project we performed cosine 
simularity to determine if the average cosine simularity between two reviews was higher if the reviews were of the same category.  After 
calculaing precison, recall, and accuracy of our cosine simularity model, we went back and constructed k-fold cross validation for 4 different
values of k: 3, 5, 10, 15.  

The final variaton of our code is a bit messy, as it performs all the above features.  To perform k fold
cross validation with k = 5 and output the average accuracy lines 134 and 135 need to be uncommented, as they call the functions which
perform those algorithms.  

Lines 387 through 395 call the methods that perform all our cosine simularity computations.   

