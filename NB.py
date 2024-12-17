import math
import os
import re

#returns the argmax conditional probability for the file's text (0 for pos, 1 for neg)
def predict(file_path, train_pos_vec, train_neg_vec, vocab_hashmap, prior_pos, prior_neg, prediction_file):

    pos_prob = math.log(prior_pos)  #initialize to prior prob
    neg_prob = math.log(prior_neg)  #initialize to prior prob

    #open and read file
    with open(file_path, 'r', encoding = "utf-8") as file:
        text = file.read()

    text = re.sub(r"[\W \d]", " ", text)    #remove digits and symbols
    text = text.lower()                     #lowercase
    text = text.split()                     #split into vector

    #sum the log prob of each word to get the log prob of the entire review
    for word in text:
        word_index = vocab_hashmap.get(word, 0)         #retreive the index of the current word

        pos_prob += math.log(train_pos_vec[word_index]) #increment the prob given pos
        neg_prob += math.log(train_neg_vec[word_index]) #increment the prob given neg

    #return 0 for pos, 1 for neg
    if(pos_prob > neg_prob):
        
        #write the prediction to a new line
        with open(prediction_file, "a") as file:
            file.write("0 \n")

        return 0
    
    else:
        
        #write the prediction to a new line
        with open(prediction_file, "a") as file:
            file.write("1 \n")

        return 1

#takes a folder and sends every file path to helper function, returns the predicted amount of negative reviews within
def unpack_test(folder_file, train_pos_vec, train_neg_vec, vocab_index_hashmap, prior_pos, prior_neg, prediction_file):

    predicted_neg = 0   #initialize total negative predictions

    #for every file in folder, send file path to helper function
    for file_name in os.listdir(folder_file):

        file_path = os.path.join(folder_file, file_name) #concatenate the folder path and file name to get complete flie path

        predicted_neg += predict(file_path, train_pos_vec, train_neg_vec, vocab_index_hashmap, prior_pos, prior_neg, prediction_file)  #increment negative predictions as necessary

    return predicted_neg
