import os
import re

#takes vocab file and returns hashmap with a word as the key and it's vector-index as the value
def hashmap_vocab(file_name):

    vocab_index_hashmap = {}            #initialize empty hashmap

    #open and read file
    with open(file_name, 'r', encoding = "utf-8") as vocab_file:
        vocab_text = vocab_file.read()      
    
    vocab_text = vocab_text.lower()     #lowercase text
    vocab_text = vocab_text.split()     #turns the text into a list

    #pair each word in the vocab with its index in the vector starting at 1 (the 0 index is for unseen words)
    for i, word in enumerate(vocab_text, 1):
        vocab_index_hashmap[word] = i

    return(vocab_index_hashmap)

#for a file, increment vector-index for each word in the text
def train_vectorize(file_path, guide_hashmap, vector):

    #open and read file
    with open(file_path, 'r', encoding = "utf-8") as file:
        text = file.read()

    text = re.sub(r"[\W \d]", " ", text)    #replace digits and symbols with whitespace
    text = text.lower()                     #lowercase
    text = text.split()                     #split into list

    #for every word, increment its respective index in the vector
    for word in text:
        position = guide_hashmap.get(word, 0)
        vector[position] += 1

    return vector

#takes folder of reviews, sends individual file paths to helper function to fill and returns full training histogram vector
def folder_vectorize(folder_path, guide_hashmap, alpha):

    vector = [alpha] * (len(guide_hashmap)+1)  #empty vector to store word occurences with add-alpha-smoothing

    #for every file, send to helper function to increment vector 
    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)            #concatenate folder path and file name to get full file path
        vector = train_vectorize(file_path, guide_hashmap, vector)  #increment vector

    return vector

#takes a vector histogram and returns the conditional probability vector
def cond_prob_from_vec(vector):

    total_words = sum(vector)    #find denominator

    #use bayes formula to calculate conditional probability for every word
    for i in range(1, len(vector)):
        vector[i] = vector[i] / total_words

    vector[0] = 1   #set the conditional prob of words outside the vocab to 1

    return vector
