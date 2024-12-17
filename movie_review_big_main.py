import preprocess
import NB

#file paths for vocab file
vocab_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\imdb.vocab"

#directory path for train folders
train_neg_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\train\\neg"
train_pos_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\train\\pos"

#directory path for test folders
test_neg_filepath = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\test\\neg"
test_pos_filepath = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\test\\pos"

#output file paths
parameter_output_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\output\\movie-review-params.NB"
prediction_output_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_big\\output\\predicted_sentiment.txt"

alpha = 1                   #set add-alpha-smoothing for add-one
prior_prob_pos = 0.5        #prior probability
prior_prob_neg = 0.5        #prior probability

num_neg_test_files = 12500  #for calculating false positives
num_test_files = 25000      #for calculating error percentage

#create hashmap of the vocabulary where key:value is word:index
vocab_index_hashmap = preprocess.hashmap_vocab(vocab_path)     

#create histograms for all classes
train_neg_histogram = preprocess.folder_vectorize(train_neg_file_path, vocab_index_hashmap, alpha)
train_pos_histogram = preprocess.folder_vectorize(train_pos_file_path, vocab_index_hashmap, alpha)

#create conditional probability vectors
train_neg_cond_prob_vec = preprocess.cond_prob_from_vec(train_neg_histogram)
train_pos_cond_prob_vec = preprocess.cond_prob_from_vec(train_pos_histogram)

#write the parameter output file with a header and the conditional probability vectors
with open(parameter_output_file_path, "w") as file:
    file.write("Conditional probability vectors (neg, pos): \n")
    file.write(str(train_neg_cond_prob_vec))
    file.write("\n")
    file.write(str(train_pos_cond_prob_vec))

#write header for prediction output file
with open(prediction_output_file_path, "w") as file:
    file.write("Predictions: neg = 1, pos = 0 (first 12500 rows are true neg next 12500 are true pos): \n") 

#predict on the test folders
neg_prediction_sum = NB.unpack_test(test_neg_filepath, train_pos_cond_prob_vec, train_neg_cond_prob_vec, vocab_index_hashmap, prior_prob_pos, prior_prob_neg, prediction_output_file_path)
pos_prediction_sum = NB.unpack_test(test_pos_filepath, train_pos_cond_prob_vec, train_neg_cond_prob_vec, vocab_index_hashmap, prior_prob_pos, prior_prob_neg, prediction_output_file_path)

#calculate errors
false_positives = (num_neg_test_files - neg_prediction_sum) #difference from truth (12500) = false positives
false_negatives = pos_prediction_sum                        #difference from truth (0) = false negatives
errors = false_negatives + false_positives                  #sum of all errors
accuracy = (num_test_files - errors) / num_test_files * 100 #correct prediction percentage

#write the accuracy of the model as the last line in the prediction output file
accuracy_message = "Model had an accuracy of " + str(accuracy) + " percent"
with open(prediction_output_file_path, "a") as file:
    file.write(accuracy_message) 
