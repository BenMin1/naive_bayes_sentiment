import preprocess
import NB

#file paths for vocab file
vocab_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_small\\vocab.txt"

#directory path for train folders
train_action_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_small\\train\\action"
train_comedy_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_small\\train\\comedy"

#directory path for test folder
test_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_small\\test"

#output file paths
parameter_output_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_small\\output\\movie-review-small.NB"
prediction_output_file_path = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\hw_2\\movie_review_small\\output\\predicted_genre.txt"

prior_prob_action = 0.6     #prior prob based on 3/5 training data being action
prior_prob_comedy = 0.4     #prior prob based on 2/5 training data being comedy
alpha = 1                   #set add-alpha-smoothing for add-one

#create hashmap of the vocabulary where key:value is word:index
vocab_index_hashmap = preprocess.hashmap_vocab(vocab_path)     

#create histograms for all classes
train_action_histogram = preprocess.folder_vectorize(train_action_file_path, vocab_index_hashmap, alpha)
train_comedy_histogram = preprocess.folder_vectorize(train_comedy_file_path, vocab_index_hashmap, alpha)

#create conditional probability vectors
train_action_cond_prob_vec = preprocess.cond_prob_from_vec(train_action_histogram)
train_comedy_cond_prob_vec = preprocess.cond_prob_from_vec(train_comedy_histogram)

#write the parameter output file with a header and the conditional prob vectors
with open(parameter_output_file_path, "w") as file:
    file.write("Conditional probability vectors (action, comedy): \n")
    file.write(str(train_action_cond_prob_vec))
    file.write("\n")
    file.write(str(train_comedy_cond_prob_vec))

#write the header of the prediction file
with open(prediction_output_file_path, "w") as file:
    file.write("Prediction comedy = 0, action = 1: \n") 

#predict on the test folder
prediction = NB.unpack_test(test_file_path, train_comedy_cond_prob_vec, train_action_cond_prob_vec, vocab_index_hashmap, prior_prob_comedy, prior_prob_action, prediction_output_file_path)
