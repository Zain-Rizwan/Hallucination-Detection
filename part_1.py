import re

def preprocess(text):
    text = text.lower()  # Converting to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    return text

def tokenize(text):
    return text.split() #splitting sentences to individual words

# Build Unigram, Bigram, and Trigram models manually using dictionaries
def build_ngram_models(tokenized_reviews):
    uni_count = {} # Initailizing empty array
    bii_count = {}
    ti_count = {}
    
    for review in tokenized_reviews:
        # Iterating through the review words for unigrams, bigrams, and trigrams
        for i in range(len(review)):
            # Unigram k liye
            unigram = review[i]
            if unigram in uni_count:
                uni_count[unigram] += 1
            else:
                uni_count[unigram] = 1

            # Bigram k liye
            if i > 0:
                bigram = (review[i-1], review[i])
                if bigram in bii_count:
                    bii_count[bigram] += 1
                else:
                    bii_count[bigram] = 1

            # Trigram k liye
            if i > 1:
                trigram = (review[i-2], review[i-1], review[i])
                if trigram in ti_count:
                    ti_count[trigram] += 1
                else:
                    ti_count[trigram] = 1

    return uni_count, bii_count, ti_count

# Function to predict the next word based on unigram, bigram, and trigram models
def predict_next_words(uni_count, bii_count, ti_count, input_sentence):
    words = tokenize(preprocess(input_sentence))

    # Initialize predictions
    unigram_prediction = None
    bigram_prediction = None
    trigram_prediction = None

    # Unigram Prediction (most frequent unigram)
    if uni_count:
        most_common_unigram = max(uni_count, key=uni_count.get)
        unigram_prediction = f"Predicted next word (unigram): {most_common_unigram}"

    # Bigram Prediction
    if len(words) >= 1:
        last_word = words[-1]
        possible_bigrams = {key[1]: count for key, count in bii_count.items() if key[0] == last_word}
        
        if possible_bigrams:
            bigram_prediction = max(possible_bigrams, key=possible_bigrams.get)
            bigram_prediction = f"Predicted next word (bigram): {bigram_prediction}"
    
    # Trigram Prediction
    if len(words) >= 2:
        last_two_words = (words[-2], words[-1])
        possible_trigrams = {key[2]: count for key, count in ti_count.items() if key[0] == last_two_words[0] and key[1] == last_two_words[1]}
        
        if possible_trigrams:
            trigram_prediction = max(possible_trigrams, key=possible_trigrams.get)
            trigram_prediction = f"Predicted next word (trigram): {trigram_prediction}"

    return unigram_prediction, bigram_prediction, trigram_prediction

def main(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # Specify encoding and ignore errors
        reviews = file.readlines()
    
    reviews = [preprocess(review.strip()) for review in reviews]
    tokenized_reviews = [tokenize(review) for review in reviews]
    
    # for maaking a probablistic array for n gram
    unigram_model, bigram_model, trigram_model = build_ngram_models(tokenized_reviews)
    
    while True:
        print("\nEnter a sentence or word (or type 'exit' to quit):")
        sentence = input().strip()
        
        if sentence == 'exit':
            break
        
        # predict karne k liye next word
        unigram_prediction, bigram_prediction, trigram_prediction = predict_next_words(unigram_model, bigram_model, trigram_model, sentence)
        
        # Display all predictions
        if unigram_prediction:
            print(unigram_prediction)
        if bigram_prediction:
            print(bigram_prediction)
        if trigram_prediction:
            print(trigram_prediction)

        if not unigram_prediction and not bigram_prediction and not trigram_prediction:
            print("No prediction can be made.")


path = 'IMDB_Dataset.csv'
main(path)
