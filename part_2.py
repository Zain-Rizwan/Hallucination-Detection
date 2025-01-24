import re

def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def tokenize(text):
    return text.split()  # Splitting sentences into individual words

def build_ngram_models(tokenized_reviews, sentiments):
    uni_count = {}  # Initializing unigram counts for positive reviews
    bii_count = {}   # Initializing bigram counts for positive reviews
    ti_count = {}  # Initializing trigram counts for positive reviews

    unigram_counts_neg = {}  # Initializing unigram counts for negative reviews
    bigram_counts_neg = {}   # Initializing bigram counts for negative reviews
    trigram_counts_neg = {}  # Initializing trigram counts for negative reviews


    for review, sentiment in zip(tokenized_reviews, sentiments):
        for i in range(len(review)):
            #  For Unigram
            unigram = review[i]
            if sentiment == "positive":
                if unigram in uni_count:
                    uni_count[unigram] += 1
                else:
                    uni_count[unigram] = 1
            elif sentiment=="negative":
                if unigram in unigram_counts_neg:
                    unigram_counts_neg[unigram] += 1
                else:
                    unigram_counts_neg[unigram] = 1

            # Bigram k liye
            if i > 0:
                bigram = (review[i-1], review[i])
                if sentiment == "positive":
                    if bigram in bii_count:
                        bii_count[bigram] += 1
                    else:
                        bii_count[bigram] = 1
                elif sentiment=="negative": # for negative
                    if bigram in bigram_counts_neg:
                        bigram_counts_neg[bigram] += 1
                    else:
                        bigram_counts_neg[bigram] = 1

            # Trigram k liye
            if i > 1:
                trigram = (review[i-2], review[i-1], review[i])
                if sentiment == "positive":
                    if trigram in ti_count:
                        ti_count[trigram] += 1
                    else:
                        ti_count[trigram] = 1
                elif sentiment=="negative": # for negative
                    if trigram in trigram_counts_neg:
                        trigram_counts_neg[trigram] += 1
                    else:
                        trigram_counts_neg[trigram] = 1

    return (uni_count, bii_count, ti_count), (unigram_counts_neg, bigram_counts_neg, trigram_counts_neg)

# Calculate the sentiment probability for each model
def calculate_sentiment_probability(counts_pos, counts_neg, ngrams):
    pos_prob, neg_prob = 1, 1
    vocab_size_pos = len(counts_pos)
    vocab_size_neg = len(counts_neg)

    for ngram in ngrams:
        # Positive probability with Laplace smoothing
        pos_prob *= (counts_pos.get(ngram, 0) + 1) / (sum(counts_pos.values()) + vocab_size_pos)
        # Negative probability with Laplace smoothing
        neg_prob *= (counts_neg.get(ngram, 0) + 1) / (sum(counts_neg.values()) + vocab_size_neg)

    return pos_prob, neg_prob

# Predict sentiment for unigrams, bigrams, and trigrams
def predict_sentiment(uni_pos, bi_pos, tri_pos, uni_neg, bi_neg, tri_neg, input_sentence):
    words = tokenize(preprocess(input_sentence))

    # Unigram Prediction
    unigram_probs = calculate_sentiment_probability(uni_pos, uni_neg, words)

    if unigram_probs[0] > unigram_probs[1]:
        unigram_prediction = "positive"
    else:
        unigram_prediction = "negative"


    # Bigram Prediction
    bigrams = [(words[i-1], words[i]) for i in range(1, len(words))]
    bigram_probs = calculate_sentiment_probability(bi_pos, bi_neg, bigrams)

    if bigram_probs[0] > bigram_probs[1]:
        bigram_prediction = "positive"
    else:
        bigram_prediction = "negative"


    # Trigram Prediction
    trigrams = [(words[i-2], words[i-1], words[i]) for i in range(2, len(words))]
    trigram_probs = calculate_sentiment_probability(tri_pos, tri_neg, trigrams)

    if trigram_probs[0] > trigram_probs[1]:
        trigram_prediction = "positive"
    else:
        trigram_prediction = "negative"


    return unigram_prediction, bigram_prediction, trigram_prediction

def main(file_path):

    reviews=[]
    sentiments = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # Specify encoding and ignore errors
        for line in file.readlines()[1:]:  # Skip header
            data = line.strip().split(',')  # Split line into review and sentiment
            if len(data) == 2:
                reviews.append(preprocess(data[0].strip()))
                sentiments.append(data[1].strip().lower())
    tokenized_reviews = [tokenize(review) for review in reviews]

    #Build the n-gram models for positive and negative reviews
    (uni_pos, bi_pos, tri_pos), (uni_neg, bi_neg, tri_neg) = build_ngram_models(tokenized_reviews, sentiments)

    #will work until break is not used
    while True:
        print("\nEnter a sentence or word (or type 'exit' to quit):")
        sentence = input().strip()

        if sentence == 'exit':
            break

        # Step 5: Predict the sentiment based on the n-gram models
        unigram_prediction, bigram_prediction, trigram_prediction = predict_sentiment(uni_pos, bi_pos, tri_pos, uni_neg, bi_neg, tri_neg, sentence)

        # Displaying predictions for unigram, bigram, and trigram
        print(f"Unigram prediction: {unigram_prediction}")
        print(f"Bigram prediction: {bigram_prediction}")
        print(f"Trigram prediction: {trigram_prediction}")


path = 'IMDB_Dataset.csv'
main(path)
