import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load model and tokenizer globally (for efficiency)
try:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


def preprocess_tweet(tweet):
    """Preprocess the tweet by replacing usernames, URLs, etc."""
    tweet_words = []
    for word in tweet.split():
        if word.startswith('@') and len(word) > 1:
            word = '@user'  # Replace usernames
        elif word.startswith('http'):
            word = 'http'   # Replace URLs
        tweet_words.append(word)
    return " ".join(tweet_words)


def analyze_sentiment(tweet):
    """Analyze sentiment of the given tweet and visualize results."""
    try:
        # Preprocess tweet
        tweet_proc = preprocess_tweet(tweet)

        # Tokenize and encode the tweet
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt').to(device)

        # Get model output
        with torch.no_grad():
            output = model(**encoded_tweet)

        # Extract scores and apply softmax
        scores = output.logits[0].cpu().numpy()
        scores = softmax(scores)

        # Sentiment labels
        labels = ['Negative', 'Neutral', 'Positive']

        # Print sentiment scores
        print("Sentiment Analysis Results:")
        for label, score in zip(labels, scores):
            print(f"{label}: {score:.4f}")

        # Visualize results using a bar chart
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        bar_colors = sns.color_palette("Set2")
        plt.bar(labels, scores, color=bar_colors, edgecolor='black')

        # Add gridlines and annotations
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(0, 1)

        # Annotate scores on bars
        for i, score in enumerate(scores):
            plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")


if __name__ == "__main__":
    tweet = "You are very beautiful! 💖"
    analyze_sentiment(tweet)
