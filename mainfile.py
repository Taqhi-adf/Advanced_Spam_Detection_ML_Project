# Spam Detection model using Logistic Regression,svc,Naive_bayes models step-by-step with code, explanations, and evaluation.
#We’ll use the SMS Spam Collection Dataset style format (label, text).
#Here’s a clean, end-to-end “Spam vs Ham” text-classifier you can drop into a single Python file or notebook. It covers data prep, modeling, evaluation, and saving the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load dataset (assuming a CSV with 'label' and 'text' columns)
# For demonstration, we’ll create a small sample dataset inline.

# --- Toy data (replace with your dataset) ---
data = {
    "label": ["spam","ham","ham","spam","ham","spam","spam","ham","spam","ham","ham"],
    "text": [
        "Hurray! You won a free lottery ticket",
        "I'll call you later",
        "Are we still meeting at 6?",
        "WINNER!! Claim your prize now",
        "See you tomorrow",
        "You won a free ticket, click here",
        "Congratulations, you have been selected",
        "Lunch at 1pm?",
        "Urgent! Your account is locked, verify now",
        "Happy birthday!",
        "Please review the report"
    ]
}
df = pd.DataFrame(data)
print(df)
print(df.columns)

# Convert labels to lowercase and strip spaces
#df["text"] = df["text"].str.lower().str.strip()

# Train /validation split

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42,stratify=df['label'])
print(X_train)
print(y_train)

# --- Pipeline: TF-IDF + Logistic Regression ---
pipe1 = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),         # unigrams + bigrams often help spam cues
        min_df=1                   # raise to 2/3 for bigger datasets
    )),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
])

pipe1.fit(X_train, y_train)

# --- Evaluation ---
y_pred1 = pipe1.predict(X_test)
print(classification_report(y_test, y_pred1, digits=3))
print("Confusion_matrix:\n", confusion_matrix(y_test, y_pred1))
accuracy_score1 = accuracy_score(y_test, y_pred1)
print("Logistic Regression Accuracy:", accuracy_score1)


sns.heatmap(confusion_matrix(y_test, y_pred1), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# --- Use the model ---
samples = [
    "free entry in a weekly sweepstake! click to claim",
    "can you share the slides from the meeting?"
]
print(pipe1.predict(samples))
print(pipe1.predict_proba(samples))  # class probabilities (if available)

pipe2 = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),         # unigrams + bigrams often help spam cues
        min_df=1                   # raise to 2/3 for bigger datasets
    )),
    ("NB", MultinomialNB())
])

pipe2.fit(X_train, y_train)

# --- Evaluation ---
y_pred2 = pipe2.predict(X_test)
print(classification_report(y_test, y_pred2, digits=3))
print("Confusion_matrix:\n", confusion_matrix(y_test, y_pred2))
accuracy_score2 = accuracy_score(y_test, y_pred2)
print("MultinomialNB Accuracy:", accuracy_score2)

sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, fmt="d", cmap="Greens")
plt.title("MultinomialNB Confusion Matrix")
plt.show()

# --- Use the model ---
samples = [
    "free entry in a weekly sweepstake! click to claim",
    "can you share the slides from the meeting?"
]
print(pipe2.predict(samples))
print(pipe2.predict_proba(samples))  # class probabilities (if available)



pipe3  = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),         # unigrams + bigrams often help spam cues
        min_df=1                   # raise to 2/3 for bigger datasets
    )),
    ("SVC", SVC(probability=True))
])

pipe3.fit(X_train, y_train)

# --- Evaluation ---
y_pred3 = pipe3.predict(X_test)
print(classification_report(y_test, y_pred3, digits=3))
print("Confusion_matrix:\n", confusion_matrix(y_test, y_pred3))
accuracy_score3 = accuracy_score(y_test, y_pred3)
print("SVC Accuracy:", accuracy_score3)


sns.heatmap(confusion_matrix(y_test, y_pred3), annot=True, fmt="d", cmap="Oranges")
plt.title("SVC Confusion Matrix")
plt.show()

# --- Use the model ---
samples = [
    "free entry in a weekly sweepstake! click to claim",
    "can you share the slides from the meeting?"
]
print(pipe3.predict(samples))
print(pipe3.predict_proba(samples))  # class probabilities (if available)


pipe1_results = pd.DataFrame(["Logistic Regression",y_pred1,confusion_matrix,accuracy_score1]).transpose()
pipe1_results.columns = ["Model", "Predictions", "Confusion Matrix", "Accuracy"]
print(pipe1_results)

pipe2_results = pd.DataFrame(["MultinomialNB",y_pred2,confusion_matrix,accuracy_score2]).transpose()
pipe2_results.columns = ["Model", "Predictions", "Confusion Matrix", "Accuracy"]
print(pipe2_results)

pipe3_results = pd.DataFrame(["SVC",y_pred3,confusion_matrix,accuracy_score3]).transpose()
pipe3_results.columns = ["Model", "Predictions", "Confusion Matrix", "Accuracy"]
print(pipe3_results)

df_pipes = pd.concat([pipe1_results, pipe2_results, pipe3_results],axis=0)

print(df_pipes.reset_index())


    