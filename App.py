import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Title
# -------------------------------
st.title("📩 Spam Detection ML App")
st.write("Compare models & predict Spam vs Ham messages")

# -------------------------------
# Dataset
# -------------------------------
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

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -------------------------------
# Models (Pipelines)
# -------------------------------
pipe1 = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])

pipe2 = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

pipe3 = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", SVC(probability=True))
])

# Train models
pipe1.fit(X_train, y_train)
pipe2.fit(X_train, y_train)
pipe3.fit(X_train, y_train)

# Predictions
y_pred1 = pipe1.predict(X_test)
y_pred2 = pipe2.predict(X_test)
y_pred3 = pipe3.predict(X_test)

# Accuracy
acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)
acc3 = accuracy_score(y_test, y_pred3)

# -------------------------------
# Sidebar - Model Selection
# -------------------------------
st.sidebar.title("⚙️ Select Model")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Naive Bayes", "SVM"]
)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area("✉️ Enter Message")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        if model_choice == "Logistic Regression":
            model = pipe1
            acc = acc1
        elif model_choice == "Naive Bayes":
            model = pipe2
            acc = acc2
        else:
            model = pipe3
            acc = acc3

        pred = model.predict([user_input])[0]
        prob = model.predict_proba([user_input]).max()

        if pred == "spam":
            st.error(f"🚨 Spam (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Ham (Confidence: {prob:.2f})")

        st.info(f"Model Accuracy: {acc:.2f}")

# -------------------------------
# Model Comparison Table
# -------------------------------
st.subheader("📊 Model Comparison")

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "SVM"],
    "Accuracy": [acc1, acc2, acc3]
})

st.dataframe(results)

# -------------------------------
# Confusion Matrix Visualization
# -------------------------------
st.subheader("📉 Confusion Matrix")

if model_choice == "Logistic Regression":
    cm = confusion_matrix(y_test, y_pred1)
elif model_choice == "Naive Bayes":
    cm = confusion_matrix(y_test, y_pred2)
else:
    cm = confusion_matrix(y_test, y_pred3)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(model_choice)

st.pyplot(fig)

# -------------------------------
# Show Dataset
# -------------------------------
if st.checkbox("Show Dataset"):
    st.write(df)