# 📈 Buy/Sell Intent Detection using NLP, POS & Sentiment Analysis

## 🚀 Project Overview

This project builds an end-to-end NLP pipeline to detect **Buy and Sell
trading intent** from financial text.

It combines:

-   Hugging Face dataset: `HugoGiddins/buy_sell_intent`
-   Keyword-based sentence extraction
-   Part-of-Speech (POS) tagging
-   Sentiment analysis
-   Feature engineering
-   Supervised machine learning classification

The goal is to simulate a lightweight **trading signal intelligence
system** capable of identifying directional intent (Buy / Sell) from
analyst commentary, financial news, and social media posts.

------------------------------------------------------------------------

## 📊 Dataset

**Source:** Hugging Face\
**Dataset:** `HugoGiddins/buy_sell_intent`

The dataset contains labeled financial sentences with buy/sell intent.

### Example Record

``` python
{
  "text": "Strong buy signal forming after breakout",
  "label": 1
}
```

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   Pandas
-   NumPy
-   Regex
-   NLTK (POS Tagging)
-   Scikit-learn
-   Hugging Face Datasets
-   Matplotlib / Seaborn (optional)

------------------------------------------------------------------------

## 🧠 Project Architecture

Raw Financial Text\
↓\
Text Cleaning & Normalization\
↓\
Buy/Sell Keyword Detection\
↓\
Sentence Extraction\
↓\
POS Tagging\
↓\
Sentiment Scoring\
↓\
Feature Engineering\
↓\
Machine Learning Model\
↓\
Buy / Sell Prediction

------------------------------------------------------------------------

## 🔍 Key Components

### 1️⃣ Keyword-Based Extraction

Custom buy/sell keyword arrays are dynamically converted into regex
patterns.

**Buy Keywords** - buy - long - bullish - accumulate - breakout

**Sell Keywords** - sell - short - bearish - dump - breakdown

------------------------------------------------------------------------

### 2️⃣ POS Tagging

NLTK is used to analyze sentence structure.

Common trading signal patterns:

-   VERB + TICKER → "buy AAPL"
-   ADJ + NOUN → "bullish breakout"
-   VERB + support/resistance → "break below support"

------------------------------------------------------------------------

### 3️⃣ Sentiment Integration

Sentiment scoring enhances directional confidence:

-   Positive sentiment → Stronger Buy bias\
-   Negative sentiment → Stronger Sell bias\
-   Neutral sentiment → Weak or no signal

------------------------------------------------------------------------

### 4️⃣ Feature Engineering

Engineered features include:

-   Keyword frequency
-   POS tag distribution
-   Sentiment polarity score
-   TF-IDF vectors
-   N-grams
-   Token length statistics

------------------------------------------------------------------------

### 5️⃣ Model Training

Models evaluated:

-   Logistic Regression
-   Random Forest
-   Gradient Boosting
-   XGBoost (optional)

**Evaluation Metrics** - Accuracy - Precision - Recall - F1 Score -
Confusion Matrix

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/yourusername/stock-intent-nlp.git
cd stock-intent-nlp
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📋 Requirements

    pandas
    numpy
    nltk
    scikit-learn
    datasets
    matplotlib
    seaborn

------------------------------------------------------------------------

## 🔬 Future Improvements

-   Integrate FinBERT / Transformer models
-   Deploy Streamlit web interface
-   Add Reddit / financial news scraping
-   Backtest signals against historical stock data
-   Implement confidence-based alert scoring

------------------------------------------------------------------------

## 🎯 Business Applications

-   Trading sentiment dashboards
-   Risk monitoring tools
-   Social media alpha extraction
-   Automated buy/sell alert systems
-   Quantitative research prototyping

------------------------------------------------------------------------

## 📌 Why This Project Matters

Financial text is unstructured and noisy.

This project demonstrates:

-   Practical NLP feature engineering\
-   POS + sentiment integration\
-   Intent classification modeling\
-   Real-world financial signal simulation

------------------------------------------------------------------------

## 👩‍💻 Author

Shikshya Bhattachan\
NLP & ML Enthusiast
