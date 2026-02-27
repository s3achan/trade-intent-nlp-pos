# 📈 Buy/Sell Intent Detection using NLP + POS + Sentiment

## 🚀 Project Overview

This project builds an NLP pipeline to detect **Buy and Sell intent signals** from financial text using:

- Hugging Face dataset: `HugoGiddins/buy_sell_intent`
- Keyword-based sentence extraction
- Part-of-Speech (POS) tagging
- Sentiment analysis
- Feature engineering
- Intent classification modeling

The goal is to simulate a lightweight **trading signal intelligence system** capable of identifying directional intent (Buy / Sell) from financial commentary, analyst notes, or social media text.

---

## 📊 Dataset

**Source:** Hugging Face  
**Dataset:** `HugoGiddins/buy_sell_intent`

The dataset contains labeled financial text with buy/sell intent classification.

Example record:

```python
{
  "text": "Strong buy signal forming after breakout",
  "label": 1
}

🛠 Tech Stack

Python

Pandas

NumPy

Regex

NLTK (POS Tagging)

Scikit-learn

Hugging Face Datasets

Matplotlib / Seaborn (optional)


🧠 Project Architecture
Raw Financial Text
        ↓
Text Cleaning & Normalization
        ↓
Buy/Sell Keyword Detection
        ↓
Sentence Extraction
        ↓
POS Tagging
        ↓
Sentiment Scoring
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
Buy / Sell Prediction
🔍 Key Components
1️⃣ Keyword-Based Extraction

Custom buy/sell keyword arrays are converted into dynamic regex patterns.

Example:

Buy: buy, long, bullish, accumulate, breakout

Sell: sell, short, bearish, dump, breakdown

This ensures flexible and scalable detection.

2️⃣ POS Tagging

We use NLTK to analyze grammatical structure of sentences.

Common trading patterns detected:

VERB + TICKER → "buy AAPL"

ADJ + NOUN → "bullish breakout"

VERB + support/resistance → "break below support"

POS signals help validate directional intent strength.

3️⃣ Sentiment Integration

Sentiment scoring strengthens prediction confidence:

Positive sentiment → stronger Buy bias

Negative sentiment → stronger Sell bias

Neutral sentiment → weak/no signal

4️⃣ Feature Engineering

Features include:

Keyword frequency

POS tag distribution

Sentiment polarity score

TF-IDF vectors

N-grams

Token length statistics

5️⃣ Model Training

Models evaluated:

Logistic Regression

Random Forest

Gradient Boosting

XGBoost (optional)

Evaluation Metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

📈 Example Output

Input:

"Strong buy signal forming on NVDA after breakout"

Model Output:

Intent: BUY
Sentiment: Positive
Confidence Score: 0.91
POS Pattern: ADJ + VERB + NOUN
📦 Installation
git clone https://github.com/yourusername/buy-sell-intent-nlp.git
cd buy-sell-intent-nlp
pip install -r requirements.txt
📋 Requirements
pandas
numpy
nltk
scikit-learn
datasets
matplotlib
seaborn
🔬 Future Improvements

Integrate FinBERT / Transformer models

Deploy Streamlit web interface

Add Reddit / Twitter scraping

Backtest signals against historical stock data

Add confidence-based alert system

🎯 Business Applications

Retail trading sentiment dashboards

Risk monitoring tools

Social media alpha detection

Automated buy/sell signal alerts

Quant research experimentation

📌 Why This Project Matters

Financial text is unstructured and noisy.

This project demonstrates:

Practical NLP feature engineering

Integration of POS + sentiment analysis

Intent classification modeling

Real-world financial signal simulation

👩‍💻 Author