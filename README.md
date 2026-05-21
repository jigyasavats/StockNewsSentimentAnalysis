# StockNewsSentimentAnalysis

AI-driven sentiment analysis system that processes and analyzes NASDAQ stock news articles to gauge market sentiment, enabling more informed investment decisions.

## 📌 Problem Statement

Investment firms struggle to stay updated with the sheer volume of news and opinions from diverse sources. This project leverages AI to automatically analyze stock-related news and classify sentiment as **Positive**, **Neutral**, or **Negative** to support stock price predictions and optimize investment strategies.

## 📊 Dataset

The dataset contains historical daily news for a NASDAQ-listed company along with stock price and trade volume data.

| Feature  | Description |
|----------|-------------|
| `Date`   | Date the news was released |
| `News`   | Content of news articles affecting stock price |
| `Open`   | Stock price ($) at the beginning of the day |
| `High`   | Highest stock price ($) during the day |
| `Low`    | Lowest stock price ($) during the day |
| `Close`  | Adjusted stock price ($) at end of the day |
| `Volume` | Number of shares traded during the day |
| `Label`  | Sentiment polarity: `1` (Positive), `0` (Neutral), `-1` (Negative) |

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras, Gensim, Sentence Transformers, Hugging Face Transformers
- **Environment:** Google Colab

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis (countplots, histograms, boxplots)
- Bivariate analysis (label vs price/volume)
- Correlation analysis

### 2. Data Preprocessing
- Dropped stock price columns (Open, High, Low, Close, Volume)
- Label encoding
- Train-test split

### 3. Word Embeddings
- **Word2Vec** (300-dimensional vectors using Gensim)
- **Sentence Transformer** (`all-MiniLM-L6-v2` from Hugging Face)

### 4. Model Building

| # | Model Name | Embedding | Algorithm |
|---|------------|-----------|-----------|
| 1 | Word2Vec_RF_M1 | Word2Vec | Random Forest (depth=3) |
| 2 | Word2Vec_RF_M2 | Word2Vec | Random Forest (depth=4) |
| 3 | SentenceTransformer_RF_M1 | Sentence Transformer | Random Forest (depth=3) |
| 4 | SentenceTransformer_RF_M2 | Sentence Transformer | Random Forest (depth=4) |
| 5 | Word2Vec_NN_M1 | Word2Vec | Neural Network |
| 6 | Word2Vec_NN_M2 | Word2Vec | Neural Network |
| 7 | SentenceTransformer_NN_M1 | Sentence Transformer | Neural Network |
| 8 | SentenceTransformer_NN_M2 | Sentence Transformer | Neural Network |
| 9 | SentenceTransformer_NN_Binary | Sentence Transformer | Neural Network (Binary) |

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy==1.26.4 scikit-learn==1.6.1 scipy==1.13.1 gensim==4.3.3 sentence-transformers==3.4.1 pandas==2.2.2
```

### Run
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/StockNewsSentimentAnalysis.git
   ```
2. Open `Learner_Notebook_Full_Code.ipynb` in Google Colab or Jupyter Notebook
3. Upload the dataset and update the file path
4. Run all cells sequentially

## 📁 Project Structure

```
StockNewsSentimentAnalysis/
├── Learner_Notebook_Full_Code.ipynb   # Main notebook with full code
├── stock_news.csv                      # Dataset
└── README.md                           # Project documentation
```

## 📝 Key Findings

- Sentence Transformer embeddings outperformed Word2Vec embeddings across models
- Neural Networks showed better performance compared to Random Forest classifiers
- Binary sentiment classification (positive vs negative) yielded higher accuracy than 3-class classification
