# 🎬 IMDB Sentiment Analysis: TF-IDF vs BERT

Binary sentiment classification of IMDB movie reviews (50k reviews),
comparing a classical ML baseline with a fine-tuned BERT model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/imdb-sentiment-analysis-bert/blob/main/sentiment_analysis_imdb.ipynb)

## Results
| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression | ~88.8% |
| BERT (bert-base-uncased, fine-tuned) | ~92% |

## Pipeline
1. Exploratory Data Analysis (class balance, review length, word clouds)
2. Text preprocessing (HTML/URL removal, stopwords)
3. Baseline: TF-IDF (1-2 grams) + Logistic Regression
4. BERT fine-tuning with PyTorch + Hugging Face Transformers
5. Evaluation: accuracy, precision/recall/F1, confusion matrix

## Tech Stack
Python, Scikit-learn, PyTorch, Transformers, Pandas, NLTK, Matplotlib, Seaborn

## Run
Open the notebook in Google Colab (GPU runtime recommended for BERT)
or run locally:
pip install datasets transformers torch scikit-learn matplotlib seaborn wordcloud nltk accelerate
