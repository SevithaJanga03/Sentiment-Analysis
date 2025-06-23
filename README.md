# 📊 Sentiment Analysis of Amazon Customer Reviews

A hybrid NLP project that performs sentiment classification on 500K+ Amazon food product reviews using both rule-based and deep learning methods. This project compares the performance of VADER (Valence Aware Dictionary for sEntiment Reasoning) and transformer-based RoBERTa for multi-class sentiment analysis: Positive, Neutral, and Negative.

## 🧠 Project Overview

- **Goal:** Classify customer sentiments from Amazon reviews into Positive, Neutral, or Negative.
- **Models Used:** 
  - ✅ VADER + Random Forest (TF-IDF features)
  - 🤖 RoBERTa (HuggingFace) + TF-IDF + Random Forest
- **Dataset:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Total Records:** ~568,000

---

## 🔧 Features & Workflow

### 🗃️ 1. Data Preprocessing
- HTML tag removal
- Lowercasing, punctuation & stop word removal
- Tokenization & Lemmatization
- Rating-to-sentiment mapping:
  - 1–2 → Negative
  - 3 → Neutral
  - 4–5 → Positive
- Balanced dataset using undersampling

### 📈 2. Feature Engineering
- TF-IDF Vectorization (Top 5000 features)
- Sentiment scores using VADER
- Sentiment labels using RoBERTa

### 🔍 3. Model Building
- **VADER:** Lexicon-based rule sentiment + Random Forest classifier
- **RoBERTa:** Transformer-based classification with Hugging Face
- **Hybrid Labeling:** Combined rating and model output

### 🧪 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves
- Stratified 5-Fold Cross-Validation

---

## 📊 Results Summary

| Model     | Accuracy | Macro F1 Score | ROC AUC Range |
|-----------|----------|----------------|----------------|
| VADER     | 93.2%    | 0.82           | 0.95 – 0.97    |
| RoBERTa   | 86.6%    | 0.84           | 0.95 – 0.98    |

---

## 🖼️ Visualizations

- Sentiment distribution before & after balancing
- Confusion matrices for VADER & RoBERTa
- ROC curves for all classes

---

## 💡 Key Learnings

- Rule-based models like VADER are fast and effective for large-scale data.
- RoBERTa performs well with complex language and nuanced sentiment but is computationally heavier.
- Balancing classes significantly improves model fairness and performance.
- Cross-validation ensures model stability and generalization.

---

## 📦 Tools & Technologies

- **Languages:** Python
- **Libraries:** Pandas, Scikit-learn, NLTK, VADER, Transformers (Hugging Face), Matplotlib, Seaborn
- **Models:** VADER, RoBERTa, Random Forest
- **Evaluation:** Sklearn metrics, Cross-validation

---

## 🚀 Future Improvements

- Ensemble methods combining VADER and RoBERTa predictions
- Aspect-Based Sentiment Analysis (ABSA)
- Real-time streaming sentiment classification
- Multilingual sentiment classification using mBERT / XLM-R

---

## 🙏 Acknowledgements

Thanks to Professor **Jesus Gonzalez Bernal** and TA **Manish Munikar**, University of Texas at Arlington, for their guidance and support.

---

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- Research papers and citations available in the full report

---

## 🧑‍💻 Authors

- **Kishore Kumar Sunke**  
- **Sevitha Janga**  
- **Sai Sree Chitturi**  

---

