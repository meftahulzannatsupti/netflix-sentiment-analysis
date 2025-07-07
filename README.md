# 🎬 Netflix Review Sentiment Analysis using NLP and Machine Learning

This project explores sentiment classification of Netflix user reviews using a mix of traditional machine learning and deep learning models. The goal is to uncover patterns in textual feedback that could enhance user experience and help optimize streaming platform recommendation systems.

---

## 🧠 Objective

To classify the sentiment of Netflix reviews using Natural Language Processing (NLP) and assess the performance of both conventional machine learning models and deep learning architectures.

---

## 📊 Summary of Findings

- **Best Performing Model:** Logistic Regression  
  - Accuracy: **72.68%**
  - Precision for Positive Sentiment: **74%**
  - Recall: **94%**
- **Baseline Model Accuracy:** 60.02%
- **Multinomial Naive Bayes:**
  - Pre-optimization accuracy: **69.11%**
  - Post-optimization accuracy: **69.65%**
- **Logistic Regression:**
  - Pre-optimization: **72.80%**
  - Post-optimization: **72.68%**

Despite traditional models achieving reasonable performance, deep learning models like LSTM and GRU underperformed significantly:

- **LSTM Accuracy:** ~39.43% (after 5 epochs)
- **GRU Accuracy:** ~39.35% (over 10 epochs)

These results reveal deep learning's difficulty in capturing the sentiment complexity in this dataset without major architecture or data enhancements.

---

## 📌 Key Metrics (Multi-Class Sentiment)

- Sentiment Score 1:  
  - Precision: 58%  
  - Recall: 0.96  
  - F1-Score: 0.72  

- Sentiment Score 2:  
  - Precision: 0%  
  - Recall: 0%  
  - F1-Score: 0  

- Sentiment Score 5:  
  - F1-Score: 0.72  

These discrepancies highlight the classifier's limitations in predicting intermediate sentiment classes accurately.

---

## 🧪 Techniques Used

- **NLP Preprocessing:** Tokenization, Stopword Removal, Stemming
- **Feature Extraction:** TF-IDF Vectorization
- **Models:**
  - Multinomial Naive Bayes
  - Logistic Regression
  - LSTM (Keras)
  - GRU (Keras)
- **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Cross-Validation:** 5-fold

---

## 📁 Project Structure

- `netflix_sentiment.ipynb` — Jupyter Notebook with preprocessing, modeling, and evaluation
- `reviews_sample.csv` — A sample dataset of user reviews and sentiment labels
- `README.md` — Project documentation
- `requirements.txt` — Required libraries (scikit-learn, nltk, pandas, keras, tensorflow)

---

## 💡 Conclusion

This study highlights the challenges and intricacies of sentiment classification, particularly when dealing with real-world, unstructured review text. While traditional machine learning models like Logistic Regression and Naive Bayes performed relatively well, deep learning models (LSTM, GRU) did not outperform them despite optimization.

The results emphasize the need for:

- More sophisticated architectures for deep learning
- Improved data preprocessing and balancing
- Possibly leveraging transfer learning (e.g., BERT, RoBERTa)

This research contributes to the ongoing development of robust sentiment analysis systems that are better suited to complex review data, such as those from streaming platforms like Netflix.

---

## 👤 Author

**Meftahul Zannat**  
MSc Artificial Intelligence & Data Science — University of Hull  
📧 meftahulzannat3598@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/meftahul-zannat-susupti98)  
🔗 [GitHub](https://github.com/meftahulzannatsupti)

---

## 🚀 Future Improvements

- Explore transformer-based models (BERT, DistilBERT)
- Implement data augmentation for imbalanced classes
- Deploy model using Streamlit for interactive visualization
