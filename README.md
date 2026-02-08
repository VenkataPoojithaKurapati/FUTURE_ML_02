# FUTURE_ML_02

## 🖥 IT Service Ticket Classification  
**Machine Learning + NLP + Deep Learning Project**  

---

## 🚀 Project Overview
This project focuses on **automatic classification of IT service tickets** into their respective categories using historical ticket data.  
It combines **Machine Learning, NLP, and Deep Learning (LSTM)** techniques to create accurate, end-to-end ticket classification pipelines.

**Objectives:**
- Automate ticket routing in IT service management  
- Improve response time and operational efficiency  
- Explore ML and Deep Learning pipelines for text classification  

---

## 📁 Dataset
- Dataset: `all_tickets_processed_improved_v3.csv.zip`  
- Data Structure:

| Column      | Description                        |
|------------|------------------------------------|
| Document   | Text content of the IT ticket       |
| Topic_group | Category/group assigned to ticket  |

The dataset is preprocessed with advanced text cleaning and feature engineering for optimal model performance.

---

## 🔧 Text Preprocessing
All pipelines include:

- Lowercasing  
- Removing emails, URLs, numbers, and punctuation  
- Removing extra whitespace  
- Tokenization (for LSTM)  
- TF-IDF vectorization (for ML models)  

---

## 🧠 Machine Learning Pipelines

### 1. Traditional ML Models
- **Logistic Regression** (baseline & optimized)  
- **Random Forest Classifier**  

**Key Features:**
- TF-IDF vectorization (unigrams + bigrams)  
- Stopwords removal  
- Stratified train-test split  

**Evaluation Metrics:**
- Accuracy  
- Classification report  
- Confusion matrix visualization  

**Real-time Prediction Function:**  
`predict_ticket_category(text)` predicts the category of new ticket text.

---

### 2. Optimized NLP Pipeline
- **Pipeline:** TF-IDF + Logistic Regression  
- **Hyperparameter Tuning:** GridSearchCV for `C` parameter  
- **Cross-Validation:** 5-fold cross-validation to evaluate performance  
- **Model Persistence:** Model & TF-IDF vectorizer saved with `joblib`

---

### 3. Deep Learning Pipeline (LSTM)
- Preprocessing: Tokenization, Padding, Label Encoding  
- Model: Bidirectional LSTM with Dropout layers  
- Input Shape: Maximum sequence length = 120  
- Output: Softmax for multi-class classification  
- Training: 6 epochs, batch size 128  
- Prediction Function: `predict_ticket_lstm(text)` for real-time inference  

**Advantages of LSTM:**
- Captures sequential patterns in text  
- Handles long-term dependencies in ticket descriptions  

---

## 📊 Model Evaluation
- **Accuracy & Classification Report** for each model  
- **Confusion Matrix** visualization for error analysis  
- Cross-validation scores for ML pipelines  
- Final LSTM accuracy reported on test set  

---

## 💾 Saved Artifacts
- `ticket_classifier_model.pkl` → Logistic Regression model  
- `tfidf_vectorizer.pkl` → TF-IDF vectorizer for ML pipelines  
- LSTM tokenizer and model (can be saved via `model.save()` for deployment)  

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn, GridSearchCV
- Matplotlib, Seaborn
- Joblib for model persistence
- TensorFlow / Keras for LSTM deep learning
- Regular Expressions & String Cleaning for NLP

---

## 📦 Project Workflow
1. Data Loading
2. Data Cleaning / Preprocessing
3. Train-Test Split (Stratified)
4. TF-IDF Vectorization (for ML models)
5. Model Training (Logistic Regression, Random Forest, GridSearch)
6. Model Evaluation (Accuracy, Classification Report, Confusion Matrix)
7. LSTM Model Training & Evaluation
8. Real-Time Prediction Functions
9. Saving Models & Vectorizers
10. Sample Predictions

---

## 🎯 Key Learning Outcomes
- End-to-end NLP and ML pipelines for text classification
- Feature engineering with TF-IDF and n-grams
- Model evaluation & visualization
- Deep learning with LSTM for sequence modeling
- Real-time inference functions for practical applications

---

## 🔮 Future Improvements
- Hyperparameter tuning for LSTM (more epochs, batch size optimization)
- Integration with live ITSM systems for automated ticket routing
- Use pretrained embeddings (GloVe / Word2Vec / BERT) for better text representation
- Handling class imbalance with SMOTE or focal loss


