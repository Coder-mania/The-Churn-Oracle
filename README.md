# The-Churn-Oracle
**The Churn Oracle: AI-Enhanced Telecom Customer Retention with Predictive Modeling and Document Intelligence**

## Overview
The Churn Oracle is a mini-project designed to predict customer churn in the telecom sector using machine learning models such as XGBoost, SVM, LDA, and KNN. It also integrates SHAP for explainability and an AI-powered assistant to query telecom-related documents.

### Model Performance
![Homepage](Images/Img 3 model results.png)

### User Interface
![UHomepage(images/UI img.png)

## Technologies and Tools
| Component                     | Technology Used |
|---------------------------|----------|
| Web Framework                  | Streamlit      | 
| ML Modeling     | XGBoost, RandomizedSearchCV     |
| Explainability | SHAP (TreeExplainer)      |
| Document Processing | Langchain, FAISS, PyMuPDF |
| OCR & Privacy Masking | Tesseract |
| RAG Assistant | Langchain + OpenAI GPT Model |
| Programming Language | Python 3.x | 

## Features
- Predicts telecom customer churn
- Provides SHAP-based interpretability
- Interactive AI Assistant for document queries
- Simple Streamlit-based user interface

## Requirements
- Python 3.x
- Required libraries (install using):
```bash
pip install -r requirements.txt
