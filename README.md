# 🚀 Machine Learning & AI Notebooks Collection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![ML](https://img.shields.io/badge/Machine-Learning-success.svg)](https://github.com/CodeRod-Solutions/Notebooks)
[![GCP](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4.svg)](https://cloud.google.com/vertex-ai)

> A comprehensive collection of machine learning, deep learning, and AI projects covering various domains from predictive modeling to natural language processing and cloud-based ML operations.

---

## 📚 Table of Contents

- [Projects Overview](#-projects-overview)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Project Details](#-project-details)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Projects Overview

This repository contains **8 distinct machine learning projects**, each focusing on different aspects of data science, AI, and cloud computing:

| Project | Domain | Tech Stack | Status |
|---------|--------|-----------|--------|
| [🏠 Linear Regression](#-linear-regression--housing-prediction) | Supervised Learning | scikit-learn, pandas | ✅ Complete |
| [📊 Customer Churn Analysis](#-customer-churn-prediction) | Classification | ML Models, EDA | ✅ Complete |
| [📰 Fake News Detection](#-fake-news-detection) | NLP | Text Classification | ✅ Complete |
| [📈 Stock Sentiment Analysis](#-stock-sentiment-analysis) | NLP & Finance | Sentiment Analysis | ✅ Complete |
| [💪 Workout ML Analysis](#-workout-ml-analysis) | Time Series | Data Analytics | ✅ Complete |
| [📄 AI Resume Builder](#-ai-resume-builder) | Generative AI | OpenAI, Gemini | ✅ Complete |
| [🤖 RAG System](#-rag-system) | LLM & RAG | Vector DB, Embeddings | ✅ Complete |
| [☁️ Vertex AI Training](#️-vertex-ai-training-service) | MLOps | GCP, Docker | ✅ Complete |

---

## 🛠️ Technologies Used

### **Core ML & Data Science**
- **Python 3.8+** - Primary programming language
- **scikit-learn** - Machine learning algorithms
- **pandas & numpy** - Data manipulation and analysis
- **matplotlib & seaborn** - Data visualization

### **Deep Learning & NLP**
- **TensorFlow / PyTorch** - Deep learning frameworks
- **sentence-transformers** - Embedding models
- **langchain** - LLM orchestration

### **Cloud & MLOps**
- **Google Cloud Platform (GCP)** - Cloud infrastructure
- **Vertex AI** - ML model training and deployment
- **BigQuery** - Data warehousing
- **Docker** - Containerization

### **Generative AI**
- **OpenAI API** - GPT models
- **Google Gemini** - Generative AI capabilities
- **ChromaDB** - Vector database for RAG

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install Jupyter Notebook
pip install jupyter
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CodeRod-Solutions/Notebooks.git
cd Notebooks
```

2. **Install dependencies** (each project has its own requirements)
```bash
# For a specific project
cd <project-directory>
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

---

## 📖 Project Details

### 🏠 Linear Regression - Housing Prediction
**Location:** `ML_LinearRegression/`

Predict US housing prices using Linear Regression and Decision Tree Regression models with comprehensive EDA and model evaluation.

**Key Features:**
- Data preprocessing and cleaning
- Exploratory Data Analysis with Seaborn
- Multiple regression models (Linear & Decision Tree)
- Performance metrics (MAE, MSE, RMSE)
- Residual analysis and visualization

**Technologies:** pandas, numpy, scikit-learn, matplotlib, seaborn

---

### 📊 Customer Churn Prediction
**Location:** `Customer_Churn/`

Analyze and predict customer churn for a telecommunications company using machine learning classification models.

**Key Features:**
- Telco customer dataset analysis
- Feature engineering and preprocessing
- Classification model development
- Churn pattern identification

**Notebook:** `Telco_Churn.ipynb`

---

### 📰 Fake News Detection
**Location:** `Fake_news/`

Build an NLP-based classifier to detect fake news articles using natural language processing techniques.

**Key Features:**
- Text preprocessing and cleaning
- Feature extraction (TF-IDF, word embeddings)
- Binary classification (Real vs. Fake)
- Model evaluation and accuracy metrics

**Files:**
- `Fake_News.ipynb` - Main implementation notebook
- `requirements.txt` - Project dependencies

---

### 📈 Stock Sentiment Analysis
**Location:** `Stock_Sentiment_Analysis/`

Perform sentiment analysis on stock-related news and social media to gauge market sentiment.

**Key Features:**
- Financial text data processing
- Sentiment classification (Positive/Negative/Neutral)
- Visualization of sentiment trends
- Correlation with stock price movements

**Files:**
- `stock_sentiment.ipynb` - Interactive notebook
- `stock_sentiment.py` - Python script for batch processing

[📖 Read Full Documentation](./Stock_Sentiment_Analysis/README.md)

---

### 💪 Workout ML Analysis
**Location:** `Workout_ML/`

Analyze workout data using machine learning to identify patterns, track progress, and provide insights.

**Key Features:**
- Time series analysis of workout metrics
- Performance trend visualization
- Statistical analysis of exercise data
- Predictive modeling for fitness goals

**Files:**
- `Workout_Analysis.ipynb` - Main analysis notebook
- `basic_analysis.py` - Helper functions

---

### 📄 AI Resume Builder
**Location:** `Resume_Builder/`

Generate professional resumes using AI-powered language models (OpenAI GPT and Google Gemini).

**Key Features:**
- Two implementations: OpenAI and Gemini
- Automated resume generation
- Customizable templates
- Content optimization using LLMs

**Files:**
- `resume_builder_OpenAI.ipynb` - OpenAI GPT implementation
- `resume_builder_Gemini.ipynb` - Google Gemini implementation

---

### 🤖 RAG System
**Location:** `RAG_SYSTEM/`

Implement a Retrieval-Augmented Generation (RAG) system for enhanced question-answering capabilities.

**Key Features:**
- PDF document processing and chunking
- Embedding generation with SentenceTransformers
- Vector database integration (ChromaDB)
- Query augmentation and result re-ranking
- UMAP visualization of embeddings
- Embedding adapter fine-tuning

**Technologies:** Google Vertex AI, ChromaDB, LangChain, UMAP

**Author:** Rod Morrison

[📖 Read Full Documentation](./RAG_SYSTEM/README.md)

---

### ☁️ Vertex AI Training Service
**Location:** `GCP_VERTEX/Vertex_ML_Training/`

Scale machine learning model training using Google Cloud's Vertex AI platform with containerization support.

**Key Features:**
- Transition from local to cloud training
- Python package organization for ML code
- BigQuery data integration
- Docker containerization
- Custom training jobs on Vertex AI
- Cloud Storage integration

**Prerequisites:**
- Google Cloud account
- Vertex AI API enabled
- BigQuery API enabled
- Cloud Storage bucket

**Files:**
- `TRAINING_AT_SCALE_VERTEX.ipynb` - Main training guide
- `taxifare/` - Training package with model and task modules

[📖 Read Full Documentation](./GCP_VERTEX/Vertex_ML_Training/README.md)

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve any of these notebooks or add new projects:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Rod Morrison** - CodeRod Solutions

- GitHub: [@CodeRod-Solutions](https://github.com/CodeRod-Solutions)
- Repository: [Notebooks](https://github.com/CodeRod-Solutions/Notebooks)

---

## 📝 License

This repository is available for educational and reference purposes. Please check individual project folders for specific licensing information.

---

## 🌟 Acknowledgments

- Google Cloud Platform for Vertex AI infrastructure
- OpenAI and Google for LLM APIs
- The open-source community for amazing ML libraries

---

<div align="center">

**⭐ If you find these projects helpful, please consider giving this repository a star!**

Made with ❤️ and ☕ by CodeRod Solutions

</div>
