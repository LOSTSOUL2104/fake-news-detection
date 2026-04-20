# 📰 Fake News Detection on Social Media Platforms

A machine learning research project that classifies news articles as **real or fake** using Natural Language Processing (NLP) techniques. Built with Python, trained on the **ISOT Fake News Dataset**, and published as an IEEE-formatted research paper.

> **Authors:** Priyansh Arora, Mukund, Paramveer, Piyush | Supervised by Dr. Ajay Katiyar  
> **Institution:** Chitkara University Institute of Engineering and Technology, Punjab, India

---

## 📁 Project Structure

```
fake-news-detection/
├── detection.ipynb       # Full notebook: end-to-end pipeline
├── Fake.csv              # Dataset of fake news articles
├── True.csv              # Dataset of real/true news articles
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/LOSTSOUL2104/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `detection.ipynb` in the browser and run all cells from top to bottom.

---

## 🧠 How It Works

1. **Data Loading** — Loads `Fake.csv` and `True.csv`, assigns labels (`0` = Fake, `1` = Real), and merges them into a single shuffled dataset of ~44,000 articles.

2. **Exploratory Data Analysis** — Visualizes class distribution, subject distribution, and article word-length distribution.

3. **Text Preprocessing** — Cleans text by lowercasing, removing non-alphabetic characters, and stripping stopwords.

4. **Feature Extraction** — Converts cleaned text into numerical features using **TF-IDF Vectorization** (`max_df=0.7`).

5. **Model Training** — Trains a **Passive Aggressive Classifier (PAC)** on the TF-IDF features — well suited for large-scale, high-dimensional text classification.

6. **Baseline Comparison** — Benchmarks PAC against Logistic Regression, Naive Bayes, and Linear SVM.

7. **Evaluation** — Reports accuracy, precision, recall, F1-score, confusion matrix, and 5-fold cross-validation results.

---

## 📊 Dataset

| File       | Description              |
| ---------- | ------------------------ |
| `Fake.csv` | Articles labeled as fake |
| `True.csv` | Articles labeled as real |

Both files contain columns: `title`, `text`, `subject`, `date`.  
Source: **ISOT Fake News Dataset** (~44,000 labeled articles).

---

## 📦 Dependencies

| Library        | Purpose                         |
| -------------- | ------------------------------- |
| `pandas`       | Data loading and manipulation   |
| `numpy`        | Numerical operations            |
| `scikit-learn` | ML models and TF-IDF vectorizer |
| `nltk`         | Stopword removal                |
| `matplotlib`   | Plotting and visualization      |
| `seaborn`      | Enhanced visualizations         |

---

## ✅ Results

| Model                    | Accuracy | Precision | Recall | F1 Score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Passive Aggressive (PAC) | ~99.4%   | ~99.4%    | ~99.4% | ~99.4%   |
| Linear SVM               | ~99.4%   | ~99.4%    | ~99.4% | ~99.4%   |
| Logistic Regression      | ~98.8%   | ~98.8%    | ~98.8% | ~98.8%   |
| Naive Bayes              | ~95.2%   | ~95.2%    | ~95.2% | ~95.2%   |

5-Fold Cross-Validation (PAC) confirms stable performance with low variance.

---

## 👥 Team Contributions

| #   | Role                       | Contributor | Steps                                                                         | Work                                                                                  |
| --- | -------------------------- | ----------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 1   | Concept & Methodology Lead | Priyansh    | Library setup, dataset loading, exploratory data analysis, text preprocessing |
| 2   | Data & Preprocessing Lead  | Mukund      | Train-test split, TF-IDF vectorization, PAC training, model evaluation        |
| 3   | Model Development Lead     | Paramveer   || Confusion matrix, baseline comparisons, visualization, cross-validation      |
| 4   | Evaluation & Writing Lead  | Piyush      || Feature analysis, predicted label distribution, final summary, manuscript preparation |

---

## 🛠️ Troubleshooting

- **NLTK stopwords not found?** Run this before preprocessing:

  ```python
  import nltk
  nltk.download('stopwords')
  ```

- **FileNotFoundError for CSVs?** Make sure `Fake.csv` and `True.csv` are in the **same directory** as `detection.ipynb`.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
