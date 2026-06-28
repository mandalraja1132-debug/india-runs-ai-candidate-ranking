                    Job Description
                           │
                           ▼
                 Text Preprocessing
                           │
                           ▼
                 TF-IDF Vectorization
                           │
                           ▼
                  Cosine Similarity
                           │
                           ▼
Candidate Profiles → Rule-Based Scoring
                           │
                           ▼
                  Hybrid Scoring Engine
                           │
                           ▼
               Top 100 Ranked Candidates
                           │
                           ▼
                  submission.csv
# 🤖 AI-Powered Candidate Recommendation & Ranking System

## 📌 Overview

This project was developed for the **INDIA.RUNS Data & AI Challenge**.

The objective is to build an intelligent candidate ranking system that automatically recommends the most suitable candidates for a given job description. The solution combines **Natural Language Processing (NLP)** with a **hybrid scoring mechanism** to rank candidates based on semantic similarity and business-relevant factors.

---

## 🎯 Problem Statement

Recruiters often receive thousands of resumes for a single job opening. Manual screening is time-consuming, inconsistent, and prone to overlooking qualified candidates.

The goal of this project is to automate candidate discovery by:

* Understanding the job description
* Analyzing candidate profiles
* Ranking candidates according to their relevance
* Producing explainable recommendations

---

## 📂 Dataset

The challenge dataset contains:

| File                  | Description                            |
| --------------------- | -------------------------------------- |
| candidates.jsonl      | 100,000 candidate profiles             |
| job_description.docx  | Job description for Senior AI Engineer |
| candidate_schema.json | Candidate data structure               |
| sample_submission.csv | Required submission format             |

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity
* python-docx
* Jupyter Notebook

---

## 🔄 Project Workflow

1. Load candidate profiles from JSONL.
2. Extract relevant fields such as headline, summary, skills, experience, title, and company.
3. Convert nested JSON into a structured dataset.
4. Read the job description from DOCX format.
5. Create a unified **tags** field by combining important candidate information.
6. Calculate rule-based scores using skills, experience, title, and company.
7. Compute semantic similarity using TF-IDF and cosine similarity.
8. Combine both scores into a hybrid ranking score.
9. Select the Top 100 candidates.
10. Generate the final submission CSV.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Loaded 100,000 candidate profiles
* Flattened nested JSON fields
* Extracted skill names
* Cleaned missing values
* Converted text to lowercase
* Created a combined **tags** field for NLP processing

---

## 🧠 Feature Engineering

Each candidate profile was represented using a combined text field:

```
Tags =
Headline
+ Summary
+ Skills
+ Current Title
+ Current Company
```

This consolidated representation improves text matching between candidate profiles and the job description.

---

## 📊 Ranking Methodology

### Rule-Based Scoring

The model evaluates candidates based on:

* Skill Match
* Years of Experience
* Current Job Title
* Company Relevance

### Semantic Similarity

The project uses **TF-IDF Vectorization** followed by **Cosine Similarity** to measure how closely each candidate profile matches the job description.

### Hybrid Score

Final ranking is based on a hybrid score that combines:

* Rule-based score (70%)
* TF-IDF semantic similarity (30%)

This approach balances explainability with semantic matching.

---

## 📈 Results

* Processed **100,000 candidate profiles**
* Generated **Top 100 recommendations**
* Produced explainable rankings with:

  * Candidate ID
  * Rank
  * Score
  * Reasoning

---

## 📁 Project Structure

```
india-runs-ai-candidate-ranking/
│
├── data/
├── notebooks/
├── output/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run

1. Clone the repository.

```
git clone <repository-url>
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Open the Jupyter notebook.

```
jupyter notebook
```

4. Run all cells in sequence.

5. The final submission file will be generated inside the `output` folder.

---

## 📌 Future Improvements

* Replace TF-IDF with transformer-based sentence embeddings.
* Integrate vector databases such as FAISS or Pinecone.
* Implement Learning-to-Rank models.
* Add recruiter feedback for continuous improvement.
* Develop a Streamlit or FastAPI web application.

---

## 👨‍💻 Author

**Raja Mandal**

BCA (Data Science)

Python • SQL • Machine Learning • Data Analytics • Power BI

---

## 🙏 Acknowledgement

This project was developed as part of the **INDIA.RUNS Data & AI Challenge** to demonstrate AI-assisted candidate recommendation and ranking using NLP and explainable scoring techniques.
