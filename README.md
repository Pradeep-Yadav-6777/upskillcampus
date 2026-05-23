# 🌾 Prediction of Agriculture Crop Production in India

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red) ![Data](https://img.shields.io/badge/Data-data.gov.in-green)

**Industrial Internship Project** — upSkill Campus + UniConverge Technologies Pvt Ltd (UCT)

---

## 📌 Project Overview

This project builds a complete end-to-end **Machine Learning pipeline** to predict crop production volumes across Indian states using **real government data** from [data.gov.in](https://data.gov.in) covering **2001–2014**.

The solution uses **Random Forest Regression** as the primary model, achieving **R² = 0.9906** (99.06% accuracy) on unseen test data.

| | |
|---|---|
| **Intern** | Pradeep Kumar Yadav |
| **College** | Khwaja Moinuddin Chishti Language University, Lucknow |
| **Domain** | Data Science & Machine Learning |
| **Organisation** | upSkill Campus + UCT |
| **Duration** | 4 Weeks |
| **Dataset** | India Agriculture Production 2001–2014 · data.gov.in |

---

## 📁 Repository Structure

```
upskillcampus/
│
├── AgricultureCropPrediction.py        ← Main Python code (all models + 9 charts)
├── streamlit_app.py                    ← Streamlit deployment app
├── agriculture_india_dataset.csv       ← Merged master dataset (686 rows)
├── datafile (1).csv                    ← Cost of cultivation & yield by crop×state
├── datafile (2).csv                    ← Year-wise production 2006-2011
├── datafile (3).csv                    ← Crop variety, season, recommended zone
├── datafile.csv                        ← Production index numbers
├── produce.csv                         ← Macro agricultural statistics 1993-2014
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
│
├── images/                             ← 9 generated charts
│   ├── state_wise_production.png
│   ├── crop_wise_production.png
│   ├── seasonal_analysis.png
│   ├── yearly_trend.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── state_crop_heatmap.png
│
├── outputs/
│   └── model_results.txt
│
└── PredictionOfAgricultureCropProductionInIndia_Pradeep_USC_UCT.pdf
```

---

## 🔗 GitHub Links

| File | Link |
|------|------|
| Code | https://github.com/pradeepkumaryadav/upskillcampus/blob/main/AgricultureCropPrediction.py |
| Report | https://github.com/pradeepkumaryadav/upskillcampus/blob/main/PredictionOfAgricultureCropProductionInIndia_Pradeep_USC_UCT.pdf |

---

## 📂 Dataset (data.gov.in — 5 Files)

| File | Records | Description |
|------|---------|-------------|
| datafile (1).csv | 49 | Crop × State: Cost A2+FL, Cost C2, CostProd C2, Yield Q/Ha |
| datafile (2).csv | 55 | Year-wise Production, Area, Yield (2006-07 to 2010-11) |
| datafile (3).csv | 78 | Crop Variety, Season Duration, Recommended Zone |
| datafile.csv | 13 | Production Index Numbers (2004-05 to 2011-12, base=100) |
| produce.csv | 429 | Macro agri stats: Foodgrains, Oilseeds, Spices (1993-2014) |

---

## 🤖 Models & Results

| Metric | Linear Regression | Random Forest ✅ |
|--------|------------------|-----------------|
| R² Score | 0.8985 | **0.9906** |
| MAE (Tonnes) | 10,886 | **3,131** |
| RMSE (Tonnes) | 16,038 | **4,890** |
| Improvement | Baseline | +10.2% R², −71% MAE |

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/pradeepkumaryadav/upskillcampus.git
cd upskillcampus

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Python script
python AgricultureCropPrediction.py

# 4. Launch Streamlit app
streamlit run streamlit_app.py
```

---

## 📊 Sample Prediction

```
Input:  Wheat | Uttar Pradesh | Rabi | 2014 | Area=45,000 Ha | Cost=₹21,000/Ha | Yield=32.5 Q/Ha
Output: Predicted Production = 71,876.24 Tonnes
```

---

*Made with ❤️ by Pradeep Kumar Yadav — upSkill Campus + UCT Internship | Data: data.gov.in*
