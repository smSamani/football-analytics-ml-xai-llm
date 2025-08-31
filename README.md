
# Football Analytics Modular Pipeline

A modular pipeline for football analytics using machine learning (K-Means, XGBoost, SHAP) and LLM-powered reporting.  
Transforms StatsBomb event data into coach-friendly tactical insights and automated, structured reports.

---

## 🚀 Project Overview

This project builds a **reproducible, modular football analytics pipeline** that:
- Processes raw StatsBomb event data (open-access)
- Extracts match/sequence features for clustering and classification
- Applies machine learning models and custom interpretability modules
- Automatically generates visuals and coach-ready reports using Gemini LLM API

The system is designed for **real-world coaching workflows**—not just data science experiments.

---

## 📁 Directory Structure

Only the `data/` and `scripts/` folders need to be provided by the user.  
All other folders (outputs, reports, processed datasets) are auto-generated.

```
project-root/
│
├── data/
│   ├── raw/
│   │   ├── Leverkusen_Matchweeks_Info.csv
│   │   └── Leverkusen_Matches/
│   │       ├── LeverkusenVSAugsburg_Home/
│   │       │   ├── event.json
│   │       │   ├── lineup.json
│   │       │   └── 360.json
│   │       ├── LeverkusenVSBochum_Away/
│   │       │   ├── event.json
│   │       │   ├── lineup.json
│   │       │   └── 360.json
│   │       └── ...
├── scripts/
│   └── [all pipeline Python scripts]
├── outputs/      # (auto-generated)
├── reports/      # (auto-generated)
├── processed/    # (auto-generated)
├── README.md
└── requirements.txt
```

**Required for each match:**  
- `event.json`: the StatsBomb event log  
- `lineup.json`: match lineup  
- `360.json`: freeze-frame data  

---

## ⚙️ Requirements

- **Python 3.9+**  
- Recommended to use a clean virtual environment.

**Python packages:**  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- shap  
- joblib  
- packaging  
- mplsoccer  
- google-generativeai  
- PyQt6  
- kneed  

_Install via:_  
```bash
pip install -r requirements.txt
```

---

## 🔑 Gemini API Key

LLM-generated reports use [Gemini API](https://ai.google.dev/) for football-language narratives.

**How to set your Gemini API key:**

Linux/macOS:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Windows PowerShell:
```powershell
setx GEMINI_API_KEY "your_api_key_here"
```

---

## 🚦 How to Run

1. Place all required input files in `data/raw/` as shown above.
2. Run the main pipeline script:
   ```bash
   python main.py
   ```
3. All outputs (processed data, visuals, reports) are generated automatically in the correct folders.

---

## 🏆 Example Outputs

> Automated coach-facing tactical report, SHAP zone heatmaps, radar profiles, etc.

---

## 📣 Disclaimer & Academic Notice

> This project and codebase were developed as part of an MSc dissertation at Kent Business School, University of Kent (2025).  
> **Full dissertation publication is pending official university approval and grading.**  
> For academic or professional inquiries, contact:  
> **Soroush Mohammadi Samani**  
> linkedin.com/in/soroush-mohammadi-samani-b14294200 | sms1618w@gmail.com

_Use for educational and non-commercial research only. StatsBomb data is open-access for research purposes._

---

## 📄 License

MIT License (see LICENSE file).  
For research and educational use. Please cite this repository if you use this pipeline in your work.

---
