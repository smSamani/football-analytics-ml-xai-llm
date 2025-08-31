# main.py

import os
from this import d
from scripts import (
    CaseDetection,
    Clustering,
    ZoneAnalysis,
    PlayerZoneAnalysis,
    DefenceClassification,
    Classification,
    DefenceCaseDetection,
    VisualAnalysis,
    k_selector_gui,
    DefenceZoneAnalysis,
    reportgenerator,
)
from scripts.k_selector_gui import show_k_selection_gui
# from utils.api_call import call_ai_with_prompt
# from utils.file_helpers import create_dirs, append_to_report


# ----------------------------- SETUP ----------------------------- #

# Ensure required output directories exist
def create_dirs():
    os.makedirs("outputs/text", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/CSV-JSON", exist_ok=True)

def append_to_report(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):  # ensure newline
            f.write("\n")

def main():
    create_dirs()  

    # --------------------- Step 1: Feature Engineering --------------------- #
    CaseDetection.run()
    DefenceCaseDetection.run()
    # --------------------- Step 2: Clustering --------------------- #
    Clustering.run()

    # --------------------- Step 3: VisualAnalysis --------------------- #
    VisualAnalysis.run()
    
    # --------------------- Step 5: Classification --------------------- #
    Classification.run()
    DefenceClassification.run()
    # --------------------- Step 6: Zone Analysis --------------------- #
    ZoneAnalysis.run()
    PlayerZoneAnalysis.run()
    DefenceZoneAnalysis.run()
    # --------------------- Step 7: Report Generation --------------------- #
    reportgenerator.run()

if __name__ == "__main__":
    main()