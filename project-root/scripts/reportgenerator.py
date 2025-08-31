import os
from pathlib import Path
from google import genai
import textwrap

# ======= CONFIGURATION =======
# Assuming the script is in the same directory as DefenceZoneAnalysis.py
# and the project structure is consistent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_OUTPUT = PROJECT_ROOT / "outputs"
TEXT_DIR = ROOT_OUTPUT / "text"

# Input file path
SOURCE_REPORT_PATH = TEXT_DIR / "FinalReport.txt"
# Output file path for the cleaned and restructured report
CLEANED_REPORT_PATH = TEXT_DIR / "CleanedFinalReport.txt"

# Model configuration
# The user requested '2.0 flash', but '2.5-pro' is much better for this complex task.
# Using the more capable model as seen in the user's sample code.
GEMINI_MODEL = "gemini-2.0-flash"


# ======= AI PROMPT DESIGN =======

def create_restructuring_prompt(report_content: str) -> str:
    """
    Creates a detailed, structured prompt for the Gemini API to
    summarize, conclude, and restructure the provided football analysis report.
    """
    # Using textwrap.dedent to keep the prompt clean and readable in the code
    return textwrap.dedent(f"""
        You are a professional football performance analyst finalizing an **Annual Performance Analysis Report** for a football club.

Your task is to process the raw report text provided below and prepare a clean, client-ready report that follows the fixed 9-part template. Treat the raw text as the body paragraphs of the final report — do not invent new content or add analysis beyond what is already in the file.

**STRICT INSTRUCTIONS:**

1. **NEW CONTENT:**
   - Write a new **Executive Summary** (3–5 sentences) based on the *entire* report. Highlight Bayer Leverkusen's overall attacking versatility, right-wing dominance, central defensive weakness, and poor left-side finishing.
   - Write a new **Conclusion** (3–5 sentences) that summarizes key takeaways and gives forward-looking recommendations.

2. **RESTRUCTURING:**
   - Reorganize all existing report content into the following exact 9-part structure. Use the text from the raw report as the body paragraphs for each section. Do not add new content beyond placing the existing text under the correct section.
   - Standardize all visuals and data references using `[IMAGE: path/to/file]` format.

   ---
   **1. Executive Summary**
   * (Insert the Executive Summary you just wrote)

   **2. Attacking Patterns Overview**
   * (Clusters text from raw report)
   * [IMAGE: project-root/outputs/plots/VisualAnalysis/KMeans_k3/fig_radar_selected_features_combined.png]

   **3. Portfolio Summary & Tactical Implications**
   * (Portfolio analysis text)
   * [IMAGE: project-root/outputs/CSV-JSON/VisualAnalysis/KMeans_k3/summary_by_cluster.csv]
   * [IMAGE: project-root/outputs/plots/VisualAnalysis/KMeans_k3/fig_goals_vs_high_threat.png]
   * [IMAGE: project-root/outputs/plots/VisualAnalysis/KMeans_k3/fig_attack_distribution_over_time.png]
   * [IMAGE: project-root/outputs/plots/VisualAnalysis/KMeans_k3/fig_high_threat_timing.png]

   **4. Shot Quality Insights**
   * (Use all Dependence Plot Insights content)
   * [IMAGE: project-root/outputs/plots/classification_median/explainability/extended_all/targeted/dep_shoot_angle__by__num_opponents_front.png]
   * [IMAGE: project-root/outputs/plots/classification_median/explainability/extended_all/targeted/dep_shoot_angle__by__pressure_on_shooter.png]

   **5. Zone-Level Performance**
   * (Part 1 – Zone-Level Performance text)
   * [IMAGE: project-root/outputs/plots/ZoneAnalysis/zone_shap_heatmap.png]

   **6. Wing & Player Analysis**
   * (Part 2 – Wing & Player Analysis text)
   * [IMAGE: project-root/outputs/plots/PlayerZoneAnalysis/zone_weak_player_bar.png]

   **7. Finishing Analysis**
   * (Part 3 – Finishing Analysis text)
   * [IMAGE: project-root/outputs/plots/PlayerZoneAnalysis/finishing_zone_bar.png]

   **8. Defensive Heatmap Analysis**
   * (Defensive Zone Heatmap section text)
   * [IMAGE: project-root/outputs/plots/DefenceAnalysis/zone_shap_heatmap.png]

   **9. Conclusion**
   * (Insert the Conclusion you just wrote)
   ---

3. **FORMATTING RULES:**
   - Remove all citation markers (e.g., `[cite: ...]`).
   - Remove original section headers (`## Section ...`). Only use the new numbered structure.
   - Keep existing `###` subheadings (e.g., clusters).
   - Correct spelling/formatting issues, remove stray symbols (like `*!@#$`).
   - Ensure the tone is professional, polished, and suitable for a club-level annual report.

**RAW REPORT TEXT TO PROCESS:**
---
{report_content}
---
    """)


# ======= GEMINI API INTEGRATION =======

def call_gemini_for_restructuring(prompt: str) -> str:
    """
    Calls the Gemini API with the provided prompt to get the restructured report.
    """
    try:
        # Use hardcoded API key as per new instructions (Client-based API)
        api_key = "AIzaSyD3Y6wabXe5rMidJTpmk6785rSI4K2gjvk"
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return getattr(response, "text", "")

    except Exception as e:
        print(f"[ERROR] Failed to call Gemini API: {e}")
        return ""


# ======= MAIN EXECUTION LOGIC =======

def run():
    """Main function to read, process, and write the report."""
    print(f"[INFO] Reading source report from: {SOURCE_REPORT_PATH}")
    if not SOURCE_REPORT_PATH.exists():
        print(f"[ERROR] Source file not found: {SOURCE_REPORT_PATH}. Aborting.")
        return

    try:
        # Read the original report content
        with SOURCE_REPORT_PATH.open("r", encoding="utf-8") as f:
            original_content = f.read()

        # Create the detailed prompt for the AI
        print("[INFO] Creating detailed prompt for AI...")
        prompt = create_restructuring_prompt(original_content)

        # Call the Gemini API to get the cleaned and restructured report
        print(f"[INFO] Calling Gemini API (model: {GEMINI_MODEL})... This may take a moment.")
        restructured_report = call_gemini_for_restructuring(prompt)

        if not restructured_report:
            print("[ERROR] Received an empty response from the AI. Aborting.")
            return

        # Write the new report to the output file
        print(f"[INFO] Writing cleaned report to: {CLEANED_REPORT_PATH}")
        with CLEANED_REPORT_PATH.open("w", encoding="utf-8") as f:
            f.write(restructured_report)

        print("\n[SUCCESS] Report has been successfully cleaned and restructured!")
        print(f"==> Check the output file: {CLEANED_REPORT_PATH}")

    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}")


if __name__ == "__main__":
    run()
