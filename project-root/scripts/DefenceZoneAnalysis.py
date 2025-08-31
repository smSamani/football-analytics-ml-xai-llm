

from pathlib import Path
import os

# ======= CONFIG (mirrors PlayerZoneAnalysis structure) =======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_OUTPUT = PROJECT_ROOT / "outputs"
OUT_PLOTS_DIR = ROOT_OUTPUT / "plots" / "DefenceAnalysis"
OUT_TEXT_DIR = ROOT_OUTPUT / "text"

# Input (image only for defence)
DEFENCE_HEATMAP_IMG = OUT_PLOTS_DIR / "zone_shap_heatmap.png"

# Output report target (append)
FINAL_REPORT_PATH = OUT_TEXT_DIR / "FinalReport.txt"

# ======= PROMPT (as provided) =======
DEFENCE_PROMPT = (
    "You are a football performance analyst.\n"
    "You are given a SHAP zone impact heatmap (image attached).\n\n"
    "Your task:\n"
    "\t•\tRead the heatmap and identify which zones are red (weakness) and which are blue (strength).\n"
    "\t•\tInterpret the meaning in simple football language, as if explaining to a head coach.\n"
    "\t•\tDo not mention SHAP or technical terms. Focus only on what the team allowed or prevented in defence.\n"
    "\t•\tWrite exactly one short paragraph (4–6 sentences).\n\n"
    "Tone:\n"
    "Clear, direct, and coach-friendly. Avoid data-science jargon.\n"
)

# ============= GEMINI AI INTEGRATION (same pattern as PlayerZoneAnalysis) =============

def call_gemini_for_defence(prompt_text: str, image_path: str) -> str:
    """Mirror Clustering/PlayerZoneAnalysis style: include image path textually in a single full_prompt."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyD3Y6wabXe5rMidJTpmk6785rSI4K2gjvk")

    image_section = f"[IMAGE PATH]\n{image_path}\n"

    full_prompt = (
        f"{prompt_text}\n\n"
        f"=== HEATMAP IMAGE ===\n{image_section}"
    )

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=full_prompt,
    )
    text = getattr(resp, "text", None)
    if text is None:
        print("[GEMINI][WARN] Empty response.text in call_gemini_for_defence; returning empty string.")
        text = ""
    return text


def append_to_report(text: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(text)


def generate_defence_analysis_report() -> None:
    # Validate image exists
    if not DEFENCE_HEATMAP_IMG.exists():
        print(f"[WARN] Missing defence heatmap image: {DEFENCE_HEATMAP_IMG}")
        return

    try:
        ai_text = call_gemini_for_defence(
            prompt_text=DEFENCE_PROMPT,
            image_path=str(DEFENCE_HEATMAP_IMG),
        )
        section_header = "\n## Section – Defensive Zone Heatmap (Season-Level)\n\n"
        append_to_report(section_header + (ai_text or "") + "\n", FINAL_REPORT_PATH)
        print(f"[INFO] Defence analysis appended to {FINAL_REPORT_PATH}")
    except Exception as e:
        print(f"[WARN] Defence analysis skipped: {e}")



def run():
    """Entry point wrapper for main.py compatibility."""
    # Ensure output dirs
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    # Run just the AI call and append
    generate_defence_analysis_report()


if __name__ == "__main__":
    run()