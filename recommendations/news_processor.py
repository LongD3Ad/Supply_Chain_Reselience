# src/recommendations/news_processor.py
import spacy
import logging
from typing import Dict, List, Tuple

# --- Load spaCy Model ---
NLP_MODEL_NAME = "en_core_web_sm"
nlp = None
try:
    nlp = spacy.load(NLP_MODEL_NAME)
    logging.info(f"spaCy model '{NLP_MODEL_NAME}' loaded successfully.")
except OSError:
    logging.warning(f"spaCy model '{NLP_MODEL_NAME}' not found. "
                    f"Run 'python -m spacy download {NLP_MODEL_NAME}' to install it. "
                    "News NER features disabled.")
except ImportError:
     logging.warning("spaCy library not installed (`pip install spacy`). News NER features disabled.")

# --- Risk Keywords (Expandable) ---
# Map canonical risk name to list of keywords
RISK_KEYWORDS = {
    "port congestion": ["port", "congestion", "backlog", "delay", "harbor", "dock", "terminal capacity", "vessel waiting"],
    "strike": ["strike", "labor dispute", "union action", "protest", "industrial action", "walkout", "picket"],
    "weather disruption": ["storm", "hurricane", "typhoon", "flood", "drought", "snow", "ice", "extreme weather", "natural disaster"],
    "geopolitical tension": ["geopolitical", "tension", "conflict", "war", "sanction", "embargo", "trade dispute", "tariff"],
    "oil price spike": ["oil price", "fuel cost", "petroleum price", "energy cost", "bunker fuel", "crude price"],
    "chip shortage": ["chip shortage", "semiconductor shortage", "microchip supply", "electronics components", "wafer production"],
    "transportation issues": ["transportation", "logistics", "freight cost", "shipping rates", "carrier capacity", "trucking", "rail"],
    "material shortage": ["material shortage", "raw material", "scarcity", "supply gap", "component shortage", "input cost"],
    "cyber attack": ["cyber attack", "ransomware", "hacking", "data breach", "it outage", "malware"]
}

# --- Sentiment Keywords ---
NEGATIVE_WORDS = ["severe", "critical", "crisis", "emergency", "disaster", "major", "significant", "serious", "extreme", "worst", "plummet", "collapse", "halt", "closure", "failure", "attack"]
POSITIVE_WORDS = ["resolve", "improve", "ease", "recover", "agreement", "solution", "growth", "stable", "strong"] # Less impactful usually

def analyze_news_summary(news_text: str) -> Tuple[List[str], Dict[str, float]]:
    """
    Analyzes news summary using spaCy NER and keyword matching for risks/severity.

    Returns:
        Tuple[List[str], Dict[str, float]]: (list_of_detected_risk_names, dict_of_risk_name_to_severity)
    """
    detected_risks = set() # Use set to avoid duplicates initially
    risk_severity = {}
    if not news_text or not nlp:
        return [], {}

    # --- Keyword Matching ---
    text_lower = news_text.lower()
    for risk_type, keywords in RISK_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            detected_risks.add(risk_type)
            # Initial severity based on keyword frequency
            severity = min(0.2 + (matches * 0.15), 1.0) # Start higher, scale faster
            risk_severity[risk_type] = severity

    # --- Sentiment Adjustment (Simple) ---
    # Count negative/positive words to adjust severity
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)

    for risk_type in detected_risks:
         current_severity = risk_severity.get(risk_type, 0.5)
         # Increase severity for negative words, decrease slightly for positive
         adjusted_severity = current_severity + (neg_count * 0.05) - (pos_count * 0.02)
         # Bound severity between 0.1 and 1.0
         risk_severity[risk_type] = round(max(0.1, min(1.0, adjusted_severity)), 2)

    # --- Optional: NER Confirmation (Example) ---
    # doc = nlp(news_text)
    # for ent in doc.ents:
    #     if ent.label_ in ["GPE", "LOC", "ORG"]: # If named locations/orgs involved
    #         # Could potentially increase severity slightly for associated risks
    #         pass # Add more sophisticated logic here later

    return sorted(list(detected_risks)), risk_severity