import logging
from typing import List, Dict, Any

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("camel_tools_test")

# Attempt to initialize CAMeL Tools Analyzer
CAMEL_ANALYZER = None
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer as CAMeLAnalyzer
    # You can specify a particular database if needed, e.g., MorphologyDB('calima-msa-r13')
    # Using the default built-in database
    db_path = MorphologyDB.builtin_db()
    CAMEL_ANALYZER = CAMeLAnalyzer(db_path)
    logger.info(f"CAMeL Tools Analyzer initialized successfully with DB: {db_path}")
except ImportError:
    logger.error("camel_tools library not found. Please install it: pip install camel-tools")
except Exception as e:
    logger.error(f"Error initializing CAMeL Tools Analyzer: {e}", exc_info=True)

def test_camel_token_analysis(token_text: str):
    """Analyzes a single token with CAMeL Tools and prints detailed output."""
    if not CAMEL_ANALYZER:
        logger.error("CAMEL_ANALYZER is not initialized. Cannot test token.")
        return

    logger.info(f"\n--- Testing token: '{token_text}' ---")
    try:
        # CAMeL Tools .analyze() on a string returns a list of lists of analyses.
        # For a single token, it's typically List[List[Analysis]], e.g., [[analysis1, analysis2,...]]
        # where each Analysis is a dict-like object.
        analyses_outer_list: List[List[Dict[str, Any]]] = CAMEL_ANALYZER.analyze(token_text)
        
        logger.debug(f"Raw output from CAMEL_ANALYZER.analyze('{token_text}'):")
        logger.debug(analyses_outer_list) # Print the raw output

        if not analyses_outer_list:
            logger.warning(f"CAMEL_ANALYZER.analyze('{token_text}') returned an empty list (no tokens found by CAMeL).")
            return

        if not isinstance(analyses_outer_list, list):
            logger.warning(f"CAMEL_ANALYZER.analyze('{token_text}') did not return a list. Type: {type(analyses_outer_list)}")
            return
            
        if len(analyses_outer_list) == 0: # Should be caught by 'if not analyses_outer_list' but good to be explicit
            logger.warning(f"CAMEL_ANALYZER.analyze('{token_text}') returned an outer list with 0 elements.")
            return

        # Assuming the first element of the outer list corresponds to the single input token
        inner_analysis_list: List[Dict[str, Any]] = analyses_outer_list[0]
        logger.debug(f"Inner list of analyses for '{token_text}':")
        logger.debug(inner_analysis_list)

        if not inner_analysis_list:
            logger.warning(f"Inner list of analyses for '{token_text}' is empty.")
            return
        
        if not isinstance(inner_analysis_list, list):
            logger.warning(f"Inner analysis structure for '{token_text}' is not a list. Type: {type(inner_analysis_list)}")
            return

        logger.info(f"Number of morphological analyses found for '{token_text}': {len(inner_analysis_list)}")
        for i, analysis_dict in enumerate(inner_analysis_list):
            logger.info(f"  Analysis {i+1} for '{token_text}': {analysis_dict}")
            # You can print specific features if you know their keys, e.g.:
            # logger.info(f"    POS: {analysis_dict.get('pos')}, Lemma: {analysis_dict.get('lex')}")

    except Exception as e:
        logger.error(f"Exception during CAMeL token-level analysis for '{token_text}':", exc_info=True)


if __name__ == "__main__":
    if CAMEL_ANALYZER:
        # List of tokens that caused issues or you want to inspect
        problematic_tokens = [
            "الولدُ",    # From your logs
            "يقرأُ",    # From your logs
            "الكتابَ",  # From your logs
            ".",         # From your logs
            "البنتُ",    # From your logs
            "تشربُ",    # From your logs
            "الحليبَ",  # From your logs
            "الطالبُ",  # From your logs
            "ك",         # Stanza split 'كتبَ' into 'ك' and 'تبَ'
            "تبَ",
            "الدرسَ",
            "هذه",      # Had "Error ...: 0"
            "مدرسة",    # Had "Error ...: 0"
            "نا",        # Had "no result"
            "السيارة",  # Stanza split 'السيارةُ'
            "ُ",         # Separate damma token from Stanza
            "سريعة",
            "ٌ",         # Separate tanween token
            "طالبات",
            "ع",
            "مجتهدات",
            "يتكم",
            "للطاولة",
            "أربعُ",
            "أرجلٍ",
            "أكلت",
            "القطة", # Dediac form of القطةُ
            "قطة",   # Dediac lemma of القطةُ
            "الطائرة", # OOV even after dediac
            "طائرة",  # Dediac form of الطائرة
            "بنى",    # OOV
            "بنية",  # Dediac form of بنى
            "القاعدة", # OOV
            "قاعدة"   # Dediac form
        ]
        
        # Also test some simple, common words
        simple_tokens = ["ولد", "كتاب", "مدرسة", "ذهب", "هو", "هي"]
        
        all_test_tokens = list(set(problematic_tokens + simple_tokens)) # Unique tokens

        for token in all_test_tokens:
            test_camel_token_analysis(token)
    else:
        logger.error("CAMEL_ANALYZER could not be initialized. Aborting tests.")


