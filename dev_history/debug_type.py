# debug_typing.py
# A comprehensive script to test the full analysis-to-diagram pipeline
# for specific, problematic sentences.

import logging
import sys
import os
import stanza
from typing import List, Dict, Tuple, Optional, Any, Set, Union 
import copy

# --- Add project directory to Python path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import all necessary components ---
try:
    # Import pipeline functions from camel_test2
    from camel_test2 import (
        #initialize_camel_analyzer_v3_1_centralized,
        analyze_arabic_sentence_with_morph,
        assign_discocat_types_v2_2,
        create_nominal_sentence_diagram_v2_7,
        create_verbal_sentence_diagram_v3_7
    )
    # Import shared types to verify results
    from common_qnlp_types import (
        LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY,
        Ty, Box, GrammarDiagram,
        ADJ_PRED_TYPE_ARABIC,
        ADJ_MOD_TYPE_ARABIC,
        VERB_TRANS_TYPE_ARABIC
    )
    logger.info("Successfully imported all necessary functions and types.")
except ImportError as e:
    logger.critical(f"Failed to import a required function or type: {e}")
    logger.critical("Please ensure camel_test2.py and common_qnlp_types.py are in the path and contain all required functions.")
    sys.exit(1)
CAMEL_ANALYZER = None
try:
    from camel_tools.morphology.database import MorphologyDB # type: ignore
    from camel_tools.morphology.analyzer import Analyzer # type: ignore
    db_path = MorphologyDB.builtin_db() # type: ignore
    CAMEL_ANALYZER = Analyzer(db_path) # type: ignore
    logger.info("camel_test2: CAMeL Tools Analyzer initialized successfully.")
except ImportError:
    logger.warning("camel_test2: CAMeL Tools not found.")
except Exception as e:
    logger.warning(f"camel_test2: Error initializing CAMeL Tools Analyzer: {e}.")

STANZA_AVAILABLE = False
nlp: Optional[stanza.Pipeline] = None 
try:
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse,mwt', verbose=False, use_gpu=False, logging_level='WARN') 
    STANZA_AVAILABLE = True
    logger.info("camel_test2: Stanza pipeline initialized successfully.")
except Exception as e:
    logger.error(f"camel_test2: Error initializing Stanza: {e}", exc_info=True)

def check_analyses_list_integrity(analyses_list: Optional[List[Any]], step_name: str):
    """Helper to check if analyses_list is a list of dictionaries."""
    if not isinstance(analyses_list, list):
        logger.error(f"CRITICAL_INTEGRITY_CHECK ({step_name}): `analyses_list` is NOT a list. Type: {type(analyses_list)}")
        return False
    if not analyses_list: # Empty list is valid but note it
        logger.info(f"INTEGRITY_CHECK ({step_name}): `analyses_list` is an empty list.")
        return True 
    
    all_dicts = True
    for i, entry in enumerate(analyses_list):
        if not isinstance(entry, dict):
            logger.error(f"CRITICAL_INTEGRITY_CHECK ({step_name}): Element at index {i} in `analyses_list` is NOT a dict. Type: {type(entry)}, Value: {str(entry)[:200]}")
            all_dicts = False
    if all_dicts:
        logger.info(f"INTEGRITY_CHECK ({step_name}): `analyses_list` (len {len(analyses_list)}) is confirmed to be a list of all dictionaries.")
    else:
        logger.error(f"INTEGRITY_CHECK ({step_name}): `analyses_list` (len {len(analyses_list)}) contains non-dictionary elements.")
    return all_dicts


def run_analysis_step(sentence: str) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], str, Dict[str, Any]]:
    logger.info("--- STEP 1: Calling `analyze_arabic_sentence_with_morph` ---")
    try:
        tokens, analyses, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug=True)
        
        logger.info(f"INFO_ANALYSIS_STEP (Post-call): `analyze_arabic_sentence_with_morph` returned `analyses` of type: {type(analyses)}")
        if isinstance(analyses, list) and analyses: # Check if it's a non-empty list
            logger.info(f"INFO_ANALYSIS_STEP (Post-call): First element of `analyses` is of type: {type(analyses[0])}")
            logger.info(f"INFO_ANALYSIS_STEP (Post-call): `analyses` length: {len(analyses)}")
        
        # Explicit integrity check
        check_analyses_list_integrity(analyses, "Post-analyze_arabic_sentence_with_morph")

        if not tokens and isinstance(analyses, list) and not analyses:
            logger.error("STEP 1 FAILED: `analyze_arabic_sentence_with_morph` returned no tokens and no analyses.")
            return None, None, "ERROR_ANALYSIS_NO_TOKENS_OR_ANALYSES", {}

        logger.info(f"  ✅  STEP 1 SUCCEEDED.")
        # logger.info(f"     >> Tokens returned: {tokens}") # Can be verbose
        logger.info(f"     >> Structure Identified: \"{structure}\"")
        key_roles = {
            'verb': roles.get('verb'), 'subject': roles.get('subject'), 'object': roles.get('object'),
            'root': roles.get('root'), 'predicate_idx': roles.get('predicate_idx')
        }
        logger.info(f"     >> Key Roles Identified: {key_roles}")
        return tokens, analyses, structure, roles
    except Exception as e:
        logger.error(f"STEP 1 FAILED: An exception occurred in `analyze_arabic_sentence_with_morph`: {e}", exc_info=True)
        return None, None, "ERROR_ANALYSIS_EXCEPTION", {}

def run_type_assignment_step(analyses_list_input: Optional[List[Dict[str, Any]]], roles_dict: Dict[str, Any], current_sentence_structure: str) -> Tuple[Optional[List[Any]], bool]:
    logger.info("\n--- STEP 2: Calling `assign_discocat_types_v2_2` for each token ---")
    
    # Make a deep copy to ensure the original analyses_list is not modified by assign_discocat_types_v2_2
    # This is a key change to test for side effects.
    analyses_list = copy.deepcopy(analyses_list_input)

    assigned_types_list = []
    all_types_assigned_flag = True
    
    if not check_analyses_list_integrity(analyses_list, "Start of run_type_assignment_step"):
        logger.error("STEP 2 SKIPPED: `analyses_list` integrity check failed at start of type assignment.")
        return None, False

    for analysis_entry in analyses_list:
        try:
            # Ensure analysis_entry is a dict before passing
            if not isinstance(analysis_entry, dict):
                logger.error(f"STEP 2 ERROR: analysis_entry is not a dict: {type(analysis_entry)}. Skipping type assignment for this entry.")
                # Decide how to handle: append None, or raise error, or skip
                assigned_types_list.append(None) # Or some error marker
                continue # Or set all_types_assigned_flag = False and break

            core_type = assign_discocat_types_v2_2(analysis=analysis_entry, roles=roles_dict, debug=True)
            assigned_types_list.append(core_type)
        except Exception as e:
            logger.error(f"STEP 2 FAILED: An exception occurred for token '{analysis_entry.get('text', 'UNKNOWN_TOKEN_IN_ERROR')}': {e}", exc_info=True)
            all_types_assigned_flag = False; break
    
    if not all_types_assigned_flag: return None, False

    logger.info(f"  ✅  STEP 2 SUCCEEDED. Types assigned.")
    print("     >> Word -> Assigned Type:")
    for i, analysis in enumerate(analyses_list): # Use the (potentially copied) analyses_list
        token_text = analysis.get('text', 'N/A')
        assigned_type = assigned_types_list[i] if i < len(assigned_types_list) else "ERROR_TYPE_MISSING"
        type_str = str(assigned_type) if assigned_type else "None"; type_class = type(assigned_type).__name__
        highlight = ""
        if analysis.get('upos') == 'ADJ' and assigned_type == ADJ_MOD_TYPE_ARABIC and current_sentence_structure.startswith("NOMINAL"):
            highlight = "  <-- NOTE: Attributive ADJ_MOD (N->N) in NOMINAL structure. Expected Predicate (N->S)?"
        elif analysis.get('upos') == 'ADJ' and assigned_type == ADJ_PRED_TYPE_ARABIC and not current_sentence_structure.startswith("NOMINAL"):
            highlight = "  <-- NOTE: Predicative ADJ_PRED (N->S) in non-NOMINAL structure."
        if analysis.get('upos') == 'X': highlight = "  <-- NOTE: Parser tagged as 'X'."
        print(f"        - '{token_text}' ({analysis.get('upos')}, {analysis.get('deprel')}) -> {type_str} ({type_class}){highlight}")
    return assigned_types_list, all_types_assigned_flag

def run_diagram_creation_step(
    tokens_list: Optional[List[str]], analyses_list: Optional[List[Dict[str, Any]]],
    assigned_types_list: Optional[List[Any]], roles_dict: Dict[str, Any], current_structure: str
) -> Optional[GrammarDiagram]:
    logger.info("\n--- STEP 3: Calling Diagram Creation Function ---")
    final_diagram_obj = None

    logger.info("DEBUG_DIAGRAM_CALL: Checking arguments before calling diagram creation function:")
    logger.info(f"  tokens_list OK: {isinstance(tokens_list, list) and bool(tokens_list)}")
    integrity_ok_for_diagram = check_analyses_list_integrity(analyses_list, "Start of run_diagram_creation_step")
    logger.info(f"  assigned_types_list OK: {isinstance(assigned_types_list, list) and bool(assigned_types_list)}")


    if not tokens_list or not integrity_ok_for_diagram or not assigned_types_list:
        logger.error("STEP 3 SKIPPED: Missing or invalid tokens, analyses, or assigned types based on integrity checks.")
        return None

    nominal_structures = ["NOMINAL_NOUN_SUBJ", "NOMINAL_NOUN_SUBJ_ADJ_PRED", "NOMINAL_ADJ_PREDICATE", 
                          "NOMINAL_ADJ_PRED_NO_SUBJ", "NOMINAL_X_PRED_WITH_SUBJ", "NOMINAL_RECLASSIFIED",
                          "NOMINAL_SUBJ_ONLY_RECLASSIFIED", "SUBJ_NO_VERB_OTHER",
                          "NOUN_PREDICATE_NO_SUBJ", "NOMINAL_ROOT_NO_EXPLICIT_SUBJ_DEP",
                          "ADJECTIVAL_PREDICATE_NO_SUBJ"]
    verbal_structures = ["VERBAL_VERB_ROOT", "SVO", "VSO", "VOS", "VERBAL_COMPLEX_ORDER", "SV", "VS",
                         "VO_NO_SUBJ", "VERB_ONLY", "VERBAL_FALLBACK_SVLIKE", "VERBAL_FALLBACK_VERB_ONLY"]
    
    original_indices = [entry['original_idx'] for entry in analyses_list] # Assumes analyses_list is now clean

    try:
        args_for_diagram_func = (tokens_list, analyses_list, roles_dict, assigned_types_list, original_indices)
        kwargs_for_diagram_func = {"debug": True}

        if current_structure.startswith("COMPLEX_"):
            if "NOMINAL" in current_structure or "ADJ_PRED" in current_structure or "NOUN_SUBJ" in current_structure:
                 logger.info(f"Attempting `create_nominal_sentence_diagram_v2_7` for COMPLEX '{current_structure}'...")
                 if "hint_predicate_original_idx" in create_nominal_sentence_diagram_v2_7.__code__.co_varnames:
                     kwargs_for_diagram_func["hint_predicate_original_idx"] = roles_dict.get("predicate_idx")
                 final_diagram_obj = create_nominal_sentence_diagram_v2_7(*args_for_diagram_func, **kwargs_for_diagram_func)
            else: 
                 logger.info(f"Attempting `create_verbal_sentence_diagram_v3_7` for COMPLEX '{current_structure}'...")
                 final_diagram_obj = create_verbal_sentence_diagram_v3_7(*args_for_diagram_func, **kwargs_for_diagram_func)
        elif current_structure in nominal_structures:
            logger.info(f"Attempting `create_nominal_sentence_diagram_v2_7` for structure '{current_structure}'...")
            if "hint_predicate_original_idx" in create_nominal_sentence_diagram_v2_7.__code__.co_varnames:
                kwargs_for_diagram_func["hint_predicate_original_idx"] = roles_dict.get("predicate_idx")
            final_diagram_obj = create_nominal_sentence_diagram_v2_7(*args_for_diagram_func, **kwargs_for_diagram_func)
        elif current_structure in verbal_structures:
            logger.info(f"Attempting `create_verbal_sentence_diagram_v3_7` for structure '{current_structure}'...")
            final_diagram_obj = create_verbal_sentence_diagram_v3_7(*args_for_diagram_func, **kwargs_for_diagram_func)
        elif current_structure.startswith("OTHER_"):
             logger.warning(f"Structure is '{current_structure}'. Attempting verbal diagram as fallback.")
             final_diagram_obj = create_verbal_sentence_diagram_v3_7(*args_for_diagram_func, **kwargs_for_diagram_func)
        else:
            logger.warning(f"No specific diagram logic for structure '{current_structure}'. Skipping.")
    except TypeError as te:
        logger.error(f"STEP 3 FAILED (TypeError) for '{current_structure}': {te}", exc_info=True)
        final_diagram_obj = None
    except Exception as e:
        logger.error(f"STEP 3 FAILED (Exception) for '{current_structure}': {e}", exc_info=True)
        final_diagram_obj = None
    return final_diagram_obj

def print_final_diagnosis(original_sentence_text: str, final_diagram_obj: Optional[GrammarDiagram], 
                          current_structure: str, analyses_list: Optional[List[Dict[str, Any]]], 
                          assigned_types_list: Optional[List[Any]], roles_dict: Dict[str, Any]):
    print("\n" + "-"*80); logger.info(f"FINAL DIAGNOSIS for \"{original_sentence_text}\":")
    if final_diagram_obj and isinstance(final_diagram_obj, GrammarDiagram):
        logger.info(f"  ✅✅✅ SUCCESS! Valid diagram created. Codomain: {final_diagram_obj.cod}. Diagram: {str(final_diagram_obj)[:200]}...")
    else:
        logger.error(f"  ❌❌❌ FAILURE! Diagram creation failed/skipped for structure '{current_structure}'.")
    print("="*80 + "\n")

if __name__ == "__main__":
    stanza_ok = False
    camel_ok = False
    try:

        if STANZA_AVAILABLE and 'nlp' in globals() and globals()['nlp'] is not None:
            logger.info("Stanza pipeline initialized successfully.")
            stanza_ok = True
        else:
            logger.error("Stanza pipeline FAILED to initialize or global 'nlp' not set.")


        if CAMEL_ANALYZER is not None:
            logger.info("CAMeL Tools analyzer initialized successfully.")
            camel_ok = True
        else:
            logger.error("CAMeL Tools analyzer FAILED to initialize.")

        if not (stanza_ok and camel_ok):
            raise Exception("One or both of Stanza/CAMeL Tools failed to initialize properly.")
    except NameError as ne:
        logger.critical(f"An initialization function was not found: {ne}. Ensure correct imports from camel_test2.py.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Initialization failed. Error: {e}", exc_info=True)
        sys.exit(1)

    sentences_to_test = [
        "الولد الذي رأيته طويل",
        "الطبيب لطيف وعطوفات",
        "بُنيَة البيت قوية"
    ]

    sentences_to_test = ["الولد الذي رأيته طويل", "الطبيب لطيف وعطوفات", "بُنيَة البيت قوية"]
    if not LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        logger.error("CRITICAL: `common_qnlp_types` Lambeq types NOT initialized.")

    for s_idx, sentence_txt in enumerate(sentences_to_test):
        logger.info(f"--- Processing Sentence {s_idx+1}/{len(sentences_to_test)}: \"{sentence_txt}\" ---")
        
        s_tokens, s_analyses, s_structure, s_roles = run_analysis_step(sentence_txt)
        
        # --- Added: Integrity check for s_analyses immediately after run_analysis_step ---
        logger.info(f"DEBUG_MAIN_LOOP: After run_analysis_step for '{sentence_txt}':")
        check_analyses_list_integrity(s_analyses, "After run_analysis_step in main")

        if s_structure.startswith("ERROR_ANALYSIS"):
            print_final_diagnosis(sentence_txt, None, s_structure, s_analyses, None, s_roles if isinstance(s_roles, dict) else {})
            continue

        s_assigned_types, types_ok = run_type_assignment_step(s_analyses, s_roles, s_structure)
        
        # --- Added: Integrity check for s_analyses after run_type_assignment_step ---
        logger.info(f"DEBUG_MAIN_LOOP: After run_type_assignment_step for '{sentence_txt}':")
        check_analyses_list_integrity(s_analyses, "After run_type_assignment_step in main")
        logger.info(f"DEBUG_MAIN_LOOP: s_assigned_types is list: {isinstance(s_assigned_types, list)}, content: {str(s_assigned_types)[:200] if s_assigned_types else 'None'}")


        if not types_ok:
            print_final_diagnosis(sentence_txt, None, s_structure, s_analyses, s_assigned_types, s_roles)
            continue
            
        s_final_diagram = run_diagram_creation_step(s_tokens, s_analyses, s_assigned_types, s_roles, s_structure)
        print_final_diagnosis(sentence_txt, s_final_diagram, s_structure, s_analyses, s_assigned_types, s_roles)

    logger.info("--- Debugging Script Finished ---")
