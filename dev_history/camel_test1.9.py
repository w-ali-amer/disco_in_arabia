# -*- coding: utf-8 -*-
import stanza
from lambeq import AtomicType, IQPAnsatz, SpiderAnsatz, StronglyEntanglingAnsatz # Added SpiderAnsatz
# *** FIX: Import normal_form explicitly if needed, or rely on diagram method ***
from lambeq import Rewriter
from lambeq.backend.grammar import Ty, Box, Cup, Id, Spider, Swap, Diagram as GrammarDiagram
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import traceback
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter # Import Parameter for type hinting
from typing import List, Dict, Tuple, Optional, Any, Set
from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram
import logging
import string
import os
import hashlib # For parameter binding hash

# --- Imports for TKET/Qiskit Conversion ---
from pytket.extensions.qiskit import tk_to_qiskit
try:
    # Use AerBackend from pytket.extensions.qiskit if available and needed
    # from pytket.extensions.qiskit import AerBackend
    PYTKET_QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: pytket-qiskit extension not found.")
    print("Please install it: pip install pytket-qiskit")
    PYTKET_QISKIT_AVAILABLE = False
# --- END Imports ---

# --- NEW: Imports for Arabic Text Display ---
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_DISPLAY_ENABLED = True
except ImportError:
    print("Warning: 'arabic_reshaper' or 'python-bidi' not found.")
    print("Arabic text in plots might not render correctly.")
    print("Install them: pip install arabic_reshaper python-bidi")
    ARABIC_DISPLAY_ENABLED = False
# --- END NEW IMPORTS ---

# Arabic diacritics (harakat, tanwin, etc):
ARABIC_DIACRITICS = set("ًٌٍَُِّْ")

# --- Configure Logging ---
# Keep your existing logging setup
logging.basicConfig(level=logging.DEBUG, # Keep DEBUG for now
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('camel_test2_arabic') # Use specific name

# --- CAMeL Tools Import ---
# Keep your existing CAMeL Tools setup
CAMEL_ANALYZER = None
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.utils.dediac import dediac_ar
    from camel_tools.morphology.analyzer import Analyzer
    db_path = MorphologyDB.builtin_db() # Use default built-in DB
    CAMEL_ANALYZER = Analyzer(db_path)
    logger.info("CAMeL Tools Analyzer initialized successfully.")
except ImportError:
    logger.warning("CAMeL Tools not found (pip install camel-tools). POS tag fallback disabled.")
except LookupError:
     logger.warning("CAMeL Tools default DB not found. Run 'camel_tools download <db_name>'. POS tag fallback disabled.")
except Exception as e:
    logger.warning(f"Error initializing CAMeL Tools Analyzer: {e}. POS tag fallback disabled.")
# --- End CAMeL Tools ---

# Initialize Stanza with Arabic models
# Keep your existing Stanza setup
try:
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse', verbose=False, download_method=None)
    STANZA_AVAILABLE = True
    logger.info("Stanza pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Stanza: {e}", exc_info=True)
    logger.error("Please ensure Stanza Arabic models are downloaded: stanza.download('ar')")
    STANZA_AVAILABLE = False

# Define DisCoCat types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE # Added for potential use
N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

# --- NEW: Helper Function for Arabic Text Display ---
def shape_arabic_text(text):
    """Reshapes and applies bidi algorithm for correct Arabic display in Matplotlib."""
    if not ARABIC_DISPLAY_ENABLED or not text or not isinstance(text, str):
        return text # Return non-strings, empty strings, or if libs not installed
    # Basic check for Arabic Unicode range
    if any('\u0600' <= char <= '\u06FF' for char in text):
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            logger.warning(f"Could not reshape/bidi Arabic text '{text[:20]}...': {e}")
            return text # Fallback to original text on error
    else:
        return text # Return text without Arabic characters as is
# --- END HELPER FUNCTION ---

# --- NEW: Configure Matplotlib Font ---
# Ensure the chosen font (e.g., 'Tahoma', 'Arial', 'Amiri') is installed on your system
try:
    plt.rcParams['font.family'] = 'sans-serif'
    # Add fonts known to support Arabic. Matplotlib will try them in order.
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Arial', 'Amiri', 'Noto Naskh Arabic']
    logger.info(f"Configured Matplotlib font.sans-serif: {plt.rcParams['font.sans-serif']}")
except Exception as e:
    logger.warning(f"Could not set Matplotlib font configuration: {e}")
# --- END FONT CONFIG ---


# ==================================
# Linguistic Analysis Function (Unchanged from promptgem.txt)
# ==================================
def analyze_arabic_sentence(sentence: str, debug: bool = True) -> Tuple[List[str], List[Dict], str, Dict]:
    """
    Analyzes an Arabic sentence using Stanza for dependency parsing and
    CAMeL Tools for morphological analysis, with improved token mismatch handling
    and TypeError fix for structure determination.
    V2.5 (More robust CAMeL token-level fallback analysis)
    """
    global nlp, STANZA_AVAILABLE, CAMEL_ANALYZER # Assuming these are global for this function

    current_logger = logging.getLogger('camel_test2_arabic') # Use a consistent logger name
    # Basic logging configuration if not already set elsewhere
    if not current_logger.hasHandlers():
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    if not STANZA_AVAILABLE or nlp is None: # Added nlp is None check
        current_logger.error("Stanza is not available or nlp model not initialized.")
        return [], [], "ERROR_STANZA_INIT", {}

    if CAMEL_ANALYZER is None:
        current_logger.warning("CAMEL_ANALYZER is not initialized. Morphological features will be missing for all tokens.")
        # Allow to proceed without CAMeL if Stanza is available

    if not sentence or not sentence.strip():
        current_logger.warning("Received empty sentence for analysis.")
        return [], [], "EMPTY_INPUT", {} # More specific error type

    try:
        doc = nlp(sentence)
    except Exception as e_nlp:
        current_logger.error(f"Stanza processing failed for sentence: '{sentence}'", exc_info=True)
        return [], [], "ERROR_STANZA_PROCESSING", {}

    tokens: List[str] = []
    analyses_data: List[Dict] = []
    # Initialize roles with a default dependency_graph
    roles: Dict[str, Any] = {"verb": None, "subject": None, "object": None, "root": None, "dependency_graph": {}, "structure": "OTHER"}


    if not doc.sentences:
        current_logger.warning(f"Stanza did not find any sentences in: '{sentence}'")
        return [], [], "NO_SENTENCES_STANZA", roles # Return initialized roles

    sent = doc.sentences[0]
    stanza_word_objects = sent.words # Store Stanza word objects

    # --- CAMeL Tools Morphological Analysis ---
    # Initialize with Nones, matching the number of Stanza tokens
    camel_morph_analyses_list: List[Dict[str, Any] | None] = [None] * len(stanza_word_objects)

    if CAMEL_ANALYZER:
        current_logger.debug(f"Attempting CAMeL Tools analysis for sentence: '{sentence}'")
        try:
            # Sentence-level analysis by CAMeL Tools
            # Returns: List[List[AnalysisDict]]
            raw_camel_analyses_sentence_level = CAMEL_ANALYZER.analyze(sentence)

            # Check if CAMeL's tokenization matches Stanza's for the current sentence
            if len(raw_camel_analyses_sentence_level) == len(stanza_word_objects):
                current_logger.debug("CAMeL Tools sentence-level analysis token count matches Stanza.")
                for i, analysis_options_for_camel_token in enumerate(raw_camel_analyses_sentence_level):
                    # analysis_options_for_camel_token is List[AnalysisDict]
                    if analysis_options_for_camel_token and isinstance(analysis_options_for_camel_token, list) and len(analysis_options_for_camel_token) > 0:
                        first_analysis = analysis_options_for_camel_token[0]
                        if isinstance(first_analysis, dict):
                            camel_morph_analyses_list[i] = first_analysis
                        else:
                            current_logger.warning(f"CAMeL sentence-level: analysis for token {i} ('{stanza_word_objects[i].text}') is not a dict: {type(first_analysis)}")
                            camel_morph_analyses_list[i] = None
                    else:
                        camel_morph_analyses_list[i] = None
                        current_logger.debug(f"CAMeL sentence-level: no morph options for Stanza token {i} ('{stanza_word_objects[i].text}') corresponding CAMeL token.")
            else:
                # Token count mismatch: Fallback to token-by-token CAMeL analysis using Stanza's tokenization
                current_logger.warning(
                    f"CAMeL sentence-level token count ({len(raw_camel_analyses_sentence_level)}) "
                    f"mismatch with Stanza ({len(stanza_word_objects)}) for sentence: '{sentence}'. "
                    f"CAMeL tokens (sample): '{[item[0].get('diac', 'N/A') if item and item[0] else 'N/A' for item in raw_camel_analyses_sentence_level[:3]]}' vs "
                    f"Stanza tokens (sample): '{[w.text for w in stanza_word_objects[:3]]}'. "
                    "Attempting token-by-token CAMeL analysis as fallback."
                )
                
                temp_camel_analyses_fallback: List[Dict[str, Any] | None] = []
                for stanza_word_obj in stanza_word_objects:
                    stanza_token_text = stanza_word_obj.text
                    try:
                        # CAMEL_ANALYZER.analyze(single_token_string) returns List[AnalysisDict]
                        camel_results_for_one_stanza_token: List[Dict[str, Any]] = CAMEL_ANALYZER.analyze(stanza_token_text)
                        
                        if (camel_results_for_one_stanza_token and 
                            isinstance(camel_results_for_one_stanza_token, list) and 
                            len(camel_results_for_one_stanza_token) > 0):
                            
                            # Successfully got a list of analyses for this Stanza token.
                            # Take the first analysis dictionary from this list.
                            first_analysis_dict = camel_results_for_one_stanza_token[0]
                            
                            if isinstance(first_analysis_dict, dict):
                                temp_camel_analyses_fallback.append(first_analysis_dict)
                            else:
                                current_logger.warning(
                                    f"CAMeL token-fallback: analysis for '{stanza_token_text}' - "
                                    f"first item in list is not a dict: {type(first_analysis_dict)}. Appending None."
                                )
                                temp_camel_analyses_fallback.append(None)
                        else:
                            current_logger.debug(
                                f"CAMeL token-fallback: no valid analyses or unexpected structure for '{stanza_token_text}': "
                                f"{camel_results_for_one_stanza_token}. Appending None."
                            )
                            temp_camel_analyses_fallback.append(None)
                            
                    except Exception as e_camel_token:
                        current_logger.error(
                            f"Exception during CAMeL token-fallback analysis for Stanza token '{stanza_token_text}'. Error: {e_camel_token}",
                            exc_info=debug # Show full traceback only if debug is True
                        )
                        temp_camel_analyses_fallback.append(None)
                
                if len(temp_camel_analyses_fallback) == len(stanza_word_objects):
                    current_logger.info("Successfully used token-by-token CAMeL analysis results for Stanza tokens.")
                    camel_morph_analyses_list = temp_camel_analyses_fallback
                else:
                    # This should not happen if the loop iterates over all stanza_word_objects
                    current_logger.error("Token-by-token CAMeL analysis fallback resulted in an unexpected list length. "
                                         "CAMeL features might be incomplete.")
                    # Pad with Nones if necessary, though ideally, the lengths should match
                    while len(temp_camel_analyses_fallback) < len(stanza_word_objects):
                        temp_camel_analyses_fallback.append(None)
                    camel_morph_analyses_list = temp_camel_analyses_fallback[:len(stanza_word_objects)]


        except Exception as e_camel_analyze_sent:
            current_logger.error(f"Error during CAMeL sentence-level analysis for '{sentence}': {e_camel_analyze_sent}", exc_info=True)
            current_logger.warning("Proceeding without CAMeL morphological features for this sentence due to error.")
            # camel_morph_analyses_list will remain all Nones
    else:
        # CAMEL_ANALYZER was None from the start
        current_logger.warning("CAMEL_ANALYZER not available. Morphological features will be missing for all tokens.")
        # camel_morph_analyses_list is already initialized to Nones

    # Populate analyses_data using Stanza's tokenization and CAMeL's morphology (if available)
    for i, word_obj in enumerate(stanza_word_objects):
        tokens.append(word_obj.text)
        head_idx = word_obj.head - 1 if word_obj.head is not None and word_obj.head > 0 else -1 # Stanza is 1-indexed
        
        current_morph_features = camel_morph_analyses_list[i] if i < len(camel_morph_analyses_list) else None
        
        # Ensure current_morph_features is a dict or None
        if current_morph_features is not None and not isinstance(current_morph_features, dict):
            current_logger.warning(
                f"Internal state error: morph features for token '{word_obj.text}' is not a dict: "
                f"{type(current_morph_features)}. Resetting to None."
            )
            current_morph_features = None

        analyses_data.append({
            "lemma": word_obj.lemma if word_obj.lemma else word_obj.text,
            "pos": word_obj.upos,
            "deprel": word_obj.deprel,
            "head": head_idx, # 0-indexed or -1 for root
            "morph": current_morph_features # This will be a dict or None
        })
        roles["dependency_graph"][i] = [] # Initialize for current token index

    # Build dependency graph from Stanza's head information
    for i, analysis_entry in enumerate(analyses_data):
        head_idx_for_dep = analysis_entry["head"]
        dep_relation = analysis_entry["deprel"]
        
        if head_idx_for_dep >= 0 and head_idx_for_dep < len(tokens):
            # Ensure the head index exists as a key in the dependency_graph
            if head_idx_for_dep not in roles["dependency_graph"]:
                 roles["dependency_graph"][head_idx_for_dep] = [] # Should have been initialized already
            roles["dependency_graph"][head_idx_for_dep].append((i, dep_relation))
        elif head_idx_for_dep != -1: # If it's not a root and not a valid index
            current_logger.warning(f"Invalid head index {head_idx_for_dep} for token {i} ('{tokens[i]}'). Skipping dependency edge.")
    
    # Determine root, verb, subject, object
    for i, analysis_entry in enumerate(analyses_data):
        if analysis_entry["deprel"] == "root" or analysis_entry["head"] == -1:
            roles["root"] = i
            break
    if roles["root"] is None and len(analyses_data) > 0:
        current_logger.warning("No 'root' dependency found, assigning first token as root as fallback.")
        roles["root"] = 0
    elif not analyses_data: # No tokens
        roles["root"] = None


    potential_verbs = [i for i, ad in enumerate(analyses_data) if ad["pos"] == "VERB"]
    num_verbs = len(potential_verbs)

    if roles["root"] is not None and roles["root"] < len(analyses_data) and analyses_data[roles["root"]]["pos"] == "VERB":
        roles["verb"] = roles["root"]
    elif potential_verbs:
        roles["verb"] = potential_verbs[0] # Default to the first verb found if root is not a verb
    else:
        roles["verb"] = None


    main_anchor_for_roles = roles.get("verb") if roles.get("verb") is not None else roles.get("root")
    subj_idx = None
    obj_idx = None

    if main_anchor_for_roles is not None and main_anchor_for_roles < len(analyses_data):
        # Check dependents of the main anchor (verb or root)
        if main_anchor_for_roles in roles["dependency_graph"]:
            for dep_idx, dep_rel in roles["dependency_graph"].get(main_anchor_for_roles, []):
                if dep_rel in ["nsubj", "csubj"] and subj_idx is None: # Prioritize nsubj
                    subj_idx = dep_idx
                if dep_rel in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and obj_idx is None: # Prioritize obj/dobj
                    obj_idx = dep_idx
                if subj_idx is not None and obj_idx is not None: # Found both
                    break
    
    roles["subject"] = subj_idx
    roles["object"] = obj_idx

    # --- Structure Determination ---
    structure = "OTHER" # Default
    verb_idx_for_structure = roles.get("verb")

    if verb_idx_for_structure is not None:
        v_pos = verb_idx_for_structure
        s_pos = roles.get("subject", -1) # Use -1 if None
        o_pos = roles.get("object", -1)  # Use -1 if None

        # Ensure positions are integers for comparison
        s_pos = s_pos if s_pos is not None else -1
        o_pos = o_pos if o_pos is not None else -1

        if s_pos != -1 and o_pos != -1: # V, S, O all present
            if v_pos < s_pos and s_pos < o_pos: structure = "VSO"
            elif s_pos < v_pos and v_pos < o_pos: structure = "SVO"
            elif s_pos < o_pos and o_pos < v_pos: structure = "SOV"
            # Add other permutations if necessary, e.g., VOS, OSV, OVS
            elif v_pos < o_pos and o_pos < s_pos: structure = "VOS"
            elif o_pos < s_pos and s_pos < v_pos: structure = "OSV"
            elif o_pos < v_pos and v_pos < s_pos: structure = "OVS"
            else: structure = "VERBAL_COMPLEX_ORDER" # All present but complex order
        elif s_pos != -1: # Only V and S
            if v_pos < s_pos: structure = "VS"
            elif s_pos < v_pos: structure = "SV"
            else: structure = "VERBAL_S_UNORDERED" # e.g. verb and subj are same token (pro-drop)
        elif o_pos != -1: # Only V and O
            if v_pos < o_pos: structure = "VO"
            elif o_pos < v_pos: structure = "OV"
            else: structure = "VERBAL_O_UNORDERED"
        else: # Only V
            structure = "VERB_ONLY"
        
        if num_verbs > 1 and not structure.startswith("COMPLEX_"):
             structure = "COMPLEX_VERBAL_" + structure
        elif num_verbs == 0 and structure != "OTHER": # Should not happen if verb_idx is not None
             current_logger.warning("Verb index present but no verbs counted. Recheck logic.")

    elif subj_idx is not None : # No verb, but subject found (nominal sentence)
        if roles["root"] is not None and roles["root"] < len(analyses_data) and \
           analyses_data[roles["root"]]["pos"] in ["NOUN", "PROPN", "ADJ", "PRON"]:
            structure = "NOMINAL"
    elif num_verbs > 1: # Multiple verbs but no clear primary verb for role assignment
        structure = "COMPLEX_MULTIVERB_OTHER"
    elif not analyses_data:
        structure = "EMPTY_ANALYSIS"


    roles["structure"] = structure

    if debug:
        current_logger.debug(f"--- Linguistic Analysis for: '{sentence}' (Stanza tokens: {' '.join(tokens)}) ---")
        for i, analysis_dict_item in enumerate(analyses_data):
            morph_dict = analysis_dict_item.get('morph')
            morph_str = str(morph_dict) if morph_dict else "N/A"
            # Limit morph string length for cleaner logs
            morph_display_limit = 100
            if len(morph_str) > morph_display_limit:
                morph_str = morph_str[:morph_display_limit] + "..."

            current_logger.debug(
                f"  Idx {i}: Token='{tokens[i]}', Lemma='{analysis_dict_item['lemma']}', "
                f"POS='{analysis_dict_item['pos']}', Dep='{analysis_dict_item['deprel']}', "
                f"Head={analysis_dict_item['head']} (token: '{tokens[analysis_dict_item['head']]}' if analysis_dict_item['head'] != -1 else 'ROOT'), " # Added head token text
                f"CAMeL Morph: {morph_str}"
            )
        current_logger.debug(f"Determined structure: {structure}")
        verb_token = tokens[roles['verb']] if roles.get('verb') is not None and 0 <= roles['verb'] < len(tokens) else 'None'
        subj_token = tokens[roles['subject']] if roles.get('subject') is not None and 0 <= roles['subject'] < len(tokens) else 'None'
        obj_token = tokens[roles['object']] if roles.get('object') is not None and 0 <= roles['object'] < len(tokens) else 'None'
        root_token_idx = roles.get('root')
        root_token = tokens[root_token_idx] if root_token_idx is not None and 0 <= root_token_idx < len(tokens) else 'None'


        current_logger.debug(f"  Verb index: {roles.get('verb')} ('{verb_token}')")
        current_logger.debug(f"  Subject index: {roles.get('subject')} ('{subj_token}')")
        current_logger.debug(f"  Object index: {roles.get('object')} ('{obj_token}')")
        current_logger.debug(f"  Root index: {root_token_idx} ('{root_token}')")
        current_logger.debug("--- End Linguistic Analysis ---")

    return tokens, analyses_data, structure, roles




# ==================================
# DisCoCat Type Assignment (REVISED - V2.1 with Standard Verb Types)
# ==================================
def assign_discocat_types_v3_morph(
    analysis_dict: Dict, # Contains lemma, pos, deprel, head, morph
    token: str, # Keep original token for logging/fallback
    is_verb: bool = False, verb_takes_subject: bool = False, verb_takes_object: bool = False,
    is_nominal_pred: bool = False,
    debug: bool = True) -> Optional[Ty]:
    """
    Assigns DisCoCat types based on POS tag, dependency relation, and role.
    V3: Accepts morphological features from CAMeL Tools (in analysis_dict['morph']).
        (Currently, morph features are only used for logging/potential future use here,
         type assignment logic remains primarily based on POS/Dep for simplicity).
    """
    n = N # Default noun type if morph info is missing
    s = S
    #v_base = V_pres # Default verb type

    # Determine specific Noun type based on morphology
    morph_features = analysis_dict.get('morph')


    # --- Define standard functional types using the determined base noun type ---
    adj_type = n @ n.l
    prep_type = n.r >> n.l
    det_type = n @ n.l
    num_type = n @ n.l
    adv_type = S @ S.l

    # Extract info from analysis_dict
    pos = analysis_dict.get('pos', 'X')
    dep_rel = analysis_dict.get('deprel', 'dep')
    lemma = analysis_dict.get('lemma', token)

    assigned_type = None
    original_pos = pos
    decision_reason = f"Initial POS='{pos}', Dep='{dep_rel}'"
    if morph_features:
        decision_reason += f" (Morph: gen={morph_features.get('gen')}, num={morph_features.get('num')}, asp={morph_features.get('asp')})"
        decision_reason += f" -> NounBase={n}, VerbBase={v_base}"

    # --- CAMeL Tools Fallback (Keep as before) ---
    if pos in ['X', 'UNK'] or pos is None:
        if CAMEL_ANALYZER:
            logger.debug(f"Attempting CAMeL Tools fallback for '{token}' (POS: {pos})")
            try:
                analyses_camel = CAMEL_ANALYZER.analyze(token)
                if analyses_camel:
                    camel_pos = analyses_camel[0].get('pos')
                    if camel_pos:
                        camel_to_upos = { # (Mapping remains the same)
                            'noun': 'NOUN', 'noun_prop': 'PROPN', 'noun_num': 'NOUN', 'noun_quant': 'NOUN',
                            'pron': 'PRON', 'pron_dem': 'PRON', 'pron_exclam': 'PRON', 'pron_interrog': 'PRON', 'pron_rel': 'PRON',
                            'verb': 'VERB', 'verb_pseudo': 'VERB',
                            'adj': 'ADJ', 'adj_comp': 'ADJ', 'adj_num': 'ADJ',
                            'adv': 'ADV', 'adv_interrog': 'ADV', 'adv_rel': 'ADV',
                            'prep': 'ADP', 'conj': 'CCONJ', 'conj_sub': 'SCONJ',
                            'part': 'PART', 'part_neg': 'PART', 'part_verb': 'PART', 'part_interrog': 'PART',
                            'punc': 'PUNCT', 'digit': 'NUM', 'latin': 'X', 'interj': 'INTJ', 'abbrev': 'X', 'symbol': 'SYM'
                        }
                        pos = camel_to_upos.get(camel_pos, 'X')
                        decision_reason += f" -> CAMeL POS='{camel_pos}' mapped to '{pos}'"
                        logger.info(f"CAMeL Tools suggested POS '{camel_pos}', mapped to '{pos}' for token '{token}'.")
                    else: logger.warning(f"CAMeL Tools analysis for '{token}' did not provide POS tag.")
                else: logger.warning(f"CAMeL Tools returned no analysis for '{token}'.")
            except Exception as e_camel: logger.error(f"Error during CAMeL Tools fallback for token '{token}': {e_camel}")
        else: logger.warning(f"Stanza POS tag is '{pos}' for token '{token}', and CAMeL Tools fallback is unavailable.")

    # --- Type Assignment Logic (Dependency Prioritized) ---

    # 1. Core Grammatical Roles (High Priority)
    if dep_rel in ["nsubj", "csubj"]: # Subject
        assigned_type = n
        decision_reason += " -> Dep=Subject => N"
    elif dep_rel in ["obj", "iobj", "dobj"]: # Object
        assigned_type = n
        decision_reason += " -> Dep=Object => N"
    elif dep_rel == "xcomp" and pos == "VERB": # Clausal complement (treat as S for now)
        assigned_type = s
        decision_reason += " -> Dep=xcomp/VERB => S"
    elif dep_rel == "ccomp" and pos == "VERB": # Clausal complement (treat as S)
        assigned_type = s
        decision_reason += " -> Dep=ccomp/VERB => S"

    # 2. Verb Type (If not assigned above) - *** CRITICAL FIX ***
    elif is_verb:
        domain_list = []
        # Standard DisCoCat: Subject from right (N.r), Object from left (N.l)
        if verb_takes_subject: domain_list.append(N.r)
        if verb_takes_object: domain_list.append(N.l)

        if not domain_list:
            domain = Ty() # Intransitive verb with no explicit arguments? Or imperative?
        elif len(domain_list) == 1:
            domain = domain_list[0] # e.g., N.r for intransitive taking subject
        else:
            # Order matters for tensor product: N.r @ N.l is standard for Subj, Obj
            if N.r in domain_list and N.l in domain_list:
                 domain = N.r @ N.l # Standard transitive
            else:
                 # Fallback if only one is present (already handled) or unexpected combo
                 domain = Ty.tensor(*domain_list) # General tensor product

        assigned_type = domain >> S
        decision_reason += f" -> is_verb=True => {domain} >> S (Standard Type)"

    # 3. Modifiers (High Priority)
    elif dep_rel == "amod": # Adjectival Modifier
        assigned_type = adj_type
        decision_reason += f" -> Dep=amod => {adj_type}"
    elif dep_rel == "advmod": # Adverbial Modifier
        assigned_type = adv_type
        decision_reason += f" -> Dep=advmod => {adv_type}"
    elif dep_rel == "nmod": # Nominal Modifier (often PP or possessive)
        if pos == "ADP":
            assigned_type = prep_type
            decision_reason += f" -> Dep=nmod/ADP => {prep_type}"
        else: # Assume it modifies a noun like an adjective
            assigned_type = adj_type
            decision_reason += f" -> Dep=nmod/NotADP => {adj_type} (assuming N modifier)"

    # 4. Prepositions (if not caught by nmod)
    elif pos == "ADP":
        assigned_type = prep_type
        decision_reason += f" -> POS=ADP => {prep_type}"

    # 5. Determiners
    elif pos == "DET":
        assigned_type = n @ n.l # Determiner modifies Noun
        decision_reason += f" -> POS=DET => {N @ N.l}"

    # 6. Nominal Predicate (Specific case)
    elif is_nominal_pred:
        # A nominal predicate (like an adjective in "The house is big")
        # takes a noun (subject) and produces a sentence.
        assigned_type = N.l >> S # Predicate expects Noun on left, outputs Sentence
        decision_reason += f" -> is_nominal_pred=True => {N.l >> S}"

    # 7. Other POS tags (Lower Priority - use POS if no strong dependency signal)
    elif assigned_type is None: # Only if not assigned by dependency yet
        if pos in ["NOUN", "PROPN", "PRON"]:
            assigned_type = n
            decision_reason += f" -> POS={pos} (fallback) => N"
        elif pos == "ADJ": # Adjective not caught by amod/nmod
            assigned_type = adj_type
            decision_reason += f" -> POS=ADJ (fallback) => {adj_type}"
        elif pos == "ADV": # Adverb not caught by advmod
            assigned_type = adv_type
            decision_reason += f" -> POS=ADV (fallback) => {adv_type}"
        elif pos == "NUM":
            assigned_type = N @ N.l # Number modifies Noun
            decision_reason += f" -> POS=NUM => {N @ N.l}"
        elif pos == "AUX":
             assigned_type = S @ S.l # Aux modifies Sentence
             decision_reason += f" -> POS=AUX => {S @ S.l}"
        elif pos == "CCONJ": # Coordinating Conjunction (e.g., 'and', 'or')
             # Connects two elements of the same type, often S
             assigned_type = S.r @ S @ S.l # Connects two sentences
             decision_reason += f" -> POS=CCONJ => {S.r @ S @ S.l}"
        elif pos == "SCONJ": # Subordinating Conjunction (e.g., 'because', 'if')
             assigned_type = S.r >> (S.l @ S) # Takes S (right), modifies S (left) - complex, simplify?
             # Simpler: S.r >> S ? Or S >> S ? Let's try S @ S.l for now
             assigned_type = S @ S.l
             decision_reason += f" -> POS=SCONJ => {assigned_type}"
        elif pos == "PART": # Particles (negation, future, etc.)
             assigned_type = S @ S.l # Simple sentence modifier
             decision_reason += f" -> POS=PART => {S @ S.l}"
        elif pos == "INTJ": # Interjection
             assigned_type = S # Standalone sentence
             decision_reason += f" -> POS=INTJ => S"
        elif pos in ["PUNCT", "SYM"]:
             assigned_type = None # Remove punctuation and symbols
             decision_reason += f" -> POS={pos} => Remove (None)"
        else: # Final fallback for truly unknown POS/Dep combinations
             logger.warning(f"Unhandled POS/Dep combination: POS='{pos}' (Orig: '{original_pos}'), Dep='{dep_rel}' for '{token}'. Assigning default N.")
             assigned_type = n
             decision_reason += f" -> Unhandled => Default N"

    if debug:
        log_level = logging.DEBUG if assigned_type is not None else logging.WARNING
        logger.log(log_level, f"Type Assignment: Token='{token}', Lemma='{lemma}', POS='{pos}'(Orig:'{original_pos}'), Dep='{dep_rel}', is_verb={is_verb}, is_nom_pred={is_nominal_pred}")
        logger.log(log_level, f"  Decision Path: {decision_reason}")
        logger.log(log_level, f"  Assigned Type: {assigned_type}")

    return assigned_type

def create_feature_boxes(morph_features: Optional[Dict]) -> List[Box]:
    """
    Creates simple identity boxes based on morphological features (v2).
    Adds more specific prefix/suffix features if present in analysis.
    Requires morphological analyzer (e.g., CAMeL Tools) to provide detailed affix info.
    """
    feature_boxes = []
    if not morph_features:
        return feature_boxes

    # --- Prefixes (Proclitics) ---
    # Definite Article (already present)
    if morph_features.get('prc1') == 'det' or morph_features.get('prc2') == 'det' or morph_features.get('prc0') == 'det':
        feature_boxes.append(Box("DEF_ART", Ty(), Ty())) # More specific name

    # Prepositions as Prefixes (already present)
    prep_prefixes = {'b': 'PREF_PREP_Bi', 'l': 'PREF_PREP_Li', 'k': 'PREF_PREP_Ka'}
    for key in ['prc0', 'prc1', 'prc2']: # Check multiple prefix slots
        lex = morph_features.get(f'{key}_lex')
        pos = morph_features.get(key)
        if pos == 'prep' and lex in prep_prefixes:
             feature_boxes.append(Box(prep_prefixes[lex], Ty(), Ty()))

    # Conjunctions as Prefixes
    conj_prefixes = {'w': 'PREF_CONJ_Wa', 'f': 'PREF_CONJ_Fa'}
    for key in ['prc0', 'prc1', 'prc2']:
        lex = morph_features.get(f'{key}_lex')
        pos = morph_features.get(key)
        if pos == 'conj' and lex in conj_prefixes:
             feature_boxes.append(Box(conj_prefixes[lex], Ty(), Ty()))

    # Future Markers as Prefixes
    future_prefixes = {'sa': 'PREF_FUT_Sa'} # Add 'sawfa' if analyzer distinguishes it
    for key in ['prc0', 'prc1']: # Usually prc1 or prc0 for 'sa'
        lex = morph_features.get(f'{key}_lex')
        pos = morph_features.get(key)
        # Check if the part-of-speech tag indicates a future particle
        if pos == 'fut_part' and lex in future_prefixes:
             feature_boxes.append(Box(future_prefixes[lex], Ty(), Ty()))


    # --- Core Features (Gender, Number, Aspect - already present) ---
    if morph_features.get('gen') == 'f':
         if morph_features.get('pos') in ['noun', 'adj', 'verb', 'verb_pseudo']:
              feature_boxes.append(Box("FEM", Ty(), Ty()))
    # else: # Optionally add Masculine box
    #     if morph_features.get('pos') in ['noun', 'adj', 'verb', 'verb_pseudo']:
    #          feature_boxes.append(Box("MASC", Ty(), Ty()))

    if morph_features.get('num') == 'p':
         feature_boxes.append(Box("PL", Ty(), Ty()))
    elif morph_features.get('num') == 'd':
         feature_boxes.append(Box("DU", Ty(), Ty()))
    # else: # Optionally add Singular box
    #      feature_boxes.append(Box("SG", Ty(), Ty()))

    if morph_features.get('asp') == 'p':
        feature_boxes.append(Box("PAST", Ty(), Ty()))
    elif morph_features.get('asp') == 'i':
        feature_boxes.append(Box("PRES", Ty(), Ty())) # Present/Imperfective
    elif morph_features.get('asp') == 'c':
        feature_boxes.append(Box("IMP", Ty(), Ty())) # Imperative

    # --- Suffixes (Enclitics & Inflectional) ---
    # Case (Example: Nominative, Accusative, Genitive)
    case_map = {'n': 'CASE_Nom', 'a': 'CASE_Acc', 'g': 'CASE_Gen'}
    case = morph_features.get('cas')
    if case in case_map:
        feature_boxes.append(Box(case_map[case], Ty(), Ty()))

    # State (Example: Definite, Indefinite, Construct)
    state_map = {'d': 'STATE_Def', 'i': 'STATE_Indef', 'c': 'STATE_Constr'}
    state = morph_features.get('stt')
    if state in state_map:
         # Avoid adding STATE_Def if DEF_ART is already added? Or keep both? Keep both for now.
         feature_boxes.append(Box(state_map[state], Ty(), Ty()))

    # Mood (Example: Indicative, Subjunctive, Jussive) - Often for verbs
    mood_map = {'i': 'MOOD_Ind', 's': 'MOOD_Subj', 'j': 'MOOD_Juss'}
    mood = morph_features.get('mod')
    if mood in mood_map:
        feature_boxes.append(Box(mood_map[mood], Ty(), Ty()))

    # Voice (Example: Active, Passive)
    voice_map = {'a': 'VOICE_Act', 'p': 'VOICE_Pass'}
    voice = morph_features.get('vox')
    if voice in voice_map:
        feature_boxes.append(Box(voice_map[voice], Ty(), Ty()))

    # Pronominal Suffixes (Enclitics) - Requires 'enc0' field from CAMeL
    enc0_type = morph_features.get('enc0')
    if enc0_type:
        # Example mapping - check CAMeL Tools tags for exact values
        pron_suffix_map = {
            '1sm_pron': 'ENC_Pron_1sg', '1pm_pron': 'ENC_Pron_1pl', # 1st person sg/pl
            '2sm_pron': 'ENC_Pron_2msg', '2sf_pron': 'ENC_Pron_2fsg', # 2nd person m/f sg
            '2pm_pron': 'ENC_Pron_2mpl', '2pf_pron': 'ENC_Pron_2fpl', # 2nd person m/f pl
            '3sm_pron': 'ENC_Pron_3msg', '3sf_pron': 'ENC_Pron_3fsg', # 3rd person m/f sg
            '3pm_pron': 'ENC_Pron_3mpl', '3pf_pron': 'ENC_Pron_3fpl', # 3rd person m/f pl
            # Add dual pronouns if needed ('2dm_pron', '3dm_pron', etc.)
        }
        if enc0_type in pron_suffix_map:
            feature_boxes.append(Box(pron_suffix_map[enc0_type], Ty(), Ty()))
        elif 'pron' in enc0_type: # Catch-all for other pronouns
             feature_boxes.append(Box("ENC_Pron_Other", Ty(), Ty()))


    # --- Final Cleanup: Remove Duplicates ---
    unique_feature_boxes = []
    seen_names = set()
    for box in feature_boxes:
        # Check name uniqueness AND ensure it's not empty/None
        if box.name and box.name not in seen_names:
            unique_feature_boxes.append(box)
            seen_names.add(box.name)
        elif not box.name:
             logger.warning("Generated a feature box with no name, skipping.")

    if debug and unique_feature_boxes: # Log only if debug is enabled
         logger.debug(f"  Generated Feature Boxes: {[b.name for b in unique_feature_boxes]}")

    return unique_feature_boxes

# ==================================
# Diagram Creation Functions (REVISED - V3.1 Simplified Composition)
# ==================================

# --- Nominal Diagram Creation (Keep v2 from promptgem.txt, it seems reasonable) ---
def create_nominal_sentence_diagram_v3_morph(tokens: List[str], analyses: List[Dict], roles: Dict, word_types: List[Ty], original_indices: List[int], debug: bool = True) -> Optional[GrammarDiagram]:
    """
    Create DisCoCat diagram for nominal sentences (Subject + Predicate).
    V2: More robust check for N and N.l >> S predicate type.
    (Code remains the same as provided in promptgem.txt)
    """
    logger.info("Attempting to create diagram for NOMINAL sentence (v2)...")
    if not word_types or len(tokens) != len(word_types) or len(tokens) != len(original_indices):
        logger.error("Mismatch/empty lists in create_nominal_v2.")
        return None
    # Allow None types initially, they will be filtered
    # if not all(isinstance(wt, Ty) for wt in word_types if wt is not None):
    #     logger.error(f"Invalid item in word_types: {word_types}")
    #     return None

    # Create boxes only for non-None types
    word_boxes = []
    box_map_nominal = {} # Map original_index -> (Box, list_index)
    current_box_index = 0
    for i, orig_idx in enumerate(original_indices):
        wt = word_types[i]
        if wt is not None:
            dom_type, cod_type = (wt.dom, wt.cod) if hasattr(wt, 'dom') and hasattr(wt, 'cod') else (Ty(), wt)
            # *** NEW: Shape box name if it contains Arabic ***
            box_name = shape_arabic_text(tokens[i])
            box = Box(box_name, dom_type, cod_type)
            word_boxes.append(box)
            box_map_nominal[orig_idx] = (box, current_box_index)
            current_box_index += 1
            morph_features = analyses[i].get('morph') # Get morph dict for this token
            feature_boxes = create_feature_boxes(morph_features)
            if feature_boxes:
                logger.debug(f"  Adding feature boxes for '{tokens[i]}': {[fb.name for fb in feature_boxes]}")
                for fb in feature_boxes:
                    current_diagram @= fb # Add feature boxes tensor product
        else:
            logger.debug(f"Skipping box creation for token '{tokens[i]}' (orig_idx {orig_idx}) due to None type.")

    if not word_boxes:
        logger.error("No valid word boxes created for nominal sentence.")
        return None

    logger.debug(f"Created {len(word_boxes)} nominal word boxes.")


  
    # Composition Logic: Try adjacent cup first, fallback to tensor
    composition_diagram = Id(Ty())
    initial_types = Ty()
    for box in word_boxes:
        composition_diagram @= box
        initial_types @= box.cod

    logger.debug(f"Initial nominal tensor diagram cod: {composition_diagram.cod}")

    # Attempt Subject-Predicate Cup
    subj_orig_idx = roles.get("subject", roles.get("root")) # Subject often root in nominal
    predicate_orig_idx = -1

    # Find the predicate (often adjective or noun modifying subject)
    if subj_orig_idx is not None:
        dep_graph = roles.get("dependency_graph", {})
        potential_preds = []
        if subj_orig_idx in dep_graph:
            potential_preds.extend([dep_idx for dep_idx, dep_rel in dep_graph[subj_orig_idx] if dep_rel in ["amod", "acl", "nmod", "appos", "cop"]]) # Common predicate relations
        # Also check if root is different and modifies subject
        root_idx = roles.get("root")
        if root_idx is not None and root_idx != subj_orig_idx and root_idx in dep_graph:
             for dep_idx, dep_rel in dep_graph[root_idx]:
                 if dep_idx == subj_orig_idx: # If root points to subject
                      potential_preds.append(root_idx)
                      break

        # Find the first potential predicate that has the N.l >> S type
        for p_idx in potential_preds:
            if p_idx in box_map_nominal:
                pred_box, _ = box_map_nominal[p_idx]
                if pred_box.cod == (N.l >> S): # Check the output type
                    predicate_orig_idx = p_idx
                    logger.info(f"Found potential predicate '{pred_box.name}' (orig_idx {p_idx}) with type {pred_box.cod}")
                    break

    cup_applied = False
    if subj_orig_idx in box_map_nominal and predicate_orig_idx in box_map_nominal:
        subj_box, subj_list_idx = box_map_nominal[subj_orig_idx]
        pred_box, pred_list_idx = box_map_nominal[predicate_orig_idx]

        # Check types directly: Subject is N, Predicate is N.l >> S
        if subj_box.cod == N and pred_box.dom == N.l and pred_box.cod == S:
            logger.info(f"Attempting nominal Cup: Subject='{subj_box.name}' (N), Predicate='{pred_box.name}' (N.l >> S)")
            # --- Simplified Composition: Assume subj and pred are the main components ---
            try:
                # Try direct composition if types match
                # Order matters: Subject must feed into the N.l input of the predicate
                if subj_list_idx < pred_list_idx:
                    # Diagram: Subj @ Pred >> Cup(N, N.l) @ Id(S) -> This is wrong.
                    # Correct composition: Subj @ Pred >> Id(N) @ Swap(N.l, S) >> Cup(N, N.l) @ Id(S) -> Complex swaps
                    # Let Lambeq handle it: Subj >> Pred
                    # Ensure the predicate box has domain N.l and codomain S
                    temp_diag = subj_box >> pred_box # This assumes subj_box outputs N and pred_box takes N.l
                    # This composition (subj >> pred) only works if types match exactly or via adjoints.
                    # For N and N.l >> S, it requires a cup. Let's try normal_form.
                    temp_diag = subj_box @ pred_box
                    logger.debug(f"Nominal composition before normal_form: {temp_diag}")
                    final_diagram = temp_diag.normal_form()
                    logger.info(f"Applied normal_form. Result cod: {final_diagram.cod}")
                    cup_applied = True # Assume normal_form applied the cup
                else: # Predicate comes first: Pred @ Subj
                     temp_diag = pred_box @ subj_box
                     logger.debug(f"Nominal composition before normal_form: {temp_diag}")
                     final_diagram = temp_diag.normal_form()
                     logger.info(f"Applied normal_form. Result cod: {final_diagram.cod}")
                     cup_applied = True

            except Exception as e_cup_nom:
                logger.warning(f"Nominal composition/normal_form failed: {e_cup_nom}. Falling back.")
                final_diagram = composition_diagram # Fallback to initial tensor
                cup_applied = False
        else:
            logger.warning(f"Type mismatch for nominal composition: Subj={subj_box.cod}, Pred={pred_box.dom} >> {pred_box.cod}")

    # Fallback if normal_form wasn't applied or failed
    if not cup_applied:
        logger.warning("Nominal composition/normal_form condition not met or failed. Using tensor product and forcing.")
        final_diagram = composition_diagram # The initial tensor product

    # Force to S if needed
    if final_diagram.cod != S:
        logger.warning(f"Final nominal diagram cod is {final_diagram.cod}. Forcing to S.")
        try:
            if final_diagram.cod != Ty():
                if all(t == S for t in final_diagram.cod.objects):
                    final_diagram >>= Spider(S, len(final_diagram.cod.objects), 1)
                else:
                    # *** NEW: Shape box name if it contains Arabic ***
                    box_name = shape_arabic_text("Force_S_NominalFallback")
                    final_diagram >>= Box(box_name, final_diagram.cod, S)
            else:
                final_diagram = Id(S) # Handle empty codomain case
        except Exception as e_force:
            logger.error(f"Could not force nominal diagram to S: {e_force}.", exc_info=True)
            return composition_diagram if not cup_applied else None # Return intermediate

    logger.info(f"Final nominal diagram created with cod: {final_diagram.cod}")
    return final_diagram


# --- Verbal Diagram Creation (REVISED - V3.1 Simplified) ---
def create_verbal_sentence_diagram_v3_morph( # Renamed
    tokens: List[str], analyses: List[Dict], roles: Dict, word_types: List[Ty],
    original_indices: List[int], svo_order: bool = False, debug: bool = False,
    output_dir: Optional[str] = None, sentence_prefix: str = "diag"
    ) -> Optional[GrammarDiagram]:
    """
    Creates DisCoCat diagrams using standard Lambeq composition. V3.2
    Adds simple identity boxes based on morphological features next to word boxes.
    """
    logger.info(f"Creating verbal diagram (v3.2 - Morph Boxes) for: {' '.join(tokens)}")
    logger.debug(f"Word Types (filtered): {word_types}")
    logger.debug(f"Original Indices: {original_indices}")
    logger.debug(f"Analyses (first few morph): {[{k: v for k, v in (a.get('morph') or {}).items() if k in ['gen', 'num', 'prc1']} for a in analyses[:3]]}")


    # --- 1. Input Validation ---
    if not word_types or len(tokens) != len(word_types) or len(tokens) != len(original_indices) or len(tokens) != len(analyses):
        logger.error("Mismatch between tokens, types, indices, or analyses in create_verbal_v3.2.")
        return None

    # --- 2. Create Boxes (Word boxes + Feature boxes) ---
    current_diagram = Id(Ty())
    box_map: Dict[int, Tuple[Box, int]] = {}
    current_box_idx = 0
    original_indices_in_diagram = []

    for i, orig_idx in enumerate(original_indices):
        wt = word_types[i]
        if wt is None: continue
        if not isinstance(wt, Ty): continue

        dom_type, cod_type = (wt.dom, wt.cod) if hasattr(wt, 'dom') and hasattr(wt, 'cod') else (Ty(), wt)
        box_name = shape_arabic_text(tokens[i])
        word_box = Box(box_name, dom_type, cod_type)
        current_diagram = current_diagram @ word_box
        box_map[orig_idx] = (word_box, current_box_idx)
        original_indices_in_diagram.append(orig_idx)
        current_box_idx += 1

        # --- Add ENHANCED Feature Boxes for this word ---
        morph_features = analyses[i].get('morph')
        feature_boxes = create_feature_boxes(morph_features) # Use v2 function
        if feature_boxes:
             # Log only if debug is True (handled inside create_feature_boxes_v2)
             for fb in feature_boxes:
                 current_diagram = current_diagram @ fb
        # --- End Feature Box Addition ---

    logger.debug(f"Initial Tensor (Words + Features v2) Cod: {current_diagram.cod}")
    # ... (Optional visualization of initial diagram) ...

    # --- Attempt Standard Composition with normal_form ---
    try:
        logger.info("Attempting composition using diagram.normal_form()...")
        simplified_diagram = current_diagram.normal_form()
        logger.info(f"Diagram after normal_form(): Cod={simplified_diagram.cod}")
        # ... (Optional visualization of simplified diagram) ...
        final_diagram = simplified_diagram
    except Exception as e_nf:
        logger.error(f"Error during normal_form composition: {e_nf}", exc_info=True)
        logger.warning("Falling back to initial tensor diagram.")
        final_diagram = current_diagram

    # --- Final Reduction to Sentence Type S ---
    if final_diagram.cod != S:
        logger.warning(f"Final verbal diagram codomain is {final_diagram.cod}, expected {S}. Attempting to force.")
        try:
            if final_diagram.cod != Ty():
                if all(t == S for t in final_diagram.cod.objects):
                    final_diagram >>= Spider(S, len(final_diagram.cod.objects), 1)
                    logger.info("Applied final Spider to merge S wires.")
                else:
                    box_name = shape_arabic_text("Force_S_Verbal")
                    final_diagram >>= Box(box_name, final_diagram.cod, S)
                    logger.info(f"Applied final Box '{box_name}' to force to S.")
            else:
                 final_diagram = Id(S)
                 logger.info("Diagram simplified to empty type, creating Id(S).")
        except Exception as e_force:
            logger.error(f"Could not force verbal diagram to S type: {e_force}. Returning intermediate.", exc_info=True)
            return final_diagram

    logger.info(f"Verbal diagram composition (v5 + enhanced fbox) finished. Final Cod: {final_diagram.cod}")
    # ... (Optional visualization of final diagram) ...
    return final_diagram

# Remove find_subwire_index_v2 and apply_cup_at_indices_v2 as they are not needed
# for the simplified standard composition approach.

# ==================================
# Main Conversion Function (REVISED - V2.1 using fixed types/diagrams)
# ==================================
def arabic_to_quantum_enhanced_v3_morph( # Renamed to v6
    sentence: str,
    ansatz_choice: str = 'IQP',
    # --- Ansatz parameters (keep as before) ---
    n_layers_ent: int = 2, entanglement_pattern: str = 'linear',
    n_layers_iqp: int = 3, n_single_qubit_params_iqp: int = 3,
    cnot_ranges: Optional[List[int]] = None,
    # --- Other parameters ---
    debug: bool = True,
    output_dir: Optional[str] = None
    ) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Dict], Dict]:
    """
    Processes Arabic sentence, creates DisCoCat diagram (v4 custom types),
    adds ENHANCED feature boxes (v2), and converts using SPECIFIED ansatz. V6.
    """
    # 1. Analyze sentence (Use analyze_arabic_sentence)
    tokens, analyses, structure, roles = [], [], "ERROR", {}
    try:
        logger.info(f"Analyzing sentence: '{sentence}'")
        tokens, analyses, structure, roles = analyze_arabic_sentence(sentence, debug)
        if structure == "ERROR" or not tokens:
             logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
             if not isinstance(roles, dict): roles = {}
             return None, None, structure, tokens, [], roles
        logger.info(f"Analysis complete. Detected structure: {structure}")
        if not isinstance(roles, dict): roles = {}
    except Exception as e_analyze_main:
         logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
         return None, None, "ERROR", tokens, [], roles

    # 2. Assign Types (Use assign_discocat_types_v4_custom)
    word_types_list = []
    original_indices_for_types = []
    filtered_tokens_for_diagram = []
    filtered_analyses_for_diagram = []
    for i, analysis_dict in enumerate(analyses):
        token = tokens[i]
        ARABIC_DIACRITICS = set("ًٌٍَُِّْ")
        if all(ch in ARABIC_DIACRITICS for ch in token) or token in string.punctuation:
            continue
        is_verb = (i == roles.get('verb'))
        is_pred = False
        if structure == "NOMINAL":
             subj_idx_orig = roles.get("subject", roles.get("root"))
             if analysis_dict['head'] == subj_idx_orig and analysis_dict['pos'] in ["ADJ", "NOUN"]: is_pred = True
             elif i == roles.get("root") and analysis_dict['pos'] in ["ADJ", "NOUN"]: is_pred = True
        verb_takes_subj = (roles.get('subject') is not None)
        verb_takes_obj = (roles.get('object') is not None)

        assigned_type = assign_discocat_types_v3_morph(
            analysis_dict, token,
            is_verb=is_verb, verb_takes_subject=verb_takes_subj, verb_takes_object=verb_takes_obj,
            is_nominal_pred=is_pred, debug=debug
        )
        word_types_list.append(assigned_type)
        original_indices_for_types.append(i)
        filtered_tokens_for_diagram.append(token)
        filtered_analyses_for_diagram.append(analysis_dict)
    if not filtered_tokens_for_diagram:
         logger.error(f"No valid tokens remained after type assignment for: '{sentence}'")
         return None, None, structure, tokens, analyses, roles

    # 3. Create Diagram (Use diagram functions v3_morph)
    diagram = None
    sentence_prefix = "sentence_" + shape_arabic_text(sentence.split()[0]) if sentence else "empty"
    try:
        logger.info("Creating DisCoCat diagram (v5 types + enhanced feature boxes)...")
        if structure == "NOMINAL":
             diagram = create_nominal_sentence_diagram_v3_morph( # Use v5 nominal
                 filtered_tokens_for_diagram, filtered_analyses_for_diagram, roles, word_types_list, original_indices_for_types, debug
             )
             if diagram is None:
                 logger.warning("Nominal diagram creation failed, attempting verbal fallback.")
                 diagram = create_verbal_sentence_diagram_v3_morph( # Use v5 verbal
                     filtered_tokens_for_diagram, filtered_analyses_for_diagram, roles, word_types_list, original_indices_for_types,
                     svo_order=(structure=="SVO"), debug=debug, output_dir=output_dir, sentence_prefix=sentence_prefix
                 )
        elif structure != "ERROR":
             diagram = create_verbal_sentence_diagram_v3_morph( # Use v5 verbal
                 filtered_tokens_for_diagram, filtered_analyses_for_diagram, roles, word_types_list, original_indices_for_types,
                 svo_order=(structure=="SVO"), debug=debug, output_dir=output_dir, sentence_prefix=sentence_prefix
             )

        if diagram is None:
            raise ValueError("Diagram creation function returned None.")
        logger.info(f"Diagram created successfully. Final Cod: {diagram.cod}")

    except Exception as e_diagram:
        logger.error(f"Exception during diagram creation phase: {e_diagram}", exc_info=True)
        return None, None, structure, tokens, analyses, roles

    # 4. Convert to Circuit (Logic remains same as v5, uses the diagram from step 3)
    circuit = None
    try:
        logger.info(f"Converting diagram to quantum circuit using {ansatz_choice} ansatz.")
        ob_map = { S: 1, N: 1 }
        logger.debug(f"Object map (ob_map) for ansatz: {ob_map}")

        if ansatz_choice.upper() == 'IQP':
             ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
        elif ansatz_choice.upper() == 'STRONGLY_ENTANGLING':
             ansatz = StronglyEntanglingAnsatz(ob_map=ob_map, n_layers=n_layers_ent, n_single_qubit_params=n_single_qubit_params_iqp, ranges=cnot_ranges, discard=False)
        elif ansatz_choice.upper() == 'SPIDER':
             ansatz = SpiderAnsatz(ob_map=ob_map)
        else:
             logger.warning(f"Unknown ansatz_choice '{ansatz_choice}'. Defaulting to IQP.")
             ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)

        if ansatz_choice.upper() == 'SPIDER' and diagram:
            try:
                debug_diag_path = os.path.join(output_dir or ".", f"{sentence_prefix}_diag_before_spider.png")
                logger.info(f"Saving diagram before SpiderAnsatz to: {debug_diag_path}")
                diagram.draw(output='png', path=debug_diag_path, figsize=(15,10), fontsize=12) # Use draw method directly if available
                # If diagram.draw doesn't work like that, use matplotlib:
                # fig = diagram.draw(figsize=(15,10), fontsize=12)
                # if fig:
                #     fig.savefig(debug_diag_path, bbox_inches='tight', dpi=150)
                #     plt.close(fig)
            except Exception as e_draw_debug:
                logger.error(f"Could not save debug diagram: {e_draw_debug}")

        quantum_diagram = ansatz(diagram)
        tket_circ = quantum_diagram.to_tk()
        circuit = tk_to_qiskit(tket_circ)
        logger.info("Circuit conversion successful.")

    # (Keep existing error handling for circuit conversion)
    except AttributeError as e_attr:
         logger.error(f"AttributeError during circuit conversion: {e_attr}.", exc_info=False)
         return None, diagram, structure, tokens, analyses, roles
    except TypeError as e_type:
         logger.error(f"TypeError during circuit conversion: {e_type}.", exc_info=True)
         return None, diagram, structure, tokens, analyses, roles
    except Exception as e_circuit_outer:
        logger.error(f"Unexpected exception during circuit conversion: {e_circuit_outer}", exc_info=True)
        return None, diagram, structure, tokens, analyses, roles


    logger.debug(f"Function returning successfully. Circuit type: {type(circuit)}")
    # Return the original full analyses list containing morph features
    return circuit, diagram, structure, tokens, analyses, roles

# ==================================
# Visualization Functions (MODIFIED for Arabic Display)
# ==================================
def visualize_dependency_tree(tokens, analyses, roles, save_path=None):
    """ Visualize the dependency tree using networkx and matplotlib. """
    if not tokens or not analyses: return None
    G = nx.DiGraph()
    if not isinstance(roles, dict): roles = {}

    # *** NEW: Prepare shaped labels for nodes ***
    node_labels = {}
    for i, token in enumerate(tokens):
        pos_tag = analyses[i][1] if i < len(analyses) else "UNK"
        # Shape the token and POS tag for display
        label_text = shape_arabic_text(f"{i}:{token}\n({pos_tag})")
        node_labels[i] = label_text

        node_color = "lightblue"
        if i == roles.get("root"): node_color = "red"
        elif i == roles.get("verb"): node_color = "green"
        elif i == roles.get("subject"): node_color = "orange"
        elif i == roles.get("object"): node_color = "purple"
        G.add_node(i, color=node_color) # Keep original index as node ID

    edge_labels = {}
    for i, (_, _, dep, head) in enumerate(analyses):
        if head >= 0 and head < len(tokens):
            G.add_edge(head, i)
            # Dependency relations are usually technical terms, may not need shaping
            edge_labels[(head, i)] = dep

    plt.figure(figsize=(max(10, len(tokens)*0.8), 6))
    try: pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except: pos = nx.spring_layout(G, seed=42)

    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    # labels = nx.get_node_attributes(G, 'label') # Use the prepared shaped labels

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True, arrowsize=20)
    # *** NEW: Draw using the shaped labels ***
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    # *** NEW: Shape the title ***
    plt.title(shape_arabic_text("شجرة تحليل الاعتمادية"))
    plt.axis('off')
    fig = plt.gcf()
    if save_path:
        try: plt.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved dependency tree to {save_path}")
        except Exception as e_save: logger.error(f"Failed to save dependency tree: {e_save}")
    plt.close(fig)
    return fig

def visualize_diagram(diagram, save_path=None):
    """ Visualize the DisCoCat diagram. V2: Checks draw() output. """
    if diagram is None or not hasattr(diagram, 'draw'):
        logger.warning("Diagram is None or has no draw method. Skipping visualization.")
        return None
    fig = None # Initialize fig to None
    try:
        # Lambeq's draw handles box labels internally (already shaped during creation)
        # Set title_size if needed, and potentially adjust figsize
        ax = diagram.draw(figsize=(max(10, len(diagram.boxes)*1.5), 6), title_size=10)

        # *** FIX: Check if ax is valid before proceeding ***
        if ax is not None and hasattr(ax, 'figure'):
            fig = ax.figure # Get the figure from the axes

            # *** NEW: Shape the title if one exists ***
            if ax.get_title():
                 ax.set_title(shape_arabic_text(ax.get_title()), fontsize=12) # Adjust fontsize as needed

            if save_path:
                try:
                    fig.savefig(save_path, bbox_inches='tight', dpi=150)
                    logger.info(f"Saved diagram to {save_path}")
                except Exception as e_save:
                    logger.error(f"Failed to save diagram: {e_save}")
            # Always close the figure associated with ax
            plt.close(fig)
            return fig # Return the figure object if needed
        else:
            logger.warning(f"diagram.draw() returned None or invalid Axes for diagram: {diagram}. Skipping visualization.")
            # Ensure any potentially created plot is closed
            plt.close()
            return None

    except Exception as e:
        logger.error(f"Could not visualize diagram: {e}", exc_info=True)
        if fig: plt.close(fig) # Attempt to close if fig was assigned
        else: plt.close() # Close the current figure context otherwise
        return None

def visualize_circuit(circuit, save_path=None):
    """ Visualize the quantum circuit. """
    if circuit is None or not hasattr(circuit, 'draw'): return None
    try:
        # Qiskit's draw might not support Arabic well in standard modes.
        # We rely on the filename being potentially shaped.
        # The internal labels are usually technical.
        fig = circuit.draw(output='mpl', fold=-1)
        if fig:
             fig.set_size_inches(max(10, circuit.depth()), max(6, circuit.num_qubits * 0.5))
             # *** NEW: Add a shaped title if possible ***
             # circuit.name might not always be set or meaningful
             circuit_name = getattr(circuit, 'name', 'Quantum Circuit')
             fig.suptitle(shape_arabic_text(f"الدارة الكمومية: {circuit_name}"), fontsize=12)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

             if save_path:
                 try: fig.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved circuit to {save_path}")
                 except Exception as e_save: logger.error(f"Failed to save circuit: {e_save}")
             plt.close(fig)
             return fig
        else:
             logger.warning("Circuit draw did not return a figure object.")
             return None
    except Exception as e:
        logger.error(f"Could not visualize circuit: {e}", exc_info=True)
        try: plt.close()
        except: pass
        return None


# ==================================
# Main Execution / Testing (Optional - Uses Fixed Functions)
# ==================================
if __name__ == "__main__":
    logger.info("Running camel_test2.py (Arabic Wrapped) directly for testing...")

    test_sentences = [
        "يقرأ الولد الكتاب",       # VSO
        "الولد يقرأ الكتاب",       # SVO
        "البيت كبير",             # NOMINAL
        "الطالبة تدرس العلوم",     # SVO (Feminine)
        "كتبت الطالبة الدرس",     # VSO (Past, Feminine)
        # "الدرس مكتوب",           # NOMINAL (Passive Participle) - Might still be tricky
        "الرجل الذي رأيته طبيب", # COMPLEX (Relative Clause) - Likely OTHER/COMPLEX
        "ذهب الولد الى المدرسة"   # VSO with PP
    ]

    test_output_dir = "camel_test2_arabic_wrapped_output" # New output dir
    os.makedirs(test_output_dir, exist_ok=True)
    logger.info(f"Test output will be saved to: {test_output_dir}")

    all_results = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n--- Testing Sentence {i+1}: '{sentence}' ---")
        # *** NEW: Shape sentence prefix if it contains Arabic ***
        sentence_prefix = f"test_{i+1}_{shape_arabic_text(sentence.split()[0])}"

        try:
            # *** Use the FIXED main function ***
            result_tuple = arabic_to_quantum_enhanced_v2_fixed(sentence, debug=True, output_dir=test_output_dir)
            circuit, diagram, structure, tokens, analyses, roles = result_tuple

            all_results.append({
                "sentence": sentence, "circuit": circuit, "diagram": diagram,
                "structure": structure, "tokens": tokens, "analyses": analyses, "roles": roles
            })
            logger.info(f"--- Result for Sentence {i+1}: Structure='{structure}', Circuit Type='{type(circuit)}', Diagram Type='{type(diagram)}' ---")

            # Visualizations are now handled inside arabic_to_quantum_enhanced_v2_fixed

        except Exception as e_test:
            logger.error(f"!!! ERROR during test for sentence: '{sentence}' !!!", exc_info=True)
            all_results.append({"sentence": sentence, "error": str(e_test)})

    logger.info("\n--- Test Summary ---")
    for i, res in enumerate(all_results):
        logger.info(f"Sentence {i+1}: '{res['sentence']}'")
        if "error" in res:
            logger.info(f"  Status: ERROR ({res['error']})")
        else:
            logger.info(f"  Structure: {res['structure']}")
            logger.info(f"  Circuit OK: {isinstance(res.get('circuit'), QuantumCircuit)}")
            logger.info(f"  Diagram OK: {isinstance(res.get('diagram'), GrammarDiagram)}")
            if res.get('diagram') is not None: logger.info(f"  Diagram Cod: {res['diagram'].cod}")

    logger.info("Script finished.")
