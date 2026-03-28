# -*- coding: utf-8 -*-
# camel_test2.py (Revised Version V3.3)
# Handles NP-internal modifiers (DET, ADJ, NP-PPs) first using build_noun_phrase_diagram.

import stanza
from lambeq import AtomicType, IQPAnsatz, SpiderAnsatz # Added SpiderAnsatz
from lambeq.backend.grammar import Ty, Box, Cup, Id, Spider, Swap, Diagram as GrammarDiagram
from lambeq.backend.grammar import Diagram as ActualDiagramFromImport
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import traceback
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter # Import Parameter for type hinting
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram
import logging
import string
import os
import hashlib # For parameter binding hash

# --- Imports for TKET/Qiskit Conversion ---
try:
    from pytket.extensions.qiskit import tk_to_qiskit
    PYTKET_QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: pytket-qiskit extension not found.")
    print("Please install it: pip install pytket-qiskit")
    PYTKET_QISKIT_AVAILABLE = False

ARABIC_DIACRITICS = set("ًٌٍَُِّْ")
# Ensure logger is configured in the main script (exp4.py)
# If running this file directly, uncomment the line below for basic logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use the name of the current module

CAMEL_ANALYZER = None
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    db_path = MorphologyDB.builtin_db() # Find default
    CAMEL_ANALYZER = Analyzer(db_path)
    logger.info("CAMeL Tools Analyzer initialized successfully.")
except ImportError:
    logger.warning("CAMeL Tools not found. Morphological feature extraction will be limited.")
except Exception as e:
    logger.warning(f"Error initializing CAMeL Tools Analyzer: {e}. Morphological feature extraction will be limited.")

STANZA_AVAILABLE = False
try:
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse,mwt', verbose=False, use_gpu=False, logging_level='WARN') # Added mwt, reduced verbosity
    STANZA_AVAILABLE = True
    logger.info("Stanza pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Stanza: {e}", exc_info=True)
    STANZA_AVAILABLE = False

# Define Atomic Types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

logger.info(f"AtomicType N: {str(N)}, Type: {type(N)}")
logger.info(f"AtomicType S: {str(S)}, Type: {type(S)}")

# --- Pre-defined Functorial Types (Boxes) ---
ADJ_MOD_TYPE: Optional[Box] = None      # Adjective modifying a Noun (N -> N)
ADJ_PRED_TYPE: Optional[Box] = None     # Adjective acting as a Predicate (N -> S)
DET_TYPE: Optional[Box] = None          # Determiner modifying a Noun (N -> N)
PREP_FUNCTOR_TYPE: Optional[Box] = None # Preposition taking Noun object (N -> N), forms PP sense
VERB_INTRANS_TYPE: Optional[Box] = None # Intransitive Verb (N -> S)
VERB_TRANS_TYPE: Optional[Box] = None   # Transitive Verb (N @ N -> S)
S_MOD_BY_N: Optional[Box] = None        # Functor for attaching a PP (represented by N) to S (S @ N -> S)
N_MOD_BY_N: Optional[Box] = None        # Functor for attaching a PP (represented by N) to N (N @ N -> N)
ADV_FUNCTOR_TYPE: Optional[Box] = None  # Adverb modifying S (S -> S)

logger.info("Attempting to define global functorial types...")
try:
    ADJ_MOD_TYPE = Box("AdjModFunctor", N, N)
    ADJ_PRED_TYPE = Box("AdjPredFunctor", N, S)
    DET_TYPE = Box("DetFunctor", N, N)
    PREP_FUNCTOR_TYPE = Box("PrepFunctor", N, N)
    VERB_INTRANS_TYPE = Box("VerbIntransFunctor", N, S)
    VERB_TRANS_TYPE = Box("VerbTransFunctor", N @ N, S)
    S_MOD_BY_N = Box("S_mod_by_N", S @ N, S)
    N_MOD_BY_N = Box("N_mod_by_N", N @ N, N)
    ADV_FUNCTOR_TYPE = Box("AdvFunctor", S, S)
    logger.info(">>> Global Box definitions for functorial types completed. <<<")
except Exception as e_box_def:
    logger.critical(f"CRITICAL ERROR defining global Box types: {e_box_def}", exc_info=True)
    ADJ_MOD_TYPE = ADJ_PRED_TYPE = DET_TYPE = PREP_FUNCTOR_TYPE = None
    VERB_INTRANS_TYPE = VERB_TRANS_TYPE = S_MOD_BY_N = N_MOD_BY_N = ADV_FUNCTOR_TYPE = None

if any(t is None for t in [ADJ_MOD_TYPE, DET_TYPE, PREP_FUNCTOR_TYPE, N_MOD_BY_N, S_MOD_BY_N]):
     logger.error("One or more essential modifier Box types failed to initialize.")

# ==================================
# Linguistic Analysis Functions (Unchanged from V3.2)
# ==================================
def parse_feats_string(feats_str: Optional[str]) -> Dict[str, str]:
    """Parses a Stanza-style features string (e.g., "Case=Nom|Gender=Masc") into a dict."""
    if not feats_str: return {}
    feats_dict = {}
    try:
        pairs = feats_str.split('|')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                feats_dict[key] = value
    except Exception as e:
        logger.warning(f"Could not parse features string '{feats_str}': {e}")
    return feats_dict

def analyze_arabic_sentence_with_morph(sentence: str, debug: bool = False) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """Analyzes an Arabic sentence using Stanza and CAMeL Tools. (Unchanged from V3.2)"""
    if not STANZA_AVAILABLE:
        logger.error("Stanza is not available or failed to initialize.")
        return [], [], "ERROR", {}
    if not sentence or not sentence.strip():
        logger.warning("Received empty sentence for analysis.")
        return [], [], "OTHER", {}
    try: doc = nlp(sentence)
    except Exception as e_nlp:
        logger.error(f"Stanza processing failed for sentence: '{sentence}'", exc_info=True)
        return [], [], "ERROR", {}

    processed_tokens_texts = []
    processed_analyses = []
    roles_dict = {"verb": None, "subject": None, "object": None, "root": None, "dependency_graph": {}, "structure": "OTHER"}
    if not doc.sentences:
        logger.warning(f"Stanza did not find any sentences in: '{sentence}'")
        return [], [], "OTHER", roles_dict
    sent = doc.sentences[0]

    # --- First Pass: Extract info & build dependency graph ---
    for i, word in enumerate(sent.words):
        token_text, lemma, upos, deprel = word.text, word.lemma, word.upos, word.deprel
        lemma = lemma if lemma else token_text
        head_idx = word.head - 1 if word.head > 0 else -1
        stanza_feats_dict = parse_feats_string(word.feats)
        camel_feats_dict = {}
        if CAMEL_ANALYZER:
            try:
                camel_analysis_list = CAMEL_ANALYZER.analyze(token_text)
                if camel_analysis_list:
                    camel_feats_str = camel_analysis_list[0].get('feat')
                    camel_feats_dict = parse_feats_string(camel_feats_str)
            except Exception as e_camel_enrich:
                logger.warning(f"Error during CAMeL enrichment for '{token_text}': {e_camel_enrich}")
        combined_feats = stanza_feats_dict.copy()
        for k, v in camel_feats_dict.items():
            if k not in combined_feats: combined_feats[k] = v
        analysis_entry = {"text": token_text, "lemma": lemma, "upos": upos, "deprel": deprel, "head": head_idx, "feats_dict": combined_feats, "original_idx": i}
        processed_tokens_texts.append(token_text)
        processed_analyses.append(analysis_entry)
        if head_idx >= 0:
             if head_idx not in roles_dict["dependency_graph"]: roles_dict["dependency_graph"][head_idx] = []
             roles_dict["dependency_graph"][head_idx].append((i, deprel))
        elif head_idx == -1 and roles_dict["root"] is None: roles_dict["root"] = i
    if roles_dict["root"] is None and len(processed_analyses) > 0: roles_dict["root"] = 0

    # --- Second Pass: Identify Roles ---
    root_idx = roles_dict["root"]
    if root_idx is not None and processed_analyses[root_idx]["upos"] == "VERB": roles_dict["verb"] = root_idx
    dependents_of_root = roles_dict["dependency_graph"].get(root_idx, [])
    for dep_idx, dep_rel_str in dependents_of_root:
        if dep_idx >= len(processed_analyses): continue
        if dep_rel_str in ["nsubj", "csubj", "nsubj:pass"] and roles_dict["subject"] is None: roles_dict["subject"] = dep_idx
        elif dep_rel_str in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and roles_dict["object"] is None: roles_dict["object"] = dep_idx
    if roles_dict["verb"] is None:
        potential_verbs = [(i, entry) for i, entry in enumerate(processed_analyses) if entry["upos"] == "VERB"]
        if potential_verbs:
            roles_dict["verb"] = potential_verbs[0][0]
            verb_idx = roles_dict["verb"]
            dependents_of_verb = roles_dict["dependency_graph"].get(verb_idx, [])
            for dep_idx, dep_rel_str in dependents_of_verb:
                 if dep_idx >= len(processed_analyses): continue
                 if dep_rel_str in ["nsubj", "csubj", "nsubj:pass"] and roles_dict["subject"] is None: roles_dict["subject"] = dep_idx
                 elif dep_rel_str in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and roles_dict["object"] is None: roles_dict["object"] = dep_idx
    if roles_dict["verb"] is not None and roles_dict["subject"] is None:
        verb_original_idx = roles_dict["verb"]
        if verb_original_idx + 1 < len(processed_analyses):
            potential_subj_entry = processed_analyses[verb_original_idx + 1]
            if potential_subj_entry["upos"] in ["NOUN", "PROPN", "PRON"] and potential_subj_entry["deprel"] != "obj":
                roles_dict["subject"] = verb_original_idx + 1; logger.debug("VSO Heuristic used.")

    # --- Determine Structure Type ---
    verb_idx, subj_idx, obj_idx = roles_dict["verb"], roles_dict["subject"], roles_dict["object"]
    num_verbs = len([i for i, entry in enumerate(processed_analyses) if entry["upos"] == "VERB"])
    structure_type = "OTHER"
    if verb_idx is not None:
        if subj_idx is not None:
            if obj_idx is not None:
                if subj_idx < verb_idx and verb_idx < obj_idx: structure_type = "SVO"
                elif verb_idx < subj_idx and subj_idx < obj_idx : structure_type = "VSO"
                elif verb_idx < obj_idx and obj_idx < subj_idx : structure_type = "VOS"
                else: structure_type = "VERBAL_COMPLEX_ORDER"
            else:
                if subj_idx < verb_idx: structure_type = "SV"
                elif verb_idx < subj_idx: structure_type = "VS"
                else: structure_type = "VERBAL_UNORDERED_SUBJ"
        elif obj_idx is not None: structure_type = "VO_NO_SUBJ"
        else: structure_type = "VERBAL_ONLY"
    elif subj_idx is not None:
        is_nominal_candidate = False
        if root_idx is not None and processed_analyses[root_idx]["upos"] in ["NOUN", "PROPN", "ADJ", "PRON"]:
                 if root_idx != subj_idx: is_nominal_candidate = True
                 elif root_idx == subj_idx:
                      root_dependents = roles_dict["dependency_graph"].get(root_idx, [])
                      for dep_idx, dep_rel_str in root_dependents:
                           if dep_idx < len(processed_analyses) and processed_analyses[dep_idx]["upos"] in ["ADJ", "NOUN", "PROPN"]:
                                is_nominal_candidate = True; break
        if is_nominal_candidate: structure_type = "NOMINAL"
        else: structure_type = "SUBJ_NO_VERB_OTHER"
    if num_verbs > 1 and not structure_type.startswith("COMPLEX_"): structure_type = "COMPLEX_" + structure_type
    roles_dict["structure"] = structure_type

    if debug: logger.debug(f"\n--- Final Analysis ---\n Tokens: {processed_tokens_texts}\n Roles: {roles_dict}\n Structure: {structure_type}\n----------------------")
    return processed_tokens_texts, processed_analyses, structure_type, roles_dict


# ==================================
# DisCoCat Type Assignment (V2.1 - Unchanged from V3.2)
# ==================================
def assign_discocat_types_v2_2( # Function name kept, but using V2.2.5 logic
    analysis: Dict[str, Any],
    roles: Dict[str, Any],
    debug: bool = True
) -> Union[Ty, GrammarDiagram, None]:
    """
    Assigns core DisCoCat types.
    V2.2.5: Aggressive predicate typing for root X/ADJ in OTHER/Nominal structures.
            Updated known verb lemmas.
    """
    token_text = analysis['text']; lemma = analysis['lemma']; pos = analysis['upos']
    dep_rel = analysis['deprel']; original_idx = analysis['original_idx']
    head_idx = analysis['head']

    is_verb_token = (original_idx == roles.get('verb'))
    is_subj_token = (original_idx == roles.get('subject'))
    is_obj_token = (original_idx == roles.get('object'))
    sentence_has_object = (roles.get('object') is not None)
    sentence_structure = roles.get('structure', 'OTHER')
    sentence_root_idx = roles.get('root')
    analyses_details = roles.get('analyses_details_for_context') # Passed from main function

    logger.debug(f"Assigning type V2.2.5 for '{token_text}' ...")

    assigned_entity: Union[Ty, GrammarDiagram, None] = None

    # --- 0. Explicit Verb Lemma Check (Updated List) ---
    known_verb_lemmas = {
    "بَنَى", "رَأَى", "اِنكَسَر", "بُنيَة", "انكسرتْ",
    "ذهب", "جاء", "قال", "كتب", "قرأ", "شرح", "درس", "عمل", "لعب", "سافر",
    "أصاب", "وجد", "عالج", "وضع", "لمس", "احتاج", "تدفق", "عَيَّن", "أَلمَأ",
    "تَحَدَّث", "أَعطَى", "اِبتَسَم", "أَدرَس", "دَرَّس", "تَقَرَّأ", "تَشَرَّح",
    "رَأيَة" # Add this if Stanza lemmatizes "رأيتُ" (verb) to "رَأيَة"
    }
    if lemma in known_verb_lemmas:
         # If roles['verb'] is None but we found a lemma match, update roles['verb']
         if roles.get('verb') is None:
             logger.warning(f"  Lemma Override: Found verb lemma '{lemma}' for token {original_idx} but no verb role. Setting roles['verb'] = {original_idx}")
             roles['verb'] = original_idx # This might help the diagram creation
             # Attempt to re-evaluate is_verb_token for subsequent logic if needed, or just proceed
             is_verb_token = True # Assume it's the verb now for role-based assignment

         assigned_entity = VERB_TRANS_TYPE if sentence_has_object else VERB_INTRANS_TYPE
         logger.debug(f"  Decision (Lemma Override): Assigned {'Transitive' if sentence_has_object else 'Intransitive'} Verb Type based on lemma '{lemma}'.")


    # --- 1. Role-Based Assignment (If not assigned by lemma) ---
    if assigned_entity is None:
        if is_verb_token: # This uses the potentially updated roles['verb']
            strongly_transitive_lemmas = {"بَنَى", "أَعطَى", "كَتَب", "قَرَأ", "شَرَح", "فَحَص", "أَكَّل", "لَمَس", "عَالَج"} # Add more
            strongly_intransitive_lemmas = {"جَاء", "ذَهَب", "جَلَس", "سَافَر", "نَامَ", "اِنكَسَر"} # Add more

            if lemma in strongly_transitive_lemmas:
                if VERB_TRANS_TYPE is not None:
                    assigned_entity = VERB_TRANS_TYPE
                    logger.debug(f"  Decision (Lemma Heuristic): Assigned Transitive Verb Type for lemma '{lemma}'.")
                else:
                    logger.error(f"VERB_TRANS_TYPE is None. Cannot assign for transitive lemma '{lemma}'. Defaulting to N.")
                    assigned_entity = N
            elif lemma in strongly_intransitive_lemmas:
                if VERB_INTRANS_TYPE is not None:
                    assigned_entity = VERB_INTRANS_TYPE
                    logger.debug(f"  Decision (Lemma Heuristic): Assigned Intransitive Verb Type for lemma '{lemma}'.")
                else:
                    logger.error(f"VERB_INTRANS_TYPE is None. Cannot assign for intransitive lemma '{lemma}'. Defaulting to N.")
                    assigned_entity = N
            else: # Fallback to original logic if lemma not in heuristic lists
                # Ensure the global types are not None before assigning
                is_adj_xcomp_obj = False
                if sentence_has_object:
                    obj_original_idx = roles.get('object')
                    # Find the analysis for the object token
                    obj_token_analysis = None
                    analyses_context = roles.get('analyses_details_for_context', [])
                    for an_detail in analyses_context:
                        if an_detail['original_idx'] == obj_original_idx:
                            obj_token_analysis = an_detail
                            break
                    
                    if obj_token_analysis and obj_token_analysis['upos'] == 'ADJ' and obj_token_analysis['deprel'] == 'xcomp':
                        is_adj_xcomp_obj = True
                        logger.debug(f"  Verb typing: Identified object '{obj_token_analysis['text']}' is an ADJ xcomp. Treating verb as intransitive for type assignment.")

                if sentence_has_object and not is_adj_xcomp_obj:
                    assigned_entity = VERB_TRANS_TYPE
                    # ... assign VERB_TRANS_TYPE ...
                else:
                    # ... assign VERB_INTRANS_TYPE ...
                    assigned_entity = VERB_INTRANS_TYPE

    # --- 2. Nominal Predicate Identification (More Aggressive for OTHER/X/ADJ roots) ---
    if pos == "DET" and dep_rel == 'nsubj' and assigned_entity is None:
        logger.debug(f"  Decision (DET as Subj): Assigning N type to DET '{token_text}' (deprel='nsubj').")
        assigned_entity = N # Treat demonstrative subject as a simple noun argument

    # --- 2. Nominal Predicate Identification (More Aggressive for OTHER/X/ADJ roots) ---
    if assigned_entity is None and \
        ( (sentence_structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"] and \
            original_idx == sentence_root_idx and \
            pos in ["ADJ", "X", "NOUN", "PROPN", "NUM"]) or \
            # More aggressive for OTHER: if a subject is found, and this token is its dependent (or root)
            (sentence_structure == "OTHER" and \
            ( (roles.get('subject') is not None and analysis.get('head') == roles.get('subject')) or \
                (original_idx == sentence_root_idx and roles.get('subject') is not None) # Predicate is root, subject is its dep
            ) and \
            pos in ["ADJ", "NOUN", "PROPN", "NUM", "X"] and \
            dep_rel in ['amod', 'nmod', 'xcomp', 'advcl', 'acl', 'appos', 'root', 'obj']) # Added obj for cases like "عينُ الطفلِ زرقاءُ" where زرقاءُ is obj of عينُ by Stanza
        ):
        current_subject_idx = roles.get('subject')
        # Try to find a subject if one isn't clearly set, especially for OTHER structures where predicate might be root
        if sentence_structure == "OTHER" and current_subject_idx is None and original_idx == sentence_root_idx:
            dependents_of_this_token = roles.get('dependency_graph', {}).get(original_idx, [])
            for dep_idx, d_rel in dependents_of_this_token:
                if d_rel == 'nsubj' and analyses_details and dep_idx < len(analyses_details) and analyses_details[dep_idx]['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                    logger.warning(f"  Predicate V2.2.6: In 'OTHER', root '{token_text}' is potential predicate. Found nsubj '{analyses_details[dep_idx]['text']}' (idx {dep_idx}). Setting roles['subject'].")
                    roles['subject'] = dep_idx
                    current_subject_idx = dep_idx
                    break

        if current_subject_idx is not None or sentence_structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"]:
            if pos == "ADJ":
                if ADJ_PRED_TYPE is not None: assigned_entity = ADJ_PRED_TYPE
                else: logger.error("ADJ_PRED_TYPE is None"); assigned_entity = N
            elif pos in ["NOUN", "PROPN", "NUM", "X"]:
                assigned_entity = Box(f"NounPred_{lemma}_{original_idx}", N, S)
            if assigned_entity: logger.debug(f"  Decision (Predicate V2.2.6): Assigned Predicate Functor to '{token_text}'.")
        else:
            logger.debug(f"  Predicate Check V2.2.6: {pos} '{token_text}' in '{sentence_structure}' but no subject identified for nominal construction.") 



    # --- 3. Dependency-Based Assignment (If still unassigned) ---
    # (Keep this section as per your V2.2 logic that yielded 83 sentences)
    if assigned_entity is None:
        if dep_rel in ['nsubj', 'obj', 'iobj', 'dobj', 'nsubj:pass', 'csubj', 'obl', 'obl:arg', 'nmod', 'appos', 'parataxis'] and pos in ["NOUN", "PROPN", "PRON", "X", "NUM"]:
             if not (dep_rel.startswith('obl') and pos == 'ADP'):
                 assigned_entity = N; logger.debug(f"  Decision (Dep): Noun Type assigned (DepRel '{dep_rel}' for {pos}).")
        elif dep_rel == 'amod' and pos in ['ADJ', 'X']:
            assigned_entity = ADJ_MOD_TYPE; logger.debug(f"  Decision (Dep): Adjective Modifier Type assigned (amod for {pos}).")
        elif dep_rel == 'det' and pos in ['DET', 'PRON', 'X']:
              assigned_entity = DET_TYPE; logger.debug(f"  Decision (Dep): Determiner Type assigned (det for {pos}).")
        elif dep_rel == 'case' and pos in ['ADP', 'PART']:
              assigned_entity = PREP_FUNCTOR_TYPE; logger.debug(f"  Decision (Dep): Preposition Functor Type assigned (case for {pos}).")
        elif dep_rel == 'advmod':
              if pos in ['ADV', 'ADJ', 'PART', 'X']:
                  assigned_entity = ADV_FUNCTOR_TYPE; logger.debug(f"  Decision (Dep): Adverb Functor Type assigned (advmod for {pos}).")

    # --- 4. POS-Based Assignment (Fallback) ---
    # (Keep this section as per your V2.2 logic)
    if assigned_entity is None:
        if pos in ["NOUN", "PROPN", "PRON", "NUM", "X"]:
            assigned_entity = N; logger.debug(f"  Decision (POS Fallback): Noun Type assigned ({pos}).")
        elif pos == "ADJ":
            assigned_entity = ADJ_MOD_TYPE; logger.debug(f"  Decision (POS Fallback): Adjective Modifier Type assigned.")
        elif pos == "DET":
             assigned_entity = DET_TYPE; logger.debug(f"  Decision (POS Fallback): Determiner Type assigned.")
        elif pos == "ADP":
            assigned_entity = PREP_FUNCTOR_TYPE; logger.debug(f"  Decision (POS Fallback): Preposition Functor Type assigned.")
        elif pos == "ADV":
            assigned_entity = ADV_FUNCTOR_TYPE; logger.debug(f"  Decision (POS Fallback): Adverb Functor Type assigned.")
        elif pos == "VERB":
             assigned_entity = VERB_TRANS_TYPE if sentence_has_object else VERB_INTRANS_TYPE
             logger.warning(f"  Decision (POS Fallback): Verb '{token_text}' wasn't assigned role/type. Defaulting.")

    # --- 5. Handle Ignored POS tags ---
    # (Keep this section as per your V2.2 logic)
    if assigned_entity is None:
        if pos in ["PUNCT", "SYM", "PART", "CCONJ", "SCONJ", "AUX", "INTJ"]:
            logger.debug(f"  Decision (POS ignore/Final): Explicitly None for POS {pos}")
        else:
            logger.warning(f"Unhandled POS/DepRel combination: POS='{pos}', DepRel='{dep_rel}'. Defaulting to N.")
            assigned_entity = N

    # --- Final Check and Logging ---
    # (Keep checks for None global types)
    if isinstance(assigned_entity, Box):
        # Use names for comparison as Box objects might be different instances
        if assigned_entity.name == "DetFunctor" and DET_TYPE is None: assigned_entity = N; logger.error("DET_TYPE is None!")
        if assigned_entity.name == "AdjModFunctor" and ADJ_MOD_TYPE is None: assigned_entity = N; logger.error("ADJ_MOD_TYPE is None!")
        if assigned_entity.name == "PrepFunctor" and PREP_FUNCTOR_TYPE is None: assigned_entity = N; logger.error("PREP_FUNCTOR_TYPE is None!")
        # Check ADJ_PRED_TYPE only if it was assigned
        if assigned_entity.name == "AdjPredFunctor" and ADJ_PRED_TYPE is None: assigned_entity = N; logger.error("ADJ_PRED_TYPE is None!")
        if assigned_entity.name == "VerbIntransFunctor" and VERB_INTRANS_TYPE is None: assigned_entity = N; logger.error("VERB_INTRANS_TYPE is None!")
        if assigned_entity.name == "VerbTransFunctor" and VERB_TRANS_TYPE is None: assigned_entity = N; logger.error("VERB_TRANS_TYPE is None!")
        if assigned_entity.name == "AdvFunctor" and ADV_FUNCTOR_TYPE is None: assigned_entity = N; logger.error("ADV_FUNCTOR_TYPE is None!")
        # Check dynamic NounPred boxes
        if assigned_entity.name.startswith("NounPred_") and (assigned_entity.dom != N or assigned_entity.cod != S):
             logger.error(f"Dynamic Noun Predicate {assigned_entity.name} has incorrect type {assigned_entity.dom}->{assigned_entity.cod}. Resetting to N.")
             assigned_entity = N


    if debug:
        final_type_str = str(assigned_entity) if assigned_entity else "None"
        logger.debug(f"  >> Final Assigned Type V2.4 for '{token_text}': {final_type_str} (Type: {type(assigned_entity).__name__})")

    return assigned_entity
# ==================================
# Diagram Creation Functions (V3.3 - NP-Internal First)
# ==================================

# --- Helper: find_subwire_index_v2 (Unchanged) ---
def find_subwire_index_v2(main_wire_type: Ty, sub_type: Ty, wire_start_index: int, path: List[int] = None) -> Optional[Tuple[int, List[int]]]:
    """Finds the start index of a sub-type within a main type."""
    if path is None: path = []
    if main_wire_type == sub_type: return wire_start_index, path
    if not hasattr(main_wire_type, 'objects') or not main_wire_type.objects: return None
    current_offset = wire_start_index
    for i, t in enumerate(main_wire_type.objects):
        result = find_subwire_index_v2(t, sub_type, current_offset, path + [i])
        if result: return result
        current_offset += len(t) # type: ignore
    return None

# --- Helper: apply_cup_at_indices_v3 (Unchanged) ---
def apply_cup_at_indices_v3(diagram: GrammarDiagram, wire1_abs_start_idx: int, wire1_type: Ty, wire1_orig_tok_idx: int, wire2_abs_start_idx: int, wire2_type: Ty, wire2_orig_tok_idx: int, current_full_wire_map: List[Dict[str, Any]]) -> Optional[Tuple[GrammarDiagram, List[Dict[str, Any]]]]:
    """Applies swaps and Cup operation. (Implementation from V3.2)"""
    logger.debug(f"Attempting Cup V3: W1({wire1_orig_tok_idx},{wire1_type}@{wire1_abs_start_idx}), W2({wire2_orig_tok_idx},{wire2_type}@{wire2_abs_start_idx}) on Cod={diagram.cod}")
    if not (wire1_type.r == wire2_type or wire1_type == wire2_type.r): logger.error(f"Cup Error: Types {wire1_type} and {wire2_type} not adjoints."); return None
    if wire1_abs_start_idx == wire2_abs_start_idx: logger.error("Cup Error: Cannot cup wire with itself."); return None

    n_wires_total = len(diagram.cod); len1 = len(wire1_type); len2 = len(wire2_type)
    left_idx, right_idx = min(wire1_abs_start_idx, wire2_abs_start_idx), max(wire1_abs_start_idx, wire2_abs_start_idx)
    left_len, right_len = (len1, len2) if wire1_abs_start_idx < wire2_abs_start_idx else (len2, len1)
    left_type, right_type = (wire1_type, wire2_type) if wire1_abs_start_idx < wire2_abs_start_idx else (wire2_type, wire1_type)

    if not (0 <= left_idx < right_idx < n_wires_total and left_idx + left_len <= right_idx and right_idx + right_len <= n_wires_total):
        logger.error(f"Cup Error: Invalid indices/lengths: Left({left_idx},{left_len}), Right({right_idx},{right_len}) on total {n_wires_total}"); return None

    perm = [i for i in range(n_wires_total) if i not in range(left_idx, left_idx + left_len) and i not in range(right_idx, right_idx + right_len)]
    perm.extend(range(left_idx, left_idx + left_len))
    perm.extend(range(right_idx, right_idx + right_len))
    logger.debug(f"Cup Permutation: {perm}")
    if len(perm) != n_wires_total: logger.error(f"Cup Error: Permutation length mismatch {len(perm)} vs {n_wires_total}."); return None

    try:
        permuted_diagram = diagram >> Swap(diagram.cod, perm) if perm != list(range(n_wires_total)) else diagram
    except Exception as e_perm: logger.error(f"Cup Error: Swap failed: {e_perm}", exc_info=True); return None

    try:
        id_wires_len = n_wires_total - (left_len + right_len)
        id_wires_type = permuted_diagram.cod[:id_wires_len] # type: ignore
        cup_op = Id(id_wires_type) @ Cup(left_type, right_type)
        if permuted_diagram.cod != cup_op.dom: logger.error(f"Cup Error: Mismatch after permute {permuted_diagram.cod} != Cup domain {cup_op.dom}"); return None
        result_diagram = permuted_diagram >> cup_op
        logger.info(f"Cup({left_type}, {right_type}) applied. New cod: {result_diagram.cod}")

        # Wire Map Update (Placeholder - needs robust implementation)
        new_wire_map = []
        remaining_units = [u for u in current_full_wire_map if u['orig_idx'] not in (wire1_orig_tok_idx, wire2_orig_tok_idx)]
        if result_diagram.cod == S or len(result_diagram.cod) == 0: logger.debug("Diagram reduced to S or Ty(). Simple wire map.")
        else:
            logger.warning("Wire map update after cup is complex (using placeholder).")
            temp_offset = 0
            for unit_data in remaining_units:
                ud = unit_data.copy(); main_len = ud['main_len']; feat_len = ud.get('feat_len', 0)
                ud['main_abs_start'] = temp_offset; ud['feat_abs_start'] = temp_offset + main_len
                ud['unit_abs_start'] = temp_offset; ud['unit_len'] = main_len + feat_len
                new_wire_map.append(ud); temp_offset += ud['unit_len']
            if temp_offset != len(result_diagram.cod): logger.error("Wire map length mismatch after cup."); # return None # Optionally fail

        return result_diagram, new_wire_map
    except Exception as e_cup: logger.error(f"Cup Error: Application failed: {e_cup}", exc_info=True); return None


# --- NEW: Build Noun Phrase Diagram ---
def build_np_diagram_v4( # Renamed
    head_noun_idx: int,
    analysis_map: Dict[int, Dict[str, Any]],
    roles: Dict[str, Any],
    core_type_map: Dict[int, Union[Ty, GrammarDiagram, None]],
    arg_producer_boxes: Dict[int, Box],
    functor_boxes: Dict[int, Box],
    processed_indices: Set[int], # Keep track of indices used
    debug: bool = True
) -> Optional[GrammarDiagram]:
    """
    Recursively builds a diagram for a Noun Phrase centered around head_noun_idx.
    Handles DET, ADJ (amod), and PP (nmod) modifiers.
    V4: Allows 'X'/'DET' heads if Subj/Obj role. Simplified PP object handling on recursion fail.
    Returns a diagram of type Ty() -> N.
    """
    if head_noun_idx in processed_indices:
        logger.warning(f"NP Head Noun {head_noun_idx} already processed. Skipping NP build.")
        return arg_producer_boxes.get(head_noun_idx)

    head_analysis = analysis_map.get(head_noun_idx)
    if not head_analysis:
        logger.error(f"Cannot build NP: Analysis not found for head index {head_noun_idx}.")
        return None

    # Check POS, allowing 'X' or 'DET' if the token is identified as Subject or Object
    allowed_pos = ["NOUN", "PROPN", "PRON"]
    is_subj_or_obj = (head_noun_idx == roles.get('subject') or head_noun_idx == roles.get('object'))
    head_pos = head_analysis['upos']
    head_deprel = head_analysis['deprel']

    can_build = head_pos in allowed_pos or \
                (head_pos == 'X' and is_subj_or_obj) or \
                (head_pos == 'DET' and is_subj_or_obj and head_deprel == 'nsubj')

    if not can_build:
        logger.error(f"Cannot build NP: Head index {head_noun_idx} ('{head_analysis['text']}') has invalid POS/Role combination ('{head_pos}', Subj/Obj={is_subj_or_obj}, DepRel='{head_deprel}') for NP head.")
        return None

    # Start with the base noun box (Ty() -> N)
    np_diagram = arg_producer_boxes.get(head_noun_idx)
    if np_diagram is None:
        if core_type_map.get(head_noun_idx) == N:
             box_name = f"{head_analysis.get('lemma', head_analysis.get('text','unk'))}_{head_noun_idx}"
             np_diagram = Box(box_name, Ty(), N)
             arg_producer_boxes[head_noun_idx] = np_diagram
             logger.info(f"Created missing Argument Producer Box for '{box_name}': Ty() -> N")
        else:
             logger.error(f"Cannot build NP: Argument producer box not found or incorrect type for head noun {head_noun_idx}.")
             return None

    logger.info(f"Building NP diagram V4 for head noun: '{head_analysis['text']}' (idx {head_noun_idx})")
    logger.debug(f"  Initial NP diagram: {np_diagram.name} ({np_diagram.dom} -> {np_diagram.cod})")
    processed_indices.add(head_noun_idx)

    # Find dependents
    dep_graph = roles.get('dependency_graph', {})
    dependents = dep_graph.get(head_noun_idx, []) if isinstance(dep_graph, dict) else []
    logger.debug(f"  Dependents of {head_noun_idx}: {dependents}")

    # --- Apply Modifiers ---
    modifiers_to_apply = []
    for dep_idx, dep_rel in dependents:
        if dep_idx in processed_indices: continue

        modifier_box = functor_boxes.get(dep_idx)
        modifier_analysis = analysis_map.get(dep_idx)

        if modifier_box and modifier_analysis:
            # Determiner
            if dep_rel == 'det' and head_pos != 'DET' and modifier_box.dom == N and modifier_box.cod == N:
                modifiers_to_apply.append({'idx': dep_idx, 'box': modifier_box, 'rel': 'det', 'order': 0})
            # Adjective Modifier
            elif dep_rel == 'amod' and modifier_box.dom == N and modifier_box.cod == N:
                 modifiers_to_apply.append({'idx': dep_idx, 'box': modifier_box, 'rel': 'amod', 'order': 1})
            # Prepositional Phrase Modifier
            elif dep_rel == 'case' and modifier_analysis['upos'] == 'ADP' and PREP_FUNCTOR_TYPE and N_MOD_BY_N:
                 prep_box = modifier_box
                 prep_dependents = dep_graph.get(dep_idx, []) if isinstance(dep_graph, dict) else []
                 pp_obj_idx = None
                 for pp_dep_idx, pp_dep_rel in prep_dependents:
                      pp_obj_analysis = analysis_map.get(pp_dep_idx)
                      if pp_obj_analysis and pp_dep_idx not in processed_indices and pp_obj_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X"]:
                           pp_obj_idx = pp_dep_idx
                           break
                 if pp_obj_idx is not None:
                      pp_obj_np_diagram = None
                      # Attempt recursive build for PP object
                      logger.debug(f"  Attempting recursive NP build for PP object {pp_obj_idx} ('{analysis_map[pp_obj_idx]['text']}')")
                      pp_obj_np_diagram = build_np_diagram_v4( # Recursive call
                          pp_obj_idx, analysis_map, roles, core_type_map,
                          arg_producer_boxes, functor_boxes, processed_indices, debug
                      )

                      # Fallback if recursive build fails
                      if pp_obj_np_diagram is None or pp_obj_np_diagram.cod != N:
                          logger.warning(f"  Recursive NP build failed for PP object {pp_obj_idx}. Falling back to simple box for PP composition.")
                          pp_obj_np_diagram = arg_producer_boxes.get(pp_obj_idx) # Use simple box
                          if pp_obj_np_diagram and pp_obj_idx not in processed_indices:
                               processed_indices.add(pp_obj_idx) # Mark simple box as used

                      # Compose PP if object diagram is valid
                      if pp_obj_np_diagram and pp_obj_np_diagram.cod == N:
                           try:
                               composed_pp = pp_obj_np_diagram >> prep_box
                               if composed_pp.cod == N:
                                    modifiers_to_apply.append({'idx': dep_idx, 'box': composed_pp, 'rel': 'pp_nmod', 'order': 2, 'pp_obj_idx': pp_obj_idx})
                                    logger.info(f"  Identified NP-PP modifier: Prep='{modifier_analysis['text']}' (idx {dep_idx}), Obj='{analysis_map[pp_obj_idx]['text']}' (idx {pp_obj_idx})")
                               else: logger.warning(f"  PP composition failed for prep {dep_idx}.")
                           except Exception as e_pp_compose: logger.error(f"  Error composing PP for prep {dep_idx}: {e_pp_compose}")
                      else: logger.warning(f"  Could not get valid diagram (even fallback) for PP object {pp_obj_idx}.")
                 else: logger.debug(f"  Could not find valid, unprocessed object for preposition {dep_idx}.")

    # Sort and apply modifiers
    modifiers_to_apply.sort(key=lambda m: m['order'])
    for mod_info in modifiers_to_apply:
        mod_idx = mod_info['idx']
        mod_box = mod_info['box']
        dep_rel = mod_info['rel']

        try:
            if dep_rel in ['det', 'amod']:
                 logger.info(f"  Applying {dep_rel.upper()} modifier: '{analysis_map[mod_idx]['text']}' (idx {mod_idx}) to NP '{head_analysis['text']}'")
                 if np_diagram.cod == mod_box.dom: # type: ignore
                     np_diagram = np_diagram >> mod_box
                     processed_indices.add(mod_idx)
                     logger.debug(f"    NP diagram after {dep_rel.upper()}: {np_diagram}")
                 else: logger.warning(f"    Type mismatch applying {dep_rel.upper()} {mod_idx}.")

            elif dep_rel == 'pp_nmod':
                 composed_pp_diag = mod_box
                 pp_obj_idx = mod_info['pp_obj_idx'] # Already marked processed if simple box used
                 if np_diagram.cod == N and composed_pp_diag.cod == N and N_MOD_BY_N is not None: # type: ignore
                     logger.info(f"  Applying NP-PP ({analysis_map[mod_idx]['text']}) to NP '{head_analysis['text']}' using N_MOD_BY_N")
                     np_diagram = (np_diagram @ composed_pp_diag) >> N_MOD_BY_N
                     processed_indices.add(mod_idx) # Mark preposition as used
                     logger.debug(f"    NP diagram after PP: {np_diagram}")
                 else: logger.warning(f"    Type mismatch applying PP {mod_idx} or N_MOD_BY_N is None.")

        except Exception as e_mod_apply:
            logger.error(f"  Error applying modifier {mod_idx} ({dep_rel}): {e_mod_apply}", exc_info=True)

    logger.info(f"Finished building NP V4 for '{head_analysis['text']}'. Final diagram cod: {np_diagram.cod if np_diagram else 'None'}. Consumed indices: {processed_indices}") # type: ignore
    if np_diagram and np_diagram.cod == N:
        return np_diagram
    else:
        logger.error(f"NP diagram build V4 for {head_noun_idx} resulted in invalid type: {np_diagram.cod if np_diagram else 'None'}. Returning None.")
        return None

def get_diagram_repr(diag_obj: Optional[GrammarDiagram]) -> str:
    if diag_obj is None:
        return "None"
    if hasattr(diag_obj, 'name'): # It's a Box
        return diag_obj.name
    # For complex diagrams, str(diag_obj) can be very long.
    # Consider a summarized representation if needed, e.g., based on its type or first/last boxes.
    # For now, str() is okay for debugging but might be verbose.
    return str(diag_obj)

def create_verbal_sentence_diagram_v3_7( # Renamed
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]],
    original_indices: List[int], # Indices of tokens included in word_core_types
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_verbal"
) -> Optional[GrammarDiagram]:
    """
    Creates a DisCoCat diagram for verbal sentences using a hybrid approach.
    Attempts to build NPs first, falls back to simple boxes if NP build fails.
    V3.7: Hybrid approach with improved fallback and nested composition checks.
    """
    logger.info(f"Creating verbal diagram (V3.7 - Hybrid/Fallback/Nested) for: {' '.join(tokens)}")
    # --- Check essential types ---
    if VERB_TRANS_TYPE is None or VERB_INTRANS_TYPE is None or S_MOD_BY_N is None or ADV_FUNCTOR_TYPE is None or N_MOD_BY_N is None or PREP_FUNCTOR_TYPE is None:
         logger.error("Cannot create verbal diagram: Essential Verb/Modifier/Prep types are not defined.")
         return None

    # --- Map data and create initial boxes ---
    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}
    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}
    processed_indices: Set[int] = set()

    for orig_idx, core_entity in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or core_entity is None: continue
        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"
        try:
            if isinstance(core_entity, Box):
                functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod)
            elif isinstance(core_entity, Ty) and core_entity == N:
                arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)
        except Exception as e_box_create:
             logger.error(f"Error creating box for {orig_idx} ('{analysis['text']}'): {e_box_create}")

    # --- Identify Core Components ---
    subj_idx = roles.get('subject')
    verb_idx = roles.get('verb')
    obj_idx = roles.get('object')

    verb_functor_box = functor_boxes.get(verb_idx) if verb_idx is not None else None
    #verb_functor_repr = verb_functor_box.name if hasattr(verb_functor_box, 'name') else str(verb_functor_box)
    if verb_functor_box is None:
         logger.error(f"Cannot form verbal diagram: Main verb functor not found for index {verb_idx}.")
         return None

    # --- Attempt to Build Subject NP, with Fallback ---
    subj_diag: Optional[GrammarDiagram] = None
   # subj_diag_repr = subj_diag.name if hasattr(subj_diag, 'name') else str(subj_diag)
    if subj_idx is not None:
        subj_diag = build_np_diagram_v4( # Use the updated NP builder
            subj_idx, analysis_map, roles, core_type_map,
            arg_producer_boxes, functor_boxes, processed_indices, debug
        )
        if subj_diag is None or subj_diag.cod != N:
            logger.warning(f"Failed/Invalid Subject NP build for index {subj_idx}. Falling back to simple box.")
            subj_diag = arg_producer_boxes.get(subj_idx) # Get the simple Ty()->N box
            if subj_diag is None:
                 logger.error(f"Fallback failed: Simple argument box missing for subject {subj_idx}.")
            elif subj_idx not in processed_indices: # Ensure index is marked if fallback used
                 processed_indices.add(subj_idx)
        # If subj_diag is still None here, the component is truly missing/invalid

    # --- Attempt to Build Object NP, with Fallback ---
    obj_diag: Optional[GrammarDiagram] = None
    #obj_diag_repr = obj_diag.name if hasattr(obj_diag, 'name') and obj_diag else "None"
    obj_analysis = analysis_map.get(obj_idx) if obj_idx is not None else None
    if obj_analysis and obj_analysis['upos'] == 'ADJ' and obj_analysis['deprel'] == 'xcomp':
        logger.warning(f"Object role points to ADJ with xcomp (idx {obj_idx}, '{obj_analysis['text']}'). Treating as non-object for NP build. Object diagram will be None.")
        obj_diag = None # Force obj_diag to be None
        if obj_idx not in processed_indices: # Mark as processed to avoid later attempts
            processed_indices.add(obj_idx)
    elif obj_idx is not None:
        # Make sure object wasn't already consumed by subject's NP build (e.g., PP object)
        if obj_idx not in processed_indices:
            obj_diag = build_np_diagram_v4( # Use the updated NP builder
                obj_idx, analysis_map, roles, core_type_map,
                arg_producer_boxes, functor_boxes, processed_indices, debug
            )
            if obj_diag is None or obj_diag.cod != N:
                logger.warning(f"Failed/Invalid Object NP build for index {obj_idx}. Falling back to simple box.")
                obj_diag = arg_producer_boxes.get(obj_idx)
                if obj_diag is None:
                    logger.error(f"Fallback failed: Simple argument box missing for object {obj_idx}.")
                elif obj_idx not in processed_indices:
                    processed_indices.add(obj_idx)
            # If obj_diag is still None here, the component is truly missing/invalid
        else:
            logger.warning(f"Object index {obj_idx} was already processed (likely part of Subject NP). Skipping object diagram.")


    # Mark verb as processed
    if verb_idx is not None: processed_indices.add(verb_idx)

    logger.debug(f"  Components after NP build/fallback: Subj(idx {subj_idx}): {subj_diag}, Verb(idx {verb_idx}): {verb_functor_box}, Obj(idx {obj_idx}): {obj_diag}")

    # --- Compose Basic Clause (Nested Checks) ---
    final_diagram: Optional[GrammarDiagram] = None
    structure_type = roles.get("structure")
    clause_composition_success = False

    try:
        # Transitive Verb
        if verb_functor_box.dom == (N @ N): # Transitive verb N @ N -> S
            subj_diag_repr = get_diagram_repr(subj_diag)
            obj_diag_repr = get_diagram_repr(obj_diag)
            verb_functor_repr = get_diagram_repr(verb_functor_box)

            if subj_diag and subj_diag.cod == N and obj_diag and obj_diag.cod == N:
                # Standard SVO/VSO with explicit subject and object
                final_diagram = (subj_diag @ obj_diag) >> verb_functor_box
                clause_composition_success = True
                logger.info(f"Composed transitive clause ({structure_type}) with S '{subj_diag_repr}' and O '{obj_diag_repr}'.")
            elif subj_diag and subj_diag.cod == N and obj_diag is None: # <<<< THIS IS THE CASE FOR SENTENCE 72
                # Subject present, but object is missing (e.g., "الرجل يكتب")
                logger.warning(f"Composing transitive verb '{verb_functor_repr}' for structure '{structure_type}' with subject '{subj_diag_repr}' and Id(N) placeholder for MISSING object.")
                final_diagram = (subj_diag @ Id(N)) >> verb_functor_box
                clause_composition_success = True
            elif obj_diag and obj_diag.cod == N and subj_diag is None and \
                (structure_type == "VO_NO_SUBJ" or structure_type == "OTHER"):
                # Object present, but subject is missing
                logger.warning(f"Composing {structure_type} for transitive verb '{verb_functor_repr}' using Id(N) placeholder for subject, with object '{obj_diag_repr}'.")
                final_diagram = (Id(N) @ obj_diag) >> verb_functor_box
                clause_composition_success = True
            else: # Fallback if specific conditions not met
                subj_status = "present_valid" if subj_diag and subj_diag.cod == N else f"missing_or_invalid ({subj_diag_repr})"
                obj_status = "present_valid" if obj_diag and obj_diag.cod == N else f"missing_or_invalid ({obj_diag_repr})"
                # If subject is present but object is not simply None but an invalid diagram, this else might be hit.
                # The previous `elif subj_diag and subj_diag.cod == N and obj_diag is None:` should catch it.
                # Let's refine the error message.
                if not (subj_diag and subj_diag.cod == N):
                    logger.error(f"Cannot compose transitive clause ({structure_type}): Subject diagram is {subj_status}. Verb: '{verb_functor_repr}'")
                elif not (obj_diag is None or (obj_diag and obj_diag.cod == N)): # Object is neither None nor a valid N-diagram
                    logger.error(f"Cannot compose transitive clause ({structure_type}): Object diagram is {obj_status}. Verb: '{verb_functor_repr}'")
                else: # General catch-all, should be rare if above conditions are exhaustive
                    logger.error(f"Cannot compose transitive clause ({structure_type}) due to unhandled S/O combination: Subj is {subj_status}, Obj is {obj_status}. Verb: '{verb_functor_repr}'")
                return None

        # Intransitive Verb
        elif verb_functor_box.dom == N:
            subj_diag_repr = get_diagram_repr(subj_diag)
            obj_diag_repr = get_diagram_repr(obj_diag)
            verb_functor_repr = get_diagram_repr(verb_functor_box)
            if subj_diag and subj_diag.cod == N:
                final_diagram = subj_diag >> verb_functor_box
                clause_composition_success = True
                logger.info(f"Composed intransitive clause ({structure_type}) with subject '{subj_diag_repr}'.")
            elif structure_type == "VERBAL_ONLY" and subj_diag is None and obj_diag is None:
                logger.warning(f"Composing VERBAL_ONLY for intransitive verb {verb_idx} ('{verb_functor_repr}') using Id(N) placeholder for subject.")
                final_diagram = Id(N) >> verb_functor_box # Apply verb to implicit N
                clause_composition_success = True
            # NEW FALLBACK for OTHER structure with missing subject for Intransitive verb
            elif subj_diag is None and structure_type == "OTHER":
                logger.warning(f"Intransitive verb '{verb_functor_repr}' in 'OTHER' structure with no subject. Searching for a fallback noun...")
                fallback_subj_diag: Optional[GrammarDiagram] = None
                fallback_subj_idx: Optional[int] = None

                # Try to find an unassigned N from arg_producer_boxes that hasn't been processed
                for idx, arg_box in arg_producer_boxes.items():
                    if idx not in processed_indices and idx != verb_idx: # Ensure it's not the verb itself
                        # A simple heuristic: pick the first available one.
                        # More complex logic could look at proximity or dependency relations if available.
                        analysis_of_arg = analysis_map.get(idx)
                        if analysis_of_arg and analysis_of_arg['upos'] in ["NOUN", "PROPN", "PRON", "X"]:
                            # Attempt to build/get NP for this fallback subject
                            temp_processed_for_fallback_subj = processed_indices.copy() # Avoid polluting main processed_indices yet

                            fb_subj_np = build_np_diagram_v4(
                                idx, analysis_map, roles, core_type_map,
                                arg_producer_boxes, functor_boxes, temp_processed_for_fallback_subj, debug=False 
                            )
                            if fb_subj_np and fb_subj_np.cod == N:
                                fallback_subj_diag = fb_subj_np
                                fallback_subj_idx = idx
                                processed_indices.update(temp_processed_for_fallback_subj)
                                # Add indices consumed by this fallback NP to the main processed_indices
                                # *after* successful composition with the verb.
                                logger.info(f"  Found fallback subject candidate: '{arg_box.name}' (idx {idx}) for intransitive verb.")
                                break 
                            elif arg_box.cod == N: # Simpler fallback if NP build fails
                                fallback_subj_diag = arg_box
                                fallback_subj_idx = idx
                                processed_indices.add(idx)
                                logger.info(f"  Found simple fallback subject candidate (arg_box): '{arg_box.name}' (idx {idx}) for intransitive verb.")
                                break


                if fallback_subj_diag:
                    fallback_subj_diag_repr = fallback_subj_diag.name if hasattr(fallback_subj_diag, 'name') else str(fallback_subj_diag)
                    final_diagram = fallback_subj_diag >> verb_functor_box
                    clause_composition_success = True
                    logger.info(f"Composed intransitive clause ({structure_type}) using fallback subject '{fallback_subj_diag.name}' (idx {fallback_subj_idx}).")
                    # Crucially, mark the fallback subject and its components as processed
                    # If build_np_diagram_v4 was used, its processed_indices need to be merged.
                    # For simplicity here, assuming build_np_diagram_v4 updates its passed 'processed_indices' set.
                    # If it returns a new set, you'd union them.
                    # For now, just add the main fallback_subj_idx. If it was an NP, its components were handled by build_np_diagram.
                    if fallback_subj_idx is not None:
                        # If build_np_diagram_v4 was successful for fb_subj_np, its components are already in its *local* processed_indices.
                        # We need to ensure those are added to the main 'processed_indices' set.
                        # This requires build_np_diagram_v4 to potentially return the set of indices it consumed,
                        # or for us to re-run it here with the main 'processed_indices' set.
                        # Let's assume build_np_diagram_v4 correctly updates the passed 'processed_indices' set.
                        # If not, a simpler processed_indices.add(fallback_subj_idx) might be initially sufficient.
                        # Re-running build_np_diagram_v4 here with the main 'processed_indices' to ensure they are marked.
                        if fallback_subj_idx is not None:
                            build_np_diagram_v4(fallback_subj_idx, analysis_map, roles, core_type_map,
                                                arg_producer_boxes, functor_boxes, processed_indices, debug=False)

                else:
                    logger.warning(f"  No suitable fallback subject found for intransitive verb in 'OTHER' structure. Attempting Id(N).")
                    final_diagram = Id(N) >> verb_functor_box
                    clause_composition_success = True

            elif obj_diag and obj_diag.cod == N and structure_type == "VO_NO_SUBJ": # Original VO_NO_SUBJ for intransitive
                logger.warning(f"Composing VO_NO_SUBJ for intransitive verb {verb_idx} using object diagram as subject.")
                final_diagram = obj_diag >> verb_functor_box # Treat object as subject semantically
                clause_composition_success = True
            else:
                logger.error(f"Cannot compose intransitive clause ({structure_type}): Missing or invalid Subject diagram. Subj={subj_diag}")
                return None

        # Verb Only
        elif verb_functor_box.dom == Ty():
             final_diagram = verb_functor_box
             clause_composition_success = True
             logger.info("Using verb box directly for VERBAL_ONLY structure.")

        # If no composition path matched
        if not clause_composition_success:
             logger.error(f"Could not compose basic clause for structure '{structure_type}' with verb type {verb_functor_box.dom}.")
             return None

        logger.info(f"Successfully composed basic clause. Diagram cod: {final_diagram.cod if final_diagram else 'None'}") # type: ignore

    except Exception as e_clause:
        logger.error(f"Error during basic clause composition ({structure_type}): {e_clause}", exc_info=True)
        return None

    # --- Attach Sentence-Level Modifiers ---
    # (Use the simpler attachment logic from V3.4/V3.6, applied *after* core clause composition)
    if final_diagram and final_diagram.cod == S:
        logger.debug("--- Attaching Sentence-Level Modifiers ---")
        dep_graph = roles.get('dependency_graph', {})
        if not isinstance(dep_graph, dict): dep_graph = {}

        remaining_indices = sorted([idx for idx in original_indices if idx not in processed_indices])

        for mod_idx in remaining_indices:
             if mod_idx in processed_indices: continue

             mod_analysis = analysis_map.get(mod_idx)
             mod_functor_box = functor_boxes.get(mod_idx)
             mod_arg_box = arg_producer_boxes.get(mod_idx)

             if not mod_analysis: continue

             head_idx = mod_analysis['head']
             dep_rel = mod_analysis['deprel']

             # Attach Adverbs modifying the main verb
             if dep_rel == 'advmod' and mod_functor_box and mod_functor_box.dom == S and mod_functor_box.cod == S:
                 if head_idx == verb_idx and ADV_FUNCTOR_TYPE:
                     try:
                         logger.info(f"Applying ADV modifier '{mod_analysis['text']}' (idx {mod_idx}) to Sentence.")
                         final_diagram = final_diagram >> mod_functor_box
                         processed_indices.add(mod_idx)
                     except Exception as e_adv: logger.error(f"Error applying ADV {mod_idx}: {e_adv}")

             # Attach Sentence-level PPs
             elif mod_analysis['upos'] == 'ADP' and mod_functor_box and mod_functor_box.dom == N and mod_functor_box.cod == N:
                 # Find PP object (must be simple arg box, not already processed)
                 pp_obj_idx = None
                 prep_dependents = dep_graph.get(mod_idx, [])
                 for pp_dep_idx, pp_dep_rel in prep_dependents:
                     if pp_dep_idx not in processed_indices and pp_dep_idx in arg_producer_boxes:
                         pp_obj_idx = pp_dep_idx; break
                 if pp_obj_idx is None and dep_rel == 'case':
                      if head_idx not in processed_indices and head_idx in arg_producer_boxes:
                           pp_obj_idx = head_idx

                 if pp_obj_idx is not None:
                     pp_obj_diag = arg_producer_boxes.get(pp_obj_idx) # Use simple box
                     pp_obj_analysis = analysis_map.get(pp_obj_idx)
                     if pp_obj_diag and pp_obj_analysis:
                         head_of_pp_obj = pp_obj_analysis['head']
                         is_sentential_pp = (head_of_pp_obj == verb_idx or mod_analysis['head'] == verb_idx)
                         if is_sentential_pp and S_MOD_BY_N:
                             try:
                                 composed_pp = pp_obj_diag >> mod_functor_box # Obj >> Prep
                                 if composed_pp.cod == N:
                                     logger.info(f"Attaching Sentence-PP '{mod_analysis['text']} {pp_obj_analysis['text']}' using S_MOD_BY_N.")
                                     final_diagram = (final_diagram @ composed_pp) >> S_MOD_BY_N # type: ignore
                                     processed_indices.add(mod_idx)
                                     processed_indices.add(pp_obj_idx)
                                 else: logger.warning(f"Composed PP for prep {mod_idx} has unexpected type {composed_pp.cod}")
                             except Exception as e_pp: logger.error(f"Error composing/attaching PP for prep {mod_idx}: {e_pp}")
                         elif not is_sentential_pp:
                              logger.debug(f"PP starting at {mod_idx} appears NP-internal. Skipping sentence attachment.")
                              processed_indices.add(mod_idx)
                              processed_indices.add(pp_obj_idx)


    # --- Final Normalization and Return ---
    if final_diagram:
        try:
            normalized_diagram = final_diagram.normal_form()
            if normalized_diagram.cod == S:
                logger.info(f"Verbal diagram (V3.7 Hybrid) normalization successful. Final cod: {normalized_diagram.cod}")
                return normalized_diagram
            else:
                logger.warning(f"Verbal diagram (V3.7 Hybrid) normalized, but final cod is {normalized_diagram.cod}, not S. Discarding.")
                return None
        except NotImplementedError as e_norm_ni:
             logger.error(f"Normalization failed (NotImplementedError) for diagram: {final_diagram}. Error: {e_norm_ni}")
             return None
        except Exception as e_norm:
            logger.error(f"Verbal diagram (V3.7 Hybrid) normal_form failed: {e_norm}", exc_info=True)
            return None

    logger.warning(f"Could not form a complete verbal diagram (V3.7 Hybrid) ending in S for structure: {structure_type}.")
    return None


def create_nominal_sentence_diagram_v2_7( # Renamed
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]],
    original_indices: List[int], # Indices of tokens included in word_core_types
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_nominal",
    hint_predicate_original_idx: Optional[int] = None # ***** NEW PARAMETER *****
) -> Optional[GrammarDiagram]:
    """
    Creates a DisCoCat diagram for nominal sentences (Subject-Predicate).
    V2.7: Hybrid approach - attempts NP build for subject, falls back to simple box.
    """
    logger.info(f"Creating nominal diagram (V2.7 - Hybrid/Fallback) for: {' '.join(tokens)}")
    # --- Check essential types ---
    if ADJ_PRED_TYPE is None: # Assuming NounPred is created dynamically
         logger.error("Cannot create nominal diagram: ADJ_PRED_TYPE is not defined.")
         return None

    # --- Map data and create initial boxes ---
    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}
    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}
    processed_indices: Set[int] = set()

    for orig_idx, core_entity in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or core_entity is None: continue
        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"
        if isinstance(core_entity, Box):
            functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod)
        elif isinstance(core_entity, Ty) and core_entity == N:
            arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)

    # --- Identify Subject and Predicate ---
    subj_idx = roles.get('subject', roles.get('root'))
    predicate_idx: Optional[int] = None
    predicate_functor_box: Optional[Box] = None
    subj_diag: Optional[GrammarDiagram] = None

    if subj_idx is not None:
        for idx, functor_box in functor_boxes.items():
            if idx == subj_idx: continue
            if functor_box.dom == N and functor_box.cod == S:
                pred_analysis = analysis_map.get(idx)
                if pred_analysis:
                    is_root_and_not_subj = (idx == roles.get('root') and idx != subj_idx)
                    is_headed_by_subj = (pred_analysis.get('head') == subj_idx)
                    # Additional check: Ensure the predicate wasn't already used in the subject NP build
                    if (is_root_and_not_subj or is_headed_by_subj) and idx not in processed_indices:
                        predicate_idx = idx
                        predicate_functor_box = functor_box
                        logger.info(f"  Found nominal predicate functor: '{predicate_functor_box.name}' (idx {predicate_idx})")
                        break

    if hint_predicate_original_idx is not None:
        logger.debug(f"  Using hint_predicate_original_idx: {hint_predicate_original_idx}")
        hinted_functor = core_type_map.get(hint_predicate_original_idx)
        if isinstance(hinted_functor, Box) and hinted_functor.dom == N and hinted_functor.cod == S:
            predicate_idx = hint_predicate_original_idx
            predicate_functor_box = hinted_functor
            logger.info(f"  Using HINTED predicate functor: '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")
        else:
            logger.warning(f"  Hinted predicate idx {hint_predicate_original_idx} does not correspond to a valid N->S functor in core_type_map. Ignoring hint.")
                
    if predicate_functor_box is None and subj_diag and subj_diag.cod == N:
        logger.debug("  No pre-assigned predicate functor. Searching for potential unassigned predicate...")
        # Try to find an unassigned N, ADJ, or NUM that could be the predicate
        potential_predicate_idx = None
        potential_predicate_analysis = None
        # Iterate over original_indices to find an unassigned token that could be a predicate
        # This requires passing the full 'analyses_details' and 'original_indices_for_diagram'
        # to this function, or a map of original_idx to its analysis and core_type.

        # Simplified: Assume 'roles' now contains 'analyses_details_for_context'
        all_analyses_details = roles.get('analyses_details_for_context', [])

        for token_analysis in all_analyses_details:
            idx = token_analysis['original_idx']
            # Ensure idx is in core_type_map and not already processed or the subject itself
            if idx in core_type_map and idx not in processed_indices and idx != subj_idx:
                core_type_of_token = core_type_map.get(idx)
                if isinstance(core_type_of_token, Ty) and core_type_of_token == N and \
                token_analysis['upos'] in ["ADJ", "NOUN", "PROPN", "NUM", "X"]:
                    # Plausibility check: is it the sentence root OR a direct dependent of the subject?
                    # Or, for SUBJ_NO_VERB_OTHER, it might be the root if subject is not.
                    is_plausible_predicate = (token_analysis['deprel'] == 'root') or \
                                            (token_analysis.get('head') == subj_idx) or \
                                            (sentence_structure == "SUBJ_NO_VERB_OTHER" and idx == roles.get('root'))

                    if is_plausible_predicate:
                        potential_predicate_idx = idx
                        potential_predicate_analysis = token_analysis
                        logger.info(f"  Found potential dynamic predicate: '{token_analysis['text']}' (idx {idx}, POS {token_analysis['upos']}, DepRel {token_analysis['deprel']})")
                        break

        if potential_predicate_idx is not None and potential_predicate_analysis is not None:
            pred_lemma = potential_predicate_analysis.get('lemma', potential_predicate_analysis.get('text', 'unk'))
            pred_pos = potential_predicate_analysis['upos']

            box_name_prefix = ""
            if pred_pos == "ADJ" and ADJ_PRED_TYPE is not None:
                # Create a unique Box instance for this specific adjective predicate
                box_name_prefix = "AdjPred"
                predicate_functor_box = Box(f"{box_name_prefix}_{pred_lemma}_{potential_predicate_idx}", N, S)
                logger.info(f"  Dynamically created Adjective Predicate Functor for '{pred_lemma}' (idx {potential_predicate_idx})")
            elif pred_pos in ["NOUN", "PROPN", "NUM", "X"]:
                box_name_prefix = "NounPred"
                predicate_functor_box = Box(f"{box_name_prefix}_{pred_lemma}_{potential_predicate_idx}", N, S)
                logger.info(f"  Dynamically created Noun/X Predicate Functor for '{pred_lemma}' (idx {potential_predicate_idx})")

            if predicate_functor_box:
                predicate_idx = potential_predicate_idx
                functor_boxes[predicate_idx] = predicate_functor_box

                # Ensure it's marked as processed if used
                # processed_indices.add(predicate_idx) # Will be added later if composition succeeds
    # --- Attempt to Build Subject NP, with Fallback ---
    if subj_idx is not None:
        logger.debug(f"Attempting to build/get subject NP for explicit subj_idx: {subj_idx} ('{analysis_map.get(subj_idx, {}).get('text', 'N/A')}')")
        # If subj_idx points to a DET that was typed as N
        for idx_loop, functor_box_iter in functor_boxes.items():
            if idx_loop == subj_idx: continue
            if isinstance(functor_box_iter, Box) and functor_box_iter.dom == N and functor_box_iter.cod == S:
                pred_analysis_loop = analysis_map.get(idx_loop)
                if pred_analysis_loop:
                    is_root_and_not_subj = (idx_loop == roles.get('root') and idx_loop != subj_idx)
                    is_headed_by_subj = (pred_analysis_loop.get('head') == subj_idx)
                    # Allow if it's the root of the sentence OR if its head is the subject
                    if (is_root_and_not_subj or is_headed_by_subj) and idx_loop not in processed_indices:
                        predicate_idx = idx_loop # Assign to function-scope variable
                        predicate_functor_box = functor_box_iter # Assign to function-scope variable
                        logger.info(f"  Found pre-assigned nominal predicate functor: '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")
                        break 
        if analysis_map.get(subj_idx, {}).get('upos') == 'DET' and core_type_map.get(subj_idx) == N:
            subj_diag = arg_producer_boxes.get(subj_idx)
            if subj_diag:
                logger.info(f"Using DET '{analysis_map[subj_idx]['text']}' (idx {subj_idx}) directly as N-type subject diagram.")
                if subj_idx not in processed_indices: processed_indices.add(subj_idx) # Mark as processed
            else:
                logger.error(f"DET subject {subj_idx} was typed N but not found in arg_producer_boxes. subj_diag remains None.")
                # subj_diag is already None or remains None

        # If not a DET subject or DET as N failed, try full NP build
        if subj_diag is None: 
            logger.debug(f"Attempting build_np_diagram_v4 for subject index {subj_idx}.")
            subj_diag = build_np_diagram_v4(
                subj_idx, analysis_map, roles, core_type_map,
                arg_producer_boxes, functor_boxes, processed_indices, debug
            )

        # Fallback to simple box if NP build failed or returned non-N
        if subj_diag is None or subj_diag.cod != N:
            logger.warning(f"Subject NP build for explicit index {subj_idx} failed or yielded non-N type ({get_diagram_repr(subj_diag)}). Falling back to simple arg_producer_box.")
            subj_diag = arg_producer_boxes.get(subj_idx) # This might re-assign subj_diag
            if subj_diag:
                logger.info(f"Using fallback arg_producer_box for subject {subj_idx}: {get_diagram_repr(subj_diag)}")
                if subj_idx not in processed_indices: processed_indices.add(subj_idx)
            else:
                logger.error(f"Fallback failed: Simple argument box also missing for explicit subject {subj_idx}. subj_diag is None.")
                # subj_diag remains None
    else: # If subj_idx was None initially (e.g. for SUBJ_NO_VERB_OTHER where Stanza doesn't pick a clear subject role)
        logger.warning("No explicit subject_idx from roles for nominal. Attempting to find a default subject from arg_producer_boxes.")
        for idx_candidate, arg_box_candidate in arg_producer_boxes.items():
            if idx_candidate not in processed_indices and arg_box_candidate.cod == N:
                logger.info(f"  Trying idx {idx_candidate} ('{analysis_map.get(idx_candidate,{}).get('text')}') as default subject.")
                # Attempt to build an NP around this candidate
                temp_processed = processed_indices.copy() # Use a copy for trial
                subj_diag_candidate = build_np_diagram_v4(
                    idx_candidate, analysis_map, roles, core_type_map,
                    arg_producer_boxes, functor_boxes, temp_processed, debug
                )
                if subj_diag_candidate and subj_diag_candidate.cod == N:
                    subj_diag = subj_diag_candidate
                    subj_idx = idx_candidate # CRITICAL: Update subj_idx
                    processed_indices.update(temp_processed) # Commit processed indices from successful NP build
                    logger.info(f"  Found and built default subject NP: '{get_diagram_repr(subj_diag)}' (orig_idx {subj_idx})")
                    break 
                elif arg_box_candidate.cod == N: # Fallback to simple box if NP build fails for this candidate
                    subj_diag = arg_box_candidate
                    subj_idx = idx_candidate # CRITICAL: Update subj_idx
                    processed_indices.add(idx_candidate)
                    logger.info(f"  Found default subject (simple box): '{get_diagram_repr(subj_diag)}' (orig_idx {subj_idx})")
                    break
        if subj_diag is None:
            logger.error("Could not find or build any suitable subject diagram for nominal sentence. subj_diag is None.")
            # subj_diag remains None
        elif subj_idx not in processed_indices: processed_indices.add(subj_idx)
        # If subj_diag is still None, error out

    # Check if required components are valid
    subj_diag_repr = get_diagram_repr(subj_diag) # Use helper for logging
    pred_func_repr = get_diagram_repr(predicate_functor_box)

    if subj_diag is None or subj_diag.cod != N: # Check if subj_diag is valid N
        logger.error(f"Cannot form nominal diagram: Subject diagram is invalid or missing ({subj_diag_repr}).")
        return None
    if predicate_idx is None or predicate_functor_box is None:
        logger.error(f"Cannot form nominal diagram: Predicate functor is missing (Predicate idx: {predicate_idx}, Box: {pred_func_repr}).")
        return None
    if predicate_functor_box.dom != N or predicate_functor_box.cod != S:
        logger.error(f"Cannot form nominal diagram: Predicate functor '{pred_func_repr}' has incorrect type {predicate_functor_box.dom} >> {predicate_functor_box.cod}.")
        return None
    logger.debug(f"PRE-DYNAMIC-PRED-SEARCH: subj_diag is {get_diagram_repr(subj_diag)}, type: {type(subj_diag)}, predicate_functor_box is {get_diagram_repr(predicate_functor_box)}")
    if predicate_functor_box is None:
        if subj_diag is not None and hasattr(subj_diag, 'cod') and subj_diag.cod == N:
            logger.debug("  No pre-assigned predicate functor and subject is valid. Searching for potential unassigned predicate...")
            potential_predicate_head_idx = None
            # Iterate over original_indices to find an unassigned token that could be a predicate HEAD
            for token_analysis in all_analyses_details: # Ensure all_analyses_details is available
                idx = token_analysis['original_idx']
                # Not already processed, not the subject, and a potential predicate POS
                if idx not in processed_indices and idx != subj_idx and \
                token_analysis['upos'] in ["ADJ", "NOUN", "PROPN", "NUM", "X"]:
                    # Plausibility: is it the sentence root OR a direct dependent of the subject?
                    # OR for SUBJ_NO_VERB_OTHER, it might be the root if subject is not.
                    is_plausible_head = (idx == roles.get('root') and token_analysis['upos'] != 'PUNCT') or \
                                        (subj_idx is not None and token_analysis.get('head') == subj_idx)
                    if is_plausible_predicate_head:
                        potential_predicate_head_idx = idx
                        logger.info(f"  Found potential dynamic predicate HEAD: '{token_analysis['text']}' (idx {idx}, POS {token_analysis['upos']})")
                        break

            if potential_predicate_head_idx is not None:
                pred_head_analysis = analysis_map.get(potential_predicate_head_idx)
                if pred_head_analysis:
                    pred_lemma = pred_head_analysis.get('lemma', pred_head_analysis.get('text', 'unk'))
                    pred_pos = pred_head_analysis['upos']
                    temp_functor_box = None
                    # Attempt to build an NP for the predicate if it's a Noun/Num/X.
                    # This NP will be the argument to the dynamically created N->S functor.
                    # The functor itself is associated with the *head* of this NP.
                    #new_functor_name = ""
                    #predicate_argument_diag = None
                    if pred_pos == "ADJ" and ADJ_PRED_TYPE is not None: # ADJ_PRED_TYPE is N->S
                        #new_functor_name = f"DynAdjPred_{pred_lemma}_{potential_predicate_head_idx}"
                        temp_functor_box = Box(f"DynAdjPred_{pred_lemma}_{potential_predicate_head_idx}", N, S)
                    elif pred_pos in ["NOUN", "PROPN", "NUM", "X"]:
                        #new_functor_name = f"DynNounPred_{pred_lemma}_{potential_predicate_head_idx}"
                        temp_functor_box = Box(f"DynNounPred_{pred_lemma}_{potential_predicate_head_idx}", N, S)

                    if temp_functor_box:
                        predicate_idx = potential_predicate_head_idx
                        predicate_functor_box = temp_functor_box # Assign to the main variable

                        # Update core_type_map and functor_boxes (as in previous version)
                        core_type_map[predicate_idx] = predicate_functor_box 
                        functor_boxes[predicate_idx] = predicate_functor_box

                        logger.info(f"  Dynamically assigned predicate functor '{get_diagram_repr(predicate_functor_box)}' to idx {predicate_idx} ('{pred_head_analysis['text']}').")

                        # Mark the predicate head and its nmod dependents as processed.
                        if predicate_idx not in processed_indices:
                            processed_indices.add(predicate_idx)
                            pred_dependents = roles.get('dependency_graph', {}).get(predicate_idx, [])
                            for dep_idx, dep_rel_str in pred_dependents:
                                # For S37, "أرجلٍ" (idx 3) is nmod of "أربعُ" (idx 2).
                                if dep_rel_str == 'nmod' and dep_idx not in processed_indices and core_type_map.get(dep_idx) == N:
                                    logger.info(f"    Marking nmod '{analysis_map.get(dep_idx,{}).get('text')}' (idx {dep_idx}) of dynamic predicate head as processed.")
                                    processed_indices.add(dep_idx)
                    else:
                        logger.warning(f"  Dynamic predicate functor creation failed for head {potential_predicate_head_idx}.")
                else:
                    logger.warning(f"  Dynamic predicate search: Analysis not found for potential head {potential_predicate_head_idx}.")
            else:
                logger.warning("  Dynamic predicate search: No suitable unassigned token found as predicate head after all strategies")

    # Mark predicate as processed if it's a valid index
    elif predicate_idx is not None: # Ensure predicate_idx was actually set
        processed_indices.add(predicate_idx)
    logger.debug(f"  Nominal components: Subj(idx {subj_idx}): {subj_diag_repr}, Pred(idx {predicate_idx}): {pred_func_repr}")

    subj_diag_repr = get_diagram_repr(subj_diag)
    pred_func_repr = get_diagram_repr(predicate_functor_box)
    logger.debug(f"FINAL CHECK for nominal composition: subj_idx={subj_idx}, predicate_idx={predicate_idx}, subj_diag={subj_diag_repr}, predicate_functor_box={pred_func_repr}")

    if subj_diag is None or not (hasattr(subj_diag, 'cod') and subj_diag.cod == N):
        logger.error(f"Cannot form nominal diagram: Subject diagram is invalid or missing ({subj_diag_repr}).")
        return None
    if predicate_idx is None or predicate_functor_box is None: # This is the failing line for S37
        logger.error(f"Cannot form nominal diagram: Predicate functor is missing (Final check: Predicate idx: {predicate_idx}, Box: {pred_func_repr}).")
        return None
    # Mark predicate as processed
    processed_indices.add(predicate_idx)
    logger.debug(f"  Nominal components after NP build/fallback: Subj(idx {subj_idx}): {subj_diag}, Pred(idx {predicate_idx}): {predicate_functor_box.name}")
    if subj_idx is None and subj_diag is not None: # If default subject was found
    # Try to get original_idx from the diagram if it's a simple Box
        if hasattr(subj_diag, 'name') and '_' in subj_diag.name:
            try: subj_idx = int(subj_diag.name.split('_')[-1])
            except ValueError: pass
        if subj_idx is None: logger.warning("Could not infer subj_idx from default subj_diag name.")
    # --- Compose Basic Predication ---
    final_diagram: Optional[GrammarDiagram] = None
    if subj_diag.cod == N and predicate_functor_box.dom == N and predicate_functor_box.cod == S:
        try:
            final_diagram = subj_diag >> predicate_functor_box
            logger.info(f"Nominal composition successful for '{sentence_prefix}'. Cod: {final_diagram.cod}")
        except Exception as e:
            logger.error(f"Nominal composition error for '{sentence_prefix}': {e}", exc_info=True)
            return None
    else:
        logger.error(f"Nominal type mismatch: Subj diag cod={subj_diag.cod}, Pred dom={predicate_functor_box.dom}")
        return None

    # --- Attach Sentence-Level Modifiers (Simplified) ---
    if final_diagram and final_diagram.cod == S:
         logger.debug("--- Attaching Sentence-Level Modifiers (Nominal - Simplified) ---")
         # Add logic similar to verbal function if needed, checking head == predicate_idx

    # --- Final Normalization and Return ---
    if final_diagram:
        try:
            normalized_diagram = final_diagram.normal_form()
            if normalized_diagram.cod == S:
                logger.info(f"Nominal diagram (V2.7 Hybrid) normalization successful. Final cod: {normalized_diagram.cod}")
                return normalized_diagram
            else:
                logger.warning(f"Nominal diagram (V2.7 Hybrid) normalized, but final cod is {normalized_diagram.cod}, not S. Discarding.")
                return None
        except NotImplementedError as e_norm_ni:
             logger.error(f"Normalization failed (NotImplementedError) for diagram: {final_diagram}. Error: {e_norm_ni}")
             return None
        except Exception as e_norm:
            logger.error(f"Nominal diagram (V2.7 Hybrid) normal_form failed: {e_norm}", exc_info=True)
            return None

    logger.warning(f"Could not form a complete nominal diagram (V2.7 Hybrid) ending in S for sentence '{sentence_prefix}'")
    return None


# ==================================
# Main Conversion Function (V2.7 - Uses Hybrid Diagram Functions)
# ==================================
def arabic_to_quantum_enhanced_v2_7( # Keeping name for consistency, but using V2.7.3 logic
    sentence: str,
    debug: bool = True,
    output_dir: Optional[str] = None,
    ansatz_choice: str = "IQP",
    # Pass ansatz parameters explicitly
    n_layers_iqp: int = 1,
    n_single_qubit_params_iqp: int = 3,
    n_layers_strong: int = 1,
    cnot_ranges: Optional[List[Tuple[int, int]]] = None,
    discard_qubits_spider: bool = True,
    **kwargs # Catch-all for other/unexpected keyword arguments
) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Dict[str,Any]], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram, and converts it to a Qiskit QuantumCircuit.
    V2.7.3: Added targeted fallback for 'OTHER' structure based on assigned functor types.
            Uses V2.2.2 type assignment logic.
    """
    if kwargs:
        logger.warning(f"Function arabic_to_quantum_enhanced_v2_7 received UNEXPECTED keyword arguments: {kwargs}")

    # --- 1. Analyze Sentence ---
    # (Keep analysis step the same)
    logger.info(f"Analyzing sentence: '{sentence}'")
    try:
        tokens, analyses_details, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug)
        if analyses_details: roles['analyses_details_for_context'] = analyses_details
        analysis_map_for_diagram_creation = {a['original_idx']: a for a in analyses_details}
        # Make it accessible via roles if sub-functions expect it there, or pass directly
        roles['analysis_map_for_diagram_creation'] = analysis_map_for_diagram_creation
        if structure == "ERROR" or not tokens:
            logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
            return None, None, structure, tokens or [], analyses_details or [], roles or {}
        logger.info(f"Analysis complete. Detected structure: {structure}. Roles: {roles}")
    except Exception as e_analyze_main:
        logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
        return None, None, "ERROR", [], [], {}

    analysis_map_for_diagram_creation = {a['original_idx']: a for a in analyses_details}
    # --- 2. Assign Core DisCoCat Types (Using V2.2.2 Logic) ---
    word_core_types_list = []
    original_indices_for_diagram = []
    filtered_tokens_for_diagram = []
    # Store map of orig_idx to assigned core type for fallback logic
    core_type_map_for_fallback: Dict[int, Union[Ty, GrammarDiagram, None]] = {}

    logger.debug(f"--- Assigning Core Types V2.2.2 for: '{sentence}' ---")
    for i, analysis_entry in enumerate(analyses_details):
        current_core_type = assign_discocat_types_v2_2( # Call V2.2.2 logic
            analysis=analysis_entry,
            roles=roles,
            debug=debug
        )
        core_type_map_for_fallback[analysis_entry['original_idx']] = current_core_type # Store for fallback
        if current_core_type is not None:
            word_core_types_list.append(current_core_type)
            original_indices_for_diagram.append(analysis_entry['original_idx'])
            filtered_tokens_for_diagram.append(analysis_entry['text'])
        else:
            logger.debug(f"  Token '{analysis_entry['text']}' (orig_idx {analysis_entry['original_idx']}) assigned None core type, excluding.")

    if not filtered_tokens_for_diagram:
        logger.error(f"No valid tokens with core types remained for diagram construction: '{sentence}'")
        return None, None, structure, tokens, analyses_details, roles

    logger.debug(f"Filtered Tokens for Diagram: {filtered_tokens_for_diagram}")
    logger.debug(f"Assigned Word Core Types: {[str(ct) if ct else 'None' for ct in word_core_types_list]}")
    logger.debug(f"Original Indices for Diagram Tokens: {original_indices_for_diagram}")

    # --- 3. Create DisCoCat Diagram (MODIFIED Logic V2.7.3) ---
    diagram: Optional[GrammarDiagram] = None
    diagram_creation_error = None
    try:
        logger.info(f"Creating DisCoCat diagram (V2.7.3 - Structure/OTHER Handling) for structure: {structure}...")
        safe_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0]) if sentence else "empty"

        # --- Decision Logic for Diagram Type ---
        attempted_diagram_type = None # Track which type we attempted

        # 1. Explicit Nominal Check
        if structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"]:
            logger.info(f"Attempting NOMINAL diagram creation based on structure '{structure}'.")
            attempted_diagram_type = "Nominal"
            diagram = create_nominal_sentence_diagram_v2_7(
                filtered_tokens_for_diagram, analyses_details, roles,
                word_core_types_list, original_indices_for_diagram, debug,
                output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_nominal"
            )
            if diagram is None:
                logger.warning(f"Nominal diagram creation failed for structure '{structure}'.")
                # Optional: Fallback to verbal if nominal fails AND a verb exists
                if roles.get('verb') is not None:
                   logger.info("Nominal failed, attempting verbal as fallback...")
                   attempted_diagram_type = "Verbal (Fallback)"
                   diagram = create_verbal_sentence_diagram_v3_7(
                       filtered_tokens_for_diagram, analyses_details, roles,
                       word_core_types_list, original_indices_for_diagram, debug,
                       output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_fallback"
                   )

        # 2. Explicit Verbal Check (or if Nominal wasn't applicable/failed and verb exists)
        elif structure not in ["ERROR", "OTHER"] or roles.get('verb') is not None:
            logger.info(f"Attempting VERBAL diagram creation based on structure '{structure}' or identified verb role.")
            attempted_diagram_type = "Verbal"
            diagram = create_verbal_sentence_diagram_v3_7(
                filtered_tokens_for_diagram, analyses_details, roles,
                word_core_types_list, original_indices_for_diagram, debug,
                output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal"
            )
            if diagram is None:
                 logger.warning(f"Verbal diagram creation failed for structure '{structure}'.")

        # 3. **MODIFIED:** Fallback for 'OTHER' structure using assigned types
        elif structure == "OTHER":
            logger.info(f"Structure is 'OTHER'. Checking assigned types for fallback strategy.")
            has_verb_functor = any(isinstance(ct, Box) and ct.name in ["VerbIntransFunctor", "VerbTransFunctor"] for ct in core_type_map_for_fallback.values())
            has_pred_functor = any(isinstance(ct, Box) and (ct.name == "AdjPredFunctor" or (hasattr(ct, 'name') and ct.name.startswith("NounPred_"))) for ct in core_type_map_for_fallback.values())
            found_verb_functor = False
            found_predicate_functor = False
            if has_verb_functor:
                logger.info("  'OTHER' structure has an assigned Verb Functor. Attempting VERBAL diagram.")
                attempted_diagram_type = "Verbal (OTHER - Assigned Verb Functor)"
                # Ensure roles['verb'] is set if it was None but a verb functor exists
                if roles.get('verb') is None:
                    for idx, core_type in core_type_map_for_fallback.items():
                        if isinstance(core_type, Box) and core_type.name in ["VerbIntransFunctor", "VerbTransFunctor"]:
                            roles['verb'] = idx
                            logger.warning(f"  Updated roles['verb'] to {idx} for 'OTHER' verbal attempt.")
                            break
                diagram = create_verbal_sentence_diagram_v3_7(
                       filtered_tokens_for_diagram, analyses_details, roles,
                       word_core_types_list, original_indices_for_diagram, debug,
                       output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_fallback"
                   )
                found_verb_functor = True
            elif has_pred_functor:
                logger.info("  'OTHER' structure has an assigned Predicate Functor. Attempting NOMINAL diagram.")
                attempted_diagram_type = "Nominal (OTHER - Assigned Pred Functor)"
                # Ensure roles['subject'] and roles['root'] (for predicate) are sensible
                if roles.get('subject') is None or roles.get('root') is None: # Or predicate_idx logic
                    # Heuristics to find subject and predicate for nominal 'OTHER'
                    potential_subj_idx, potential_pred_idx = None, None
                    for idx, core_type in core_type_map_for_fallback.items():
                        analysis = analysis_map_for_diagram_creation.get(idx)
                        if isinstance(core_type, Box) and (core_type.name == "AdjPredFunctor" or (hasattr(core_type, 'name') and core_type.name.startswith("NounPred_"))):
                            potential_pred_idx = idx
                        elif core_type == N and analysis and analysis.get('deprel') == 'nsubj': # A noun that is a subject
                            potential_subj_idx = idx

                    if potential_pred_idx is not None and roles.get('root') != potential_pred_idx : roles['root'] = potential_pred_idx # Predicate is often root
                    if potential_subj_idx is not None and roles.get('subject') is None: roles['subject'] = potential_subj_idx
                    logger.warning(f"  Updated roles for 'OTHER' nominal attempt: subject={roles.get('subject')}, root/predicate_anchor={roles.get('root')}")

                diagram = create_nominal_sentence_diagram_v2_7(
                                    filtered_tokens_for_diagram, 
                                    analyses_details, 
                                    roles, 
                                    temp_word_core_types_list, 
                                    original_indices_for_diagram, debug
                                )

            else:
                # NEW: If 'OTHER' and NO functor, try to dynamically make one (e.g. root noun/adj as predicate)
                logger.warning("  'OTHER' structure with NO pre-assigned functor. Attempting dynamic predicate identification.")
                root_idx = roles.get('root')
                if root_idx is not None and root_idx in core_type_map_for_fallback:
                    root_analysis = analysis_map_for_diagram_creation.get(root_idx)
                    root_core_type = core_type_map_for_fallback.get(root_idx)
                    # Try to find a subject for this root
                    current_subject_idx = roles.get('subject')
                    if current_subject_idx is None:
                        dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                        for dep_idx, d_rel in dependents_of_root:
                            dep_analysis = analysis_map_for_diagram_creation.get(dep_idx)
                            if d_rel == 'nsubj' and analyses_details and dep_idx < len(analyses_details) and analyses_details[dep_idx]['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                roles['subject'] = dep_idx; current_subject_idx = dep_idx; break

                    if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["ADJ", "NOUN", "X", "PROPN", "NUM"]:
                        logger.info(f"  Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}) in 'OTHER' structure.")
                        # Temporarily update core_type_map for this attempt
                        temp_core_type_map = core_type_map.copy() # or word_core_types_list if that's what's passed
                        # This requires word_core_types_list to be a dict or to find the right index
                        original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                        list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                        if list_idx_of_root is not None:
                            # Create a new word_core_types_list for this attempt
                            temp_word_core_types_list = list(word_core_types_list) # Make a mutable copy
                            if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE:
                                temp_word_core_types_list[list_idx_of_root] = ADJ_PRED_TYPE
                            else:
                                temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicNounPred_{root_analysis['lemma']}_{root_idx}", N, S)

                            logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate at root {root_idx}.")
                            attempted_diagram_type = "Nominal (OTHER - Dynamic Root Predicate)"
                            diagram = create_nominal_sentence_diagram_v2_7(
                                filtered_tokens_for_diagram, analyses_details, roles,
                                temp_word_core_types_list, # Pass the modified list
                                original_indices_for_diagram, debug, # ...
                            )
                        else:
                            logger.warning(f"  Root index {root_idx} not in original_indices_for_diagram, cannot dynamically assign predicate for 'OTHER'.")

                if diagram is None: # If dynamic predicate also failed
                    logger.warning("  'OTHER' structure with NO pre-assigned functor. Attempting dynamic predicate identification.")
                    root_idx = roles.get('root')
                    if root_idx is not None and root_idx in core_type_map_for_fallback:
                        # Use the analysis_map_for_diagram_creation defined earlier
                        root_analysis = analysis_map_for_diagram_creation.get(root_idx) 

                        current_subject_idx = roles.get('subject')
                        if current_subject_idx is None and root_analysis: # Check root_analysis exists
                            dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                            for dep_idx, d_rel in dependents_of_root:
                                # Use analysis_map_for_diagram_creation to get dep_analysis
                                dep_analysis = analysis_map_for_diagram_creation.get(dep_idx)
                                if d_rel == 'nsubj' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                    roles['subject'] = dep_idx; current_subject_idx = dep_idx
                                    logger.info(f"  Dynamic 'OTHER': Set subject to {dep_idx} ('{dep_analysis['text']}') for root predicate candidate '{root_analysis['text']}'.")
                                    break

                        if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["ADJ", "NOUN", "X", "PROPN", "NUM"]:
                            logger.info(f"  Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}) in 'OTHER' structure.")
                            # ... (rest of dynamic predicate assignment to temp_word_core_types_list) ...
                            # Ensure temp_word_core_types_list is correctly created and modified
                            original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                            list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                            if list_idx_of_root is not None and list_idx_of_root < len(word_core_types_list):
                                temp_word_core_types_list = list(word_core_types_list) 
                                if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE:
                                    temp_word_core_types_list[list_idx_of_root] = ADJ_PRED_TYPE
                                else:
                                    temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicNounPred_{root_analysis.get('lemma', root_analysis.get('text','unk'))}_{root_idx}", N, S)

                                logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate at root {root_idx}.")
                                # Pass analyses_details so create_nominal_sentence_diagram_v2_7 can build its own analysis_map
                                diagram = create_nominal_sentence_diagram_v2_7(
                                    filtered_tokens_for_diagram, 
                                    analyses_details, 
                                    roles, 
                                    temp_word_core_types_list, 
                                    original_indices_for_diagram, debug
                                )

            # Check for any assigned verb functor
            for idx, core_type in core_type_map_for_fallback.items():
                if isinstance(core_type, Box) and core_type.name in ["VerbIntransFunctor", "VerbTransFunctor"]:
                    logger.info(f"  Found assigned Verb Functor ('{core_type.name}' for token idx {idx}) in 'OTHER' structure. Attempting VERBAL diagram.")
                    attempted_diagram_type = "Verbal (OTHER Fallback - Assigned Type)"
                    # If main verb role wasn't set, set it to the first one found
                    if roles.get('verb') is None:
                        logger.warning(f"  Updating roles['verb'] heuristically to {idx} for diagram creation.")
                        roles['verb'] = idx
                        # Optional: Try to find subj/obj based on this verb? Risky.
                    diagram = create_verbal_sentence_diagram_v3_7(
                        filtered_tokens_for_diagram, analyses_details, roles,
                        word_core_types_list, original_indices_for_diagram, debug,
                        output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_other"
                    )
                    found_verb_functor = True
                    break

            # If no verb functor, check for any assigned predicate functor
            if not found_verb_functor:
                for idx, core_type in core_type_map_for_fallback.items():
                     if isinstance(core_type, Box) and (core_type.name == "AdjPredFunctor" or core_type.name.startswith("NounPred_")):
                         logger.info(f"  Found assigned Predicate Functor ('{core_type.name}' for token idx {idx}) in 'OTHER' structure. Attempting NOMINAL diagram.")
                         attempted_diagram_type = "Nominal (OTHER Fallback - Assigned Type)"
                         # Need to ensure subject is identified for nominal path
                         if roles.get('subject') is None:
                              # Try to find nsubj dependent of this predicate
                              dep_graph = roles.get('dependency_graph', {})
                              potential_subj_idx = None
                              if isinstance(dep_graph, dict):
                                   dependents = dep_graph.get(idx, [])
                                   for dep_idx, rel in dependents:
                                        if rel == 'nsubj':
                                             potential_subj_idx = dep_idx
                                             break
                              if potential_subj_idx is not None:
                                   logger.warning(f"  Updating roles['subject'] heuristically to {potential_subj_idx} for nominal diagram creation.")
                                   roles['subject'] = potential_subj_idx
                              else:
                                   logger.warning(f"  Found predicate functor but no subject identified for 'OTHER' structure. Nominal diagram likely to fail.")
                         # Update root if necessary? Maybe not needed if predicate is found.
                         # if roles.get('root') != idx : roles['root'] = idx

                         diagram = create_nominal_sentence_diagram_v2_7(
                             filtered_tokens_for_diagram, analyses_details, roles,
                             word_core_types_list, original_indices_for_diagram, debug,
                             output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_nominal_other"
                         )
                         found_predicate_functor = True
                         break

            # If neither found for 'OTHER'
            if not found_verb_functor and not found_predicate_functor:
                # logger.warning(f"  Structure is 'OTHER', but no assigned Verb or Predicate functor found. Skipping diagram creation.")
                attempted_diagram_type = "Skipped (OTHER - No Functor)"
                logger.warning("  'OTHER' structure with NO pre-assigned and USED functor. Attempting dynamic predicate identification.")
                root_idx = roles.get('root')
                analysis_map_from_roles = roles.get('analysis_map_for_diagram_creation', {}) # Get the map from roles
                potential_pred_idx = None
                potential_subj_idx = None
                if root_idx is not None:
                    root_analysis = analysis_map_from_roles.get(root_idx)
                    if root_analysis and root_analysis['upos'] in ["ADJ", "X", "NUM"]: # Potential predicate POS
                        dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                        for dep_idx, d_rel in dependents_of_root:
                            if d_rel == 'nsubj':
                                potential_subj_idx = dep_idx
                                potential_pred_idx = root_idx
                                logger.info(f"  Dynamic 'OTHER' (Attempt 1): Root '{root_analysis['text']}' (ADJ/X/NUM) is predicate, its nsubj '{analysis_map_from_roles.get(dep_idx,{}).get('text')}' is subject.")
                                break

                # Attempt 2: If root is NOUN/X, look for an ADJ/X dependent that could be the predicate.
                # Example: "عينُ الطفلِ زرقاءُ" (S51). Root="عينُ"(N/X). "زرقاءُ"(X) is nmod of "الطفلِ"(obj of "عينُ").
                # This requires finding "زرقاءُ" as predicate and "عينُ الطفلِ" as subject.
                if potential_pred_idx is None and root_analysis and root_analysis['upos'] in ["NOUN", "X"]:
                    # Iterate through all tokens to find a potential adjectival/nominal predicate
                    for token_idx, token_analysis_iter in analysis_map_from_roles.items():
                        if token_idx == root_idx: continue # Don't pick root as its own predicate here
                        if token_analysis_iter['upos'] in ["ADJ", "X", "NUM"]: # Potential predicate
                            # Check if this token is related to the root or its main arguments
                            # For S51: "زرقاءُ" (idx 2) head is "الطفلِ" (idx 1), head of "الطفلِ" is "عينُ" (idx 0, root)
                            head_of_candidate_pred = token_analysis_iter.get('head')
                            if head_of_candidate_pred is not None:
                                head_of_head = analysis_map_from_roles.get(head_of_candidate_pred, {}).get('head')
                                if head_of_candidate_pred == root_idx or head_of_head == root_idx or \
                                (roles.get('object') is not None and head_of_candidate_pred == roles.get('object')): # Predicate modifies object of root

                                    # If this is our predicate, the root (or an NP around it) is the subject
                                    potential_pred_idx = token_idx
                                    potential_subj_idx = root_idx # Assume root is subject, or head of subject NP
                                    logger.info(f"  Dynamic 'OTHER' (Attempt 2): Found predicate '{token_analysis_iter['text']}' (idx {token_idx}) for subject (around) root '{root_analysis['text']}'.")
                                    break

                # If we found a potential subject and predicate for nominal construction:
                if potential_pred_idx is not None and potential_subj_idx is not None:
                    roles['subject'] = potential_subj_idx # Update roles
                    # The dynamic predicate functor will be assigned to potential_pred_idx
                    pred_cand_analysis = analysis_map_from_roles.get(potential_pred_idx)
                    if pred_cand_analysis:
                        logger.info(f"  Dynamically assigning N->S functor to '{pred_cand_analysis['text']}' (idx {potential_pred_idx}) in 'OTHER' structure.")
                        original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                        list_idx_of_pred_cand = original_idx_to_list_idx.get(potential_pred_idx)

                        if list_idx_of_pred_cand is not None and list_idx_of_pred_cand < len(word_core_types_list):
                            temp_word_core_types_list = list(word_core_types_list) 
                            new_functor_name_base = pred_cand_analysis.get('lemma', pred_cand_analysis.get('text','unk'))
                            if pred_cand_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE: # ADJ_PRED_TYPE is N->S Box
                                temp_word_core_types_list[list_idx_of_pred_cand] = Box(f"DynamicAdjPred_{new_functor_name_base}_{potential_pred_idx}", N, S)
                            else: # NOUN, X, NUM as predicate
                                temp_word_core_types_list[list_idx_of_pred_cand] = Box(f"DynamicNounPred_{new_functor_name_base}_{potential_pred_idx}", N, S)

                            logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate '{get_diagram_repr(temp_word_core_types_list[list_idx_of_pred_cand])}' and subject idx {potential_subj_idx}.")
                            attempted_diagram_type = "Nominal (OTHER - Dynamic Subj/Pred)"
                            diagram = create_nominal_sentence_diagram_v2_7(
                                filtered_tokens_for_diagram, analyses_details, roles, 
                                temp_word_core_types_list, original_indices_for_diagram, debug,
                                hint_predicate_original_idx=potential_pred_idx # ***** NEW HINT *****
                            )
                        # ...
                """ if root_idx is not None and root_idx in core_type_map_for_fallback:
                    root_analysis = analysis_map_from_roles.get(root_idx) 
                    current_subject_idx = roles.get('subject')
                    if current_subject_idx is None and root_analysis:
                        logger.debug(f"  Dynamic 'OTHER': Root '{root_analysis['text']}' is predicate candidate. Searching for its subject.")
                        # Strategy 1: Look for 'nsubj' dependent of the root.
                        dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                        for dep_idx, d_rel in dependents_of_root:
                            dep_analysis = analysis_map_from_roles.get(dep_idx)
                            if d_rel == 'nsubj' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                roles['subject'] = dep_idx; current_subject_idx = dep_idx
                                logger.info(f"    Found subject (nsubj of root): '{dep_analysis['text']}' (idx {dep_idx}) for predicate '{root_analysis['text']}'.")
                                break
                        # Strategy 2: If root is ADJ/X and has no nsubj, look for a preceding N/PRON/DET that might be the subject.
                        if current_subject_idx is None and root_analysis['upos'] in ["ADJ", "X"] :
                            for i in range(root_idx - 1, -1, -1): # Look backwards
                                prev_token_analysis = analysis_map_from_roles.get(i)
                                if prev_token_analysis and prev_token_analysis['upos'] in ["NOUN", "PRON", "DET", "X"] and prev_token_analysis.get('head') == root_idx: # Check if it's a dependent or just preceding
                                    roles['subject'] = i; current_subject_idx = i
                                    logger.info(f"    Found potential preceding subject: '{prev_token_analysis['text']}' (idx {i}) for predicate '{root_analysis['text']}'.")
                                    break
                        # Strategy 3: For S81 "اللاعبانِ ماهرانِ .", "ماهرانِ" is root, "اللاعبانِ" is its nmod. This is unusual.
                        # If root is X/ADJ and subject is still None, check nmod as last resort.
                        if current_subject_idx is None and root_analysis['upos'] in ["ADJ", "X"]:
                            for dep_idx, d_rel in dependents_of_root:
                                dep_analysis = analysis_map_from_roles.get(dep_idx)
                                if d_rel == 'nmod' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                    roles['subject'] = dep_idx; current_subject_idx = dep_idx
                                    logger.info(f"    Found subject (nmod of root): '{dep_analysis['text']}' (idx {dep_idx}) for predicate '{root_analysis['text']}'.")
                                    break
                    if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["ADJ", "NOUN", "X", "PROPN", "NUM"]:
                        logger.info(f"  Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}) in 'OTHER' structure.")
                        original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                        list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                        if list_idx_of_root is not None and list_idx_of_root < len(word_core_types_list):
                            temp_word_core_types_list = list(word_core_types_list) 
                            new_functor_name_base = root_analysis.get('lemma', root_analysis.get('text','unk'))
                            if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE:
                                temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicAdjPred_{new_functor_name_base}_{root_idx}", N, S)
                            else:
                                temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicNounPred_{new_functor_name_base}_{root_idx}", N, S)

                            logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate at root {root_idx} ('{get_diagram_repr(temp_word_core_types_list[list_idx_of_root])}').")
                            attempted_diagram_type = "Nominal (OTHER - Dynamic Root Predicate)"
                            diagram = create_nominal_sentence_diagram_v2_7(
                                filtered_tokens_for_diagram, analyses_details, roles, 
                                temp_word_core_types_list, original_indices_for_diagram, debug
                            )
                        else:
                            logger.warning(f"  Root index {root_idx} not in original_indices_for_diagram or word_core_types_list, cannot dynamically assign predicate for 'OTHER'.")
 """
                if diagram is None: # If dynamic predicate also failed
                    logger.warning(f"  'OTHER' structure: All fallbacks failed. Skipping diagram creation.")
                    attempted_diagram_type = "Skipped (OTHER - All Fallbacks Failed)"


        # 4. Handle ERROR or unhandled cases
        else:
            logger.warning(f"Structure is '{structure}'. Cannot determine diagram type. Skipping diagram creation.")
            attempted_diagram_type = f"Skipped ({structure})"


        # --- Logging Outcome ---
        if diagram is None:
            # Log error only if we actually attempted a diagram type
            if attempted_diagram_type and not attempted_diagram_type.startswith("Skipped"):
                logger.error(f"Diagram creation ({attempted_diagram_type}) returned None for sentence '{sentence}'.")
        else:
            logger.info(f"Diagram ({attempted_diagram_type}) created successfully for '{sentence}'. Final Cod: {diagram.cod}")

    except Exception as e_diagram:
        logger.error(f"Exception during diagram creation phase for '{sentence}': {e_diagram}", exc_info=True)
        diagram_creation_error = str(e_diagram)

    # --- Steps 4 & 5 (Circuit Conversion & Return) ---
    # (Keep the rest of the function the same as in camel_test2.py v3.3)
    # ... (circuit conversion logic using selected_ansatz) ...

    # --- 4. Convert Diagram to Quantum Circuit ---
    circuit: Optional[QuantumCircuit] = None
    # Ensure diagram exists before proceeding
    if diagram is None:
        logger.error("Diagram is None after creation attempt, cannot proceed to circuit conversion.")
        # Return existing info, circuit and diagram are None
        return None, None, structure, tokens, analyses_details, roles

    try:
        logger.info(f"Converting diagram to quantum circuit using ansatz: {ansatz_choice}")
        # Define object map (adjust if you use different atomic types like P, ADJ etc.)
        ob_map = {N: 1, S: 1} # Basic map, assumes N=1 qubit, S=1 qubit

        selected_ansatz = None
        # --- Ansatz Selection (Ensure names match Lambeq's classes) ---
        if ansatz_choice.upper() == "IQP":
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
            logger.info(f"Using IQPAnsatz (layers={n_layers_iqp}, params/q={n_single_qubit_params_iqp})")
        elif ansatz_choice.upper() == "STRONGLY_ENTANGLING":
             # Determine number of qubits needed for the StrongAnsatz
             # This depends on the input type of the diagram (diagram.dom)
             # If diagram.dom is Ty(), it might represent 0 qubits, handle appropriately.
             num_qubits_required = len(diagram.dom) if diagram.dom else 1 # Default to 1 if dom is Ty()
             logger.info(f"Diagram domain requires {num_qubits_required} qubits for StronglyEntanglingAnsatz.")
             # Note: Lambeq's StronglyEntanglingAnsatz might take ob_map directly now. Check documentation.
             # If it requires num_qubits explicitly:
             # selected_ansatz = StronglyEntanglingAnsatz(num_qubits=num_qubits_required, n_layers=n_layers_strong, ranges=cnot_ranges)
             # If it takes ob_map:
             selected_ansatz = StronglyEntanglingAnsatz(ob_map=ob_map, n_layers=n_layers_strong, ranges=cnot_ranges)
             logger.info(f"Using StronglyEntanglingAnsatz (layers={n_layers_strong}, qubits based on ob_map)")
        elif ansatz_choice.upper() == "SPIDER":
            selected_ansatz = SpiderAnsatz(ob_map=ob_map, discard_qubits=discard_qubits_spider)
            logger.info(f"Using SpiderAnsatz (discard_qubits={discard_qubits_spider})")
        else:
            logger.warning(f"Unknown ansatz_choice: '{ansatz_choice}'. Defaulting to IQPAnsatz.")
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3) # Default IQP

        if selected_ansatz is None:
            raise ValueError("Ansatz object could not be created.")

        # Apply the ansatz to the diagram (already normalized in create_* functions)
        quantum_diagram = selected_ansatz(diagram)
        logger.info(f"Applied {ansatz_choice} ansatz to the diagram.")

        # --- Convert to Qiskit Circuit ---
        if PYTKET_QISKIT_AVAILABLE:
            logger.debug("Attempting conversion via Pytket...")
            tket_circ = quantum_diagram.to_tk()
            circuit = tk_to_qiskit(tket_circ)
            logger.info("Circuit conversion via Tket successful.")
        elif hasattr(quantum_diagram, 'to_qiskit'):
             logger.debug("Attempting direct conversion using Lambeq's to_qiskit...")
             circuit = quantum_diagram.to_qiskit()
             logger.info("Direct circuit conversion to Qiskit successful.")
        else:
             logger.error("No available method (Tket or direct) to convert Lambeq diagram to Qiskit circuit.")
             circuit = None # Ensure circuit is None if conversion fails

    except NotImplementedError as e_nf_main:
        # This might occur if normal_form was called implicitly or if ansatz fails
        logger.error(f"Diagram normalization or ansatz application failed: {e_nf_main}", exc_info=True)
        # Return the diagram that was created, but circuit is None
        return None, diagram, structure, tokens, analyses_details, roles
    except Exception as e_circuit_outer:
        logger.error(f"Exception during circuit conversion: {e_circuit_outer}", exc_info=True)
        # Return the diagram, but circuit is None
        return None, diagram, structure, tokens, analyses_details, roles

    if circuit is None:
        logger.error("Circuit conversion resulted in None.")
        # Return diagram, circuit is None
        return None, diagram, structure, tokens, analyses_details, roles

    # --- 5. Return Results ---
    logger.info(f"Successfully processed sentence '{sentence}' into circuit.")
    return circuit, diagram, structure, tokens, analyses_details, roles


# ==================================
# Visualization Functions (Unchanged)
# ==================================
def visualize_diagram(diagram, save_path=None):
    """Visualizes a Lambeq diagram."""
    if diagram is None or not hasattr(diagram, 'draw'): return None
    try:
        figsize = (max(10, len(diagram.boxes) * 0.8), max(6, len(diagram.dom) * 0.5 + 2))
        ax = diagram.draw(figsize=figsize, fontsize=9, aspect='auto')
        fig = ax.figure
        if save_path:
            try: os.makedirs(os.path.dirname(save_path), exist_ok=True); fig.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved diagram to {save_path}")
            except Exception as e_save: logger.error(f"Failed to save diagram: {e_save}")
        return fig
    except Exception as e: logger.error(f"Diagram visualization error: {e}", exc_info=True); plt.close(); return None

def visualize_circuit(circuit, save_path=None):
    """Visualizes a Qiskit circuit."""
    if circuit is None or not hasattr(circuit, 'draw'): return None
    try:
        fig = circuit.draw(output='mpl', fold=-1, scale=0.7)
        if fig:
             fig.set_size_inches(12, max(6, circuit.num_qubits * 0.4))
             plt.tight_layout()
             if save_path:
                 try: os.makedirs(os.path.dirname(save_path), exist_ok=True); fig.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved circuit to {save_path}")
                 except Exception as e_save: logger.error(f"Failed to save circuit: {e_save}")
             return fig
        else: logger.warning("Circuit draw failed."); return None
    except Exception as e: logger.error(f"Circuit visualization error: {e}", exc_info=True); plt.close(); return None

# ==================================
# Main Execution / Testing (Updated for V3.3)
# ==================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running camel_test2.py (V3.3) directly for testing...")

    test_sentences = [
        "يقرأ الولد الكتاب",               # VSO
        "الولد يقرأ الكتاب",               # SVO
        "البيت كبير",                     # NOMINAL (Adj Pred)
        "هذا الرجل طبيب",                 # NOMINAL (Noun Pred + D
    ]