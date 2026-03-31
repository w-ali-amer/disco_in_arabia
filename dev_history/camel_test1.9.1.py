# -*- coding: utf-8 -*-
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
from pytket.extensions.qiskit import tk_to_qiskit
try:
    PYTKET_QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: pytket-qiskit extension not found.")
    print("Please install it: pip install pytket-qiskit")
    PYTKET_QISKIT_AVAILABLE = False

ARABIC_DIACRITICS = set("ًٌٍَُِّْ")
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

CAMEL_ANALYZER = None
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    db_path = MorphologyDB.builtin_db()
    CAMEL_ANALYZER = Analyzer(db_path)
    logger.info("CAMeL Tools Analyzer initialized successfully.")
except ImportError:
    logger.warning("CAMeL Tools not found. Morphological feature extraction will be limited.")
except Exception as e:
    logger.warning(f"Error initializing CAMeL Tools Analyzer: {e}. Morphological feature extraction will be limited.")

try:
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse,mwt', verbose=False, use_gpu=False) # Added mwt
    STANZA_AVAILABLE = True
    logger.info("Stanza pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Stanza: {e}", exc_info=True)
    STANZA_AVAILABLE = False

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE 
C = AtomicType.CONJUNCTION          
ADJ = AtomicType.NOUN_PHRASE       

logger.info(f"Initial N: {str(N)}, Type: {type(N)}")
logger.info(f"Initial S: {str(S)}, Type: {type(S)}")

# --- Pre-defined Functorial Types - Defined Explicitly as Boxes (Diagrams) ---
ADJ_MOD_TYPE: Optional[Box] = None
ADJ_PRED_TYPE: Optional[Box] = None
DET_TYPE: Optional[Box] = None
PREP_FUNCTOR_TYPE: Optional[Box] = None
VERB_INTRANS_TYPE: Optional[Box] = None
VERB_TRANS_TYPE: Optional[Box] = None
S_MOD_BY_N: Optional[Box] = None # For PP attachment to Sentence
N_MOD_BY_N: Optional[Box] = None
ADV_FUNCTOR_TYPE: Optional[Box] = None # For Adverbs/Adjuncts modifying sentence: S -> S

logger.info("Attempting to define global functorial types...")
try:
    logger.debug("Defining ADJ_MOD_TYPE...")
    ADJ_MOD_TYPE = Box("AdjModFunctor", N, N)
    logger.info(f"  ADJ_MOD_TYPE: {str(ADJ_MOD_TYPE)}")

    logger.debug("Defining ADJ_PRED_TYPE...")
    ADJ_PRED_TYPE = Box("AdjPredFunctor", N, S)
    logger.info(f"  ADJ_PRED_TYPE: {str(ADJ_PRED_TYPE)}")

    logger.debug("Defining DET_TYPE...")
    DET_TYPE = Box("DetFunctor", N, N)
    logger.info(f"  DET_TYPE: {str(DET_TYPE)}")
    
    _intermediate_prep_type: Ty = N 
    logger.info(f"  Defined _intermediate_prep_type for prepositions as: {str(_intermediate_prep_type)}")
    
    logger.debug("Defining PREP_FUNCTOR_TYPE...")
    PREP_FUNCTOR_TYPE = Box("PrepFunctor", N, _intermediate_prep_type) # N -> N
    logger.info(f"  PREP_FUNCTOR_TYPE: {str(PREP_FUNCTOR_TYPE)} (Dom: {PREP_FUNCTOR_TYPE.dom}, Cod: {PREP_FUNCTOR_TYPE.cod})")
    
    logger.debug("Defining VERB_INTRANS_TYPE...")
    VERB_INTRANS_TYPE = Box("VerbIntransFunctor", N, S) # Takes N, returns S
    logger.info(f"  VERB_INTRANS_TYPE: {str(VERB_INTRANS_TYPE)}")

    logger.debug("Defining VERB_TRANS_TYPE...")
    VERB_TRANS_TYPE = Box("VerbTransFunctor", N @ N, S) # Takes N @ N, returns S
    logger.info(f"  VERB_TRANS_TYPE: {str(VERB_TRANS_TYPE)}")

    # --- Specific try-except for S_MOD_BY_N ---
    logger.debug("Defining S_MOD_BY_N (for PP attachment)...")
    try:
        S_MOD_BY_N = Box("S_mod_by_N", S @ N, S) # Takes S and N (PP), returns S
        logger.info(f"  Successfully defined S_MOD_BY_N: {str(S_MOD_BY_N)}")
    except BaseException as e_smod: # Catch more general errors for this specific definition
        logger.critical(f"CRITICAL ERROR defining S_MOD_BY_N specifically: {e_smod}", exc_info=True)
        S_MOD_BY_N = None # Ensure it's None if this specific definition fails

    logger.debug("Defining N_MOD_BY_N (for PP attachment to Noun)...")
    try:
        N_MOD_BY_N = Box("N_mod_by_N", N @ N, N) # Takes N and N (PP), returns N
        logger.info(f"  Successfully defined N_MOD_BY_N: {str(N_MOD_BY_N)}")
    except BaseException as e_nmod:
        logger.critical(f"CRITICAL ERROR defining N_MOD_BY_N specifically: {e_nmod}", exc_info=True)
        N_MOD_BY_N = None
    try:
        ADV_FUNCTOR_TYPE = Box("AdvFunctor", S, S) # Takes S, returns S
        logger.info(f"  Successfully defined ADV_FUNCTOR_TYPE: {str(ADV_FUNCTOR_TYPE)}")
    except BaseException as e_adv:
        logger.critical(f"CRITICAL ERROR defining ADV_FUNCTOR_TYPE specifically: {e_adv}", exc_info=True)
        ADV_FUNCTOR_TYPE = None
    if S_MOD_BY_N is None: 
        logger.error("S_MOD_BY_N is None immediately after its specific try-except block during global definitions.")
    if N_MOD_BY_N is None:
        logger.error("N_MOD_BY_N is None immediately after its specific try-except block during global definitions.")
    if ADV_FUNCTOR_TYPE is None:
        logger.error("ADV_FUNCTOR_TYPE is None immediately after its specific try-except block during global definitions.")
    
    logger.info(">>> Explicit Box definitions for functorial types completed (status of S_MOD_BY_N logged above). <<<")

except Exception as e_box_def_outer: # Outer catch-all for other definitions
    logger.critical(f"CRITICAL ERROR during outer block of global Box type definitions (excluding S_MOD_BY_N specific try-catch): {e_box_def_outer}", exc_info=True)
    # Reset other types if this outer block fails
    ADJ_MOD_TYPE = ADJ_MOD_TYPE or None 
    ADJ_PRED_TYPE = ADJ_PRED_TYPE or None
    DET_TYPE = DET_TYPE or None
    PREP_FUNCTOR_TYPE = PREP_FUNCTOR_TYPE or None
    VERB_INTRANS_TYPE = VERB_INTRANS_TYPE or None
    VERB_TRANS_TYPE = VERB_TRANS_TYPE or None
    if S_MOD_BY_N is None: # If it wasn't set by its specific try-catch, ensure it remains None
        logger.debug("S_MOD_BY_N remains None due to outer exception block.")
# ==================================
# Linguistic Analysis Function (MODIFIED to include feats_dict)
# ==================================
def parse_feats_string(feats_str: Optional[str]) -> Dict[str, str]:
    """Parses a Stanza-style features string (e.g., "Case=Nom|Gender=Masc") into a dict."""
    if not feats_str:
        return {}
    feats_dict = {}
    try:
        pairs = feats_str.split('|')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                feats_dict[key] = value
            else:
                # Handle boolean features or features without explicit values if necessary
                # For now, we only parse key=value pairs
                logger.debug(f"Skipping feature part without '=': {pair} in {feats_str}")
    except Exception as e:
        logger.warning(f"Could not parse features string '{feats_str}': {e}")
    return feats_dict

def analyze_arabic_sentence_with_morph(sentence: str, debug: bool = True) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    if not STANZA_AVAILABLE:
        logger.error("Stanza is not available or failed to initialize.")
        return [], [], "ERROR", {}
    if not sentence or not sentence.strip():
        logger.warning("Received empty sentence for analysis.")
        return [], [], "OTHER", {}
    try:
        doc = nlp(sentence)
    except Exception as e_nlp:
        logger.error(f"Stanza processing failed for sentence: '{sentence}'", exc_info=True)
        return [], [], "ERROR", {}

    processed_tokens_texts = []
    processed_analyses = [] 
    roles_dict = {"verb": None, "subject": None, "object": None, "root": None, "dependency_graph": {}}

    if not doc.sentences:
        logger.warning(f"Stanza did not find any sentences in: '{sentence}'")
        return [], [], "OTHER", roles_dict
    sent = doc.sentences[0] 
    for i, word in enumerate(sent.words):
        processed_tokens_texts.append(word.text)
        head_idx = word.head - 1 if word.head > 0 else -1 
        feats_dict = parse_feats_string(word.feats)
        if CAMEL_ANALYZER and not feats_dict.get('case') and not feats_dict.get('gen') and not feats_dict.get('num'): 
            try:
                camel_analysis_list = CAMEL_ANALYZER.analyze(word.text)
                if camel_analysis_list:
                    camel_feats_str = camel_analysis_list[0].get('feat')
                    camel_feats_parsed = parse_feats_string(camel_feats_str)
                    for k, v in camel_feats_parsed.items():
                        if k not in feats_dict: 
                            feats_dict[k] = v
                    if camel_feats_parsed:
                         logger.debug(f"Enriched token '{word.text}' with CAMeL feats: {camel_feats_parsed}")
            except Exception as e_camel_enrich:
                logger.warning(f"Error during CAMeL enrichment for '{word.text}': {e_camel_enrich}")
        analysis_entry = {
            "text": word.text, "lemma": word.lemma if word.lemma else word.text,
            "upos": word.upos, "deprel": word.deprel, "head": head_idx, "feats_dict": feats_dict, "original_idx": i
        }
        processed_analyses.append(analysis_entry)
        roles_dict["dependency_graph"][i] = [] 
    for i, analysis_entry in enumerate(processed_analyses):
        dep_rel = analysis_entry["deprel"]
        head_idx = analysis_entry["head"]
        if head_idx >= 0 and head_idx < len(processed_tokens_texts):
            if head_idx not in roles_dict["dependency_graph"]:
                roles_dict["dependency_graph"][head_idx] = []
            roles_dict["dependency_graph"][head_idx].append((i, dep_rel))
        elif head_idx != -1:
            logger.warning(f"Invalid head index {head_idx} for token {i} ('{processed_tokens_texts[i]}'). Skipping dependency edge.")
    if debug:
        logger.debug("\nParsed sentence with dependencies & features:")
        for i, entry in enumerate(processed_analyses):
            logger.debug(f"  {i}: Token='{entry['text']}', Lemma='{entry['lemma']}', POS='{entry['upos']}', Dep='{entry['deprel']}', Head={entry['head']}, Feats={entry['feats_dict']}")
    
    # Determine Root and Verb
    for i, entry in enumerate(processed_analyses):
        if entry["deprel"] == "root" or entry["head"] == -1: 
            roles_dict["root"] = i
            if entry["upos"] == "VERB":
                roles_dict["verb"] = i
            break
    if roles_dict["root"] is None and len(processed_analyses) > 0:
        roles_dict["root"] = 0 
        if processed_analyses[0]["upos"] == "VERB":
            roles_dict["verb"] = 0

    if roles_dict["verb"] is None: 
        potential_verbs = [i for i, entry in enumerate(processed_analyses) if entry["upos"] == "VERB"]
        if potential_verbs:
            roles_dict["verb"] = potential_verbs[0]
            if roles_dict["root"] is None : roles_dict["root"] = potential_verbs[0] 

    # Determine Subject and Object based on the identified verb
    if roles_dict["verb"] is not None:
        verb_original_idx = roles_dict["verb"]
        for i, entry in enumerate(processed_analyses):
            if entry["head"] == verb_original_idx:
                if entry["deprel"] in ["nsubj", "csubj"] and roles_dict["subject"] is None: 
                    roles_dict["subject"] = i
                elif entry["deprel"] in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and roles_dict["object"] is None: 
                    roles_dict["object"] = i
    
    # Fallback for subject if not directly linked to verb but to root
    if roles_dict["subject"] is None and roles_dict["root"] is not None:
        root_original_idx = roles_dict["root"]
        for i, entry in enumerate(processed_analyses):
            if entry["head"] == root_original_idx and entry["deprel"] in ["nsubj", "csubj"]:
                 if roles_dict["subject"] is None: roles_dict["subject"] = i; break
    
    # If still no subject, and there's a verb, check for nouns immediately following a VSO verb
    if roles_dict["subject"] is None and roles_dict["verb"] is not None:
        verb_original_idx = roles_dict["verb"]
        if verb_original_idx + 1 < len(processed_analyses):
            potential_subj_entry = processed_analyses[verb_original_idx + 1]
            if potential_subj_entry["upos"] in ["NOUN", "PROPN", "PRON"] and potential_subj_entry["deprel"] != "obj": # Avoid taking obj as subj
                # A simple VSO heuristic: if the word after verb is Noun/Pron and not already obj
                roles_dict["subject"] = verb_original_idx + 1
                logger.debug(f"VSO Heuristic: Assigning token {verb_original_idx+1} ('{potential_subj_entry['text']}') as subject for verb {verb_original_idx}.")


    verb_idx = roles_dict.get("verb") 
    subj_idx = roles_dict.get("subject")
    obj_idx = roles_dict.get("object")
    structure_type = "OTHER" 
    num_verbs = len([i for i, entry in enumerate(processed_analyses) if entry["upos"] == "VERB"])

    if verb_idx is not None:
        if subj_idx is not None:
            if obj_idx is not None: 
                if subj_idx < verb_idx and verb_idx < obj_idx: structure_type = "SVO"
                elif verb_idx < subj_idx and verb_idx < obj_idx : structure_type = "VSO"
                elif verb_idx < obj_idx and obj_idx < subj_idx : structure_type = "VOS"
                else: structure_type = "VERBAL_COMPLEX_ORDER" 
            else: 
                if subj_idx < verb_idx: structure_type = "SV"
                elif verb_idx < subj_idx: structure_type = "VS"
                else: structure_type = "VERBAL_UNORDERED_SUBJ"
        else: # Verb but no subject found by rules
            structure_type = "VERBAL_NO_EXPLICIT_SUBJ" 
        if num_verbs > 1 and not structure_type.startswith("COMPLEX_"):
             structure_type = "COMPLEX_" + structure_type
    elif subj_idx is not None: 
        is_nominal_candidate = False
        if roles_dict["root"] is not None:
            root_pos = processed_analyses[roles_dict["root"]]["upos"]
            if root_pos in ["NOUN", "PROPN", "ADJ", "PRON"]:
                if roles_dict["root"] != subj_idx: is_nominal_candidate = True
                else:
                    for i, entry in enumerate(processed_analyses):
                        if entry["head"] == subj_idx and entry["upos"] in ["ADJ", "NOUN", "PROPN"]:
                            is_nominal_candidate = True; break
        if is_nominal_candidate: structure_type = "NOMINAL"
        else: structure_type = "SUBJ_NO_VERB_OTHER" 
    elif num_verbs > 1 : structure_type = "COMPLEX_NO_MAIN_VERB_ID"
    
    roles_dict["structure"] = structure_type 
    if debug:
        logger.debug(f"\nDetected structure: {structure_type}")
        logger.debug(f"  Verb index: {verb_idx} ({processed_tokens_texts[verb_idx] if verb_idx is not None else 'None'})")
        logger.debug(f"  Subject index: {subj_idx} ({processed_tokens_texts[subj_idx] if subj_idx is not None else 'None'})")
        logger.debug(f"  Object index: {obj_idx} ({processed_tokens_texts[obj_idx] if obj_idx is not None else 'None'})")
        logger.debug(f"  Root index: {roles_dict.get('root')} ({processed_tokens_texts[roles_dict.get('root')] if roles_dict.get('root') is not None else 'None'})")
    return processed_tokens_texts, processed_analyses, structure_type, roles_dict



# ==================================
# DisCoCat Type Assignment (assign_discocat_types_v2 from camel_test3.py)
# This function determines the *core grammatical type*. Features are handled separately.
# No major changes needed here for now, but ensure it uses 'upos' from the new 'analyses' structure.
# ==================================
def assign_discocat_types_v2(
    pos: str, dep_rel: str, token_text: str, lemma: str,
    is_verb: bool = False, verb_takes_subject: bool = False, verb_takes_object: bool = False,
    is_nominal_pred: bool = False, is_adj_modifier: bool = False, debug: bool = True
) -> Union[Ty, GrammarDiagram, None]: 
    logger.debug(f"Assigning type for '{token_text}', POS: {pos}, Lemma: '{lemma}', DepRel: '{dep_rel}', is_verb: {is_verb}, is_nominal_pred: {is_nominal_pred}, is_adj_mod: {is_adj_modifier}")
    logger.debug(f"  Global types check: VERB_TRANS_TYPE is {type(VERB_TRANS_TYPE).__name__}, PREP_FUNCTOR_TYPE is {type(PREP_FUNCTOR_TYPE).__name__}, S_MOD_BY_N is {type(S_MOD_BY_N).__name__}, DET_TYPE is {type(DET_TYPE).__name__}")
    
    assigned_entity: Union[Ty, GrammarDiagram, None] = None

    # 1. Handle explicit verb role
    if is_verb:
        if verb_takes_subject and verb_takes_object: assigned_entity = VERB_TRANS_TYPE
        elif verb_takes_subject: assigned_entity = VERB_INTRANS_TYPE
        else: assigned_entity = VERB_INTRANS_TYPE 
        logger.debug(f"  Decision (is_verb): Verb Type (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
        if assigned_entity is None: logger.error(f"  CRITICAL: Global verb type is None for '{token_text}'.")
        elif not isinstance(assigned_entity, Box): logger.error(f"  CRITICAL: Verb type {str(assigned_entity)} is not a Box!")
    
    # 2. Handle explicit nominal predicate role
    elif is_nominal_pred:
        if pos == "ADJ": assigned_entity = ADJ_PRED_TYPE
        elif pos in ["NOUN", "PROPN"]: assigned_entity = Box(f"NounPred_{lemma}", N, S)
        else: 
            logger.warning(f"Nominal predicate '{token_text}' has POS '{pos}'. Assigning NounPred Box based on role.")
            assigned_entity = Box(f"NounPred_{lemma}", N, S)
        logger.debug(f"  Decision (is_nominal_pred): Nominal Predicate Type (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
        if assigned_entity is None and pos == "ADJ": logger.error(f"  CRITICAL: ADJ_PRED_TYPE global is None for '{token_text}'.")
        elif not isinstance(assigned_entity, Box): logger.error(f"  CRITICAL: Nominal Predicate type {str(assigned_entity)} is not a Box!")

    # 3. Handle explicit adjective modifier role
    elif is_adj_modifier and pos == "ADJ":
        assigned_entity = ADJ_MOD_TYPE
        logger.debug(f"  Decision (is_adj_modifier): ADJ_MOD_TYPE (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
        if assigned_entity is None: logger.error(f"  CRITICAL: ADJ_MOD_TYPE global is None for '{token_text}'.")

    # 4. NEW: Prioritize dep_rel for subjects/objects/obliques if not already typed as verb/predicate
    elif dep_rel in ['nsubj', 'obj', 'iobj', 'dobj', 'nsubj:pass', 'csubj', 'obl', 'obl:arg'] : 
        if pos in ["NOUN", "PROPN", "PRON", "X"]: 
            assigned_entity = N 
            logger.debug(f"  Decision (dep_rel priority): N (due to dep_rel '{dep_rel}') for token '{token_text}' with POS '{pos}'")
        # If it's an oblique/arg but ADJ (e.g., adverbial adj), handle below. If ADP, handle below.
    
    # 5. Handle standard POS tags for arguments if not caught by dep_rel and not a functor
    elif pos in ["NOUN", "PROPN", "PRON"]: 
        assigned_entity = N 
        logger.debug(f"  Decision (POS): N (due to POS '{pos}') for token '{token_text}'")

    # 6. Handle "ADJ" based on dep_rel if not already handled
    elif pos == "ADJ":
        if dep_rel == "amod": # Adjectival modifier of a noun
            assigned_entity = ADJ_MOD_TYPE
            logger.debug(f"  Decision (amod): ADJ_MOD_TYPE by amod (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
            if assigned_entity is None: logger.error(f"  CRITICAL: ADJ_MOD_TYPE global is None for '{token_text}'.")
        elif dep_rel in ['advmod', 'obl', 'xcomp']: # Adjective acting adverbially or as complement
            assigned_entity = ADV_FUNCTOR_TYPE # Treat as S->S modifier for now
            logger.debug(f"  Decision (Adj as Adv/Obl/Xcomp): ADV_FUNCTOR_TYPE (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
            if assigned_entity is None: logger.error(f"  CRITICAL: ADV_FUNCTOR_TYPE global is None for '{token_text}'.")
        else: # Fallback for ADJ with other dep_rel
            logger.warning(f"Unhandled ADJ dep_rel: '{dep_rel}' for '{token_text}'. Defaulting to N.")
            assigned_entity = N

    # 7. Handle Determiners (DET) - Check if it's modifying something or acting as subject
    elif pos == "DET":
        logger.debug(f"  POS is DET. Global DET_TYPE is: {str(DET_TYPE)} of type {type(DET_TYPE)}")
        if dep_rel == 'det': # Clearly modifying a noun
            if DET_TYPE is not None and isinstance(DET_TYPE, Box):
                assigned_entity = DET_TYPE # N -> N
                logger.debug(f"  Decision (det): DET_TYPE assigned (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
            else: logger.error(f"  CRITICAL: DET_TYPE global is None or not a Box for '{token_text}'. Defaulting to N.") ; assigned_entity = N
        elif dep_rel == 'nsubj': # Acting as subject pronoun
             assigned_entity = N
             logger.debug(f"  Decision (DET as nsubj): N assigned for token '{token_text}'.")
        else: # Fallback for DET with other dep_rel
             logger.warning(f"Unhandled DET dep_rel: '{dep_rel}' for '{token_text}'. Defaulting to N.")
             assigned_entity = N

    # 8. Handle other specific functor POS tags
    elif pos == "ADP":
        assigned_entity = PREP_FUNCTOR_TYPE # N -> N
        logger.debug(f"  Decision (POS): PREP_FUNCTOR_TYPE (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
        if assigned_entity is None: logger.error(f"  CRITICAL: PREP_FUNCTOR_TYPE global is None for '{token_text}'.")
        elif not (isinstance(assigned_entity, Box) and assigned_entity.dom == N and assigned_entity.cod == N): logger.error(f"  CRITICAL: PREP_FUNCTOR_TYPE is not Box(N,N)!")
    elif pos == "ADV": 
        assigned_entity = ADV_FUNCTOR_TYPE # S -> S
        logger.debug(f"  Decision (POS): ADV_FUNCTOR_TYPE (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
        if assigned_entity is None: logger.error(f"  CRITICAL: ADV_FUNCTOR_TYPE global is None for '{token_text}'.")
    elif pos == "NUM": 
        assigned_entity = Box("NumFunctor", N, N) # Treat as N->N modifier for now
        logger.debug(f"  Decision (POS): Dynamic NumFunctor Box (Value: {str(assigned_entity)}, Type: {type(assigned_entity)})")
    
    # 9. Handle explicitly ignored POS tags or fallback if still None
    if assigned_entity is None:
        if pos in ["PUNCT", "SYM", "PART", "CCONJ", "SCONJ", "AUX", "INTJ", "X"]:
            logger.debug(f"  Decision (POS ignore): Explicitly None for POS {pos}")
        else: 
            logger.warning(f"Unhandled POS/DepRel combination: POS='{pos}', DepRel='{dep_rel}' for '{token_text}'. Defaulting to N.")
            assigned_entity = N 

    if debug: logger.debug(f"  TypeAssignV2 FINAL RESULT for '{token_text}': Entity='{str(assigned_entity)}', Instance='{type(assigned_entity).__name__}', IsDiagram: {isinstance(assigned_entity, GrammarDiagram)}")
    return assigned_entity

# Diagram Creation Functions (REVISED - V3.1 with Feature Boxes)
# ==================================

# `find_subwire_index_v2` and `apply_cup_at_indices_v2` from camel_test3.py
# might need adjustments if the wire map changes significantly.
# For now, let's assume they are adaptable or we'll refine them based on the new wire map.

def find_subwire_index_v2(main_wire_type: Ty, sub_type: Ty, wire_start_index: int, path: List[int] = None) -> Optional[Tuple[int, List[int]]]:
    """(Copied from camel_test3.py, may need review based on usage)"""
    if path is None: path = []
    if main_wire_type == sub_type:
        return wire_start_index, path
    if not hasattr(main_wire_type, 'objects') or not main_wire_type.objects:
        return None
    current_offset = wire_start_index
    for i, t in enumerate(main_wire_type.objects):
        result = find_subwire_index_v2(t, sub_type, current_offset, path + [i])
        if result:
            return result
        current_offset += len(t)
    return None

def apply_cup_at_indices_v3(diagram: GrammarDiagram,
                          wire1_abs_start_idx: int, wire1_type: Ty, wire1_orig_tok_idx: int,
                          wire2_abs_start_idx: int, wire2_type: Ty, wire2_orig_tok_idx: int,
                          current_full_wire_map: List[Dict[str, Any]] # New wire map structure
                          ) -> Optional[Tuple[GrammarDiagram, List[Dict[str, Any]]]]:
    """
    Applies swaps to bring wires adjacent and then applies Cup. V3 for new wire map.
    Wires to be cupped are main grammatical wires. Feature wires are permuted alongside.

    Args:
        diagram: The current diagram.
        wire1_abs_start_idx, wire1_type, wire1_orig_tok_idx: Info for the first main grammatical wire.
        wire2_abs_start_idx, wire2_type, wire2_orig_tok_idx: Info for the second main grammatical wire.
        current_full_wire_map: List of dicts, each describing a token unit (main + features).
                               Example: {'orig_idx': 0, 'token': 'الولد',
                                         'main_box_type': N, 'main_abs_start': 0, 'main_len': 1,
                                         'feature_boxes_types': [Ty(), Ty()], 'feat_abs_start': 1, 'feat_len': 0,
                                         'unit_abs_start': 0, 'unit_len': 1}
                                         (feat_len is sum of len(Ty()) which is 0)

    Returns:
        Tuple (new_diagram, new_wire_map) or None on failure.
    """
    logger.debug(f"Attempting Cup V3: Wire1(TokIdx {wire1_orig_tok_idx}) type {wire1_type} at abs_idx {wire1_abs_start_idx}, "
                 f"Wire2(TokIdx {wire2_orig_tok_idx}) type {wire2_type} at abs_idx {wire2_abs_start_idx} on Cod={diagram.cod}")

    # 1. Check Adjoints for the main grammatical types
    if not (wire1_type.r == wire2_type or wire1_type == wire2_type.r):
        logger.error(f"Cannot apply Cup: Main types {wire1_type} and {wire2_type} are not adjoints.")
        return None
    if wire1_abs_start_idx == wire2_abs_start_idx: # Should not happen if distinct wires
        logger.error("Cannot cup a wire with itself (same start index).")
        return None

    # 2. Calculate Permutation for individual wires
    n_wires_total = len(diagram.cod) # Total number of elementary wires (Ty() counts as 0, N counts as 1)
    len1 = len(wire1_type) # Length of the first main grammatical wire block
    len2 = len(wire2_type) # Length of the second main grammatical wire block

    # Ensure index1 refers to the leftmost of the two main grammatical wires
    left_main_abs_idx, right_main_abs_idx = min(wire1_abs_start_idx, wire2_abs_start_idx), max(wire1_abs_start_idx, wire2_abs_start_idx)
    left_main_len, right_main_len = (len1, len2) if wire1_abs_start_idx < wire2_abs_start_idx else (len2, len1)
    left_main_type, right_main_type = (wire1_type, wire2_type) if wire1_abs_start_idx < wire2_abs_start_idx else (wire2_type, wire1_type)

    # Check indices are valid
    if not (0 <= left_main_abs_idx < right_main_abs_idx < n_wires_total and \
            left_main_abs_idx + left_main_len <= right_main_abs_idx and \
            right_main_abs_idx + right_main_len <= n_wires_total):
        logger.error(f"Invalid indices/lengths for cup: LeftMain({left_main_abs_idx}, len {left_main_len}), "
                     f"RightMain({right_main_abs_idx}, len {right_main_len}) on total {n_wires_total} wires.")
        return None

    # Create the permutation array for Swap
    perm = []
    # These are blocks of individual wires making up the main grammatical types
    left_main_block_indices = list(range(left_main_abs_idx, left_main_abs_idx + left_main_len))
    right_main_block_indices = list(range(right_main_abs_idx, right_main_abs_idx + right_main_len))
    
    indices_to_be_cupped = set(left_main_block_indices + right_main_block_indices)

    # Wires not part of the cup, to be placed first
    for i in range(n_wires_total):
        if i not in indices_to_be_cupped:
            perm.append(i)
    
    # Append the main grammatical wire blocks that will be cupped, in order
    perm.extend(left_main_block_indices)
    perm.extend(right_main_block_indices)

    logger.debug(f"Calculated individual wire permutation for Swap: {perm}")
    if len(perm) != n_wires_total:
        logger.error(f"Permutation length {len(perm)} does not match total wires {n_wires_total}.")
        return None

    # 3. Apply Permutation (Swap layer)
    try:
        if perm != list(range(n_wires_total)): # Only apply if non-trivial
            swap_layer = Swap(diagram.cod, perm) # diagram.cod is the type of all wires
            permuted_diagram = diagram >> swap_layer
            logger.debug(f"Applied Swap. Diagram after permute cod: {permuted_diagram.cod}")
        else:
            permuted_diagram = diagram
            logger.debug("Permutation is identity, skipping Swap.")
    except Exception as e_perm:
        logger.error(f"Failed to apply Swap permutation {perm} on cod {diagram.cod}: {e_perm}", exc_info=True)
        return None

    # 4. Apply Cup
    try:
        # The cupped wires (left_main_type, right_main_type) are now at the end of permuted_diagram.cod
        # The type of wires before the cupped pair:
        id_wires_len = n_wires_total - (left_main_len + right_main_len)
        id_wires_type = permuted_diagram.cod[:id_wires_len] 

        cup_op = Id(id_wires_type) @ Cup(left_main_type, right_main_type)
        logger.debug(f"Applying Cup op: Id({id_wires_type}) @ Cup({left_main_type}, {right_main_type})")
        logger.debug(f"Permuted diagram cod: {permuted_diagram.cod}, Cup op domain: {cup_op.dom}")

        if permuted_diagram.cod != cup_op.dom:
            logger.error(f"Critical Mismatch: Codomain after permute {permuted_diagram.cod} != Cup domain {cup_op.dom}")
            # This can happen if feature wires (Ty()) were miscounted in n_wires_total or lengths
            return None
        
        result_diagram = permuted_diagram >> cup_op
        logger.info(f"Cup({left_main_type}, {right_main_type}) applied. New cod: {result_diagram.cod}")

        # 5. Update Wire Map (This is the complex part)
        new_wire_map = []
        # The wires that were cupped (wire1_orig_tok_idx, wire2_orig_tok_idx) are now gone.
        # Their associated feature wires (if any) are part of id_wires_type.
        # All other token units' wires are also part of id_wires_type.
        # We need to reconstruct the map for the remaining wires.

        # The `id_wires_type` now contains all wires that were NOT cupped, in their new permuted order.
        # We need to map these back to their original token units and features.
        
        # This is a simplified update logic. A more robust one would track each original wire.
        # For now, we assume that cupping removes the *entire* token units involved if their main wire is cupped.
        # This is an oversimplification if feature wires are meant to persist.
        # A truly robust solution needs to track each feature wire individually through permutations.

        temp_cod_offset = 0
        surviving_wires_type = result_diagram.cod # This is id_wires_type

        # Iterate through the *original* full_wire_map to see what remains
        # The order of wires in surviving_wires_type is based on the permutation `perm`
        # for those wires that were *not* in `indices_to_be_cupped`.

        # Create a list of original indices of token units whose main wires were NOT cupped
        surviving_orig_indices = []
        for unit_desc in current_full_wire_map:
            if unit_desc['orig_idx'] != wire1_orig_tok_idx and unit_desc['orig_idx'] != wire2_orig_tok_idx:
                surviving_orig_indices.append(unit_desc['orig_idx'])
        
        # The new wire map should reflect the order in `surviving_wires_type`.
        # This requires knowing how `perm` reordered the non-cupped wires.
        # The non-cupped wires in `perm` are `perm[:id_wires_len]`.
        # Each `j` in `perm[:id_wires_len]` is an *original absolute wire index*.

        # This is difficult to reconstruct perfectly without detailed tracking of each elementary wire.
        # Let's attempt a pragmatic rebuild:
        
        current_abs_pos_in_new_diag = 0
        # We need to map the old absolute indices (from `perm`) to new token units.
        # This is very tricky. The current `apply_cup_at_indices_v2` in camel_test3.py
        # has a wire map update that might be a better starting point if adapted.
        # For now, returning an empty map to signal it needs proper implementation.
        # logger.warning("Wire map update after cup is complex and needs full implementation.")
        
        # Simplified approach: Rebuild map from remaining units, assuming their internal structure is preserved
        # and they are now contiguous in the new diagram.cod
        
        # Get the original token units that were *not* cupped
        remaining_units_orig_data = [
            unit for unit in current_full_wire_map 
            if unit['orig_idx'] not in (wire1_orig_tok_idx, wire2_orig_tok_idx)
        ]
        
        # The order of these remaining units in the new diagram depends on the permutation.
        # This is the hardest part. The `perm` array tells us the original indices of the wires
        # that form the `id_wires_type`. We need to group these back.

        # Fallback: if the resulting diagram is simple (e.g. just S), the wire map is trivial.
        if result_diagram.cod == S: # If it reduced to a sentence
            logger.debug("Diagram reduced to S, wire map is empty or represents S.")
            # new_wire_map could be [{'orig_idx': -1, 'token': 'SENTENCE', ... type: S}]
        elif len(result_diagram.cod) == 0: # Reduced to Ty()
             logger.debug("Diagram reduced to Ty(), wire map is empty.")
        else:
            # TODO: Implement robust wire map reconstruction. This is a major source of errors.
            # For now, if it's not simple S or Ty(), we can't reliably update the map with this logic.
            logger.error("Robust wire map update after cup is not fully implemented for complex resulting types. Diagram may become disconnected.")
            # To prevent downstream errors, it might be better to return None if map is uncertain
            # return None
            # Or, as a placeholder, assume remaining units are just concatenated:
            temp_offset = 0
            for unit_data in remaining_units_orig_data: # This assumes their relative order is preserved, which is not guaranteed by `perm`
                updated_unit_data = unit_data.copy()
                
                main_len = unit_data['main_len']
                feat_len = unit_data['feat_len'] # Sum of lengths of Ty() which is 0

                updated_unit_data['main_abs_start'] = temp_offset
                updated_unit_data['feat_abs_start'] = temp_offset + main_len
                updated_unit_data['unit_abs_start'] = temp_offset
                updated_unit_data['unit_len'] = main_len + feat_len # total length of this unit in the diagram

                new_wire_map.append(updated_unit_data)
                temp_offset += updated_unit_data['unit_len']
            
            if temp_offset != len(result_diagram.cod):
                logger.error(f"Reconstructed wire map length {temp_offset} does not match result diagram cod length {len(result_diagram.cod)}. Map is unreliable.")
                return None # Critical failure in map update


        logger.debug(f"Wire map updated (or attempted). New map size: {len(new_wire_map)}")
        return result_diagram, new_wire_map

    except ValueError as e_cup_val:
        logger.error(f"ValueError during Cup application: {e_cup_val}. Types might be wrong or diagram structure issue.")
        return None
    except Exception as e_cup_unexpected:
        logger.error(f"Unexpected error applying Cup after permutation: {e_cup_unexpected}", exc_info=True)
        return None


def create_verbal_sentence_diagram_v3_1_features(
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]], 
    original_indices: List[int],
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_verbal"
) -> Optional[GrammarDiagram]:
    logger.info(f"Creating verbal diagram for: {' '.join(tokens)}")
    arg_producer_boxes: Dict[int, Box] = {} 
    functor_boxes: Dict[int, Box] = {}    

    for i, orig_idx in enumerate(original_indices):
        core_entity = word_core_types[i] 
        analysis = next((a for a in analyses_details if a['original_idx'] == orig_idx), None)
        
        if not analysis or core_entity is None:
            logger.debug(f"Skipping token at orig_idx {orig_idx} for verbal diagram: No analysis or core_entity is None.")
            continue
        
        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"
        logger.debug(f"Processing for Verbal Box '{box_name}': core_entity='{str(core_entity)}', type(core_entity)='{type(core_entity)}'")

        if isinstance(core_entity, Box): 
            functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod) 
            logger.info(f"  Created/Registered Functor Box for '{box_name}': {core_entity.dom} -> {core_entity.cod} from Box object.")
        elif isinstance(core_entity, Ty): 
            arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)
            logger.info(f"  Created Argument Producer Box for '{box_name}': Ty() -> {core_entity}")
        else:
            logger.error(f"Unknown or unhandled core_entity type for '{box_name}': {type(core_entity)}. Cannot create box.")
            continue 
            
    subj_idx, verb_idx, obj_idx = roles.get('subject'), roles.get('verb'), roles.get('object')
    
    subj_diag = arg_producer_boxes.get(subj_idx) if subj_idx is not None else None
    verb_functor_box = functor_boxes.get(verb_idx) if verb_idx is not None else None
    obj_diag = arg_producer_boxes.get(obj_idx) if obj_idx is not None else None

    logger.debug(f"  Components for verbal diagram: Subj (idx {subj_idx}): {subj_diag}, Verb (idx {verb_idx}): {verb_functor_box}, Obj (idx {obj_idx}): {obj_diag}")

    if verb_idx is not None and verb_functor_box is None: 
        verb_token_info = "UnknownVerb"
        verb_analysis = next((a for a in analyses_details if a['original_idx'] == verb_idx), None)
        if verb_analysis: verb_token_info = verb_analysis.get('lemma', verb_analysis.get('text', 'UnknownVerb'))
        logger.error(f"CRITICAL ERROR: Verb '{verb_token_info}' (idx {verb_idx}) was identified by roles but not found in functor_boxes. Check its core_type assignment.")
        original_verb_core_type = None
        if verb_idx in original_indices: 
            try: original_verb_core_type = word_core_types[original_indices.index(verb_idx)]
            except ValueError: logger.error(f"  Verb idx {verb_idx} not found in original_indices: {original_indices}")
        else: logger.error(f"  Verb idx {verb_idx} not included in original_indices: {original_indices}")
        logger.error(f"  Original core_type assigned to verb token at idx {verb_idx} was: {str(original_verb_core_type)} (Type: {type(original_verb_core_type)})")
        return None 
    
    final_diagram = None
    structure_type = roles.get("structure")

    # --- Basic Clause Construction ---
    if structure_type == "SVO" and subj_diag and verb_functor_box and obj_diag:
        if subj_diag.cod == N and obj_diag.cod == N and verb_functor_box.dom == (N@N) and verb_functor_box.cod == S:
            try: final_diagram = (subj_diag @ obj_diag) >> verb_functor_box; logger.info(f"SVO composition successful for '{sentence_prefix}'.")
            except Exception as e: logger.error(f"SVO composition error for '{sentence_prefix}': {e}", exc_info=True)
        else: logger.warning(f"SVO type mismatch for '{sentence_prefix}': Subj.cod={subj_diag.cod}, Obj.cod={obj_diag.cod}, Verb.dom={verb_functor_box.dom}")
    
    elif structure_type == "SV" and subj_diag and verb_functor_box:
        if subj_diag.cod == N and verb_functor_box.dom == N and verb_functor_box.cod == S:
            try: final_diagram = subj_diag >> verb_functor_box; logger.info(f"SV composition successful for '{sentence_prefix}'.")
            except Exception as e: logger.error(f"SV composition error for '{sentence_prefix}': {e}", exc_info=True)
        else: logger.warning(f"SV type mismatch for '{sentence_prefix}': Subj.cod={subj_diag.cod}, Verb.dom={verb_functor_box.dom}")
    
    elif structure_type == "VS" and subj_diag and verb_functor_box: 
        if verb_functor_box.dom == N and subj_diag.cod == N and verb_functor_box.cod == S :
             try: final_diagram = subj_diag >> verb_functor_box; logger.info(f"VS composition (treating as S >> V) successful for '{sentence_prefix}'.")
             except Exception as e: logger.error(f"VS composition error for '{sentence_prefix}': {e}", exc_info=True)
        else: logger.warning(f"VS type mismatch for '{sentence_prefix}'. Subj.cod={subj_diag.cod if subj_diag else 'None'}, Verb.dom={verb_functor_box.dom if verb_functor_box else 'None'}")
    
    elif structure_type == "VSO" and subj_diag and verb_functor_box and obj_diag:
        if verb_functor_box.dom == (N @ N) and subj_diag.cod == N and obj_diag.cod == N and verb_functor_box.cod == S:
            try: final_diagram = (subj_diag @ obj_diag) >> verb_functor_box; logger.info(f"VSO composition (S@O >> V) successful for '{sentence_prefix}'.")
            except Exception as e: logger.error(f"VSO composition error for '{sentence_prefix}': {e}", exc_info=True)
        else: logger.warning(f"VSO type mismatch for '{sentence_prefix}'.")

    elif structure_type == "VERBAL_NO_EXPLICIT_SUBJ" and verb_functor_box:
        if verb_functor_box.dom == Ty() and verb_functor_box.cod == S:
            final_diagram = verb_functor_box
            logger.info(f"VERBAL_NO_EXPLICIT_SUBJ: Using verb as Ty()->S diagram for '{sentence_prefix}'.")
        elif obj_diag and verb_functor_box.dom == N and verb_functor_box.cod == S: 
            try:
                final_diagram = obj_diag >> verb_functor_box 
                logger.info(f"VERBAL_NO_EXPLICIT_SUBJ (V O with N->S verb) composition for '{sentence_prefix}' successful.")
            except Exception as e:
                logger.error(f"VERBAL_NO_EXPLICIT_SUBJ (V O like) composition error for '{sentence_prefix}': {e}", exc_info=True)
        elif obj_diag and verb_functor_box.dom == (N @ N) and verb_functor_box.cod == S: 
            logger.warning(f"VERBAL_NO_EXPLICIT_SUBJ: Transitive verb '{verb_functor_box.name}' (N@N->S) has an object but no explicit subject for '{sentence_prefix}'. Attempting to form N->S by applying object.")
            try:
                final_diagram = (Id(N) @ obj_diag) >> verb_functor_box
                logger.info(f"VERBAL_NO_EXPLICIT_SUBJ: Transitive verb with object. Created N->S diagram for '{sentence_prefix}'.")
            except Exception as e:
                logger.error(f"VERBAL_NO_EXPLICIT_SUBJ (Transitive V O) composition error for '{sentence_prefix}': {e}", exc_info=True)
        else:
            logger.warning(f"VERBAL_NO_EXPLICIT_SUBJ: Unhandled verb type {verb_functor_box.dom} -> {verb_functor_box.cod} or missing object for '{sentence_prefix}'.")
    
    # --- PP Attachment Logic (Refined using Dependency Parse) ---
    if final_diagram and final_diagram.cod == S:
        if S_MOD_BY_N is None:
            logger.error("  Cannot attempt PP attachment: S_MOD_BY_N global Box is not defined (checked before attachment loop).")
        else:
            processed_pp_nouns = set() # Keep track of nouns already used as objects of prepositions
            
            # Iterate through all functor boxes to find prepositions
            for prep_orig_idx, prep_box in functor_boxes.items():
                # Check if it's a preposition functor (N->N)
                # Check name prefix for safety, assuming PREP_FUNCTOR_TYPE is defined
                is_prep_functor = (isinstance(prep_box, Box) and 
                                   prep_box.dom == N and prep_box.cod == N and 
                                   PREP_FUNCTOR_TYPE is not None and 
                                   prep_box.name.startswith(PREP_FUNCTOR_TYPE.name[:-9])) # Check based on global name
                
                if not is_prep_functor:
                    continue

                prep_analysis = next((a for a in analyses_details if a['original_idx'] == prep_orig_idx), None)
                if not prep_analysis: continue

                # Find the noun object of this preposition using dependency graph
                noun_obj_orig_idx = None
                
                # Look for a noun that has this preposition as its 'case' dependent
                for head_candidate_idx, dependents in roles.get('dependency_graph', {}).items():
                    for dep_idx, dep_rel_str in dependents:
                        if dep_idx == prep_orig_idx and dep_rel_str == 'case':
                            if head_candidate_idx in arg_producer_boxes and head_candidate_idx not in processed_pp_nouns:
                                noun_obj_orig_idx = head_candidate_idx
                                break
                    if noun_obj_orig_idx: break
                
                # If not found via 'case', check if the prep is head of an 'obl' relation
                if noun_obj_orig_idx is None:
                     prep_dependents = roles.get('dependency_graph', {}).get(prep_orig_idx, [])
                     for dep_idx, dep_rel_str in prep_dependents:
                          if dep_rel_str in ['obl', 'obl:arg', 'nmod']: # nmod can sometimes link prep to its object
                               if dep_idx in arg_producer_boxes and dep_idx not in processed_pp_nouns:
                                    noun_obj_orig_idx = dep_idx
                                    break
                
                if noun_obj_orig_idx is not None:
                    noun_obj_diag = arg_producer_boxes[noun_obj_orig_idx]
                    
                    # Determine if this PP modifies the main verb
                    pp_modifies_verb = False
                    noun_obj_analysis = next((a for a in analyses_details if a['original_idx'] == noun_obj_orig_idx), None)
                    # Check if noun object is obl/obl:arg of the verb
                    if noun_obj_analysis and noun_obj_analysis.get('head') == verb_idx and noun_obj_analysis.get('deprel') in ['obl', 'obl:arg']:
                        pp_modifies_verb = True
                    # Check if preposition is attached to the verb (less common for case, but possible for advmod etc.)
                    elif prep_analysis.get('head') == verb_idx and prep_analysis.get('deprel') in ['obl', 'obl:arg', 'advmod']: 
                         pp_modifies_verb = True

                    if pp_modifies_verb:
                        try:
                            pp_diag_composed = noun_obj_diag >> prep_box 
                            pp_log_name = f"({noun_obj_diag.name} >> {prep_box.name})"
                            logger.info(f"  Identified PP (modifying verb/sentence): {pp_log_name} -> type {pp_diag_composed.cod}")
                            
                            if final_diagram.cod == S and pp_diag_composed.cod == N : 
                                logger.info(f"Attempting to attach PP ({pp_log_name}) to S diagram using S_MOD_BY_N.")
                                final_diagram = (final_diagram @ pp_diag_composed) >> S_MOD_BY_N
                                logger.info(f"  Successfully attached PP. New diagram cod: {final_diagram.cod}")
                                processed_pp_nouns.add(noun_obj_orig_idx) # Mark noun as used in this PP
                            else:
                                logger.warning(f"  Skipping attachment of PP {pp_log_name}. Type mismatch: final_diag.cod={final_diagram.cod}, pp_diag.cod={pp_diag_composed.cod}")
                        except Exception as e_pp_attach:
                            logger.error(f"Error forming or attaching PP with {prep_box.name} and {noun_obj_diag.name}: {e_pp_attach}", exc_info=True)
                    else:
                        logger.info(f"  PP ({prep_box.name} with obj {noun_obj_diag.name if noun_obj_diag else 'N/A'}) does not appear to modify the main verb directly. NP-internal PP attachment or other PP roles not yet fully handled.")
                else:
                    logger.debug(f"Could not find a clear noun object for preposition '{prep_box.name}' (idx {prep_orig_idx}) using dependency graph.")


    if final_diagram:
        try: 
            final_diagram.normal_form() 
            if final_diagram.cod == S: 
                logger.info(f"Verbal diagram normal_form successful for '{sentence_prefix}'. Final cod: {final_diagram.cod}")
                return final_diagram
            else:
                logger.warning(f"Verbal diagram for '{sentence_prefix}' normalized, but final cod is {final_diagram.cod}, not S. Discarding.")
                return None
        except Exception as e: 
            logger.error(f"Verbal diagram normal_form failed for '{sentence_prefix}': {e}", exc_info=True)
            return None 
            
    logger.warning(f"Could not form a complete verbal diagram ending in S for structure: {structure_type} for sentence '{sentence_prefix}'.")
    return None


def create_nominal_sentence_diagram_v2_1_features(
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]], 
    original_indices: List[int],
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_nominal"
) -> Optional[GrammarDiagram]:
    logger.info(f"Creating nominal diagram for: {' '.join(tokens)}")
    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}

    for i, orig_idx in enumerate(original_indices):
        core_entity = word_core_types[i]
        analysis = next((a for a in analyses_details if a['original_idx'] == orig_idx), None)
        if not analysis or core_entity is None:
            logger.debug(f"Skipping token at orig_idx {orig_idx} for nominal diagram: No analysis or core_entity is None.")
            continue
        
        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"
        logger.debug(f"Processing for Nominal Box '{box_name}': core_entity='{str(core_entity)}', type(core_entity)='{type(core_entity)}'")

        if isinstance(core_entity, Box): 
            functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod) 
            logger.info(f"  Created/Registered Nominal Functor Box for '{box_name}': {core_entity.dom} -> {core_entity.cod} from Box object.")
        elif isinstance(core_entity, Ty): 
            arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)
            logger.info(f"  Created Nominal Argument Producer Box for '{box_name}': Ty() -> {core_entity}")
        else:
            logger.error(f"Unknown or unhandled core_entity type for nominal '{box_name}': {type(core_entity)}. Cannot create box.")
            continue
            
    subj_idx_from_roles = roles.get('subject', roles.get('root')) 
    predicate_orig_idx = None 
    
    if subj_idx_from_roles is not None:
        for token_orig_idx_in_diagram in original_indices: 
            if token_orig_idx_in_diagram == subj_idx_from_roles: continue 
            if token_orig_idx_in_diagram in functor_boxes:
                potential_pred_box = functor_boxes[token_orig_idx_in_diagram]
                if potential_pred_box.dom == N and potential_pred_box.cod == S:
                    pred_analysis = next((a for a in analyses_details if a['original_idx'] == token_orig_idx_in_diagram), None)
                    if pred_analysis:
                        is_root = (pred_analysis['original_idx'] == roles.get('root'))
                        is_headed_by_subject = (pred_analysis.get('head') == subj_idx_from_roles) 
                        if is_root or is_headed_by_subject:
                            predicate_orig_idx = token_orig_idx_in_diagram
                            logger.info(f"Found nominal predicate functor: {pred_analysis['text']} (orig_idx {predicate_orig_idx}) of type {potential_pred_box.dom} -> {potential_pred_box.cod}")
                            break 
    
    subj_diag = arg_producer_boxes.get(subj_idx_from_roles) if subj_idx_from_roles is not None else None
    pred_functor_box = functor_boxes.get(predicate_orig_idx) if predicate_orig_idx is not None else None

    if predicate_orig_idx is not None and pred_functor_box is None: 
        pred_token_info = "UnknownPredicate"
        pred_analysis_check = next((a for a in analyses_details if a['original_idx'] == predicate_orig_idx), None)
        if pred_analysis_check: pred_token_info = pred_analysis_check.get('lemma', pred_analysis_check.get('text', 'UnknownPredicate'))
        logger.error(f"CRITICAL ERROR: Nominal Predicate '{pred_token_info}' (idx {predicate_orig_idx}) was identified but not found in functor_boxes.")
        original_pred_core_type = None
        if predicate_orig_idx in original_indices: 
            try: original_pred_core_type = word_core_types[original_indices.index(predicate_orig_idx)]
            except ValueError: logger.error(f"  Predicate idx {predicate_orig_idx} not found in original_indices: {original_indices}")
        else: logger.error(f" Predicate idx {predicate_orig_idx} was not part of the tokens selected for diagram construction.")
        logger.error(f"  Original core_type assigned to predicate token at idx {predicate_orig_idx} was: {str(original_pred_core_type)} (Type: {type(original_pred_core_type)})")
        return None
        
    final_diagram = None
    if subj_diag and pred_functor_box:
        if subj_diag.cod == N and pred_functor_box.dom == N and pred_functor_box.cod == S:
            try: final_diagram = subj_diag >> pred_functor_box; logger.info(f"Nominal composition successful for '{sentence_prefix}'.")
            except Exception as e: logger.error(f"Nominal composition error for '{sentence_prefix}': {e}", exc_info=True)
        else: logger.warning(f"Nominal type mismatch for '{sentence_prefix}': Subj.cod={subj_diag.cod}, Pred.dom={pred_functor_box.dom}")
    
    elif subj_diag and not pred_functor_box:
        logger.warning(f"Nominal sentence for '{sentence_prefix}' has a subject ('{subj_diag.name}') but no N->S predicate functor was found/applied. Returning subject diagram (type N).")
        final_diagram = subj_diag 
    
    elif not subj_diag and pred_functor_box: 
        logger.warning(f"Nominal sentence for '{sentence_prefix}' has predicate functor ('{pred_functor_box.name}') but no subject found/applied. This is unusual.")
            
    if final_diagram:
        try: 
            final_diagram.normal_form()
            if final_diagram.cod == S:
                 logger.info(f"Nominal diagram normal_form successful for '{sentence_prefix}'. Final cod: {final_diagram.cod}")
                 return final_diagram
            else:
                 logger.warning(f"Nominal diagram for '{sentence_prefix}' normalized, but final cod is {final_diagram.cod}, not S. Discarding.")
                 return None
        except Exception as e: 
            logger.error(f"Nominal diagram normal_form failed for '{sentence_prefix}': {e}", exc_info=True)
            return None 
            
    logger.warning(f"Could not form a complete nominal diagram ending in S for sentence '{sentence_prefix}' with subj_idx: {subj_idx_from_roles}, pred_idx: {predicate_orig_idx}")
    return None

# ==================================
# Main Conversion Function (REVISED - V2.1 using new features logic)
# ==================================
def arabic_to_quantum_enhanced_v2_1_features(
    sentence: str,
    debug: bool = True,
    output_dir: Optional[str] = None,
    ansatz_choice: str = "IQP",
    n_layers_iqp: int = 1,
    n_single_qubit_params_iqp: int = 3,
    # Parameters for other ansatzes, matching original file structure
    n_layers_strong: int = 1,
    cnot_ranges: Optional[List[Tuple[int, int]]] = None,
    discard_qubits_spider: bool = True,
    **kwargs # Catch-all for other/unexpected keyword arguments
) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Dict[str,Any]], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram,
    and converts it to a Qiskit QuantumCircuit. V2.1.
    Accepts ansatz configuration parameters and **kwargs for flexibility.
    """
    # Log any unexpected keyword arguments received if **kwargs is used
    if kwargs:
        logger.warning(f"Function arabic_to_quantum_enhanced_v2_1_features received UNEXPECTED keyword arguments: {kwargs}")

    tokens = []
    analyses_details = [] # This will store list of dicts, each dict is one token's full analysis
    structure = "ERROR"
    roles = {} # Will store subject, verb, object indices etc.
    diagram = None
    circuit = None # Qiskit circuit

    # 1. Analyze sentence (ensure this populates analyses_details with 'original_idx')
    try:
        logger.info(f"Analyzing sentence with morph: '{sentence}'")
        # Ensure your analyze_arabic_sentence_with_morph returns analyses_details
        # where each dict has an 'original_idx' key.
        tokens, analyses_details, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug)
        if structure == "ERROR" or not tokens:
            logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
            return None, None, structure, tokens, analyses_details, roles
        logger.info(f"Analysis complete. Detected structure: {structure}. Roles: {roles}")
        # Log a sample of analyses_details to confirm 'original_idx'
        if debug and analyses_details:
            logger.debug(f"  Sample analysis detail (first token): {analyses_details[0]}")

    except Exception as e_analyze_main:
        logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
        return None, None, "ERROR", tokens, analyses_details, roles

    # --- Filter tokens and assign core types ---
    word_core_types_list = [] # Stores the DisCoCat type for each token to be included in diagram
    original_indices_for_diagram = [] # Stores original sentence index for tokens in diagram
    filtered_tokens_for_diagram = []  # Stores text of tokens in diagram

    logger.debug(f"--- Assigning Core Types for: '{sentence}' ---")
    for i, analysis_entry in enumerate(analyses_details): # i is the original index from Stanza/CAMeL
        token_text = analysis_entry['text']
        lemma = analysis_entry['lemma']
        pos = analysis_entry['upos']
        dep_rel = analysis_entry['deprel']
        
        # Skip punctuation/diacritics (ensure this logic is correct)
        # ARABIC_DIACRITICS should be defined globally
        # string.punctuation should be imported
        # if all(ch in ARABIC_DIACRITICS for ch in token_text) or token_text in string.punctuation:
        #     logger.debug(f"Skipping token '{token_text}' (punctuation/diacritic).")
        #     continue # This continue was problematic if core_type was used after it unconditionally

        is_verb_flag = (analysis_entry['original_idx'] == roles.get('verb')) # Use original_idx for comparison
        is_nominal_pred_flag = False
        if structure == "NOMINAL":
            subj_idx_orig = roles.get("subject", roles.get("root"))
            if analysis_entry['original_idx'] == roles.get("root") and analysis_entry['original_idx'] != subj_idx_orig and pos in ["ADJ", "NOUN", "PROPN"]:
                 is_nominal_pred_flag = True
            elif subj_idx_orig is not None and analysis_entry.get('head') == subj_idx_orig and pos in ["ADJ", "NOUN", "PROPN"]:
                is_nominal_pred_flag = True

        verb_takes_subj_flag = (roles.get('subject') is not None)
        verb_takes_obj_flag = (roles.get('object') is not None)
        
        logger.debug(f"  Token '{token_text}' (orig_idx {analysis_entry['original_idx']}): POS='{pos}', Dep='{dep_rel}', is_verb={is_verb_flag}, verb_takes_subj={verb_takes_subj_flag}, verb_takes_obj={verb_takes_obj_flag}, is_nominal_pred={is_nominal_pred_flag}")

        current_core_type = assign_discocat_types_v2(
            pos, dep_rel, token_text, lemma,
            is_verb=is_verb_flag,
            verb_takes_subject=verb_takes_subj_flag,
            verb_takes_object=verb_takes_obj_flag,
            is_nominal_pred=is_nominal_pred_flag,
            debug=debug
        )

        logger.debug(f"    Raw core_type from assign_discocat_types_v2: {str(current_core_type)} (Type: {type(current_core_type)})")
        if isinstance(current_core_type, GrammarDiagram): logger.info(f"    SUCCESS: Core type for '{token_text}' is a GrammarDiagram.")
        # ... other logging for core_type type

        if current_core_type is not None:
            word_core_types_list.append(current_core_type)
            original_indices_for_diagram.append(analysis_entry['original_idx']) # Use original_idx
            filtered_tokens_for_diagram.append(token_text)
        else:
            logger.debug(f"  Token '{token_text}' (orig_idx {analysis_entry['original_idx']}) was assigned None core type, excluding from diagram construction.")

    if not filtered_tokens_for_diagram:
        logger.error(f"No valid tokens with core types remained for diagram construction: '{sentence}'")
        return None, None, structure, tokens, analyses_details, roles

    logger.debug(f"Filtered Tokens for Diagram: {filtered_tokens_for_diagram}")
    logger.debug(f"Assigned Word Core Types (list for diagram construction): {[str(ct) for ct in word_core_types_list]}")
    logger.debug(f"Original Indices for Diagram Tokens: {original_indices_for_diagram}")

    # 3. Create DisCoCat Diagram
    diagram = None
    try:
        logger.info("Creating DisCoCat diagram...")
        # Ensure 'original_indices_for_diagram' is passed to these functions
        if structure == "NOMINAL":
            diagram = create_nominal_sentence_diagram_v2_1_features(
                filtered_tokens_for_diagram, analyses_details, roles,
                word_core_types_list, original_indices_for_diagram, debug,
                output_dir=output_dir, sentence_prefix=f"sent_{tokens[0] if tokens else 'empty'}_nominal"
            )
        elif structure != "ERROR":
            diagram = create_verbal_sentence_diagram_v3_1_features(
                filtered_tokens_for_diagram, analyses_details, roles,
                word_core_types_list, original_indices_for_diagram, debug,
                output_dir=output_dir, sentence_prefix=f"sent_{tokens[0] if tokens else 'empty'}_verbal"
            )
        if diagram is None:
            logger.error(f"Diagram creation returned None for sentence '{sentence}' with structure '{structure}'.")
        else:
            logger.info(f"Diagram created successfully for '{sentence}'. Final Cod: {diagram.cod}")
    except Exception as e_diagram:
        logger.error(f"Exception during diagram creation phase for '{sentence}': {e_diagram}", exc_info=True)
        return None, diagram, structure, tokens, analyses_details, roles

    if diagram is None:
        logger.error("Diagram is None after creation attempt, cannot proceed to circuit conversion.")
        return None, None, structure, tokens, analyses_details, roles

    # 4. Convert diagram to quantum circuit
    circuit = None
    try:
        logger.info(f"Converting diagram to quantum circuit using ansatz: {ansatz_choice}")
        ob_map = {N: 1, S: 1}
        if ADJ != N : ob_map[ADJ] = 1

        selected_ansatz = None
        if ansatz_choice.upper() == "IQP":
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
        elif ansatz_choice.upper() == "STRONGLY_ENTANGLING": # Corrected name
             # Add logic to determine num_qubits_for_strong and final_cnot_ranges if needed
            selected_ansatz = StronglyEntanglingAnsatz(ob_map=ob_map, n_layers=n_layers_strong, ranges=cnot_ranges)
        elif ansatz_choice.upper() == "SPIDER":
            selected_ansatz = SpiderAnsatz(ob_map=ob_map, discard_qubits=discard_qubits_spider)
        else:
            logger.warning(f"Unknown ansatz_choice: '{ansatz_choice}'. Defaulting to IQPAnsatz.")
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3)

        simplified_diagram = diagram.normal_form()
        quantum_diagram = selected_ansatz(simplified_diagram)
        
        # Assuming PYTKET_QISKIT_AVAILABLE is true and tk_to_qiskit is imported
        from pytket.extensions.qiskit import tk_to_qiskit
        tket_circ = quantum_diagram.to_tk()
        circuit = tk_to_qiskit(tket_circ)
        #circuit = quantum_diagram.to_qiskit() # Lambeq >=0.4.0 has direct to_qiskit
        logger.info("Circuit conversion successful.")
    except NotImplementedError as e_nf_main:
        logger.error(f"NORMAL_FORM FAILED in main conversion for diagram of '{sentence}': {e_nf_main}", exc_info=True)
        return None, diagram, structure, tokens, analyses_details, roles
    except Exception as e_circuit_outer:
        logger.error(f"Exception during circuit conversion for '{sentence}': {e_circuit_outer}", exc_info=True)
        return None, diagram, structure, tokens, analyses_details, roles

    return circuit, diagram, structure, tokens, analyses_details, roles



# ==================================
# Visualization Functions (from camel_test3.py - ensure they are present)
# ==================================
def visualize_diagram(diagram, save_path=None):
    if diagram is None or not hasattr(diagram, 'draw'): return None
    try:
        # Adjust figsize dynamically based on diagram complexity
        width = max(10, len(diagram.boxes) * 1.0 + len(diagram.offsets) * 0.3) 
        # Approximate height based on max wires at any point + vertical spacing
        max_wires = 0
        if diagram.dom: max_wires = max(max_wires, len(diagram.dom))
        if diagram.cod: max_wires = max(max_wires, len(diagram.cod))
        for b in diagram.boxes:
            if b.dom: max_wires = max(max_wires, len(b.dom))
            if b.cod: max_wires = max(max_wires, len(b.cod))
        
        height = max(6, max_wires * 0.5 + 2)
        
        ax = diagram.draw(figsize=(width, height), fontsize=9, aspect='auto') # 'auto' aspect for better fit
        fig = ax.figure 
        if save_path:
            try: 
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved diagram to {save_path}")
            except Exception as e_save: logger.error(f"Failed to save diagram: {e_save}")
        plt.close(fig)
        return fig
    except Exception as e:
        logger.error(f"Could not visualize diagram: {e}", exc_info=True)
        plt.close() 
        return None

def visualize_circuit(circuit, save_path=None):
    if circuit is None or not hasattr(circuit, 'draw'): return None
    try:
        # Dynamic figsize for circuits
        depth = circuit.depth()
        num_qubits = circuit.num_qubits
        width = max(10, depth * 0.3 + num_qubits * 0.5) 
        height = max(6, num_qubits * 0.6 + 2)

        fig = circuit.draw(output='mpl', fold=-1, scale=0.7) # scale might need adjustment
        if fig: 
             fig.set_size_inches(width, height)
             plt.tight_layout() # Apply tight layout to prevent labels from overlapping
             if save_path:
                 try: 
                     fig.savefig(save_path, bbox_inches='tight', dpi=150)
                     logger.info(f"Saved circuit to {save_path}")
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

# (visualize_dependency_tree would also go here if used)

# ==================================
# Main Execution / Testing
# ==================================
if __name__ == "__main__":
    logger.info("Running camel_test3_features.py directly for testing...")

    test_sentences = [
        "يقرأ الولد الكتاب",       # VSO (The boy reads the book)
        "الولد يقرأ الكتاب",       # SVO (The boy reads the book)
        "البيت كبير",             # NOMINAL (The house is big)
        "الطالبة الذكية تدرس العلوم بجد", # SVO (The smart student studies science diligently)
        "كتبت الطالبة الدرس بسرعة", # VSO (The student wrote the lesson quickly)
        "الرجل الذي رأيته طبيب", # COMPLEX (The man whom I saw is a doctor) - Likely complex/other
        "ذهب الولد الى المدرسة صباحا"   # VSO with PP and Adverb (The boy went to school in the morning)
    ]

    test_output_dir = "camel_test3_features_output"
    os.makedirs(test_output_dir, exist_ok=True)
    logger.info(f"Test output will be saved to: {test_output_dir}")

    all_results_summary = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n--- Testing Sentence {i+1}/{len(test_sentences)}: '{sentence}' ---")
        # Sanitize sentence for use in file prefix
        safe_sentence_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0])
        current_sentence_prefix = f"test_{i+1}_{safe_sentence_prefix}"

        try:
            result_tuple = arabic_to_quantum_enhanced_v2_1_features(sentence, debug=True, output_dir=test_output_dir)
            circuit, diagram, structure, _, _, _ = result_tuple # Unpack relevant parts

            all_results_summary.append({
                "sentence": sentence, 
                "structure": structure,
                "diagram_ok": diagram is not None,
                "diagram_cod": diagram.cod if diagram else "N/A",
                "circuit_ok": circuit is not None,
                "error": None
            })
            logger.info(f"--- Result for Sentence {i+1}: Structure='{structure}', Diagram OK='{diagram is not None}', Circuit OK='{circuit is not None}' ---")
            if diagram: logger.info(f" Diagram Cod: {diagram.cod}")

        except Exception as e_test:
            logger.error(f"!!! ERROR during test for sentence: '{sentence}' !!!", exc_info=True)
            all_results_summary.append({"sentence": sentence, "error": str(e_test), "diagram_ok": False, "circuit_ok": False, "structure": "ERROR"})

    logger.info("\n\n" + "="*20 + " Test Summary " + "="*20)
    for i, res in enumerate(all_results_summary):
        logger.info(f"Sentence {i+1}: '{res['sentence']}'")
        if res["error"]:
            logger.info(f"  Status: ERROR ({res['error']})")
        else:
            logger.info(f"  Structure: {res['structure']}")
            logger.info(f"  Diagram OK: {res['diagram_ok']} (Cod: {res.get('diagram_cod', 'N/A')})")
            logger.info(f"  Circuit OK: {res['circuit_ok']}")
    logger.info("="*54)