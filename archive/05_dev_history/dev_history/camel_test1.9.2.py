# -*- coding: utf-8 -*-
# camel_test2.py (Revised Version)
# Incorporates improvements based on log analysis, focusing on DET, ADJ, and PP logic.

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
    # Use a specific path or let CAMeL Tools find the default
    # db_path = MorphologyDB.builtin_db('calima-msa-r13') # Example specific DB
    db_path = MorphologyDB.builtin_db() # Find default
    CAMEL_ANALYZER = Analyzer(db_path)
    logger.info("CAMeL Tools Analyzer initialized successfully.")
except ImportError:
    logger.warning("CAMeL Tools not found. Morphological feature extraction will be limited.")
except Exception as e:
    logger.warning(f"Error initializing CAMeL Tools Analyzer: {e}. Morphological feature extraction will be limited.")

STANZA_AVAILABLE = False
try:
    # Download model if not present (optional, Stanza usually handles this)
    # stanza.download('ar', verbose=False)
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse,mwt', verbose=False, use_gpu=False, logging_level='WARN') # Added mwt, reduced verbosity
    STANZA_AVAILABLE = True
    logger.info("Stanza pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Stanza: {e}", exc_info=True)
    STANZA_AVAILABLE = False

# Define Atomic Types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
# P = AtomicType.PREPOSITIONAL_PHRASE # Not typically used as a base type in DisCoCat directly
# C = AtomicType.CONJUNCTION          # Often handled by structure or specific boxes
# ADJ = AtomicType.NOUN_PHRASE       # Let ADJ map to N for modifiers, or N >> S for predicates

logger.info(f"AtomicType N: {str(N)}, Type: {type(N)}")
logger.info(f"AtomicType S: {str(S)}, Type: {type(S)}")

# --- Pre-defined Functorial Types (Boxes) ---
# These represent functions transforming types (e.g., Noun -> Noun)
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
    logger.debug("Defining ADJ_MOD_TYPE (N->N)...")
    ADJ_MOD_TYPE = Box("AdjModFunctor", N, N)
    logger.info(f"  ADJ_MOD_TYPE: {str(ADJ_MOD_TYPE)}")

    logger.debug("Defining ADJ_PRED_TYPE (N->S)...")
    ADJ_PRED_TYPE = Box("AdjPredFunctor", N, S)
    logger.info(f"  ADJ_PRED_TYPE: {str(ADJ_PRED_TYPE)}")

    logger.debug("Defining DET_TYPE (N->N)...")
    DET_TYPE = Box("DetFunctor", N, N)
    logger.info(f"  DET_TYPE: {str(DET_TYPE)}")

    # Preposition takes a Noun object and returns a Noun (representing the PP sense)
    logger.debug("Defining PREP_FUNCTOR_TYPE (N->N)...")
    PREP_FUNCTOR_TYPE = Box("PrepFunctor", N, N)
    logger.info(f"  PREP_FUNCTOR_TYPE: {str(PREP_FUNCTOR_TYPE)}")

    logger.debug("Defining VERB_INTRANS_TYPE (N->S)...")
    VERB_INTRANS_TYPE = Box("VerbIntransFunctor", N, S)
    logger.info(f"  VERB_INTRANS_TYPE: {str(VERB_INTRANS_TYPE)}")

    logger.debug("Defining VERB_TRANS_TYPE (N@N->S)...")
    VERB_TRANS_TYPE = Box("VerbTransFunctor", N @ N, S)
    logger.info(f"  VERB_TRANS_TYPE: {str(VERB_TRANS_TYPE)}")

    logger.debug("Defining S_MOD_BY_N (S@N->S, for PP/Adv attachment to Sentence)...")
    S_MOD_BY_N = Box("S_mod_by_N", S @ N, S)
    logger.info(f"  S_MOD_BY_N: {str(S_MOD_BY_N)}")

    logger.debug("Defining N_MOD_BY_N (N@N->N, for PP attachment to Noun)...")
    N_MOD_BY_N = Box("N_mod_by_N", N @ N, N)
    logger.info(f"  N_MOD_BY_N: {str(N_MOD_BY_N)}")

    logger.debug("Defining ADV_FUNCTOR_TYPE (S->S)...")
    ADV_FUNCTOR_TYPE = Box("AdvFunctor", S, S)
    logger.info(f"  ADV_FUNCTOR_TYPE: {str(ADV_FUNCTOR_TYPE)}")

    logger.info(">>> Global Box definitions for functorial types completed. <<<")

except Exception as e_box_def:
    logger.critical(f"CRITICAL ERROR defining global Box types: {e_box_def}", exc_info=True)
    # Ensure all are None if any definition fails to prevent downstream errors
    ADJ_MOD_TYPE = ADJ_PRED_TYPE = DET_TYPE = PREP_FUNCTOR_TYPE = None
    VERB_INTRANS_TYPE = VERB_TRANS_TYPE = S_MOD_BY_N = N_MOD_BY_N = ADV_FUNCTOR_TYPE = None

# Check if essential types are defined after initialization attempt
if S_MOD_BY_N is None or N_MOD_BY_N is None or PREP_FUNCTOR_TYPE is None:
     logger.error("One or more essential modifier Box types (S_MOD_BY_N, N_MOD_BY_N, PREP_FUNCTOR_TYPE) failed to initialize.")

# ==================================
# Linguistic Analysis Functions
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
            # else: logger.debug(f"Skipping feature part without '=': {pair} in {feats_str}")
    except Exception as e:
        logger.warning(f"Could not parse features string '{feats_str}': {e}")
    return feats_dict

def analyze_arabic_sentence_with_morph(sentence: str, debug: bool = False) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Analyzes an Arabic sentence using Stanza and CAMeL Tools.
    Returns tokens, detailed analyses, detected structure type, and grammatical roles.
    """
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
    roles_dict = {"verb": None, "subject": None, "object": None, "root": None, "dependency_graph": {}, "structure": "OTHER"}

    if not doc.sentences:
        logger.warning(f"Stanza did not find any sentences in: '{sentence}'")
        return [], [], "OTHER", roles_dict

    sent = doc.sentences[0] # Process the first sentence found

    # --- First Pass: Extract basic info and build dependency graph structure ---
    for i, word in enumerate(sent.words):
        # Basic info
        token_text = word.text
        lemma = word.lemma if word.lemma else token_text # Fallback lemma
        upos = word.upos
        deprel = word.deprel
        head_idx = word.head - 1 if word.head > 0 else -1 # Stanza is 1-based, convert to 0-based; -1 for root

        # Parse Stanza features
        stanza_feats_dict = parse_feats_string(word.feats)

        # Enrich with CAMeL Tools morphological features if available and needed
        camel_feats_dict = {}
        if CAMEL_ANALYZER:
            # Check if key features are missing from Stanza before calling CAMeL
            # This avoids unnecessary calls for words Stanza handles well (like verbs)
            # if not all(k in stanza_feats_dict for k in ['asp', 'gen', 'num', 'per', 'vox']) or upos in ['NOUN', 'ADJ', 'PROPN']:
            try:
                camel_analysis_list = CAMEL_ANALYZER.analyze(token_text)
                if camel_analysis_list:
                    # Take the first CAMeL analysis (often the most likely)
                    camel_feats_str = camel_analysis_list[0].get('feat')
                    camel_feats_dict = parse_feats_string(camel_feats_str)
                    if debug and camel_feats_dict:
                         logger.debug(f"  CAMeL feats for '{token_text}': {camel_feats_dict}")
            except Exception as e_camel_enrich:
                logger.warning(f"Error during CAMeL enrichment for '{token_text}': {e_camel_enrich}")

        # Combine features: Stanza takes precedence, CAMeL fills gaps
        combined_feats = stanza_feats_dict.copy()
        for k, v in camel_feats_dict.items():
            if k not in combined_feats:
                combined_feats[k] = v

        # Store analysis
        analysis_entry = {
            "text": token_text, "lemma": lemma, "upos": upos,
            "deprel": deprel, "head": head_idx,
            "feats_dict": combined_feats, # Store combined features
            "original_idx": i # Store original 0-based index
        }
        processed_tokens_texts.append(token_text)
        processed_analyses.append(analysis_entry)

        # Build dependency graph structure (adjacency list: head_idx -> list of (dependent_idx, relation))
        if head_idx >= 0: # Don't add edges from the dummy root -1
             if head_idx not in roles_dict["dependency_graph"]:
                 roles_dict["dependency_graph"][head_idx] = []
             roles_dict["dependency_graph"][head_idx].append((i, deprel))
        elif head_idx == -1: # Identify the root word
             if roles_dict["root"] is None: # Take the first root found
                 roles_dict["root"] = i
             else:
                 logger.warning(f"Multiple roots found? Previous root: {roles_dict['root']}, new candidate: {i} ('{token_text}'). Keeping first.")

    # Handle cases where Stanza might not find a root (e.g., single word)
    if roles_dict["root"] is None and len(processed_analyses) > 0:
        logger.warning("No explicit root found, assigning first token as root.")
        roles_dict["root"] = 0

    # --- Second Pass: Identify Roles (Verb, Subject, Object) ---
    root_idx = roles_dict["root"]
    if root_idx is not None and processed_analyses[root_idx]["upos"] == "VERB":
        roles_dict["verb"] = root_idx # If root is a verb, it's the main verb

    # Find dependents of the root/verb
    dependents_of_root = roles_dict["dependency_graph"].get(root_idx, [])

    for dep_idx, dep_rel_str in dependents_of_root:
        if dep_idx >= len(processed_analyses): continue # Safety check

        # Find Subject
        if dep_rel_str in ["nsubj", "csubj", "nsubj:pass"] and roles_dict["subject"] is None:
            roles_dict["subject"] = dep_idx

        # Find Object
        elif dep_rel_str in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and roles_dict["object"] is None:
            roles_dict["object"] = dep_idx

    # If root is not verb, check if any other token is a verb and assign roles relative to it
    if roles_dict["verb"] is None:
        potential_verbs = [(i, entry) for i, entry in enumerate(processed_analyses) if entry["upos"] == "VERB"]
        if potential_verbs:
            # Heuristic: Pick the first verb found as the main verb
            roles_dict["verb"] = potential_verbs[0][0]
            verb_idx = roles_dict["verb"]
            dependents_of_verb = roles_dict["dependency_graph"].get(verb_idx, [])
            # Re-check for subj/obj relative to this verb
            for dep_idx, dep_rel_str in dependents_of_verb:
                 if dep_idx >= len(processed_analyses): continue
                 if dep_rel_str in ["nsubj", "csubj", "nsubj:pass"] and roles_dict["subject"] is None:
                     roles_dict["subject"] = dep_idx
                 elif dep_rel_str in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and roles_dict["object"] is None:
                     roles_dict["object"] = dep_idx

    # VSO Heuristic: If a verb is found but no subject via deps, check the next word if it's a Noun/Pronoun
    if roles_dict["verb"] is not None and roles_dict["subject"] is None:
        verb_original_idx = roles_dict["verb"]
        if verb_original_idx + 1 < len(processed_analyses):
            potential_subj_entry = processed_analyses[verb_original_idx + 1]
            if potential_subj_entry["upos"] in ["NOUN", "PROPN", "PRON"] and potential_subj_entry["deprel"] != "obj":
                roles_dict["subject"] = verb_original_idx + 1
                logger.debug(f"VSO Heuristic: Assigning token {verb_original_idx+1} ('{potential_subj_entry['text']}') as subject.")


    # --- Determine Structure Type ---
    verb_idx = roles_dict.get("verb")
    subj_idx = roles_dict.get("subject")
    obj_idx = roles_dict.get("object")
    num_verbs = len([i for i, entry in enumerate(processed_analyses) if entry["upos"] == "VERB"])
    structure_type = "OTHER" # Default

    if verb_idx is not None:
        if subj_idx is not None:
            if obj_idx is not None: # V, S, O all present
                if subj_idx < verb_idx and verb_idx < obj_idx: structure_type = "SVO"
                elif verb_idx < subj_idx and subj_idx < obj_idx : structure_type = "VSO" # V S O
                elif verb_idx < obj_idx and obj_idx < subj_idx : structure_type = "VOS" # V O S
                else: structure_type = "VERBAL_COMPLEX_ORDER"
            else: # V, S present, no O
                if subj_idx < verb_idx: structure_type = "SV"
                elif verb_idx < subj_idx: structure_type = "VS"
                else: structure_type = "VERBAL_UNORDERED_SUBJ"
        elif obj_idx is not None: # V, O present, no S
             structure_type = "VO_NO_SUBJ"
        else: # Only V present
             structure_type = "VERBAL_ONLY"
    elif subj_idx is not None: # No V, but S present
        # Check if it's likely nominal predication
        is_nominal_candidate = False
        if root_idx is not None:
            root_pos = processed_analyses[root_idx]["upos"]
            if root_pos in ["NOUN", "PROPN", "ADJ", "PRON"]:
                 # If root is nominal and is NOT the subject itself
                 if root_idx != subj_idx: is_nominal_candidate = True
                 # Or if root IS the subject, check if it has a nominal dependent acting as predicate
                 elif root_idx == subj_idx:
                      root_dependents = roles_dict["dependency_graph"].get(root_idx, [])
                      for dep_idx, dep_rel_str in root_dependents:
                           if dep_idx < len(processed_analyses) and processed_analyses[dep_idx]["upos"] in ["ADJ", "NOUN", "PROPN"]:
                                is_nominal_candidate = True; break
        if is_nominal_candidate: structure_type = "NOMINAL"
        else: structure_type = "SUBJ_NO_VERB_OTHER"
    # else: structure_type remains "OTHER"

    # Refine structure type if multiple verbs detected
    if num_verbs > 1 and not structure_type.startswith("COMPLEX_"):
        structure_type = "COMPLEX_" + structure_type

    roles_dict["structure"] = structure_type

    if debug:
        logger.debug("\n--- Final Analysis ---")
        logger.debug(f" Tokens: {processed_tokens_texts}")
        for i, entry in enumerate(processed_analyses):
            logger.debug(f"  {i}: {entry['text']} ({entry['lemma']}/{entry['upos']}) "
                         f"-> Head: {entry['head']} ({processed_tokens_texts[entry['head']] if entry['head'] != -1 else 'ROOT'}) "
                         f"via '{entry['deprel']}', Feats: {entry['feats_dict']}")
        logger.debug(f" Roles: {roles_dict}")
        logger.debug(f" Detected Structure: {structure_type}")
        logger.debug("----------------------")

    return processed_tokens_texts, processed_analyses, structure_type, roles_dict


# ==================================
# DisCoCat Type Assignment (V2.1 - Minor Refinements)
# ==================================
def assign_discocat_types_v2_1( # Renamed slightly for clarity
    pos: str, dep_rel: str, token_text: str, lemma: str,
    feats: Dict[str, str], # Added feats dictionary
    head_pos: Optional[str], # Added POS tag of the head word
    is_verb_role: bool = False, # Explicit role flags from structure analysis
    is_subj_role: bool = False,
    is_obj_role: bool = False,
    is_nominal_pred_role: bool = False,
    debug: bool = True
) -> Union[Ty, GrammarDiagram, None]:
    """
    Assigns core DisCoCat types based on POS, dependency relation, and grammatical role.
    V2.1: Includes feats and head_pos for potentially more nuanced decisions (though not heavily used yet).
          Prioritizes explicit roles, then dependency, then POS.
    """
    logger.debug(f"Assigning type for '{token_text}' ({lemma}/{pos}), DepRel: '{dep_rel}', HeadPOS: {head_pos}, "
                 f"Roles: V={is_verb_role}, S={is_subj_role}, O={is_obj_role}, NomPred={is_nominal_pred_role}")
    # Log global type status (ensure they are accessible)
    # logger.debug(f"  Global types: DET={DET_TYPE}, ADJ_MOD={ADJ_MOD_TYPE}, ADJ_PRED={ADJ_PRED_TYPE}, "
    #              f"PREP={PREP_FUNCTOR_TYPE}, V_INTR={VERB_INTRANS_TYPE}, V_TR={VERB_TRANS_TYPE}, ADV={ADV_FUNCTOR_TYPE}")

    assigned_entity: Union[Ty, GrammarDiagram, None] = None

    # --- 1. Role-Based Assignment (Highest Priority) ---
    if is_verb_role:
        # Determine transitivity based on whether object role was also assigned
        if is_obj_role: assigned_entity = VERB_TRANS_TYPE # N @ N -> S
        else: assigned_entity = VERB_INTRANS_TYPE   # N -> S
        logger.debug(f"  Decision (Role): Verb Type assigned -> {str(assigned_entity)}")
    elif is_subj_role or is_obj_role:
        # Subjects and Objects are fundamentally Nouns in the grammar
        assigned_entity = N
        logger.debug(f"  Decision (Role): Noun Type assigned (Subj/Obj)")
    elif is_nominal_pred_role:
        if pos == "ADJ": assigned_entity = ADJ_PRED_TYPE # N -> S
        elif pos in ["NOUN", "PROPN"]: assigned_entity = Box(f"NounPred_{lemma}", N, S) # N -> S
        else:
            logger.warning(f"Nominal predicate '{token_text}' has unexpected POS '{pos}'. Assigning N->S Box.")
            assigned_entity = Box(f"NomPred_{lemma}", N, S)
        logger.debug(f"  Decision (Role): Nominal Predicate Type assigned -> {str(assigned_entity)}")

    # --- 2. Dependency-Based Assignment (If no explicit role matched) ---
    if assigned_entity is None:
        if dep_rel in ['nsubj', 'obj', 'iobj', 'dobj', 'nsubj:pass', 'csubj', 'obl', 'obl:arg'] and pos in ["NOUN", "PROPN", "PRON", "X", "NUM"]:
            assigned_entity = N
            logger.debug(f"  Decision (Dep): Noun Type assigned (DepRel '{dep_rel}' for {pos})")
        elif dep_rel == 'amod' and pos == 'ADJ':
            assigned_entity = ADJ_MOD_TYPE # N -> N
            logger.debug(f"  Decision (Dep): Adjective Modifier Type assigned (amod)")
        elif dep_rel == 'det' and pos == 'DET':
            assigned_entity = DET_TYPE # N -> N
            logger.debug(f"  Decision (Dep): Determiner Type assigned (det)")
        elif dep_rel == 'case' and pos == 'ADP':
            assigned_entity = PREP_FUNCTOR_TYPE # N -> N
            logger.debug(f"  Decision (Dep): Preposition Functor Type assigned (case)")
        elif dep_rel == 'advmod':
             if pos == 'ADV': assigned_entity = ADV_FUNCTOR_TYPE # S -> S
             elif pos == 'ADJ': assigned_entity = ADV_FUNCTOR_TYPE # Treat adverbial adjectives as S->S for now
             else: logger.warning(f"advmod relation on non-ADV/ADJ POS '{pos}' for '{token_text}'. Skipping dep assignment.")
             if assigned_entity: logger.debug(f"  Decision (Dep): Adverb Functor Type assigned (advmod for {pos})")

    # --- 3. POS-Based Assignment (Fallback if no role or specific dependency matched) ---
    if assigned_entity is None:
        if pos in ["NOUN", "PROPN", "PRON", "NUM", "X"]: # Treat numerals and unknowns as Nouns by default
            assigned_entity = N
            logger.debug(f"  Decision (POS): Noun Type assigned (POS fallback for {pos})")
        elif pos == "ADJ":
            # Fallback for ADJ: assume modifier unless context suggests otherwise (hard to determine here)
            assigned_entity = ADJ_MOD_TYPE # Default ADJ to modifier (N->N)
            logger.debug(f"  Decision (POS): Adjective Modifier Type assigned (POS fallback)")
        elif pos == "DET":
            # Fallback for DET: assume modifier
            assigned_entity = DET_TYPE # Default DET to modifier (N->N)
            logger.debug(f"  Decision (POS): Determiner Type assigned (POS fallback)")
        elif pos == "ADP":
            assigned_entity = PREP_FUNCTOR_TYPE # N -> N
            logger.debug(f"  Decision (POS): Preposition Functor Type assigned (POS fallback)")
        elif pos == "ADV":
            assigned_entity = ADV_FUNCTOR_TYPE # S -> S
            logger.debug(f"  Decision (POS): Adverb Functor Type assigned (POS fallback)")
        elif pos == "VERB":
             # If a VERB wasn't assigned a role, default to intransitive
             assigned_entity = VERB_INTRANS_TYPE
             logger.warning(f"  Decision (POS): Verb '{token_text}' wasn't assigned a role. Defaulting to Intransitive Type.")

    # --- 4. Handle Ignored POS tags ---
    if assigned_entity is None:
        if pos in ["PUNCT", "SYM", "PART", "CCONJ", "SCONJ", "AUX", "INTJ"]:
            logger.debug(f"  Decision (POS ignore): Explicitly None for POS {pos}")
        else:
            # Final fallback for completely unhandled cases
            logger.warning(f"Unhandled POS/DepRel combination: POS='{pos}', DepRel='{dep_rel}' for '{token_text}'. Defaulting to N.")
            assigned_entity = N

    # --- Final Check and Logging ---
    # Ensure essential functor types were actually defined
    if isinstance(assigned_entity, Box):
        if assigned_entity.name == "DetFunctor" and DET_TYPE is None: assigned_entity = N; logger.error("DET_TYPE is None!")
        if assigned_entity.name == "AdjModFunctor" and ADJ_MOD_TYPE is None: assigned_entity = N; logger.error("ADJ_MOD_TYPE is None!")
        # ... add checks for other Box types if needed ...

    if debug:
        final_type_str = str(assigned_entity) if assigned_entity else "None"
        logger.debug(f"  >> Final Assigned Type for '{token_text}': {final_type_str} (Type: {type(assigned_entity).__name__})")

    return assigned_entity


# ==================================
# Diagram Creation Functions (V3.2 - Refined PP Attachment)
# ==================================

# --- Helper functions for diagram manipulation (find_subwire_index, apply_cup_at_indices) ---
# These are complex and depend heavily on the internal structure of lambeq diagrams.
# Assuming they exist and function correctly from previous versions for now.
# The core logic change is *where* and *how* PPs are attached, not the cup mechanics themselves.
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
        current_offset += len(t) # type: ignore
    return None

def apply_cup_at_indices_v3(diagram: GrammarDiagram,
                          wire1_abs_start_idx: int, wire1_type: Ty, wire1_orig_tok_idx: int,
                          wire2_abs_start_idx: int, wire2_type: Ty, wire2_orig_tok_idx: int,
                          current_full_wire_map: List[Dict[str, Any]] # Assuming this map structure is still relevant
                          ) -> Optional[Tuple[GrammarDiagram, List[Dict[str, Any]]]]:
    """
    Applies swaps to bring wires adjacent and then applies Cup. V3 for wire map.
    (Implementation copied from previous version - assumes correctness for now)
    """
    # --- Implementation from camel_test2.py ---
    logger.debug(f"Attempting Cup V3: Wire1(TokIdx {wire1_orig_tok_idx}) type {wire1_type} at abs_idx {wire1_abs_start_idx}, "
                 f"Wire2(TokIdx {wire2_orig_tok_idx}) type {wire2_type} at abs_idx {wire2_abs_start_idx} on Cod={diagram.cod}")

    if not (wire1_type.r == wire2_type or wire1_type == wire2_type.r):
        logger.error(f"Cannot apply Cup: Main types {wire1_type} and {wire2_type} are not adjoints.")
        return None
    if wire1_abs_start_idx == wire2_abs_start_idx:
        logger.error("Cannot cup a wire with itself (same start index).")
        return None

    n_wires_total = len(diagram.cod)
    len1 = len(wire1_type)
    len2 = len(wire2_type)

    left_main_abs_idx, right_main_abs_idx = min(wire1_abs_start_idx, wire2_abs_start_idx), max(wire1_abs_start_idx, wire2_abs_start_idx)
    left_main_len, right_main_len = (len1, len2) if wire1_abs_start_idx < wire2_abs_start_idx else (len2, len1)
    left_main_type, right_main_type = (wire1_type, wire2_type) if wire1_abs_start_idx < wire2_abs_start_idx else (wire2_type, wire1_type)

    if not (0 <= left_main_abs_idx < right_main_abs_idx < n_wires_total and \
            left_main_abs_idx + left_main_len <= right_main_abs_idx and \
            right_main_abs_idx + right_main_len <= n_wires_total):
        logger.error(f"Invalid indices/lengths for cup: LeftMain({left_main_abs_idx}, len {left_main_len}), "
                     f"RightMain({right_main_abs_idx}, len {right_main_len}) on total {n_wires_total} wires.")
        return None

    perm = []
    left_main_block_indices = list(range(left_main_abs_idx, left_main_abs_idx + left_main_len))
    right_main_block_indices = list(range(right_main_abs_idx, right_main_abs_idx + right_main_len))
    indices_to_be_cupped = set(left_main_block_indices + right_main_block_indices)

    for i in range(n_wires_total):
        if i not in indices_to_be_cupped:
            perm.append(i)
    perm.extend(left_main_block_indices)
    perm.extend(right_main_block_indices)

    logger.debug(f"Calculated individual wire permutation for Swap: {perm}")
    if len(perm) != n_wires_total:
        logger.error(f"Permutation length {len(perm)} does not match total wires {n_wires_total}.")
        return None

    try:
        if perm != list(range(n_wires_total)):
            swap_layer = Swap(diagram.cod, perm)
            permuted_diagram = diagram >> swap_layer
            logger.debug(f"Applied Swap. Diagram after permute cod: {permuted_diagram.cod}")
        else:
            permuted_diagram = diagram
            logger.debug("Permutation is identity, skipping Swap.")
    except Exception as e_perm:
        logger.error(f"Failed to apply Swap permutation {perm} on cod {diagram.cod}: {e_perm}", exc_info=True)
        return None

    try:
        id_wires_len = n_wires_total - (left_main_len + right_main_len)
        id_wires_type = permuted_diagram.cod[:id_wires_len] # type: ignore

        cup_op = Id(id_wires_type) @ Cup(left_main_type, right_main_type)
        logger.debug(f"Applying Cup op: Id({id_wires_type}) @ Cup({left_main_type}, {right_main_type})")
        logger.debug(f"Permuted diagram cod: {permuted_diagram.cod}, Cup op domain: {cup_op.dom}")

        if permuted_diagram.cod != cup_op.dom:
            logger.error(f"Critical Mismatch: Codomain after permute {permuted_diagram.cod} != Cup domain {cup_op.dom}")
            return None

        result_diagram = permuted_diagram >> cup_op
        logger.info(f"Cup({left_main_type}, {right_main_type}) applied. New cod: {result_diagram.cod}")

        # --- Wire Map Update (Simplified Placeholder) ---
        new_wire_map = []
        remaining_units_orig_data = [
            unit for unit in current_full_wire_map
            if unit['orig_idx'] not in (wire1_orig_tok_idx, wire2_orig_tok_idx)
        ]

        if result_diagram.cod == S:
            logger.debug("Diagram reduced to S, wire map is empty or represents S.")
        elif len(result_diagram.cod) == 0:
             logger.debug("Diagram reduced to Ty(), wire map is empty.")
        else:
            logger.warning("Wire map update after cup is complex. Using simplified placeholder logic.")
            temp_offset = 0
            for unit_data in remaining_units_orig_data: # Assumes relative order preserved (not guaranteed)
                updated_unit_data = unit_data.copy()
                main_len = unit_data['main_len']
                feat_len = unit_data.get('feat_len', 0) # Handle potential absence

                updated_unit_data['main_abs_start'] = temp_offset
                updated_unit_data['feat_abs_start'] = temp_offset + main_len
                updated_unit_data['unit_abs_start'] = temp_offset
                updated_unit_data['unit_len'] = main_len + feat_len

                new_wire_map.append(updated_unit_data)
                temp_offset += updated_unit_data['unit_len']

            if temp_offset != len(result_diagram.cod):
                logger.error(f"Reconstructed wire map length {temp_offset} != result diagram cod length {len(result_diagram.cod)}. Map unreliable.")
                # Return None or handle error appropriately
                # For now, return the potentially incorrect map with a warning
                # return None # Option: Fail if map is unreliable

        logger.debug(f"Wire map updated (placeholder). New map size: {len(new_wire_map)}")
        return result_diagram, new_wire_map

    except ValueError as e_cup_val:
        logger.error(f"ValueError during Cup application: {e_cup_val}. Types might be wrong or diagram structure issue.")
        return None
    except Exception as e_cup_unexpected:
        logger.error(f"Unexpected error applying Cup after permutation: {e_cup_unexpected}", exc_info=True)
        return None
    # --- End of apply_cup_at_indices_v3 ---


def create_verbal_sentence_diagram_v3_2( # Renamed for clarity
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]],
    original_indices: List[int], # Indices of tokens included in word_core_types
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_verbal"
) -> Optional[GrammarDiagram]:
    """
    Creates a DisCoCat diagram for verbal sentences.
    V3.2: Refined PP attachment logic based on head POS.
          Assumes N_MOD_BY_N and S_MOD_BY_N are defined globally.
    """
    logger.info(f"Creating verbal diagram V3.2 for: {' '.join(tokens)}")
    # --- Check if essential modifier types are available ---
    if PREP_FUNCTOR_TYPE is None or S_MOD_BY_N is None or N_MOD_BY_N is None:
         logger.error("Cannot create verbal diagram: Essential Prep/Modifier types (PREP_FUNCTOR_TYPE, S_MOD_BY_N, N_MOD_BY_N) are not defined.")
         return None

    # --- Map original indices to their data for easier lookup ---
    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}

    # --- Create initial boxes for arguments and functors ---
    arg_producer_boxes: Dict[int, Box] = {} # {original_idx: Box(Ty(), N)}
    functor_boxes: Dict[int, Box] = {}    # {original_idx: Box(Dom, Cod)}

    for orig_idx, core_entity in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or core_entity is None:
            logger.debug(f"Skipping token at orig_idx {orig_idx}: No analysis or core_entity is None.")
            continue

        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"

        if isinstance(core_entity, Box): # Functors (Verbs, Preps, Adjs, Dets)
            # Create a new Box instance to avoid modifying the global ones
            functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod)
            logger.info(f"  Registered Functor Box for '{box_name}': {core_entity.dom} -> {core_entity.cod}")
        elif isinstance(core_entity, Ty): # Arguments (Nouns, Pronouns)
            arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)
            logger.info(f"  Registered Argument Producer Box for '{box_name}': Ty() -> {core_entity}")
        else:
            logger.error(f"Unknown core_entity type for '{box_name}': {type(core_entity)}")

    # --- Identify Core Components (Verb, Subject, Object) ---
    subj_idx = roles.get('subject')
    verb_idx = roles.get('verb')
    obj_idx = roles.get('object')

    subj_diag = arg_producer_boxes.get(subj_idx) if subj_idx is not None else None
    verb_functor_box = functor_boxes.get(verb_idx) if verb_idx is not None else None
    obj_diag = arg_producer_boxes.get(obj_idx) if obj_idx is not None else None

    logger.debug(f"  Core components: Subj(idx {subj_idx}): {subj_diag}, Verb(idx {verb_idx}): {verb_functor_box}, Obj(idx {obj_idx}): {obj_diag}")

    # --- Basic Clause Construction (SVO, VSO, SV, VS etc.) ---
    # (This part remains similar to v3.1, ensuring types match before composition)
    final_diagram: Optional[GrammarDiagram] = None
    structure_type = roles.get("structure")
    clause_composition_success = False

    try:
        if structure_type == "SVO" and subj_diag and verb_functor_box and obj_diag:
            if subj_diag.cod == N and obj_diag.cod == N and verb_functor_box.dom == (N@N) and verb_functor_box.cod == S:
                final_diagram = (subj_diag @ obj_diag) >> verb_functor_box
                clause_composition_success = True
            else: logger.warning(f"SVO type mismatch: Subj={subj_diag.cod}, Obj={obj_diag.cod}, Verb.dom={verb_functor_box.dom}")
        elif structure_type == "VSO" and subj_diag and verb_functor_box and obj_diag:
             if subj_diag.cod == N and obj_diag.cod == N and verb_functor_box.dom == (N@N) and verb_functor_box.cod == S:
                 # Lambeq composition order is arguments first: (S @ O) >> V
                 final_diagram = (subj_diag @ obj_diag) >> verb_functor_box
                 clause_composition_success = True
             else: logger.warning(f"VSO type mismatch: Subj={subj_diag.cod}, Obj={obj_diag.cod}, Verb.dom={verb_functor_box.dom}")
        elif structure_type == "SV" and subj_diag and verb_functor_box:
            if subj_diag.cod == N and verb_functor_box.dom == N and verb_functor_box.cod == S:
                final_diagram = subj_diag >> verb_functor_box
                clause_composition_success = True
            else: logger.warning(f"SV type mismatch: Subj={subj_diag.cod}, Verb.dom={verb_functor_box.dom}")
        elif structure_type == "VS" and subj_diag and verb_functor_box:
            if subj_diag.cod == N and verb_functor_box.dom == N and verb_functor_box.cod == S:
                 # Lambeq composition order: S >> V
                 final_diagram = subj_diag >> verb_functor_box
                 clause_composition_success = True
            else: logger.warning(f"VS type mismatch: Subj={subj_diag.cod}, Verb.dom={verb_functor_box.dom}")
        elif structure_type == "VERBAL_NO_EXPLICIT_SUBJ" and verb_functor_box:
             if obj_diag and verb_functor_box.dom == N and verb_functor_box.cod == S: # Treat as V O
                 final_diagram = obj_diag >> verb_functor_box
                 clause_composition_success = True
                 logger.info(f"Composed diagram for VO (no explicit subject) structure.")
             else:
                 logger.warning(f"VERBAL_NO_EXPLICIT_SUBJ: Cannot compose diagram for verb type {verb_functor_box.dom}->{verb_functor_box.cod} with object {obj_diag}")
        # Add other structure types if needed (VOS, etc.)

        if clause_composition_success:
            logger.info(f"Successfully composed basic clause for structure '{structure_type}'. Diagram cod: {final_diagram.cod if final_diagram else 'None'}") # type: ignore
        elif verb_functor_box: # If there's a verb but composition failed
             logger.warning(f"Could not compose basic clause for structure '{structure_type}'. Verb: {verb_functor_box.name}")
             # Maybe return just the verb box? Or fail? For now, proceed to modifiers if verb exists.
             # final_diagram = verb_functor_box # Or some representation?
        else:
            logger.error(f"Cannot form verbal diagram: No main verb functor found for index {verb_idx}.")
            return None

    except Exception as e_clause:
        logger.error(f"Error during basic clause composition ({structure_type}): {e_clause}", exc_info=True)
        return None # Stop if basic clause fails

    # --- Modifier Attachment (PPs, Adjectives, Determiners) ---
    # This part is complex. We'll focus on PP attachment refinement first.
    # A full solution requires robust tracking of which box/wire corresponds to which token.

    processed_modifier_indices = set([verb_idx, subj_idx, obj_idx]) # Keep track of used indices

    # --- Refined PP Attachment Logic ---
    if final_diagram and final_diagram.cod == S: # Only attach PPs if we have a sentence diagram
        logger.debug("--- Attaching Prepositional Phrases (PPs) ---")
        # Iterate through potential prepositions (N->N functors)
        for prep_orig_idx, prep_box in functor_boxes.items():
            if prep_orig_idx in processed_modifier_indices: continue # Skip core SVO components

            # Check if it's likely a preposition based on type and maybe name prefix
            is_prep_functor = (isinstance(prep_box, Box) and
                               prep_box.dom == N and prep_box.cod == N)
                               # Optionally check name: and prep_box.name.startswith("PrepFunctor"))

            if not is_prep_functor: continue

            prep_analysis = analysis_map.get(prep_orig_idx)
            if not prep_analysis: continue

            # Find the noun object of this preposition
            noun_obj_orig_idx = None
            # Look for 'obl', 'nmod' dependents of the preposition
            prep_dependents = roles.get('dependency_graph', {}).get(prep_orig_idx, [])
            for dep_idx, dep_rel_str in prep_dependents:
                if dep_rel_str in ['obl', 'obl:arg', 'nmod'] and dep_idx in arg_producer_boxes:
                    noun_obj_orig_idx = dep_idx
                    break
            # If not found, look for nouns where the prep is the 'case' dependent
            if noun_obj_orig_idx is None:
                 for head_idx, dependents in roles.get('dependency_graph', {}).items():
                     for dep_idx, dep_rel_str in dependents:
                         if dep_idx == prep_orig_idx and dep_rel_str == 'case' and head_idx in arg_producer_boxes:
                             noun_obj_orig_idx = head_idx
                             break
                     if noun_obj_orig_idx: break

            if noun_obj_orig_idx is not None and noun_obj_orig_idx not in processed_modifier_indices:
                noun_obj_diag = arg_producer_boxes.get(noun_obj_orig_idx)
                noun_obj_analysis = analysis_map.get(noun_obj_orig_idx)

                if noun_obj_diag and noun_obj_analysis:
                    try:
                        # Compose the PP diagrammatically (Object >> Preposition)
                        pp_diag_composed = noun_obj_diag >> prep_box
                        pp_log_name = f"({noun_obj_diag.name} >> {prep_box.name})"
                        logger.info(f"  Identified potential PP: {pp_log_name} (PrepIdx: {prep_orig_idx}, NounObjIdx: {noun_obj_orig_idx}) -> type {pp_diag_composed.cod}")

                        # Determine attachment target based on the head of the *noun object*
                        head_of_noun_obj_idx = noun_obj_analysis.get('head')
                        attach_to_sentence = False
                        attach_to_noun = False

                        if head_of_noun_obj_idx == verb_idx:
                            # If noun object directly depends on the main verb (e.g., obl relation)
                            attach_to_sentence = True
                            logger.debug(f"    PP Noun Object '{noun_obj_analysis['text']}' head is main verb. Attaching PP to Sentence.")
                        elif head_of_noun_obj_idx is not None and head_of_noun_obj_idx in analysis_map:
                            head_pos = analysis_map[head_of_noun_obj_idx]['upos']
                            if head_pos == 'VERB': # Could be another verb or the main one via different path
                                attach_to_sentence = True
                                logger.debug(f"    PP Noun Object '{noun_obj_analysis['text']}' head is a VERB (idx {head_of_noun_obj_idx}). Attaching PP to Sentence.")
                            elif head_pos == 'NOUN':
                                attach_to_noun = True
                                logger.debug(f"    PP Noun Object '{noun_obj_analysis['text']}' head is a NOUN (idx {head_of_noun_obj_idx}). NP attachment needed (Deferred).")
                            else: # Head is ADP, ADJ, etc. - Treat as potentially adverbial/sentential
                                attach_to_sentence = True
                                logger.debug(f"    PP Noun Object '{noun_obj_analysis['text']}' head is {head_pos} (idx {head_of_noun_obj_idx}). Assuming Sentence attachment.")
                        else: # Head is root or unknown, default to sentence attachment for now
                             attach_to_sentence = True
                             logger.debug(f"    PP Noun Object '{noun_obj_analysis['text']}' head is ROOT or unknown. Assuming Sentence attachment.")

                        # Perform attachment
                        if attach_to_sentence:
                            if final_diagram.cod == S and pp_diag_composed.cod == N and S_MOD_BY_N is not None:
                                logger.info(f"  Attaching PP {pp_log_name} to Sentence diagram using S_MOD_BY_N.")
                                final_diagram = (final_diagram @ pp_diag_composed) >> S_MOD_BY_N # type: ignore
                                logger.info(f"    Successfully attached PP to S. New diagram cod: {final_diagram.cod}")
                                processed_modifier_indices.add(prep_orig_idx)
                                processed_modifier_indices.add(noun_obj_orig_idx)
                            else:
                                logger.warning(f"    Skipping S-attachment of PP {pp_log_name}. Type mismatch or S_MOD_BY_N undefined. "
                                               f"(Diagram Cod: {final_diagram.cod}, PP Cod: {pp_diag_composed.cod}, S_MOD_BY_N: {S_MOD_BY_N})")
                        elif attach_to_noun:
                             # TODO: Implement NP-attachment using N_MOD_BY_N
                             # This requires finding the diagram box corresponding to head_of_noun_obj_idx
                             # and applying: (head_noun_box @ pp_diag_composed) >> N_MOD_BY_N
                             logger.warning(f"  NP-attachment for PP {pp_log_name} modifying Noun {head_of_noun_obj_idx} is not yet implemented. Skipping attachment.")
                             # Optionally mark as processed to avoid re-attempting sentence attachment later
                             # processed_modifier_indices.add(prep_orig_idx)
                             # processed_modifier_indices.add(noun_obj_orig_idx)

                    except Exception as e_pp_attach:
                        logger.error(f"Error forming or attaching PP with {prep_box.name} and {noun_obj_diag.name}: {e_pp_attach}", exc_info=True) # type: ignore
                else:
                     logger.debug(f"Could not find argument box or analysis for noun object (idx {noun_obj_orig_idx}) of prep {prep_orig_idx}.")
            else:
                logger.debug(f"Could not find a suitable noun object for preposition '{prep_analysis['text']}' (idx {prep_orig_idx}) or object already used.")


    # --- Placeholder for Adjective/Determiner Attachment ---
    # Loop through remaining functors (ADJ_MOD_TYPE, DET_TYPE)
    # Find the noun they modify via dependency graph
    # Apply to the corresponding noun box (requires index-to-box mapping)
    logger.warning("Attachment of Adjective/Determiner modifiers within NPs is not fully implemented in this version.")


    # --- Final Normalization and Return ---
    if final_diagram:
        try:
            final_diagram.normal_form()
            if final_diagram.cod == S:
                logger.info(f"Verbal diagram normalization successful for '{sentence_prefix}'. Final cod: {final_diagram.cod}")
                return final_diagram
            else:
                logger.warning(f"Verbal diagram for '{sentence_prefix}' normalized, but final cod is {final_diagram.cod}, not S. Discarding.")
                return None
        except Exception as e_norm:
            logger.error(f"Verbal diagram normal_form failed for '{sentence_prefix}': {e_norm}", exc_info=True)
            return None # Fail if normalization error occurs

    logger.warning(f"Could not form a complete verbal diagram ending in S for structure: {structure_type} for sentence '{sentence_prefix}'.")
    return None


def create_nominal_sentence_diagram_v2_2( # Renamed for clarity
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]],
    original_indices: List[int], # Indices of tokens included in word_core_types
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_nominal"
) -> Optional[GrammarDiagram]:
    """
    Creates a DisCoCat diagram for nominal sentences (Subject-Predicate).
    V2.2: Simplified structure, attaches modifiers after basic predication.
    """
    logger.info(f"Creating nominal diagram V2.2 for: {' '.join(tokens)}")

    # --- Map original indices to their data ---
    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}

    # --- Create initial boxes ---
    arg_producer_boxes: Dict[int, Box] = {} # {original_idx: Box(Ty(), N)}
    functor_boxes: Dict[int, Box] = {}    # {original_idx: Box(Dom, Cod)}

    for orig_idx, core_entity in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or core_entity is None: continue
        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"
        if isinstance(core_entity, Box):
            functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod)
            logger.info(f"  Registered Nominal Functor Box for '{box_name}': {core_entity.dom} -> {core_entity.cod}")
        elif isinstance(core_entity, Ty):
            arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)
            logger.info(f"  Registered Nominal Argument Box for '{box_name}': Ty() -> {core_entity}")
        else: logger.error(f"Unknown core_entity type for nominal '{box_name}': {type(core_entity)}")

    # --- Identify Subject and Predicate ---
    # Subject is often determined by 'nsubj' dep or root if nominal
    subj_idx = roles.get('subject', roles.get('root')) # Prioritize subject role if found
    predicate_idx = None

    # Find the predicate: Look for an N->S functor whose head is the subject, or is the root itself (if root != subject)
    if subj_idx is not None:
        for idx, functor_box in functor_boxes.items():
            if idx == subj_idx: continue # Predicate cannot be the subject itself
            if functor_box.dom == N and functor_box.cod == S:
                pred_analysis = analysis_map.get(idx)
                if pred_analysis:
                    # Check if it's the root (and not the subject) OR if its head is the subject
                    is_root_and_not_subj = (idx == roles.get('root') and idx != subj_idx)
                    is_headed_by_subj = (pred_analysis.get('head') == subj_idx)
                    if is_root_and_not_subj or is_headed_by_subj:
                        predicate_idx = idx
                        logger.info(f"  Found nominal predicate functor: '{functor_box.name}' (idx {predicate_idx})")
                        break # Found the predicate

    subj_diag = arg_producer_boxes.get(subj_idx) if subj_idx is not None else None
    pred_functor_box = functor_boxes.get(predicate_idx) if predicate_idx is not None else None

    logger.debug(f"  Nominal components: Subj(idx {subj_idx}): {subj_diag}, Pred(idx {predicate_idx}): {pred_functor_box}")

    # --- Basic Predication ---
    final_diagram: Optional[GrammarDiagram] = None
    if subj_diag and pred_functor_box:
        if subj_diag.cod == N and pred_functor_box.dom == N and pred_functor_box.cod == S:
            try:
                final_diagram = subj_diag >> pred_functor_box
                logger.info(f"Nominal composition successful for '{sentence_prefix}'. Cod: {final_diagram.cod}")
            except Exception as e:
                logger.error(f"Nominal composition error for '{sentence_prefix}': {e}", exc_info=True)
                return None
        else:
            logger.warning(f"Nominal type mismatch: Subj.cod={subj_diag.cod}, Pred.dom={pred_functor_box.dom}")
            return None # Cannot compose if types don't match
    elif subj_diag and not pred_functor_box:
         logger.warning(f"Nominal sentence for '{sentence_prefix}' has subject but no predicate functor found. Returning subject diagram (type N).")
         # Cannot form a sentence diagram in this case. Return None or handle differently?
         return None # Returning None as it doesn't form a sentence.
    else: # No subject or other issue
         logger.error(f"Cannot form nominal diagram for '{sentence_prefix}': Missing subject or predicate.")
         return None

    # --- Modifier Attachment (Simplified: Attach remaining N->N functors to Subject) ---
    # This is a heuristic and might attach modifiers incorrectly if they modify the predicate.
    if final_diagram and final_diagram.cod == S and subj_idx is not None:
        logger.debug("--- Attaching remaining N->N modifiers (simplified) ---")
        processed_modifier_indices = {subj_idx, predicate_idx}

        # Find the subject box *within* the composed diagram (this is the hard part)
        # For Subject >> Predicate, the subject wire is consumed. Modifiers need to apply *before* predication.
        # Let's rebuild with modifiers applied to subject first.

        # --- Rebuild with Subject Modifiers ---
        logger.debug("Rebuilding nominal diagram to apply modifiers to subject first.")
        modified_subj_diag = subj_diag # Start with the original subject box

        # Apply N->N functors (ADJ_MOD, DET) that modify the subject
        for mod_idx, mod_box in functor_boxes.items():
             if mod_idx not in processed_modifier_indices and mod_box.dom == N and mod_box.cod == N:
                 mod_analysis = analysis_map.get(mod_idx)
                 # Check if this modifier's head is the subject
                 if mod_analysis and mod_analysis.get('head') == subj_idx:
                     try:
                         logger.info(f"  Applying modifier '{mod_box.name}' (idx {mod_idx}) to subject '{modified_subj_diag.name}'.") # type: ignore
                         modified_subj_diag = modified_subj_diag >> mod_box
                         processed_modifier_indices.add(mod_idx)
                     except Exception as e_mod_apply:
                          logger.error(f"    Error applying modifier {mod_box.name} to subject: {e_mod_apply}")

        # Re-compose with the potentially modified subject
        if modified_subj_diag and pred_functor_box:
             if modified_subj_diag.cod == N and pred_functor_box.dom == N and pred_functor_box.cod == S:
                  try:
                      final_diagram = modified_subj_diag >> pred_functor_box
                      logger.info(f"Re-composed nominal diagram with subject modifiers. Final cod: {final_diagram.cod}")
                  except Exception as e_recomp:
                       logger.error(f"Error re-composing nominal diagram: {e_recomp}")
                       final_diagram = None # Failed recomposition
             else:
                  logger.error("Type mismatch after applying subject modifiers.")
                  final_diagram = None
        else: # Should not happen if initial composition worked, but safety check
             final_diagram = None


    # --- Final Normalization and Return ---
    if final_diagram:
        try:
            final_diagram.normal_form()
            if final_diagram.cod == S:
                logger.info(f"Nominal diagram normalization successful for '{sentence_prefix}'. Final cod: {final_diagram.cod}")
                return final_diagram
            else:
                logger.warning(f"Nominal diagram for '{sentence_prefix}' normalized, but final cod is {final_diagram.cod}, not S. Discarding.")
                return None
        except Exception as e_norm:
            logger.error(f"Nominal diagram normal_form failed for '{sentence_prefix}': {e_norm}", exc_info=True)
            return None

    logger.warning(f"Could not form a complete nominal diagram ending in S for sentence '{sentence_prefix}' with subj_idx: {subj_idx}, pred_idx: {predicate_idx}")
    return None


# ==================================
# Main Conversion Function (V2.2 - Using updated types/diagram functions)
# ==================================
def arabic_to_quantum_enhanced_v2_2( # Renamed for clarity
    sentence: str,
    debug: bool = True,
    output_dir: Optional[str] = None,
    ansatz_choice: str = "IQP",
    # Pass ansatz parameters explicitly
    n_layers_iqp: int = 1,
    n_single_qubit_params_iqp: int = 3,
    n_layers_strong: int = 1,
    cnot_ranges: Optional[List[Tuple[int, int]]] = None, # Note: Lambeq uses List[int] or Tuple[int, int]
    discard_qubits_spider: bool = True,
    **kwargs # Catch-all for other/unexpected keyword arguments
) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Dict[str,Any]], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram (using V3.2/V2.2 functions),
    and converts it to a Qiskit QuantumCircuit. V2.2.
    """
    if kwargs:
        logger.warning(f"Function arabic_to_quantum_enhanced_v2_2 received UNEXPECTED keyword arguments: {kwargs}")

    # --- 1. Analyze Sentence ---
    logger.info(f"Analyzing sentence with morph: '{sentence}'")
    try:
        tokens, analyses_details, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug)
        if structure == "ERROR" or not tokens:
            logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
            return None, None, structure, tokens or [], analyses_details or [], roles or {}
        logger.info(f"Analysis complete. Detected structure: {structure}. Roles: {roles}")
    except Exception as e_analyze_main:
        logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
        return None, None, "ERROR", [], [], {}

    # --- 2. Assign Core DisCoCat Types ---
    word_core_types_list = []
    original_indices_for_diagram = []
    filtered_tokens_for_diagram = []

    logger.debug(f"--- Assigning Core Types V2.1 for: '{sentence}' ---")
    for i, analysis_entry in enumerate(analyses_details):
        token_text = analysis_entry['text']
        lemma = analysis_entry['lemma']
        pos = analysis_entry['upos']
        dep_rel = analysis_entry['deprel']
        feats = analysis_entry['feats_dict']
        head_idx = analysis_entry['head']
        head_pos = analyses_details[head_idx]['upos'] if head_idx != -1 and head_idx < len(analyses_details) else None

        # Determine roles explicitly for type assignment function
        is_verb = (i == roles.get('verb'))
        is_subj = (i == roles.get('subject'))
        is_obj = (i == roles.get('object'))
        is_nom_pred = False
        if structure == "NOMINAL":
             # A token is a nominal predicate if it's the root and not the subject,
             # OR if its head is the subject and it's a Noun/Adj/Propn
             subj_idx_nom = roles.get("subject", roles.get("root")) # Subject might be root
             if subj_idx_nom is not None:
                 is_root_and_not_subj = (i == roles.get("root") and i != subj_idx_nom)
                 is_headed_by_subj = (head_idx == subj_idx_nom)
                 if (is_root_and_not_subj or is_headed_by_subj) and pos in ["ADJ", "NOUN", "PROPN"]:
                      is_nom_pred = True

        current_core_type = assign_discocat_types_v2_1( # Use updated function
            pos, dep_rel, token_text, lemma, feats, head_pos,
            is_verb_role=is_verb,
            is_subj_role=is_subj,
            is_obj_role=is_obj,
            is_nominal_pred_role=is_nom_pred,
            debug=debug
        )

        if current_core_type is not None:
            word_core_types_list.append(current_core_type)
            original_indices_for_diagram.append(i) # Store original index
            filtered_tokens_for_diagram.append(token_text)
        else:
            logger.debug(f"  Token '{token_text}' (orig_idx {i}) assigned None core type, excluding.")

    if not filtered_tokens_for_diagram:
        logger.error(f"No valid tokens with core types remained for diagram construction: '{sentence}'")
        return None, None, structure, tokens, analyses_details, roles

    logger.debug(f"Filtered Tokens for Diagram: {filtered_tokens_for_diagram}")
    logger.debug(f"Assigned Word Core Types: {[str(ct) for ct in word_core_types_list]}")
    logger.debug(f"Original Indices for Diagram Tokens: {original_indices_for_diagram}")

    # --- 3. Create DisCoCat Diagram ---
    diagram: Optional[GrammarDiagram] = None
    diagram_creation_error = None
    try:
        logger.info("Creating DisCoCat diagram...")
        safe_sentence_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0]) if sentence else "empty"

        if structure == "NOMINAL":
            diagram = create_nominal_sentence_diagram_v2_2( # Use updated function
                filtered_tokens_for_diagram, analyses_details, roles,
                word_core_types_list, original_indices_for_diagram, debug,
                output_dir=output_dir, sentence_prefix=f"sent_{safe_sentence_prefix}_nominal"
            )
        elif structure != "ERROR": # Attempt verbal for others
            diagram = create_verbal_sentence_diagram_v3_2( # Use updated function
                filtered_tokens_for_diagram, analyses_details, roles,
                word_core_types_list, original_indices_for_diagram, debug,
                output_dir=output_dir, sentence_prefix=f"sent_{safe_sentence_prefix}_verbal"
            )

        if diagram is None:
            logger.error(f"Diagram creation returned None for sentence '{sentence}' with structure '{structure}'.")
        else:
            logger.info(f"Diagram created successfully for '{sentence}'. Final Cod: {diagram.cod}")
            # Optional: Save diagram visualization here if needed
            # visualize_diagram(diagram, save_path=os.path.join(output_dir or ".", f"{safe_sentence_prefix}_diag.png"))

    except Exception as e_diagram:
        logger.error(f"Exception during diagram creation phase for '{sentence}': {e_diagram}", exc_info=True)
        diagram_creation_error = str(e_diagram)
        # Keep diagram as None, return other info

    if diagram is None:
        logger.error("Diagram is None after creation attempt, cannot proceed to circuit conversion.")
        # Return analysis details even if diagram fails
        return None, None, structure, tokens, analyses_details, roles

    # --- 4. Convert Diagram to Quantum Circuit ---
    circuit: Optional[QuantumCircuit] = None
    try:
        logger.info(f"Converting diagram to quantum circuit using ansatz: {ansatz_choice}")
        # Define ob_map (mapping types to qubit counts) - ensure consistency
        ob_map = {N: 1, S: 1} # Basic map: Noun=1 qubit, Sentence=1 qubit
        # Add other types if they are used as distinct atomic types in your grammar
        # e.g., if you had ADJ = AtomicType.ADJECTIVE, you'd add: ob_map[ADJ] = 1

        selected_ansatz = None
        if ansatz_choice.upper() == "IQP":
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
        elif ansatz_choice.upper() == "STRONGLY_ENTANGLING":
            # Check qubit count required by diagram
            num_qubits_required = sum(ob_map.get(obj, 0) for obj in diagram.dom.objects) if diagram.dom else 0
            if num_qubits_required == 0: num_qubits_required = 1 # Handle Ty() case
            # Lambeq's StronglyEntanglingAnsatz takes num_qubits directly
            selected_ansatz = StronglyEntanglingAnsatz(ob_map=ob_map, n_layers=n_layers_strong, num_qubits=num_qubits_required) # Pass num_qubits
        elif ansatz_choice.upper() == "SPIDER":
            selected_ansatz = SpiderAnsatz(ob_map=ob_map, discard_qubits=discard_qubits_spider)
        else:
            logger.warning(f"Unknown ansatz_choice: '{ansatz_choice}'. Defaulting to IQPAnsatz.")
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3)

        if selected_ansatz is None:
            raise ValueError("Ansatz object could not be created.")

        # Apply ansatz to the (already normalized) diagram
        quantum_diagram = selected_ansatz(diagram) # Diagram should be normalized in creation step

        # Convert to Qiskit circuit
        if PYTKET_QISKIT_AVAILABLE:
            tket_circ = quantum_diagram.to_tk()
            circuit = tk_to_qiskit(tket_circ)
            logger.info("Circuit conversion via Tket successful.")
        elif hasattr(quantum_diagram, 'to_qiskit'): # Check for direct qiskit conversion (lambeq >= 0.4.0)
             circuit = quantum_diagram.to_qiskit()
             logger.info("Direct circuit conversion to Qiskit successful.")
        else:
             logger.error("No available method (Tket or direct) to convert Lambeq diagram to Qiskit circuit.")
             circuit = None


    except NotImplementedError as e_nf_main:
        logger.error(f"Diagram normalization or ansatz application failed in main conversion for '{sentence}': {e_nf_main}", exc_info=True)
        # Return analysis details even if circuit fails
        return None, diagram, structure, tokens, analyses_details, roles
    except Exception as e_circuit_outer:
        logger.error(f"Exception during circuit conversion for '{sentence}': {e_circuit_outer}", exc_info=True)
        # Return analysis details even if circuit fails
        return None, diagram, structure, tokens, analyses_details, roles

    if circuit is None:
         logger.error("Circuit conversion resulted in None.")

    return circuit, diagram, structure, tokens, analyses_details, roles


# ==================================
# Visualization Functions (Ensure presence or import)
# ==================================
def visualize_diagram(diagram, save_path=None):
    """Visualizes a Lambeq diagram."""
    if diagram is None or not hasattr(diagram, 'draw'): return None
    try:
        # Simple fixed size for consistency, adjust if needed
        figsize = (max(10, len(diagram.boxes) * 0.8), max(6, len(diagram.dom) * 0.5 + 2))
        ax = diagram.draw(figsize=figsize, fontsize=9, aspect='auto')
        fig = ax.figure
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved diagram to {save_path}")
            except Exception as e_save: logger.error(f"Failed to save diagram: {e_save}")
        # plt.close(fig) # Close after saving
        return fig # Return figure object
    except Exception as e:
        logger.error(f"Could not visualize diagram: {e}", exc_info=True)
        if 'fig' in locals() and fig is not None: plt.close(fig)
        else: plt.close()
        return None

def visualize_circuit(circuit, save_path=None):
    """Visualizes a Qiskit circuit."""
    if circuit is None or not hasattr(circuit, 'draw'): return None
    try:
        # Simple fixed size, adjust if needed
        fig = circuit.draw(output='mpl', fold=-1, scale=0.7)
        if fig:
             fig.set_size_inches(12, max(6, circuit.num_qubits * 0.4)) # Adjust size dynamically
             plt.tight_layout()
             if save_path:
                 try:
                     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                     fig.savefig(save_path, bbox_inches='tight', dpi=150)
                     logger.info(f"Saved circuit to {save_path}")
                 except Exception as e_save: logger.error(f"Failed to save circuit: {e_save}")
             # plt.close(fig) # Close after saving
             return fig # Return figure object
        else:
             logger.warning("Circuit draw did not return a figure object.")
             return None
    except Exception as e:
        logger.error(f"Could not visualize circuit: {e}", exc_info=True)
        if 'fig' in locals() and fig is not None: plt.close(fig)
        else: plt.close()
        return None

# ==================================
# Main Execution / Testing (Example)
# ==================================
if __name__ == "__main__":
    # Configure logging for direct execution
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running camel_test2.py (V3.2/V2.2) directly for testing...")

    test_sentences = [
        # --- Standard Cases ---
        "يقرأ الولد الكتاب",       # VSO (The boy reads the book)
        "الولد يقرأ الكتاب",       # SVO (The boy reads the book)
        "البيت كبير",             # NOMINAL (The house is big)
        "الطالبة الذكية تدرس العلوم بجد", # SVO with modifiers (The smart student studies science diligently)
        "كتبت الطالبة الدرس بسرعة", # VSO with adverb (The student wrote the lesson quickly)
        # --- PP Attachment Cases ---
        "ذهب الولد الى المدرسة صباحا",   # VSO + PP(verb) + Adv(time) (The boy went to school in the morning)
        "رأيت رجلاً في السوق",          # VSO + PP(noun) (I saw a man in the market)
        "الكتاب على الطاولة",           # NOMINAL + PP(noun) (The book is on the table) - Stanza might parse differently
        "الطفل يلعب بالكرة",           # SVO + PP(verb) (The child plays with the ball)
        # --- Determiner/Adjective Cases ---
        "هذا الرجل طبيب",             # DET + N + N_Pred (This man is a doctor)
        "الرجل الطويل جاء",           # N + ADJ_Mod + V (The tall man came)
        # --- Complex ---
        # "الرجل الذي رأيته طبيب", # COMPLEX (The man whom I saw is a doctor) - Requires relative clause handling
    ]

    test_output_dir = "camel_test2_v3_2_output"
    os.makedirs(test_output_dir, exist_ok=True)
    logger.info(f"Test output will be saved to: {test_output_dir}")

    all_results_summary = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n--- Testing Sentence {i+1}/{len(test_sentences)}: '{sentence}' ---")
        safe_sentence_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0]) if sentence else "empty"
        current_sentence_prefix = f"test_{i+1}_{safe_sentence_prefix}"
        current_output_subdir = os.path.join(test_output_dir, current_sentence_prefix)
        os.makedirs(current_output_subdir, exist_ok=True)

        try:
            # Use the updated main conversion function
            result_tuple = arabic_to_quantum_enhanced_v2_2(
                sentence,
                debug=True,
                output_dir=current_output_subdir, # Save outputs per sentence
                ansatz_choice="IQP" # Or other ansatz
                # Add other ansatz params if needed
            )
            circuit, diagram, structure, tokens, analyses, roles = result_tuple

            summary = {
                "sentence": sentence,
                "structure": structure,
                "diagram_ok": diagram is not None,
                "diagram_cod": str(diagram.cod) if diagram else "N/A",
                "circuit_ok": circuit is not None,
                "num_qubits": circuit.num_qubits if circuit else "N/A",
                "error": None
            }
            all_results_summary.append(summary)

            logger.info(f"--- Result for Sentence {i+1}: Structure='{structure}', Diagram OK='{summary['diagram_ok']}', Circuit OK='{summary['circuit_ok']}' ---")
            if diagram: logger.info(f" Diagram Cod: {diagram.cod}")
            if circuit: logger.info(f" Circuit Qubits: {circuit.num_qubits}")

            # Visualize successful diagrams/circuits
            if diagram:
                visualize_diagram(diagram, save_path=os.path.join(current_output_subdir, "diagram.png"))
            if circuit:
                 visualize_circuit(circuit, save_path=os.path.join(current_output_subdir, "circuit.png"))


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
            logger.info(f"  Circuit OK: {res['circuit_ok']} (Qubits: {res.get('num_qubits', 'N/A')})")
    logger.info("="*54)

