# -*- coding: utf-8 -*-
import stanza
from lambeq import AtomicType, IQPAnsatz, SpiderAnsatz # Added SpiderAnsatz
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

# Arabic diacritics (harakat, tanwin, etc):
ARABIC_DIACRITICS = set("ًٌٍَُِّْ")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__) # Use module-level logger
# Set higher level for Lambeq internal logs if desired
# logging.getLogger('lambeq').setLevel(logging.WARNING)

# --- CAMeL Tools Import ---
CAMEL_ANALYZER = None
try:
    from camel_tools.morphology.database import MorphologyDB
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
try:
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse', verbose=False)
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
C = AtomicType.CONJUNCTION # Added for potential use
ADJ = AtomicType.NOUN_PHRASE # Explicit ADJ type

# ==================================
# Linguistic Analysis Function (Unchanged from previous)
# ==================================
def analyze_arabic_sentence(sentence, debug=True):
    """
    Analyzes an Arabic sentence using Stanza for dependency parsing.
    (Code remains the same as provided in promptgem.txt)
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

    tokens = []
    analyses = []
    roles = {"verb": None, "subject": None, "object": None, "root": None, "dependency_graph": {}}

    if not doc.sentences:
         logger.warning(f"Stanza did not find any sentences in: '{sentence}'")
         return [], [], "OTHER", roles

    sent = doc.sentences[0]

    for i, word in enumerate(sent.words):
        tokens.append(word.text)
        head_idx = word.head - 1 if word.head > 0 else -1
        analyses.append((
            word.lemma if word.lemma else word.text,
            word.upos,
            word.deprel,
            head_idx
        ))
        roles["dependency_graph"][i] = []

    for i, (_, _, dep, head) in enumerate(analyses):
        if head >= 0 and head < len(tokens):
            # Ensure the head index exists in the graph keys before appending
            if head not in roles["dependency_graph"]:
                roles["dependency_graph"][head] = [] # Initialize if missing
            roles["dependency_graph"][head].append((i, dep))
        elif head != -1:
             logger.warning(f"Invalid head index {head} for token {i} ('{tokens[i]}'). Skipping dependency edge.")


    if debug:
        logger.debug("\nParsed sentence with dependencies:")
        for i, (token, (lemma, pos, dep, head)) in enumerate(zip(tokens, analyses)):
            logger.debug(f"  {i}: Token='{token}', Lemma='{lemma}', POS='{pos}', Dep='{dep}', Head={head}")

    # Determine sentence structure and roles
    for i, (_, _, dep, head) in enumerate(analyses):
        if dep == "root" or head == -1:
            roles["root"] = i
            break
    if roles["root"] is None and len(analyses) > 0:
        logger.warning("No 'root' dependency found, assigning first token as root as fallback.")
        roles["root"] = 0

    potential_verbs = [i for i, (_, pos, _, _) in enumerate(analyses) if pos == "VERB"]

    if roles["root"] is not None and analyses[roles["root"]][1] == "VERB":
         roles["verb"] = roles["root"]
    elif potential_verbs:
         roles["verb"] = potential_verbs[0]

    if roles["verb"] is not None:
        verb_head_idx = roles["verb"]
        for i, (_, _, dep, head) in enumerate(analyses):
            if head == verb_head_idx:
                if dep in ["nsubj", "csubj"] and roles["subject"] is None:
                     roles["subject"] = i
                elif dep in ["obj", "iobj", "dobj", "ccomp", "xcomp"] and roles["object"] is None:
                     roles["object"] = i
    if roles["subject"] is None and roles["root"] is not None:
         for i, (_, _, dep, head) in enumerate(analyses):
              if head == roles["root"] and dep in ["nsubj", "csubj"]:
                   roles["subject"] = i; break

    verb_idx = roles.get("verb")
    subj_idx = roles.get("subject")
    obj_idx = roles.get("object")

    structure = "OTHER"
    num_verbs = len(potential_verbs)

    if verb_idx is not None:
         if subj_idx is not None:
              if verb_idx < subj_idx: structure = "VSO"
              elif subj_idx < verb_idx: structure = "SVO"
         else: structure = "VERBAL_OTHER"
         if num_verbs > 1: structure = "COMPLEX_" + structure
    elif subj_idx is not None:
         if roles["root"] is not None and analyses[roles["root"]][1] in ["NOUN", "PROPN", "ADJ", "PRON"]:
              structure = "NOMINAL"
    elif num_verbs > 1: structure = "COMPLEX_OTHER"

    roles["structure"] = structure

    if debug:
        logger.debug(f"\nDetected structure: {structure}")
        logger.debug(f"  Verb index: {verb_idx} ({tokens[verb_idx] if verb_idx is not None else 'None'})")
        logger.debug(f"  Subject index: {subj_idx} ({tokens[subj_idx] if subj_idx is not None else 'None'})")
        logger.debug(f"  Object index: {obj_idx} ({tokens[obj_idx] if obj_idx is not None else 'None'})")
        logger.debug(f"  Root index: {roles.get('root')} ({tokens[roles.get('root')] if roles.get('root') is not None else 'None'})")

    return tokens, analyses, structure, roles


# ==================================
# DisCoCat Type Assignment (REVISED - V2 with Dependency Focus)
# ==================================
def assign_discocat_types_v2(
    pos: str, dep_rel: str, token: str, lemma: str,
    is_verb: bool = False, verb_takes_subject: bool = False, verb_takes_object: bool = False,
    is_nominal_pred: bool = False,
    morph_features: Optional[Dict] = None, # Placeholder
    debug: bool = True) -> Optional[Ty]:
    """
    Assigns DisCoCat types based on POS tag, dependency relation, and role.
    V2: Prioritizes dependency relation for key roles (subj, obj, modifiers).
    """
    n = AtomicType.NOUN
    s = AtomicType.SENTENCE
    adj_type = N @ N.l # Adjective modifies Noun (takes N, outputs modified N) - Standard Lambeq
    # Define a simpler preposition type: takes Noun object (right), outputs Noun modifier (left)
    prep_type = N.r >> N.l

    assigned_type = None
    original_pos = pos # Keep original POS for logging
    decision_reason = f"Initial POS='{pos}', Dep='{dep_rel}'"

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

    # 2. Verb Type (If not assigned above)
    elif is_verb:
        domain_list = []
        if verb_takes_subject: domain_list.append(N.r)
        if verb_takes_object: domain_list.append(N.l)

        if not domain_list: domain = Ty()
        elif len(domain_list) == 1: domain = domain_list[0]
        else: domain = Ty.tensor(*domain_list)

        assigned_type = domain >> S
        decision_reason += f" -> is_verb=True => {domain} >> S"

    # 3. Modifiers (High Priority)
    elif dep_rel == "amod": # Adjectival Modifier
        assigned_type = adj_type
        decision_reason += f" -> Dep=amod => {adj_type}"
    elif dep_rel == "advmod": # Adverbial Modifier
        # Simple Sentence modifier type
        assigned_type = S.r >> S.l # Takes S from right, outputs modified S on left
        decision_reason += f" -> Dep=advmod => {assigned_type}"
    elif dep_rel == "nmod": # Nominal Modifier (often PP or possessive)
        # If POS is ADP, use prep_type. Otherwise, maybe noun modifier N @ N.l?
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
        assigned_type = N @ N.l # Determiner modifies Noun
        decision_reason += f" -> POS=DET => {N @ N.l}"

    # 6. Nominal Predicate (Specific case)
    elif is_nominal_pred:
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
            assigned_type = S.r >> S.l
            decision_reason += f" -> POS=ADV (fallback) => {S.r >> S.l}"
        elif pos == "NUM":
            assigned_type = N @ N.l # Number modifies Noun
            decision_reason += f" -> POS=NUM => {N @ N.l}"
        elif pos == "AUX":
             # Treat Aux as identity on S for simplicity, or S @ S.l if it modifies
             assigned_type = S @ S.l
             decision_reason += f" -> POS=AUX => {S @ S.l}"
        elif pos == "CCONJ": # Coordinating Conjunction (e.g., 'and', 'or')
             assigned_type = S.r @ S @ S.l # Connects two sentences
             decision_reason += f" -> POS=CCONJ => {S.r @ S @ S.l}"
        elif pos == "SCONJ": # Subordinating Conjunction (e.g., 'because', 'if')
             assigned_type = S.r >> S # Takes Sentence, outputs Sentence
             decision_reason += f" -> POS=SCONJ => {S.r >> S}"
        elif pos == "PART": # Particles (negation, future, etc.)
             # Simple sentence modifier for now
             assigned_type = S.r >> S
             decision_reason += f" -> POS=PART => {S.r >> S}"
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


# ==================================
# Diagram Creation Functions (REVISED - V3 with Dependency Composition)
# ==================================

def create_nominal_sentence_diagram_v2(tokens: List[str], analyses: List[Tuple], roles: Dict, word_types: List[Ty], original_indices: List[int], debug: bool = True) -> Optional[GrammarDiagram]:
    """
    Create DisCoCat diagram for nominal sentences (Subject + Predicate).
    V2: More robust check for N and N.l >> S predicate type.
    """
    logger.info("Attempting to create diagram for NOMINAL sentence (v2)...")
    if not word_types or len(tokens) != len(word_types) or len(tokens) != len(original_indices):
        logger.error("Mismatch/empty lists in create_nominal_v2.")
        return None
    if not all(isinstance(wt, Ty) for wt in word_types if wt is not None): # Allow None types initially
        logger.error(f"Invalid item in word_types: {word_types}")
        return None

    # Create boxes only for non-None types
    word_boxes = []
    box_map_nominal = {} # Map original_index -> (Box, list_index)
    current_box_index = 0
    for i, orig_idx in enumerate(original_indices):
        wt = word_types[i]
        if wt is not None:
            # Split arrow types if needed
            dom_type, cod_type = (wt.dom, wt.cod) if hasattr(wt, 'dom') and hasattr(wt, 'cod') else (Ty(), wt)
            box = Box(tokens[i], dom_type, cod_type)
            word_boxes.append(box)
            box_map_nominal[orig_idx] = (box, current_box_index)
            current_box_index += 1
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
                if pred_box.cod == (N.l >> S):
                    predicate_orig_idx = p_idx
                    logger.info(f"Found potential predicate '{pred_box.name}' (orig_idx {p_idx}) with type {pred_box.cod}")
                    break

    cup_applied = False
    if subj_orig_idx in box_map_nominal and predicate_orig_idx in box_map_nominal:
        subj_box, subj_list_idx = box_map_nominal[subj_orig_idx]
        pred_box, pred_list_idx = box_map_nominal[predicate_orig_idx]

        # Check types directly
        if subj_box.cod == N and pred_box.cod == (N.l >> S):
            logger.info(f"Attempting nominal Cup: Subject='{subj_box.name}' (N), Predicate='{pred_box.name}' (N.l >> S)")
            # --- Simplified Composition: Assume subj and pred are the main components ---
            # This avoids complex wire tracking for nominal cases
            try:
                # Try direct composition if types match
                # This assumes the order might be Subj @ Pred or Pred @ Subj
                if subj_list_idx < pred_list_idx:
                    # Create diagram with just subj and pred, apply cup
                    temp_diag = subj_box @ pred_box >> Cup(N, N.l)
                    # How to integrate back? Difficult without full tracking.
                    # Fallback: Just return the cupped pair forced to S
                    final_diagram = temp_diag
                    if final_diagram.cod != S: final_diagram >>= Box("Force_S_NominalCup", final_diagram.cod, S)
                    cup_applied = True
                    logger.info("Applied simplified nominal cup. Result forced to S.")
                    return final_diagram # Return the simplified result
                else: # Predicate comes first
                     temp_diag = pred_box @ subj_box # Order matters for >>
                     # Need Swap then Cup: pred @ subj >> Swap(N.l >> S, N) >> Id(N) @ Swap(N.l >> S, Ty()) >> Cup(N, N.l) ... complex
                     logger.warning("Predicate before Subject in nominal tensor - complex swap needed. Falling back.")

            except Exception as e_cup_nom:
                logger.warning(f"Nominal Cup application failed: {e_cup_nom}. Falling back.")
        else:
            logger.warning(f"Type mismatch for nominal cup: Subj={subj_box.cod}, Pred={pred_box.cod}")

    # Fallback if cup wasn't applied or failed
    if not cup_applied:
        logger.warning("Nominal Cup condition not met or failed. Using tensor product and forcing.")
        final_diagram = composition_diagram # The initial tensor product

    # Force to S if needed
    if final_diagram.cod != S:
        logger.warning(f"Final nominal diagram cod is {final_diagram.cod}. Forcing to S.")
        try:
            if final_diagram.cod != Ty():
                # Use Spider if codomain is just multiple S, otherwise Box
                if all(t == S for t in final_diagram.cod.objects):
                    final_diagram >>= Spider(S, len(final_diagram.cod.objects), 1)
                else:
                    final_diagram >>= Box("Force_S_NominalFallback", final_diagram.cod, S)
            else:
                final_diagram = Id(S) # Handle empty codomain case
        except Exception as e_force:
            logger.error(f"Could not force nominal diagram to S type: {e_force}.", exc_info=True)
            # Return the diagram before forcing if forcing fails
            return composition_diagram if not cup_applied else None

    logger.info(f"Final nominal diagram created with cod: {final_diagram.cod}")
    return final_diagram


def find_subwire_index_v2(main_wire_type: Ty, sub_type: Ty, wire_start_index: int, path: List[int] = None) -> Optional[Tuple[int, List[int]]]:
    """
    Finds the absolute start index and path of a sub_type within a main_wire_type's codomain.
    Returns (absolute_start_index, path_list) or None if not found.
    Path list contains indices to navigate composite types.
    """
    if path is None: path = []

    if main_wire_type == sub_type:
        return wire_start_index, path

    # Check if main_wire_type is composite (has .objects)
    if not hasattr(main_wire_type, 'objects') or not main_wire_type.objects:
        return None # Not composite or empty

    current_offset = wire_start_index
    for i, t in enumerate(main_wire_type.objects):
        # Recursively search within the component 't'
        result = find_subwire_index_v2(t, sub_type, current_offset, path + [i])
        if result:
            return result # Found it

        # If not found in composite sub-type 't', update offset by its length
        current_offset += len(t) # len(Ty()) is 0, len(AtomicType) is 1

    return None # Not found in any component


def apply_cup_at_indices_v2(diagram: GrammarDiagram,
                          wire1_info: Tuple[int, Ty, int], # (start_index, type, original_box_index)
                          wire2_info: Tuple[int, Ty, int], # (start_index, type, original_box_index)
                          wire_map: List[Tuple[int, int, int, Ty]] # (start, end, orig_idx, type)
                          ) -> Optional[Tuple[GrammarDiagram, List[Tuple[int, int, int, Ty]]]]:
    """
    Applies swaps to bring wires adjacent and then applies Cup.
    Uses absolute start indices and types. Updates wire map. V2.

    Args:
        diagram: The current diagram.
        wire1_info: Tuple (start_index, type, original_box_index) for the first wire.
        wire2_info: Tuple (start_index, type, original_box_index) for the second wire.
        wire_map: Current wire map [(start_abs, end_abs, original_box_idx, type), ...].

    Returns:
        Tuple (new_diagram, new_wire_map) or None on failure.
    """
    idx1, type1, orig_idx1 = wire1_info
    idx2, type2, orig_idx2 = wire2_info

    logger.debug(f"Attempting Cup V2: Wire1(OrigIdx {orig_idx1}) starts at {idx1}({type1}), Wire2(OrigIdx {orig_idx2}) starts at {idx2}({type2}) on Cod={diagram.cod}")
    logger.debug(f"Current Wire Map: {wire_map}")

    # --- 1. Check Adjoints ---
    if not (type1.r == type2 or type1 == type2.r):
        logger.error(f"Cannot apply Cup: Types {type1} and {type2} are not adjoints.")
        return None
    if idx1 == idx2:
        logger.error("Cannot cup a wire with itself.")
        return None

    # --- 2. Calculate Permutation ---
    n_wires_total = len(diagram.cod)
    len1 = len(type1)
    len2 = len(type2)

    # Ensure index1 refers to the leftmost wire of the pair
    left_idx, right_idx = min(idx1, idx2), max(idx1, idx2)
    left_len, right_len = (len1, len2) if idx1 < idx2 else (len2, len1)
    left_type, right_type = (type1, type2) if idx1 < idx2 else (type2, type1)
    left_orig_idx, right_orig_idx = (orig_idx1, orig_idx2) if idx1 < idx2 else (orig_idx2, orig_idx1)

    # Check indices are valid relative to the current diagram codomain length
    if not (0 <= left_idx < right_idx < n_wires_total and left_idx + left_len <= right_idx and right_idx + right_len <= n_wires_total):
         logger.error(f"Invalid indices/lengths for cup: Left({left_idx}, len {left_len}), Right({right_idx}, len {right_len}) on total {n_wires_total} wires.")
         return None

    # Define the target permutation: [wires_before] + [wires_between] + [wires_after] + [left_wire_block] + [right_wire_block]
    perm = []
    left_block = list(range(left_idx, left_idx + left_len))
    right_block = list(range(right_idx, right_idx + right_len))
    indices_to_move = set(left_block + right_block)

    for i in range(n_wires_total):
        if i not in indices_to_move:
            perm.append(i)

    # Append the blocks to be cupped at the end, in the correct order
    perm.extend(left_block)
    perm.extend(right_block)

    logger.debug(f"Calculated permutation: {perm}")

    # --- 3. Apply Permutation ---
    try:
        # *** FIX: Use Swap layer for permutation ***
        # Check if permutation is non-trivial
        if perm != list(range(n_wires_total)):
            swap_layer = Swap(diagram.cod, perm)
            permuted_diagram = diagram >> swap_layer
            logger.debug(f"Applied Swap. Diagram after permute cod: {permuted_diagram.cod}")
        else:
            permuted_diagram = diagram # No swap needed
            logger.debug("Permutation is identity, skipping Swap.")

    except ValueError as e_perm_val:
         # This might happen if len(perm) doesn't match len(diagram.cod)
         logger.error(f"Failed applying Swap permutation {perm}. Length mismatch? Cod={diagram.cod}. Error: {e_perm_val}")
         return None
    except Exception as e_perm:
        logger.error(f"Failed to apply Swap permutation {perm}: {e_perm}", exc_info=True)
        return None

    # --- 4. Apply Cup ---
    try:
        # Wires to be cupped are now at the end
        id_wires_type = permuted_diagram.cod[:-(left_len + right_len)] # Type of wires before the cupped pair
        # Use the types corresponding to the left/right blocks
        cup_op = Id(id_wires_type) @ Cup(left_type, right_type)

        logger.debug(f"Applying Cup op {Cup(left_type, right_type)} with Id({id_wires_type})")
        logger.debug(f"Permuted diagram cod: {permuted_diagram.cod}")
        logger.debug(f"Cup op domain: {cup_op.dom}")

        if permuted_diagram.cod == cup_op.dom:
            result_diagram = permuted_diagram >> cup_op
            logger.info(f"Cup({left_type}, {right_type}) applied successfully. New cod: {result_diagram.cod}")

            # --- 5. Update Wire Map ---
            new_wire_map = []
            current_offset = 0
            # Iterate through the original wire map, skipping the cupped ones
            for start, end, orig_idx, w_type in wire_map:
                if orig_idx != left_orig_idx and orig_idx != right_orig_idx:
                    w_len = len(w_type)
                    new_start = current_offset
                    new_end = current_offset + w_len
                    new_wire_map.append((new_start, new_end, orig_idx, w_type))
                    current_offset += w_len

            # Verification
            if current_offset != len(result_diagram.cod):
                 logger.error(f"Wire map update length mismatch! Expected {len(result_diagram.cod)}, got {current_offset}.")
                 # Attempt to rebuild map from codomain (less reliable)
                 # return None # Safer to return None if update fails
                 logger.warning("Attempting wire map rebuild from codomain (may be inaccurate).")
                 new_wire_map = []
                 current_offset = 0
                 # This part is tricky - matching remaining boxes to codomain parts
                 # For now, return None to indicate failure
                 return None

            logger.debug(f"Wire map updated successfully. New map: {new_wire_map}")
            return result_diagram, new_wire_map
        else:
            logger.error(f"Cup failed: Codomain after permute {permuted_diagram.cod} != Cup domain {cup_op.dom}")
            return None
    except ValueError as e_cup_val:
         logger.error(f"ValueError during Cup application: {e_cup_val}. Types might be wrong despite check.")
         return None
    except Exception as e_cup:
        logger.error(f"Unexpected error applying Cup after permutation: {e_cup}", exc_info=True)
        return None


def create_verbal_sentence_diagram_v3(tokens: List[str], analyses: List[Tuple], roles: Dict, word_types: List[Ty], original_indices: List[int], svo_order: bool = False, debug: bool = False, output_dir: Optional[str] = None, sentence_prefix: str = "diag") -> Optional[GrammarDiagram]:
    """
    Attempts to create DisCoCat diagrams dynamically using wire tracking (v3).
    Uses filtered tokens, word_types, and original_indices.
    Roles dictionary MUST contain indices relative to the original sentence.
    Includes dependency-based composition attempts.
    """
    logger.info(f"Creating verbal diagram (v3 - Dependency Focus) for: {' '.join(tokens)}")
    logger.debug(f"Word Types (filtered): {word_types}")
    logger.debug(f"Original Indices: {original_indices}")
    logger.debug(f"Roles (original indices): {roles}")

    # --- 1. Input Validation ---
    if not word_types or len(tokens) != len(word_types) or len(tokens) != len(original_indices):
        logger.error("Mismatch between tokens, word_types, or original_indices in create_verbal_v3.")
        return None
    if not all(isinstance(wt, Ty) for wt in word_types if wt is not None):
        logger.error(f"Invalid item found in word_types: {word_types}")
        return None

    # --- 2. Create Initial Tensor Diagram & Box Map ---
    box_map: Dict[int, Tuple[Box, int]] = {} # Map original_index -> (Box, box_list_index)
    word_boxes: List[Box] = []
    current_box_idx = 0
    for i, orig_idx in enumerate(original_indices):
        wt = word_types[i]
        if wt is None: # Skip tokens with None type
            logger.debug(f"Skipping box for token '{tokens[i]}' (orig_idx {orig_idx}) due to None type.")
            continue
        dom_type, cod_type = (wt.dom, wt.cod) if hasattr(wt, 'dom') and hasattr(wt, 'cod') else (Ty(), wt)
        box = Box(tokens[i], dom_type, cod_type)
        word_boxes.append(box)
        box_map[orig_idx] = (box, current_box_idx)
        current_box_idx += 1

    if not word_boxes:
        logger.error("No valid word boxes created for verbal sentence.")
        return None

    current_diagram = Id(Ty())
    for box in word_boxes:
        current_diagram @= box
    logger.debug(f"Initial Tensor Cod: {current_diagram.cod}")

    # Create initial wire map: List of [start_abs_idx, end_abs_idx, original_token_idx, type]
    current_wire_map: List[Tuple[int, int, int, Ty]] = []
    current_offset = 0
    for i, box in enumerate(word_boxes):
         box_cod_len = len(box.cod)
         # Find the original index corresponding to this box
         orig_idx = -1
         for k, v in box_map.items():
             if v[1] == i:
                 orig_idx = k
                 break
         if orig_idx == -1:
              logger.error(f"Could not find original index for box {i} ('{box.name}') in box_map.")
              return None # Critical error

         current_wire_map.append((current_offset, current_offset + box_cod_len, orig_idx, box.cod))
         current_offset += box_cod_len

    if current_offset != len(current_diagram.cod):
         logger.error(f"Initial wire map length mismatch! Expected {len(current_diagram.cod)}, got {current_offset}")
         return None
    logger.debug(f"Initial Wire Map (Start, End, OrigIdx, Type): {current_wire_map}")

    # Optional: Save initial diagram
    if debug and output_dir:
        diag_path = os.path.join(output_dir, f"{sentence_prefix}_0_initial.png")
        visualize_diagram(current_diagram, save_path=diag_path)


    # --- 3. Identify Roles & Process Dependencies ---
    verb_orig_idx = roles.get('verb', -1)
    subj_orig_idx = roles.get('subject', -1)
    obj_orig_idx = roles.get('object', -1)
    root_orig_idx = roles.get('root', -1)
    dep_graph = roles.get('dependency_graph', {})

    # Check if key roles exist in the current diagram (i.e., weren't filtered out)
    verb_box_info = box_map.get(verb_orig_idx)
    subj_box_info = box_map.get(subj_orig_idx)
    obj_box_info = box_map.get(obj_orig_idx)

    if verb_box_info is None:
        logger.warning(f"Verb (original index {verb_orig_idx}) not found in filtered boxes. Attempting dependency composition if possible.")
        # Proceed to dependency composition without SVO/VSO logic

    processed_dependencies: Set[Tuple[int, int]] = set() # Track (head, dependent) pairs

    # --- 4. Apply Compositions Iteratively (Dependency-Driven) ---
    # We can iterate through dependencies. Start with verb/subject/object if available.

    composition_step = 1
    made_change_in_pass = True
    max_passes = len(word_boxes) # Limit passes to prevent infinite loops

    while made_change_in_pass and composition_step <= max_passes:
        made_change_in_pass = False
        logger.debug(f"\n--- Composition Pass {composition_step} ---")

        # --- A. Try Standard Subject-Verb and Verb-Object Cups ---
        if verb_box_info and subj_box_info:
            dep_pair = (verb_orig_idx, subj_orig_idx) # Or (subj, verb)? Check dep graph
            if dep_pair not in processed_dependencies:
                verb_box, _ = verb_box_info
                subj_box, _ = subj_box_info
                logger.info(f"Attempting Subject-Verb Cup (Subj orig_idx {subj_orig_idx}, Verb orig_idx {verb_orig_idx})...")

                cup_subj_type = N
                cup_verb_req_type = N.r # Verb expects subject on the right

                # Find wires using the current map
                subj_map_entry = next((entry for entry in current_wire_map if entry[2] == subj_orig_idx), None)
                verb_map_entry = next((entry for entry in current_wire_map if entry[2] == verb_orig_idx), None)

                if subj_map_entry and verb_map_entry and subj_map_entry[3] == cup_subj_type:
                    subj_wire_start = subj_map_entry[0]
                    verb_wire_start = verb_map_entry[0]
                    verb_full_type = verb_map_entry[3] # Type from map

                    # Find the specific N.r wire within the verb's *domain* (input)
                    # This requires looking at the original box type, not codomain from map
                    verb_input_type = verb_box.dom
                    verb_nr_input_info = find_subwire_index_v2(verb_input_type, cup_verb_req_type, 0) # Find relative index in domain

                    if verb_nr_input_info:
                        # This cup is different - it consumes an input wire of the verb
                        # Requires more complex diagram manipulation (not simple codomain cup)
                        logger.warning("Subject-Verb cup requires domain manipulation - Skipping for now.")
                        # TODO: Implement domain cup if needed (more complex)
                    else:
                        # Check if verb *output* contains N.r (less standard DisCoCat)
                        verb_nr_output_info = find_subwire_index_v2(verb_full_type, cup_verb_req_type, verb_wire_start)
                        if verb_nr_output_info:
                             verb_nr_wire_start, _ = verb_nr_output_info
                             logger.debug(f"Found Subj wire at {subj_wire_start}({cup_subj_type}), Verb N.r output wire at {verb_nr_wire_start}({cup_verb_req_type})")
                             cup_result = apply_cup_at_indices_v2(current_diagram,
                                                                  (subj_wire_start, cup_subj_type, subj_orig_idx),
                                                                  (verb_nr_wire_start, cup_verb_req_type, verb_orig_idx),
                                                                  current_wire_map)
                             if cup_result:
                                 current_diagram, current_wire_map = cup_result
                                 processed_dependencies.add(dep_pair)
                                 made_change_in_pass = True
                                 logger.info("Subject-Verb Cup successful.")
                                 if debug and output_dir: visualize_diagram(current_diagram, os.path.join(output_dir, f"{sentence_prefix}_{composition_step}_subj_verb_cup.png"))
                             else: logger.warning("Subject-Verb cup application failed.")
                        else: logger.warning(f"Type mismatch/wire not found for Subj-Verb Cup: Subj={subj_map_entry[3]}, Verb output={verb_full_type} doesn't contain {cup_verb_req_type}")
                else: logger.warning("Could not find map entries or type mismatch for Subj-Verb.")
                processed_dependencies.add(dep_pair) # Mark as attempted

        if verb_box_info and obj_box_info and not made_change_in_pass: # Try only if previous didn't change
            dep_pair = (verb_orig_idx, obj_orig_idx)
            if dep_pair not in processed_dependencies:
                verb_box, _ = verb_box_info
                obj_box, _ = obj_box_info
                logger.info(f"Attempting Verb-Object Cup (Obj orig_idx {obj_orig_idx}, Verb orig_idx {verb_orig_idx})...")

                cup_obj_type = N
                cup_verb_req_type = N.l # Verb expects object on the left

                obj_map_entry = next((entry for entry in current_wire_map if entry[2] == obj_orig_idx), None)
                verb_map_entry = next((entry for entry in current_wire_map if entry[2] == verb_orig_idx), None)

                if obj_map_entry and verb_map_entry and obj_map_entry[3] == cup_obj_type:
                    obj_wire_start = obj_map_entry[0]
                    verb_wire_start = verb_map_entry[0]
                    verb_full_type = verb_map_entry[3]
                    verb_input_type = verb_box.dom

                    verb_nl_input_info = find_subwire_index_v2(verb_input_type, cup_verb_req_type, 0)

                    if verb_nl_input_info:
                        logger.warning("Verb-Object cup requires domain manipulation - Skipping for now.")
                        # TODO: Implement domain cup if needed
                    else:
                         verb_nl_output_info = find_subwire_index_v2(verb_full_type, cup_verb_req_type, verb_wire_start)
                         if verb_nl_output_info:
                             verb_nl_wire_start, _ = verb_nl_output_info
                             logger.debug(f"Found Verb N.l output wire at {verb_nl_wire_start}({cup_verb_req_type}), Obj wire at {obj_wire_start}({cup_obj_type})")
                             # Cup order: Verb(N.l) then Obj(N)
                             cup_result = apply_cup_at_indices_v2(current_diagram,
                                                                  (verb_nl_wire_start, cup_verb_req_type, verb_orig_idx),
                                                                  (obj_wire_start, cup_obj_type, obj_orig_idx),
                                                                  current_wire_map)
                             if cup_result:
                                 current_diagram, current_wire_map = cup_result
                                 processed_dependencies.add(dep_pair)
                                 made_change_in_pass = True
                                 logger.info("Verb-Object Cup successful.")
                                 if debug and output_dir: visualize_diagram(current_diagram, os.path.join(output_dir, f"{sentence_prefix}_{composition_step}_verb_obj_cup.png"))
                             else: logger.warning("Verb-Object cup application failed.")
                         else: logger.warning(f"Type mismatch/wire not found for Verb-Obj Cup: Verb output={verb_full_type} doesn't contain {cup_verb_req_type}, Obj={obj_map_entry[3]}")
                else: logger.warning("Could not find map entries or type mismatch for Verb-Obj.")
                processed_dependencies.add(dep_pair) # Mark as attempted

        # --- B. Iterate through other dependencies ---
        if not made_change_in_pass: # Only if standard cups didn't apply
            logger.debug("Attempting general dependency compositions...")
            for head_orig_idx, dependents in dep_graph.items():
                if head_orig_idx not in box_map: continue # Skip if head was filtered

                head_map_entry = next((entry for entry in current_wire_map if entry[2] == head_orig_idx), None)
                if not head_map_entry: continue
                head_start, _, _, head_type = head_map_entry

                for dep_orig_idx, dep_rel in dependents:
                    if dep_orig_idx not in box_map: continue # Skip if dependent was filtered
                    dep_pair = (head_orig_idx, dep_orig_idx)
                    if dep_pair in processed_dependencies: continue # Already handled

                    dep_map_entry = next((entry for entry in current_wire_map if entry[2] == dep_orig_idx), None)
                    if not dep_map_entry: continue
                    dep_start, _, _, dep_type = dep_map_entry

                    logger.debug(f"Checking dependency: Head {head_orig_idx} ('{box_map[head_orig_idx][0].name}', {head_type}) -> Dep {dep_orig_idx} ('{box_map[dep_orig_idx][0].name}', {dep_type}) via '{dep_rel}'")

                    # --- Apply Composition Rules based on Types/Rel ---
                    applied_cup = False
                    # Rule 1: Noun modified by Adjective (N @ (N @ N.l)) -> Cup(N, N.l)
                    if head_type == N and dep_type == (N @ N.l): # Adjective modifies Noun
                        logger.debug(f"Applying N + Adj composition for {head_orig_idx} and {dep_orig_idx}")
                        # Adjective type is N @ N.l. We need N.l wire from Adj and N wire from Head.
                        adj_nl_info = find_subwire_index_v2(dep_type, N.l, dep_start)
                        if adj_nl_info:
                            adj_nl_start, _ = adj_nl_info
                            cup_result = apply_cup_at_indices_v2(current_diagram,
                                                                 (head_start, N, head_orig_idx), # Head N wire
                                                                 (adj_nl_start, N.l, dep_orig_idx), # Adj N.l wire
                                                                 current_wire_map)
                            if cup_result: applied_cup = True
                            else: logger.warning("N + Adj cup failed.")
                        else: logger.warning("Could not find N.l wire in Adjective.")

                    # Rule 2: Noun modified by Prepositional Phrase (N @ (N.r >> N.l)) -> ??? Complex
                    # Rule 2 (Simpler): Noun modified by PP (N @ N.l) -> Cup(N, N.l)
                    elif head_type == N and dep_type == (N.r >> N.l): # PP modifies Noun
                         logger.debug(f"Applying N + PP (as N.l modifier) composition for {head_orig_idx} and {dep_orig_idx}")
                         # PP type is N.r >> N.l. We need N.l wire from PP and N wire from Head.
                         pp_nl_info = find_subwire_index_v2(dep_type, N.l, dep_start) # Should be the whole output
                         if pp_nl_info:
                             pp_nl_start, _ = pp_nl_info
                             cup_result = apply_cup_at_indices_v2(current_diagram,
                                                                  (head_start, N, head_orig_idx), # Head N wire
                                                                  (pp_nl_start, N.l, dep_orig_idx), # PP N.l wire
                                                                  current_wire_map)
                             if cup_result: applied_cup = True
                             else: logger.warning("N + PP (as N.l) cup failed.")
                         else: logger.warning("Could not find N.l wire in PP.")


                    # Add more rules here based on common dependency patterns and types

                    if applied_cup:
                        current_diagram, current_wire_map = cup_result
                        processed_dependencies.add(dep_pair)
                        made_change_in_pass = True
                        logger.info(f"Applied dependency composition for {head_orig_idx} -> {dep_orig_idx}.")
                        if debug and output_dir: visualize_diagram(current_diagram, os.path.join(output_dir, f"{sentence_prefix}_{composition_step}_dep_cup_{head_orig_idx}_{dep_orig_idx}.png"))
                        # Break inner loop and restart pass if change was made? Or continue? Continue for now.

        composition_step += 1
        if not made_change_in_pass:
             logger.debug("No composition changes made in this pass.")


    # --- 5. Final Reduction to Sentence Type S ---
    final_diagram = current_diagram
    if final_diagram.cod != S:
        logger.warning(f"Final verbal diagram codomain is {final_diagram.cod}, expected {S}. Attempting to force.")
        try:
            if final_diagram.cod != Ty():
                # Use Spider if codomain is just multiple S, otherwise Box
                if all(t == S for t in final_diagram.cod.objects):
                    final_diagram >>= Spider(S, len(final_diagram.cod.objects), 1)
                    logger.info("Applied final Spider to merge S wires.")
                else:
                    final_diagram >>= Box("Force_S_Verbal", final_diagram.cod, S)
                    logger.info("Applied final Box to force to S.")
            else:
                 final_diagram = Id(S) # Handle empty codomain case
        except Exception as e_force:
            logger.error(f"Could not force verbal diagram to S type: {e_force}. Returning intermediate.", exc_info=True)
            return current_diagram # Return diagram before forcing

    logger.info(f"Verbal diagram composition (v3) finished. Final Cod: {final_diagram.cod}")
    if debug and output_dir: visualize_diagram(final_diagram, os.path.join(output_dir, f"{sentence_prefix}_final.png"))
    return final_diagram


# ==================================
# Main Conversion Function (REVISED - V2 using new types/diagrams)
# ==================================
def arabic_to_quantum_enhanced_v2(sentence: str, debug: bool = True, output_dir: Optional[str] = None) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Tuple], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram (using v3 logic),
    and converts it to a Qiskit QuantumCircuit. V2.
    """
    tokens, analyses, structure, roles = [], [], "ERROR", {}
    diagram = None
    circuit = None
    sentence_prefix = "sentence_" + sentence.split()[0] if sentence else "empty" # Basic prefix

    # 1. Analyze sentence
    try:
        logger.info(f"Analyzing sentence: '{sentence}'")
        tokens, analyses, structure, roles = analyze_arabic_sentence(sentence, debug)
        if structure == "ERROR" or not tokens:
             logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
             if not isinstance(roles, dict): roles = {}
             return None, None, structure, tokens, analyses, roles
        logger.info(f"Analysis complete. Detected structure: {structure}")
        # Ensure roles dict is populated
        if not isinstance(roles, dict): roles = {}

        # Visualize dependency tree if output dir provided
        if debug and output_dir:
             dep_path = os.path.join(output_dir, f"{sentence_prefix}_dep_tree.png")
             visualize_dependency_tree(tokens, analyses, roles, save_path=dep_path)

    except Exception as e_analyze_main:
         logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
         return None, None, "ERROR", tokens, analyses, roles

    # --- 2. Assign Types and Prepare Boxes (Using V2 type assignment) ---
    word_types_list = []
    original_indices_for_types = []
    filtered_tokens_for_diagram = []

    for i, (token, (lemma, pos, dep_rel, head)) in enumerate(zip(tokens, analyses)):
        # Skip pure diacritic or punctuation tokens early
        if all(ch in ARABIC_DIACRITICS for ch in token) or token in string.punctuation:
            logger.debug(f"Skipping token '{token}' (punctuation/diacritic).")
            continue

        is_verb = (i == roles.get('verb'))
        # Refined nominal predicate check: ADJ or NOUN linked to subject
        is_pred = False
        if structure == "NOMINAL":
             subj_idx_orig = roles.get("subject", roles.get("root"))
             if head == subj_idx_orig and pos in ["ADJ", "NOUN"]:
                  is_pred = True
             # Also consider if this token IS the root and is ADJ/NOUN
             elif i == roles.get("root") and pos in ["ADJ", "NOUN"]:
                  is_pred = True


        verb_takes_subj = (roles.get('subject') is not None)
        verb_takes_obj = (roles.get('object') is not None)

        assigned_type = assign_discocat_types_v2(
            pos, dep_rel, token, lemma,
            is_verb=is_verb,
            verb_takes_subject=verb_takes_subj,
            verb_takes_object=verb_takes_obj,
            is_nominal_pred=is_pred,
            debug=debug
        )

        # Store type even if None, diagram creation will handle filtering
        word_types_list.append(assigned_type)
        original_indices_for_types.append(i)
        filtered_tokens_for_diagram.append(token)


    if not filtered_tokens_for_diagram:
        logger.error(f"No valid tokens remained after type assignment for: '{sentence}'")
        return None, None, structure, tokens, analyses, roles

    logger.debug(f"Filtered Tokens for Diagram: {filtered_tokens_for_diagram}")
    logger.debug(f"Assigned Word Types (incl. None): {word_types_list}")
    logger.debug(f"Original Indices for Types: {original_indices_for_types}")

    # 3. Create DisCoCat Diagram (Using V3 verbal or V2 nominal)
    diagram = None
    try:
        logger.info("Creating DisCoCat diagram...")
        if structure == "NOMINAL":
            logger.debug("Using create_nominal_sentence_diagram_v2...")
            diagram = create_nominal_sentence_diagram_v2(
                filtered_tokens_for_diagram, analyses, roles, word_types_list, original_indices_for_types, debug
            )
        elif structure != "ERROR": # Includes SVO, VSO, VERBAL_OTHER, COMPLEX_, OTHER
            logger.debug(f"Using create_verbal_sentence_diagram_v3 for structure '{structure}'...")
            diagram = create_verbal_sentence_diagram_v3(
                filtered_tokens_for_diagram, analyses, roles, word_types_list, original_indices_for_types,
                svo_order=(structure=="SVO"), debug=debug, output_dir=output_dir, sentence_prefix=sentence_prefix
            )
        # Removed explicit 'OTHER' fallback here, as v3 attempts dependency composition

        if diagram is None:
            raise ValueError("Diagram creation function returned None.")
        logger.info(f"Diagram created successfully. Final Cod: {diagram.cod}")
        # Visualize final diagram
        if debug and output_dir:
             diag_path = os.path.join(output_dir, f"{sentence_prefix}_final_diagram.png")
             visualize_diagram(diagram, save_path=diag_path)


    except Exception as e_diagram:
        logger.error(f"Exception during diagram creation phase: {e_diagram}", exc_info=True)
        return None, None, structure, tokens, analyses, roles

    # 4. Convert diagram to quantum circuit
    circuit = None
    try:
        logger.info("Converting diagram to quantum circuit.")
        # Define object map (qubits per atomic type)
        ob_map = {N: 1, S: 1, ADJ: 1} # Assign 1 qubit to N, S, ADJ
        # Note: S:0 was used before, changing to S:1 might increase complexity but could be more expressive

        # Choose an ansatz (IQP or Spider)
        # IQPAnsatz is generally good for parameterization
        # SpiderAnsatz might be better if structure is complex but less params
        ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3)
        # ansatz = SpiderAnsatz(ob_map=ob_map) # Alternative

        # Simplify the diagram before applying ansatz
        # Use `normalize=True` for potentially better simplification
        simplified_diagram = diagram.normal_form()
        logger.debug(f"Simplified diagram: {simplified_diagram}")

        # Apply the ansatz to get the quantum diagram
        quantum_diagram = ansatz(simplified_diagram)
        logger.debug(f"Quantum diagram created. Num qubits: {quantum_diagram.n_qubits}")

        # Convert to Tket circuit, then to Qiskit
        tket_circ = quantum_diagram.to_tk()
        # Optimize circuit using Tket (optional but recommended)
        # from pytket.passes import auto_rebase_pass, auto_squash_pass
        # from pytket.architecture import Architecture # If targeting specific hardware
        # from pytket.extensions.qiskit import qiskit_dag_to_tk
        # qiskit_pass = auto_rebase_pass({'cx', 'rz', 'h', 'rx'}) >> auto_squash_pass({'cx', 'rz', 'h', 'rx'})
        # qiskit_pass.apply(tket_circ)
        # logger.debug("Applied Tket optimization passes.")

        circuit = tk_to_qiskit(tket_circ)
        logger.info("Circuit conversion successful.")
        # Visualize circuit
        if debug and output_dir:
             circ_path = os.path.join(output_dir, f"{sentence_prefix}_circuit.png")
             visualize_circuit(circuit, save_path=circ_path)


    except Exception as e_circuit_outer:
        logger.error(f"Exception during circuit conversion: {e_circuit_outer}", exc_info=True)
        return None, diagram, structure, tokens, analyses, roles

    # Final checks and return
    if circuit is None:
        logger.warning(f"Failed to convert diagram to Qiskit circuit for: '{sentence}'")
    elif not isinstance(circuit, QuantumCircuit):
         logger.error(f"Conversion resulted in unexpected type {type(circuit)}. Expected qiskit.QuantumCircuit.")
         circuit = None # Set to None if type is wrong

    logger.debug(f"Function returning successfully. Circuit type: {type(circuit)}")
    return circuit, diagram, structure, tokens, analyses, roles


# ==================================
# Visualization Functions (Optional but Recommended - Unchanged)
# ==================================
def visualize_dependency_tree(tokens, analyses, roles, save_path=None):
    """ Visualize the dependency tree using networkx and matplotlib. """
    if not tokens or not analyses: return None
    G = nx.DiGraph()
    # Ensure roles is a dictionary
    if not isinstance(roles, dict): roles = {}

    for i, token in enumerate(tokens):
        pos_tag = analyses[i][1] if i < len(analyses) else "UNK"
        node_color = "lightblue" # Default
        if i == roles.get("root"): node_color = "red"
        elif i == roles.get("verb"): node_color = "green"
        elif i == roles.get("subject"): node_color = "orange"
        elif i == roles.get("object"): node_color = "purple"
        G.add_node(i, label=f"{i}:{token}\n({pos_tag})", color=node_color)

    edge_labels = {}
    for i, (_, _, dep, head) in enumerate(analyses):
        if head >= 0 and head < len(tokens): # Ensure head is valid index
            G.add_edge(head, i)
            edge_labels[(head, i)] = dep

    plt.figure(figsize=(max(10, len(tokens)*0.8), 6))
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') # Requires graphviz/pygraphviz
    except:
        pos = nx.spring_layout(G, seed=42) # Fallback layout

    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    labels = nx.get_node_attributes(G, 'label')

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    plt.title("Dependency Parse Tree")
    plt.axis('off')
    fig = plt.gcf() # Get current figure
    if save_path:
        try: plt.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved dependency tree to {save_path}")
        except Exception as e_save: logger.error(f"Failed to save dependency tree: {e_save}")
    plt.close(fig) # Close the figure
    return fig

def visualize_diagram(diagram, save_path=None):
    """ Visualize the DisCoCat diagram. """
    if diagram is None or not hasattr(diagram, 'draw'): return None
    try:
        # Draw returns a matplotlib Axes object
        ax = diagram.draw(figsize=(max(10, len(diagram.boxes)*1.5), 6))
        fig = ax.figure # Get the figure from the axes
        if save_path:
            try: fig.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved diagram to {save_path}")
            except Exception as e_save: logger.error(f"Failed to save diagram: {e_save}")
        plt.close(fig) # Close the figure
        return fig
    except Exception as e:
        logger.error(f"Could not visualize diagram: {e}", exc_info=True)
        return None

def visualize_circuit(circuit, save_path=None):
    """ Visualize the quantum circuit. """
    if circuit is None or not hasattr(circuit, 'draw'): return None
    try:
        fig = circuit.draw(output='mpl', fold=-1)
        if fig: # Check if draw returned a figure
             fig.set_size_inches(max(10, circuit.depth()), max(6, circuit.num_qubits * 0.5))
             plt.tight_layout()
             if save_path:
                 try: fig.savefig(save_path, bbox_inches='tight', dpi=150); logger.info(f"Saved circuit to {save_path}")
                 except Exception as e_save: logger.error(f"Failed to save circuit: {e_save}")
             plt.close(fig) # Close the figure
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
# Main Execution / Testing (Optional)
# ==================================
if __name__ == "__main__":
    logger.info("Running camel_test3.py (Revised) directly for testing...")

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

    test_output_dir = "camel_test3_revised_output"
    os.makedirs(test_output_dir, exist_ok=True)
    logger.info(f"Test output will be saved to: {test_output_dir}")

    all_results = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n--- Testing Sentence {i+1}: '{sentence}' ---")
        sentence_prefix = f"test_{i+1}" # Prefix for saving files
        try:
            # Use the revised main function
            result_tuple = arabic_to_quantum_enhanced_v2(sentence, debug=True, output_dir=test_output_dir)
            circuit, diagram, structure, tokens, analyses, roles = result_tuple

            all_results.append({
                "sentence": sentence, "circuit": circuit, "diagram": diagram,
                "structure": structure, "tokens": tokens, "analyses": analyses, "roles": roles
            })
            logger.info(f"--- Result for Sentence {i+1}: Structure='{structure}', Circuit Type='{type(circuit)}', Diagram Type='{type(diagram)}' ---")

            # Visualizations are now handled inside arabic_to_quantum_enhanced_v2 if output_dir is provided

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
