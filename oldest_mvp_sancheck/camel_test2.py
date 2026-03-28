# -*- coding: utf-8 -*-
import stanza
from lambeq import AtomicType, IQPAnsatz, SpiderAnsatz # Added SpiderAnsatz
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
ADJ = AtomicType.NOUN_PHRASE # Explicit ADJ type (can also be N @ N.l)

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
            if head not in roles["dependency_graph"]:
                roles["dependency_graph"][head] = []
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
# DisCoCat Type Assignment (REVISED - V2.1 with Standard Verb Types)
# ==================================
def assign_discocat_types_v2_fixed( # Renamed to avoid confusion
    pos: str, dep_rel: str, token: str, lemma: str,
    is_verb: bool = False, verb_takes_subject: bool = False, verb_takes_object: bool = False,
    is_nominal_pred: bool = False,
    morph_features: Optional[Dict] = None, # Placeholder
    debug: bool = True) -> Optional[Ty]:
    """
    Assigns DisCoCat types based on POS tag, dependency relation, and role.
    V2.1: Uses standard verb types (e.g., N.r @ N.l >> S) and prioritizes dependency.
    (Code remains the same as provided in promptgem.txt)
    """
    n = AtomicType.NOUN
    s = AtomicType.SENTENCE
    # Standard adjective type: N -> N (or N @ N.l in Lambeq notation)
    adj_type = N @ N.l
    # Standard preposition type: Takes Noun object (right), outputs Noun modifier (left)
    prep_type = N.r >> N.l
    # Standard adverb type: Modifies a sentence S -> S (or S @ S.l)
    adv_type = S @ S.l # Or S.r >> S.l depending on convention

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


# ==================================
# Diagram Creation Functions (REVISED - V3.1 Simplified Composition)
# ==================================

# --- Nominal Diagram Creation (Keep v2 from promptgem.txt, it seems reasonable) ---
def create_nominal_sentence_diagram_v2(tokens: List[str], analyses: List[Tuple], roles: Dict, word_types: List[Ty], original_indices: List[int], debug: bool = True) -> Optional[GrammarDiagram]:
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
def create_verbal_sentence_diagram_v3_fixed( # Renamed
    tokens: List[str], analyses: List[Tuple], roles: Dict, word_types: List[Ty],
    original_indices: List[int], svo_order: bool = False, debug: bool = False,
    output_dir: Optional[str] = None, sentence_prefix: str = "diag"
    ) -> Optional[GrammarDiagram]:
    """
    Attempts to create DisCoCat diagrams using standard Lambeq composition. V3.1
    Uses filtered tokens, word_types, and original_indices.
    Relies on correct standard type assignment (e.g., N.r @ N.l >> S for verbs).
    """
    logger.info(f"Creating verbal diagram (v3.1 - Standard Composition) for: {' '.join(tokens)}")
    logger.debug(f"Word Types (filtered): {word_types}")
    logger.debug(f"Original Indices: {original_indices}")
    logger.debug(f"Roles (original indices): {roles}")

    # --- 1. Input Validation ---
    if not word_types or len(tokens) != len(word_types) or len(tokens) != len(original_indices):
        logger.error("Mismatch between tokens, word_types, or original_indices in create_verbal_v3.1.")
        return None

    # --- 2. Create Boxes (only for non-None types) ---
    word_boxes = []
    box_map: Dict[int, Tuple[Box, int]] = {} # Map original_index -> (Box, box_list_index)
    current_box_idx = 0
    original_indices_in_diagram = [] # Track original indices actually used

    for i, orig_idx in enumerate(original_indices):
        wt = word_types[i]
        if wt is None:
            logger.debug(f"Skipping box for token '{tokens[i]}' (orig_idx {orig_idx}) due to None type.")
            continue

        # Ensure wt is a Ty object before accessing dom/cod
        if not isinstance(wt, Ty):
             logger.error(f"Invalid type found for token '{tokens[i]}': {wt}. Skipping.")
             continue

        dom_type, cod_type = (wt.dom, wt.cod) if hasattr(wt, 'dom') and hasattr(wt, 'cod') else (Ty(), wt)
        # *** NEW: Shape box name if it contains Arabic ***
        box_name = shape_arabic_text(tokens[i])
        box = Box(box_name, dom_type, cod_type)
        word_boxes.append(box)
        box_map[orig_idx] = (box, current_box_idx)
        original_indices_in_diagram.append(orig_idx)
        current_box_idx += 1

    if not word_boxes:
        logger.error("No valid word boxes created for verbal sentence.")
        return None

    # --- 3. Determine Composition Order (SVO/VSO/Other) ---
    # We need the indices *within the word_boxes list* for S, V, O
    verb_list_idx = -1
    subj_list_idx = -1
    obj_list_idx = -1

    verb_orig_idx = roles.get('verb', -1)
    subj_orig_idx = roles.get('subject', -1)
    obj_orig_idx = roles.get('object', -1)

    if verb_orig_idx in box_map: verb_list_idx = box_map[verb_orig_idx][1]
    if subj_orig_idx in box_map: subj_list_idx = box_map[subj_orig_idx][1]
    if obj_orig_idx in box_map: obj_list_idx = box_map[obj_orig_idx][1]

    # Create the initial tensor product based on the order of boxes
    # This assumes word_boxes are already in sentence order
    current_diagram = Id(Ty())
    for box in word_boxes:
        current_diagram @= box
    logger.debug(f"Initial Tensor Cod: {current_diagram.cod}")
    if debug and output_dir: visualize_diagram(current_diagram, os.path.join(output_dir, f"{sentence_prefix}_0_initial.png"))


    # --- 4. Attempt Standard Composition with normal_form ---
    # Lambeq's normal_form applies cups based on types (N @ N.r -> Id(), etc.)
    # This is the preferred way to compose if types are standard.
    try:
        logger.info("Attempting composition using diagram.normal_form()...")
        # Use Rewriter for potentially better simplification control
        # rewriter = Rewriter(['normal_form']) # or ['inner_diagram_normal_form']
        # simplified_diagram = rewriter(current_diagram)
        simplified_diagram = current_diagram.normal_form()

        logger.info(f"Diagram after normal_form(): Cod={simplified_diagram.cod}")
        if debug and output_dir: visualize_diagram(simplified_diagram, os.path.join(output_dir, f"{sentence_prefix}_1_normal_form.png"))

        final_diagram = simplified_diagram

    except Exception as e_nf:
        logger.error(f"Error during normal_form composition: {e_nf}", exc_info=True)
        logger.warning("Falling back to initial tensor diagram.")
        final_diagram = current_diagram # Fallback to the initial tensor

    # --- 5. Final Reduction to Sentence Type S ---
    if final_diagram.cod != S:
        logger.warning(f"Final verbal diagram codomain is {final_diagram.cod}, expected {S}. Attempting to force.")
        try:
            if final_diagram.cod != Ty():
                # Use Spider if codomain is just multiple S, otherwise Box
                if all(t == S for t in final_diagram.cod.objects):
                    final_diagram >>= Spider(S, len(final_diagram.cod.objects), 1)
                    logger.info("Applied final Spider to merge S wires.")
                else:
                    # Check if the codomain is just N (e.g., failed composition)
                    if final_diagram.cod == N:
                         logger.warning("Diagram reduced to N, forcing to S.")
                    # *** NEW: Shape box name if it contains Arabic ***
                    box_name = shape_arabic_text("Force_S_Verbal")
                    final_diagram >>= Box(box_name, final_diagram.cod, S)
                    logger.info("Applied final Box to force to S.")
            else:
                 final_diagram = Id(S) # Handle empty codomain case
        except Exception as e_force:
            logger.error(f"Could not force verbal diagram to S type: {e_force}. Returning intermediate.", exc_info=True)
            return current_diagram # Return diagram before forcing

    logger.info(f"Verbal diagram composition (v3.1) finished. Final Cod: {final_diagram.cod}")
    if debug and output_dir: visualize_diagram(final_diagram, os.path.join(output_dir, f"{sentence_prefix}_final.png"))
    return final_diagram

# Remove find_subwire_index_v2 and apply_cup_at_indices_v2 as they are not needed
# for the simplified standard composition approach.

# ==================================
# Main Conversion Function (REVISED - V2.1 using fixed types/diagrams)
# ==================================
def arabic_to_quantum_enhanced_v2_fixed(sentence: str, debug: bool = True, output_dir: Optional[str] = None) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Tuple], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram (using v3.1 logic),
    and converts it to a Qiskit QuantumCircuit. V2.1.
    """
    tokens, analyses, structure, roles = [], [], "ERROR", {}
    diagram = None
    circuit = None
    # *** NEW: Shape sentence prefix if it contains Arabic ***
    sentence_prefix = "sentence_" + shape_arabic_text(sentence.split()[0]) if sentence else "empty" # Basic prefix

    # 1. Analyze sentence
    try:
        logger.info(f"Analyzing sentence: '{sentence}'")
        tokens, analyses, structure, roles = analyze_arabic_sentence(sentence, debug)
        if structure == "ERROR" or not tokens:
             logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
             if not isinstance(roles, dict): roles = {}
             return None, None, structure, tokens, analyses, roles
        logger.info(f"Analysis complete. Detected structure: {structure}")
        if not isinstance(roles, dict): roles = {}

        if debug and output_dir:
             dep_path = os.path.join(output_dir, f"{sentence_prefix}_dep_tree.png")
             visualize_dependency_tree(tokens, analyses, roles, save_path=dep_path)

    except Exception as e_analyze_main:
         logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
         return None, None, "ERROR", tokens, analyses, roles

    # --- 2. Assign Types and Prepare Boxes (Using FIXED V2.1 type assignment) ---
    word_types_list = []
    original_indices_for_types = []
    filtered_tokens_for_diagram = []

    for i, (token, (lemma, pos, dep_rel, head)) in enumerate(zip(tokens, analyses)):
        if all(ch in ARABIC_DIACRITICS for ch in token) or token in string.punctuation:
            logger.debug(f"Skipping token '{token}' (punctuation/diacritic).")
            continue

        is_verb = (i == roles.get('verb'))
        is_pred = False
        if structure == "NOMINAL":
             subj_idx_orig = roles.get("subject", roles.get("root"))
             if head == subj_idx_orig and pos in ["ADJ", "NOUN"]: is_pred = True
             elif i == roles.get("root") and pos in ["ADJ", "NOUN"]: is_pred = True

        verb_takes_subj = (roles.get('subject') is not None)
        verb_takes_obj = (roles.get('object') is not None)

        # *** USE THE FIXED TYPE ASSIGNMENT FUNCTION ***
        assigned_type = assign_discocat_types_v2_fixed(
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

    # 3. Create DisCoCat Diagram (Using FIXED V3.1 verbal or V2 nominal)
    diagram = None
    try:
        logger.info("Creating DisCoCat diagram...")
        if structure == "NOMINAL":
            logger.debug("Using create_nominal_sentence_diagram_v2...")
            # Pass the filtered lists to the diagram creation function
            diagram = create_nominal_sentence_diagram_v2(
                filtered_tokens_for_diagram, analyses, roles, word_types_list, original_indices_for_types, debug
            )
        elif structure != "ERROR": # Includes SVO, VSO, VERBAL_OTHER, COMPLEX_, OTHER
            logger.debug(f"Using create_verbal_sentence_diagram_v3_fixed for structure '{structure}'...")
            # *** USE THE FIXED VERBAL DIAGRAM FUNCTION ***
            diagram = create_verbal_sentence_diagram_v3_fixed(
                filtered_tokens_for_diagram, analyses, roles, word_types_list, original_indices_for_types,
                svo_order=(structure=="SVO"), debug=debug, output_dir=output_dir, sentence_prefix=sentence_prefix
            )

        if diagram is None:
            raise ValueError("Diagram creation function returned None.")
        logger.info(f"Diagram created successfully. Final Cod: {diagram.cod}")
        # Visualize final diagram
        if debug and output_dir:
             diag_path = os.path.join(output_dir, f"{sentence_prefix}_final_diagram.png")
             visualize_diagram(diagram, save_path=diag_path) # Use the fixed visualize_diagram

    except Exception as e_diagram:
        logger.error(f"Exception during diagram creation phase: {e_diagram}", exc_info=True)
        return None, None, structure, tokens, analyses, roles

    # 4. Convert diagram to quantum circuit
    circuit = None
    try:
        logger.info("Converting diagram to quantum circuit.")
        # Define object map (qubits per atomic type)
        ob_map = {N: 1, S: 1} # Assign 1 qubit to N, S. ADJ/ADV handled by functional types.
                              # S:1 is more standard than S:0 for sentence meaning.

        ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3)

        # Simplify the diagram before applying ansatz (already done in diagram creation)
        # simplified_diagram = diagram.normal_form() # Potentially redundant if v3.1 worked
        simplified_diagram = diagram # Use the diagram from v3.1
        logger.debug(f"Diagram passed to ansatz: {simplified_diagram}")

        # Apply the ansatz to get the quantum diagram
        quantum_diagram = ansatz(simplified_diagram)

        # *** FIX: Check quantum_diagram validity before proceeding ***
        if not isinstance(quantum_diagram, LambeqQuantumDiagram):
             logger.error(f"Ansatz did not return a valid Lambeq Quantum Diagram. Got: {type(quantum_diagram)}")
             raise TypeError("Invalid quantum diagram from ansatz.")
        if not hasattr(quantum_diagram, 'to_tk'):
             logger.error("Quantum diagram object lacks 'to_tk' method.")
             raise AttributeError("Missing 'to_tk' method.")

        logger.debug(f"Quantum diagram created. Attempting to get n_qubits...")
        # Access n_qubits safely
        num_q = getattr(quantum_diagram, 'n_qubits', -1)
        if num_q == -1: logger.warning("Could not retrieve n_qubits from quantum diagram.")
        else: logger.debug(f"Quantum diagram n_qubits: {num_q}")

        # Convert to Tket circuit, then to Qiskit
        tket_circ = quantum_diagram.to_tk()
        circuit = tk_to_qiskit(tket_circ)
        logger.info("Circuit conversion successful.")

        if debug and output_dir:
             circ_path = os.path.join(output_dir, f"{sentence_prefix}_circuit.png")
             visualize_circuit(circuit, save_path=circ_path) # Use the fixed visualize_circuit

    except AttributeError as e_attr:
         logger.error(f"AttributeError during circuit conversion: {e_attr}. Likely issue with diagram structure or Lambeq/Qiskit interaction.", exc_info=False) # Keep traceback minimal for this specific error
         return None, diagram, structure, tokens, analyses, roles
    except TypeError as e_type:
         logger.error(f"TypeError during circuit conversion: {e_type}. Check diagram validity.", exc_info=True)
         return None, diagram, structure, tokens, analyses, roles
    except Exception as e_circuit_outer:
        # Catch the specific 'super' object error if it persists
        if "'super' object has no attribute '__getattr__'" in str(e_circuit_outer):
             logger.error(f"Caught specific 'super __getattr__' error during circuit conversion. This often indicates an issue within Lambeq's quantum backend or version incompatibility.", exc_info=True)
        else:
             logger.error(f"Unexpected exception during circuit conversion: {e_circuit_outer}", exc_info=True)
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
