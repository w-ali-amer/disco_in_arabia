# -*- coding: utf-8 -*-
# camel_test2.py (Revised Version V3.9 - Unconditional Lambeq Imports)
import logging
logger = logging.getLogger(__name__) 

# --- Import shared Lambeq types and Box functors ---
# These are now the single source of truth for these types.
try:
    from common_qnlp_types import (
        N_ARABIC, S_ARABIC, # These are the primary AtomicTypes or Ty fallbacks
        ADJ_MOD_TYPE_ARABIC, ADJ_PRED_TYPE_ARABIC, DET_TYPE_ARABIC,
        PREP_FUNCTOR_TYPE_ARABIC, VERB_INTRANS_TYPE_ARABIC, VERB_TRANS_TYPE_ARABIC,
        S_MOD_BY_N_ARABIC, N_MOD_BY_N_ARABIC, ADV_FUNCTOR_TYPE_ARABIC,
        NOUN_TYPE_BOX_FALLBACK_ARABIC, LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY,
        # Import Lambeq base classes needed for isinstance checks and Box creation
        AtomicType, Ty, Box, GrammarDiagram, Cup, Id, Spider, Swap, Word, Functor,
        # Import Ansaetze
        IQPAnsatz, SpiderAnsatz, StronglyEntanglingAnsatz 
    )
    # Assign to module-level variables for convenience and to match existing code.
    N = N_ARABIC
    S = S_ARABIC
    ADJ_MOD_TYPE = ADJ_MOD_TYPE_ARABIC
    ADJ_PRED_TYPE = ADJ_PRED_TYPE_ARABIC
    DET_TYPE = DET_TYPE_ARABIC
    PREP_FUNCTOR_TYPE = PREP_FUNCTOR_TYPE_ARABIC
    VERB_INTRANS_TYPE = VERB_INTRANS_TYPE_ARABIC
    VERB_TRANS_TYPE = VERB_TRANS_TYPE_ARABIC
    S_MOD_BY_N = S_MOD_BY_N_ARABIC
    N_MOD_BY_N = N_MOD_BY_N_ARABIC
    ADV_FUNCTOR_TYPE = ADV_FUNCTOR_TYPE_ARABIC
    NOUN_TYPE_BOX_FALLBACK = NOUN_TYPE_BOX_FALLBACK_ARABIC
    
    # Make Lambeq classes available directly if needed by other functions in this file
          # For Ty()

    if not LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        logger.warning("camel_test2: common_qnlp_types reported that Lambeq types were NOT initialized successfully. Expect issues with Box functors.")
    else:
        logger.info("camel_test2: Successfully imported types and functors from common_qnlp_types.")

except ImportError as e_common_types_ct2:
    logger.critical(f"camel_test2: CRITICAL - Failed to import from common_qnlp_types.py: {e_common_types_ct2}. This module will not function correctly.", exc_info=True)
    # Define very basic dummies to prevent immediate NameErrors during parsing of the rest of the file.
    class FallbackTyCT2:
        def __init__(self, name): self.name = name
        def __str__(self): return self.name
        def __matmul__(self, other): return self
        def __hash__(self): return hash(self.name)
        def __eq__(self, other): return isinstance(other, FallbackTyCT2) and self.name == other.name #type: ignore
    N = FallbackTyCT2('n_ct2_dummy_NI') # type: ignore
    S = FallbackTyCT2('s_ct2_dummy_NI') # type: ignore
    AtomicType = type(N) # type: ignore 
    Ty = type(N) # type: ignore
    class Box: # type: ignore
        def __init__(self, name: str, dom, cod, data=None, _dagger=False):
            self.name=name; self.dom=dom; self.cod=cod; self.data=data if data is not None else {}
            self._dagger = _dagger
    class GrammarDiagram: pass # type: ignore
    ADJ_MOD_TYPE = ADJ_PRED_TYPE = DET_TYPE = PREP_FUNCTOR_TYPE = None
    VERB_INTRANS_TYPE = VERB_TRANS_TYPE = S_MOD_BY_N = N_MOD_BY_N = ADV_FUNCTOR_TYPE = None
    NOUN_TYPE_BOX_FALLBACK = None


# Import other necessary non-lambeq libraries
import stanza
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import traceback # type: ignore
from qiskit import QuantumCircuit, transpile # type: ignore
from qiskit.circuit import Parameter # type: ignore
from typing import List, Dict, Tuple, Optional, Any, Set, Union 
import os
import hashlib 

# Lambeq specific ansaetze (can also be moved to common_qnlp_types if they are shared)


PYTKET_QISKIT_AVAILABLE = False
try:
    from pytket.extensions.qiskit import tk_to_qiskit # type: ignore
    PYTKET_QISKIT_AVAILABLE = True
    logger.info("camel_test2: Pytket-Qiskit extension found.")
except ImportError:
    logger.warning("camel_test2: Pytket-Qiskit extension not found.")

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

def analyze_arabic_sentence_with_morph(
    sentence: str, debug: bool = False
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Analyzes an Arabic sentence using Stanza for dependency parsing and
    CAMeL Tools for morphological analysis.
    V4.5: Adds attempt to identify NOUN predicates for NOMINAL_NOUN_SUBJ structures.
    """
    global nlp, STANZA_AVAILABLE, CAMEL_ANALYZER
    logger = logging.getLogger('camel_test2')

    # ... (Initial checks for Stanza, sentence empty, nlp(sentence) error handling - as in V4.4) ...
    if not STANZA_AVAILABLE or nlp is None: return [], [], "ERROR_STANZA_INIT", {}
    if not sentence or not sentence.strip(): return [], [], "EMPTY_INPUT", {}
    try: doc = nlp(sentence)
    except Exception: return [], [], "ERROR_STANZA_PROCESSING", {}

    processed_tokens_texts: List[str] = []; processed_analyses: List[Dict[str, Any]] = []
    roles: Dict[str, Any] = {
        "verb": None, "subject": None, "object": None, "root": None, "predicate_idx": None,
        "dependency_graph": {}, "structure": "OTHER_DEFAULT", "main_verb_indices": [], 
        "subject_indices": [], "direct_object_indices": [], "prepositional_object_indices": {}, 
        "all_noun_indices": [], "all_verb_indices": [], "all_adj_indices": [], 
        "all_prep_indices": [], "all_det_indices": [], 
        "analyses_details_for_context": [], "analysis_map_for_diagram_creation": {}
    }
    if not doc.sentences: return [], [], "NO_SENTENCES_STANZA", roles
    sent = doc.sentences[0]; stanza_word_objects = sent.words
    camel_morph_analyses: List[Optional[Dict[str, Any]]] = [None] * len(stanza_word_objects)
    if CAMEL_ANALYZER: # CAMeL Tools Integration (as in V4.4)
        try:
            raw_camel = CAMEL_ANALYZER.analyze(sentence)
            if len(raw_camel) == len(stanza_word_objects):
                for i, ana in enumerate(raw_camel):
                    if ana and isinstance(ana, list) and ana[0]: camel_morph_analyses[i] = ana[0]
            else:
                logger.warning(f"CAMeL/Stanza token mismatch for '{sentence}'. Fallback.")
                for i, sw in enumerate(stanza_word_objects):
                    camel_res = CAMEL_ANALYZER.analyze(sw.text)
                    if camel_res and isinstance(camel_res,list) and camel_res[0] and isinstance(camel_res[0],dict): camel_morph_analyses[i] = camel_res[0]
        except Exception as e: logger.error(f"CAMeL error: {e}", exc_info=debug)

    for i, word in enumerate(stanza_word_objects): # First Pass (as in V4.4)
        ana = {"text": word.text, "lemma": word.lemma or word.text, "upos": word.upos, "deprel": word.deprel,
               "head": word.head-1 if word.head else -1, "original_idx": i,
               "stanza_feats_dict": parse_feats_string(word.feats),
               "camel_analysis": camel_morph_analyses[i] or {},
               "combined_feats_dict": {**parse_feats_string(word.feats), **parse_feats_string((camel_morph_analyses[i]or{}).get('feat'))}}
        processed_tokens_texts.append(word.text); processed_analyses.append(ana)
        if ana["head"] >= 0: roles["dependency_graph"].setdefault(ana["head"], []).append((i, ana["deprel"]))
        if ana["deprel"] == "root" or ana["head"] == -1:
            if roles["root"] is None: roles["root"] = i
        if ana["upos"] == "VERB": roles["all_verb_indices"].append(i)
        elif ana["upos"] == "NOUN": roles["all_noun_indices"].append(i)
        elif ana["upos"] == "ADJ": roles["all_adj_indices"].append(i)
    if roles["root"] is None and processed_analyses: roles["root"] = 0
    elif not processed_analyses: return [], [], "ERROR_NO_TOKENS", roles

    # --- Second Pass V4.5: Root-Driven Role and Structure Identification ---
    root_idx = roles["root"]; final_structure_label = "OTHER_UNCLASSIFIED"
    if root_idx is not None and 0 <= root_idx < len(processed_analyses):
        root_ana = processed_analyses[root_idx]; root_pos = root_ana["upos"]
        root_deps = roles["dependency_graph"].get(root_idx, [])

        if root_pos == "VERB":
            roles["verb"] = root_idx; roles["main_verb_indices"] = [root_idx]
            s_idx, o_idx = None, None
            for dep_i, dep_r in root_deps:
                if dep_r in ["nsubj","nsubj:pass","csubj"] and s_idx is None: s_idx=dep_i
                elif dep_r in ["obj","dobj","ccomp","xcomp"] and o_idx is None: o_idx=dep_i
            roles["subject"], roles["object"] = s_idx, o_idx
            if s_idx is not None: roles["subject_indices"]=[s_idx]
            if o_idx is not None: roles["direct_object_indices"]=[o_idx]
            if s_idx is not None and o_idx is not None: final_structure_label = "SVO"
            elif s_idx is not None: final_structure_label = "SV"
            elif o_idx is not None: final_structure_label = "VO_NO_SUBJ"
            else: final_structure_label = "VERB_ONLY"

        elif root_pos in ["NOUN", "PROPN", "PRON", "NUM"]:
            roles["subject"] = root_idx; roles["subject_indices"] = [root_idx]
            current_nominal_label = "NOMINAL_NOUN_SUBJ"
            predicate_found = False
            for i_tok, tok_ana in enumerate(processed_analyses):
                if i_tok == root_idx: continue # Subject cannot be its own predicate in this primary check
                if tok_ana["upos"] == "ADJ": # Prefer ADJ predicates
                    is_direct = (tok_ana["head"] == root_idx and tok_ana["deprel"] in ["amod", "acl", "xcomp", "conj", "advcl", "nmod", "obl"])
                    ultimate_head = get_ultimate_head(i_tok, processed_analyses)
                    is_indirect = (ultimate_head == root_idx)
                    is_in_rel_clause = False # Simplified from V4.3
                    if tok_ana.get("head", -1) != -1:
                        head_of_adj = processed_analyses[tok_ana["head"]]
                        if head_of_adj["upos"] == "VERB" and head_of_adj["deprel"] == "acl:relcl" and head_of_adj["head"] == root_idx:
                            is_in_rel_clause = True
                    if is_direct or is_indirect or is_in_rel_clause:
                        roles["predicate_idx"] = i_tok
                        current_nominal_label = "NOMINAL_NOUN_SUBJ_ADJ_PRED"; predicate_found = True
                        logger.info(f"  V4.5 Predicate Found (ADJ for NOUN_SUBJ): '{tok_ana['text']}' (idx {i_tok}) for subject '{root_ana['text']}' (idx {root_idx})")
                        break
            if not predicate_found: # If no ADJ predicate, look for NOUN predicate (e.g. "الجمل سفينة")
                for i_tok, tok_ana in enumerate(processed_analyses):
                    if i_tok == root_idx: continue
                    if tok_ana["upos"] in ["NOUN", "PROPN"] and \
                       (tok_ana["head"] == root_idx and tok_ana["deprel"] in ["xcomp", "appos", "nmod", "obl", "conj"]): # Typical deprels for predicate nouns
                        roles["predicate_idx"] = i_tok
                        current_nominal_label = "NOMINAL_NOUN_SUBJ_NOUN_PRED"; predicate_found = True
                        logger.info(f"  V4.5 Predicate Found (NOUN for NOUN_SUBJ): '{tok_ana['text']}' (idx {i_tok}) for subject '{root_ana['text']}' (idx {root_idx})")
                        break
            final_structure_label = current_nominal_label

        elif root_pos == "ADJ":
            roles["predicate_idx"] = root_idx; final_structure_label = "NOMINAL_ADJ_PREDICATE"
            for dep_i, dep_r in root_deps:
                if dep_r == "nsubj" and roles["subject"] is None: roles["subject"]=dep_i; roles["subject_indices"]=[dep_i]; break
            if roles["subject"] is None: final_structure_label = "NOMINAL_ADJ_PRED_NO_SUBJ"
        
        elif root_pos == "X": # (As in V4.3/V4.4)
            for dep_i, dep_r in root_deps:
                if dep_r == "nsubj" and roles["subject"] is None:
                    roles["subject"]=dep_i; roles["subject_indices"]=[dep_i]; roles["predicate_idx"]=root_idx
                    final_structure_label = "NOMINAL_X_PRED_WITH_SUBJ"; break
            if roles["predicate_idx"] is None: final_structure_label = "OTHER_ROOT_X"
        else: final_structure_label = f"OTHER_ROOT_{root_pos}"


    if len(roles["all_verb_indices"]) > 1 and not structure_label.startswith("COMPLEX_") and structure_label not in ["VERBAL_COMPLEX_ORDER"]:
        structure_label = "COMPLEX_" + structure_label
    elif len(roles["all_verb_indices"]) == 0 and structure_label.startswith("VERB"):
        logger.warning(f"Structure labeled verbal ('{structure_label}') but no verbs in 'all_verb_indices'. Reclassifying.")
        if roles["subject"] is not None and roles["predicate_idx"] is not None: structure_label = "NOMINAL_RECLASSIFIED"
        elif roles["subject"] is not None: structure_label = "NOMINAL_SUBJ_ONLY_RECLASSIFIED"
        else: structure_label = "OTHER_NO_VERB_RECLASSIFIED"

    roles["structure"] = structure_label
    roles["analyses_details_for_context"] = processed_analyses
    roles["analysis_map_for_diagram_creation"] = {entry["original_idx"]: entry for entry in processed_analyses}

    if debug:
        logger.debug(f"\n--- Final Analysis (V4.3) for: '{sentence}' ---")
        logger.debug(f" Stanza Tokens: {processed_tokens_texts}")
        logger.debug(f" Determined Structure: {roles['structure']}")
        pred_token_idx = roles.get('predicate_idx')
        pred_token = processed_analyses[pred_token_idx]['text'] if pred_token_idx is not None and 0 <= pred_token_idx < len(processed_analyses) else 'None'
        subj_token = processed_analyses[roles['subject']]['text'] if roles.get('subject') is not None and 0 <= roles['subject'] < len(processed_analyses) else 'None'
        logger.debug(f" Roles: Subject='{subj_token}'(idx {roles.get('subject')}), Predicate='{pred_token}'(idx {pred_token_idx}), Root='{processed_analyses[roles['root']]['text'] if roles.get('root') is not None and 0 <= roles['root'] < len(processed_analyses) else 'None'}'(idx {roles.get('root')})")
        logger.debug("----------------------------------------------------")

    return processed_tokens_texts, processed_analyses, roles["structure"], roles

def get_ultimate_head(token_idx: int, analyses: List[Dict[str,Any]], max_depth=10) -> int:
    """Helper to find the ultimate head of a token by traversing up the dependency tree."""
    current_idx = token_idx
    visited_for_head_search = set()
    # logger = logging.getLogger('camel_test2') # Use existing logger
    # logger.debug(f"      get_ultimate_head: Starting for token_idx {token_idx} ('{analyses[token_idx]['text'] if 0 <= token_idx < len(analyses) else 'OOB'}')")
    for depth in range(max_depth): 
        if current_idx == -1 or current_idx in visited_for_head_search:
            # logger.debug(f"      get_ultimate_head: Stopping at depth {depth}. current_idx: {current_idx}. Visited: {current_idx in visited_for_head_search}")
            break
        visited_for_head_search.add(current_idx)
        if 0 <= current_idx < len(analyses):
            head_of_current = analyses[current_idx].get('head', -1)
            # logger.debug(f"      get_ultimate_head: Depth {depth}, current_idx {current_idx} ('{analyses[current_idx]['text']}'), head_of_current: {head_of_current}")
            if head_of_current == -1: 
                break
            current_idx = head_of_current
        else: 
            # logger.debug(f"      get_ultimate_head: current_idx {current_idx} out of bounds for analyses (len {len(analyses)})")
            break
    # logger.debug(f"      get_ultimate_head: Returning ultimate head {current_idx} for original token_idx {token_idx}")
    return current_idx # Returns -1 if original was root, or the highest valid head found.

# ==================================
# DisCoCat Type Assignment (V2.1 - Unchanged from V3.2)
# ==================================
def assign_discocat_types_v2_2( # Function name kept, but using V2.2.5 logic
    analysis: Dict[str, Any],
    roles: Dict[str, Any],
    debug: bool = True # This debug flag is not currently used by the print statements
) -> Union['Ty', 'GrammarDiagram', None]: # Use string literals for forward references if Ty/GrammarDiagram not defined yet
    """
    Assigns core DisCoCat types.
    V2.2.5: Aggressive predicate typing for root X/ADJ in OTHER/Nominal structures.
            Updated known verb lemmas.
    """
    # --- Make sure these global type variables are accessible ---
    # These would typically be imported from common_qnlp_types.py
    global N, S, VERB_INTRANS_TYPE, VERB_TRANS_TYPE, ADJ_MOD_TYPE, ADJ_PRED_TYPE, DET_TYPE, \
           PREP_FUNCTOR_TYPE, ADV_FUNCTOR_TYPE, NOUN_TYPE_BOX_FALLBACK, Ty, Box, AtomicType, GrammarDiagram
           # Added Box, AtomicType, GrammarDiagram for completeness if they are used

    print(f"\nDEBUG_PRINT: >>> ENTERING assign_discocat_types_v2_2 for token: {analysis.get('text', 'UNK_TEXT')} <<<")
    print(f"DEBUG_PRINT: Input analysis keys: {list(analysis.keys())}")
    print(f"DEBUG_PRINT: Input roles keys: {list(roles.keys())}")

    # --- Aggressive check for global types at the start of each call ---
    print(f"DEBUG_PRINT: Initial Global Type Check: N type: {type(N)}, S type: {type(S)}")
    if any(v is None for v in [N, S]): # N and S are most critical
        logger.critical(f"camel_test2.assign_discocat_types: N ({N}) or S ({S}) is None AT CALL TIME. This is fatal. Bailing out early for token '{analysis.get('text', 'UNK')}'.")
        print(f"DEBUG_PRINT: CRITICAL - N or S is None. Returning None.")
        return None # Cannot proceed if N or S are not valid types

    # Check other global functor types
    functor_types_status = {
        "VERB_INTRANS_TYPE": VERB_INTRANS_TYPE is not None,
        "VERB_TRANS_TYPE": VERB_TRANS_TYPE is not None,
        "ADJ_MOD_TYPE": ADJ_MOD_TYPE is not None,
        "ADJ_PRED_TYPE": ADJ_PRED_TYPE is not None,
        "DET_TYPE": DET_TYPE is not None,
        "PREP_FUNCTOR_TYPE": PREP_FUNCTOR_TYPE is not None,
        "ADV_FUNCTOR_TYPE": ADV_FUNCTOR_TYPE is not None,
        "NOUN_TYPE_BOX_FALLBACK": NOUN_TYPE_BOX_FALLBACK is not None,
    }
    print(f"DEBUG_PRINT: Global Functor Types Status: {functor_types_status}")
    if not all(functor_types_status.values()):
        logger.error("camel_test2.assign_discocat_types: One or more global functorial Box types are None. Attempting re-initialization (simulated here).")
        # Here you might call a re-initialization function if you have one.
        # For debugging, just log their state again.
        print(f"DEBUG_PRINT: Post re-init attempt: VT={VERB_TRANS_TYPE}, VI={VERB_INTRANS_TYPE}, ADJ_MOD={ADJ_MOD_TYPE}")


    token_text = analysis['text']; lemma = analysis['lemma']; pos = analysis['upos']
    dep_rel = analysis['deprel']; original_idx = analysis['original_idx']
    head_idx = analysis['head']
    print(f"DEBUG_PRINT: Token Details: text='{token_text}', lemma='{lemma}', pos='{pos}', dep_rel='{dep_rel}', original_idx={original_idx}, head_idx={head_idx}")

    is_verb_token = (original_idx == roles.get('verb'))
    is_subj_token = (original_idx == roles.get('subject'))
    is_obj_token = (original_idx == roles.get('object'))
    sentence_has_object = (roles.get('object') is not None)
    sentence_structure = roles.get('structure', 'OTHER')
    sentence_root_idx = roles.get('root')
    analyses_details = roles.get('analyses_details_for_context') # Passed from main function
    print(f"DEBUG_PRINT: Role/Sentence Context: is_verb_token={is_verb_token}, is_subj_token={is_subj_token}, is_obj_token={is_obj_token}")
    print(f"DEBUG_PRINT: Sentence Context: has_object={sentence_has_object}, structure='{sentence_structure}', root_idx={sentence_root_idx}")

    is_designated_predicate = (roles.get('predicate_idx') == original_idx)
    print(f"DEBUG_PRINT: Token '{token_text}' (idx {original_idx}) - Is designated predicate? {is_designated_predicate}")

    logger.debug(f"Assigning type V2.2.5 for '{token_text}' ...") # User's existing log

    assigned_entity: Union[Ty, GrammarDiagram, None] = None

    # --- 0. Explicit Verb Lemma Check (Updated List) ---
    known_verb_lemmas = {
        "بَنَى", "رَأَى", "اِنكَسَر", "بُنيَة", "انكسرتْ", # "بُنيَة", "رَأيَة" look like nouns, ensure lemmatization is correct
        "ذهب", "جاء", "قال", "كتب", "قرأ", "شرح", "درس", "عمل", "لعب", "سافر",
        "أصاب", "وجد", "عالج", "وضع", "لمس", "احتاج", "تدفق", "عَيَّن", "أَلمَأ",
        "تَحَدَّث", "أَعطَى", "اِبتَسَم", "أَدرَس", "دَرَّس", "تَقَرَّأ", "تَشَرَّح",
        "رَأيَة", "قَرّ", "شَرِب", "كَتَب", "سَافَر", "جَلَس", "نَامَ", "سَال", "أَهَاب", "أَكَل", "مَثَّل", "وَجَب", "اِكتَسَب", "شَاهَد",
        "تَدَفَّق", "اِحتَاج", "أَولَى", "اِستَقبَل", "حَلَّل", "أَمكَن", "رَكَّب", "اِحتَرَم",
        "أَرفَر", # from يرفرف
        "بَدَد"  # from تبدا
    }
    print(f"DEBUG_PRINT: Section 0: Explicit Verb Lemma Check. is_verb_token={is_verb_token}, lemma='{lemma}' in known_verb_lemmas? {lemma in known_verb_lemmas}")
    if is_verb_token or lemma in known_verb_lemmas:
        logger.debug(f"  Attempting verb type assignment for '{token_text}' (lemma: '{lemma}').") # User's log
        print(f"DEBUG_PRINT:   Attempting verb type assignment for '{token_text}' (lemma: '{lemma}').")
        # Ensure N and S are valid AtomicTypes before creating temporary verb boxes if needed
        # THIS CHECK IS PROBLEMATIC if AtomicType is the class lambeq.AtomicType.
        # N and S are instances of Ty.
        # The corrected check `isinstance(N, Ty) and str(N) == 'n'` is used later and is better.
        # For now, let's see what this original check yields.
        is_N_AtomicType = isinstance(N, AtomicType)
        is_S_AtomicType = isinstance(S, AtomicType)
        print(f"DEBUG_PRINT:     Check `isinstance(N, AtomicType)`: {is_N_AtomicType} (N is {type(N)})")
        print(f"DEBUG_PRINT:     Check `isinstance(S, AtomicType)`: {is_S_AtomicType} (S is {type(S)})")

        # Using the more robust check for Ty('n') and Ty('s')
        is_N_Ty_n = isinstance(N, Ty) and str(N) == 'n'
        is_S_Ty_s = isinstance(S, Ty) and str(S) == 's'
        print(f"DEBUG_PRINT:     Robust Check: is_N_Ty_n={is_N_Ty_n}, is_S_Ty_s={is_S_Ty_s}")

        if not (is_N_Ty_n and is_S_Ty_s): # Using the robust check here
            logger.error(f"  Cannot assign verb/predicate type for '{token_text}': N ('{N}') or S ('{S}') are not valid Ty('n')/Ty('s'). Defaulting to basic assignment.")
            print(f"DEBUG_PRINT:     ERROR - N or S not valid Ty('n')/Ty('s').")
            if analysis.get('upos') == "VERB":
                assigned_entity = Ty('s') if isinstance(S, Ty) else _DummyTy('s_fallback_assign_0')
                print(f"DEBUG_PRINT:       Fallback for VERB: {assigned_entity}")
            else:
                assigned_entity = Ty('n') if isinstance(N, Ty) else _DummyTy('n_fallback_assign_0')
                print(f"DEBUG_PRINT:       Fallback for non-VERB predicate: {assigned_entity}")
        elif sentence_has_object:
            print(f"DEBUG_PRINT:     Verb is transitive (sentence_has_object=True). VERB_TRANS_TYPE is None? {VERB_TRANS_TYPE is None}")
            if VERB_TRANS_TYPE is not None:
                assigned_entity = VERB_TRANS_TYPE
                logger.debug(f"  Decision (Verb Logic): Assigned VERB_TRANS_TYPE to '{token_text}'.") # User's log
                print(f"DEBUG_PRINT:       Assigned VERB_TRANS_TYPE: {assigned_entity}")
            else:
                logger.error(f"  VERB_TRANS_TYPE is None. Creating temporary Box(N@N >> S) for '{token_text}'.") # User's log
                assigned_entity = Box(f"TempVerbTrans_{lemma}_{original_idx}", N @ N, S)
                print(f"DEBUG_PRINT:       VERB_TRANS_TYPE is None. Created temporary Box: {assigned_entity}")
        else: # Intransitive
            print(f"DEBUG_PRINT:     Verb is intransitive (sentence_has_object=False). VERB_INTRANS_TYPE is None? {VERB_INTRANS_TYPE is None}")
            if VERB_INTRANS_TYPE is not None:
                assigned_entity = VERB_INTRANS_TYPE
                logger.debug(f"  Decision (Verb Logic): Assigned VERB_INTRANS_TYPE to '{token_text}'.") # User's log
                print(f"DEBUG_PRINT:       Assigned VERB_INTRANS_TYPE: {assigned_entity}")
            else:
                logger.error(f"  VERB_INTRANS_TYPE is None. Creating temporary Box(N >> S) for '{token_text}'.") # User's log
                assigned_entity = Box(f"TempVerbIntrans_{lemma}_{original_idx}", N, S)
                print(f"DEBUG_PRINT:       VERB_INTRANS_TYPE is None. Created temporary Box: {assigned_entity}")

    if assigned_entity is None and is_designated_predicate:
        print(f"DEBUG_PRINT:   Token '{token_text}' is the designated predicate (idx {original_idx}). POS: {pos}")
        is_N_Ty_n = isinstance(N, Ty) and str(N) == 'n'
        is_S_Ty_s = isinstance(S, Ty) and str(S) == 's'
        if not (is_N_Ty_n and is_S_Ty_s):
            logger.error(f"Cannot assign predicate type for '{token_text}': N or S not valid Ty('n')/Ty('s'). Defaulting to N.")
            assigned_entity = N if is_N_Ty_n else _LocalTy('n_fallback_pred')
        elif pos == "ADJ":
            print(f"DEBUG_PRINT:     Designated predicate is ADJ. ADJ_PRED_TYPE is None? {ADJ_PRED_TYPE is None}")
            if ADJ_PRED_TYPE is not None: assigned_entity = ADJ_PRED_TYPE
            else: assigned_entity = Box(f"TempAdjPred_{lemma}_{original_idx}", N, S)
        elif pos in ["NOUN", "PROPN", "NUM", "X"]: # X can be a predicate in some nominals
            print(f"DEBUG_PRINT:     Designated predicate is {pos}. Creating NounPred Box.")
            assigned_entity = Box(f"NounPred_{lemma}_{original_idx}", N, S)
        else:
            logger.warning(f"Designated predicate '{token_text}' has unexpected POS '{pos}'. Defaulting to N->S Box.")
            assigned_entity = Box(f"TempGenericPred_{lemma}_{original_idx}", N, S)
        
        if assigned_entity: print(f"DEBUG_PRINT:     Assigned by designated_predicate logic: {assigned_entity}")


    # --- 1. Role-Based Assignment (If not assigned by lemma) ---
    print(f"DEBUG_PRINT: Section 1: Role-Based Assignment. assigned_entity is None? {assigned_entity is None}. is_verb_token? {is_verb_token}")
    if assigned_entity is None:
        if is_verb_token:
            print(f"DEBUG_PRINT:   Processing role-based for VERB token '{token_text}'. Lemma: '{lemma}'")
            strongly_transitive_lemmas = {"بَنَى", "أَعطَى", "كَتَب", "قَرَأ", "شَرَح", "فَحَص", "أَكَّل", "لَمَس", "عَالَج"}
            strongly_intransitive_lemmas = {"جَاء", "ذَهَب", "جَلَس", "سَافَر", "نَامَ", "اِنكَسَر"}

            if lemma in strongly_transitive_lemmas:
                print(f"DEBUG_PRINT:     Lemma '{lemma}' in strongly_transitive_lemmas. VERB_TRANS_TYPE is None? {VERB_TRANS_TYPE is None}")
                if VERB_TRANS_TYPE is not None:
                    assigned_entity = VERB_TRANS_TYPE
                    logger.debug(f"  Decision (Lemma Heuristic): Assigned Transitive Verb Type for lemma '{lemma}'.") # User's log
                    print(f"DEBUG_PRINT:       Assigned VERB_TRANS_TYPE by strong lemma.")
                else:
                    logger.error(f"VERB_TRANS_TYPE is None. Cannot assign for transitive lemma '{lemma}'. Defaulting to N.") # User's log
                    assigned_entity = N
                    print(f"DEBUG_PRINT:       ERROR - VERB_TRANS_TYPE is None for strong transitive lemma. Defaulted to N.")
            elif lemma in strongly_intransitive_lemmas:
                print(f"DEBUG_PRINT:     Lemma '{lemma}' in strongly_intransitive_lemmas. VERB_INTRANS_TYPE is None? {VERB_INTRANS_TYPE is None}")
                if VERB_INTRANS_TYPE is not None:
                    assigned_entity = VERB_INTRANS_TYPE
                    logger.debug(f"  Decision (Lemma Heuristic): Assigned Intransitive Verb Type for lemma '{lemma}'.") # User's log
                    print(f"DEBUG_PRINT:       Assigned VERB_INTRANS_TYPE by strong lemma.")
                else:
                    logger.error(f"VERB_INTRANS_TYPE is None. Cannot assign for intransitive lemma '{lemma}'. Defaulting to N.") # User's log
                    assigned_entity = N
                    print(f"DEBUG_PRINT:       ERROR - VERB_INTRANS_TYPE is None for strong intransitive lemma. Defaulted to N.")
            else: # Fallback to original logic if lemma not in heuristic lists
                print(f"DEBUG_PRINT:     Lemma '{lemma}' not in strong heuristic lists. Fallback role-based logic.")
                is_adj_xcomp_obj = False
                if sentence_has_object:
                    obj_original_idx = roles.get('object')
                    obj_token_analysis = None
                    analyses_context = roles.get('analyses_details_for_context', [])
                    print(f"DEBUG_PRINT:       Checking for ADJ xcomp object. Object index: {obj_original_idx}. analyses_context length: {len(analyses_context)}")
                    for an_detail in analyses_context:
                        if an_detail['original_idx'] == obj_original_idx:
                            obj_token_analysis = an_detail
                            break
                    if obj_token_analysis:
                        print(f"DEBUG_PRINT:         Object analysis found: UPOS='{obj_token_analysis['upos']}', DepRel='{obj_token_analysis['deprel']}'")
                        if obj_token_analysis['upos'] == 'ADJ' and obj_token_analysis['deprel'] == 'xcomp':
                            is_adj_xcomp_obj = True
                            logger.debug(f"  Verb typing: Identified object '{obj_token_analysis['text']}' is an ADJ xcomp. Treating verb as intransitive for type assignment.") # User's log
                            print(f"DEBUG_PRINT:         Object is ADJ xcomp. is_adj_xcomp_obj = True.")
                    else:
                        print(f"DEBUG_PRINT:         No analysis found for object index {obj_original_idx} in analyses_context.")


                print(f"DEBUG_PRINT:       Final check for role-based verb type: sentence_has_object={sentence_has_object}, is_adj_xcomp_obj={is_adj_xcomp_obj}")
                if sentence_has_object and not is_adj_xcomp_obj:
                    assigned_entity = VERB_TRANS_TYPE
                    print(f"DEBUG_PRINT:         Assigned VERB_TRANS_TYPE (role-based fallback): {VERB_TRANS_TYPE}")
                else:
                    assigned_entity = VERB_INTRANS_TYPE
                    print(f"DEBUG_PRINT:         Assigned VERB_INTRANS_TYPE (role-based fallback): {VERB_INTRANS_TYPE}")
        else: # Not is_verb_token
             print(f"DEBUG_PRINT:   Skipping role-based assignment for non-verb token '{token_text}'.")


    # --- DET as Subj ---
    # This block was duplicated in the user's original code. I'm keeping the first instance.
    # The second instance of "Nominal Predicate Identification" seems to be the more complete one.
    print(f"DEBUG_PRINT: Section DET as Subj: assigned_entity is None? {assigned_entity is None}. POS='{pos}', DepRel='{dep_rel}'")
    if pos == "DET" and dep_rel == 'nsubj' and assigned_entity is None:
        logger.debug(f"  Decision (DET as Subj): Assigning N type to DET '{token_text}' (deprel='nsubj').") # User's log
        assigned_entity = N
        print(f"DEBUG_PRINT:   Assigned N to DET '{token_text}' (nsubj).")

    # --- 2. Nominal Predicate Identification (More Aggressive for OTHER/X/ADJ roots) ---
    print(f"DEBUG_PRINT: Section 2: Nominal Predicate Identification. assigned_entity is None? {assigned_entity is None}")
    # Condition components for predicate check
    cond1_nominal_structure = sentence_structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"] and \
                              original_idx == sentence_root_idx and \
                              pos in ["ADJ", "X", "NOUN", "PROPN", "NUM"]
    
    subject_exists_in_roles = roles.get('subject') is not None
    head_is_subject = analysis.get('head') == roles.get('subject')
    token_is_root_and_subj_exists = original_idx == sentence_root_idx and subject_exists_in_roles

    cond2_other_structure = sentence_structure == "OTHER" and \
                            ( (subject_exists_in_roles and head_is_subject) or \
                              (token_is_root_and_subj_exists) \
                            ) and \
                            pos in ["ADJ", "NOUN", "PROPN", "NUM", "X"] and \
                            dep_rel in ['amod', 'nmod', 'xcomp', 'advcl', 'acl', 'appos', 'root', 'obj']

    print(f"DEBUG_PRINT:   Predicate Check Cond1 (Nominal): {cond1_nominal_structure}")
    print(f"DEBUG_PRINT:     Sentence Structure: {sentence_structure}, Original Idx: {original_idx}, Sentence Root Idx: {sentence_root_idx}, POS: {pos}")
    print(f"DEBUG_PRINT:   Predicate Check Cond2 (OTHER): {cond2_other_structure}")
    print(f"DEBUG_PRINT:     Subject exists: {subject_exists_in_roles}, Head is subject: {head_is_subject}, Token is root and subj exists: {token_is_root_and_subj_exists}, DepRel: {dep_rel}")


    if assigned_entity is None and (cond1_nominal_structure or cond2_other_structure):
        print(f"DEBUG_PRINT:     Attempting Nominal Predicate assignment for '{token_text}'.")
        # Problematic check: isinstance(N, AtomicType)
        # is_N_AtomicType_pred = isinstance(N, AtomicType)
        # is_S_AtomicType_pred = isinstance(S, AtomicType)
        # print(f"DEBUG_PRINT:       Predicate N/S check: isinstance(N, AtomicType)={is_N_AtomicType_pred}, isinstance(S, AtomicType)={is_S_AtomicType_pred}")

        # Using robust check
        is_N_Ty_n_pred = isinstance(N, Ty) and str(N) == 'n'
        is_S_Ty_s_pred = isinstance(S, Ty) and str(S) == 's'
        print(f"DEBUG_PRINT:       Predicate N/S robust check: is_N_Ty_n={is_N_Ty_n_pred}, is_S_Ty_s={is_S_Ty_s_pred}")

        if not (is_N_Ty_n_pred and is_S_Ty_s_pred): # Using robust check
            logger.error(f"  Cannot assign predicate type for '{token_text}': N or S are not valid AtomicTypes. Defaulting to N.") # User's log
            assigned_entity = N
            print(f"DEBUG_PRINT:         ERROR - N/S not valid for predicate. Defaulted to N.")
        elif pos == "ADJ":
            print(f"DEBUG_PRINT:         Predicate is ADJ. ADJ_PRED_TYPE is None? {ADJ_PRED_TYPE is None}")
            if ADJ_PRED_TYPE is not None:
                assigned_entity = ADJ_PRED_TYPE
            else:
                logger.error(f"  ADJ_PRED_TYPE is None for '{token_text}'. Creating temporary N->S Box.") # User's log
                assigned_entity = Box(f"TempAdjPred_{lemma}_{original_idx}", N, S)
                print(f"DEBUG_PRINT:           ADJ_PRED_TYPE is None. Created temp Box.")
        elif pos in ["NOUN", "PROPN", "NUM", "X"]:
            print(f"DEBUG_PRINT:         Predicate is {pos}. Creating dynamic NounPred Box.")
            assigned_entity = Box(f"NounPred_{lemma}_{original_idx}", N, S) # Dynamic Noun Predicate
        
        if assigned_entity:
            logger.debug(f"  Decision (Predicate): Assigned Predicate Functor to '{token_text}'.") # User's log
            print(f"DEBUG_PRINT:       Assigned Predicate Functor: {assigned_entity}")

        # This block seems to be a refinement or alternative for predicate identification,
        # but it's inside the `if assigned_entity is None and (cond1_nominal_structure or cond2_other_structure):`
        # which means if the above assigned something, this won't run.
        # The logic for `current_subject_idx` update and then re-assigning `assigned_entity`
        # might be tricky. Let's add prints.
        print(f"DEBUG_PRINT:     Predicate V2.2.6 refinement block. Current assigned_entity: {assigned_entity}")
        current_subject_idx = roles.get('subject')
        print(f"DEBUG_PRINT:       Initial current_subject_idx: {current_subject_idx}")

        if sentence_structure == "OTHER" and current_subject_idx is None and original_idx == sentence_root_idx:
            print(f"DEBUG_PRINT:         In 'OTHER', root, no subject. Looking for nsubj dependent for '{token_text}'.")
            dependents_of_this_token = roles.get('dependency_graph', {}).get(original_idx, [])
            print(f"DEBUG_PRINT:           Dependents of '{token_text}': {dependents_of_this_token}")
            for dep_idx, d_rel in dependents_of_this_token:
                if d_rel == 'nsubj' and analyses_details and dep_idx < len(analyses_details) and analyses_details[dep_idx]['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                    logger.warning(f"  Predicate V2.2.6: In 'OTHER', root '{token_text}' is potential predicate. Found nsubj '{analyses_details[dep_idx]['text']}' (idx {dep_idx}). Setting roles['subject'].") # User's log
                    roles['subject'] = dep_idx # Modifying roles dict - be careful
                    current_subject_idx = dep_idx
                    print(f"DEBUG_PRINT:           Found nsubj '{analyses_details[dep_idx]['text']}'. Updated current_subject_idx to {current_subject_idx}.")
                    break
        
        print(f"DEBUG_PRINT:       After potential subject update, current_subject_idx: {current_subject_idx}")
        if current_subject_idx is not None or sentence_structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"]:
            # This might overwrite a previously assigned entity if the outer conditions were met.
            print(f"DEBUG_PRINT:         Predicate V2.2.6 refinement: Has subject or nominal structure. POS: {pos}")
            if pos == "ADJ":
                if ADJ_PRED_TYPE is not None:
                    assigned_entity = ADJ_PRED_TYPE
                    print(f"DEBUG_PRINT:           Assigned ADJ_PRED_TYPE (refinement).")
                else:
                    logger.error("ADJ_PRED_TYPE is None") # User's log
                    assigned_entity = N
                    print(f"DEBUG_PRINT:           ERROR - ADJ_PRED_TYPE is None (refinement). Defaulted to N.")
            elif pos in ["NOUN", "PROPN", "NUM", "X"]:
                assigned_entity = Box(f"NounPred_{lemma}_{original_idx}", N, S)
                print(f"DEBUG_PRINT:           Assigned NounPred Box (refinement) for {pos}.")
            
            if assigned_entity: # Check if this block actually assigned something
                logger.debug(f"  Decision (Predicate V2.2.6): Assigned Predicate Functor to '{token_text}'.") # User's log
                print(f"DEBUG_PRINT:         Predicate V2.2.6 assigned: {assigned_entity}")
        else:
            logger.debug(f"  Predicate Check V2.2.6: {pos} '{token_text}' in '{sentence_structure}' but no subject identified for nominal construction.") # User's log
            print(f"DEBUG_PRINT:       Predicate V2.2.6: No subject for nominal construction for {pos} '{token_text}'.")


    # --- 3. Dependency-Based Assignment (If still unassigned) ---
    print(f"DEBUG_PRINT: Section 3: Dependency-Based. assigned_entity is None? {assigned_entity is None}. DepRel='{dep_rel}', POS='{pos}'")
    if assigned_entity is None:
        print(f"DEBUG_PRINT:   Attempting dependency-based assignment.")
        if dep_rel in ['nsubj', 'obj', 'iobj', 'dobj', 'nsubj:pass', 'csubj', 'obl', 'obl:arg', 'nmod', 'appos', 'parataxis'] and pos in ["NOUN", "PROPN", "PRON", "X", "NUM"]:
            if not (dep_rel.startswith('obl') and pos == 'ADP'): # This condition seems specific
                assigned_entity = N
                logger.debug(f"  Decision (Dep): Noun Type assigned (DepRel '{dep_rel}' for {pos}).") # User's log
                print(f"DEBUG_PRINT:     Assigned N (Dep: {dep_rel}, POS: {pos}).")
        elif dep_rel == 'amod' and pos in ['ADJ', 'X']:
            assigned_entity = ADJ_MOD_TYPE
            logger.debug(f"  Decision (Dep): Adjective Modifier Type assigned (amod for {pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned ADJ_MOD_TYPE (Dep: amod, POS: {pos}). ADJ_MOD_TYPE is None? {ADJ_MOD_TYPE is None}")
        elif dep_rel == 'det' and pos in ['DET', 'PRON', 'X']:
            assigned_entity = DET_TYPE
            logger.debug(f"  Decision (Dep): Determiner Type assigned (det for {pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned DET_TYPE (Dep: det, POS: {pos}). DET_TYPE is None? {DET_TYPE is None}")
        elif dep_rel == 'case' and pos in ['ADP', 'PART']:
            assigned_entity = PREP_FUNCTOR_TYPE
            logger.debug(f"  Decision (Dep): Preposition Functor Type assigned (case for {pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned PREP_FUNCTOR_TYPE (Dep: case, POS: {pos}). PREP_FUNCTOR_TYPE is None? {PREP_FUNCTOR_TYPE is None}")
        elif dep_rel == 'advmod':
            if pos in ['ADV', 'ADJ', 'PART', 'X']:
                assigned_entity = ADV_FUNCTOR_TYPE
                logger.debug(f"  Decision (Dep): Adverb Functor Type assigned (advmod for {pos}).") # User's log
                print(f"DEBUG_PRINT:     Assigned ADV_FUNCTOR_TYPE (Dep: advmod, POS: {pos}). ADV_FUNCTOR_TYPE is None? {ADV_FUNCTOR_TYPE is None}")
        else:
            print(f"DEBUG_PRINT:     No dependency rule matched for DepRel='{dep_rel}', POS='{pos}'.")


    # --- 4. POS-Based Assignment (Fallback) ---
    print(f"DEBUG_PRINT: Section 4: POS-Based Fallback. assigned_entity is None? {assigned_entity is None}. POS='{pos}'")
    if assigned_entity is None:
        print(f"DEBUG_PRINT:   Attempting POS-based fallback assignment.")
        if pos in ["NOUN", "PROPN", "PRON", "NUM", "X"]:
            assigned_entity = N
            logger.debug(f"  Decision (POS Fallback): Noun Type assigned ({pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned N (POS Fallback: {pos}).")
        elif pos == "ADJ":
            assigned_entity = ADJ_MOD_TYPE
            logger.debug(f"  Decision (POS Fallback): Adjective Modifier Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned ADJ_MOD_TYPE (POS Fallback). ADJ_MOD_TYPE is None? {ADJ_MOD_TYPE is None}")
        elif pos == "DET":
            assigned_entity = DET_TYPE
            logger.debug(f"  Decision (POS Fallback): Determiner Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned DET_TYPE (POS Fallback). DET_TYPE is None? {DET_TYPE is None}")
        elif pos == "ADP":
            assigned_entity = PREP_FUNCTOR_TYPE
            logger.debug(f"  Decision (POS Fallback): Preposition Functor Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned PREP_FUNCTOR_TYPE (POS Fallback). PREP_FUNCTOR_TYPE is None? {PREP_FUNCTOR_TYPE is None}")
        elif pos == "ADV":
            assigned_entity = ADV_FUNCTOR_TYPE
            logger.debug(f"  Decision (POS Fallback): Adverb Functor Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned ADV_FUNCTOR_TYPE (POS Fallback). ADV_FUNCTOR_TYPE is None? {ADV_FUNCTOR_TYPE is None}")
        elif pos == "VERB":
            assigned_entity = VERB_TRANS_TYPE if sentence_has_object else VERB_INTRANS_TYPE
            logger.warning(f"  Decision (POS Fallback): Verb '{token_text}' wasn't assigned role/type. Defaulting.") # User's log
            print(f"DEBUG_PRINT:     Assigned VERB type (POS Fallback): {'Transitive' if sentence_has_object else 'Intransitive'}. VT None? {VERB_TRANS_TYPE is None}, VI None? {VERB_INTRANS_TYPE is None}")
        else:
            print(f"DEBUG_PRINT:     No POS fallback rule matched for POS='{pos}'.")


    # --- 5. Handle Ignored POS tags ---
    print(f"DEBUG_PRINT: Section 5: Ignored POS. assigned_entity is None? {assigned_entity is None}. POS='{pos}'")
    if assigned_entity is None:
        print(f"DEBUG_PRINT:   Attempting to handle ignored POS tags.")
        if pos in ["PUNCT", "SYM", "PART", "CCONJ", "SCONJ", "AUX", "INTJ"]:
            logger.debug(f"  Decision (POS ignore/Final): Explicitly None for POS {pos}") # User's log
            # assigned_entity remains None, which is correct for ignored tags.
            print(f"DEBUG_PRINT:     POS '{pos}' is ignored. assigned_entity remains None.")
        else: # This is the ultimate fallback if nothing else caught it
            logger.warning(f"Unhandled POS/DepRel combination: POS='{pos}', DepRel='{dep_rel}'. Defaulting to N.") # User's log
            assigned_entity = N
            print(f"DEBUG_PRINT:     ULTIMATE FALLBACK: Unhandled POS='{pos}'. Defaulted to N.")

    # --- Final Check and Logging ---
    print(f"DEBUG_PRINT: Final Checks. Current assigned_entity: {assigned_entity} (type: {type(assigned_entity).__name__})")
    if assigned_entity is None:
        # This critical log was here, but the return was missing. Adding return.
        logger.critical(f"  CRITICAL FALLBACK: assign_discocat_types_v2_2 is returning None for '{token_text}' (POS: {pos}, Lemma: {lemma}). This will break diagram construction.")
        print(f"DEBUG_PRINT:   CRITICAL - assigned_entity is STILL None. Returning None.")
        return None # Explicitly return None if it's still None here.
    else:
        # This log was here, but the return was missing.
        logger.debug(f"  FINAL RETURN for '{token_text}' (POS {pos}): Type='{type(assigned_entity).__name__}', Value='{str(assigned_entity)}'")
        # The actual return is further down after more checks in the original code.
        pass


    # --- User's original final checks for Box types ---
    # This block seems to intend to correct assignments if the global functor types were None
    # but a Box with that name was somehow assigned (e.g. a dummy Box).
    if isinstance(assigned_entity, Box):
        print(f"DEBUG_PRINT:   Final check for Box instance: Name='{assigned_entity.name}'")
        original_assigned_entity_name_for_check = assigned_entity.name # Store for logging if changed
        changed_in_final_check = False

        if assigned_entity.name == "DetFunctor" and DET_TYPE is None:
            assigned_entity = N; logger.error("DET_TYPE is None!"); changed_in_final_check = True
        if assigned_entity.name == "AdjModFunctor" and ADJ_MOD_TYPE is None:
            assigned_entity = N; logger.error("ADJ_MOD_TYPE is None!"); changed_in_final_check = True
        if assigned_entity.name == "PrepFunctor" and PREP_FUNCTOR_TYPE is None:
            assigned_entity = N; logger.error("PREP_FUNCTOR_TYPE is None!"); changed_in_final_check = True
        if assigned_entity.name == "AdjPredFunctor" and ADJ_PRED_TYPE is None:
            assigned_entity = N; logger.error("ADJ_PRED_TYPE is None!"); changed_in_final_check = True
        if assigned_entity.name == "VerbIntransFunctor" and VERB_INTRANS_TYPE is None:
            assigned_entity = N; logger.error("VERB_INTRANS_TYPE is None!"); changed_in_final_check = True
        if assigned_entity.name == "VerbTransFunctor" and VERB_TRANS_TYPE is None:
            assigned_entity = N; logger.error("VERB_TRANS_TYPE is None!"); changed_in_final_check = True
        if assigned_entity.name == "AdvFunctor" and ADV_FUNCTOR_TYPE is None:
            assigned_entity = N; logger.error("ADV_FUNCTOR_TYPE is None!"); changed_in_final_check = True
        
        if assigned_entity.name.startswith("NounPred_") and (assigned_entity.dom != N or assigned_entity.cod != S):
            logger.error(f"Dynamic Noun Predicate {assigned_entity.name} has incorrect type {assigned_entity.dom}->{assigned_entity.cod}. Resetting to N.") # User's log
            assigned_entity = N; changed_in_final_check = True
        
        if changed_in_final_check:
            print(f"DEBUG_PRINT:     Assigned entity '{original_assigned_entity_name_for_check}' was reset to N due to its corresponding global functor being None or malformed.")


    if debug: # User's original debug log flag
        final_type_str = str(assigned_entity) if assigned_entity else "None"
        logger.debug(f"  >> Final Assigned Type V2.4 for '{token_text}': {final_type_str} (Type: {type(assigned_entity).__name__})") # User's log

    print(f"DEBUG_PRINT: >>> EXITING assign_discocat_types_v2_2 for '{token_text}'. Returning: {str(assigned_entity)} (Type: {type(assigned_entity).__name__}) <<<\n")
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
def build_np_diagram_v4( # User's function name, logic updated to V7.2
    head_noun_idx: int,
    analysis_map: Dict[int, Dict[str, Any]],
    roles: Dict[str, Any], 
    core_type_map: Dict[int, Union['Ty', 'GrammarDiagram', None]], 
    arg_producer_boxes: Dict[int, 'Box'], 
    functor_boxes: Dict[int, 'Box'],      
    globally_processed_indices: Set[int], 
    debug: bool = True
) -> Optional['GrammarDiagram']:
    """
    Recursively builds a diagram for a Noun Phrase centered around head_noun_idx.
    V7.2: Ensures 'indices_consumed_by_this_np_build' is defined and used correctly,
          matching the variable name in the user's NameError traceback.
          Improved nmod (Idafa) handling, stricter processed_indices logic.
    Returns a diagram of type N.
    """
    global N, S, ADJ_MOD_TYPE, DET_TYPE, PREP_FUNCTOR_TYPE, N_MOD_BY_N_ARABIC, Ty, Box, GrammarDiagram, Word, Id, Cup, Swap

    # V7.2: Define and consistently use 'indices_consumed_by_this_np_build'
    indices_consumed_by_this_np_build: Set[int] = set()

    if head_noun_idx in globally_processed_indices:
        logger.warning(f"NP_V7.2: Head Noun {head_noun_idx} ('{analysis_map.get(head_noun_idx,{}).get('text')}') is in `globally_processed_indices`. Cannot be head of new NP. Returning None.")
        return None 

    head_analysis = analysis_map.get(head_noun_idx)
    if not head_analysis:
        logger.error(f"NP_V7.2: Analysis not found for head_noun_idx {head_noun_idx}.")
        return None

    allowed_pos = ["NOUN", "PROPN", "PRON"]
    is_subj_or_obj_role = (head_noun_idx == roles.get('subject') or head_noun_idx == roles.get('object'))
    head_pos = head_analysis['upos']; head_deprel = head_analysis['deprel']
    can_be_np_head = head_pos in allowed_pos or (head_pos == 'X' and is_subj_or_obj_role) or \
                     (head_pos == 'DET' and is_subj_or_obj_role and head_deprel == 'nsubj')
    if not can_be_np_head and core_type_map.get(head_noun_idx) != N:
        logger.error(f"NP_V7.2: Head {head_noun_idx} ('{head_analysis['text']}') POS '{head_pos}' not N type. Cannot form NP.")
        return None

    np_diagram = arg_producer_boxes.get(head_noun_idx)
    if np_diagram is None: 
        if core_type_map.get(head_noun_idx) == N:
            box_name = f"{head_analysis.get('lemma', head_analysis.get('text','unk'))}_{head_noun_idx}"
            np_diagram = Box(box_name, Ty(), N)
            logger.info(f"NP_V7.2: Created missing Argument Producer Box for NP head '{box_name}'.")
        else:
            logger.error(f"NP_V7.2: No N-type for head {head_noun_idx}. Cannot form NP.")
            return None
    
    logger.info(f"NP_V7.2: Building NP for head: '{head_analysis['text']}' (idx {head_noun_idx}), Initial diagram: {get_diagram_repr(np_diagram)}")
    indices_consumed_by_this_np_build.add(head_noun_idx)

    dep_graph = roles.get('dependency_graph', {})
    dependents = dep_graph.get(head_noun_idx, [])
    logger.debug(f"  NP_V7.2: Dependents of {head_noun_idx} ('{head_analysis['text']}'): {dependents}")

    modifiers_to_apply_data = [] 

    for dep_idx, dep_rel in dependents:
        if dep_idx in globally_processed_indices or dep_idx in indices_consumed_by_this_np_build:
            logger.debug(f"  NP_V7.2: Dependent idx {dep_idx} already processed. Skipping.")
            continue

        modifier_analysis = analysis_map.get(dep_idx)
        if not modifier_analysis: continue
        logger.debug(f"  NP_V7.2: Considering dependent: '{modifier_analysis['text']}' (idx {dep_idx}), deprel='{dep_rel}', POS='{modifier_analysis['upos']}'")

        current_mod_indices_subtree: Set[int] = {dep_idx} 
        mod_diagram_component: Optional[GrammarDiagram] = None

        if dep_rel == 'det' and modifier_analysis['upos'] == 'DET':
            det_box = functor_boxes.get(dep_idx)
            if det_box and det_box.dom == N and det_box.cod == N:
                mod_diagram_component = det_box
                modifiers_to_apply_data.append({'box': mod_diagram_component, 'order': 0, 'indices_of_mod_subtree': current_mod_indices_subtree, 'rel': 'det'})
        
        elif dep_rel == 'amod' and modifier_analysis['upos'] == 'ADJ':
            adj_box = functor_boxes.get(dep_idx)
            if adj_box and adj_box.dom == N and adj_box.cod == N:
                mod_diagram_component = adj_box
                modifiers_to_apply_data.append({'box': mod_diagram_component, 'order': 1, 'indices_of_mod_subtree': current_mod_indices_subtree, 'rel': 'amod'})
        
        elif dep_rel == 'nmod' and modifier_analysis['upos'] in ["NOUN", "PROPN", "PRON"]:
            if N_MOD_BY_N_ARABIC is None: logger.warning(f"    N_MOD_BY_N_ARABIC is None. Cannot process nmod."); continue
            
            indices_for_recursive_call_exclusion = globally_processed_indices.union(indices_consumed_by_this_np_build)
            
            logger.debug(f"    NP_V7.2 nmod: Recursive NP build for: '{modifier_analysis['text']}' (idx {dep_idx}). Excluding: {indices_for_recursive_call_exclusion}")
            modifier_np_sub_diagram = build_np_diagram_v4(
                dep_idx, analysis_map, roles, core_type_map,
                arg_producer_boxes, functor_boxes, 
                indices_for_recursive_call_exclusion, 
                debug
            )
            if modifier_np_sub_diagram and modifier_np_sub_diagram.cod == N:
                mod_diagram_component = modifier_np_sub_diagram
                # V7.2: If modifier_np_sub_diagram is complex, its own consumed indices should be captured.
                # For now, 'indices_of_mod_subtree' primarily tracks the head of the modifier.
                # A more robust 'build_np_diagram_v4' would return the set of indices it consumed.
                # Here, we assume the recursive call handles its own additions to `indices_consumed_by_this_np_build`
                # if it modifies a shared set, or we'd need to capture its returned consumed set.
                # The current `indices_of_mod_subtree : {dep_idx}` is a simplification.
                modifiers_to_apply_data.append({
                    'box': mod_diagram_component, 
                    'nmod_functor': N_MOD_BY_N_ARABIC, 
                    'order': 2, 
                    'indices_of_mod_subtree': current_mod_indices_subtree, 
                    'rel': 'nmod_functor'
                })
            else:
                logger.warning(f"    NP_V7.2 nmod: Recursive NP build for nmod '{modifier_analysis['text']}' (idx {dep_idx}) failed or not N type.")
        
        # elif dep_rel == 'case' ... (PP Logic)

    modifiers_to_apply_data.sort(key=lambda m: m['order'])

    for mod_info in modifiers_to_apply_data:
        mod_component = mod_info['box']
        can_apply_modifier = True
        for mod_sub_idx in mod_info['indices_of_mod_subtree']:
            if mod_sub_idx != head_noun_idx and mod_sub_idx in indices_consumed_by_this_np_build:
                logger.warning(f"  NP_V7.2: Index {mod_sub_idx} for modifier '{get_diagram_repr(mod_component)}' already consumed by this NP build. Skipping.")
                can_apply_modifier = False; break
        if not can_apply_modifier: continue

        try:
            if mod_info['rel'] in ['det', 'amod'] or mod_info['rel'] == 'pp_nmod_N_to_NtoN':
                if np_diagram and np_diagram.cod == mod_component.dom: # type: ignore
                    np_diagram = np_diagram >> mod_component
                    indices_consumed_by_this_np_build.update(mod_info['indices_of_mod_subtree'])
            elif mod_info['rel'] == 'nmod_functor': 
                nmod_functor_to_apply = mod_info['nmod_functor']
                if np_diagram and np_diagram.cod == N and mod_component.cod == N: # type: ignore
                    np_diagram = (np_diagram @ mod_component) >> nmod_functor_to_apply
                    indices_consumed_by_this_np_build.update(mod_info['indices_of_mod_subtree'])
            # ... (PP application logic) ...
        except Exception as e_mod_apply:
            logger.error(f"    NP_V7.2 Error applying modifier for {mod_info.get('rel')}: {e_mod_apply}", exc_info=True)
    
    # V7.2: Using the exact logging format from the user's traceback that caused NameError
    logger.info(f"NP_V7.1: Finished NP for '{head_analysis['text']}'. Final diagram: {get_diagram_repr(np_diagram)}, Cod: {np_diagram.cod if np_diagram else 'None'}. Indices consumed by THIS NP build call: {indices_consumed_by_this_np_build - {head_noun_idx}} (plus head {head_noun_idx})")

    if np_diagram and hasattr(np_diagram, 'cod') and np_diagram.cod == N:
        return np_diagram
    else:
        logger.error(f"NP_V7.2: Final NP for head {head_noun_idx} ('{head_analysis['text']}') is invalid or not N-type.")
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


def create_nominal_sentence_diagram_v2_7( # User's function name, logic updated to V3.0
    tokens: List[str],
    analyses_details: List[Dict[str, Any]],
    roles: Dict,
    word_core_types: List[Union['Ty', 'GrammarDiagram', None]],
    original_indices: List[int], # These are all original_indices present in word_core_types
    debug: bool = True,
    output_dir: Optional[str] = None, 
    sentence_prefix: str = "diag_nominal",
    hint_predicate_original_idx: Optional[int] = None
) -> Optional['GrammarDiagram']:
    """
    Creates a DisCoCat diagram for nominal sentences (Subject-Predicate).
    V3.0: Correctly calls build_np_diagram_v4 with positional arg for processed_indices.
          Relies on build_np_diagram_v4 (V7) for full subject NP construction.
    """
    global N, S, ADJ_PRED_TYPE, Ty, Box, GrammarDiagram, Word, Id, Cup, Swap, build_np_diagram_v4
    
    logger.info(f"NOMINAL_DIAG_V3.0: Creating diagram for: \"{' '.join(tokens)}\"")
    logger.info(f"  Hinted Predicate Idx (from roles['predicate_idx']): {hint_predicate_original_idx}")
    logger.info(f"  Roles: Subject Idx={roles.get('subject')}, Root Idx={roles.get('root')}")

    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map: Dict[int, Union[Ty, Box, Word, GrammarDiagram, None]] = {}
    if len(original_indices) == len(word_core_types):
        core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}
    else:
        logger.error("NOMINAL_DIAG_V3.0: Mismatch: original_indices vs word_core_types lengths.")
        return None

    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}    

    for orig_idx, core_entity in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or core_entity is None: continue
        box_name_base = analysis.get('lemma', analysis.get('text', 'UNK'))
        box_name = f"{box_name_base}_{orig_idx}"
        if isinstance(core_entity, Box): functor_boxes[orig_idx] = core_entity
        elif isinstance(core_entity, Ty) and core_entity == N: arg_producer_boxes[orig_idx] = Box(box_name, Ty(), N)
        elif isinstance(core_entity, Word): 
            if core_entity.cod == N: arg_producer_boxes[orig_idx] = Box(core_entity.name, Ty(), N)
            elif isinstance(core_entity.cod, Ty) and core_entity.cod != Ty(): functor_boxes[orig_idx] = Box(core_entity.name, core_entity.dom, core_entity.cod)

    selected_predicate_idx: Optional[int] = None
    selected_predicate_functor: Optional[Box] = None

    if hint_predicate_original_idx is not None:
        hinted_core_type = core_type_map.get(hint_predicate_original_idx)
        if isinstance(hinted_core_type, Box) and hinted_core_type.dom == N and hinted_core_type.cod == S:
            selected_predicate_idx = hint_predicate_original_idx
            selected_predicate_functor = hinted_core_type
            logger.info(f"  V3.0 Predicate: SUCCESSFULLY used HINTED predicate: '{get_diagram_repr(selected_predicate_functor)}' (idx {selected_predicate_idx})")
        else:
            logger.warning(f"  V3.0 Predicate: Hinted predicate idx {hint_predicate_original_idx} is NOT a valid N->S Box. Ignoring hint.")
    
    if selected_predicate_functor is None: # Fallback search (less reliable)
        logger.warning("  V3.0 Predicate: No valid hinted predicate. Attempting fallback search for N->S functor.")
        for idx in original_indices:
            if idx == roles.get('subject'): continue
            functor = functor_boxes.get(idx)
            if functor and functor.dom == N and functor.cod == S:
                selected_predicate_idx = idx; selected_predicate_functor = functor
                logger.warning(f"  V3.0 Predicate: Found FALLBACK predicate: '{get_diagram_repr(selected_predicate_functor)}' (idx {selected_predicate_idx}).")
                break 
    
    if selected_predicate_functor is None or selected_predicate_idx is None:
        logger.error(f"NOMINAL_DIAG_V3.0: No suitable N->S predicate functor selected or found.")
        return None
    logger.info(f"NOMINAL_DIAG_V3.0: FINAL Selected Predicate: '{get_diagram_repr(selected_predicate_functor)}' (idx {selected_predicate_idx})")

    subject_head_idx = roles.get('subject')
    if subject_head_idx is None:
        logger.error("NOMINAL_DIAG_V3.0: Subject index not found in roles.")
        return None
    logger.info(f"NOMINAL_DIAG_V3.0: Main Subject Head Index: {subject_head_idx} ('{analysis_map.get(subject_head_idx, {}).get('text')}')")

    # Indices to be excluded from NP construction (i.e., the main predicate)
    indices_to_exclude_for_np_build: Set[int] = set()
    if selected_predicate_idx is not None:
        indices_to_exclude_for_np_build.add(selected_predicate_idx)

    logger.info(f"NOMINAL_DIAG_V3.0: Calling build_np_diagram_v4 for subject head idx {subject_head_idx}. Excluding: {indices_to_exclude_for_np_build}")
    
    # ***** MODIFIED CALL to build_np_diagram_v4 *****
    # Pass the 7th argument (globally_processed_indices) positionally
    subject_np_diagram = build_np_diagram_v4(
        subject_head_idx,                   # 1st: head_noun_idx
        analysis_map,                       # 2nd: analysis_map
        roles,                              # 3rd: roles
        core_type_map,                      # 4th: core_type_map
        arg_producer_boxes,                 # 5th: arg_producer_boxes
        functor_boxes,                      # 6th: functor_boxes
        indices_to_exclude_for_np_build,    # 7th: globally_processed_indices (Set[int])
        debug=debug                         # Keyword argument
    )

    if subject_np_diagram is None or not hasattr(subject_np_diagram, 'cod') or subject_np_diagram.cod != N:
        logger.error(f"NOMINAL_DIAG_V3.0: Subject NP construction failed or yielded non-N type. Diagram: {get_diagram_repr(subject_np_diagram)}")
        subject_np_diagram = arg_producer_boxes.get(subject_head_idx) # Fallback
        if subject_np_diagram is None or subject_np_diagram.cod != N:
             logger.error(f"  Fallback to simple arg_producer_box for subject {subject_head_idx} also failed. Cannot proceed.")
             return None
        logger.warning(f"  Using simple arg_producer_box '{get_diagram_repr(subject_np_diagram)}' as subject NP fallback.")

    logger.info(f"NOMINAL_DIAG_V3.0: Subject NP Diagram: {get_diagram_repr(subject_np_diagram)} (cod: {subject_np_diagram.cod})")
    final_diagram: Optional[GrammarDiagram] = None
    try:
        if subject_np_diagram.cod == N and selected_predicate_functor.dom == N and selected_predicate_functor.cod == S:
            final_diagram = subject_np_diagram >> selected_predicate_functor
            logger.info(f"  Nominal composition successful. Diagram Cod: {final_diagram.cod}")
            # ... (Simplified sentence-level modifier attachment logic) ...
        else:
            logger.error(f"NOMINAL_DIAG_V3.0: Type mismatch for composition. Subj cod: {subject_np_diagram.cod}, Pred dom: {selected_predicate_functor.dom}")
            return None
    except Exception as e_compose:
        logger.error(f"NOMINAL_DIAG_V3.0: Error during composition: {e_compose}", exc_info=True)
        return None

    if final_diagram and final_diagram.cod == S:
        try:
            normalized_diagram = final_diagram.normal_form()
            logger.info(f"NOMINAL_DIAG_V3.0: Normalization successful. Final cod: {normalized_diagram.cod}")
            return normalized_diagram
        except Exception as e_norm:
            logger.error(f"NOMINAL_DIAG_V3.0: Normalization failed: {e_norm}. Returning unnormalized.", exc_info=True)
            return final_diagram 
    
    logger.error("NOMINAL_DIAG_V3.0: Could not form a complete diagram ending in S.")
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
    n_layers_iqp: int = 2,
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
    global ARABIC_DISCOCIRC_PIPELINE_AVAILABLE # Allow modification of the global flag

    # --- MOVED IMPORT INSIDE FUNCTION ---
    generate_discocirc_ready_diagram_func = None
    try:
        from arabic_discocirc_pipeline import generate_discocirc_ready_diagram
        generate_discocirc_ready_diagram_func = generate_discocirc_ready_diagram
        ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = True # Set to True on successful import
    except ImportError as e_discocirc_runtime:
        # Log this error if it occurs at runtime, even if the top-level one was removed
        logger.error(f"Runtime import of generate_discocirc_ready_diagram failed: {e_discocirc_runtime}. Enriched diagrams unavailable.")
        ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = False # Ensure it's False

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
    diagram: Optional[GrammarDiagram] = None
    used_enriched_diagram_path = False

    if ARABIC_DISCOCIRC_PIPELINE_AVAILABLE:
        logger.info(f"Attempting to generate feature-enriched DisCoCirc diagram for: '{sentence}'")
        try:
            enriched_diagram = generate_discocirc_ready_diagram(
                sentence_str=sentence,
                debug=debug
                #classical_feature_dim_for_enrichment=classical_feature_dim_for_discocirc_enrichment
            )
            if enriched_diagram is not None:
                logger.info(f"Successfully generated feature-enriched diagram via arabic_discocirc_pipeline. Boxes: {len(enriched_diagram.boxes)}")
                # Log data of first few boxes if debug
                if debug and enriched_diagram.boxes:
                    for i, b in enumerate(enriched_diagram.boxes[:3]): # Log first 3 boxes
                        logger.debug(f"  Enriched Box {i} ('{b.name}') data keys: {list(getattr(b, 'data', {}).keys()) if getattr(b, 'data', {}) else 'No data'}")

                diagram = enriched_diagram # Use this diagram
                used_enriched_diagram_path = True
            else:
                logger.warning("Feature-enriched diagram generation (arabic_discocirc_pipeline) returned None. Proceeding with fallback diagram logic.")
        except Exception as e_discocirc_call:
            logger.error(f"Error calling generate_discocirc_ready_diagram: {e_discocirc_call}", exc_info=True)
            logger.warning("Proceeding with fallback diagram logic due to error in enriched pipeline.")
    else:
        logger.info("arabic_discocirc_pipeline not available. Using fallback diagram logic directly.")

    # --- 3. Fallback or Original Diagram Creation Logic (IF enriched diagram was not created) ---
    if not used_enriched_diagram_path:
        logger.info("Using internal camel_test2.py logic for diagram generation (fallback path).")
        # This part is your existing logic from arabic_to_quantum_enhanced_v2_7 for type assignment and diagram creation
        word_core_types_list = []
        original_indices_for_diagram = []
        filtered_tokens_for_diagram = []
        core_type_map_for_fallback: Dict[int, Union[Ty, GrammarDiagram, None]] = {}

        logger.debug(f"--- Assigning Core Types (Fallback Path) V2.2.2 for: '{sentence}' ---")
        for i, analysis_entry in enumerate(analyses_details):
            current_core_type = assign_discocat_types_v2_2(
                analysis=analysis_entry,
                roles=roles,
                debug=debug
            )
            core_type_map_for_fallback[analysis_entry['original_idx']] = current_core_type
            if current_core_type is not None:
                word_core_types_list.append(current_core_type)
                original_indices_for_diagram.append(analysis_entry['original_idx'])
                filtered_tokens_for_diagram.append(analysis_entry['text'])
            else:
                logger.debug(f"  Fallback Token '{analysis_entry['text']}' (orig_idx {analysis_entry['original_idx']}) assigned None core type, excluding.")

        if not filtered_tokens_for_diagram:
            logger.error(f"Fallback Path: No valid tokens with core types remained for diagram construction: '{sentence}'")
            return None, None, structure, tokens, analyses_details, roles
        
        logger.debug(f"Fallback Path Filtered Tokens: {filtered_tokens_for_diagram}")
        diagram_creation_error = None
        try:
            logger.info(f"Creating DisCoCat diagram (V2.7.3 - Structure/OTHER Handling) for structure: {structure}...")
            safe_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0]) if sentence else "empty"

            attempted_diagram_type = "Unknown"
            
            # Define all known nominal structures from analyze_arabic_sentence_with_morph (V4.x)
            known_nominal_structures = [
                "NOMINAL_NOUN_SUBJ", "NOMINAL_NOUN_SUBJ_ADJ_PRED", "NOMINAL_NOUN_SUBJ_NOUN_PRED",
                "NOMINAL_ADJ_PREDICATE", "NOMINAL_ADJ_PRED_NO_SUBJ",
                "NOMINAL_X_PRED_WITH_SUBJ", "NOMINAL_RECLASSIFIED",
                "NOMINAL_SUBJ_ONLY_RECLASSIFIED", "SUBJ_NO_VERB_OTHER", # Older label
                "NOMINAL" # Generic older label
            ]
            # Add COMPLEX versions
            complex_nominal_structures = ["COMPLEX_" + s for s in known_nominal_structures]
            all_recognized_nominal_structures = known_nominal_structures + complex_nominal_structures

            if structure in all_recognized_nominal_structures:
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
            elif structure.startswith("VERBAL_") or structure in ["SVO", "VSO", "SV", "VS"] or \
                 (structure.startswith("COMPLEX_") and ("VERBAL" in structure or "SVO" in structure or "VSO" in structure)) or \
                 (roles.get('verb') is not None and structure not in ["ERROR_ANALYSIS_EXCEPTION", "ERROR_STANZA_INIT", "EMPTY_INPUT", "NO_SENTENCES_STANZA"]): # If a verb is identified, try verbal
                logger.info(f"Attempting VERBAL diagram creation for structure '{structure}' (or verb role present).")
                attempted_diagram_type = "Verbal"
                diagram = create_verbal_sentence_diagram_v3_7(
                    filtered_tokens_for_diagram, analyses_details, roles,
                    word_core_types_list, original_indices_for_diagram, debug=debug,
                    output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal"
                )
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
            n_single_qubit_params_calculated = n_layers_iqp * 2 
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
            logger.debug(f"Created IQPAnsatz with {n_layers_iqp} layers and {n_single_qubit_params_calculated} params per qubit.")

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
            nounS = AtomicType.NOUN
            sentS = AtomicType.SENTENCE
            ob_map = {nounS: 1, sentS: 1}
            selected_ansatz = SpiderAnsatz(ob_map=ob_map)
            logger.info(f"Using SpiderAnsatz (discard_qubits={discard_qubits_spider})")
        else:
            logger.warning(f"Unknown ansatz_choice: '{ansatz_choice}'. Defaulting to IQPAnsatz.")
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3) # Default IQP

        if selected_ansatz is None:
            raise ValueError("Ansatz object could not be created.")
        logger.debug(f"Attempting to apply ansatz. Grammatical diagram for sentence '{sentence}':")
        logger.debug(f"Diagram: {diagram}") # This might be too verbose
        logger.debug(f"Boxes in diagram ({len(diagram.boxes)}):")
        for i, box in enumerate(diagram.boxes):
            logger.debug(f"  Box {i}: name='{box.name}', dom={box.dom}, cod={box.cod}, data={getattr(box, 'data', None)}") # Log box.data if it exists

        logger.info(f"--- SPIDER DEBUG for sentence: '{sentence}' ---")
        logger.info(f"Grammatical diagram: {diagram}")
        logger.info("Boxes in grammatical diagram:")
        for i, box in enumerate(diagram.boxes):
            logger.info(f"  GRMR_BOX {i}: name='{box.name}', dom={box.dom}, cod={box.cod}")

        if "spider" in str(type(selected_ansatz)).lower(): # Check if it's SpiderAnsatz
            logger.info("Applying SPIDER ansatz functor to each grammatical box individually:")
            problematic_box_found = False
            for i, box in enumerate(diagram.boxes):
                try:
                    # selected_ansatz is the SpiderAnsatz instance
                    quantum_box = selected_ansatz(box) # Apply functor to individual box
                    logger.info(f"  QUANTUM_BOX from GRMR_BOX {i} ('{box.name}'): q_dom={quantum_box.dom}, q_cod={quantum_box.cod}")
                    if quantum_box.dom == Ty():
                        logger.error(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        logger.error(f"    !!!! PROBLEM: GRMR_BOX {i} ('{box.name}', dom={box.dom}, cod={cod.cod})")
                        logger.error(f"    !!!!   resulted in QUANTUM_BOX with dom=Ty() and cod={quantum_box.cod} via SPIDER !!!!")
                        logger.error(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        problematic_box_found = True
                except Exception as e_box:
                    logger.error(f"  ERROR applying SPIDER to GRMR_BOX {i} ('{box.name}', dom={box.dom}, cod={box.cod}): {e_box}", exc_info=False)
                    problematic_box_found = True

            if problematic_box_found:
                logger.error("SPIDER Debug: Problematic box found. See details above. Full diagram conversion will likely fail.")
                # Consider returning a failure state here for the single sentence test
                # return None, None, None, None # (match your function's error return)

        # Now, let the original call happen, which will likely raise the error
        # if problematic_box_found was true
        try:
            quantum_diagram = selected_ansatz(diagram)
        except TypeError as e_diag:
            logger.error(f"SPIDER TypeError on full diagram conversion (as expected if box issues found): {e_diag}")
            # If not already done by the individual box check, re-log the failing diagram's boxes
            if not problematic_box_found: # Log only if individual check didn't catch it
                logger.error("Original grammatical diagram that failed on full conversion:")
                for i, box in enumerate(diagram.boxes):
                    logger.error(f"  GRMR_BOX {i}: name='{box.name}', dom={box.dom}, cod={box.cod}")
            raise

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