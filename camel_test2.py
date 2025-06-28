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

    structure_label = final_structure_label

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
    debug: bool = True,
    handle_lexical_ambiguity: bool = False # NEW PARAMETER     # This debug flag is not currently used by the print statements
) -> Union['Ty', 'GrammarDiagram', None]: # Use string literals for forward references if Ty/GrammarDiagram not defined yet
    """
    Assigns core DisCoCat types.
    V2.2.5: Aggressive predicate typing for root X/ADJ in OTHER/Nominal structures.
            Updated known verb lemmas.
    V_FIX_CCONJ_PART: Added fix to assign N to CCONJ, PART etc. to prevent None return.
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
    if handle_lexical_ambiguity and assigned_entity is None:
        known_ambiguous_words: Dict[str, Dict[str, Any]] = {
            "رجل": {"senses": ["man", "leg"], "base_type": N_ARABIC},
            "عين": {"senses": ["eye", "spring"], "base_type": N_ARABIC},
            "جمل": {"senses": ["camel", "sentences"], "base_type": N_ARABIC},
            "علم": {"senses": ["flag", "knowledge"], "base_type": N_ARABIC},
            "ملك": {"senses": ["king", "possess_verb_sense_if_pos_verb"], "base_type": N_ARABIC}, # Handle POS variation for 'ملك' later
            "ضرب": {"senses": ["hit_verb_sense", "multiply_verb_sense"], "base_type": S_ARABIC}, # Base type S for verb
            "فتح": {"senses": ["open_verb_sense", "conquer_verb_sense"], "base_type": S_ARABIC},# Base type S for verb
        }
        
        # Special handling for ambiguous verbs like 'ملك', 'ضرب', 'فتح'
        # Their base_type depends on whether they are nouns or verbs in context
        # For simplicity in this example, we'll assume 'رجل', 'عين', 'جمل', 'علم' are nouns
        # and 'ضرب', 'فتح' are verbs when ambiguous.
        # 'ملك' is tricky - it can be Noun (king) or Verb (possess).
        # `assign_discocat_types_v2_2` logic for VERB POS will typically handle verb senses.
        # This explicit block is more for nouns that are lexically ambiguous but same POS.

        # Refined check for known ambiguous words, prioritizing nouns for this explicit AmbiguousLexicalBox
        if token_text in known_ambiguous_words:
            amb_info = known_ambiguous_words[token_text]
            # Only apply if the expected POS matches (e.g., 'رجل' as NOUN)
            # The current `assign_discocat_types_v2_2` might later assign a verb type if POS is VERB
            # This is a simple check. More sophisticated logic might be needed.
            current_pos = analysis.get('upos', 'X').upper()
            
            # If the ambiguous word is a verb, its type should be S-related.
            # If noun, N-related.
            expected_base_type_for_amb_box = N_ARABIC
            if current_pos == "VERB" and token_text in ["ضرب", "فتح"]: # Ambiguous verbs
                expected_base_type_for_amb_box = S_ARABIC # Base type S for verb outputs
            elif current_pos == "VERB" and token_text == "ملك":
                # If "ملك" is a VERB, it will be handled by verb logic later.
                # If it's a NOUN, amb_info['base_type'] (N_ARABIC) is fine.
                # We only create AmbiguousLexicalBox if we expect it to be a noun.
                if amb_info['base_type'] != N_ARABIC: # if "ملك" configured for VERB sense here
                    logger.debug(f"Skipping AmbiguousLexicalBox for VERB '{token_text}'. Verb logic will handle.")
                    pass # Let verb logic handle it.
                else: # "ملك" configured as Noun (king), current_pos might be something else
                    if current_pos not in ["NOUN", "PROPN"]: # if 'ملك' is NOT noun, but amb_info is for Noun sense
                         pass # skip
                    else: # current_pos is NOUN, and amb_info is for Noun. Create Box.
                         from common_qnlp_types import AmbiguousLexicalBox
                         assigned_entity = AmbiguousLexicalBox(
                             name=f"{token_text}_ambiguous",
                             base_type=amb_info['base_type'], # Should be N_ARABIC
                             senses=amb_info['senses']
                             #data={'original_idx': analysis['original_idx'], 'original_text': token_text}
                         )
                         print(f"DEBUG_PRINT: Assigned AmbiguousLexicalBox to NOUN '{token_text}'")


            elif amb_info['base_type'] == expected_base_type_for_amb_box:
                # Check if N_ARABIC and S_ARABIC are correctly imported/defined AtomicType instances
                if not isinstance(amb_info['base_type'], Ty):
                    logger.error(f"Misconfigured base_type for ambiguous word '{token_text}'. Expected AtomicType instance. Got {type(amb_info['base_type'])}")
                else:
                    from common_qnlp_types import AmbiguousLexicalBox
                    assigned_entity = AmbiguousLexicalBox(
                        name=f"{token_text}_ambiguous",
                        base_type=amb_info['base_type'],
                        senses=amb_info['senses']
                        #data={'original_idx': analysis['original_idx'], 'original_text': token_text}
                    )
                    print(f"DEBUG_PRINT: Assigned AmbiguousLexicalBox to '{token_text}' (POS: {current_pos}, BaseType: {str(amb_info['base_type'])})")
            else:
                 logger.debug(f"Skipping AmbiguousLexicalBox for '{token_text}': POS '{current_pos}' might not match expected base type logic for ambiguity, or handled by other rules.")
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
        logger.debug(f"  Attempting verb type assignment for '{token_text}' (lemma: '{lemma}').") # User's log
        print(f"DEBUG_PRINT:   Attempting verb type assignment for '{token_text}' (lemma: '{lemma}').")
        
        is_N_Ty_n = isinstance(N, Ty) and str(N) == 'n'
        is_S_Ty_s = isinstance(S, Ty) and str(S) == 's'
        print(f"DEBUG_PRINT:     Robust Check: is_N_Ty_n={is_N_Ty_n}, is_S_Ty_s={is_S_Ty_s}")

        if not (is_N_Ty_n and is_S_Ty_s): 
            logger.error(f"  Cannot assign verb/predicate type for '{token_text}': N ('{N}') or S ('{S}') are not valid Ty('n')/Ty('s'). Defaulting to basic assignment.")
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
                logger.debug(f"  Decision (Verb Logic): Assigned VERB_TRANS_TYPE to '{token_text}'.") # User's log
                print(f"DEBUG_PRINT:       Assigned VERB_TRANS_TYPE: {assigned_entity}")
            else:
                logger.error(f"  VERB_TRANS_TYPE is None. Creating temporary Box(N@N >> S) for '{token_text}'.") # User's log
                assigned_entity = Box(f"TempVerbTrans_{lemma}_{original_idx}", N @ N, S)
                print(f"DEBUG_PRINT:       VERB_TRANS_TYPE is None. Created temporary Box: {assigned_entity}")
        else: # Intransitive
            print(f"DEBUG_PRINT:     Verb is intransitive (sentence_has_object=False). VERB_INTRANS_TYPE is None? {VERB_INTRANS_TYPE is None}")
            if VERB_INTRANS_TYPE is not None:
                assigned_entity = VERB_INTRANS_TYPE
                logger.debug(f"  Decision (Verb Logic): Assigned VERB_INTRANS_TYPE to '{token_text}'.") # User's log
                print(f"DEBUG_PRINT:       Assigned VERB_INTRANS_TYPE: {assigned_entity}")
            else:
                logger.error(f"  VERB_INTRANS_TYPE is None. Creating temporary Box(N >> S) for '{token_text}'.") # User's log
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
                    logger.debug(f"  Decision (Lemma Heuristic): Assigned Transitive Verb Type for lemma '{lemma}'.") # User's log
                    print(f"DEBUG_PRINT:       Assigned VERB_TRANS_TYPE by strong lemma.")
                else:
                    logger.error(f"VERB_TRANS_TYPE is None. Cannot assign for transitive lemma '{lemma}'. Defaulting to N.") # User's log
                    assigned_entity = N
                    print(f"DEBUG_PRINT:       ERROR - VERB_TRANS_TYPE is None for strong transitive lemma. Defaulted to N.")
            elif lemma in strongly_intransitive_lemmas:
                print(f"DEBUG_PRINT:     Lemma '{lemma}' in strongly_intransitive_lemmas. VERB_INTRANS_TYPE is None? {VERB_INTRANS_TYPE is None}")
                if VERB_INTRANS_TYPE is not None:
                    assigned_entity = VERB_INTRANS_TYPE
                    logger.debug(f"  Decision (Lemma Heuristic): Assigned Intransitive Verb Type for lemma '{lemma}'.") # User's log
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
                            logger.debug(f"  Verb typing: Identified object '{obj_token_analysis['text']}' is an ADJ xcomp. Treating verb as intransitive for type assignment.") # User's log
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
    print(f"DEBUG_PRINT: Section DET as Subj: assigned_entity is None? {assigned_entity is None}. POS='{pos}', DepRel='{dep_rel}'")
    if pos == "DET" and dep_rel == 'nsubj' and assigned_entity is None:
        logger.debug(f"  Decision (DET as Subj): Assigning N type to DET '{token_text}' (deprel='nsubj').") # User's log
        assigned_entity = N
        print(f"DEBUG_PRINT:   Assigned N to DET '{token_text}' (nsubj).")

    # --- 2. Nominal Predicate Identification (More Aggressive for OTHER/X/ADJ roots) ---
    print(f"DEBUG_PRINT: Section 2: Nominal Predicate Identification. assigned_entity is None? {assigned_entity is None}")
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
        is_N_Ty_n_pred = isinstance(N, Ty) and str(N) == 'n'
        is_S_Ty_s_pred = isinstance(S, Ty) and str(S) == 's'
        print(f"DEBUG_PRINT:       Predicate N/S robust check: is_N_Ty_n={is_N_Ty_n_pred}, is_S_Ty_s={is_S_Ty_s_pred}")

        if not (is_N_Ty_n_pred and is_S_Ty_s_pred): 
            logger.error(f"  Cannot assign predicate type for '{token_text}': N or S are not valid AtomicTypes. Defaulting to N.") # User's log
            assigned_entity = N
            print(f"DEBUG_PRINT:         ERROR - N/S not valid for predicate. Defaulted to N.")
        elif pos == "ADJ":
            print(f"DEBUG_PRINT:         Predicate is ADJ. ADJ_PRED_TYPE is None? {ADJ_PRED_TYPE is None}")
            if ADJ_PRED_TYPE is not None:
                assigned_entity = ADJ_PRED_TYPE
            else:
                logger.error(f"  ADJ_PRED_TYPE is None for '{token_text}'. Creating temporary N->S Box.") # User's log
                assigned_entity = Box(f"TempAdjPred_{lemma}_{original_idx}", N, S)
                print(f"DEBUG_PRINT:           ADJ_PRED_TYPE is None. Created temp Box.")
        elif pos in ["NOUN", "PROPN", "NUM", "X"]:
            print(f"DEBUG_PRINT:         Predicate is {pos}. Creating dynamic NounPred Box.")
            assigned_entity = Box(f"NounPred_{lemma}_{original_idx}", N, S) # Dynamic Noun Predicate
        
        if assigned_entity:
            logger.debug(f"  Decision (Predicate): Assigned Predicate Functor to '{token_text}'.") # User's log
            print(f"DEBUG_PRINT:       Assigned Predicate Functor: {assigned_entity}")

        print(f"DEBUG_PRINT:     Predicate V2.2.6 refinement block. Current assigned_entity: {assigned_entity}")
        current_subject_idx = roles.get('subject')
        print(f"DEBUG_PRINT:       Initial current_subject_idx: {current_subject_idx}")

        if sentence_structure == "OTHER" and current_subject_idx is None and original_idx == sentence_root_idx:
            print(f"DEBUG_PRINT:         In 'OTHER', root, no subject. Looking for nsubj dependent for '{token_text}'.")
            dependents_of_this_token = roles.get('dependency_graph', {}).get(original_idx, [])
            print(f"DEBUG_PRINT:           Dependents of '{token_text}': {dependents_of_this_token}")
            if analyses_details: # Ensure analyses_details is not None
                for dep_idx, d_rel in dependents_of_this_token:
                    # Check dep_idx bounds for analyses_details
                    if d_rel == 'nsubj' and 0 <= dep_idx < len(analyses_details) and analyses_details[dep_idx]['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                        logger.warning(f"  Predicate V2.2.6: In 'OTHER', root '{token_text}' is potential predicate. Found nsubj '{analyses_details[dep_idx]['text']}' (idx {dep_idx}). Setting roles['subject'].") # User's log
                        roles['subject'] = dep_idx # Modifying roles dict - be careful
                        current_subject_idx = dep_idx
                        print(f"DEBUG_PRINT:             Found nsubj '{analyses_details[dep_idx]['text']}'. Updated current_subject_idx to {current_subject_idx}.")
                        break
        
        print(f"DEBUG_PRINT:       After potential subject update, current_subject_idx: {current_subject_idx}")
        if current_subject_idx is not None or sentence_structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"]:
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
            
            if assigned_entity: 
                logger.debug(f"  Decision (Predicate V2.2.6): Assigned Predicate Functor to '{token_text}'.") # User's log
                print(f"DEBUG_PRINT:         Predicate V2.2.6 assigned: {assigned_entity}")
        else:
            logger.debug(f"  Predicate Check V2.2.6: {pos} '{token_text}' in '{sentence_structure}' but no subject identified for nominal construction.") # User's log
            print(f"DEBUG_PRINT:       Predicate V2.2.6: No subject for nominal construction for {pos} '{token_text}'.")


    # --- 3. Dependency-Based Assignment (If still unassigned) ---
    print(f"DEBUG_PRINT: Section 3: Dependency-Based. assigned_entity is None? {assigned_entity is None}. DepRel='{dep_rel}', POS='{pos}'")
    if assigned_entity is None:
        print(f"DEBUG_PRINT:   Attempting dependency-based assignment.")
        if dep_rel in ['nsubj', 'obj', 'iobj', 'dobj', 'nsubj:pass', 'csubj', 'obl', 'obl:arg', 'nmod', 'appos', 'parataxis'] and pos in ["NOUN", "PROPN", "PRON", "X", "NUM"]:
            if not (dep_rel.startswith('obl') and pos == 'ADP'): 
                assigned_entity = N
                logger.debug(f"  Decision (Dep): Noun Type assigned (DepRel '{dep_rel}' for {pos}).") # User's log
                print(f"DEBUG_PRINT:     Assigned N (Dep: {dep_rel}, POS: {pos}).")
        elif dep_rel == 'amod' and pos in ['ADJ', 'X']:
            assigned_entity = ADJ_MOD_TYPE
            logger.debug(f"  Decision (Dep): Adjective Modifier Type assigned (amod for {pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned ADJ_MOD_TYPE (Dep: amod, POS: {pos}). ADJ_MOD_TYPE is None? {ADJ_MOD_TYPE is None}")
        elif dep_rel == 'det' and pos in ['DET', 'PRON', 'X']:
            assigned_entity = DET_TYPE
            logger.debug(f"  Decision (Dep): Determiner Type assigned (det for {pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned DET_TYPE (Dep: det, POS: {pos}). DET_TYPE is None? {DET_TYPE is None}")
        elif dep_rel == 'case' and pos in ['ADP', 'PART']: # Note: PART was also in the "ignore" list. If it gets here, it's handled.
            assigned_entity = PREP_FUNCTOR_TYPE
            logger.debug(f"  Decision (Dep): Preposition Functor Type assigned (case for {pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned PREP_FUNCTOR_TYPE (Dep: case, POS: {pos}). PREP_FUNCTOR_TYPE is None? {PREP_FUNCTOR_TYPE is None}")
        elif dep_rel == 'advmod':
            if pos in ['ADV', 'ADJ', 'PART', 'X']:
                assigned_entity = ADV_FUNCTOR_TYPE
                logger.debug(f"  Decision (Dep): Adverb Functor Type assigned (advmod for {pos}).") # User's log
                print(f"DEBUG_PRINT:     Assigned ADV_FUNCTOR_TYPE (Dep: advmod, POS: {pos}). ADV_FUNCTOR_TYPE is None? {ADV_FUNCTOR_TYPE is None}")
        else:
            print(f"DEBUG_PRINT:     No dependency rule matched for DepRel='{dep_rel}', POS='{pos}'.")


    # --- 4. POS-Based Assignment (Fallback) ---
    print(f"DEBUG_PRINT: Section 4: POS-Based Fallback. assigned_entity is None? {assigned_entity is None}. POS='{pos}'")
    if assigned_entity is None:
        print(f"DEBUG_PRINT:   Attempting POS-based fallback assignment.")
        if pos in ["NOUN", "PROPN", "PRON", "NUM", "X"]:
            assigned_entity = N
            logger.debug(f"  Decision (POS Fallback): Noun Type assigned ({pos}).") # User's log
            print(f"DEBUG_PRINT:     Assigned N (POS Fallback: {pos}).")
        elif pos == "ADJ":
            assigned_entity = ADJ_MOD_TYPE
            logger.debug(f"  Decision (POS Fallback): Adjective Modifier Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned ADJ_MOD_TYPE (POS Fallback). ADJ_MOD_TYPE is None? {ADJ_MOD_TYPE is None}")
        elif pos == "DET":
            assigned_entity = DET_TYPE
            logger.debug(f"  Decision (POS Fallback): Determiner Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned DET_TYPE (POS Fallback). DET_TYPE is None? {DET_TYPE is None}")
        elif pos == "ADP":
            assigned_entity = PREP_FUNCTOR_TYPE
            logger.debug(f"  Decision (POS Fallback): Preposition Functor Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned PREP_FUNCTOR_TYPE (POS Fallback). PREP_FUNCTOR_TYPE is None? {PREP_FUNCTOR_TYPE is None}")
        elif pos == "ADV":
            assigned_entity = ADV_FUNCTOR_TYPE
            logger.debug(f"  Decision (POS Fallback): Adverb Functor Type assigned.") # User's log
            print(f"DEBUG_PRINT:     Assigned ADV_FUNCTOR_TYPE (POS Fallback). ADV_FUNCTOR_TYPE is None? {ADV_FUNCTOR_TYPE is None}")
        elif pos == "VERB":
            assigned_entity = VERB_TRANS_TYPE if sentence_has_object else VERB_INTRANS_TYPE
            logger.warning(f"  Decision (POS Fallback): Verb '{token_text}' wasn't assigned role/type. Defaulting.") # User's log
            print(f"DEBUG_PRINT:     Assigned VERB type (POS Fallback): {'Transitive' if sentence_has_object else 'Intransitive'}. VT None? {VERB_TRANS_TYPE is None}, VI None? {VERB_INTRANS_TYPE is None}")
        else:
            print(f"DEBUG_PRINT:     No POS fallback rule matched for POS='{pos}'.")


    # --- 5. Handle Ignored POS tags ---
    print(f"DEBUG_PRINT: Section 5: Ignored POS. assigned_entity is None? {assigned_entity is None}. POS='{pos}'")
    if assigned_entity is None:
        print(f"DEBUG_PRINT:   Attempting to handle ignored/problematic POS tags.")
        if pos in ["PUNCT", "SYM"]: # These are truly ignored and will result in None if no other rule caught them.
            logger.debug(f"  Decision (POS ignore/Final): Explicitly None for POS {pos} (e.g., punctuation).")
            # assigned_entity remains None, this is intended for these specific POS tags to be filtered out later.
            print(f"DEBUG_PRINT:     POS '{pos}' is truly ignored. assigned_entity remains None.")
        elif pos in ["PART", "CCONJ", "SCONJ", "AUX", "INTJ"]: # FIX: These were previously ignored, potentially causing 'None' return.
            logger.warning(f"POS '{pos}' (e.g., CCONJ, PART) was previously leading to 'None'. Assigning default N to prevent critical error. Original DepRel: '{dep_rel}'.")
            assigned_entity = N # Assign N as a fallback to prevent returning None.
            print(f"DEBUG_PRINT:     POS '{pos}' (was leading to None) defaulted to N. assigned_entity: {assigned_entity}")
        else: # This is the ultimate fallback if not PUNCT/SYM and not one of the newly handled (PART, CCONJ etc.)
            logger.warning(f"Unhandled POS/DepRel combination in final fallback: POS='{pos}', DepRel='{dep_rel}'. Defaulting to N.")
            assigned_entity = N
            print(f"DEBUG_PRINT:     ULTIMATE FALLBACK: Unhandled POS='{pos}'. Defaulted to N.")

    # --- Final Check and Logging ---
    print(f"DEBUG_PRINT: Final Checks. Current assigned_entity: {assigned_entity} (type: {type(assigned_entity).__name__ if assigned_entity else 'NoneType'})")
    if assigned_entity is None:
        # This critical log was here, but the return was missing in the original snippet. Adding return.
        logger.critical(f"  CRITICAL FALLBACK: assign_discocat_types_v2_2 is returning None for '{token_text}' (POS: {pos}, Lemma: {lemma}). This will break diagram construction.")
        print(f"DEBUG_PRINT:   CRITICAL - assigned_entity is STILL None. Returning None.")
        return None # Explicitly return None if it's still None here.
    # else: # This log was in the original snippet, but the return was missing.
    # logger.debug(f"  FINAL RETURN for '{token_text}' (POS {pos}): Type='{type(assigned_entity).__name__}', Value='{str(assigned_entity)}'")
    # pass # The actual return is further down after more checks in the original code.


    # --- User's original final checks for Box types ---
    # This block seems to intend to correct assignments if the global functor types were None
    # but a Box with that name was somehow assigned (e.g. a dummy Box).
    if isinstance(assigned_entity, Box): # Make sure Box is defined
        print(f"DEBUG_PRINT:   Final check for Box instance: Name='{assigned_entity.name}'")
        original_assigned_entity_name_for_check = assigned_entity.name # Store for logging if changed
        changed_in_final_check = False

        # Check if global functor types are valid Box instances or Ty instances before comparing names
        # This section assumes global functor types are Box instances with a .name attribute.
        if assigned_entity.name == "DetFunctor" and (not isinstance(DET_TYPE, Box) or DET_TYPE is None):
            assigned_entity = N; logger.error("DET_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        if assigned_entity.name == "AdjModFunctor" and (not isinstance(ADJ_MOD_TYPE, Box) or ADJ_MOD_TYPE is None):
            assigned_entity = N; logger.error("ADJ_MOD_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        if assigned_entity.name == "PrepFunctor" and (not isinstance(PREP_FUNCTOR_TYPE, Box) or PREP_FUNCTOR_TYPE is None):
            assigned_entity = N; logger.error("PREP_FUNCTOR_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        if assigned_entity.name == "AdjPredFunctor" and (not isinstance(ADJ_PRED_TYPE, Box) or ADJ_PRED_TYPE is None):
            assigned_entity = N; logger.error("ADJ_PRED_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        if assigned_entity.name == "VerbIntransFunctor" and (not isinstance(VERB_INTRANS_TYPE, Box) or VERB_INTRANS_TYPE is None):
            assigned_entity = N; logger.error("VERB_INTRANS_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        if assigned_entity.name == "VerbTransFunctor" and (not isinstance(VERB_TRANS_TYPE, Box) or VERB_TRANS_TYPE is None):
            assigned_entity = N; logger.error("VERB_TRANS_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        if assigned_entity.name == "AdvFunctor" and (not isinstance(ADV_FUNCTOR_TYPE, Box) or ADV_FUNCTOR_TYPE is None):
            assigned_entity = N; logger.error("ADV_FUNCTOR_TYPE is None or not a Box! Resetting to N."); changed_in_final_check = True
        
        if assigned_entity.name.startswith("NounPred_") and (assigned_entity.dom != N or assigned_entity.cod != S):
            logger.error(f"Dynamic Noun Predicate {assigned_entity.name} has incorrect type {assigned_entity.dom}->{assigned_entity.cod}. Resetting to N.") # User's log
            assigned_entity = N; changed_in_final_check = True
        
        if changed_in_final_check:
            print(f"DEBUG_PRINT:     Assigned entity '{original_assigned_entity_name_for_check}' was reset to N due to its corresponding global functor being None/malformed or Box type mismatch.")


    # if debug: # User's original debug log flag # This debug flag is from the function signature
    final_type_str = str(assigned_entity) if assigned_entity else "None"
    # Ensure logger.debug is used if debug is True, or just print for general debug prints
    logger.debug(f"  >> Final Assigned Type V2.4 for '{token_text}': {final_type_str} (Type: {type(assigned_entity).__name__ if assigned_entity else 'NoneType'})")

    print(f"DEBUG_PRINT: >>> EXITING assign_discocat_types_v2_2 for '{token_text}'. Returning: {str(assigned_entity)} (Type: {type(assigned_entity).__name__ if assigned_entity else 'NoneType'}) <<<\n")
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
def build_np_diagram_v4(
    head_noun_idx: int,
    analysis_map: Dict[int, Dict[str, Any]],
    roles: Dict[str, Any],
    core_type_map: Dict[int, Union[Word, Box, None]], # Expects Word or Box
    arg_producer_boxes: Dict[int, Box],
    functor_boxes: Dict[int, Box],
    globally_processed_indices: Set[int], # Indices already processed outside this NP build
    debug: bool = True
) -> Optional[GrammarDiagram]: # Changed GrammarDiagram to Diagram
    """
    Recursively builds a diagram for a Noun Phrase centered around head_noun_idx.
    Handles DET, ADJ (amod), and PP (nmod) modifiers.
    V4.1 (Corrected Type Handling): Adjusted to work with core_type_map containing Word/Box.
    Correctly checks types and uses arg_producer_boxes.
    Returns a diagram of type N (i.e., codomain is N).
    """
    # These types should be globally available or imported
    # N, S, ADJ_MOD_TYPE, DET_TYPE, PREP_FUNCTOR_TYPE, N_MOD_BY_N, Ty, Box, Word, GrammarDiagram

    # Keep track of indices consumed specifically by *this* NP build call to avoid re-processing within the same NP.
    indices_consumed_by_this_np_build: Set[int] = set()

    if head_noun_idx in globally_processed_indices:
        logger.warning(f"NP_V4.1_CORRECTED: Head Noun {head_noun_idx} ('{analysis_map.get(head_noun_idx,{}).get('text')}') is in `globally_processed_indices`. Cannot be head of new NP. Returning None.")
        return None 

    head_analysis = analysis_map.get(head_noun_idx)
    if not head_analysis:
        logger.error(f"NP_V4.1_CORRECTED: Analysis not found for head_noun_idx {head_noun_idx}.")
        return None

    # Check if the head token can indeed form an NP head based on its POS and role.
    # And importantly, check its DisCoCat type from core_type_map.
    head_entity_from_map = core_type_map.get(head_noun_idx)
    is_head_n_type_word = isinstance(head_entity_from_map, Word) and head_entity_from_map.cod == N
    
    allowed_pos_for_head = ["NOUN", "PROPN", "PRON"]
    is_subj_or_obj_role = (head_noun_idx == roles.get('subject') or head_noun_idx == roles.get('object'))
    head_pos = head_analysis['upos']
    head_deprel = head_analysis['deprel']

    can_be_np_head_by_pos_role = head_pos in allowed_pos_for_head or \
                                (head_pos == 'X' and is_subj_or_obj_role) or \
                                (head_pos == 'DET' and is_subj_or_obj_role and head_deprel == 'nsubj')

    if not (can_be_np_head_by_pos_role and is_head_n_type_word):
        logger.error(f"NP_V4.1_CORRECTED: Head {head_noun_idx} ('{head_analysis['text']}') - POS/Role: '{head_pos}'/{head_deprel}, Subj/Obj={is_subj_or_obj_role}. Core type: {head_entity_from_map}. Not a valid N-type NP head. Cannot form NP.")
        return None

    # Start with the base noun box (Ty() -> N) from arg_producer_boxes.
    # arg_producer_boxes should have been populated correctly by the calling function.
    np_diagram = arg_producer_boxes.get(head_noun_idx)
    if np_diagram is None:
        # This case should ideally not happen if arg_producer_boxes was populated correctly
        # for all N-type Words. But as a fallback:
        logger.warning(f"NP_V4.1_CORRECTED: Arg producer box not found for head {head_noun_idx} ('{head_analysis['text']}'). Creating one.")
        box_name = f"{head_analysis.get('lemma', head_analysis.get('text','unk'))}_{head_noun_idx}_arg"
        np_diagram = Box(box_name, Ty(), N) # Create the Ty() -> N box
        # arg_producer_boxes[head_noun_idx] = np_diagram # Optionally add to shared dict
    
    logger.info(f"NP_V4.1_CORRECTED: Building NP for head: '{head_analysis['text']}' (idx {head_noun_idx}), Initial diagram: {get_diagram_repr(np_diagram)}")
    indices_consumed_by_this_np_build.add(head_noun_idx) # Mark head as consumed by this NP

    dep_graph = roles.get('dependency_graph', {})
    dependents = dep_graph.get(head_noun_idx, []) if isinstance(dep_graph, dict) else []
    logger.debug(f"  NP_V4.1_CORRECTED: Dependents of {head_noun_idx} ('{head_analysis['text']}'): {dependents}")

    modifiers_to_apply_data = [] 

    for dep_idx, dep_rel in dependents:
        if dep_idx in globally_processed_indices or dep_idx in indices_consumed_by_this_np_build:
            logger.debug(f"  NP_V4.1_CORRECTED: Dependent idx {dep_idx} already processed globally or by this NP. Skipping.")
            continue

        modifier_analysis = analysis_map.get(dep_idx)
        if not modifier_analysis: continue
        
        logger.debug(f"  NP_V4.1_CORRECTED: Considering dependent: '{modifier_analysis['text']}' (idx {dep_idx}), deprel='{dep_rel}', POS='{modifier_analysis['upos']}'")

        current_mod_indices_subtree: Set[int] = {dep_idx} 
        mod_diagram_component: Optional[Diagram] = None # Can be Box or Diagram from recursive call

        modifier_functor_box = functor_boxes.get(dep_idx) # Check pre-built functors

        if dep_rel == 'det' and modifier_analysis['upos'] == 'DET':
            if modifier_functor_box and modifier_functor_box.dom == N and modifier_functor_box.cod == N:
                mod_diagram_component = modifier_functor_box
                modifiers_to_apply_data.append({'box': mod_diagram_component, 'order': 0, 'indices_of_mod_subtree': current_mod_indices_subtree, 'rel': 'det'})
        
        elif dep_rel == 'amod' and modifier_analysis['upos'] == 'ADJ':
            if modifier_functor_box and modifier_functor_box.dom == N and modifier_functor_box.cod == N:
                mod_diagram_component = modifier_functor_box
                modifiers_to_apply_data.append({'box': mod_diagram_component, 'order': 1, 'indices_of_mod_subtree': current_mod_indices_subtree, 'rel': 'amod'})
        
        elif dep_rel == 'nmod' and modifier_analysis['upos'] in ["NOUN", "PROPN", "PRON"]: # Idafa construction
            if N_MOD_BY_N is None: # N_MOD_BY_N should be (N @ N) -> N
                logger.warning(f"    NP_V4.1_CORRECTED nmod: N_MOD_BY_N functor is None. Cannot process Idafa.")
                continue
            
            # Exclude current NP's head and already processed items from recursive call's "globally_processed"
            # Also exclude items already consumed by *this current* NP build further up the modifier chain.
            indices_for_recursive_call_exclusion = globally_processed_indices.union(indices_consumed_by_this_np_build)
            
            logger.debug(f"    NP_V4.1_CORRECTED nmod: Recursive NP build for: '{modifier_analysis['text']}' (idx {dep_idx}). Excluding: {indices_for_recursive_call_exclusion}")
            
            modifier_np_sub_diagram = build_np_diagram_v4( # Recursive call
                dep_idx, analysis_map, roles, core_type_map,
                arg_producer_boxes, functor_boxes, 
                indices_for_recursive_call_exclusion, # Pass the combined set
                debug
            )
            if modifier_np_sub_diagram and modifier_np_sub_diagram.cod == N:
                mod_diagram_component = modifier_np_sub_diagram
                # The recursive call will add its consumed indices to its *copy* of processed_indices.
                # We need to capture which indices it *newly* processed to add to our current NP's consumption.
                # For simplicity, we assume build_np_diagram_v4_corrected would return the set of indices it consumed,
                # or we rely on the fact that it modifies the passed set (if it's the same object, which it is here).
                # The `indices_consumed_by_this_np_build` will be updated after successful application.
                modifiers_to_apply_data.append({
                    'box': mod_diagram_component, 
                    'nmod_functor': N_MOD_BY_N, 
                    'order': 2, 
                    'indices_of_mod_subtree': {dep_idx}, # Head of the nmod NP
                    'rel': 'nmod_idafa'
                })
            else:
                logger.warning(f"    NP_V4.1_CORRECTED nmod: Recursive NP build for nmod '{modifier_analysis['text']}' (idx {dep_idx}) failed or not N type.")
        
        # TODO: Add PP (prepositional phrase) modifier handling if needed
        # This would involve finding a 'case' ADP, its object (another NP build),
        # composing them into an N->N modifier, then applying to np_diagram.

    modifiers_to_apply_data.sort(key=lambda m: m['order'])

    for mod_info in modifiers_to_apply_data:
        mod_component = mod_info['box']
        
        # Check if any part of this modifier's subtree has already been consumed by *this current* NP build
        # (e.g., if modifiers were not strictly hierarchical and shared sub-dependents)
        can_apply_modifier = True
        for mod_sub_idx in mod_info['indices_of_mod_subtree']: # This currently only contains the head of the modifier
            if mod_sub_idx != head_noun_idx and mod_sub_idx in indices_consumed_by_this_np_build:
                logger.warning(f"  NP_V4.1_CORRECTED: Index {mod_sub_idx} for modifier '{get_diagram_repr(mod_component)}' already consumed by this NP build. Skipping this modifier application.")
                can_apply_modifier = False; break
        if not can_apply_modifier: continue

        try:
            if mod_info['rel'] in ['det', 'amod']:
                if np_diagram and np_diagram.cod == mod_component.dom: # type: ignore
                    np_diagram = np_diagram >> mod_component
                    indices_consumed_by_this_np_build.update(mod_info['indices_of_mod_subtree'])
                    logger.debug(f"    Applied {mod_info['rel']} modifier. NP cod: {np_diagram.cod}") # type: ignore
            elif mod_info['rel'] == 'nmod_idafa': 
                nmod_functor_to_apply = mod_info['nmod_functor']
                # mod_component is the NP diagram of the second noun in Idafa (cod N)
                # np_diagram is the NP diagram of the first noun in Idafa (cod N)
                if np_diagram and np_diagram.cod == N and mod_component.cod == N: # type: ignore
                    np_diagram = (np_diagram @ mod_component) >> nmod_functor_to_apply
                    indices_consumed_by_this_np_build.update(mod_info['indices_of_mod_subtree'])
                    logger.debug(f"    Applied nmod_idafa. NP cod: {np_diagram.cod}") # type: ignore
            # ... (PP application logic if added) ...
        except Exception as e_mod_apply:
            logger.error(f"    NP_V4.1_CORRECTED Error applying modifier for {mod_info.get('rel')}: {e_mod_apply}", exc_info=True)
    
    # Update the globally_processed_indices with everything consumed by this successful NP build
    globally_processed_indices.update(indices_consumed_by_this_np_build)
    
    logger.info(f"NP_V4.1_CORRECTED: Finished NP for '{head_analysis['text']}'. Final diagram: {get_diagram_repr(np_diagram)}, Cod: {np_diagram.cod if np_diagram else 'None'}. Indices consumed by THIS NP build call: {indices_consumed_by_this_np_build - {head_noun_idx}} (plus head {head_noun_idx})")

    if np_diagram and hasattr(np_diagram, 'cod') and np_diagram.cod == N:
        return np_diagram
    else:
        logger.error(f"NP_V4.1_CORRECTED: Final NP for head {head_noun_idx} ('{head_analysis['text']}') is invalid or not N-type. Diagram: {get_diagram_repr(np_diagram)}")
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

def create_verbal_sentence_diagram_v3_7(
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Word, Box, None]], # Expects Word or Box
    original_indices: List[int],
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_verbal"
) -> Optional[GrammarDiagram]: # Changed GrammarDiagram
    """
    Creates a DisCoCat diagram for verbal sentences.
    V3.7.1 (Corrected Type Handling): Adjusted for word_core_types containing Word/Box.
    Calls build_np_diagram_v4_corrected.
    """
    logger.info(f"VERBAL_DIAG_V3.7.1_CORRECTED: Creating diagram for: \"{' '.join(tokens)}\"")
    # Ensure essential global types are available (N, S, VERB_TRANS_TYPE, etc.)
    if any(v is None for v in [N, S, VERB_TRANS_TYPE, VERB_INTRANS_TYPE, S_MOD_BY_N, ADV_FUNCTOR_TYPE]):
         logger.error("VERBAL_DIAG_V3.7.1_CORRECTED: Essential Verb/Modifier types are not defined.")
         return None

    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map: Dict[int, Union[Word, Box, None]] = {}
    if len(original_indices) == len(word_core_types):
        core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}
    else:
        logger.error("VERBAL_DIAG_V3.7.1_CORRECTED: Mismatch: original_indices vs word_core_types lengths.")
        return None

    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}
    processed_indices: Set[int] = set()

    for orig_idx, entity_from_map in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or entity_from_map is None: continue
        box_name_base = analysis.get('lemma', analysis.get('text', 'UNK'))
        box_name = f"{box_name_base}_{orig_idx}"

        if isinstance(entity_from_map, Box):
            temp_box = Box(box_name, entity_from_map.dom, entity_from_map.cod) # Create Box
            if hasattr(entity_from_map, 'data') and entity_from_map.data is not None:
                temp_box.data = entity_from_map.data.copy() # Assign data
            else:
                temp_box.data = {} # Ensure data attribute exists
            functor_boxes[orig_idx] = temp_box
        elif isinstance(entity_from_map, Word):
            if entity_from_map.cod == N:
                temp_box = Box(box_name, Ty(), entity_from_map.cod) # Create Box
                if hasattr(entity_from_map, 'data') and entity_from_map.data is not None:
                    temp_box.data = entity_from_map.data.copy() # Assign data
                else:
                    temp_box.data = {}
                arg_producer_boxes[orig_idx] = temp_box
            # elif entity_from_map.dom != Ty(): # If Word could be a functor
            #     temp_box_f = Box(box_name, entity_from_map.dom, entity_from_map.cod)
            #     if hasattr(entity_from_map, 'data') and entity_from_map.data is not None:
            #         temp_box_f.data = entity_from_map.data.copy()
            #     else:
            #         temp_box_f.data = {}
            #     functor_boxes[orig_idx] = temp_box_f

    subj_idx = roles.get('subject')
    verb_idx = roles.get('verb')
    obj_idx = roles.get('object')

    verb_functor_box = functor_boxes.get(verb_idx) if verb_idx is not None else None
    if verb_functor_box is None:
         verb_entity_core = core_type_map.get(verb_idx) if verb_idx is not None else None
         if isinstance(verb_entity_core, Box) and verb_entity_core.cod == S:
             verb_analysis = analysis_map.get(verb_idx)
             vb_name = f"{verb_analysis.get('lemma', 'verb')}_{verb_idx}" if verb_analysis else f"verb_{verb_idx}"
             
             temp_verb_box = Box(vb_name, verb_entity_core.dom, verb_entity_core.cod)
             if hasattr(verb_entity_core, 'data') and verb_entity_core.data is not None:
                 temp_verb_box.data = verb_entity_core.data.copy()
             else:
                 temp_verb_box.data = {}
             verb_functor_box = temp_verb_box
             functor_boxes[verb_idx] = verb_functor_box
             logger.warning(f"  Used verb functor directly from core_type_map for idx {verb_idx}: {get_diagram_repr(verb_functor_box)}")
         else:
            logger.error(f"VERBAL_DIAG_V3.7.2: Main verb functor not found or invalid for index {verb_idx}.")
            return None

    subj_np_diagram: Optional[Diagram] = None
    if subj_idx is not None:
        subj_np_diagram = build_np_diagram_v4(
            subj_idx, analysis_map, roles, core_type_map,
            arg_producer_boxes, functor_boxes, processed_indices, debug
        )
        if not (subj_np_diagram and subj_np_diagram.cod == N):
            logger.warning(f"  Subject NP build for {subj_idx} failed/invalid. Falling back to simple arg box.")
            subj_np_diagram = arg_producer_boxes.get(subj_idx)
            if subj_np_diagram and subj_idx not in processed_indices: processed_indices.add(subj_idx)
        # If still None or invalid, error will be caught before composition

    obj_np_diagram: Optional[Diagram] = None
    obj_analysis = analysis_map.get(obj_idx) if obj_idx is not None else None
    if obj_analysis and obj_analysis['upos'] == 'ADJ' and obj_analysis['deprel'] == 'xcomp':
        logger.warning(f"  Object role {obj_idx} is ADJ xcomp. Treating as non-object for NP build.")
        if obj_idx not in processed_indices: processed_indices.add(obj_idx)
    elif obj_idx is not None and obj_idx not in processed_indices:
        obj_np_diagram = build_np_diagram_v4(
            obj_idx, analysis_map, roles, core_type_map,
            arg_producer_boxes, functor_boxes, processed_indices, debug
        )
        if not (obj_np_diagram and obj_np_diagram.cod == N):
            logger.warning(f"  Object NP build for {obj_idx} failed/invalid. Falling back to simple arg box.")
            obj_np_diagram = arg_producer_boxes.get(obj_idx)
            if obj_np_diagram and obj_idx not in processed_indices: processed_indices.add(obj_idx)
    elif obj_idx is not None and obj_idx in processed_indices:
         logger.warning(f"  Object index {obj_idx} already processed. Skipping object diagram build.")


    if verb_idx is not None: processed_indices.add(verb_idx)
    logger.debug(f"  Components: Subj(idx {subj_idx}): {get_diagram_repr(subj_np_diagram)}, Verb(idx {verb_idx}): {get_diagram_repr(verb_functor_box)}, Obj(idx {obj_idx}): {get_diagram_repr(obj_np_diagram)}")

    final_diagram: Optional[Diagram] = None
    structure_type = roles.get("structure", "UNKNOWN_VERBAL")

    try:
        if verb_functor_box.dom == (N @ N): # Transitive N @ N >> S
            if subj_np_diagram and subj_np_diagram.cod == N and obj_np_diagram and obj_np_diagram.cod == N:
                final_diagram = (subj_np_diagram @ obj_np_diagram) >> verb_functor_box
            elif subj_np_diagram and subj_np_diagram.cod == N and obj_np_diagram is None:
                logger.warning(f"  Transitive verb '{get_diagram_repr(verb_functor_box)}' with subject but MISSING object. Using Id(N) for object.")
                final_diagram = (subj_np_diagram @ Id(N)) >> verb_functor_box # Id(N) is Lambeq's identity for type N
            elif obj_np_diagram and obj_np_diagram.cod == N and subj_np_diagram is None:
                logger.warning(f"  Transitive verb '{get_diagram_repr(verb_functor_box)}' with object but MISSING subject. Using Id(N) for subject.")
                final_diagram = (Id(N) @ obj_np_diagram) >> verb_functor_box
            else:
                s_stat = "valid" if subj_np_diagram and subj_np_diagram.cod == N else "invalid/missing"
                o_stat = "valid" if obj_np_diagram and obj_np_diagram.cod == N else "invalid/missing (or intentionally None for xcomp)"
                logger.error(f"  Cannot compose transitive clause ({structure_type}): Subj is {s_stat}, Obj is {o_stat}.")
                return None
        elif verb_functor_box.dom == N: # Intransitive N >> S
            if subj_np_diagram and subj_np_diagram.cod == N:
                final_diagram = subj_np_diagram >> verb_functor_box
            elif subj_np_diagram is None and structure_type == "VERBAL_ONLY": # e.g. "يكتب"
                 logger.warning(f"  Intransitive verb '{get_diagram_repr(verb_functor_box)}' in VERBAL_ONLY. Using Id(N) for subject.")
                 final_diagram = Id(N) >> verb_functor_box
            elif subj_np_diagram is None: # Try to find default subject
                 logger.warning(f"  Intransitive verb '{get_diagram_repr(verb_functor_box)}' with no explicit subject. Searching for default N.")
                 default_subj_diag: Optional[Diagram] = None
                 for idx_cand, arg_box_cand in arg_producer_boxes.items():
                     if idx_cand not in processed_indices and idx_cand != verb_idx:
                         default_subj_diag = arg_box_cand; processed_indices.add(idx_cand); break
                 if default_subj_diag and default_subj_diag.cod == N:
                     final_diagram = default_subj_diag >> verb_functor_box
                 else:
                     logger.warning(f"  No default subject found. Using Id(N).")
                     final_diagram = Id(N) >> verb_functor_box
            else: # Subject was present but invalid type
                logger.error(f"  Cannot compose intransitive clause ({structure_type}): Subject diagram invalid: {get_diagram_repr(subj_np_diagram)}")
                return None
        elif verb_functor_box.dom == Ty(): # Verb is Ty() >> S
            final_diagram = verb_functor_box
        else:
            logger.error(f"  Unhandled verb functor type: {verb_functor_box.dom} >> {verb_functor_box.cod}")
            return None
        
        logger.info(f"  Successfully composed basic clause. Diagram cod: {final_diagram.cod if final_diagram else 'None'}")

    except Exception as e_clause:
        logger.error(f"  Error during basic clause composition ({structure_type}): {e_clause}", exc_info=True)
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


def create_nominal_sentence_diagram_v2_7(
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Word, Box, None]], # Corrected: Expects Word or Box
    original_indices: List[int],
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_nominal",
    hint_predicate_original_idx: Optional[int] = None
) -> Optional[GrammarDiagram]: # Changed GrammarDiagram to Diagram for consistency with placeholders
    """
    Creates a DisCoCat diagram for nominal sentences (Subject-Predicate).
    V2.7.3 (Corrected Box data instantiation & Type Handling):
    - Fixes TypeError for Box.__init__() by assigning .data attribute after instantiation.
    - Adjusted to work with word_core_types containing Word/Box.
    - Calls build_np_diagram_v4_corrected (which also needs this Box data fix).
    """
    global N, S, ADJ_PRED_TYPE, Ty, Box, Word, GrammarDiagram, Id # Ensure access

    logger.info(f"NOMINAL_DIAG_V2.7.3 (Box data fix): Creating diagram for: \"{' '.join(tokens)}\"")

    if ADJ_PRED_TYPE is None:
         logger.error("NOMINAL_DIAG_V2.7.3: ADJ_PRED_TYPE is not defined.")
         return None

    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map: Dict[int, Union[Word, Box, None]] = {}
    if len(original_indices) == len(word_core_types):
        core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}
    else:
        logger.error("NOMINAL_DIAG_V2.7.3: Mismatch: original_indices vs word_core_types lengths.")
        return None

    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}
    processed_indices: Set[int] = set()

    for orig_idx, entity_from_map in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or entity_from_map is None: continue
        
        box_name_base = analysis.get('lemma', analysis.get('text', 'UNK'))
        box_name = f"{box_name_base}_{orig_idx}"

        if isinstance(entity_from_map, Box):
            temp_box = Box(box_name, entity_from_map.dom, entity_from_map.cod) # Create Box
            if hasattr(entity_from_map, 'data') and entity_from_map.data is not None:
                temp_box.data = entity_from_map.data.copy() # Assign data
            else:
                temp_box.data = {} # Ensure data attribute exists
            functor_boxes[orig_idx] = temp_box
        elif isinstance(entity_from_map, Word):
            if entity_from_map.cod == N:
                temp_box = Box(box_name, Ty(), entity_from_map.cod) # Create Box
                if hasattr(entity_from_map, 'data') and entity_from_map.data is not None:
                    temp_box.data = entity_from_map.data.copy() # Assign data
                else:
                    temp_box.data = {}
                arg_producer_boxes[orig_idx] = temp_box

    subj_head_idx = roles.get('subject')
    predicate_idx: Optional[int] = None
    predicate_functor_box: Optional[Box] = None

    if hint_predicate_original_idx is not None:
        hinted_functor_candidate = functor_boxes.get(hint_predicate_original_idx)
        if hinted_functor_candidate and hinted_functor_candidate.dom == N and hinted_functor_candidate.cod == S:
            predicate_idx = hint_predicate_original_idx
            predicate_functor_box = hinted_functor_candidate
            logger.info(f"  Using HINTED predicate from functor_boxes: '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")
        else:
            entity_at_hint = core_type_map.get(hint_predicate_original_idx)
            if isinstance(entity_at_hint, Box) and entity_at_hint.dom == N and entity_at_hint.cod == S:
                predicate_idx = hint_predicate_original_idx
                box_name_base_hint = analysis_map.get(predicate_idx, {}).get('lemma', 'hinted_pred')
                box_name_hint = f"{box_name_base_hint}_{predicate_idx}"
                
                temp_box_hint = Box(box_name_hint, entity_at_hint.dom, entity_at_hint.cod)
                if hasattr(entity_at_hint, 'data') and entity_at_hint.data is not None:
                    temp_box_hint.data = entity_at_hint.data.copy()
                else:
                    temp_box_hint.data = {}
                predicate_functor_box = temp_box_hint
                functor_boxes[predicate_idx] = predicate_functor_box # Ensure it's in functor_boxes
                logger.info(f"  Using HINTED predicate (from core_type_map Box, new instance): '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")
            else:
                logger.warning(f"  Hinted predicate idx {hint_predicate_original_idx} not a valid N->S functor.")

    if predicate_functor_box is None:
        logger.debug("  No valid hinted predicate. Searching for pre-assigned N->S functor in functor_boxes.")
        # Simplified search, prioritize root if it's a predicate
        potential_candidates = []
        if roles.get('root') is not None and roles.get('root') != subj_head_idx:
            pfb = functor_boxes.get(roles['root'])
            if pfb and pfb.dom == N and pfb.cod == S:
                potential_candidates.append({'idx': roles['root'], 'box': pfb, 'priority': 0}) # Highest priority

        for idx, f_box in functor_boxes.items():
            if idx == subj_head_idx or idx == roles.get('root'): continue
            if f_box.dom == N and f_box.cod == S:
                priority = 1 # Default for other N->S functors
                pred_analysis = analysis_map.get(idx)
                if pred_analysis and pred_analysis.get('head') == subj_head_idx:
                    priority = 0 # Higher priority if headed by subject (but not subject itself)
                potential_candidates.append({'idx': idx, 'box': f_box, 'priority': priority})
        
        if potential_candidates:
            potential_candidates.sort(key=lambda x: x['priority'])
            selected_candidate = potential_candidates[0]
            predicate_idx = selected_candidate['idx']
            predicate_functor_box = selected_candidate['box']
            logger.info(f"  Found FALLBACK predicate functor: '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")

    subj_np_diagram: Optional[Diagram] = None
    if subj_head_idx is None:
        logger.error("NOMINAL_DIAG_V2.7.1_CORRECTED: Subject head index is None. Cannot build subject NP.")
        # Attempt to find a default subject if predicate is known
        if predicate_idx is not None:
            logger.warning("  Attempting to find a default subject as predicate is known but subject_head_idx is None.")
            for idx_candidate, arg_box_candidate in arg_producer_boxes.items():
                if idx_candidate not in processed_indices and idx_candidate != predicate_idx:
                    temp_processed_for_default_subj = processed_indices.copy()
                    if predicate_idx is not None: temp_processed_for_default_subj.add(predicate_idx)

                    subj_diag_candidate = build_np_diagram_v4( # Call corrected version
                        idx_candidate, analysis_map, roles, core_type_map,
                        arg_producer_boxes, functor_boxes, temp_processed_for_default_subj, debug
                    )
                    if subj_diag_candidate and subj_diag_candidate.cod == N:
                        subj_np_diagram = subj_diag_candidate
                        subj_head_idx = idx_candidate
                        processed_indices.update(temp_processed_for_default_subj - {predicate_idx} if predicate_idx is not None else temp_processed_for_default_subj)
                        logger.info(f"  Found and built default subject NP: '{get_diagram_repr(subj_np_diagram)}' (orig_idx {subj_head_idx})")
                        break
                    # Fallback to simple arg_box_candidate if NP build fails is implicitly handled if build_np_diagram_v4_corrected returns it
            if subj_np_diagram is None:
                logger.error("  Could not find or build any suitable default subject diagram.")
                return None
        else:
            logger.error("NOMINAL_DIAG_V2.7.1_CORRECTED: Subject head index is None and no predicate to anchor default subject search.")
            return None
    else:
        logger.debug(f"Attempting to build subject NP for explicit subj_head_idx: {subj_head_idx}")
        temp_processed_for_subj_np = processed_indices.copy()
        if predicate_idx is not None: temp_processed_for_subj_np.add(predicate_idx)

        subj_np_diagram = build_np_diagram_v4( # Call corrected version
            subj_head_idx, analysis_map, roles, core_type_map,
            arg_producer_boxes, functor_boxes, temp_processed_for_subj_np, debug
        )
        
        if subj_np_diagram and subj_np_diagram.cod == N:
            processed_indices.update(temp_processed_for_subj_np - {predicate_idx} if predicate_idx is not None else temp_processed_for_subj_np)
            logger.info(f"  Successfully built subject NP for idx {subj_head_idx}: {get_diagram_repr(subj_np_diagram)}")
        else:
            logger.warning(f"  Subject NP build for idx {subj_head_idx} failed or not N-type. Falling back to simple arg_producer_box.")
            subj_np_diagram = arg_producer_boxes.get(subj_head_idx)
            if subj_np_diagram and subj_np_diagram.cod == N:
                logger.info(f"  Using fallback arg_producer_box for subject {subj_head_idx}: {get_diagram_repr(subj_np_diagram)}")
                if subj_head_idx not in processed_indices: processed_indices.add(subj_head_idx)
            else:
                logger.error(f"  Fallback failed: Simple arg box missing/invalid for subject {subj_head_idx}.")
                return None

    if subj_head_idx is not None and subj_head_idx not in processed_indices: # Should be processed by now
        processed_indices.add(subj_head_idx)

    if predicate_functor_box is None and subj_np_diagram and subj_np_diagram.cod == N:
        logger.debug("  No pre-assigned predicate. Searching for dynamic predicate for subject: " + get_diagram_repr(subj_np_diagram))
        all_analyses_for_dyn_pred = roles.get('analyses_details_for_context', analyses_details)
        potential_dyn_pred_idx: Optional[int] = None
        potential_dyn_pred_analysis: Optional[Dict[str, Any]] = None

        for token_analysis in all_analyses_for_dyn_pred:
            idx = token_analysis['original_idx']
            if idx in processed_indices or idx == subj_head_idx: continue

            # A dynamic predicate is often a NOUN or ADJ.
            # Its original type in core_type_map might be Word(name,N) or Word(name, N>>N for ADJ_MOD)
            entity_at_idx = core_type_map.get(idx)
            is_potential_pred_type = False
            if isinstance(entity_at_idx, Word) and entity_at_idx.cod == N: # Is it an N-type Word?
                is_potential_pred_type = True
            elif isinstance(entity_at_idx, Box) and entity_at_idx.dom == N and entity_at_idx.cod == N: # Is it an N->N Box (like AdjMod)?
                 is_potential_pred_type = True # Could still be turned into N->S

            if token_analysis['upos'] in ["ADJ", "NOUN", "PROPN", "NUM", "X"] and is_potential_pred_type:
                sentence_structure = roles.get('structure', "UNKNOWN")
                is_plausible_role = (token_analysis.get('deprel') == 'root') or \
                                    (subj_head_idx is not None and token_analysis.get('head') == subj_head_idx) or \
                                    (sentence_structure == "SUBJ_NO_VERB_OTHER" and idx == roles.get('root'))
                if is_plausible_role:
                    potential_dyn_pred_idx = idx
                    potential_dyn_pred_analysis = token_analysis
                    logger.info(f"  Found potential dynamic predicate: '{token_analysis['text']}' (idx {idx}, POS {token_analysis['upos']})")
                    break
        
        if potential_dyn_pred_idx is not None and potential_dyn_pred_analysis is not None:
            pred_lemma = potential_dyn_pred_analysis.get('lemma', potential_dyn_pred_analysis.get('text', 'unk'))
            pred_pos = potential_dyn_pred_analysis['upos']
            temp_dyn_pred_functor: Optional[Box] = None
            if pred_pos == "ADJ" and ADJ_PRED_TYPE is not None:
                temp_dyn_pred_functor = Box(f"DynAdjPred_{pred_lemma}_{potential_dyn_pred_idx}", N, S)
            elif pred_pos in ["NOUN", "PROPN", "NUM", "X"]:
                temp_dyn_pred_functor = Box(f"DynNounPred_{pred_lemma}_{potential_dyn_pred_idx}", N, S)
            
            if temp_dyn_pred_functor:
                predicate_idx = potential_dyn_pred_idx
                predicate_functor_box = temp_dyn_pred_functor
                functor_boxes[predicate_idx] = predicate_functor_box # Add to functor_boxes
                logger.info(f"  Dynamically created Predicate Functor: '{get_diagram_repr(predicate_functor_box)}'")
            else:
                logger.warning(f"  Dynamic predicate functor creation failed for {potential_dyn_pred_idx}.")
        else:
            logger.warning("  Dynamic predicate search: No suitable token found.")

    if not (subj_np_diagram and subj_np_diagram.cod == N):
        logger.error(f"Final Subject NP diagram invalid: {get_diagram_repr(subj_np_diagram)}")
        return None
    if not (predicate_idx is not None and predicate_functor_box and predicate_functor_box.dom == N and predicate_functor_box.cod == S):
        logger.error(f"Final Predicate functor invalid: Idx={predicate_idx}, Box={get_diagram_repr(predicate_functor_box)}")
        return None

    if predicate_idx not in processed_indices: # Mark predicate as processed
        processed_indices.add(predicate_idx)

    final_diagram: Optional[Diagram] = None
    try:
        logger.info(f"  Composing: Subject NP '{get_diagram_repr(subj_np_diagram)}' >> Predicate Functor '{get_diagram_repr(predicate_functor_box)}'")
        final_diagram = subj_np_diagram >> predicate_functor_box
        logger.info(f"  Nominal composition successful. Cod: {final_diagram.cod}")
    except Exception as e:
        logger.error(f"  Nominal composition error: {e}", exc_info=True)
        return None
    
    # Simplified sentence-level modifier attachment (add if needed)

    if final_diagram:
        try:
            normalized_diagram = final_diagram.normal_form()
            if normalized_diagram.cod == S:
                logger.info(f"  Nominal diagram normalization successful. Final cod: {normalized_diagram.cod}")
                return normalized_diagram
            else:
                logger.warning(f"  Nominal diagram normalized, but final cod is {normalized_diagram.cod}, not S. Discarding.")
                return None
        except Exception as e_norm:
            logger.error(f"  Nominal diagram normal_form failed: {e_norm}", exc_info=True)
            return None # Or return final_diagram if unnormalized is acceptable

    logger.warning(f"Could not form a complete nominal diagram ending in S for '{sentence_prefix}'")
    return None

# ==================================
# Main Conversion Function (V2.7 - Uses Hybrid Diagram Functions)
# ==================================

logger = logging.getLogger(__name__)


def arabic_to_quantum_enhanced_v2_7(
    sentence: str,
    debug: bool = True,
    output_dir: Optional[str] = None,
    ansatz_choice: str = "IQP",
    n_layers_iqp: int = 1,
    n_single_qubit_params_iqp: int = 2,
    n_layers_strong: int = 1,
    cnot_ranges: Optional[List[Tuple[int, int]]] = None,
    discard_qubits_spider: bool = True,
    handle_lexical_ambiguity_in_typing: bool = False,
    pre_initialized_functor: Optional[Functor] = None, # NEW # NEW PARAMETER
    **kwargs
) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Dict[str,Any]], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram, and converts it to a Qiskit QuantumCircuit.
    V2.7.6: Calls to diagram builders EXCLUDE core_type_map if not in their definition.
            Relies on word_core_types_list (List[Word|Box]) for type info.
    """
    global ARABIC_DISCOCIRC_PIPELINE_AVAILABLE, N, S, ADJ_PRED_TYPE, NOUN_TYPE_BOX_FALLBACK # Ensure globals

    generate_discocirc_ready_diagram_func = None
    try:
        from arabic_discocirc_pipeline import generate_discocirc_ready_diagram
        generate_discocirc_ready_diagram_func = generate_discocirc_ready_diagram
        ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = True
    except ImportError as e_discocirc_runtime:
        logger.error(f"Runtime import of generate_discocirc_ready_diagram from arabic_discocirc_pipeline failed: {e_discocirc_runtime}. Enriched diagrams unavailable.")
        ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = False

    if kwargs:
        logger.warning(f"Function arabic_to_quantum_enhanced_v2_7 received UNEXPECTED keyword arguments: {kwargs}")

    logger.info(f"Analyzing sentence: '{sentence}'")
    try:
        tokens, analyses_details, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug=debug)
        if not analyses_details and not tokens :
             logger.error(f"Core sentence analysis (analyze_arabic_sentence_with_morph) failed completely for: '{sentence}'")
             return None, None, structure if structure else "ERROR_ANALYSIS_EMPTY", [], [], {}

        roles['analysis_map_for_diagram_creation'] = {a['original_idx']: a for a in analyses_details} if analyses_details else {}
        roles['analyses_details_for_context'] = analyses_details

        if structure == "ERROR" or not tokens:
            logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'. Structure: {structure}")
            return None, None, structure, tokens or [], analyses_details or [], roles or {}
        logger.info(f"Analysis complete. Detected structure: {structure}.")
        if debug: logger.debug(f"Roles from analysis: {roles}")

    except Exception as e_analyze_main:
        logger.error(f"Sentence analysis (analyze_arabic_sentence_with_morph) failed unexpectedly: {e_analyze_main}", exc_info=True)
        return None, None, "ERROR_ANALYSIS_EXCEPTION", [], [], {}

    diagram: Optional[GrammarDiagram] = None
    used_enriched_diagram_path = False

    if ARABIC_DISCOCIRC_PIPELINE_AVAILABLE and generate_discocirc_ready_diagram_func:
        logger.info(f"Attempting to generate feature-enriched DisCoCirc diagram for: '{sentence}' via arabic_discocirc_pipeline")
        try:
            enriched_diagram = generate_discocirc_ready_diagram_func(
                sentence_str=sentence,
                sentence_analyzer_func=analyze_arabic_sentence_with_morph,
                type_assigner_func=assign_discocat_types_v2_2,
                debug=debug
            )
            if enriched_diagram is not None:
                logger.info(f"Successfully generated feature-enriched diagram via arabic_discocirc_pipeline. Boxes: {len(enriched_diagram.boxes) if hasattr(enriched_diagram, 'boxes') else 'N/A'}") # type: ignore
                diagram = enriched_diagram
                used_enriched_diagram_path = True
            else:
                logger.warning("Feature-enriched diagram generation (arabic_discocirc_pipeline) returned None. Proceeding with fallback diagram logic.")
        except Exception as e_discocirc_call:
            logger.error(f"Error calling generate_discocirc_ready_diagram: {e_discocirc_call}", exc_info=True)
            logger.warning("Proceeding with fallback diagram logic due to error in enriched pipeline.")
    else:
        logger.info("arabic_discocirc_pipeline not available or function not loaded. Using fallback diagram logic directly.")

    if not used_enriched_diagram_path:
        logger.info("Using internal camel_test2.py logic for diagram generation (fallback path).")
        word_core_types_list: List[Union[Word, Box]] = [] 
        original_indices_for_diagram: List[int] = []
        filtered_tokens_for_diagram: List[str] = []
        # This map stores the direct output of assign_discocat_types_v2_2 (Ty or Box or None)
        # It can be used by diagram creation functions if they need to look up the original assigned type for an index.
        # However, the primary input to diagram builders should be word_core_types_list.
        core_type_map_for_fallback: Dict[int, Union[Ty, Box, None]] = {}


        logger.debug(f"--- Assigning Core Types (Fallback Path) V2.2.5 logic for: '{sentence}' ---")
        for i, analysis_entry in enumerate(analyses_details):
            current_core_type = assign_discocat_types_v2_2(
                analysis=analysis_entry, roles=roles, debug=debug, handle_lexical_ambiguity=handle_lexical_ambiguity_in_typing # PASS THE FLAG
            )
            core_type_map_for_fallback[analysis_entry['original_idx']] = current_core_type

            if current_core_type is not None:
                diagrammatic_entity: Optional[Union[Word, Box]] = None
                token_text_fb = analysis_entry.get('text', f"unk_fb_{i}")
                token_lemma_fb = analysis_entry.get('lemma', token_text_fb)
                
                if isinstance(current_core_type, Ty):
                    diagrammatic_entity = Word(token_lemma_fb, current_core_type)
                elif isinstance(current_core_type, Box): 
                    diagrammatic_entity = current_core_type
                
                if diagrammatic_entity:
                    word_core_types_list.append(diagrammatic_entity)
                    original_indices_for_diagram.append(analysis_entry['original_idx'])
                    filtered_tokens_for_diagram.append(token_text_fb)
            else:
                logger.debug(f"  Fallback Token '{analysis_entry.get('text')}' (orig_idx {analysis_entry.get('original_idx')}) assigned None core type, excluding.")

        if not word_core_types_list: # Check the list that will be passed to diagram builders
            logger.error(f"Fallback Path: No valid tokens with assignable diagrammatic types (Word or Box) remained for diagram construction: '{sentence}'")
            return None, None, structure, tokens, analyses_details, roles
        
        logger.debug(f"Fallback Path Filtered Tokens with Diagrammatic Types: {[str(d) for d in word_core_types_list]}")
        
        try:
            logger.info(f"Creating DisCoCat diagram (Fallback V2.7.5) for structure: {structure}...")
            safe_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0]) if sentence else "empty"
            attempted_diagram_type = "Unknown"
            
            known_nominal_structures = [
                "NOMINAL_NOUN_SUBJ", "NOMINAL_NOUN_SUBJ_ADJ_PRED", "NOMINAL_NOUN_SUBJ_NOUN_PRED",
                "NOMINAL_ADJ_PREDICATE", "NOMINAL_ADJ_PRED_NO_SUBJ",
                "NOMINAL_X_PRED_WITH_SUBJ", "NOMINAL_RECLASSIFIED", "NOMINAL_LIKE",
                "NOMINAL_SUBJ_ONLY_RECLASSIFIED", "SUBJ_NO_VERB_OTHER", "NOMINAL"
            ]
            complex_nominal_structures = ["COMPLEX_" + s for s in known_nominal_structures]
            all_recognized_nominal_structures = known_nominal_structures + complex_nominal_structures

            if structure in all_recognized_nominal_structures:
                logger.info(f"Attempting NOMINAL diagram creation for structure '{structure}'.")
                attempted_diagram_type = "Nominal"
                diagram = create_nominal_sentence_diagram_v2_7(
                    tokens=filtered_tokens_for_diagram, 
                    analyses_details=analyses_details, 
                    roles=roles,
                    word_core_types=word_core_types_list, # Pass the list of Word/Box
                    original_indices=original_indices_for_diagram, 
                    debug=debug, 
                    output_dir=output_dir, 
                    sentence_prefix=f"sent_{safe_prefix}_nominal",
                    hint_predicate_original_idx=roles.get('predicate_idx')
                )
            elif structure.startswith("VERBAL_") or structure in ["SVO", "VSO", "SV", "VS", "VO_NO_SUBJ"] or \
                 (structure.startswith("COMPLEX_") and ("VERBAL" in structure or "SVO" in structure or "VSO" in structure)) or \
                 (roles.get('verb') is not None and structure not in ["ERROR_ANALYSIS_EXCEPTION", "ERROR_STANZA_INIT", "EMPTY_INPUT", "NO_SENTENCES_STANZA", "OTHER_ROOT_X", "OTHER_UNCLASSIFIED", "OTHER_STRUCTURE", "OTHER_DEFAULT"]):
                logger.info(f"Attempting VERBAL diagram creation for structure '{structure}'.")
                attempted_diagram_type = "Verbal"
                diagram = create_verbal_sentence_diagram_v3_7(
                    tokens=filtered_tokens_for_diagram, 
                    analyses_details=analyses_details, 
                    roles=roles,
                    word_core_types=word_core_types_list, # Pass the list of Word/Box
                    original_indices=original_indices_for_diagram, 
                    debug=debug, 
                    output_dir=output_dir, 
                    sentence_prefix=f"sent_{safe_prefix}_verbal"
                )
            else: 
                logger.warning(f"Structure is '{structure}'. Attempting robust fallback strategy for OTHER-like structures.")
                verb_idx_roles = roles.get('verb')
                has_assigned_verb_functor = verb_idx_roles is not None and \
                                            isinstance(core_type_map_for_fallback.get(verb_idx_roles), Box) and \
                                            core_type_map_for_fallback[verb_idx_roles].name in ["VerbIntransFunctor", "VerbTransFunctor"] # type: ignore
                
                if verb_idx_roles is not None and has_assigned_verb_functor :
                    logger.info(f"  '{structure}': Verb role and functor found. Attempting VERBAL diagram.")
                    attempted_diagram_type = "Verbal (OTHER - Verb Role & Functor Fallback)"
                    diagram = create_verbal_sentence_diagram_v3_7(
                        tokens=filtered_tokens_for_diagram, analyses_details=analyses_details, roles=roles,
                        word_core_types=word_core_types_list, #core_type_map=core_type_map_for_fallback, 
                        original_indices=original_indices_for_diagram, debug=debug,
                        output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_other_v_fallback"
                    )
                
                if diagram is None: # If verbal failed or not applicable
                    pred_idx_roles = roles.get('predicate_idx')
                    has_assigned_pred_functor = pred_idx_roles is not None and \
                                                isinstance(core_type_map_for_fallback.get(pred_idx_roles), Box) and \
                                                (core_type_map_for_fallback[pred_idx_roles].name == "AdjPredFunctor" or \
                                                 (hasattr(core_type_map_for_fallback[pred_idx_roles], 'name') and \
                                                  core_type_map_for_fallback[pred_idx_roles].name.startswith("NounPred_"))) # type: ignore
                    
                    if pred_idx_roles is not None and has_assigned_pred_functor:
                        logger.info(f"  '{structure}': Predicate role and functor found. Attempting NOMINAL diagram.")
                        attempted_diagram_type = "Nominal (OTHER - Predicate Role & Functor Fallback)"
                        diagram = create_nominal_sentence_diagram_v2_7(
                             tokens=filtered_tokens_for_diagram, analyses_details=analyses_details, roles=roles,
                             word_core_types=word_core_types_list, #core_type_map=core_type_map_for_fallback, 
                             original_indices=original_indices_for_diagram, debug=debug,
                             output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_nominal_other_p_fallback",
                             hint_predicate_original_idx=pred_idx_roles
                        )
                
                if diagram is None and structure in ["OTHER_ROOT_X", "OTHER_UNCLASSIFIED", "OTHER_STRUCTURE", "OTHER_DEFAULT"]:
                    root_idx = roles.get('root')
                    analysis_map_from_roles = roles.get('analysis_map_for_diagram_creation', {})
                    if root_idx is not None and root_idx in core_type_map_for_fallback:
                        root_analysis = analysis_map_from_roles.get(root_idx)
                        current_subject_idx = roles.get('subject')
                        if current_subject_idx is None and root_analysis:
                            dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                            for dep_idx, d_rel in dependents_of_root:
                                dep_analysis = analysis_map_from_roles.get(dep_idx)
                                if d_rel == 'nsubj' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                    roles['subject'] = dep_idx; current_subject_idx = dep_idx; break
                        
                        if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["NOUN", "ADJ", "X", "PROPN", "NUM"]:
                            logger.info(f"  '{structure}': Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}).")
                            temp_diagrammatic_types = list(word_core_types_list)
                            original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                            list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                            if list_idx_of_root is not None and list_idx_of_root < len(temp_diagrammatic_types):
                                new_functor_name_base = root_analysis.get('lemma', root_analysis.get('text','unk'))
                                if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE and isinstance(ADJ_PRED_TYPE, Box): # type: ignore
                                    temp_diagrammatic_types[list_idx_of_root] = Box(f"DynamicAdjPred_{new_functor_name_base}_{root_idx}", N, S, data={'original_idx': root_idx}) # type: ignore
                                else:
                                    temp_diagrammatic_types[list_idx_of_root] = Box(f"DynamicNounPred_{new_functor_name_base}_{root_idx}", N, S, data={'original_idx': root_idx}) # type: ignore
                                roles['predicate_idx'] = root_idx 
                                logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate '{get_diagram_repr(temp_diagrammatic_types[list_idx_of_root])}' (orig_idx {root_idx}) and subject idx {current_subject_idx}.")
                                attempted_diagram_type = "Nominal (OTHER - Dynamic Root Predicate)"
                                diagram = create_nominal_sentence_diagram_v2_7(
                                    tokens=filtered_tokens_for_diagram, analyses_details=analyses_details, roles=roles,
                                    word_core_types=temp_diagrammatic_types, 
                                    #core_type_map=core_type_map_for_fallback,
                                    original_indices=original_indices_for_diagram, debug=debug,
                                    output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_nominal_dyn_pred",
                                    hint_predicate_original_idx=root_idx
                                )
                
                if diagram is None: # Absolute last resort
                    logger.warning(f"  '{structure}': All specific diagram strategies failed. Attempting simple tensor product.")
                    if word_core_types_list:
                        temp_diag = word_core_types_list[0]
                        for next_diag_piece in word_core_types_list[1:]:
                            temp_diag = temp_diag @ next_diag_piece # type: ignore
                        diagram = temp_diag.normal_form() if hasattr(temp_diag, 'normal_form') else temp_diag # type: ignore
                        attempted_diagram_type = "Tensor Product (OTHER - Last Resort Fallback)"
                        logger.info(f"    '{structure}': Created last-resort tensor product diagram. Codomain: {diagram.cod if diagram and hasattr(diagram, 'cod') else 'None'}") # type: ignore
                    else:
                        logger.error(f"  '{structure}': No diagrammatic core types available even for tensor product fallback.")
                        attempted_diagram_type = "Skipped (OTHER - No types for tensor)"

            if diagram is None:
                if attempted_diagram_type and not attempted_diagram_type.startswith("Skipped"):
                    logger.error(f"Diagram creation ({attempted_diagram_type}) returned None for sentence '{sentence}'.")
            else:
                logger.info(f"Diagram ({attempted_diagram_type}) created successfully for '{sentence}'. Final Cod: {diagram.cod if hasattr(diagram, 'cod') else 'N/A'}") # type: ignore

        except Exception as e_diagram:
            logger.error(f"Exception during fallback diagram creation phase for '{sentence}': {e_diagram}", exc_info=True)
            diagram_creation_error = str(e_diagram)

    # --- Circuit Conversion ---
    circuit: Optional[QuantumCircuit] = None
    if diagram is None:
        logger.error("Diagram is None after all attempts, cannot proceed to circuit conversion.")
        return None, None, structure, tokens, analyses_details, roles

    try:
        logger.info(f"Converting diagram to quantum circuit using ansatz: {ansatz_choice}")
        ob_map = {N: 1, S: 1}
        selected_ansatz = None
        if handle_lexical_ambiguity_in_typing:
        # Use ControlledSenseIQPAnsatz if ambiguity was handled at type assignment
            from common_qnlp_types import ControlledSenseFunctor # ensure import
            logger.info(f"Using ControlledSenseIQPAnsatz for sentence '{sentence}' due to ambiguity handling flag.")
            # Use n_single_qubit_params_iqp for ControlledSenseIQPAnsatz if that's the intended config
            selected_ansatz = ControlledSenseFunctor(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
        if ansatz_choice.upper() == "IQP":
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
        elif ansatz_choice.upper() == "STRONGLY_ENTANGLING":
            selected_ansatz = StronglyEntanglingAnsatz(ob_map=ob_map, n_layers=n_layers_strong, ranges=cnot_ranges)
        elif ansatz_choice.upper() == "SPIDER":
            selected_ansatz = SpiderAnsatz(ob_map=ob_map)
        else:
            logger.warning(f"Unknown ansatz_choice: '{ansatz_choice}'. Defaulting to IQPAnsatz.")
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=2)

        if selected_ansatz is None: raise ValueError("Ansatz object could not be created.")
        
        quantum_diagram = selected_ansatz(diagram)
        logger.info(f"Applied {ansatz_choice} ansatz to the diagram.")

        if PYTKET_QISKIT_AVAILABLE and hasattr(quantum_diagram, 'to_tk'):
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
            circuit = None
            
    except Exception as e_circuit_outer:
        logger.error(f"Exception during circuit conversion: {e_circuit_outer}", exc_info=True)
        return None, diagram, structure, tokens, analyses_details, roles

    if circuit is None:
        logger.error("Circuit conversion resulted in None.")
        return None, diagram, structure, tokens, analyses_details, roles

    logger.info(f"Successfully processed sentence '{sentence}' into circuit.")
    return circuit, diagram, structure, tokens, analyses_details, roles, word_core_types_list


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