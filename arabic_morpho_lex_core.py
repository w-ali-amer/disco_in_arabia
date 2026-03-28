# arabic_morpho_lex_core.py
# Core QNLP pipeline enhancements for Arabic.
# Focus: Representing words as Root + Transformation.
# TransformationBox.data is enriched with detailed morphological features.

# -*- coding: utf-8 -*-

# arabic_morpho_lex_core.py
import logging
logger = logging.getLogger(__name__)

# --- Import shared Lambeq types ---
try:
    from common_qnlp_types import (
        N_ARABIC, S_ARABIC, ROOT_TYPE_ARABIC, 
        LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY,
        AtomicType, Ty, Box as CommonBox, # Import Box as CommonBox
        GrammarDiagram as CommonGrammarDiagram,
        IQPAnsatz as CommonIQPAnsatz, 
        SpiderAnsatz as CommonSpiderAnsatz, 
        StronglyEntanglingAnsatz as CommonStronglyEntanglingAnsatz,
        Cup as CommonCup, Id as CommonId, Spider as CommonSpider, Swap as CommonSwap,
        Functor as CommonFunctor, Word as CommonWord
    )
    if not LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        logger.warning("arabic_morpho_lex_core: common_qnlp_types reported that Lambeq types were NOT initialized successfully. Expect issues.")
    else:
        logger.info("arabic_morpho_lex_core: Successfully imported N_ARABIC, S_ARABIC, ROOT_TYPE_ARABIC from common_qnlp_types.")
    
    N = N_ARABIC
    S = S_ARABIC
    ROOT_TYPE = ROOT_TYPE_ARABIC 
    
    # Make Lambeq classes available directly if needed
    AtomicType = AtomicType
    Ty = Ty
    Box = CommonBox
    GrammarDiagram = CommonGrammarDiagram
    IQPAnsatz = CommonIQPAnsatz
    SpiderAnsatz = CommonSpiderAnsatz
    StronglyEntanglingAnsatz = CommonStronglyEntanglingAnsatz
    Cup, Id, Spider, Swap = CommonCup, CommonId, CommonSpider, CommonSwap
    Functor = CommonFunctor
    Word = CommonWord


except ImportError as e_common_types_core:
    logger.critical(f"arabic_morpho_lex_core: CRITICAL - Failed to import from common_qnlp_types.py: {e_common_types_core}. This module will not function correctly.", exc_info=True)
    class FallbackTyPlaceholderCore:
        def __init__(self, name): self.name = name
        def __str__(self): return self.name
        def __rshift__(self, other): return self # Dummy for >>
        def __matmul__(self, other): return self # Dummy for @

    N = FallbackTyPlaceholderCore('n_core_dummy_NI') # type: ignore
    S = FallbackTyPlaceholderCore('s_core_dummy_NI') # type: ignore
    ROOT_TYPE = N # type: ignore
    # Dummies for other lambeq components if their direct import (below) also fails
    if 'AtomicType' not in globals():
        class AtomicType: # type: ignore
            pass 
    if 'Ty' not in globals():
        class Ty: # type: ignore
            def __init__(self, *args): pass
            def __rshift__(self, other): return self
            def __matmul__(self, other): return self
    if 'Box' not in globals():
        class Box: # type: ignore
            def __init__(self, name: str, dom: Any, cod: Any, data: Optional[Dict] = None):
                self.name=name
                self.dom=dom
                self.cod=cod
                self.data=data if data is not None else {}
            def __rshift__(self, other): return self # Dummy for >>
            def __matmul__(self, other): return self # Dummy for @
            def normal_form(self): return self

    if 'GrammarDiagram' not in globals():
        class GrammarDiagram: # type: ignore
            def __init__(self, dom: Any, cod: Any, boxes: List[Any], offsets: List[Any]):
                self.dom = dom
                self.cod = cod
                self.boxes = boxes
                self.offsets = offsets
            def __rshift__(self, other): return self # Dummy for >>
            def __matmul__(self, other): return self # Dummy for @
            def normal_form(self): return self
            @property
            def name(self): return "DummyDiagram"

    if 'Cup' not in globals(): Cup = Box
    if 'Id' not in globals(): Id = lambda x: Box("Id", x, x)
    if 'Spider' not in globals(): Spider = Box
    if 'Swap' not in globals(): Swap = Box
    if 'Functor' not in globals():
        class Functor:
            pass
    if 'Word' not in globals(): Word = Box


# Import other necessary non-lambeq libraries
import stanza
import numpy as np
import traceback # type: ignore
from qiskit import QuantumCircuit # type: ignore
from qiskit.circuit import Parameter # type: ignore
from typing import List, Dict, Tuple, Optional, Any, Set, Union, Sequence, Mapping, Callable
from collections import Counter, defaultdict
import os
import hashlib
import io
import base64
import re
import copy
import json

# Lambeq specific imports (if not covered by common_qnlp_types aliasing)
# These are here to ensure they are available if common_qnlp_types import had issues with them.
try:
    if 'BobcatParser' not in globals(): from lambeq import BobcatParser # type: ignore
    if 'RewriteRule' not in globals(): from lambeq.rewrite import RewriteRule, Rewriter # type: ignore
    if 'quantum' not in globals(): import lambeq.backend.quantum as quantum # type: ignore
except ImportError:
    logger.error("arabic_morpho_lex_core: Failed to import some additional lambeq components directly.")

try:
    from pytket import Circuit as TketCircuit # For type hinting if needed
    from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
    PYTKET_QISKIT_AVAILABLE = True
    logger.info("Pytket-Qiskit extension found in core module.")
except ImportError:
    logger.warning("Pytket-Qiskit extension not found in core module. Circuit conversion will rely on Lambeq's default.")
    PYTKET_QISKIT_AVAILABLE = False
    # Define dummy for type hinting if needed, though not strictly necessary for runtime
    class TketCircuit: pass # type: ignore
    def tk_to_qiskit(tk_circ): return None # type: ignore
    def qiskit_to_tk(qc_circ): return None # type: ignore
# --- CAMeL Tools, Stanza, etc. Initializations ---
CAMEL_ANALYZER = None
CAMEL_DISAMBIGUATOR_MLE = None
DEDIAC_AR_FUNC = None
SIMPLE_WORD_TOKENIZE_CAMEL = None
STANZA_PIPELINE = None

try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.word import simple_word_tokenize as camel_simple_tokenizer
    from camel_tools.utils.dediac import dediac_ar

    DEDIAC_AR_FUNC = dediac_ar
    SIMPLE_WORD_TOKENIZE_CAMEL = camel_simple_tokenizer
    # Initialize with a specific DB variant if needed, e.g., 'calima-msa-r13'
    # db = MorphologyDB.builtin_db('calima-msa-r13') 
    db = MorphologyDB.builtin_db() # Default
    CAMEL_ANALYZER = Analyzer(db)
    disamb_analyzer_for_mle = Analyzer(MorphologyDB.builtin_db()) 
    CAMEL_DISAMBIGUATOR_MLE = MLEDisambiguator.pretrained(analyzer=disamb_analyzer_for_mle)
    logger.info("CAMeL Tools components initialized in core module.")
except Exception as e: logger.warning(f"Error initializing CAMeL Tools in core module: {e}.")

try:
    STANZA_PIPELINE = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse,mwt', verbose=False, use_gpu=False, logging_level='WARN')
    logger.info("Stanza pipeline initialized in core module.")
except Exception as e: logger.error(f"Error initializing Stanza in core module: {e}", exc_info=True)

# --- Core Atomic Types (EXISTING ONLY) ---
# Roots are represented with the Noun type by default.

# --- Helper Functions ---
def shape_arabic_text_core(text: Optional[str]) -> str: return text if text is not None else ""
def sanitize_filename_core(filename: str) -> str: return re.sub(r'[^\w\-_\.]', '_', filename.strip())[:100]
def parse_feats_string_core(feats_str: Optional[str]) -> Dict[str, str]:
    if not feats_str: return {}
    try: return {k: v for k, v in (pair.split('=', 1) for pair in feats_str.split('|') if '=' in pair)}
    except: return {}

# ==================================
# Classical Feature Extraction for Surface Words
# ==================================
def extract_classical_surface_word_features(camel_analysis: Dict[str, Any], feature_dim: int = 16) -> np.ndarray:
    """
    Extracts classical features from a CAMeL Tools analysis dictionary for a surface word.
    Args:
        camel_analysis: A dictionary representing the morphological analysis of a word.
        feature_dim: The desired dimensionality of the output feature vector.
    Returns:
        A NumPy array representing the classical features.
    """
    vec = np.full(feature_dim, 0.1) # Initialize with a neutral/NA value
    if not camel_analysis:
        logger.debug("extract_classical_surface_word_features: No CAMeL analysis provided, returning neutral vector.")
        return vec

    pos_map = {'noun': 0.1, 'verb': 0.2, 'adj': 0.3, 'adv':0.35, 'adp': 0.4, 'pron': 0.5,
               'part': 0.6, 'det':0.7, 'propn':0.15, 'intj':0.75, 'num':0.8, 'punct':0.9, 'x':0.95, 'conj':0.65, 'sconj':0.66}
    vec[0] = pos_map.get(camel_analysis.get('pos', 'x'), 0.95)

    gen = camel_analysis.get('gen')
    if gen == 'm': vec[1] = 0.0
    elif gen == 'f': vec[1] = 1.0
    else: vec[1] = 0.5

    num = camel_analysis.get('num')
    if num == 's': vec[2] = 0.0
    elif num == 'd': vec[2] = 0.5
    elif num == 'p': vec[2] = 1.0
    else: vec[2] = 0.2 

    has_al_det = False
    for prc_key in ['prc0', 'prc1', 'prc2']:
        if camel_analysis.get(prc_key) == 'Al_det':
            has_al_det = True
            break
        if camel_analysis.get(prc_key) == 'det' and camel_analysis.get(f'{prc_key}_lex', '').lower() == 'al':
             has_al_det = True
             break
    vec[3] = 1.0 if has_al_det else 0.0

    verb_form = camel_analysis.get('form_verb', camel_analysis.get('form'))
    form_map = {'I': 0.1, 'II': 0.2, 'III': 0.3, 'IV': 0.4, 'V': 0.5, 'VI':0.6, 'VII':0.7, 'VIII':0.8, 'IX':0.85, 'X':0.9,
                'XI': 0.91, 'XII': 0.92, 'XIII': 0.93, 'XIV': 0.94, 'XV': 0.95, 'XVI': 0.96}
    if verb_form and verb_form in form_map:
        vec[4] = form_map[verb_form]
    else:
        vec[4] = 0.0

    cas = camel_analysis.get('cas')
    if cas == 'n': vec[5] = 0.2
    elif cas == 'a': vec[5] = 0.5
    elif cas == 'g': vec[5] = 0.8
    else: vec[5] = 0.0

    asp = camel_analysis.get('asp')
    if asp == 'p': vec[6] = 0.2
    elif asp == 'i': vec[6] = 0.5
    elif asp == 'c': vec[6] = 0.8
    else: vec[6] = 0.0

    vox = camel_analysis.get('vox')
    if vox == 'a': vec[7] = 0.3
    elif vox == 'p': vec[7] = 0.7
    else: vec[7] = 0.0

    mod = camel_analysis.get('mod')
    if mod == 'i': vec[8] = 0.2
    elif mod == 's': vec[8] = 0.5
    elif mod == 'j': vec[8] = 0.8
    else: vec[8] = 0.0

    per = camel_analysis.get('per')
    if per == '1': vec[9] = 0.2
    elif per == '2': vec[9] = 0.5
    elif per == '3': vec[9] = 0.8
    else: vec[9] = 0.0

    enc0 = camel_analysis.get('enc0')
    vec[10] = 1.0 if enc0 and enc0 != '0' and 'pron' in enc0 else 0.0

    stt = camel_analysis.get('stt')
    if stt == 'i': vec[11] = 0.2
    elif stt == 'd': vec[11] = 0.5
    elif stt == 'c': vec[11] = 0.8
    else: vec[11] = 0.0

    has_common_prefix = any(camel_analysis.get(prc, '0') != '0' and camel_analysis.get(f'{prc}_lex') in ['w', 'f', 'b', 'l', 'k']
                            for prc in ['prc0', 'prc1', 'prc2'])
    vec[12] = 1.0 if has_common_prefix else 0.0
    
    vec[13] = 0.1 

    word_len = len(camel_analysis.get('diac', camel_analysis.get('lex', '')))
    if word_len <= 3: vec[14] = 0.2
    elif word_len <= 6: vec[14] = 0.5
    else: vec[14] = 0.8
    
    vec[15] = 0.1

    if not np.all(np.isfinite(vec)):
         logger.warning(f"NaN or Inf detected in classical_features_of_surface_word for '{camel_analysis.get('lex', 'UNK')}'. Resetting to neutral.")
         vec = np.full(feature_dim, 0.1)
    return vec

# ==================================
# Linguistic Analysis (Focus on Words and their Roots for Transformation)
# ==================================
def analyze_sentence_for_root_transform(
    sentence_str: str,
    chosen_camel_analyses_per_word: Optional[List[Dict[str, Any]]] = None,
    classical_feature_dim: int = 16,
    debug: bool = False
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Analyzes a sentence using Stanza for basic structure and CAMeL Tools for rich
    morphological details of each word, focusing on root and transformation.
    Args:
        sentence_str: The input Arabic sentence string.
        chosen_camel_analyses_per_word: Optional list of pre-selected CAMeL analysis dicts.
        classical_feature_dim: The dimensionality for classical features of surface words.
        debug: Boolean flag for verbose logging.
    Returns:
        A tuple containing:
        - linguistic_stream_of_words: List of dictionaries detailing surface words.
        - structure_label: A string label for the overall sentence structure.
        - roles_and_global_info: Dictionary with sentence-level info (dependency graph, etc.).
    """
    if not STANZA_PIPELINE:
        logger.error("Stanza pipeline not initialized. Cannot analyze sentence.")
        return [], "ERROR", {"original_sentence": sentence_str, "error": "Stanza not initialized"}
    if not CAMEL_ANALYZER and not chosen_camel_analyses_per_word:
        logger.warning("CAMeL Analyzer not initialized and no pre-chosen analyses. Morphological details will be basic.")

    logger.debug(f"RootTransformAnalysis: Processing sentence '{sentence_str}'. Has chosen CAMeL: {chosen_camel_analyses_per_word is not None}")

    try:
        doc = STANZA_PIPELINE(sentence_str)
    except Exception as e_stanza:
        logger.error(f"Stanza processing failed for '{sentence_str}': {e_stanza}", exc_info=debug)
        return [], "ERROR", {"original_sentence": sentence_str, "error": f"Stanza failed: {e_stanza}"}

    if not doc.sentences:
        logger.warning(f"Stanza found no sentences in: '{sentence_str}'")
        return [], "OTHER", {"original_sentence": sentence_str, "notes": "No sentences by Stanza"}

    sent = doc.sentences[0]
    
    linguistic_stream_of_words: List[Dict[str, Any]] = []
    roles_and_global_info: Dict[str, Any] = {
        "original_sentence": sentence_str,
        "dependency_graph_stanza": defaultdict(list), # Adjacency list for dependency graph
        "root_stanza_idx": None, 
        "verb_stanza_idx": None, "subject_stanza_idx": None, "object_stanza_idx": None,
        "structure_label_stanza": "OTHER_STRUCTURE",
        "stanza_word_analyses_for_sentence": [],
        "stanza_token_to_word_map": {} # For MWT expansion if Stanza handles it
    }

    # Store Stanza word analyses (0-indexed)
    # Stanza words are 1-indexed for head, 0-indexed for id if not MWT
    for s_word_idx, s_word in enumerate(sent.words):
        # s_word.id is a tuple for MWTs (e.g. (6,7,'بعضها')), string for single tokens
        # We need a consistent 0-indexed ID for our internal use.
        # The s_word_idx (enumerate index) is the 0-indexed position in sent.words list.
        
        head_idx_stanza = s_word.head - 1 if s_word.head and s_word.head > 0 else -1 # 0-indexed head
        
        word_analysis_entry = {
            'text': s_word.text, 'lemma': s_word.lemma or s_word.text,
            'upos': s_word.upos, 'xpos': s_word.xpos,
            'head_stanza_idx': head_idx_stanza, 
            'deprel': s_word.deprel,
            'stanza_feats_dict': parse_feats_string_core(s_word.feats),
            'original_stanza_idx': s_word_idx, # This is our reliable 0-indexed ID for this word in the Stanza list
            'stanza_internal_id': s_word.id # Stanza's own ID, can be complex for MWTs
        }
        roles_and_global_info["stanza_word_analyses_for_sentence"].append(word_analysis_entry)

        if head_idx_stanza != -1:
            # Ensure head_idx_stanza is also a valid 0-indexed ID from the same list
            roles_and_global_info["dependency_graph_stanza"][head_idx_stanza].append(
                {'dependent_idx': s_word_idx, 'deprel': s_word.deprel}
            )
        elif roles_and_global_info["root_stanza_idx"] is None: # First word with no head is a candidate for root
             roles_and_global_info["root_stanza_idx"] = s_word_idx
    
    # If no explicit root (e.g. all words have heads, which is unusual), default to first word if any
    if roles_and_global_info["root_stanza_idx"] is None and roles_and_global_info["stanza_word_analyses_for_sentence"]:
        # Try to find a word with deprel 'root' if Stanza provides it explicitly
        for i, sw_info in enumerate(roles_and_global_info["stanza_word_analyses_for_sentence"]):
            if sw_info['deprel'] == 'root':
                roles_and_global_info["root_stanza_idx"] = i
                break
        if roles_and_global_info["root_stanza_idx"] is None: # Still not found
            logger.warning(f"No explicit 'root' deprel found by Stanza for '{sentence_str}'. Defaulting root_stanza_idx to 0.")
            roles_and_global_info["root_stanza_idx"] = 0


    for i, sw_info in enumerate(roles_and_global_info["stanza_word_analyses_for_sentence"]):
        if sw_info['upos'] == 'VERB' and roles_and_global_info['verb_stanza_idx'] is None:
            roles_and_global_info['verb_stanza_idx'] = i
        # More robust role finding might iterate through dependents of the verb
        if roles_and_global_info['verb_stanza_idx'] is not None:
            for dep_info in roles_and_global_info["dependency_graph_stanza"].get(roles_and_global_info['verb_stanza_idx'], []):
                dep_idx = dep_info['dependent_idx']
                deprel = dep_info['deprel']
                if deprel in ['nsubj', 'nsubj:pass'] and roles_and_global_info['subject_stanza_idx'] is None:
                    roles_and_global_info['subject_stanza_idx'] = dep_idx
                elif deprel == 'obj' and roles_and_global_info['object_stanza_idx'] is None:
                    roles_and_global_info['object_stanza_idx'] = dep_idx


    if roles_and_global_info['verb_stanza_idx'] is not None:
        roles_and_global_info['structure_label_stanza'] = "VERBAL_LIKE"
    elif roles_and_global_info['subject_stanza_idx'] is not None:
        roles_and_global_info['structure_label_stanza'] = "NOMINAL_LIKE"

    for s_word_idx, s_word_analysis_from_stanza in enumerate(roles_and_global_info["stanza_word_analyses_for_sentence"]):
        surface_word_text = s_word_analysis_from_stanza['text']
        
        camel_analysis_for_this_word: Dict[str, Any]
        if chosen_camel_analyses_per_word and s_word_idx < len(chosen_camel_analyses_per_word):
            camel_analysis_for_this_word = chosen_camel_analyses_per_word[s_word_idx]
            if debug: logger.debug(f"  Using pre-chosen CAMeL for '{surface_word_text}'")
        elif CAMEL_ANALYZER:
            camel_analyses_list = CAMEL_ANALYZER.analyze(surface_word_text)
            if camel_analyses_list:
                camel_analysis_for_this_word = camel_analyses_list[0] 
                if debug: logger.debug(f"  CAMeL (top) for '{surface_word_text}': POS={camel_analysis_for_this_word.get('pos')}, Root={camel_analysis_for_this_word.get('root')}")
            else:
                logger.warning(f"  CAMeL no analysis for '{surface_word_text}'. Fallback.")
                camel_analysis_for_this_word = {
                    'diac': surface_word_text, 'lex': s_word_analysis_from_stanza['lemma'],
                    'pos': s_word_analysis_from_stanza['upos'].lower(), 
                    'root': s_word_analysis_from_stanza['lemma'], 
                    'pattern': 'UNKNOWN_PATTERN', 'form': 'UNKNOWN_FORM',
                    'camel_analysis_failed': True
                }
        else: 
            logger.warning(f"  No CAMeL Analyzer for '{surface_word_text}'. Fallback.")
            camel_analysis_for_this_word = {
                'diac': surface_word_text, 'lex': s_word_analysis_from_stanza['lemma'],
                'pos': s_word_analysis_from_stanza['upos'].lower(),
                'root': s_word_analysis_from_stanza['lemma'],
                'pattern': 'NO_CAMEL_PATTERN', 'form': 'NO_CAMEL_FORM',
                'no_camel_analyzer': True
            }

        raw_camel_root = camel_analysis_for_this_word.get('root')
        stanza_lemma_as_concept = s_word_analysis_from_stanza.get('lemma', surface_word_text)
        if not stanza_lemma_as_concept or not stanza_lemma_as_concept.strip():
            stanza_lemma_as_concept = surface_word_text

        core_representation_payload: Union[str, Dict[str, str]]
        conceptual_root_for_display: str 

        if raw_camel_root and isinstance(raw_camel_root, str) and raw_camel_root.strip() and raw_camel_root != 'NOAN':
            core_representation_payload = raw_camel_root
            conceptual_root_for_display = raw_camel_root
        else:
            reason_for_missing = f"CAMeL root was '{raw_camel_root if raw_camel_root else 'None'}' or invalid ('NOAN')."
            core_representation_payload = {
                "status": "missing_camel_root", 
                "reason": reason_for_missing,
                "fallback_conceptual_root": stanza_lemma_as_concept 
            }
            conceptual_root_for_display = stanza_lemma_as_concept

        classical_feats_vec = extract_classical_surface_word_features(camel_analysis_for_this_word, classical_feature_dim)

        linguistic_stream_item = {
            'surface_text': surface_word_text, 
            'lemma_stanza': s_word_analysis_from_stanza.get('lemma', surface_word_text), 
            'upos_stanza': s_word_analysis_from_stanza.get('upos'),   
            'deprel_stanza': s_word_analysis_from_stanza.get('deprel'),
            'head_idx_stanza': s_word_analysis_from_stanza.get('head_stanza_idx'), 
            'original_stanza_idx': s_word_idx, 
            'camel_analysis_of_surface_word': copy.deepcopy(camel_analysis_for_this_word),
            'extracted_root': conceptual_root_for_display, 
            'core_linguistic_representation': core_representation_payload,
            'classical_features_of_surface_word': classical_feats_vec,
        }
        linguistic_stream_of_words.append(linguistic_stream_item)
            
    return linguistic_stream_of_words, roles_and_global_info.get('structure_label_stanza', "OTHER_STRUCTURE"), roles_and_global_info

# ==================================
# Type Assignment & Diagram Creation (Root-Transformation Model)
# ==================================
def assign_types_for_root_transform(
    word_stream_item: Dict[str, Any]
) -> Tuple[Optional[Ty], Ty]:
    """
    Determines Lambeq type for RootWord and output type of TransformationBox.
    """
    # Ensure N and S are valid Ty objects, not placeholders if imports failed
    global N, S, ROOT_TYPE


    root_word_type = ROOT_TYPE # By default, roots are Nouns (or whatever ROOT_TYPE is defined as)
    
    surface_pos = word_stream_item.get('upos_stanza', 'X').upper()
    transformation_output_type: Ty

    if surface_pos in ['NOUN', 'PROPN', 'PRON', 'ADJ', 'NUM', 'DET', 'X', 'ADP', 'PART', 'PUNCT', 'SYM', 'INTJ', 'AUX']:
        transformation_output_type = N 
    elif surface_pos == 'VERB':
        transformation_output_type = S
    else: 
        logger.warning(f"Surface word '{word_stream_item.get('surface_text')}' UPOS '{surface_pos}' defaulting to N type for transformation output.")
        transformation_output_type = N
        
    return root_word_type, transformation_output_type


def create_diagram_for_word_as_root_transform(
    word_item_data: Dict[str, Any],
    root_type: Ty, 
    transformation_output_type: Ty 
) -> GrammarDiagram: 
    """
    Creates RootWord(Ty() -> root_type) >> TransformationBox(root_type -> transformation_output_type)
    """
    surface_text = word_item_data.get('surface_text', 'UNKSURFACE')
    raw_root_text = word_item_data.get('extracted_root')
    root_text = raw_root_text if isinstance(raw_root_text, str) and raw_root_text.strip() else surface_text
    
    original_stanza_idx = word_item_data.get('original_stanza_idx', 'X')
    sane_root_text = sanitize_filename_core(root_text)
    sane_surface_text = sanitize_filename_core(surface_text)

    root_word_name = f"RootConcept_{sane_root_text}_{original_stanza_idx}"
    # Ensure Box and Ty are callable
    if not callable(Box) or not callable(Ty):
        raise TypeError("Box or Ty is not callable. Lambeq types likely not initialized.")

    root_word_box_obj = Box(root_word_name, Ty(), root_type) 
    root_word_box_obj.data = {
        'role': 'root_concept', 'root_string': root_text,
        'surface_form_it_belongs_to': surface_text, 'original_stanza_idx': original_stanza_idx,
        'discocirc_enrichment_source': 'arabic_morpho_lex_core_root_word'
    }
    # A Box is a simple diagram.
    root_word_diagram = root_word_box_obj

    camel_analysis = word_item_data.get('camel_analysis_of_surface_word', {})
    affixes_info = {
        prc_key: camel_analysis.get(f"{prc_key}_diac", camel_analysis.get(prc_key))
        for prc_key in ['prc0', 'prc1', 'prc2', 'prc3'] if camel_analysis.get(prc_key) and camel_analysis.get(prc_key) != '0'
    }
    affixes_info.update({
        enc_key: camel_analysis.get(f"{enc_key}_diac", camel_analysis.get(enc_key))
        for enc_key in ['enc0', 'enc1'] if camel_analysis.get(enc_key) and camel_analysis.get(enc_key) != '0'
    })
    pattern_form_suffix = camel_analysis.get('pattern', camel_analysis.get('form', 'PATT_UNK'))
    transformation_box_name = f"MorphTransform_{sane_root_text}_to_{sane_surface_text}_as_{pattern_form_suffix}_{original_stanza_idx}"
    
    transformation_box_obj = Box(transformation_box_name, root_type, transformation_output_type)
    transformation_box_obj.data = {
        'role': 'morphological_transformation', 'surface_form': surface_text, 'root_form': root_text,
        'pos_of_surface': camel_analysis.get('pos'), 'pattern_camel': camel_analysis.get('pattern'),
        'form_camel': camel_analysis.get('form'), 'affixes_from_camel': affixes_info,
        'all_camel_features_of_surface': copy.deepcopy(camel_analysis),
        'classical_features_vector': copy.deepcopy(word_item_data.get('classical_features_of_surface_word')),
        'original_stanza_idx': original_stanza_idx,
        'discocirc_enrichment_source': 'arabic_morpho_lex_core_transform_box'
    }
    
    word_diagram = root_word_diagram >> transformation_box_obj
    logger.debug(f"  Created Root-Transform diagram for '{surface_text}': {word_diagram.name if hasattr(word_diagram, 'name') else 'Composite'} ({word_diagram.dom} -> {word_diagram.cod})")
    return word_diagram

# --- Syntactic Composition Helpers ---
def _create_applicator_box(name: str, arg_type: Ty, head_type: Ty, output_type: Ty, is_left_arg: bool = True) -> Box:
    """Creates a generic applicator box."""
    dom = arg_type @ head_type if is_left_arg else head_type @ arg_type
    box = Box(name, dom, output_type)
    box.data = {'rule_type': name, 'is_left_arg': is_left_arg}
    logger.debug(f"Created applicator box: {name} ({dom} -> {output_type})")
    return box

# Predefined applicator boxes (can be customized)
# These assume the "head" (verb/main noun) is on the right for subject, left for object/modifier
# This might need adjustment based on typical Arabic word order or dependency direction.
# For VSO: Verb (S-like) then Subject (N) then Object (N)
# ApplySubj: S @ N -> S (Verb receives Subject on its right)
# ApplyObj:  S @ N -> S (Verb-Subj complex receives Object on its right)
# ApplyMod:  N @ N -> N (Noun receives Adjective on its right) - or Adjective on left N @ N -> N
# Let's assume standard DisCoCat argument order for now: arguments on left/right of predicate.
# If word diagrams are Ty() -> Type, then these applicators act on the output types.

# Subject (N) on left of Predicate (S) -> S
APPLY_SUBJ_BOX = _create_applicator_box("ApplySubject", N, S, S, is_left_arg=True)
# Predicate (S) on left of Object (N) -> S
APPLY_OBJ_BOX = _create_applicator_box("ApplyObject", S, N, S, is_left_arg=True) # Object is right arg
# Modifier (N) on left of Noun (N) -> N
APPLY_ADJ_MOD_BOX = _create_applicator_box("ApplyAdjModifier", N, N, N, is_left_arg=True) 
# Preposition (N) on left of Noun (N) -> N (representing PP)
APPLY_PREP_MOD_BOX = _create_applicator_box("ApplyPrepModifier", N, N, N, is_left_arg=True)


def _recursive_compose_diagrams(
    node_idx: int,
    word_diagrams_map: Dict[int, GrammarDiagram],
    stanza_analyses: List[Dict[str, Any]], # List of Stanza word analyses
    dependency_graph: Dict[int, List[Dict[str, Any]]], # From roles_and_global_info
    composed_indices: Set[int] # To track which nodes have been incorporated
) -> Optional[GrammarDiagram]:
    """
    Recursively composes diagrams based on Stanza dependency tree.
    This is a simplified bottom-up like approach.
    """
    if node_idx in composed_indices: # Already part of a larger structure handled by its parent
        return None 
    
    current_word_diag = word_diagrams_map.get(node_idx)
    if not current_word_diag:
        logger.warning(f"No diagram found for node_idx {node_idx} in _recursive_compose_diagrams.")
        return None

    # Mark current node as being processed at this level
    composed_indices.add(node_idx) 
    
    # Get dependents of the current node
    dependents_info = sorted(dependency_graph.get(node_idx, []), key=lambda d: d['dependent_idx'])

    # Order of composition: typically modifiers first, then arguments.
    # For simplicity, let's process dependents from left to right (by original_stanza_idx).
    # More sophisticated ordering might be needed.

    composed_diag_for_node = current_word_diag
    node_upos = stanza_analyses[node_idx]['upos']

    for dep_info in dependents_info:
        dep_idx = dep_info['dependent_idx']
        deprel = dep_info['deprel']

        # Recursively compose for the dependent first (to get its full subtree diagram)
        dep_subtree_diag = _recursive_compose_diagrams(dep_idx, word_diagrams_map, stanza_analyses, dependency_graph, composed_indices)

        if not dep_subtree_diag:
            # logger.debug(f"Dependent {dep_idx} ({stanza_analyses[dep_idx]['text']}) already composed or no diagram. Skipping direct composition with head {node_idx}.")
            continue # This dependent was already handled or had no diagram

        # Now, compose dep_subtree_diag with composed_diag_for_node (which is current_word_diag initially, then grows)
        # The types of composed_diag_for_node and dep_subtree_diag must be N or S (output of Root>>Transform)
        
        # Determine applicator box based on deprel and POS of head/dependent
        # This is a very simplified logic.
        # Assumes head is `composed_diag_for_node`, dependent is `dep_subtree_diag`.
        # Word order (left/right application) is crucial here.
        # Stanza indices give original sentence order.
        
        # Default: tensor product if no specific rule applies
        applicator_to_use: Optional[Box] = None
        # Determine if dependent is to the left or right of the head for application
        # This is crucial for choosing the correct applicator box or swapping.
        # For now, we assume applicators handle standard (e.g. subject-left) DisCoCat order.
        # We might need Swaps if sentence order doesn't match applicator's expected order.
        
        # Let's assume dependent applies to head.
        # If dep_idx < node_idx, dependent is to the left.
        # If dep_idx > node_idx, dependent is to the right.

        # Simplified logic:
        # If head is VERB:
        if node_upos == 'VERB':
            if deprel in ['nsubj', 'nsubj:pass']: # Subject
                if composed_diag_for_node.cod == S and dep_subtree_diag.cod == N:
                    applicator_to_use = APPLY_SUBJ_BOX # N @ S -> S
                    # If subject (dep) is to the right of verb (head) in Arabic (e.g. VSO)
                    # and APPLY_SUBJ_BOX expects N (left) @ S (right), we might need a Swap.
                    # For now, assume APPLY_SUBJ_BOX is N @ S -> S.
                    # If dep_idx > node_idx (subj is right of verb): S @ N. Need Swap(S,N) >> N @ S >> APPLY_SUBJ_BOX
                    if dep_idx < node_idx: # Subj is left of Verb
                        composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node >> applicator_to_use
                    else: # Subj is right of Verb (V S O)
                         # We need S @ N. APPLY_SUBJ_BOX is N @ S.
                         # We'd need a box S @ N -> S or swap.
                         # For now, let's make a right-applying subject box for VSO
                        apply_subj_right_box = _create_applicator_box("ApplySubjRight", N, S, S, is_left_arg=False) # S @ N -> S
                        composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag >> apply_subj_right_box
                    logger.info(f"Applied Subject rule for head '{stanza_analyses[node_idx]['text']}' and dep '{stanza_analyses[dep_idx]['text']}'")

            elif deprel == 'obj': # Object
                if composed_diag_for_node.cod == S and dep_subtree_diag.cod == N:
                    applicator_to_use = APPLY_OBJ_BOX # S @ N -> S
                    # If obj (dep) is left of verb (head) - less common for Arabic main obj
                    if dep_idx < node_idx: # Obj is left of Verb-Subj complex
                        # Need N @ S. APPLY_OBJ_BOX is S @ N.
                        apply_obj_left_box = _create_applicator_box("ApplyObjLeft", N, S, S, is_left_arg=True) # N @ S -> S
                        composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node >> apply_obj_left_box
                    else: # Obj is right of Verb-Subj complex
                        composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag >> applicator_to_use
                    logger.info(f"Applied Object rule for head '{stanza_analyses[node_idx]['text']}' and dep '{stanza_analyses[dep_idx]['text']}'")
            
            elif deprel in ['advmod', 'obl']: # Adverbial/oblique argument or modifier
                 # Simplified: treat as N modifying S (e.g. S @ N -> S or N @ S -> S)
                if composed_diag_for_node.cod == S and dep_subtree_diag.cod == N:
                    adv_mod_box = _create_applicator_box("ApplyAdvModifierToVerb", S, N, S, is_left_arg=(dep_idx > node_idx)) # S @ N or N @ S
                    if dep_idx < node_idx: # Modifier left of verb
                         composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node >> adv_mod_box
                    else: # Modifier right of verb
                         composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag >> adv_mod_box
                    logger.info(f"Applied Adverbial/Oblique rule for head '{stanza_analyses[node_idx]['text']}' and dep '{stanza_analyses[dep_idx]['text']}'")


        # If head is NOUN:
        elif node_upos in ['NOUN', 'PROPN', 'PRON']:
            if deprel in ['amod', 'acl']: # Adjectival modifier or clausal modifier acting as adjective
                if composed_diag_for_node.cod == N and dep_subtree_diag.cod == N:
                    applicator_to_use = APPLY_ADJ_MOD_BOX # N @ N -> N
                    if dep_idx < node_idx: # Adj is left of Noun
                        composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node >> applicator_to_use
                    else: # Adj is right of Noun (common in Arabic)
                        apply_adj_mod_right_box = _create_applicator_box("ApplyAdjModRight", N, N, N, is_left_arg=False) # N @ N -> N (Head @ Mod)
                        composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag >> apply_adj_mod_right_box
                    logger.info(f"Applied Adjectival Modifier rule for head '{stanza_analyses[node_idx]['text']}' and dep '{stanza_analyses[dep_idx]['text']}'")

            elif deprel == 'nmod': # Nominal modifier (e.g., possession, prepositional phrase attached to noun)
                # This is complex. If it's a possessive (idaafa), it's N @ N -> N.
                # If it's a PP, the PP itself needs to be formed first.
                # For now, treat as N @ N -> N.
                if composed_diag_for_node.cod == N and dep_subtree_diag.cod == N: # Assuming PP also results in N-type diagram for now
                    nmod_box = _create_applicator_box("ApplyNMod", N, N, N, is_left_arg=(dep_idx < node_idx))
                    if dep_idx < node_idx:
                        composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node >> nmod_box
                    else:
                        composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag >> nmod_box
                    logger.info(f"Applied Nominal Modifier (nmod) rule for head '{stanza_analyses[node_idx]['text']}' and dep '{stanza_analyses[dep_idx]['text']}'")


        # If head is ADP (Preposition):
        elif node_upos == 'ADP':
            if deprel == 'obj' or deprel == 'case': # Object of preposition (or 'case' marker for some languages)
                # Preposition (N) + Noun (N) -> PP (represented as N for now)
                if composed_diag_for_node.cod == N and dep_subtree_diag.cod == N:
                    # Assume Prep is left of its object: Prep @ Noun_Obj
                    applicator_to_use = APPLY_PREP_MOD_BOX # N @ N -> N
                    if dep_idx > node_idx: # Object is to the right of preposition
                         composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag >> applicator_to_use
                    else: # Object is to the left (unusual for typical prepositions)
                         prep_obj_left_box = _create_applicator_box("ApplyPrepObjLeft", N, N, N, is_left_arg=True)
                         composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node >> prep_obj_left_box
                    logger.info(f"Applied Preposition-Object rule for head '{stanza_analyses[node_idx]['text']}' and dep '{stanza_analyses[dep_idx]['text']}'")


        if applicator_to_use is None and composed_diag_for_node and dep_subtree_diag : # Fallback if no specific rule
            logger.warning(f"No specific composition rule for head '{stanza_analyses[node_idx]['text']}' (UPOS: {node_upos}, cod: {composed_diag_for_node.cod}) "
                           f"and dependent '{stanza_analyses[dep_idx]['text']}' (deprel: {deprel}, cod: {dep_subtree_diag.cod}). Using tensor product as fallback.")
            try:
                if dep_idx < node_idx: # Dependent is to the left
                    composed_diag_for_node = dep_subtree_diag @ composed_diag_for_node
                else: # Dependent is to the right
                    composed_diag_for_node = composed_diag_for_node @ dep_subtree_diag
            except Exception as e_tensor_fallback:
                 logger.error(f"Error in fallback tensor product for {node_idx} and {dep_idx}: {e_tensor_fallback}")


    if composed_diag_for_node:
        try:
            composed_diag_for_node = composed_diag_for_node.normal_form()
        except Exception as e_norm:
            logger.warning(f"Could not normalize diagram for node {node_idx} ('{stanza_analyses[node_idx]['text']}'): {e_norm}")
            
    return composed_diag_for_node


def create_sentence_diagram_from_root_transforms(
    per_word_diagrams: List[Optional[GrammarDiagram]],
    word_stream_for_sentence: List[Dict[str, Any]], 
    roles_and_global_info: Dict[str, Any],
    use_bobcat_sentence_structure: bool = False # Currently not used by this strategy
) -> Optional[GrammarDiagram]:
    """
    Composes individual word diagrams (RootWord >> TransformationBox) into a sentence diagram
    using Stanza dependency parse to guide composition with applicator boxes.
    """
    valid_word_diagrams_with_indices = []
    for i, diag in enumerate(per_word_diagrams):
        if diag and i < len(word_stream_for_sentence): # Ensure alignment
            original_idx = word_stream_for_sentence[i].get('original_stanza_idx', -1)
            if original_idx != -1 : # Make sure we have a valid original_stanza_idx
                 valid_word_diagrams_with_indices.append({
                     'diagram': diag, 
                     'original_stanza_idx': original_idx, 
                     'stream_idx': i # Index in per_word_diagrams and word_stream_for_sentence
                })
            else:
                logger.warning(f"Word at stream_idx {i} ('{word_stream_for_sentence[i].get('surface_text')}') missing original_stanza_idx. Skipping.")
        elif not diag:
             logger.debug(f"No diagram for word at stream_idx {i} ('{word_stream_for_sentence[i].get('surface_text')}').")


    if not valid_word_diagrams_with_indices:
        logger.warning("No valid word diagrams with original_stanza_idx to compose for the sentence.")
        return None

    # Create a map from original_stanza_idx to the diagram object
    word_diagrams_map: Dict[int, GrammarDiagram] = {
        item['original_stanza_idx']: item['diagram'] for item in valid_word_diagrams_with_indices
    }
    
    # Get Stanza analyses (contains UPOS, text, etc. needed by recursive composer)
    stanza_analyses = roles_and_global_info.get("stanza_word_analyses_for_sentence")
    if not stanza_analyses:
        logger.error("Missing 'stanza_word_analyses_for_sentence' in roles_and_global_info. Cannot compose.")
        return None

    # Get the dependency graph and the root of the sentence from Stanza's analysis
    dependency_graph = roles_and_global_info.get("dependency_graph_stanza")
    sentence_root_idx = roles_and_global_info.get("root_stanza_idx")

    if dependency_graph is None or sentence_root_idx is None:
        logger.error("Dependency graph or sentence root index not found in roles_and_global_info. Falling back to tensor product.")
        # Fallback to simple tensor product if essential info is missing
        sorted_diagrams = [item['diagram'] for item in sorted(valid_word_diagrams_with_indices, key=lambda x: x['original_stanza_idx'])]
        if not sorted_diagrams: return None
        final_diag = sorted_diagrams[0]
        for i in range(1, len(sorted_diagrams)):
            final_diag = final_diag @ sorted_diagrams[i]
        return final_diag.normal_form() if final_diag else None

    logger.info(f"Starting syntactic composition. Sentence root index: {sentence_root_idx} ('{stanza_analyses[sentence_root_idx]['text'] if sentence_root_idx < len(stanza_analyses) else 'OOB'}').")
    
    composed_indices: Set[int] = set() # To track nodes already incorporated
    final_sentence_diagram = _recursive_compose_diagrams(
        sentence_root_idx,
        word_diagrams_map,
        stanza_analyses,
        dependency_graph,
        composed_indices
    )

    if final_sentence_diagram:
        logger.info(f"Syntactic composition successful. Final diagram codomain: {final_sentence_diagram.cod}")
        # Check if all words were incorporated. If not, it might indicate issues in recursion or graph traversal.
        if len(composed_indices) != len(word_diagrams_map):
            logger.warning(f"Not all word diagrams were composed into the final structure. Composed: {len(composed_indices)}, Available: {len(word_diagrams_map)}")
            uncomposed_words = []
            for idx, data in enumerate(stanza_analyses):
                if idx not in composed_indices and idx in word_diagrams_map:
                    uncomposed_words.append(f"'{data['text']}' (idx {idx})")
            if uncomposed_words:
                 logger.warning(f"Uncomposed words: {', '.join(uncomposed_words)}")
            # As a fallback, tensor the remaining diagrams with the main composed structure.
            # This is a simple way to ensure all semantic content is present, though not perfectly structured.
            # This part can be complex to implement robustly. For now, we'll just log.

        return final_sentence_diagram.normal_form() if hasattr(final_sentence_diagram, 'normal_form') else final_sentence_diagram
    else:
        logger.error("Syntactic composition failed to produce a final diagram. Check logs for issues in _recursive_compose_diagrams.")
        # Fallback to simple tensor product if syntactic composition fails entirely
        logger.warning("Falling back to simple tensor product due to syntactic composition failure.")
        sorted_diagrams = [item['diagram'] for item in sorted(valid_word_diagrams_with_indices, key=lambda x: x['original_stanza_idx'])]
        if not sorted_diagrams: return None
        final_diag = sorted_diagrams[0]
        for i in range(1, len(sorted_diagrams)):
            final_diag = final_diag @ sorted_diagrams[i]
        return final_diag.normal_form() if final_diag else None


# ==================================
# Main Orchestration Function for a Single Sentence
# ==================================
def process_sentence_for_qnlp_core(
    sentence_str: str,
    ansatz_functor: Any, 
    use_bobcat_for_sentence_structure: bool = False, 
    max_senses: int = 1, 
    classical_feature_dim_for_surface_words: int = 16,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Processes a single Arabic sentence through the Root-Transformation model.
    Generates linguistic analysis, DisCoCat diagram, and quantum circuit for each sense.
    """
    all_sense_results_for_sentence: List[Dict[str, Any]] = []
    logger.info(f"CORE_PROCESS: Starting for sentence: '{sentence_str}'")

    base_word_stream, base_structure_label, base_roles_dict = \
        analyze_sentence_for_root_transform(sentence_str,
                                            chosen_camel_analyses_per_word=None,
                                            classical_feature_dim=classical_feature_dim_for_surface_words,
                                            debug=debug)

    if not base_word_stream:
        logger.error(f"CORE_PROCESS: Baseline linguistic (root-transform) analysis FAILED for: '{sentence_str}'.")
        return [{
            'error': "Baseline linguistic (root-transform) analysis failed (empty base_word_stream)",
            'sentence_text': sentence_str, 'sense_id': 'baseline_error', 'sense_score': 0.0,
            'linguistic_stream_of_words': [] 
        }]

    logger.debug(f"CORE_PROCESS: Baseline analysis for '{sentence_str}' yielded {len(base_word_stream)} words. Structure: {base_structure_label}")

    sense_variant_definitions: List[Tuple[str, Optional[List[Dict[str,Any]]], float]] = []
    baseline_camel_analyses_from_stream = [item.get('camel_analysis_of_surface_word', {}) for item in base_word_stream]
    sense_variant_definitions.append(("baseline_top_senses", baseline_camel_analyses_from_stream, 1.0))

    # Placeholder for disambiguation logic if CAMEL_DISAMBIGUATOR_MLE is used
    if max_senses > 1 and CAMEL_DISAMBIGUATOR_MLE and base_roles_dict.get("stanza_word_analyses_for_sentence"):
        try:
            original_words = [sw_info['text'] for sw_info in base_roles_dict["stanza_word_analyses_for_sentence"]]
            disambiguated_options = CAMEL_DISAMBIGUATOR_MLE.disambiguate(original_words)
            
            # Process disambiguated_options to fit into sense_variant_definitions
            # This is a simplified example; you'll need to map scores and select top N senses
            for i, disamb_sent in enumerate(disambiguated_options[:max_senses -1]): # -1 because baseline is one sense
                sense_id = f"mle_sense_{i+1}"
                # Each disamb_word in disamb_sent.disambiguated_words is a CAMeL analysis dict
                chosen_analyses = [dw.analysis for dw in disamb_sent.disambiguated_words]
                score = disamb_sent.score # Assuming score is available
                sense_variant_definitions.append((sense_id, chosen_analyses, score))
                if len(sense_variant_definitions) >= max_senses: break
            logger.info(f"Generated {len(disambiguated_options)} disambiguation options via CAMeL MLE.")
        except Exception as e_disamb:
            logger.error(f"Error during CAMeL MLE disambiguation: {e_disamb}", exc_info=debug)


    for sense_id, chosen_analyses_for_variant, sense_score in sense_variant_definitions:
        logger.info(f"  CORE_PROCESS: Processing Sense Variant: ID='{sense_id}', Score={sense_score:.4f} for '{sentence_str}'")
        
        current_linguistic_stream_for_this_sense: Optional[List[Dict[str, Any]]] = None

        current_word_stream, current_structure_label, current_roles_dict = \
            analyze_sentence_for_root_transform(sentence_str,
                                                chosen_camel_analyses_per_word=chosen_analyses_for_variant,
                                                classical_feature_dim=classical_feature_dim_for_surface_words,
                                                debug=debug)
        
        current_linguistic_stream_for_this_sense = current_word_stream 

        if not current_word_stream:
            logger.error(f"    CORE_PROCESS: Linguistic analysis FAILED for sense '{sense_id}' of '{sentence_str}'.")
            all_sense_results_for_sentence.append({
                'error': f"Linguistic analysis for sense {sense_id} failed",
                'sentence_text': sentence_str, 'sense_id': sense_id, 'sense_score': sense_score,
                'linguistic_stream_of_words': [] 
            })
            continue
        
        logger.debug(f"    CORE_PROCESS: Sense '{sense_id}' - Linguistic stream has {len(current_word_stream)} words.")

        per_word_diagrams_for_sentence: List[Optional[GrammarDiagram]] = []
        valid_diag_count = 0
        temp_word_stream_with_diagrams = [] # To store stream items updated with their diagrams

        for word_idx, word_item_data in enumerate(current_word_stream):
            updated_word_item_data = word_item_data.copy() # Operate on a copy
            try:
                root_type_for_word, transform_output_type_for_word = assign_types_for_root_transform(updated_word_item_data)
                word_diag = create_diagram_for_word_as_root_transform(updated_word_item_data, root_type_for_word, transform_output_type_for_word)
                if word_diag:
                    updated_word_item_data['word_diagram_from_core'] = word_diag # Store the diagram in the stream item
                    valid_diag_count += 1
                else: # Should not happen if create_diagram_for_word_as_root_transform guarantees a diagram
                    logger.warning(f"    CORE_PROCESS: Diagram creation returned None for word '{updated_word_item_data.get('surface_text')}' (idx {word_idx}) in sense '{sense_id}'.")
                    updated_word_item_data['word_diagram_from_core'] = None
                per_word_diagrams_for_sentence.append(word_diag) # Append diagram (or None) to list for sentence composition
            except Exception as e_word_diag_creation:
                logger.error(f"    CORE_PROCESS: EXCEPTION during diagram creation for word '{updated_word_item_data.get('surface_text')}': {e_word_diag_creation}", exc_info=debug)
                updated_word_item_data['word_diagram_from_core'] = None
                per_word_diagrams_for_sentence.append(None)

            temp_word_stream_with_diagrams.append(updated_word_item_data) # Add to the potentially updated stream
        
        current_linguistic_stream_for_this_sense = temp_word_stream_with_diagrams # This is the stream to save
        
        if valid_diag_count == 0:
            logger.error(f"    CORE_PROCESS: No valid word diagrams created for any word in sense '{sense_id}' of '{sentence_str}'.")
            all_sense_results_for_sentence.append({
                'error': "No valid word diagrams created for this sense", 'sense_id': sense_id,
                'sentence_text': sentence_str,
                'linguistic_stream_of_words': current_linguistic_stream_for_this_sense, 
                'sense_score': sense_score
            })
            continue

        # Pass the full current_roles_dict which contains the dependency graph
        final_sentence_diagram = create_sentence_diagram_from_root_transforms(
            per_word_diagrams_for_sentence, 
            current_linguistic_stream_for_this_sense, # This stream has original_stanza_idx
            current_roles_dict, # This has dependency_graph_stanza, root_stanza_idx, stanza_word_analyses
            use_bobcat_sentence_structure=use_bobcat_for_sentence_structure # Flag might be repurposed
        )
        
        if not final_sentence_diagram:
            logger.error(f"    CORE_PROCESS: Sentence diagram composition FAILED for sense '{sense_id}' of '{sentence_str}'.")
            all_sense_results_for_sentence.append({
                'error': "Sentence diagram composition failed", 'sense_id': sense_id,
                'sentence_text': sentence_str,
                'linguistic_stream_of_words': current_linguistic_stream_for_this_sense, 
                'sense_score': sense_score,
                'per_word_diagrams_count': valid_diag_count
            })
            continue
        logger.info(f"    CORE_PROCESS: Sense '{sense_id}' - Sentence diagram composed. Codomain: {final_sentence_diagram.cod}")
            
        circuit = None
        try:
            if not callable(ansatz_functor) and not isinstance(ansatz_functor, Functor): # Functor is also callable
                raise ValueError("Provided ansatz_functor is not callable (e.g., an initialized ansatz object or Functor).")
            
            quantum_sentence_diagram = ansatz_functor(final_sentence_diagram)
            logger.debug(f"    CORE_PROCESS: Sense '{sense_id}' - Applied ansatz. Quantum diagram type: {type(quantum_sentence_diagram)}")

            if PYTKET_QISKIT_AVAILABLE and isinstance(quantum_sentence_diagram, GrammarDiagram) and hasattr(quantum_sentence_diagram, 'to_tk'):
                logger.debug(f"      CORE_PROCESS: Attempting circuit conversion via Pytket for sense '{sense_id}'...")
                tket_circ = quantum_sentence_diagram.to_tk()
                if tket_circ:
                    circuit = tk_to_qiskit(tket_circ) 
                    logger.info(f"      CORE_PROCESS: Pytket conversion successful for sense '{sense_id}'.")
                else:
                    logger.warning(f"      CORE_PROCESS: .to_tk() returned None for sense '{sense_id}'.")
            
            if circuit is None and hasattr(quantum_sentence_diagram, 'to_qiskit'):
                logger.debug(f"      CORE_PROCESS: Attempting direct .to_qiskit() for sense '{sense_id}'...")
                circuit_candidate = quantum_sentence_diagram.to_qiskit()
                if isinstance(circuit_candidate, QuantumCircuit): 
                    circuit = circuit_candidate
                    logger.info(f"      CORE_PROCESS: Direct .to_qiskit() successful for sense '{sense_id}'.")
                else:
                    logger.warning(f"      CORE_PROCESS: .to_qiskit() did not return Qiskit Circuit for sense '{sense_id}' (type: {type(circuit_candidate)}).")

            if circuit is None: 
                raise ValueError("Circuit conversion to Qiskit format returned None or failed.")
                
        except Exception as e_circ:
            logger.error(f"    CORE_PROCESS: Circuit conversion FAILED for sense '{sense_id}' of '{sentence_str}': {e_circ}", exc_info=debug)
            all_sense_results_for_sentence.append({
                'error': f"Circuit conversion failed: {str(e_circ)}", 'sense_id': sense_id,
                'diagram_str': str(final_sentence_diagram), # Save diagram string if circuit fails
                'sentence_text': sentence_str,
                'linguistic_stream_of_words': current_linguistic_stream_for_this_sense, 
                'sense_score': sense_score
            })
            continue
            
        all_sense_results_for_sentence.append({
            'sentence_text': sentence_str,
            'sense_id': sense_id,
            'sense_score': sense_score,
            'diagram': final_sentence_diagram, # Store the composed diagram object
            'circuit': circuit, # Store the Qiskit circuit object
            'linguistic_stream_of_words': current_linguistic_stream_for_this_sense, 
            'roles_for_sense': current_roles_dict,
            'structure_for_sense': current_structure_label,
            'error': None
        })
        logger.info(f"  CORE_PROCESS: Successfully processed sense '{sense_id}' for '{sentence_str}'.")

    if not all_sense_results_for_sentence:
        logger.error(f"CORE_PROCESS: No sense results generated for sentence '{sentence_str}' after all attempts.")
        return [{
            'error': "No sense results generated after all attempts", 'sentence_text': sentence_str,
            'sense_id': 'no_results_final', 'sense_score': 0.0,
            'linguistic_stream_of_words': base_word_stream 
        }]
        
    return all_sense_results_for_sentence
