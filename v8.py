# quantum_kernel_v8.py (Combined Clustering/Visualization & Classification Evaluation)
# -*- coding: utf-8 -*-
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
# Corrected Typing import for Mapping
from typing import List, Dict, Tuple, Optional, Any, Sequence, Set, Mapping, Union
import matplotlib.pyplot as plt
import pickle
import os
import traceback
import logging
from collections import Counter
import hashlib
import io
import base64
import re
  # Import re for sanitizing filenames

# --- Imports for Arabic Text Display ---
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_DISPLAY_ENABLED = True
except ImportError:
    print("Warning: 'arabic_reshaper' or 'python-bidi' not found.")
    ARABIC_DISPLAY_ENABLED = False

logger = logging.getLogger(__name__)
# Configure basic logging - Ensure this is configured in your main script (exp4.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.exceptions import QiskitError
    from qiskit_aer import AerSimulator
    from qiskit import qasm2 as Qasm
    from qiskit import qpy
    from qiskit.primitives import Sampler, Estimator
    from qiskit.quantum_info import SparsePauliOp, partial_trace
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit or Qiskit-Aer not found. Quantum execution will fail.")
    QISKIT_AVAILABLE = False # Ensure it's False if import fails
    class QuantumCircuit: pass
    class AerSimulator: pass
    class ClassicalRegister: pass
    class Parameter: pass
    class Sampler: pass
    class Estimator: pass
    class SparsePauliOp: pass
    partial_trace = None

# --- Qiskit Visualization ---
try:
    from qiskit.visualization import plot_state_city, plot_histogram
except ImportError:
    plot_state_city = None; plot_histogram = None

# --- Lambeq Imports ---
try:
    from common_qnlp_types import (
        N_ARABIC, S_ARABIC, ROOT_TYPE_ARABIC, 
        LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY,
        AtomicType, GrammarDiagram, Ty, Word, Box, 
        IQPAnsatz, SpiderAnsatz, StronglyEntanglingAnsatz , AmbiguousLexicalBox # Import for isinstance checks
    )
    if not LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        logger.warning("v8: common_qnlp_types reported that Lambeq types were NOT initialized successfully. Core enhancements might fail.")
    else:
        logger.info("v8: Successfully imported N_ARABIC, S_ARABIC, ROOT_TYPE_ARABIC from common_qnlp_types.")
    
    # These will be passed to ADP's ansatz_config
    N = N_ARABIC
    S = S_ARABIC
    ROOT_TYPE= ROOT_TYPE_ARABIC

except ImportError as e_common_types_v8:
    logger.critical(f"v8: CRITICAL - Failed to import from common_qnlp_types.py: {e_common_types_v8}. Core enhancements will likely fail.", exc_info=True)
    class FallbackTyPlaceholderV8:
        def __init__(self, name): self.name = name
        def __str__(self): return self.name
    N = FallbackTyPlaceholderV8('n_v8_dummy_NI') # type: ignore
    S = FallbackTyPlaceholderV8('s_v8_dummy_NI') # type: ignore
    ROOT_TYPE = N # type: ignore
    #if 'LambeqAtomicType' not in globals(): class LambeqAtomicType: pass # type: ignore

# --- Gensim Imports ---
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    logger.warning("Warning: gensim not found. Parameter binding via embeddings disabled.")
    GENSIM_AVAILABLE = False # Ensure it's False if import fails

# --- Scikit-learn Imports ---
SKLEARN_AVAILABLE = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC  # <-- Added SVC
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler # <-- Added StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise # Added for heatmap
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("scikit-learn is required for classification and PCA.")
    # Define dummy classes if sklearn not available to avoid NameErrors later
    class GaussianNB: pass
    class SVC: pass
    class StandardScaler: pass
    class PCA: pass
    def train_test_split(*args, **kwargs): return [], [], [], []
    def accuracy_score(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return "scikit-learn not available"
    def pairwise(*args, **kwargs): return None # Dummy for heatmap

CORE_MODULE_AVAILABLE = False
try:
    from arabic_morpho_lex_core import process_sentence_for_qnlp_core
    CORE_MODULE_AVAILABLE = True
    logger.info("v8: Successfully imported 'process_sentence_for_qnlp_core' from arabic_morpho_lex_core.")
except ImportError as e_core_module_v8:
    logger.warning(f"v8: WARNING: Cannot import from 'arabic_morpho_lex_core.py'. Error: {e_core_module_v8}")
    def process_sentence_for_qnlp_core(*args, **kwargs): # type: ignore
        logger.error("v8: Dummy process_sentence_for_qnlp_core called (import failed).")
        return [{'error': 'arabic_morpho_lex_core.py not available'}]

# --- Dependency on camel_test2 ---
# Use the latest version that includes the desired type assignment and feature boxes
try:
    from camel_test2 import arabic_to_quantum_enhanced_v2_7 as main_sentence_processor
    MAIN_PROCESSOR_AVAILABLE = True
    logger.info("Successfully imported 'arabic_to_quantum_enhanced_v2_7' as main_sentence_processor.")
except ImportError as e_camel_test2:
    logger.error(f"ERROR: Cannot import 'arabic_to_quantum_enhanced_v2_7' from 'camel_test2.py'. Error: {e_camel_test2}", exc_info=True)
    MAIN_PROCESSOR_AVAILABLE = False
    def main_sentence_processor(*args, **kwargs): # Dummy function if import fails
        logger.error("Dummy main_sentence_processor called (import failed).")
        return None, None, "ERROR_CAMEL_TEST2_IMPORT", [], [], {}
# --- Helper Functions ---
def shape_arabic_text(text):
    """Reshapes and applies bidi algorithm for correct Arabic display."""
    if not ARABIC_DISPLAY_ENABLED or not text or not isinstance(text, str): return text
    if any('\u0600' <= char <= '\u06FF' for char in text):
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            return get_display(reshaped_text)
        except Exception as e:
            logger.warning(f"Could not reshape/bidi Arabic text '{text[:20]}...': {e}")
            return text
    return text

def _ensure_circuit_name(circuit: Any, default_name: str = "qiskit_circuit") -> Any:
    """Assigns a default name to a Qiskit circuit if it doesn't have one."""
    if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
        if not getattr(circuit, 'name', None):
            try: setattr(circuit, 'name', default_name)
            except Exception: pass
    return circuit

# <<< MODIFICATION >>> Add sanitization functions here for broader use
def sanitize_filename_v8(filename: str) -> str:
    """Removes or replaces characters unsafe for filenames."""
    # Remove leading/trailing whitespace
    sanitized = filename.strip()
    # Replace potentially problematic characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\s+,;=\[\]]+', '_', sanitized)
    # Limit length (optional, but good practice)
    max_len = 100
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized

def mpl_sanitize(text: str) -> str:
    """Removes characters that might cause issues with Matplotlib text rendering."""
    # Basic sanitization: remove common problematic characters for titles/labels
    # This might need refinement based on specific errors encountered
    sanitized = text.replace('$', '').replace('{', '').replace('}', '').replace('_', ' ')
    return sanitized


# ============================================
#        Combined Kernel Class (v8)
# ============================================
class ArabicQuantumMeaningKernel:
    """
    Combines exploratory analysis (clustering/visualization on quantum embeddings)
    with quantitative evaluation (classification using quantum embeddings).
    Generates both quantum and combined (quantum + classical) embeddings.
    V8.
    """
    def __init__(self,
                 embedding_dim: int = 30,
                 num_clusters: int = 5,
                 embedding_model_path: Optional[str] = None,
                 ansatz_choice_for_main_processor: str = 'IQP', # Ansatz used by camel_test2
                 n_layers_iqp: int = 2,
                 n_layers_strong: int = 2,
                 output_dir_set: str = "qnlp_kernel_output_current_set",
                 embedding_type_for_classification: str = 'quantum',
                 params_per_word_fallback: int = 6, # Used by fallback _bind_parameters
                 shots: int = 8192,
                 morph_param_weight_fallback: float = 0.3,
                 config_n_layers_iqp: int = 2, 
                 config_n_single_qubit_params_iqp: int = 4, # Default if 2 layers
                 config_n_layers_strong: int = 1, # For fallback _bind_parameters
                 combine_feature_weight: float = 0.5, # Weight for combining quantum and classical features
                 classical_feature_dim_from_core: int = 16): # Expected dim from core module

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ArabicQuantumMeaningKernel.")
        self.ansatz_choice = ansatz_choice_for_main_processor
        self.config_n_layers_iqp = config_n_layers_iqp
        self.config_n_single_qubit_params_iqp = config_n_single_qubit_params_iqp
        self.config_n_layers_strong = config_n_layers_strong
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.params_per_word_fallback = params_per_word_fallback # How many params to try and bind per "concept" in fallback
        self._embedding_model_path = embedding_model_path
        self.shots = shots
        self.morph_param_weight_fallback = max(0.0, min(1.0, morph_param_weight_fallback))
        self.combine_feature_weight = max(0.0, min(1.0, combine_feature_weight))
        self.ansatz_choice_for_main_processor = ansatz_choice_for_main_processor # For record keeping
        self.n_layers_iqp = n_layers_iqp
        self.n_layers_strong = n_layers_strong
        self.output_dir_set = output_dir_set
        self.embedding_type_for_classification = embedding_type_for_classification
        self.classical_feature_dim_from_core = classical_feature_dim_from_core # Store this

        self.current_set_name = "UnknownSet"
        self.final_results_for_set: List[Dict[str, Any]] = [] # Stores results for the current set
        self.classification_results_for_set: Optional[Dict[str, Any]] = None

        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
            self.estimator = Estimator(options={'shots': self.shots})
        else:
            self.simulator = None; self.estimator = None

        self.kmeans_model = None; self.cluster_labels = None; self.meaning_clusters = None
        self.meaning_map = {}
        self.reference_sentences: Dict[int, str] = {} # original_idx -> sentence string
        self.circuit_embeddings: Dict[int, np.ndarray] = {}
        self.sentence_embeddings: Dict[int, np.ndarray] = {}
        self.linguistic_features: Dict[int, np.ndarray] = {} # Classical features
        # Store diagrams from main_processor (camel_test2) for parameter binding
        self.sentence_diagrams_main: Dict[int, Optional[GrammarDiagram]] = {}


        self.camel_analyzer = None # For fallback get_morph_vector
        if CORE_MODULE_AVAILABLE: # CAMeL tools are initialized within arabic_morpho_lex_core
            logger.info("CAMeL Tools expected to be initialized by arabic_morpho_lex_core.")
        else: # Fallback if core module not available
            try:
                from camel_tools.morphology.database import MorphologyDB
                from camel_tools.morphology.analyzer import Analyzer
                db_path = MorphologyDB.builtin_db()
                self.camel_analyzer = Analyzer(db_path)
                logger.info("Kernel initialized its own CAMeL Analyzer (core module not available).")
            except Exception as e_camel_kernel:
                logger.warning(f"Failed to initialize CAMeL Analyzer in Kernel: {e_camel_kernel}")


        self.embedding_model = None # For fallback parameter binding
        if GENSIM_AVAILABLE and embedding_model_path:
            logger.info(f"Attempting to load word embedding model from: {embedding_model_path}")
            if not os.path.exists(embedding_model_path): logger.error(f"Embedding file not found: {embedding_model_path}")
            else:
                try:
                    from gensim.models import Word2Vec
                    m = Word2Vec.load(embedding_model_path)
                    if hasattr(m, 'wv'): self.embedding_model = m.wv
                except:
                    try: self.embedding_model = KeyedVectors.load(embedding_model_path)
                    except Exception as e_gensim: logger.error(f"Failed to load embedding model: {e_gensim}")
        
        self.pca_per_category_path = os.path.join(self.output_dir_set, "pca_plots_per_category")
        os.makedirs(self.pca_per_category_path, exist_ok=True)

        # --- Semantic Templates (for reporting) ---
        self.semantic_templates = {
            'VSO': {'declarative': "ACTION performed by SUBJECT on OBJECT"},
            'SVO': {'declarative': "SUBJECT performs ACTION on OBJECT"},
            'NOMINAL': {'declarative': "SUBJECT is PREDICATE"},
            'OTHER': {'declarative': "General statement about TOPIC"},
            'VERBAL_OTHER': {'declarative': "Verbal action involving SUBJECT/OBJECT"},
            'COMPLEX_VSO': {'declarative': "Complex action (VSO base) involving SUBJECT/OBJECT"},
            'COMPLEX_SVO': {'declarative': "Complex action (SVO base) involving SUBJECT/OBJECT"},
            'COMPLEX_OTHER': {'declarative': "Complex statement about TOPIC"}
        }
        # --- Initialize CAMeL Tools ---
        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            db_path = MorphologyDB.builtin_db()
            self.camel_analyzer = Analyzer(db_path)
            logger.info("CAMeL Tools Analyzer initialized successfully.")
        except ImportError:
             logger.warning("CAMeL Tools not found (pip install camel-tools). Morphological analysis disabled.")
        except LookupError:
             logger.warning("CAMeL Tools default DB not found. Run 'camel_tools download default'. Morphological analysis disabled.")
        except Exception as e:
            logger.warning(f"Error initializing CAMeL Tools Analyzer: {e}. Morphological analysis disabled.")

    # --- get_morph_vector (Unchanged) ---
    def get_morph_vector(self, morph_features: Optional[Dict]) -> np.ndarray:
        """
        Converts CAMeL Tools morphological analysis dictionary into a richer
        fixed-size numerical vector (16 dimensions).
        """
        # Define the size and initialize the vector
        vec_size = 16
        vec = np.full(vec_size, 0.2) # Initialize with a neutral 'na' value (e.g., 0.2)

        if not morph_features:
            return vec # Return neutral vector if no features provided

        # --- Feature Mapping ---
        # Index 0: Gender (gen) - m=0, f=1, na=0.5
        gen = morph_features.get('gen')
        if gen == 'm': vec[0] = 0.0
        elif gen == 'f': vec[0] = 1.0
        else: vec[0] = 0.5

        # Index 1: Number (num) - s=0, d=0.5, p=1, na=0.2
        num = morph_features.get('num')
        if num == 's': vec[1] = 0.0
        elif num == 'd': vec[1] = 0.5
        elif num == 'p': vec[1] = 1.0
        # else: vec[1] remains 0.2 (na)

        # Index 2: Aspect (asp) - p=0, i=0.5, c=1, na=0.2
        asp = morph_features.get('asp')
        if asp == 'p': vec[2] = 0.0 # Perfective (Past)
        elif asp == 'i': vec[2] = 0.5 # Imperfective (Present/Future)
        elif asp == 'c': vec[2] = 1.0 # Command (Imperative)
        # else: vec[2] remains 0.2 (na)

        # Index 3: Case (cas) - n=0, a=0.5, g=1, na=0.2
        cas = morph_features.get('cas')
        if cas == 'n': vec[3] = 0.0 # Nominative
        elif cas == 'a': vec[3] = 0.5 # Accusative
        elif cas == 'g': vec[3] = 1.0 # Genitive
        # else: vec[3] remains 0.2 (na)

        # Index 4: State (stt) - i=0, d=0.5, c=1, na=0.2
        stt = morph_features.get('stt')
        if stt == 'i': vec[4] = 0.0 # Indefinite
        elif stt == 'd': vec[4] = 0.5 # Definite
        elif stt == 'c': vec[4] = 1.0 # Construct
        # else: vec[4] remains 0.2 (na)

        # Index 5: Voice (vox) - a=0, p=1, na=0.5
        vox = morph_features.get('vox')
        if vox == 'a': vec[5] = 0.0 # Active
        elif vox == 'p': vec[5] = 1.0 # Passive
        else: vec[5] = 0.5

        # Index 6: Mood (mod) - i=0, s=0.5, j=1, na=0.2
        mod = morph_features.get('mod')
        if mod == 'i': vec[6] = 0.0 # Indicative
        elif mod == 's': vec[6] = 0.5 # Subjunctive
        elif mod == 'j': vec[6] = 1.0 # Jussive
        # else: vec[6] remains 0.2 (na)

        # Index 7: Person (per) - 1=0, 2=0.5, 3=1, na=0.2
        per = morph_features.get('per')
        if per == '1': vec[7] = 0.0
        elif per == '2': vec[7] = 0.5
        elif per == '3': vec[7] = 1.0
        # else: vec[7] remains 0.2 (na)

        # --- Presence Features (Binary 0/1, default 0) ---
        vec[8:] = 0.0 # Reset presence features to 0

        # Index 8: Definite Article Prefix ('al-')
        has_det = False
        for key in ['prc0', 'prc1', 'prc2']:
            if morph_features.get(key) == 'det':
                has_det = True; break
        if has_det: vec[8] = 1.0

        # Index 9: Prep Prefix 'b'
        has_prep_b = False
        for key in ['prc0', 'prc1', 'prc2']:
            if morph_features.get(key) == 'prep' and morph_features.get(f'{key}_lex') == 'b':
                has_prep_b = True; break
        if has_prep_b: vec[9] = 1.0

        # Index 10: Prep Prefix 'l'
        has_prep_l = False
        for key in ['prc0', 'prc1', 'prc2']:
            if morph_features.get(key) == 'prep' and morph_features.get(f'{key}_lex') == 'l':
                has_prep_l = True; break
        if has_prep_l: vec[10] = 1.0

        # Index 11: Prep Prefix 'k'
        has_prep_k = False
        for key in ['prc0', 'prc1', 'prc2']:
            if morph_features.get(key) == 'prep' and morph_features.get(f'{key}_lex') == 'k':
                has_prep_k = True; break
        if has_prep_k: vec[11] = 1.0

        # Index 12: Conj Prefix 'w'
        has_conj_w = False
        for key in ['prc0', 'prc1', 'prc2']:
            if morph_features.get(key) == 'conj' and morph_features.get(f'{key}_lex') == 'w':
                has_conj_w = True; break
        if has_conj_w: vec[12] = 1.0

        # Index 13: Conj Prefix 'f'
        has_conj_f = False
        for key in ['prc0', 'prc1', 'prc2']:
            if morph_features.get(key) == 'conj' and morph_features.get(f'{key}_lex') == 'f':
                has_conj_f = True; break
        if has_conj_f: vec[13] = 1.0

        # Index 14: Future Prefix 'sa'
        has_fut_sa = False
        for key in ['prc0', 'prc1']: # Usually prc0 or prc1
            if morph_features.get(key) == 'fut_part' and morph_features.get(f'{key}_lex') == 'sa':
                has_fut_sa = True; break
        if has_fut_sa: vec[14] = 1.0

        # Index 15: Pronominal Suffix (enc0 indicates pronoun)
        enc0_type = morph_features.get('enc0')
        if enc0_type and 'pron' in enc0_type:
            vec[15] = 1.0

        # Final check for NaNs (shouldn't happen with this logic, but good practice)
        if not np.all(np.isfinite(vec)):
             logger.warning("NaN or Inf detected in morphological vector! Resetting to neutral.")
             vec = np.full(vec_size, 0.2)

        return vec

    # --- Parameter Binding (Unchanged) ---
    def _bind_parameters(self,
                            circuit: Any,
                            diagram_main: Optional[GrammarDiagram] = None,
                            tokens_from_main_processor: List[str] = None,
                            analyses_from_main_processor: List[Dict] = None,
                            linguistic_stream_core: Optional[List[Dict[str, Any]]] = None,
                            ambiguous_word_info_map: Optional[Dict[int, Dict[str, Any]]] = None
                            ) -> Optional[Dict[Parameter, float]]:
        """
        Unified parameter binding strategy.
        V8.2: Uses duck-typing for circuit validation and simplified categorization.
        """
        import re
        import hashlib
        from typing import Set, Tuple, Optional
        
        is_valid_circuit = (
            hasattr(circuit, 'qregs') and 
            hasattr(circuit, 'parameters') and 
            callable(getattr(circuit, 'draw', None))
        )

        if not QISKIT_AVAILABLE or not is_valid_circuit:
            logger.error(f"Invalid circuit or Qiskit unavailable. QISKIT_AVAILABLE: {QISKIT_AVAILABLE}, is_valid_circuit: {is_valid_circuit}")
            return None
        
        circuit_name = getattr(circuit, 'name', 'unnamed')
        logger.debug(f"Starting unified parameter binding for circuit '{circuit_name}'")
        
        bound_params: Dict[Parameter, float] = {}
        circuit_params: Set[Parameter] = circuit.parameters
        num_total_params = len(circuit_params)
        
        if num_total_params == 0:
            return {}
        
        token_map = self._build_token_mapping(
            diagram_main, tokens_from_main_processor, 
            analyses_from_main_processor, linguistic_stream_core, 
            ambiguous_word_info_map
        )
        
        param_categories = self._categorize_parameters(circuit_params, token_map)
        
        self._bind_feature_parameters(param_categories['feature'], token_map, bound_params)
        self._bind_variational_parameters(param_categories['variational'], bound_params)
        self._bind_unknown_parameters(param_categories['unknown'], bound_params)
        
        self._validate_parameter_binding(circuit_params, bound_params, circuit_name)
        
        return bound_params

    def _build_token_mapping(self,
                            diagram_main: Optional[GrammarDiagram],
                            tokens: Optional[List[str]],
                            analyses: Optional[List[Dict]],
                            core_stream: Optional[List[Dict[str, Any]]],
                            ambiguous_map: Optional[Dict[int, Dict[str, Any]]]
                        ) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive mapping from parameter prefixes to token information."""
        
        token_map: Dict[str, Dict[str, Any]] = {}
        
        # Strategy 1: Diagram-based mapping (most accurate)
        if diagram_main and hasattr(diagram_main, 'boxes'):
            for box in diagram_main.boxes:
                if not isinstance(box.data, dict) or 'original_stanza_idx' not in box.data:
                    continue
                    
                token_idx = box.data['original_stanza_idx']
                analysis = self._find_analysis_by_idx(analyses, token_idx)
                core_data = self._find_core_data_by_idx(core_stream, token_idx)
                
                if analysis:
                    token_map[box.name] = {
                        'source': 'diagram',
                        'token_idx': token_idx,
                        'analysis': analysis,
                        'core_data': core_data,
                        'is_ambiguous': isinstance(box, AmbiguousLexicalBox),
                        'senses': box.data.get('senses', []) if isinstance(box, AmbiguousLexicalBox) else [],
                        'box_name': box.name
                    }
        
        # Strategy 2: Direct analysis mapping (fallback)
        if analyses and not token_map:
            for analysis in analyses:
                token_idx = analysis.get('original_idx')
                if token_idx is None:
                    continue
                    
                # Generate likely parameter prefix
                lemma = analysis.get('lemma', analysis.get('text', ''))
                is_ambiguous = ambiguous_map and token_idx in ambiguous_map
                
                if is_ambiguous:
                    prefix = f"{lemma}_ambiguous"
                    senses = ambiguous_map[token_idx].get('senses', [])
                else:
                    prefix = f"{lemma}_std"
                    senses = []
                
                core_data = self._find_core_data_by_idx(core_stream, token_idx)
                
                token_map[prefix] = {
                    'source': 'analysis',
                    'token_idx': token_idx,
                    'analysis': analysis,
                    'core_data': core_data,
                    'is_ambiguous': is_ambiguous,
                    'senses': senses,
                    'box_name': prefix
                }
        
        logger.debug(f"Built token mapping with {len(token_map)} entries from {token_map.get(next(iter(token_map)), {}).get('source', 'unknown') if token_map else 'none'}")
        return token_map

    def _categorize_parameters(self, 
                            circuit_params: Set[Parameter],
                            token_map: Dict[str, Dict[str, Any]]
                            ) -> Dict[str, List[Tuple[Parameter, Dict[str, Any]]]]:
        """
        Categorize parameters by their binding strategy.
        V8.3: Robust parameter categorization.
        """
        
        categories = {'feature': [], 'variational': [], 'unknown': []}
        
        for param in circuit_params:
            if '_ancilla_' in param.name:
                categories['variational'].append((param, {'type': 'variational', 'param_name': param.name}))
                continue

            # --- MODIFICATION START ---
            # Match parameter names against the box names from the token map.
            # This is more robust than parsing parameter names directly.
            best_match_info = None
            longest_prefix = ""
            for box_name, token_info in token_map.items():
                # The parameter name is often a composite like 'box_name_sense_q0_p0'
                # or 'box_name__type_0'. We find the box_name that is the longest prefix.
                if param.name.startswith(box_name) and len(box_name) > len(longest_prefix):
                    longest_prefix = box_name
                    best_match_info = token_info
            # --- MODIFICATION END ---

            if best_match_info:
                param_info = {'type': 'feature', 'param_name': param.name, 'token_info': best_match_info}
                categories['feature'].append((param, param_info))
            else:
                categories['unknown'].append((param, {'type': 'unknown', 'param_name': param.name}))
        
        logger.debug(f"Parameter categorization: {len(categories['feature'])} feature, "
                    f"{len(categories['variational'])} variational, {len(categories['unknown'])} unknown")
        
        return categories

    def _parse_parameter_name(self, 
                            param_name: str,
                            patterns: List[re.Pattern],
                            token_map: Dict[str, Dict[str, Any]]
                            ) -> Dict[str, Any]:
        """Parse parameter name and determine its properties."""
        
        # Check for variational ancilla parameters first
        if '_anc_' in param_name:
            return {
                'type': 'variational',
                'param_name': param_name,
                'token_info': None,
                'sense': None,
                'param_idx': 0
            }
        
        # Try to match against known patterns
        for pattern in patterns:
            match = pattern.match(param_name)
            if match:
                groups = match.groups()
                
                if len(groups) >= 3:  # Has param index
                    prefix = groups[0]
                    param_idx = int(groups[-1])
                    sense = groups[1] if len(groups) == 4 else None
                    
                    # Find matching token info
                    token_info = self._find_token_info_for_prefix(prefix, token_map, sense)
                    
                    return {
                        'type': 'feature',
                        'param_name': param_name,
                        'token_info': token_info,
                        'sense': sense,
                        'param_idx': param_idx,
                        'prefix': prefix
                    }
        
        # No pattern matched - unknown parameter
        return {
            'type': 'unknown',
            'param_name': param_name,
            'token_info': None,
            'sense': None,
            'param_idx': 0
        }

    def _find_token_info_for_prefix(self,
                                prefix: str,
                                token_map: Dict[str, Dict[str, Any]],
                                sense: Optional[str] = None
                                ) -> Optional[Dict[str, Any]]:
        """Find token information for a given parameter prefix."""
        
        # Direct match
        if prefix in token_map:
            return token_map[prefix]
        
        # Partial match - find best candidate
        candidates = []
        for box_name, token_info in token_map.items():
            if prefix.startswith(box_name) or box_name.startswith(prefix):
                # Score based on match quality and sense compatibility
                score = len(set(prefix) & set(box_name))
                if sense and sense in token_info.get('senses', []):
                    score += 10  # Bonus for sense match
                candidates.append((score, token_info))
        
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        
        return None

    def _bind_feature_parameters(self,
                            feature_params: List[Tuple[Parameter, Dict[str, Any]]],
                            token_map: Dict[str, Dict[str, Any]],
                            bound_params: Dict[Parameter, float]
                            ) -> None:
        """Bind parameters using linguistic features."""
        
        for param, param_info in feature_params:
            token_info = param_info['token_info']
            if not token_info:
                continue
                
            sense = param_info.get('sense')
            param_idx = param_info.get('param_idx', 0)
            
            # Get feature vector for this token/sense
            feature_vec = self._get_feature_vector(token_info, sense)
            
            if feature_vec is not None and feature_vec.size > 0:
                # Use parameter index to select from feature vector
                feat_val = feature_vec[param_idx % feature_vec.size]
                angle = np.tanh(feat_val) * np.pi
                angle = max(-np.pi, min(np.pi, angle))
                
                if not np.isfinite(angle):
                    angle = 0.0
                    
                bound_params[param] = angle
                
                logger.debug(f"Bound feature param '{param.name}' to {angle:.3f} "
                            f"(token='{token_info['analysis'].get('text', 'unknown')}', "
                            f"sense='{sense}', idx={param_idx})")
            else:
                # Fallback to hash if no features available
                bound_params[param] = self._hash_to_angle(param.name)
                logger.warning(f"No features for param '{param.name}', using hash fallback")

    def _bind_variational_parameters(self,
                                variational_params: List[Tuple[Parameter, Dict[str, Any]]],
                                bound_params: Dict[Parameter, float]
                                ) -> None:
        """Handle variational parameters (typically left unbound for optimization)."""
        
        variational_names = [param.name for param, _ in variational_params]
        if variational_names:
            logger.info(f"Identified {len(variational_names)} variational parameters: {variational_names[:5]}{'...' if len(variational_names) > 5 else ''}")
        
        # Variational parameters are intentionally not added to bound_params
        # They will be handled by the quantum optimization process

    def _bind_unknown_parameters(self,
                            unknown_params: List[Tuple[Parameter, Dict[str, Any]]],
                            bound_params: Dict[Parameter, float]
                            ) -> None:
        """Bind unknown parameters using hash fallback."""
        
        for param, param_info in unknown_params:
            bound_params[param] = self._hash_to_angle(param.name)
            logger.warning(f"Unknown parameter '{param.name}' bound using hash fallback")

    def _get_feature_vector(self,
                        token_info: Dict[str, Any],
                        sense: Optional[str] = None
                        ) -> Optional[np.ndarray]:
        """Get feature vector for token/sense with priority-based selection."""
        
        analysis = token_info['analysis']
        lemma = analysis.get('lemma', analysis.get('text', ''))
        
        # Priority 1: Sense-specific embedding
        if sense and hasattr(self, 'embedding_model') and self.embedding_model:
            sense_key = f"{lemma}_{sense}"
            if self.embedding_model.has_index_for(sense_key):
                return self.embedding_model.get_vector(sense_key)
        
        # Priority 2: Generic embedding (with sense perturbation if needed)
        if hasattr(self, 'embedding_model') and self.embedding_model and self.embedding_model.has_index_for(lemma):
            vec = self.embedding_model.get_vector(lemma)
            if sense:
                # Add deterministic perturbation for sense differentiation
                sense_hash = int(hashlib.sha256(sense.encode()).hexdigest(), 16)
                perturb_factor = 0.01 * ((sense_hash % 100) / 100.0)
                rng = np.random.RandomState(sense_hash)
                vec = vec + perturb_factor * rng.randn(len(vec))
            return vec
        
        # Priority 3: Core linguistic features
        if token_info.get('core_data') and 'classical_feature_vector' in token_info['core_data']:
            core_vec = token_info['core_data']['classical_feature_vector']
            if core_vec:
                return np.array(core_vec)
        
        # Priority 4: Morphological features (assumes get_morph_vector is a class method)
        if hasattr(self, 'get_morph_vector'):
            morph_vec = self.get_morph_vector(analysis)
            if morph_vec is not None and morph_vec.size > 0:
                return morph_vec
        
        # Priority 5: Fallback to class default
        if hasattr(self, 'params_per_word_fallback'):
            return np.array([0.1] * self.params_per_word_fallback)
        
        return None

    def _hash_to_angle(self, param_name: str) -> float:
        """Convert parameter name to angle using consistent hash."""
        hash_val = int(hashlib.sha256(param_name.encode('utf-8')).hexdigest(), 16)
        return ((hash_val % (2 * np.pi * 10000)) / 10000.0) - np.pi

    def _find_analysis_by_idx(self, analyses: Optional[List[Dict]], idx: int) -> Optional[Dict]:
        """Find analysis entry by original_idx."""
        if not analyses:
            return None
        return next((ana for ana in analyses if ana.get('original_idx') == idx), None)

    def _find_core_data_by_idx(self, core_stream: Optional[List[Dict]], idx: int) -> Optional[Dict]:
        """Find core data entry by original_stanza_idx."""
        if not core_stream:
            return None
        return next((core for core in core_stream if core.get('original_stanza_idx') == idx), None)

    def _validate_parameter_binding(self,
                                circuit_params: Set[Parameter],
                                bound_params: Dict[Parameter, float],
                                circuit_name: str
                                ) -> None:
        """Validate parameter binding results and log summary."""
        
        total_params = len(circuit_params)
        bound_count = len(bound_params)
        unbound_params = circuit_params - set(bound_params.keys())
        
        logger.info(f"Parameter binding summary for '{circuit_name}': "
                f"{bound_count}/{total_params} parameters bound")
        
        if unbound_params:
            unbound_names = [p.name for p in unbound_params]
            logger.info(f"Unbound (variational) parameters: {unbound_names}")
        
        # Validate bound values
        invalid_params = []
        for param, value in bound_params.items():
            if not np.isfinite(value):
                invalid_params.append(param.name)
                logger.error(f"Invalid value for parameter '{param.name}': {value}")
        
        if invalid_params:
            logger.error(f"Found {len(invalid_params)} parameters with invalid values")

        # --- End of _bind_parameters implementation ---

    # --- Quantum Feature Extraction (Unchanged) ---
    def get_enhanced_circuit_features(self,
                                      circuit: QuantumCircuit,
                                      tokens_from_main_processor: List[str], 
                                      analyses_from_main_processor: List[Dict], 
                                      linguistic_stream_core: Optional[List[Dict]] # ADD THIS PARAMETER
                                     ) -> np.ndarray:
        fallback_features = np.zeros(self.embedding_dim)
        if not QISKIT_AVAILABLE or self.estimator is None: return fallback_features
        if not isinstance(circuit, QuantumCircuit): return fallback_features

        circuit_name = f"circ_{tokens_from_main_processor[0]}" if tokens_from_main_processor else "unnamed_circ"
        circuit = _ensure_circuit_name(circuit, circuit_name)
        num_qubits = circuit.num_qubits
        if num_qubits == 0: return fallback_features

        try:
            # MODIFY THIS CALL to include linguistic_stream_core
            parameter_binds_map = self._bind_parameters(circuit, tokens_from_main_processor, analyses_from_main_processor, linguistic_stream_core)
            if parameter_binds_map is None: return fallback_features
            
            circuit_to_estimate = circuit.remove_final_measurements(inplace=False)
            params_list = sorted(list(circuit.parameters), key=lambda p: p.name)
            param_values_ordered = [parameter_binds_map.get(p, 0.0) for p in params_list]

            observables = []
            for i in range(num_qubits): 
                for pauli_char in ["Z", "X", "Y"]:
                    pauli_str = ["I"] * num_qubits
                    pauli_str[i] = pauli_char
                    observables.append(SparsePauliOp.from_list([("".join(pauli_str), 1)]))
            if not observables: return fallback_features

            param_values_for_job = [param_values_ordered] * len(observables) if params_list else [[]] * len(observables)
            job = self.estimator.run(circuits=[circuit_to_estimate] * len(observables), observables=observables, parameter_values=param_values_for_job)
            result = job.result()
            feature_vector = np.array(result.values.tolist())

            current_len = len(feature_vector)
            if current_len == 0: return fallback_features
            elif current_len < self.embedding_dim: feature_vector = np.pad(feature_vector, (0, self.embedding_dim - current_len), 'constant')
            elif current_len > self.embedding_dim: feature_vector = feature_vector[:self.embedding_dim]
            
            norm = np.linalg.norm(feature_vector)
            final_features = feature_vector / norm if norm > 1e-9 else feature_vector
            return final_features if np.all(np.isfinite(final_features)) else fallback_features
        except Exception as e:
            logger.error(f"Error in get_enhanced_circuit_features: {e}", exc_info=True)
            return fallback_features

        # --- 6. Pad/Truncate, Normalize ---
        final_features = np.array(feature_vector)
        current_len = len(final_features)

        if current_len == 0:
             logger.warning("Estimator feature vector is empty after processing. Returning zeros.")
             return fallback_features
        elif current_len < self.embedding_dim:
             final_features = np.pad(final_features, (0, self.embedding_dim - current_len), 'constant')
             if debug: logger.debug(f"Padded Estimator feature vector from {current_len} to {self.embedding_dim}")
        elif current_len > self.embedding_dim:
             final_features = final_features[:self.embedding_dim]
             if debug: logger.debug(f"Truncated Estimator feature vector from {current_len} to {self.embedding_dim}")

        # Normalize final vector
        norm = np.linalg.norm(final_features)
        if norm > 1e-9: # Avoid division by zero
            final_features = final_features / norm
            if debug: logger.debug(f"Final Estimator feature vector normalized (Norm: {norm:.4f}).")
        else:
             if debug: logger.warning("Final Estimator feature vector norm is near zero. Not normalizing.")

        # Final check for NaNs
        if not np.all(np.isfinite(final_features)):
             logger.error("NaN or Inf found in final Estimator feature vector! Returning zeros.")
             return fallback_features

        return final_features
        # --- End of get_enhanced_circuit_features implementation ---


    # --- Linguistic Feature Extraction (Unchanged) ---
    def extract_complex_linguistic_features_v2(self,
                                               tokens_main: List[str], 
                                               analyses_main: List[Dict], # From camel_test2
                                               structure_main: str, # From camel_test2
                                               roles_main: Dict, # From camel_test2
                                               linguistic_stream_core: Optional[List[Dict[str, Any]]] # From core module
                                              ) -> np.ndarray:
        """ Extracts classical linguistic features. Prefers linguistic_stream_core if available. """
        target_dim = self.classical_feature_dim_from_core 
        features = np.zeros(target_dim)
        
        current_feature_idx = 0
        def add_feature_value(val):
            nonlocal current_feature_idx
            if current_feature_idx < target_dim:
                features[current_feature_idx] = max(0.0, min(1.0, float(val)))
                current_feature_idx += 1
        
        if linguistic_stream_core:
            logger.debug(f"Extracting linguistic features using CORE stream for '{tokens_main[0] if tokens_main else 'unk'}...'")
            all_core_word_feats = [
                item.get('classical_features_of_surface_word') 
                for item in linguistic_stream_core 
                if isinstance(item.get('classical_features_of_surface_word'), np.ndarray) and \
                   item.get('classical_features_of_surface_word').shape == (self.classical_feature_dim_from_core,)
            ]
            
            if all_core_word_feats:
                avg_core_feats = np.mean(all_core_word_feats, axis=0)
                norm_avg = np.linalg.norm(avg_core_feats)
                avg_core_feats_norm = avg_core_feats / norm_avg if norm_avg > 1e-9 else avg_core_feats
                for val in avg_core_feats_norm:
                    add_feature_value(val)
            else: 
                logger.warning(f"Core stream for '{tokens_main[0] if tokens_main else 'unk'}' had no valid classical_features. Using fallback structure/roles.")
                structure_map = {'VSO': 0, 'SVO': 0.2, 'NOMINAL': 0.4, 'OTHER': 0.6, 'VERBAL_LIKE':0.1, 'NOMINAL_LIKE':0.3, 'OTHER_STRUCTURE':0.5 }
                add_feature_value(structure_map.get(structure_main.split('_')[0], 0.6))
                add_feature_value(1.0 if roles_main.get('verb') is not None else 0.0)
        else: 
            logger.debug(f"Extracting linguistic features using MAIN (camel_test2) analysis for '{tokens_main[0] if tokens_main else 'unk'}...'")
            structure_map = {'VSO': 0, 'SVO': 0.2, 'NOMINAL': 0.4, 'OTHER': 0.6, 'VERBAL_LIKE':0.1, 'NOMINAL_LIKE':0.3, 'OTHER_STRUCTURE':0.5 }
            add_feature_value(structure_map.get(structure_main.split('_')[0], 0.6))
            pos_counts = Counter(a.get('upos', 'UNK') for a in analyses_main)
            total_toks = max(1, len(tokens_main))
            for pos_tag in ['VERB', 'NOUN', 'ADJ', 'ADP', 'PRON']: # Add more if needed
                add_feature_value(pos_counts.get(pos_tag,0) / total_toks)
        
        while current_feature_idx < target_dim: # Pad remaining
            add_feature_value(0.05) 

        norm_final_ling = np.linalg.norm(features)
        return features / norm_final_ling if norm_final_ling > 1e-9 else features
          # --- End of extract_complex_linguistic_features_v2 implementation ---

    # --- Combine Features (Unchanged) ---
    def combine_features_with_attention(self, quantum_features, linguistic_features, structure):
        """ Combine features with attention mechanism. V2 """
        # Keep the attention logic based on structure
        base_structure = structure.split('_')[0]
        if base_structure == 'NOMINAL': quantum_weight, linguistic_weight = 0.4, 0.6 # Give more weight to linguistics for nominal
        elif base_structure in ['VSO', 'SVO']: quantum_weight, linguistic_weight = 0.5, 0.5
        elif base_structure == 'COMPLEX': quantum_weight, linguistic_weight = 0.6, 0.4
        else: quantum_weight, linguistic_weight = 0.5, 0.5

        # Ensure dimensions match self.embedding_dim
        q_len = len(quantum_features); l_len = len(linguistic_features)
        if q_len != self.embedding_dim: # Quantum features should already be padded/truncated
             logger.warning(f"Quantum feature length mismatch ({q_len} vs {self.embedding_dim}). Padding/truncating.")
             if q_len < self.embedding_dim: quantum_features = np.pad(quantum_features, (0, self.embedding_dim - q_len), 'constant')
             elif q_len > self.embedding_dim: quantum_features = quantum_features[:self.embedding_dim]
        if l_len != self.embedding_dim: # Linguistic features might be shorter
             logger.debug(f"Linguistic feature length mismatch ({l_len} vs {self.embedding_dim}). Padding.")
             if l_len < self.embedding_dim: linguistic_features = np.pad(linguistic_features, (0, self.embedding_dim - l_len), 'constant')
             elif l_len > self.embedding_dim: linguistic_features = linguistic_features[:self.embedding_dim] # Should not happen if max_features is set

        # Combine and normalize
        combined = quantum_weight * quantum_features + linguistic_weight * linguistic_features
        norm = np.linalg.norm(combined)
        combined = combined / norm if norm > 1e-9 else combined
        # Final check for NaNs
        if not np.all(np.isfinite(combined)):
             logger.error("NaN or Inf found in combined feature vector! Returning zeros.")
             return np.zeros_like(combined)
        return combined


    # --- Training (Modified to store linguistic features) ---
    def train(self,
              original_indices: List[int], 
              sentences_str: List[str],
              circuits_main: List[QuantumCircuit],
              diagrams_main: List[Optional[GrammarDiagram]],
              tokens_main: List[List[str]],
              analyses_main: List[List[Dict]], 
              structures_main: List[str],
              roles_main: List[Dict],
              linguistic_streams_core: List[Optional[List[Dict[str, Any]]]],
              assigned_diagram_elements_main: List[List[Union[Word,Box]]], # List of lists of diagram elements
              ambiguous_maps_for_sentences: Optional[List[Optional[Dict[int, Dict[str, Any]]]]] = None # ADD THIS
             ):

        self.ambiguous_maps_for_training_data = ambiguous_maps_for_sentences # Store
        self.assigned_diagram_elements_for_training_data = assigned_diagram_elements_main # Store
        self.reference_sentences = {orig_idx: sentences_str[i] for i, orig_idx in enumerate(original_indices)}
        self.circuit_embeddings = {}
        self.sentence_embeddings = {}
        self.linguistic_features = {}
        self.sentence_diagrams_main = {orig_idx: diagrams_main[i] for i, orig_idx in enumerate(original_indices)}

        logger.info(f"\n--- Kernel Training (Core Conditionally Integrated) for {len(sentences_str)} sentences ---")

        for i in range(len(sentences_str)): # i is the dense index for these lists
            original_idx = original_indices[i] 
            try:
                current_circuit_main = circuits_main[i]
                current_tokens_main = tokens_main[i]
                current_analyses_main = analyses_main[i]
                current_structure_main = structures_main[i]
                current_roles_main = roles_main[i]
                # current_diagram_main is not directly used in feature extraction, but was stored above
                current_amb_map_for_sent = None
                if self.ambiguous_maps_for_training_data and i < len(self.ambiguous_maps_for_training_data):
                    current_amb_map_for_sent = self.ambiguous_maps_for_training_data[i]
        
                current_linguistic_stream_core = linguistic_streams_core[i] # This is now aligned

                if not isinstance(current_circuit_main, QuantumCircuit): 
                    logger.warning(f"Skipping original_idx {original_idx}: Invalid circuit.")
                    continue

                quantum_features = self.get_enhanced_circuit_features(
                    current_circuit_main, current_tokens_main, current_analyses_main, current_linguistic_stream_core, current_ambiguous_word_info_map=current_amb_map_for_sent # Pass stream
                )
                self.circuit_embeddings[original_idx] = quantum_features

                linguistic_features_vec = self.extract_complex_linguistic_features_v2(
                    current_tokens_main, current_analyses_main, current_structure_main, current_roles_main, current_linguistic_stream_core # Pass stream
                )
                self.linguistic_features[original_idx] = linguistic_features_vec

                combined_embedding = self.combine_features_with_attention(
                    quantum_features, linguistic_features_vec, current_structure_main 
                )
                self.sentence_embeddings[original_idx] = combined_embedding
                logger.debug(f"Embeddings for original_idx {original_idx}: Q({quantum_features.shape}), L({linguistic_features_vec.shape}), C({combined_embedding.shape})")
            except Exception as e:
                logger.error(f"Error processing sentence original_idx {original_idx} ('{sentences_str[i][:30]}...'): {e}", exc_info=True)
        
        self.learn_meaning_clusters_from_quantum()
        
        # Prepare data for assign_meaning_to_clusters, ensuring alignment by original_idx
        # These lists should correspond to the items that successfully got embeddings.
        # The keys of self.circuit_embeddings are the original_indices that were successful.
        indices_with_embeddings = sorted(list(self.circuit_embeddings.keys()))

        sentences_for_meaning = [self.reference_sentences[idx] for idx in indices_with_embeddings if idx in self.reference_sentences]
        
        # Find the dense index in original_indices to fetch corresponding structures/roles/streams
        dense_indices_for_kernel_data = [original_indices.index(idx) for idx in indices_with_embeddings if idx in original_indices]

        structures_for_meaning = [structures_main[d_idx] for d_idx in dense_indices_for_kernel_data]
        roles_for_meaning = [roles_main[d_idx] for d_idx in dense_indices_for_kernel_data]
        
        # For linguistic_streams_core, it might contain Nones. Filter or handle Nones in assign_meaning.
        # For simplicity, pass the aligned (potentially None-containing) streams.
        streams_for_meaning = []
        for d_idx in dense_indices_for_kernel_data:
            if hasattr(self, 'linguistic_streams_core_for_kernel_train_data') and self.linguistic_streams_core_for_kernel_train_data is not None:
                if d_idx < len(self.linguistic_streams_core_for_kernel_train_data):
                    stream_data = self.linguistic_streams_core_for_kernel_train_data[d_idx]
                else:
                    logger.error(f"d_idx {d_idx} out of bounds for self.linguistic_streams_core_for_kernel_train_data (len {len(self.linguistic_streams_core_for_kernel_train_data)})")
                    stream_data = [] # Or handle error appropriately
            else:
                logger.warning("self.linguistic_streams_core_for_kernel_train_data not found or is None in kernel.train().")
                stream_data = [] # Default to empty list to avoid further errors # This is the list from arabic_morpho_lex_core
            # The 'analyses' in assign_meaning_to_clusters comes from the 'linguistic_stream_of_words'
            # within each entry of kernel_train_linguistic_streams_core.
            # Each item in kernel_train_linguistic_streams_core should be a list of dicts (the stream itself),
            # or None if core processing failed.
            if stream_data is None: # If the entire stream for a sentence is None
                logger.warning(f"No linguistic stream from core module for sentence at dense_idx {d_idx} (original_idx {indices_with_embeddings[dense_indices_for_kernel_data.index(d_idx)]}). Passing empty list to assign_meaning_to_clusters.")
                streams_for_meaning.append([]) # Pass an empty list for 'analyses'
            else:
                streams_for_meaning.append(stream_data) # Pass the actual stream (list of word dicts)

        self.assign_meaning_to_clusters(sentences_for_meaning, structures_for_meaning, roles_for_meaning, streams_for_meaning)
        return self

    # --- Clustering (On Quantum Embeddings - Unchanged) ---
    def learn_meaning_clusters_from_quantum(self) -> None:
        """ Learn meaning clusters from the stored quantum embeddings using KMeans. """
        if not self.circuit_embeddings:
            logger.warning("No quantum embeddings available for clustering."); return
        embeddings_list = list(self.circuit_embeddings.values())
        indices = list(self.circuit_embeddings.keys())

        if not embeddings_list: logger.warning("Embeddings list is empty."); return

        X = np.array(embeddings_list)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
             logger.warning("Non-finite values found in quantum embeddings. Removing problematic rows for clustering.")
             X = X[finite_mask]
             original_indices = indices
             indices = [idx for i, idx in enumerate(original_indices) if finite_mask[i]]
             if X.shape[0] == 0: logger.error("All quantum embeddings contained non-finite values."); return
             logger.info(f"Removed {len(finite_mask) - X.shape[0]} non-finite rows before clustering.")

        n_samples = X.shape[0]
        if n_samples == 0: logger.warning("Quantum embedding array is empty after filtering. Cannot cluster."); return

        n_clusters = min(self.num_clusters, n_samples)
        if n_clusters <= 0: n_clusters = 1
        self.num_clusters = n_clusters # Update actual number of clusters used

        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(X)
            self.kmeans_model = kmeans # Store the fitted model
            self.meaning_clusters = kmeans.cluster_centers_ # Store cluster centers

            # Map labels back to original indices (handling filtered NaNs)
            self.cluster_labels = np.full(len(self.reference_sentences), -1, dtype=int) # Initialize with -1
            for i, original_idx in enumerate(indices):
                 if original_idx < len(self.cluster_labels):
                      self.cluster_labels[original_idx] = raw_labels[i]
                 else:
                      logger.warning(f"Original index {original_idx} out of bounds for cluster labels array (size {len(self.cluster_labels)}).")

            logger.info(f"KMeans clustering performed on {n_samples} quantum embeddings into {self.num_clusters} clusters.")

        except Exception as e:
            logger.error(f"Error during KMeans clustering on quantum embeddings: {e}", exc_info=True)
            self.cluster_labels = None; self.meaning_clusters = None; self.kmeans_model = None
        # --- End of learn_meaning_clusters_from_quantum ---


    # --- Meaning Assignment (Unchanged - Describes clusters) ---
    def assign_meaning_to_clusters(self, sentences: List[str], structures: List[str], roles_list: List[Dict], analyses_list: List[List[Dict]]) -> Dict:
        """
        Assign meaning templates to clusters based on linguistic analysis.
        V6.3: Correctly handles list of analysis dictionaries.
        """
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            logger.warning("Cluster labels not found. Cannot assign meanings.")
            self.meaning_map = {}
            return self.meaning_map

        cluster_data_grouped = {label: [] for label in range(self.num_clusters)} # Initialize for all potential clusters
        min_len = min(len(sentences), len(structures), len(roles_list), len(analyses_list), len(self.cluster_labels))

        # Group data by cluster label
        for i in range(min_len):
            label = self.cluster_labels[i]
            if label == -1: continue # Skip unassigned sentences
            if label not in cluster_data_grouped: # Should not happen if initialized correctly
                 cluster_data_grouped[label] = []
            cluster_data_grouped[label].append({
                'sentence': sentences[i],
                'structure': structures[i],
                'roles': roles_list[i],
                'analyses': analyses_list[i], # This is the list of dicts for sentence i
                'index': i
            })

        # Process each cluster
        self.meaning_map = {} # Reset meaning map
        for cluster_id, cluster_data in cluster_data_grouped.items():
            if not cluster_data:
                self.meaning_map[cluster_id] = {'deduced_template': 'Empty Cluster', 'examples': []}
                continue

            # Counters for features within the cluster
            verb_lemmas = Counter(); subject_lemmas = Counter(); object_lemmas = Counter()
            common_preps = Counter(); verb_tenses = Counter(); verb_moods = Counter()
            structure_counts = Counter(); pos_counts = Counter()

            # Iterate through items (sentences) in the current cluster
            for item in cluster_data:
                structure_counts[item['structure']] += 1
                # tokens = item['sentence'].split() # Not needed directly here
                roles = item['roles']
                analyses = item['analyses'] # This should now be a list (possibly empty), not None
                if analyses is None: # Add a safeguard, though the above change should prevent it
                    logger.warning(f"assign_meaning_to_clusters: 'analyses' for sentence '{item['sentence']}' is None. Skipping morph/POS details for it.")
                    analyses = [] # Default to empty list to prevent len() error# List of analysis dicts for this sentence

                verb_idx = roles.get('verb')
                subj_idx = roles.get('subject')
                obj_idx = roles.get('object')

                # Extract features using the analysis dictionaries
                if verb_idx is not None and verb_idx < len(analyses):
                    verb_lemmas[analyses[verb_idx].get('lemma', 'UNK_LEMMA')] += 1
                if subj_idx is not None and subj_idx < len(analyses):
                    subject_lemmas[analyses[subj_idx].get('lemma', 'UNK_LEMMA')] += 1
                if obj_idx is not None and obj_idx < len(analyses):
                    object_lemmas[analyses[obj_idx].get('lemma', 'UNK_LEMMA')] += 1

                # Iterate through analysis dicts for POS, Morph, Prepositions
                for analysis_dict in analyses:
                    pos_counts[analysis_dict.get('upos', 'UNK_POS')] += 1 # Use upos
                    # Use combined feats_dict for morph features
                    morph = analysis_dict.get('feats_dict')
                    if morph: # Morphological features (if available)
                        verb_tenses[morph.get('Aspect', 'UNK')] += 1 # Use Stanza/CAMeL keys
                        verb_moods[morph.get('Mood', 'UNK')] += 1

                    # Extract preposition info
                    if analysis_dict.get('upos') == 'ADP':
                        prep_lemma = analysis_dict.get('lemma', 'UNK_PREP')
                        common_preps[prep_lemma] += 1

            # Determine dominant features for the cluster
            dominant_structure = structure_counts.most_common(1)[0][0] if structure_counts else 'OTHER'
            dominant_verb = verb_lemmas.most_common(1)[0][0] if verb_lemmas else None
            dominant_subj = subject_lemmas.most_common(1)[0][0] if subject_lemmas else "SUBJECT"
            dominant_obj = object_lemmas.most_common(1)[0][0] if object_lemmas else "OBJECT"
            top_prep = common_preps.most_common(1)[0][0] if common_preps else None
            dominant_tense = verb_tenses.most_common(1)[0][0] if verb_tenses else None
            dominant_mood = verb_moods.most_common(1)[0][0] if verb_moods else None

            # Deduce a template based on dominant features
            deduced_template = f"{dominant_subj} (did something involving) {dominant_obj}"
            verb_class = self._classify_verb(dominant_verb)

            # Refine template based on verb class and structure
            if verb_class == "MOTION":
                dest = top_prep if top_prep and top_prep in ['إلى', 'ل'] else "DESTINATION"
                deduced_template = f"{dominant_subj} went to {dest}"
            elif verb_class == "COMMUNICATION":
                msg = dominant_obj if dominant_obj != "OBJECT" else "MESSAGE"
                deduced_template = f"{dominant_subj} said {msg}"
            elif verb_class == "POSSESSION":
                item = dominant_obj if dominant_obj != "OBJECT" else "ITEM"
                deduced_template = f"{dominant_subj} has/got {item}"
            elif verb_class == "COGNITION":
                thought = dominant_obj if dominant_obj != "OBJECT" else "IDEA"
                deduced_template = f"{dominant_subj} thinks about {thought}"
            elif verb_class == "EMOTION":
                stimulus = dominant_obj if dominant_obj != "OBJECT" else "SOMETHING"
                deduced_template = f"{dominant_subj} feels emotion about {stimulus}"
            elif dominant_verb:
                base_structure = dominant_structure.split('_')[0]
                if base_structure == 'VSO':
                    action_desc = f"{dominant_verb} performed by {dominant_subj}"
                    action_desc += f" on {dominant_obj}" if dominant_obj != "OBJECT" else ""
                    deduced_template = action_desc
                elif base_structure == 'SVO':
                    action_desc = f"{dominant_subj} performs {dominant_verb}"
                    action_desc += f" on {dominant_obj}" if dominant_obj != "OBJECT" else ""
                    deduced_template = action_desc
                elif base_structure == 'NOMINAL':
                    predicate = "ATTRIBUTE"
                    pred_lemmas = Counter()
                    for item_inner in cluster_data:
                        subj_idx_inner = item_inner['roles'].get('subject')
                        if subj_idx_inner is not None:
                            for k, analysis_dict_k in enumerate(item_inner['analyses']):
                                pos_k = analysis_dict_k.get('upos')
                                head_k = analysis_dict_k.get('head')
                                lemma_k = analysis_dict_k.get('lemma')
                                if head_k == subj_idx_inner and pos_k == 'ADJ':
                                    pred_lemmas[lemma_k] += 1
                                    break
                    if pred_lemmas: predicate = pred_lemmas.most_common(1)[0][0]
                    deduced_template = f"{dominant_subj} is {predicate}"
                else:
                    deduced_template = f"Statement about {dominant_subj} involving {dominant_verb}"

            # Add tense information
            tense_map = {'Perf': ' (past)', 'Impf': ' (present)', 'Subj': ' (subjunctive)', 'Imper': ' (command)'} # Use Stanza/CAMeL keys
            deduced_template += tense_map.get(dominant_tense, "")

            sentiment_label = None # Add sentiment logic if needed

            # Update the meaning map for the cluster
            self.meaning_map[cluster_id] = {
                'structure': dominant_structure,
                'deduced_template': deduced_template,
                'dominant_verb': dominant_verb,
                'dominant_subject': dominant_subj,
                'dominant_object': dominant_obj,
                'common_prep_phrase': top_prep, # Store the lemma directly
                'sentiment': sentiment_label,
                'examples': [item['sentence'] for item in cluster_data[:3]],
                'original_templates': self.semantic_templates.get(dominant_structure.split('_')[0], self.semantic_templates.get('OTHER', {}))
            }

        return self.meaning_map
        # --- End of assign_meaning_to_clusters ---


    # --- Interpretation (Modified to return BOTH embeddings & handle context) ---
    def interpret_sentence(self,
                           original_idx: int, 
                           circuit_main: QuantumCircuit,
                           tokens_main: List[str], analyses_main: List[Dict],
                           structure_main: str, roles_main: Dict, 
                           diagram_main: Optional[GrammarDiagram], # Diagram from camel_test2
                           linguistic_stream_core: Optional[List[Dict[str, Any]]], # ADD THIS
                           previous_analysis_dict=None) -> Dict:
        sentence_text = self.reference_sentences.get(original_idx, ' '.join(tokens_main))
        base_result = {
            'sentence': sentence_text, 'structure': structure_main, 'roles': roles_main, 
            'quantum_embedding': None, 'combined_embedding': None, 'linguistic_features_vector': None, 
            'error': None, 'discourse_relations': []
        }

        if not isinstance(circuit_main, QuantumCircuit):
            base_result['error'] = 'Invalid circuit object'; return base_result

        quantum_features = self.get_enhanced_circuit_features(
            circuit_main, tokens_main, analyses_main, linguistic_stream_core # Pass stream
        )
        if not np.all(np.isfinite(quantum_features)):
            base_result['error'] = 'Failed to generate valid quantum features'; return base_result
        base_result['quantum_embedding'] = quantum_features

        linguistic_features_vec = self.extract_complex_linguistic_features_v2(
            tokens_main, analyses_main, structure_main, roles_main, linguistic_stream_core # Pass stream
        )
        base_result['linguistic_features_vector'] = linguistic_features_vec

        combined_embedding = self.combine_features_with_attention(
            quantum_features, linguistic_features_vec, structure_main
        )
        base_result['combined_embedding'] = combined_embedding
        
        if self.kmeans_model is not None and base_result['quantum_embedding'] is not None:
             try:
                 q_emb_reshaped = base_result['quantum_embedding'].reshape(1, -1)
                 if np.all(np.isfinite(q_emb_reshaped)): 
                    cluster_pred = self.kmeans_model.predict(q_emb_reshaped)
                    base_result['nearest_cluster_id'] = int(cluster_pred[0])
                    base_result['nearest_cluster_desc'] = self.meaning_map.get(int(cluster_pred[0]), {}).get('deduced_template', 'N/A')
                 else: logger.warning(f"NaN/Inf in quantum embedding for original_idx {original_idx} before cluster prediction.")
             except Exception as e_pred: logger.warning(f"Could not predict nearest cluster for original_idx {original_idx}: {e_pred}")
        return base_result


    # --- Context Analysis (Modified to take base_interpretation) ---
    def analyze_sentence_in_context(self, current_interpretation_dict, previous_analysis_dict):
        """ Analyze a sentence considering the previous sentence's context. V8"""
        if previous_analysis_dict is None:
            return current_interpretation_dict # No context to analyze
        if current_interpretation_dict.get('error'): # Propagate error
             return current_interpretation_dict

        previous_tokens = previous_analysis_dict.get('tokens')
        current_tokens = current_interpretation_dict.get('sentence', '').split() # Get tokens from current dict

        discourse_info = self.find_discourse_relations(current_tokens, previous_tokens)
        current_interpretation_dict['discourse_relations'] = discourse_info

        # Optional: Contextual Embedding Adjustment (using combined embeddings)
        previous_embedding = previous_analysis_dict.get('interpretation', {}).get('combined_embedding')
        current_embedding = current_interpretation_dict.get('combined_embedding')

        if previous_embedding is not None and current_embedding is not None:
            context_influence = 0.2 # How much the previous sentence influences the current
            if isinstance(current_embedding, np.ndarray) and isinstance(previous_embedding, np.ndarray) and current_embedding.shape == previous_embedding.shape:
                context_aware_embedding = (1 - context_influence) * current_embedding + context_influence * previous_embedding
                norm = np.linalg.norm(context_aware_embedding);
                context_aware_embedding = context_aware_embedding / norm if norm > 1e-9 else context_aware_embedding
                current_interpretation_dict['context_aware_embedding'] = context_aware_embedding
            else:
                logger.warning("Combined embedding dimension/type mismatch. Skipping context blending.")
        elif current_embedding is None:
             logger.warning("Current combined embedding is None. Skipping context blending.")
        elif previous_embedding is None:
             logger.debug("Previous combined embedding is None. No context blending needed.")

        return current_interpretation_dict


    # --- Discourse Relation Finder (Unchanged) ---
    def find_discourse_relations(self, current_tokens, previous_tokens):
        """ Basic discourse relation detection based on markers. """
        discourse_relations = []
        if not current_tokens or not previous_tokens: return discourse_relations
        discourse_markers = {
            'CONTINUATION': ['و', 'ثم', 'ف', 'بعد ذلك', 'بعدها'], 'CAUSE': ['لذلك', 'وبالتالي', 'لهذا السبب', 'بسبب', 'نتيجة'],
            'CONTRAST': ['لكن', 'غير أن', 'ومع ذلك', 'بالرغم', 'بينما', 'إلا أن'], 'ELABORATION': ['أي', 'يعني', 'بمعنى'],
            'EXAMPLE': ['مثل', 'على سبيل المثال', 'مثلا'], 'CONDITION': ['إذا', 'لو', 'إن'], 'TEMPORAL': ['عندما', 'حين', 'قبل', 'بعد'],
        }
        first_token = current_tokens[0]
        for relation_type, markers in discourse_markers.items():
            if first_token in markers:
                discourse_relations.append({'type': relation_type, 'marker': first_token})
                break
        pronouns = ['هذا', 'ذلك', 'تلك', 'هذه']
        if first_token in pronouns and len(current_tokens) > 1 and current_tokens[1] in ['الأمر', 'الشيء', 'الحدث', 'الفكرة', 'القول']:
             relation = {'type': 'REFERENCE', 'marker': f"{first_token} {current_tokens[1]}"}
             if relation not in discourse_relations: discourse_relations.append(relation)
        return discourse_relations


    # --- Classification Evaluation (Unchanged) ---
    def evaluate_classification_accuracy(self,
                                         labels: List[Any],
                                         embedding_type: str = 'quantum',
                                         test_size: float = 0.3,
                                         random_state: int = 42) -> Optional[Dict[str, Dict]]:
        if not SKLEARN_AVAILABLE: logger.error("scikit-learn not available."); return None
        embeddings_dict = self.circuit_embeddings if embedding_type == 'quantum' else self.sentence_embeddings
        if not embeddings_dict: logger.error(f"No {embedding_type} embeddings."); return None
        
        X_list, y_list = [], []
        for idx, embedding_vec in embeddings_dict.items():
            if idx < len(labels):
                label_for_idx = labels[idx]
                if embedding_vec is not None and np.all(np.isfinite(embedding_vec)) and \
                   label_for_idx is not None and isinstance(label_for_idx, (str, int, float, bool)):
                    X_list.append(embedding_vec); y_list.append(label_for_idx)
                else: logger.warning(f"Skipping index {idx} for classification (invalid data/label).")
        
        if len(X_list) < 2 or len(y_list) < 2:
             logger.error(f"Not enough valid samples for classification ({len(X_list)})."); return None
        X, y = np.array(X_list), np.array(y_list); num_classes = len(set(y))
        if num_classes < 2 and X.shape[0] > 0:
            logger.warning("Only one distinct class. Classification metrics may be trivial."); X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=min(test_size, 0.5 if len(y)>1 else 0.0), random_state=random_state)
        elif num_classes < 2: logger.error("Need >=2 classes and samples."); return None
        else:
            try: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            except ValueError: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if not X_train.size or not X_test.size: logger.error("Train/test set empty."); return None
        
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
        classifiers = {"GaussianNB": GaussianNB(), "SVC": SVC(random_state=random_state)} # type: ignore
        results_summary = {}
        for name, clf in classifiers.items():
            try:
                clf.fit(X_train_scaled, y_train); y_pred = clf.predict(X_test_scaled) # type: ignore
                accuracy = accuracy_score(y_test, y_pred); report = classification_report(y_test, y_pred, zero_division=0) # type: ignore
                results_summary[name] = {'classifier': name, 'accuracy': accuracy, 'report': report, 'num_samples_train': len(X_train), 'num_samples_test': len(X_test), 'num_classes': num_classes, 'embedding_type_used': embedding_type}
            except Exception as e: results_summary[name] = {'classifier': name, 'error': str(e), 'embedding_type_used': embedding_type}
        return results_summary if results_summary else None

        # --- End of evaluate_classification_accuracy ---


    # --- Visualization (Modified to plot specified embeddings) ---
    def _generate_pca_plots_for_current_set(self, embedding_type_to_plot: str = 'quantum', per_category_plots: bool = True):
        """
        Generates PCA plots for the specified embedding type ('quantum', 'combined', 'linguistic').
        Can generate an overall plot for the set and/or plots for each category within the set.
        """
        # <<< MODIFICATION >>> Select embedding dictionary based on type
        if embedding_type_to_plot == 'quantum':
            embeddings_dict = self.circuit_embeddings
        elif embedding_type_to_plot == 'combined':
            embeddings_dict = self.sentence_embeddings
        elif embedding_type_to_plot == 'linguistic':
            embeddings_dict = self.linguistic_features
        else:
            logger.error(f"[{self.current_set_name}] Invalid embedding_type_to_plot: '{embedding_type_to_plot}'. Choose 'quantum', 'combined', or 'linguistic'.")
            return

        if not embeddings_dict:
            logger.warning(f"[{self.current_set_name}] No '{embedding_type_to_plot}' embeddings available to generate PCA plot from.")
            return

        logger.info(f"--- [{self.current_set_name}] Generating PCA Plot(s) (using {embedding_type_to_plot} embeddings) ---")

        all_embeddings_in_set = []
        all_category_labels_in_set = [] # Labels like 'WordOrder_SVO'
        all_sentence_ids_in_set = []

        # Collect all valid embeddings and labels from the current set's results
        # We need the labels from final_results_for_set, but embeddings from the kernel dict
        valid_indices = sorted(embeddings_dict.keys()) # Use indices from the chosen embedding dict

        for idx in valid_indices:
            emb_vec = embeddings_dict.get(idx)
            # Find the corresponding result item in final_results_for_set
            res_item = next((item for item in self.final_results_for_set if item.get('original_index') == idx), None)

            if emb_vec is not None and hasattr(emb_vec, '__len__') and len(emb_vec) > 0 and res_item is not None:
                all_embeddings_in_set.append(emb_vec)
                all_category_labels_in_set.append(res_item.get('label', 'N/A_Label')) # Get label from result item
                all_sentence_ids_in_set.append(idx) # Use the index as ID
            else:
                logger.debug(f"[{self.current_set_name}/{idx}] Missing or empty {embedding_type_to_plot} embedding or result item. Skipping for PCA.")

        # Generate overall PCA plot for the entire set
        if len(all_embeddings_in_set) < 2:
            logger.warning(f"[{self.current_set_name}] Could not generate overall PCA plot (type: {embedding_type_to_plot}): "
                        f"Not enough valid embeddings found (got {len(all_embeddings_in_set)}). Need at least 2.")
        else:
            self._create_single_pca_plot( # Call the helper function
                embeddings_list=all_embeddings_in_set,
                labels_list=all_category_labels_in_set, # Pass the full labels
                ids_list=all_sentence_ids_in_set,
                embedding_type=embedding_type_to_plot,
                plot_title_suffix=f"Overall Set - {self.current_set_name}",
                save_filename_suffix=f"overall_{sanitize_filename_v8(self.current_set_name)}",
                output_subdir=self.output_dir_set # Save in the main set output directory
            )

        # Generate Per-Category PCA Plots if enabled and data exists
        if per_category_plots and len(all_embeddings_in_set) >= 2 :
            logger.info(f"--- [{self.current_set_name}] Generating Per-Category PCA Plots (using {embedding_type_to_plot} embeddings) ---")
            # Group embeddings/labels by the category part of the label (e.g., 'WordOrder')
            categories_data = {}
            for i, full_label in enumerate(all_category_labels_in_set):
                category_name = full_label.split('_')[0] if '_' in full_label else full_label # Extract category name
                if category_name not in categories_data:
                    categories_data[category_name] = {'embeddings': [], 'labels': [], 'ids': []}
                categories_data[category_name]['embeddings'].append(all_embeddings_in_set[i])
                categories_data[category_name]['labels'].append(full_label) # Store the full label for coloring
                categories_data[category_name]['ids'].append(all_sentence_ids_in_set[i])

            for category_name, data in categories_data.items():
                cat_embeddings = data['embeddings']
                cat_labels = data['labels'] # Use the full labels for coloring within the category plot
                cat_ids = data['ids']

                if len(cat_embeddings) < 2:
                    logger.warning(f"[{self.current_set_name} - Category: {category_name}] Not enough valid embeddings "
                                f"(found {len(cat_embeddings)}) for PCA plot (type: {embedding_type_to_plot}). Need at least 2.")
                    continue

                self._create_single_pca_plot( # Call the helper function
                    embeddings_list=cat_embeddings,
                    labels_list=cat_labels, # Pass full labels for coloring
                    ids_list=cat_ids,
                    embedding_type=embedding_type_to_plot,
                    plot_title_suffix=f"Category: {mpl_sanitize(category_name)} - Set: {self.current_set_name}",
                    save_filename_suffix=f"category_{sanitize_filename_v8(category_name)}_{sanitize_filename_v8(self.current_set_name)}",
                    output_subdir=self.pca_per_category_path # Save in the dedicated subdir
                )


    # --- PCA Plotting Helper (Unchanged) ---
    def _create_single_pca_plot(self, embeddings_list: List[np.ndarray], labels_list: List[str], ids_list: List[int], # Changed ID type
                                embedding_type: str, plot_title_suffix: str, save_filename_suffix: str, output_subdir: str):
        """ Helper function to create a single PCA plot. """
        s_context = f"[{self.current_set_name} - {plot_title_suffix}]"
        try:
            X_embeddings = np.array(embeddings_list)
            if X_embeddings.ndim == 1:
                logger.warning(f"{s_context} Embeddings for PCA are 1D. Reshaping.")
                X_embeddings = X_embeddings.reshape(-1, 1)

            if X_embeddings.shape[1] == 0 : # No features
                logger.warning(f"{s_context} Embeddings have 0 features. Cannot perform PCA.")
                return

            # Check for non-finite values before scaling
            finite_mask = np.all(np.isfinite(X_embeddings), axis=1)
            if not np.all(finite_mask):
                num_removed = np.sum(~finite_mask)
                logger.warning(f"{s_context} Removing {num_removed} non-finite rows before PCA.")
                X_embeddings = X_embeddings[finite_mask]
                # Filter corresponding labels and IDs
                original_labels = labels_list
                original_ids = ids_list
                labels_list = [label for i, label in enumerate(original_labels) if finite_mask[i]]
                ids_list = [id_val for i, id_val in enumerate(original_ids) if finite_mask[i]]
                if X_embeddings.shape[0] < 2:
                    logger.warning(f"{s_context} Not enough finite samples ({X_embeddings.shape[0]}) for PCA after filtering.")
                    return

            if X_embeddings.shape[0] < 2:
                logger.warning(f"{s_context} Not enough valid embeddings ({X_embeddings.shape[0]}) for PCA. Need at least 2.")
                return

            logger.info(f"{s_context} Performing PCA on {X_embeddings.shape[0]} samples, {X_embeddings.shape[1]} features ({embedding_type} embeddings).")
            X_scaled = StandardScaler().fit_transform(X_embeddings)
            n_pca_components = min(2, X_scaled.shape[0], X_scaled.shape[1])

            if n_pca_components < 1: # Need at least 1 component
                logger.warning(f"{s_context} Cannot perform PCA (target n_components={n_pca_components}). Skipping plot.")
                return

            pca_model = PCA(n_components=n_pca_components)
            X_pca_transformed = pca_model.fit_transform(X_scaled)

            plt.figure(figsize=(12, 10))
            # Use the passed labels_list (which might be full labels) for coloring
            unique_cats_in_plot = sorted(list(set(labels_list)))
            colors = plt.cm.get_cmap('tab10', len(unique_cats_in_plot)) if len(unique_cats_in_plot) <= 10 else plt.cm.get_cmap('viridis', len(unique_cats_in_plot))

            for i, category_val in enumerate(unique_cats_in_plot):
                indices = [j for j, lbl in enumerate(labels_list) if lbl == category_val]
                if not indices: continue

                pc1_data = X_pca_transformed[indices, 0]
                # Handle case where only 1 component is available
                pc2_data = X_pca_transformed[indices, 1] if X_pca_transformed.shape[1] > 1 else np.zeros_like(pc1_data)

                # Use the potentially full label for the legend
                legend_label = mpl_sanitize(str(category_val))
                plt.scatter(pc1_data, pc2_data,
                            label=legend_label, color=colors(i), alpha=0.75, s=60)

            title_str = (f"PCA: {plot_title_suffix}\n"
                        f"({embedding_type.capitalize()} Emb, Ansatz: {self.ansatz_choice})")
            plt.title(mpl_sanitize(title_str), fontsize=15)

            # Add explained variance info if available
            if hasattr(pca_model, 'explained_variance_ratio_') and pca_model.explained_variance_ratio_ is not None and len(pca_model.explained_variance_ratio_) > 0:
                plt.xlabel(f"PC 1 (Expl. Var: {pca_model.explained_variance_ratio_[0]:.2%})", fontsize=12)
                if n_pca_components > 1 and len(pca_model.explained_variance_ratio_) > 1:
                    plt.ylabel(f"PC 2 (Expl. Var: {pca_model.explained_variance_ratio_[1]:.2%})", fontsize=12)
                elif n_pca_components == 1:
                    plt.ylabel(" (Single Component)", fontsize=12)
            else: # Fallback if explained_variance_ratio_ is not available
                plt.xlabel("Principal Component 1", fontsize=12)
                if n_pca_components > 1: plt.ylabel("Principal Component 2", fontsize=12)

            # Adjust legend placement
            if len(unique_cats_in_plot) > 1:
                plt.legend(title=mpl_sanitize("Labels"), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust rect to make space for legend
            else:
                plt.tight_layout()

            pca_plot_filename = f"pca_plot_{embedding_type}_{save_filename_suffix}.png"
            save_path = os.path.join(output_subdir, pca_plot_filename)
            os.makedirs(output_subdir, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"{s_context} PCA plot for {embedding_type} embeddings saved to {save_path}")

        except Exception as e_pca_single:
            logger.error(f"{s_context} Error generating single PCA plot for {embedding_type} embeddings: {e_pca_single}", exc_info=True)
            if plt.gcf().get_axes(): plt.close() # Close figure if error occurred during plotting
        # --- End of _create_single_pca_plot ---


    # --- Quantum State Visualization (Unchanged) ---
    def analyze_quantum_states(self, circuits_dict, tokens_dict, analyses_dict, save_path_prefix=None):
        """ Analyze and visualize the quantum states of circuits. """
        if not QISKIT_AVAILABLE or self.simulator is None:
            logger.error("Qiskit/Simulator not available for state analysis.")
            return None
        if not circuits_dict: logger.warning("No circuits for state analysis."); return None
        logger.info(f"Analyzing quantum states for {len(circuits_dict)} circuits...")
        num_circuits = len(circuits_dict); ncols = 3; nrows = (num_circuits + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False); axes = axes.flatten()
        i = 0; processed_ids = set()
        for circuit_id, circuit in circuits_dict.items():
            if i >= len(axes) or circuit_id in processed_ids: continue
            ax = axes[i]; processed_ids.add(circuit_id)
            original_sentence = self.reference_sentences.get(circuit_id, f"Circuit {circuit_id}")
            sentence_label = shape_arabic_text(original_sentence);
            if len(sentence_label) > 40: sentence_label = sentence_label[:37] + "..."
            try:
                if not isinstance(circuit, QuantumCircuit): logger.warning(f"Skipping {circuit_id}: Not a QCircuit"); i += 1; continue
                tokens = tokens_dict.get(circuit_id, []); analyses = analyses_dict.get(circuit_id, [])
                parameter_binds = self._bind_parameters(circuit, tokens, analyses);
                # Estimator uses parameter_values, Simulator uses parameter_binds
                bindings_list = [parameter_binds] if parameter_binds else None

                # Use simulator for statevector
                job = self.simulator.run(circuit, parameter_binds=bindings_list, shots=1) # Use 1 shot for statevector
                result = job.result()

                if not result.success: logger.error(f"Statevector sim failed for {circuit_id}. Status: {getattr(result, 'status', 'N/A')}"); i += 1; continue
                if hasattr(result, 'get_statevector'):
                    statevector = result.get_statevector();
                    if plot_state_city: plot_state_city(statevector, title=sentence_label, ax=ax); ax.tick_params(axis='both', which='major', labelsize=8); ax.title.set_size(10)
                    else: ax.text(0.5, 0.5, shape_arabic_text("العرض معطل"), ha='center', va='center', fontsize=9)
                    if save_path_prefix:
                         try:
                              ind_fig = plt.figure(figsize=(8, 6));
                              if plot_state_city: plot_state_city(statevector, title=f"{shape_arabic_text('الحالة:')} {sentence_label}", fig=ind_fig)
                              else: # Add fallback text if plot_state_city is None
                                   ax_ind = ind_fig.add_subplot(111)
                                   ax_ind.text(0.5, 0.5, shape_arabic_text("العرض معطل"), ha='center', va='center', fontsize=12)
                                   ax_ind.set_title(f"{shape_arabic_text('الحالة:')} {sentence_label}")
                                   ax_ind.set_xticks([]); ax_ind.set_yticks([])
                              ind_path = f"{save_path_prefix}circuit_{circuit_id}_state.png"; ind_fig.savefig(ind_path, dpi=150, bbox_inches='tight'); plt.close(ind_fig)
                         except Exception as save_e: logger.warning(f"Failed to save individual state plot {circuit_id}: {save_e}")
                else: ax.text(0.5, 0.5, shape_arabic_text("الحالة غير متوفرة"), ha='center', va='center', fontsize=9); ax.set_title(sentence_label, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            except Exception as e: logger.error(f"Error visualizing circuit {circuit_id}: {e}", exc_info=True);
            i += 1
        for j in range(i, len(axes)): fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.suptitle(shape_arabic_text("تصورات الحالة الكمومية"), fontsize=16)
        if save_path_prefix:
            try: overview_path = f"{save_path_prefix}all_states_overview.png"; fig.savefig(overview_path, dpi=150); logger.info(f"Saved states overview to {overview_path}")
            except Exception as overview_save_e: logger.warning(f"Failed to save overview state plot: {overview_save_e}")
        # Return the figure object so it can be displayed or closed by the caller
        # return fig # <<< MODIFICATION >>> Uncomment if you want to return the figure
        plt.close(fig) # Close the figure here if not returning it
        # --- End of analyze_quantum_states ---


    # --- Reporting (Modified to show both embeddings potentially) ---
    def generate_html_report(self, analysis_results_list: List[Optional[Dict[str, Any]]],
                            report_title: str = "QNLP Analysis Report",
                            current_set_name: Optional[str] = None) -> str:
        # Helper for Arabic text display (ensure _reshaper_display is defined in your class)
        # def _reshaper_display(text):
        #     if not text or not ARABIC_DISPLAY_ENABLED: return str(text)
        #     try:
        #         return get_display(arabic_reshaper.reshape(str(text)))
        #     except Exception:
        #         return str(text)

        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{self._reshaper_display(report_title)}</title>",
            "<script src='https://cdn.tailwindcss.com'></script>",
            "<style>",
            "body { font-family: 'Arial', sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }",
            ".container { max-width: 95%; margin: 20px auto; padding: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }",
            "h1, h2 { color: #333; text-align: center; }",
            "table { width: 100%; border-collapse: collapse; margin-top: 20px; table-layout: fixed; }",
            "th, td { border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; word-wrap: break-word; }",
            "th { background-color: #f0f0f0; font-weight: bold; }",
            "td.rtl-text { direction: rtl; font-family: ' Amiri', 'Noto Naskh Arabic', 'Tahoma', serif; font-size: 1.1em; }",
            "img.diagram, img.circuit { max-width: 100%; height: auto; display: block; margin: 5px auto; border: 1px solid #eee; border-radius: 4px; }",
            ".error-message { color: #D8000C; background-color: #FFD2D2; padding: 8px; border-radius: 4px; margin: 5px 0; }",
            ".linguistic-stream ul { list-style-type: none; padding-left: 0; }",
            ".linguistic-stream li { margin-bottom: 8px; padding: 5px; border-left: 3px solid #007bff; background-color: #f9f9f9; }",
            ".linguistic-stream b { color: #0056b3; }",
            ".status-missing { color: #e85d04; font-style: italic; }",
            ".status-error { color: #d00000; font-weight: bold; }",
            "</style>",
            "</head>",
            "<body>",
            f"<div class='container'><h1>{self._reshaper_display(report_title)}</h1>"
        ]
        if current_set_name:
            html_parts.append(f"<h2>Set: {self._reshaper_display(current_set_name)}</h2>")

        html_parts.append("<div class='overflow-x-auto'><table class='min-w-full divide-y divide-gray-200'>")
        html_parts.append("<thead class='bg-gray-50'><tr>")
        headers = ["#", "Sentence (النص)", "Sense ID", "Score", "Diagram", "Circuit", "Linguistic Stream"]
        col_widths = ["w-1/12", "w-3/12", "w-1/12", "w-1/12", "w-2/12", "w-2/12", "w-2/12"] # Adjusted for 7 columns
        for header, width in zip(headers, col_widths):
            html_parts.append(f"<th class='px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider {width}'>{header}</th>")
        html_parts.append("</tr></thead><tbody class='bg-white divide-y divide-gray-200'>")

        if not analysis_results_list:
            html_parts.append(f"<tr><td colspan='{len(headers)}' class='text-center p-4 text-gray-500'>No analysis results to display.</td></tr>")
        else:
            for idx, interpretation_data in enumerate(analysis_results_list):
                html_parts.append("<tr>")
                html_parts.append(f"<td class='border p-2 text-center align-top text-sm'>{idx + 1}</td>")

                # --- ADDED BLOCK TO HANDLE POTENTIAL None ITEMS and ensure robust .get() ---
                if interpretation_data is None:
                    logger.warning(f"Item {idx} in analysis_results_list is None. Skipping for HTML report.")
                    sentence_text_display = "N/A"
                    sense_id_display = "N/A"
                    sense_score_display = "N/A"
                    error_msg_display = "<div class='error-message'>Error: Analysis data for this item was corrupted or missing.</div>"
                    diagram_html_display = error_msg_display
                    circuit_html_display = error_msg_display
                    linguistic_html_display = error_msg_display
                    
                    html_parts.append(f"<td class='border p-2 align-top text-right font-arabic rtl-text text-sm'>{sentence_text_display}</td>")
                    html_parts.append(f"<td class='border p-2 align-top text-center text-sm'>{sense_id_display}</td>")
                    html_parts.append(f"<td class='border p-2 align-top text-center text-sm'>{sense_score_display}</td>")
                    html_parts.append(f"<td colspan='3' class='border p-2 align-top text-center text-sm'>{error_msg_display}</td>")

                else:
                    sentence_text = interpretation_data.get('sentence_text', 'N/A')
                    sentence_text_display = self._reshaper_display(sentence_text)
                    
                    sense_id = interpretation_data.get('sense_id', 'N/A')
                    sense_id_display = self._reshaper_display(str(sense_id))

                    sense_score = interpretation_data.get('sense_score', 'N/A')
                    if isinstance(sense_score, float):
                        sense_score_display = f"{sense_score:.4f}"
                    else:
                        sense_score_display = str(sense_score)

                    error_msg = interpretation_data.get('error')

                    html_parts.append(f"<td class='border p-2 align-top text-right font-arabic rtl-text text-sm'>{sentence_text_display}</td>")
                    html_parts.append(f"<td class='border p-2 align-top text-center text-sm'>{sense_id_display}</td>")
                    html_parts.append(f"<td class='border p-2 align-top text-center text-sm'>{sense_score_display}</td>")

                    if error_msg:
                        error_msg_display = f"<div class='error-message'><strong>Error:</strong> {self._reshaper_display(str(error_msg))}</div>"
                        html_parts.append(f"<td colspan='3' class='border p-2 align-top text-center text-sm'>{error_msg_display}</td>")
                    else:
                        diagram_html_display = interpretation_data.get('diagram_image_html', "<i class='status-missing'>Diagram not available</i>")
                        circuit_html_display = interpretation_data.get('circuit_draw_html', "<i class='status-missing'>Circuit not available</i>")
                        linguistic_html_display = interpretation_data.get('linguistic_stream_html', "<i class='status-missing'>Linguistic stream not available</i>")
                        
                        html_parts.append(f"<td class='border p-2 align-top text-center text-sm'>{diagram_html_display}</td>")
                        html_parts.append(f"<td class='border p-2 align-top text-center text-sm'>{circuit_html_display}</td>")
                        html_parts.append(f"<td class='border p-2 align-top text-xs text-left leading-tight linguistic-stream'>{linguistic_html_display}</td>")
                # --- END OF MODIFIED BLOCK ---
                html_parts.append("</tr>")

        html_parts.append("</tbody></table></div></div></body></html>")
        return "".join(html_parts)

    # In v8.py
    # Update or replace the existing format_linguistic_stream_for_html method 
    # (or the logic that performs this task) in your ArabicQuantumMeaningKernel class

    def format_linguistic_stream_for_html(self, linguistic_stream_list: Optional[List[Dict[str, Any]]]) -> str:
        if not linguistic_stream_list:
            return "<i class='status-missing'>No linguistic stream data available.</i>"

        # Helper for Arabic text display (ensure _reshaper_display is defined in your class)
        # def _reshaper_display(text): (same as above)

        html_parts = ["<ul class='list-none p-0 m-0'>"] # Changed from list-disc for cleaner look
        for i, word_data in enumerate(linguistic_stream_list):
            if word_data is None:
                html_parts.append("<li class='mb-2 p-1 border-l-2 border-red-500 bg-red-50'><i class='status-error'>Corrupted word data entry.</i></li>")
                continue

            surface = self._reshaper_display(word_data.get('word_surface', 'N/A'))
            lemma = self._reshaper_display(word_data.get('word_lemma', 'N/A'))
            pos = word_data.get('pos_tag', 'N/A')
            
            core_rep = word_data.get('core_linguistic_representation')
            core_html = ""
            if isinstance(core_rep, dict) and core_rep.get("status") == "missing":
                reason = self._reshaper_display(core_rep.get('reason', ''))
                core_html = f"<span class='status-missing'>Core: Missing ({self._reshaper_display(reason)})</span>"
            elif core_rep is not None:
                core_html = f"Core: {self._reshaper_display(str(core_rep))}"
            else:
                core_html = "<span class='status-missing'>Core: Not Provided</span>"

            morph_features = word_data.get('morph_features', {})
            morph_feat_str = ""
            if isinstance(morph_features, dict) and morph_features:
                morph_feat_str = ", ".join([f"{k}:{self._reshaper_display(str(v))}" for k,v in morph_features.items()])
            elif isinstance(morph_features, str): # Handle if it's already a string
                morph_feat_str = self._reshaper_display(morph_features)
            else:
                morph_feat_str = "N/A"
            
            box_str = word_data.get('box_str', 'N/A')
            box_type_info = word_data.get('box_type', {})
            box_type_str = f"Dom: {box_type_info.get('dom', 'N/A')}, Cod: {box_type_info.get('cod', 'N/A')}"

            html_parts.append(
                f"<li class='mb-2 p-2 border-l-2 border-blue-500 bg-blue-50 rounded-r-md'>"
                f"<b>{surface}</b> (L: {lemma}, P: {pos})<br/>"
                f"<span class='text-gray-700 text-xs'>&nbsp;&nbsp;M: {morph_feat_str}</span><br/>"
                f"<span class='text-gray-700 text-xs'>&nbsp;&nbsp;{core_html}</span><br/>"
                f"<span class='text-gray-500 text-xs'>&nbsp;&nbsp;Box: {box_str} ({box_type_str})</span>"
                f"</li>"
            )
        html_parts.append("</ul>")
        return "".join(html_parts)


    # --- Utility Methods (Save/Load - Unchanged) ---
    def save_model(self, filename: str = 'arabic_quantum_kernel_v8.pkl'):
        """ Saves the kernel state (embeddings, config) to a file. V8 """
        logger.info(f"Saving kernel state (v8) to {filename}...")
        model_data = {
            'embedding_dim': self.embedding_dim, 'num_clusters': self.num_clusters,
            'combine_feature_weight': self.combine_feature_weight, 'shots': self.shots,
            '_embedding_model_path': getattr(self, '_embedding_model_path', None),
            'ansatz_choice': self.ansatz_choice, # Store ansatz info
            'n_layers_iqp': self.n_layers_iqp,
            'config_n_layers_iqp': self.config_n_layers_iqp,
            'config_n_single_qubit_params_iqp': self.config_n_single_qubit_params_iqp,
            'config_n_layers_strong': self.config_n_layers_strong,
            'n_layers_strong': self.n_layers_strong,
            'reference_sentences': self.reference_sentences,
            'circuit_embeddings': self.circuit_embeddings, # Save quantum embeddings
            'sentence_embeddings': self.sentence_embeddings, # Save combined embeddings
            'linguistic_features': self.linguistic_features, # <<< MODIFICATION >>> Save linguistic
            'kmeans_model': self.kmeans_model, # Save fitted KMeans model
            'cluster_labels': self.cluster_labels, 'meaning_map': self.meaning_map,
            'classification_results_for_set': self.classification_results_for_set, # <<< MODIFICATION >>> Save classification results
            'embedding_type_for_classification': self.embedding_type_for_classification, # <<< MODIFICATION >>> Save classification type
        }
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f: pickle.dump(model_data, f)
            logger.info(f"Kernel state saved successfully to {filename}")
        except Exception as e: logger.error(f"Error saving kernel state: {e}", exc_info=True)

    def load_model(self, filename: str = 'arabic_quantum_kernel_v8.pkl'):
        """ Loads a kernel state from a file. V8 """
        if not os.path.exists(filename): logger.error(f"Error: Kernel state file {filename} not found."); return self
        logger.info(f"Loading kernel state (v8) from {filename}...")
        try:
            with open(filename, 'rb') as f: model_data = pickle.load(f)
            saved_model_path = model_data.get('_embedding_model_path')
            # Re-initialize with saved parameters
            self.__init__(
                embedding_dim=model_data.get('embedding_dim', 30),
                num_clusters=model_data.get('num_clusters', 5),
                embedding_model_path=saved_model_path,
                ansatz_choice=model_data.get('ansatz_choice', 'IQP'),
                n_layers_iqp=model_data.get('n_layers_iqp', 2),
                n_layers_strong=model_data.get('n_layers_strong', 2),
                # output_dir_set will be set by the caller pipeline
                embedding_type_for_classification=model_data.get('embedding_type_for_classification', 'quantum'),
                #params_per_word=model_data.get('params_per_word', 3),
                shots=model_data.get('shots', 8192),
                morph_param_weight=model_data.get('morph_param_weight', 0.3),
                combine_feature_weight=model_data.get('combine_feature_weight', 0.5)
            )
            # Load state
            self.reference_sentences = model_data.get('reference_sentences', {})
            self.circuit_embeddings = model_data.get('circuit_embeddings', {})
            self.sentence_embeddings = model_data.get('sentence_embeddings', {})
            self.linguistic_features = model_data.get('linguistic_features', {}) # <<< MODIFICATION >>> Load linguistic
            self.kmeans_model = model_data.get('kmeans_model')
            self.cluster_labels = model_data.get('cluster_labels')
            self.meaning_map = model_data.get('meaning_map', {})
            self.classification_results_for_set = model_data.get('classification_results_for_set') # <<< MODIFICATION >>> Load classification results

            logger.info(f"Kernel state loaded successfully.")
        except Exception as e: logger.error(f"Error loading kernel state: {e}", exc_info=True)
        return self

    # --- Verb Classification Helper (Unchanged) ---
    def _classify_verb(self, verb_lemma: Optional[str]) -> str:
        """ Classify verb lemma into semantic categories. """
        if not verb_lemma: return "UNKNOWN"
        motion_verbs = ["ذهب", "جاء", "مشى", "سار", "رجع", "دخل", "خرج", "وصل", "سافر", "عاد"]
        possession_verbs = ["أخذ", "أعطى", "ملك", "وهب", "منح", "سلم", "اشترى", "باع", "امتلك", "حاز"]
        communication_verbs = ["قال", "تكلم", "صرح", "أخبر", "سأل", "أجاب", "حدث", "نادى", "أعلن", "ذكر", "خاطب"]
        cognition_verbs = ["فكر", "اعتقد", "ظن", "علم", "فهم", "نسي", "عرف", "تذكر", "درس"]
        emotion_verbs = ["أحب", "كره", "خاف", "فرح", "حزن", "شعر", "غضب"]
        creation_verbs = ["بنى", "كتب", "صنع", "خلق", "أنشأ", "رسم"]
        perception_verbs = ["رأى", "سمع", "نظر", "شاهد", "لمس", "شم"]
        if verb_lemma in motion_verbs: return "MOTION"
        if verb_lemma in possession_verbs: return "POSSESSION"
        if verb_lemma in communication_verbs: return "COMMUNICATION"
        if verb_lemma in cognition_verbs: return "COGNITION"
        if verb_lemma in emotion_verbs: return "EMOTION"
        if verb_lemma in creation_verbs: return "CREATION"
        if verb_lemma in perception_verbs: return "PERCEPTION"
        return "ACTION"

    # --- Discourse Formatting Helper (Unchanged) ---
    def format_discourse_relations(self, discourse_relations):
        """ Creates a user-friendly description of discourse relations. """
        if not discourse_relations: return shape_arabic_text("لم يتم اكتشاف علاقات خطاب محددة.") # Shape default msg
        formatted_output = []
        descriptions = {
            'CONTINUATION': "تواصل هذه الجملة الفكرة السابقة باستخدام '{}'", 'CAUSE': "تظهر هذه الجملة نتيجة أو عاقبة للجملة السابقة باستخدام '{}'",
            'CONTRAST': "تتناقض هذه الجملة مع المعلومات السابقة باستخدام '{}'", 'ELABORATION': "توضح هذه الجملة المعلومات السابقة باستخدام '{}'",
            'EXAMPLE': "تقدم هذه الجملة مثالاً على المفهوم السابق باستخدام '{}'", 'CONDITION': "تحدد هذه الجملة شرطًا متعلقًا بالجملة السابقة باستخدام '{}'",
            'TEMPORAL': "تحدد هذه الجملة علاقة زمنية مع الجملة السابقة باستخدام '{}'", 'REFERENCE': "تشير هذه الجملة إلى المحتوى السابق باستخدام '{}'"
        }
        for relation in discourse_relations:
            rel_type = relation.get('type', 'UNKNOWN'); marker = relation.get('marker', '')
            desc_template = descriptions.get(rel_type, f"تم اكتشاف علاقة من نوع {rel_type} باستخدام '{marker}'")
            formatted_output.append(shape_arabic_text(desc_template.format(shape_arabic_text(marker))))
        return "\n".join(formatted_output) # Already shaped

    def _reshaper_display(self, text: Any) -> str:
        """Helper method to correctly reshape Arabic text for display."""
        if not text or not ARABIC_DISPLAY_ENABLED:
            return str(text)
        try:
            return get_display(arabic_reshaper.reshape(str(text)))
        except Exception:
            # Fallback if reshaping fails for any reason
            return str(text)

# ============================================
# Example Pipeline Function (V8 - Combined Approach)
# ============================================
def prepare_quantum_nlp_pipeline_v8(
    sentences_for_current_set: List[Dict[str, Any]], # Input: list of {'sentence': str, 'label': str}
    current_set_name: str,
    max_sentences_to_process: Optional[int] = None,
    embedding_model_path: Optional[str] = None,
    ansatz_choice: str = 'IQP', # This is for main_sentence_processor (camel_test2)
    n_layers_iqp: int = 2,
    n_layers_strong: int = 2,
    # cnot_ranges for main_sentence_processor if it uses StronglyEntanglingAnsatz
    cnot_ranges: Optional[List[Tuple[int, int]]] = None, 
    run_clustering_viz: bool = True, run_state_viz: bool = False, run_classification: bool = True,
    output_dir_set: str = "qnlp_pipeline_output_default_set",
    embedding_type_for_classification: str = 'quantum',
    debug_circuits: bool = False, # For main_sentence_processor and core module debug
    generate_per_category_pca_plots: bool = True,
    classical_feature_dim_config_for_core: int = 16, # Dim for core module's classical features
    max_senses_from_core: int = 1, # How many senses from core module
    morpho_lex_enhanced_sets: List[str] = ["MorphologyTestSet", "LexicalTestSet", "Morphology"],
    handle_lexical_ambiguity_conditionally: bool = True, # NEW: Master switch for this feature
    lexical_ambiguity_sets: List[str] = ["LexicalAmbiguity"] # Sets where ambiguity handling is active
    ) -> Tuple[Optional['ArabicQuantumMeaningKernel'], List[Dict[str, Any]], Optional[Dict[str, Any]]]:

    # Create output directory if it doesn't exist
    os.makedirs(output_dir_set, exist_ok=True)

    activate_ambiguity_typing_for_this_set = (
        handle_lexical_ambiguity_conditionally and
        current_set_name in lexical_ambiguity_sets
    )
    logger.info(f"[{current_set_name}] Lexical ambiguity explicit typing active: {activate_ambiguity_typing_for_this_set}")

    if not MAIN_PROCESSOR_AVAILABLE:
        logger.critical(f"[{current_set_name}] Main sentence processor (from camel_test2.py) not available. Exiting.")
        return None, [], None
    
    use_core_enhancements = CORE_MODULE_AVAILABLE and current_set_name in morpho_lex_enhanced_sets
    logger.info(f"Current set: '{current_set_name}'. Enhanced sets: {morpho_lex_enhanced_sets}. Using core enhancements: {use_core_enhancements}")
    logger.info(f"\n--- [{current_set_name}] Pipeline V8.3 --- Core Enhancements Active: {use_core_enhancements} ---")

    # Data lists for kernel training (dense lists, for items successfully processed up to kernel.train)
    processed_original_indices_for_kernel: List[int] = [] 
    kernel_train_sentences_str: List[str] = []
    kernel_train_original_labels: List[str] = [] 
    
    kernel_train_circuits_main: List[QuantumCircuit] = []
    kernel_train_diagrams_main: List[Optional[GrammarDiagram]] = []
    kernel_train_tokens_main: List[List[str]] = []
    kernel_train_analyses_main: List[List[Dict]] = []
    kernel_train_structures_main: List[str] = []
    kernel_train_roles_main: List[Dict] = []
    kernel_train_linguistic_streams_core: List[Optional[List[Dict[str, Any]]]] = []
    kernel_train_assigned_elements_main: List[List[Any]] = []  # FIXED: Initialize this list
    
    # For final reporting: list of dicts, one per *original input sentence*
    # Initialize with basic info for all input sentences.
    analysis_results_list_for_report: List[Dict[str, Any]] = [
        {'sentence': s_data.get("sentence"), 'label': s_data.get("label"), 
         'original_index': idx, 'error': None, 'interpretation': None, 
         'circuit_representation_str': None, 'circuit_representation_type': None}
        for idx, s_data in enumerate(sentences_for_current_set)
    ]

    sentences_to_process_input = sentences_for_current_set
    if max_sentences_to_process is not None and 0 < max_sentences_to_process < len(sentences_to_process_input):
        sentences_to_process_input = sentences_to_process_input[:max_sentences_to_process]
        # Adjust analysis_results_list_for_report if input is truncated
        analysis_results_list_for_report = analysis_results_list_for_report[:max_sentences_to_process]

    ansatz_functor_for_core = None
    if use_core_enhancements and CORE_MODULE_AVAILABLE and LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        if N and S and ROOT_TYPE and isinstance(N, Ty):
            ob_map_core = {N: 1, S: 1, ROOT_TYPE: 1}
            try:
                # Use the same ansatz choice and layers for consistency, but on core_types
                if ansatz_choice.upper() == 'IQP': 
                    ansatz_functor_for_core = IQPAnsatz(ob_map_core, n_layers=n_layers_iqp, n_single_qubit_params=n_layers_iqp*2)
                elif ansatz_choice.upper() == 'SPIDER': 
                    ansatz_functor_for_core = SpiderAnsatz(ob_map_core)
                elif ansatz_choice.upper() == 'STRONGLY_ENTANGLING':
                    ansatz_functor_for_core = StronglyEntanglingAnsatz(ob_map_core, n_layers=n_layers_strong) # Add ranges if supported
                else: 
                    ansatz_functor_for_core = IQPAnsatz(ob_map_core, n_layers=1, n_single_qubit_params=2)
                logger.info(f"Created '{ansatz_choice}' ansatz functor for Core Module processing.")
            except Exception as e_ans: 
                logger.error(f"Failed to create ansatz for core module: {e_ans}"); ansatz_functor_for_core = None
        else: logger.warning("Core atomic types not loaded or not valid Lambeq Ty, cannot create ansatz for core module.")
    elif use_core_enhancements:
         if not CORE_MODULE_AVAILABLE: logger.warning("Core enhancements requested, but arabic_morpho_lex_core.py is not available.")
         if not LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY: logger.warning("Core enhancements requested, but Lambeq is not available for core ansatz.")

    failed_processing_log = [] 

    for original_idx, sentence_data_dict in enumerate(sentences_to_process_input):
        sentence_str = sentence_data_dict.get("sentence")
        original_label = sentence_data_dict.get("label")
        report_item_ref = analysis_results_list_for_report[original_idx] # Reference to update

        if not sentence_str or not original_label:
            report_item_ref['error'] = 'Missing sentence or label in input'
            failed_processing_log.append(report_item_ref)
            continue

        logger.debug(f"[{current_set_name}] Processing original_idx {original_idx}: '{sentence_str[:30]}...'")
        
        current_linguistic_stream_core: Optional[List[Dict[str, Any]]] = None
        if use_core_enhancements and ansatz_functor_for_core:
            core_sense_results = process_sentence_for_qnlp_core(
                sentence_str=sentence_str, ansatz_functor=ansatz_functor_for_core,
                max_senses=max_senses_from_core,
                classical_feature_dim_for_surface_words=classical_feature_dim_config_for_core,
                debug=debug_circuits 
            )
            if core_sense_results and not core_sense_results[0].get('error'):
                current_linguistic_stream_core = core_sense_results[0].get('linguistic_stream_of_words')
                if current_linguistic_stream_core:
                    logger.info(f"Successfully obtained linguistic stream from core module for original_idx {original_idx}.")
                    report_item_ref['linguistic_stream_of_words_core'] = current_linguistic_stream_core # Store for report
                else: logger.warning(f"Core module processed original_idx {original_idx} but linguistic_stream_of_words was empty.")
            else:
                error_detail_core = core_sense_results[0].get('error', 'Unknown core processing error') if core_sense_results else 'No results from core module'
                logger.warning(f"Core processing failed for original_idx {original_idx}: {error_detail_core}. Proceeding with main processor only for enrichment data.")
        
        # FIXED: Initialize assigned_elements_main before use
        assigned_elements_main = []
        
        try:
            # Call the processor ONCE and store all returned values in a tuple
            result = main_sentence_processor(
                sentence_str, debug=debug_circuits, ansatz_choice=ansatz_choice, 
                n_layers_iqp=n_layers_iqp, n_layers_strong=n_layers_strong, cnot_ranges=cnot_ranges, 
                handle_lexical_ambiguity_in_typing=activate_ambiguity_typing_for_this_set
            )
            
            # Unpack the 6 required values, which we assume will always be there
            circuit_main, diagram_main, structure_main, tokens_main, analyses_main, roles_main = result[:6]
            
            # Conditionally get the 7th value (assigned_elements_main) if it exists
            assigned_elements_main = result[6] if len(result) >= 7 else []

            # Proceed with validation
            if circuit_main is None: # A simple check is enough here
                err_msg = 'Main processor (camel_test2) failed to produce a valid circuit.'
                report_item_ref['error'] = err_msg
                failed_processing_log.append(report_item_ref)
                continue

        except Exception as e_main_proc:
            err_msg = f'Exception in main_sentence_processor: {e_main_proc}'
            logger.error(f"Error in main_sentence_processor for original_idx {original_idx}: {e_main_proc}", exc_info=True)
            report_item_ref['error'] = err_msg
            failed_processing_log.append(report_item_ref)
            continue

        # Populate lists for kernel training
        processed_original_indices_for_kernel.append(original_idx)
        kernel_train_sentences_str.append(sentence_str)
        kernel_train_original_labels.append(original_label)
        kernel_train_circuits_main.append(circuit_main)
        kernel_train_diagrams_main.append(diagram_main)
        kernel_train_tokens_main.append(tokens_main)
        kernel_train_analyses_main.append(analyses_main)
        kernel_train_structures_main.append(structure_main)
        kernel_train_roles_main.append(roles_main)
        kernel_train_linguistic_streams_core.append(current_linguistic_stream_core)
        kernel_train_assigned_elements_main.append(assigned_elements_main if assigned_elements_main else [])
        
        # Update report item with main processor's outputs
        report_item_ref.update({
            'structure_from_main_processor': structure_main, 
            'roles_from_main_processor': roles_main,
            'tokens_from_main_processor': tokens_main,
            'analyses_from_main_processor': analyses_main, # Could be large, consider summarizing
            'circuit_object': circuit_main
        })
        if QISKIT_AVAILABLE and isinstance(circuit_main, QuantumCircuit):
            try:
                report_item_ref['circuit_representation_str'] = Qasm().dumps(circuit_main)
                report_item_ref['circuit_representation_type'] = "qasm"
            except Exception: 
                try: 
                    with io.BytesIO() as b: 
                        qpy.dump([circuit_main], b)
                        report_item_ref['circuit_representation_str'] = base64.b64encode(b.getvalue()).decode('utf-8')
                        report_item_ref['circuit_representation_type'] = "qpy_b64"
                except Exception as e_s: 
                    report_item_ref['circuit_representation_str'] = f"SERIALIZATION_ERROR: {e_s}"

    if not kernel_train_circuits_main:
        logger.error(f"[{current_set_name}] No sentences successfully processed for kernel training.")
        # Save failed_processing_log (already contains all original items with errors if any)
        if failed_processing_log: # This log now contains only items that failed before kernel stage
            err_path = os.path.join(output_dir_set, f"pipeline_failures_before_kernel_{current_set_name}.json")
            with open(err_path, 'w', encoding='utf-8') as f_err: 
                json.dump(failed_processing_log, f_err, indent=2, ensure_ascii=False)
            logger.info(f"Pre-kernel failure log saved to {err_path}")
        return None, analysis_results_list_for_report, None 
        
    n_single_qubit_params_for_iqp_config = n_layers_iqp * 2
    kernel = ArabicQuantumMeaningKernel(
        embedding_dim=30, num_clusters=min(5, max(1, len(kernel_train_circuits_main)//2 if len(kernel_train_circuits_main)>1 else 1)),
        embedding_model_path=embedding_model_path,
        ansatz_choice_for_main_processor=ansatz_choice, 
        n_layers_iqp=n_layers_iqp, n_layers_strong=n_layers_strong,
        output_dir_set=output_dir_set,
        embedding_type_for_classification=embedding_type_for_classification,
        classical_feature_dim_from_core=classical_feature_dim_config_for_core,
        config_n_layers_iqp=n_layers_iqp,
        config_n_single_qubit_params_iqp=n_single_qubit_params_for_iqp_config,
        config_n_layers_strong=n_layers_strong 
    )
    kernel.current_set_name = current_set_name
    
    kernel_train_ambiguous_maps: List[Optional[Dict[int, Dict[str, Any]]]] = []
    if activate_ambiguity_typing_for_this_set: # More direct, build only if ambiguity was handled
        for sent_idx_in_kernel_batch in range(len(kernel_train_assigned_elements_main)): # Use the new list
            current_assigned_elements = kernel_train_assigned_elements_main[sent_idx_in_kernel_batch]
            amb_map_for_sentence: Dict[int, Dict[str, Any]] = {}
            if current_assigned_elements:
                for diag_element in current_assigned_elements: # Iterate through Word/Box/AmbiguousLexicalBox
                    if isinstance(diag_element, AmbiguousLexicalBox): # from common_qnlp_types
                        # The 'original_idx' for AmbiguousLexicalBox should be in its .data
                        # This was planned to be set in assign_discocat_types_v2_2
                        # Let's assume assign_discocat_types has ensured Word/Box/AmbiguousLexicalBox
                        # in `word_core_types_list` have `diag_element.data['original_idx']`
                        
                        original_token_idx_within_sentence = diag_element.data.get('original_idx')
                        if original_token_idx_within_sentence is None and '_' in diag_element.name: # Fallback to parsing name
                            try:
                                original_token_idx_within_sentence = int(diag_element.name.split('_')[-1])
                            except ValueError: 
                                pass
                        
                        if original_token_idx_within_sentence is not None:
                            amb_map_for_sentence[original_token_idx_within_sentence] = {
                                'senses': diag_element.data.get('senses', []),
                                'box_name': diag_element.name 
                            }
                        else:
                            logger.warning(f"Could not determine original_idx for AmbiguousLexicalBox: {diag_element.name}")
            kernel_train_ambiguous_maps.append(amb_map_for_sentence if amb_map_for_sentence else None)
    else:
        kernel_train_ambiguous_maps = [None] * len(kernel_train_circuits_main) # List of Nones

    kernel.train(
        assigned_diagram_elements_main=kernel_train_assigned_elements_main, # NEW, pass the elements themselves
        ambiguous_maps_for_sentences=kernel_train_ambiguous_maps, # Pass the derived map
        original_indices=processed_original_indices_for_kernel, 
        sentences_str=kernel_train_sentences_str,
        circuits_main=kernel_train_circuits_main,
        diagrams_main=kernel_train_diagrams_main,
        tokens_main=kernel_train_tokens_main,
        analyses_main=kernel_train_analyses_main,
        structures_main=kernel_train_structures_main,
        roles_main=kernel_train_roles_main,
        linguistic_streams_core=kernel_train_linguistic_streams_core
    )
    
    # Update analysis_results_list_for_report with interpretations from the kernel
    for dense_idx, original_idx_val in enumerate(processed_original_indices_for_kernel):
        # Find the corresponding item in the report list by original_idx
        report_item_to_update = next((item for item in analysis_results_list_for_report if item.get('original_index') == original_idx_val), None)
        if report_item_to_update:
            # Fetch interpretation data from kernel (embeddings are keyed by original_idx in kernel now)
            q_emb = kernel.circuit_embeddings.get(original_idx_val)
            c_emb = kernel.sentence_embeddings.get(original_idx_val)
            l_emb = kernel.linguistic_features.get(original_idx_val)
            
            interpretation = {
                'sentence': kernel_train_sentences_str[dense_idx], 
                'structure': kernel_train_structures_main[dense_idx],
                'roles': kernel_train_roles_main[dense_idx],
                'quantum_embedding': q_emb, 'combined_embedding': c_emb, 'linguistic_features_vector': l_emb,
                'error': None 
            }
            if (kernel.kmeans_model and q_emb is not None and kernel.cluster_labels is not None and 
                original_idx_val < len(kernel.cluster_labels) and kernel.cluster_labels[original_idx_val] != -1):
                cluster_id = kernel.cluster_labels[original_idx_val]
                interpretation['nearest_cluster_id'] = int(cluster_id)
                interpretation['nearest_cluster_desc'] = kernel.meaning_map.get(int(cluster_id), {}).get('deduced_template', 'N/A')
            
            report_item_to_update['interpretation'] = interpretation
        else:
            logger.warning(f"Could not find report item for original_idx {original_idx_val} to update with kernel interpretation.")

    kernel.final_results_for_set = analysis_results_list_for_report 

    classification_summary = None
    if run_classification and kernel_train_original_labels:
        # More robust: use processed_original_indices_for_kernel to get labels
        labels_for_kernel_items = [sentences_for_current_set[orig_idx]['label'] for orig_idx in processed_original_indices_for_kernel]

        if len(set(labels_for_kernel_items)) < 2:
            logger.warning(f"[{current_set_name}] Skipping classification: Need >= 2 distinct labels among kernel-processed sentences.")
        else:
            classification_summary = kernel.evaluate_classification_accuracy(
                labels=labels_for_kernel_items, # Pass only labels for items kernel processed
                embedding_type=embedding_type_for_classification
            ) # evaluate_classification_accuracy needs to be robust to the length of labels_all_set
            kernel.classification_results_for_set = classification_summary

    if run_clustering_viz: 
        kernel._generate_pca_plots_for_current_set('quantum', generate_per_category_pca_plots)
        kernel._generate_pca_plots_for_current_set('combined', generate_per_category_pca_plots)
        if any(s is not None for s in kernel_train_linguistic_streams_core): 
             kernel._generate_pca_plots_for_current_set('linguistic', generate_per_category_pca_plots)

    logger.debug(f"[{current_set_name}] Starting sanitization of analysis_results_list_for_report before HTML generation.")
    sanitized_list_for_report = []
    for item_idx, report_data_item in enumerate(analysis_results_list_for_report):
        if report_data_item is None:
            logger.error(
                f"[{current_set_name}] CRITICAL: Found a None item at index {item_idx} "
                f"in analysis_results_list_for_report before HTML generation. This indicates a prior processing failure "
                f"that was not correctly captured as an error dictionary. Replacing with an error placeholder."
            )
            
            sentence_context = "Unknown sentence (original data lost)"
            label_context = "N/A (original data lost)"
            original_index_val = item_idx # This is the index in the current list

            # Try to get more context if sentences_to_process_input is available and matches length
            # Note: sentences_to_process_input might have been truncated by max_sentences_to_process
            if item_idx < len(sentences_to_process_input):
                context_sentence_data = sentences_to_process_input[item_idx]
                sentence_context = context_sentence_data.get("sentence", sentence_context)
                label_context = context_sentence_data.get("label", label_context)
                # The 'original_index' from the initial list construction would be lost if the item is None.
                # We use item_idx as a fallback for the current position.

            placeholder_item = {
                'sentence_text': sentence_context, # Key expected by generate_html_report
                'label': label_context,
                'original_index': original_index_val, 
                'sense_id': 'CORRUPTED_DATA',    # Placeholder
                'sense_score': 'N/A',           # Placeholder
                'error': 'CRITICAL ERROR: Sentence data item was lost or corrupted (became None) during processing pipeline.',
                
                # Placeholders for other fields generate_html_report might access
                'diagram_image_html': "<div class='error-message'>Data Corrupted: Diagram unavailable due to critical error.</div>",
                'circuit_draw_html': "<div class='error-message'>Data Corrupted: Circuit unavailable due to critical error.</div>",
                'linguistic_stream_html': "<div class='error-message'>Data Corrupted: Linguistic stream unavailable due to critical error.</div>",
                
                # Maintaining structure of other original keys, if possible
                'interpretation': {
                    'error': 'Parent data item was corrupted, no detailed interpretation available.',
                    'sentence': sentence_context, # repeat for consistency if interpretation is directly used
                    # Add other interpretation sub-fields as None or error strings if generate_html_report expects them
                    'structure': 'N/A', 'roles': {}, 'quantum_embedding': None, 'combined_embedding': None, 
                    'linguistic_features_vector': None, 'nearest_cluster_id': None, 'nearest_cluster_desc': 'N/A'
                },
                'circuit_representation_str': 'N/A_CORRUPTED',
                'circuit_representation_type': 'N/A_CORRUPTED',
                'structure_from_main_processor': 'N/A_CORRUPTED', 
                'roles_from_main_processor': {},
                'tokens_from_main_processor': ['N/A_CORRUPTED'],
                'analyses_from_main_processor': [{'error': 'Data corrupted'}],
                'linguistic_stream_of_words_core': [{'error': 'Data corrupted'}] if use_core_enhancements else None
            }
            sanitized_list_for_report.append(placeholder_item)
        else:
            # Ensure 'sentence_text' key exists if 'sentence' key exists, for generate_html_report compatibility
            if 'sentence' in report_data_item and 'sentence_text' not in report_data_item:
                report_data_item['sentence_text'] = report_data_item['sentence']
            sanitized_list_for_report.append(report_data_item)
            
    analysis_results_list_for_report = sanitized_list_for_report # Overwrite with the sanitized list
    logger.debug(f"[{current_set_name}] Sanitization of analysis_results_list_for_report complete. Final list length: {len(analysis_results_list_for_report)}")

    if kernel:
        html_report_content = kernel.generate_html_report(analysis_results_list=analysis_results_list_for_report) 
        report_path = os.path.join(output_dir_set, f'report_{sanitize_filename_v8(current_set_name)}.html')
        try:
            with open(report_path, 'w', encoding='utf-8') as f: 
                f.write(html_report_content)
            logger.info(f"[{current_set_name}] HTML report saved to {report_path}")
        except Exception as e_html: 
            logger.error(f"Error saving HTML report: {e_html}")
    
    if kernel:
        kernel_save_path = os.path.join(output_dir_set, f'kernel_state_{sanitize_filename_v8(current_set_name)}.pkl')
        kernel.save_model(kernel_save_path)

    logger.info(f"\n--- [{current_set_name}] Pipeline Finished. Processed {len(kernel_train_circuits_main)} sentences for kernel. ---")
    return kernel, analysis_results_list_for_report, classification_summary

# ============================================
# Utility to transform data (Example - Keep as needed)
# ============================================
def transform_data(original_data):
    """Transforms the simple list format to the required dict format."""
    new_data = {}
    for set_name, sentences in original_data.items():
        new_data[set_name] = []
        # Basic labeling logic (replace with your actual logic)
        for i, sentence in enumerate(sentences):
            label_suffix = "TypeA" if i % 2 == 0 else "TypeB" # Example label
            new_data[set_name].append({"sentence": sentence, "label": f"{set_name}_{label_suffix}"})
    return new_data

# ============================================
# Main execution block (Example - Keep for testing v8.py directly)
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running v8.py directly for testing conditional core integration...")

    test_sentences_data = {
        "GeneralTestSet": [
            {"sentence": "الولد يقرأ الكتاب.", "label": "General_SVO"},
            {"sentence": "البيت كبير.", "label": "General_Nominal"}
        ],
        "MorphologyTestSet": [ 
            {"sentence": "يكتبون الدرس بسرعة.", "label": "Morpho_VerbPlural"},
            {"sentence": "المعلمة ماهرة.", "label": "Morpho_NounFem"}
        ]
    }
    if not os.path.exists("sentences.json"):
        with open("sentences.json", "w", encoding="utf-8") as f_json:
            json.dump(test_sentences_data, f_json, ensure_ascii=False, indent=2)
        logger.info("Created a dummy sentences.json for testing.")
    else: # Overwrite with test data if it exists, for consistent testing
        with open("sentences.json", "w", encoding="utf-8") as f_json:
            json.dump(test_sentences_data, f_json, ensure_ascii=False, indent=2)
        logger.info("Overwrote sentences.json with test data for this run.")


    base_output_dir_main_test = "v8_conditional_core_test_outputs"
    os.makedirs(base_output_dir_main_test, exist_ok=True)

    for set_name_from_data, sentences_list_for_set in test_sentences_data.items():
        set_specific_output_dir = os.path.join(base_output_dir_main_test, sanitize_filename_v8(set_name_from_data))
        os.makedirs(set_specific_output_dir, exist_ok=True)

        logger.info(f"\n>>> Processing test set: {set_name_from_data} <<<")
        k, results, class_sum = prepare_quantum_nlp_pipeline_v8(
            sentences_for_current_set=sentences_list_for_set,
            current_set_name=set_name_from_data,
            output_dir_set=set_specific_output_dir,
            morpho_lex_enhanced_sets=["MorphologyTestSet", "Morphology"] # Ensure your test set name is here
        )
        if k: logger.info(f"Finished set {set_name_from_data}. Kernel has {len(k.circuit_embeddings)} embeddings.")
        else: logger.error(f"Kernel processing failed for set {set_name_from_data}.")