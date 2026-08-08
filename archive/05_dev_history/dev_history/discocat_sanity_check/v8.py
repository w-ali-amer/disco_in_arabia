# quantum_kernel_v8.py (Combined Clustering/Visualization & Classification Evaluation)
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
# Corrected Typing import for Mapping
from typing import List, Dict, Tuple, Optional, Any, Sequence, Set, Mapping
import matplotlib.pyplot as plt
import pickle
import os
import traceback
import logging
from collections import Counter
import hashlib

# --- Imports for Arabic Text Display ---
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_DISPLAY_ENABLED = True
except ImportError:
    print("Warning: 'arabic_reshaper' or 'python-bidi' not found.")
    ARABIC_DISPLAY_ENABLED = False

logger = logging.getLogger(__name__)
# Configure basic logging - Ensure this is configured in your main script (exp3.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.exceptions import QiskitError
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler, Estimator
    from qiskit.quantum_info import SparsePauliOp, partial_trace
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit or Qiskit-Aer not found. Quantum execution will fail.")
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
    from lambeq import AtomicType, IQPAnsatz, SpiderAnsatz, StronglyEntanglingAnsatz
    from lambeq.backend.grammar import Diagram as GrammarDiagram, Ty, Box, Cup, Id, Spider, Swap
    LAMBEQ_AVAILABLE = True
except ImportError:
    logger.warning("Warning: Lambeq not found.")
    class GrammarDiagram: pass
    class Ty: pass
    class Box: pass
    AtomicType = None
    LAMBEQ_AVAILABLE = False

# --- Gensim Imports ---
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    logger.warning("Warning: gensim not found. Parameter binding via embeddings disabled.")
    GENSIM_AVAILABLE = False

# --- Scikit-learn Imports ---
SKLEARN_AVAILABLE = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC  # <-- Added SVC
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler # <-- Added StandardScaler
    from sklearn.decomposition import PCA
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

# --- Dependency on camel_test2 ---
# Use the latest version that includes the desired type assignment and feature boxes
try:
    # Import the REVISED function
    from camel_test2 import arabic_to_quantum_enhanced_v2_7 as arabic_to_quantum_enhanced
    print("Successfully imported 'arabic_to_quantum_enhanced_v2_fixed' as 'arabic_to_quantum_enhanced'.")
except ImportError:
    print("ERROR: Cannot import 'arabic_to_quantum_enhanced_v2_fixed' from 'camel_test2'.")
    def arabic_to_quantum_enhanced(*args, **kwargs):
        print("ERROR: Dummy arabic_to_quantum_enhanced called because import failed.")
        return None, None, "ERROR", [], [], {}


# --- Helper Functions ---
def shape_arabic_text(text):
    """Reshapes and applies bidi algorithm for correct Arabic display."""
    # (Implementation as before)
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
    # (Implementation as before)
    if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
        if not getattr(circuit, 'name', None):
            try: setattr(circuit, 'name', default_name)
            except Exception: pass
    return circuit

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
                 embedding_dim: int = 30, # Dimensionality for BOTH quantum and combined features
                 num_clusters: int = 5,   # Default clusters for K-Means viz
                 embedding_model_path: Optional[str] = None,
                 params_per_word: int = 5,
                 shots: int = 8192,
                 morph_param_weight: float = 0.3,
                 combine_feature_weight: float = 0.5): # Weight for quantum features in combined embedding
        """
        Initialize the combined quantum meaning kernel. V8.1 (Handles .mdl load)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ArabicQuantumMeaningKernel.")

        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.params_per_word = params_per_word
        self._embedding_model_path = embedding_model_path # Store the path
        self.shots = shots
        self.morph_param_weight = max(0.0, min(1.0, morph_param_weight))
        self.combine_feature_weight = max(0.0, min(1.0, combine_feature_weight)) # Weight for combining features

        # --- Simulator Initialization ---
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
            # Initialize Estimator primitive with specified shots
            self.estimator = Estimator(options={'shots': self.shots})
            logger.info(f"Initialized Qiskit AerSimulator and Estimator primitive (shots={self.shots}).")
        else:
            self.simulator = None; self.estimator = None
            logger.warning("Qiskit not available. Quantum execution will be skipped.")

        # --- Kernel State ---
        self.kmeans_model = None     # Stores the fitted KMeans model (on quantum embeddings)
        self.cluster_labels = None   # K-Means labels assigned to sentences
        self.meaning_clusters = None # K-Means cluster centers
        self.meaning_map = {}        # Descriptive info about K-Means clusters
        self.reference_sentences = {} # Map index to original sentence text
        self.circuit_embeddings = {} # Stores PURELY QUANTUM features {index: np.array}
        self.sentence_embeddings = {} # Stores COMBINED (quantum+classical) features {index: np.array}

        self.camel_analyzer = None # CAMeL Tools Analyzer instance

        # --- Load Embedding Model ---
        self.embedding_model = None # This will store the KeyedVectors (wv part)
        # Check if Gensim is available AND a path was provided
        if GENSIM_AVAILABLE and embedding_model_path:
            logger.info(f"Attempting to load word embedding model from path: {embedding_model_path}...")
            if not os.path.exists(embedding_model_path):
                 logger.error(f"Embedding file/path not found: {embedding_model_path}")
            else:
                loaded_model_object = None
                try:
                    # *** MODIFICATION START: Handle .mdl loading ***
                    logger.info("Attempting to load using Word2Vec.load() for potential .mdl format...")
                    # Assumes Word2Vec, change to FastText.load if it's a FastText model
                    from gensim.models import Word2Vec # Import here if not already global
                    loaded_model_object = Word2Vec.load(embedding_model_path)
                    logger.info(f"Successfully loaded full model object (type: {type(loaded_model_object)}).")
                    # Extract the KeyedVectors part (usually in .wv)
                    if hasattr(loaded_model_object, 'wv'):
                        self.embedding_model = loaded_model_object.wv
                        logger.info(f"Extracted KeyedVectors (.wv). Vector size: {self.embedding_model.vector_size}")
                    else:
                        logger.error("Loaded model object does not have a 'wv' attribute containing vectors.")
                        self.embedding_model = None
                    # *** MODIFICATION END ***

                except Exception as e_load_full:
                    logger.warning(f"Failed to load as full Word2Vec model ({e_load_full}). Trying KeyedVectors.load() as fallback...")
                    try:
                        # Fallback: Try loading just KeyedVectors (for .npy format etc.)
                        self.embedding_model = KeyedVectors.load(embedding_model_path)
                        logger.info(f"Successfully loaded KeyedVectors directly. Vector size: {self.embedding_model.vector_size}")
                    except Exception as e_load_kv:
                        logger.error(f"Failed to load word embedding model using both Word2Vec.load and KeyedVectors.load: {e_load_kv}", exc_info=True)
                        self.embedding_model = None # Ensure model is None if all loading fails

        elif not GENSIM_AVAILABLE:
             logger.warning("Gensim library not found, cannot load word embeddings.")
        elif not embedding_model_path:
             logger.info("No embedding model path provided, skipping embedding loading.")


        # --- Semantic Templates (for reporting) ---
        # (Keep existing templates)
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
        # (Keep existing CAMeL Tools initialization)
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

    def get_morph_vector(self, morph_features: Optional[Dict]) -> np.ndarray:
        """
        Converts CAMeL Tools morphological analysis dictionary into a richer
        fixed-size numerical vector (16 dimensions).

        Args:
            morph_features (Optional[Dict]): The dictionary from CAMeL Tools analysis
                                             (e.g., analyses[i]['morph']).

        Returns:
            np.ndarray: A 16-dimensional numpy array representing morphological features.
                        Values are generally scaled between 0 and 1.
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
    def _bind_parameters(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Dict]) -> Optional[Dict[Parameter, float]]:
        """
        Creates parameter bindings using word embeddings (tanh scaling)
        AND enhanced morphological features. Uses hash-based fallback.
        V8.10 (Adds dediacritization for embedding lookup).
        """
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit) or not hasattr(circuit, 'parameters'):
            logger.warning("Invalid circuit passed to _bind_parameters."); return None

        params: Set[Parameter] = circuit.parameters
        if not params: return {} # No parameters

        num_params_needed = len(params)
        param_values: Dict[Parameter, float] = {} # Dict to store {Parameter: value}
        logger.debug(f"Binding {num_params_needed} parameters for circuit '{circuit.name}'...")

        embedding_source = None
        if self.embedding_model is None:
             logger.warning("!!! Embedding model (self.embedding_model) is None at start of _bind_parameters !!!")
        elif hasattr(self.embedding_model, 'vector_size') and hasattr(self.embedding_model, 'key_to_index'):
             logger.debug("Embedding model appears to be KeyedVectors (or .wv).")
             embedding_source = self.embedding_model
        else:
             logger.warning(f"!!! Unknown or unexpected embedding model type in self.embedding_model: {type(self.embedding_model)}. Cannot use embeddings. !!!")

        # --- Import dediac_ar from camel_tools if available ---
        dediac_ar_func = None
        if self.camel_analyzer: # Check if CAMeL Tools are available
            try:
                from camel_tools.utils.dediac import dediac_ar
                dediac_ar_func = dediac_ar
                logger.debug("Dediacritization function (dediac_ar) is available.")
            except ImportError:
                logger.warning("camel_tools.utils.dediac not found. Cannot perform dediacritization for lookup.")
        else:
            logger.warning("CAMeL Analyzer not available, skipping dediacritization for lookup.")
        # --- End dediac_ar import ---


        # Helper for Hash-based Fallback
        def get_hashed_value(param_name: str) -> float:
            param_hash_input = param_name.encode('utf-8')
            param_hash_val = int(hashlib.sha256(param_hash_input).hexdigest(), 16)
            scaled_value = ((param_hash_val % (2 * np.pi * 10000)) / 10000.0) - np.pi
            return scaled_value

        params_list = sorted(list(params), key=lambda p: p.name)
        param_idx = 0
        word_idx = 0
        assigned_params: Set[Parameter] = set()

        if not analyses or len(tokens) != len(analyses):
            logger.warning(f"Tokens/Analyses mismatch for '{' '.join(tokens[:3])}...'. Using fallback morph.")
            analyses = [{"lemma": tok, "pos": "UNK", "deprel": "dep", "head": -1, "morph": None} for tok in tokens]

        logger.debug(f"Iterating through {len(analyses)} words/analyses to assign {self.params_per_word} params each...")

        while param_idx < num_params_needed and word_idx < len(analyses):
            analysis_dict = analyses[word_idx]
            lemma = analysis_dict.get('lemma', tokens[word_idx])
            token = tokens[word_idx]
            morph_features = analysis_dict.get('morph')

            embed_vector = None
            word_in_vocab = False
            embedding_reason = "No embedding source"

            if embedding_source:
                # --- MODIFIED LOOKUP LOGIC WITH DEDIACRITIZATION ---
                lookup_attempts = [
                    (lemma, "lemma"),
                    (token, "token")
                ]
                if dediac_ar_func:
                    try:
                        lemma_dediac = dediac_ar_func(lemma) if lemma else None
                        token_dediac = dediac_ar_func(token) if token else None
                        if lemma_dediac and lemma_dediac != lemma:
                            lookup_attempts.append((lemma_dediac, "dediac_lemma"))
                        if token_dediac and token_dediac != token:
                            lookup_attempts.append((token_dediac, "dediac_token"))
                    except Exception as e_dediac:
                        logger.warning(f"Error during dediacritization for '{token}'/'{lemma}': {e_dediac}")


                found_key = None
                for key_to_try, key_type in lookup_attempts:
                    if not key_to_try: continue # Skip if key is empty
                    try:
                        if key_to_try in embedding_source:
                            embed_vector = embedding_source[key_to_try]
                            word_in_vocab = True
                            embedding_reason = f"Found {key_type} '{key_to_try}' in vocab"
                            found_key = key_to_try
                            logger.debug(f"  Found embedding for: '{key_to_try}' (type: {key_type})")
                            break # Stop if found
                    except Exception as e_embed_lookup:
                        logger.warning(f"Error looking up {key_type} '{key_to_try}': {e_embed_lookup}")
                        continue # Try next key

                if not found_key:
                    embedding_reason = f"All forms OOV for token='{token}', lemma='{lemma}'"
                    if dediac_ar_func: # Add dediac forms to reason if attempted
                         embedding_reason += f" (tried dediac: '{dediac_ar_func(token)}', '{dediac_ar_func(lemma)}')"
                    logger.debug(f"  OOV: Word {word_idx} ('{token}', lemma '{lemma}') - {embedding_reason}")
                # --- END MODIFIED LOOKUP LOGIC ---

            else:
                 embedding_reason = "Embedding model not loaded or invalid."

            morph_vector = self.get_morph_vector(morph_features)
            morph_vec_len = len(morph_vector)

            for i in range(self.params_per_word):
                if param_idx >= num_params_needed: break
                current_param = params_list[param_idx]
                if current_param in assigned_params:
                    param_idx += 1
                    continue

                param_value = 0.0
                value_source = "Fallback"
                embed_contrib = 0.0
                morph_contrib = 0.0

                if embed_vector is not None and word_in_vocab:
                    hash_input_embed = f"{found_key}_embed_{i}".encode('utf-8') # Use the key that was actually found
                    hash_val_embed = int(hashlib.sha256(hash_input_embed).hexdigest(), 16)
                    vec_size = getattr(embedding_source, 'vector_size', 1)
                    if vec_size <= 0: vec_size = 1
                    embed_idx = hash_val_embed % vec_size
                    raw_embed_val = float(embed_vector[embed_idx])
                    embed_contrib = np.tanh(raw_embed_val) * np.pi
                    value_source = f"Embed (tanh, from '{found_key}')"
                else:
                    embed_contrib = get_hashed_value(f"{current_param.name}_embed_fallback")
                    value_source = f"Fallback ({embedding_reason})"

                if morph_vec_len > 0:
                    hash_input_morph = f"{lemma}_morph_{i}".encode('utf-8') # Use original lemma for morph hash consistency
                    hash_val_morph = int(hashlib.sha256(hash_input_morph).hexdigest(), 16)
                    morph_idx = hash_val_morph % morph_vec_len
                    raw_morph_val = morph_vector[morph_idx]
                    morph_contrib = (raw_morph_val * 2 * np.pi) - np.pi
                    if value_source.startswith("Fallback"):
                         value_source += "+Morph"
                else:
                    morph_contrib = get_hashed_value(f"{current_param.name}_morph_fallback")
                    if value_source.startswith("Fallback"):
                         value_source += "+MorphFallback"

                param_value = (1.0 - self.morph_param_weight) * embed_contrib + self.morph_param_weight * morph_contrib
                param_value = max(-np.pi, min(np.pi, param_value))
                param_values[current_param] = param_value
                logger.debug(f"  Bound param {current_param.name} (idx {param_idx}, word {word_idx} '{token}') to {param_value:.3f} (Source: {value_source})")
                assigned_params.add(current_param)
                param_idx += 1
            word_idx += 1

        num_remaining = num_params_needed - param_idx
        if num_remaining > 0:
             logger.warning(f"Circuit '{circuit.name}' has {num_remaining} more parameters than expected from words ({len(analyses)} words * {self.params_per_word} params/word = {len(analyses) * self.params_per_word}). Assigning fallbacks.")

        while param_idx < num_params_needed:
             current_param = params_list[param_idx]
             if current_param not in assigned_params:
                  fallback_value = get_hashed_value(current_param.name)
                  param_values[current_param] = fallback_value
                  logger.warning(f"  Bound remaining param {current_param.name} (idx {param_idx}) to HASHED fallback -> {fallback_value:.3f}")
                  assigned_params.add(current_param)
             param_idx += 1

        if len(param_values) != num_params_needed:
             logger.error(f"Parameter binding mismatch! Expected {num_params_needed}, got {len(param_values)}. Assigning emergency fallbacks.")
             missing_params = params - set(param_values.keys())
             for p_missing in missing_params:
                 param_values[p_missing] = get_hashed_value(p_missing.name)

        return param_values
        # --- End of _bind_parameters implementation ---


    # --- Quantum Feature Extraction (Unchanged) ---
    def get_enhanced_circuit_features(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Dict], debug: bool = False) -> np.ndarray:
        """
        Extracts features using Pauli X, Y, Z expectation values via Qiskit Estimator.
        V8.5.

        Args:
            circuit: The Qiskit QuantumCircuit (potentially parameterized).
            tokens: List of tokens for the sentence.
            analyses: List of analysis dictionaries for the sentence.
            debug: Print debug info.

        Returns:
            A numpy array of features derived from the circuit execution probabilities.
        """
        fallback_features = np.zeros(self.embedding_dim)

        if not QISKIT_AVAILABLE or self.estimator is None:
            logger.error("Qiskit Estimator not available in get_enhanced_circuit_features.")
            return fallback_features
        if not isinstance(circuit, QuantumCircuit):
             logger.error(f"get_enhanced_circuit_features received non-Qiskit object: {type(circuit)}")
             return fallback_features

        circuit_name = f"circuit_{tokens[0]}_{len(tokens)}" if tokens else "circuit_unnamed"
        circuit = _ensure_circuit_name(circuit, circuit_name)
        num_qubits = circuit.num_qubits

        if num_qubits == 0:
            logger.warning(f"Circuit '{circuit.name}' has 0 qubits. Returning zero features.")
            return fallback_features

        feature_vector = []

        try:
            # --- 1. Bind Parameters ---
            parameter_binds_map = self._bind_parameters(circuit, tokens, analyses)
            if parameter_binds_map is None:
                 logger.error("Parameter binding failed. Returning zero features.")
                 return fallback_features
            if not parameter_binds_map and circuit.parameters:
                 logger.warning(f"Circuit '{circuit.name}' has parameters but binding map is empty.")
                 # Proceeding, Estimator might handle unbound params depending on version/options

            # --- 2: Remove Measurements (Estimator works on measurement-free circuits) ---
            circuit_to_estimate = circuit.remove_final_measurements(inplace=False)
            if debug: logger.debug(f"Removed final measurements from circuit '{circuit_name}' for Estimator.")

            # --- 3: Format parameters for Estimator ---
            params_list = sorted(list(circuit.parameters), key=lambda p: p.name)
            param_values_ordered_list = [parameter_binds_map.get(p, 0.0) for p in params_list]

            if debug and params_list:
                 logger.debug("Parameter values list for Estimator:")
                 for p, v in zip(params_list, param_values_ordered_list): logger.debug(f"  {p.name}: {v:.4f}")

            # --- 4. Define Observables (Pauli Z, X, AND Y for each qubit) ---
            observables = []
            for i in range(num_qubits):
                # Create Pauli strings for Z, X, Y on qubit i
                pauli_z_str = "I" * i + "Z" + "I" * (num_qubits - 1 - i)
                pauli_x_str = "I" * i + "X" + "I" * (num_qubits - 1 - i)
                pauli_y_str = "I" * i + "Y" + "I" * (num_qubits - 1 - i)
                # Append SparsePauliOp objects for each
                observables.append(SparsePauliOp.from_list([(pauli_z_str, 1)]))
                observables.append(SparsePauliOp.from_list([(pauli_x_str, 1)]))
                observables.append(SparsePauliOp.from_list([(pauli_y_str, 1)]))

            num_observables = len(observables)
            if debug: logger.debug(f"Defined {num_observables} observables (Z, X, Y for {num_qubits} qubits).")

            # --- 5. Run Estimator ---
            if not observables:
                 logger.warning("No observables defined. Returning zero features.")
                 return fallback_features

            # Prepare parameter values for the Estimator run
            # Need one parameter list for each circuit-observable pair
            parameter_values_for_estimator: Sequence[Sequence[float]]
            if not params_list:
                 # If circuit has no parameters, pass list of empty lists
                 parameter_values_for_estimator = [[]] * num_observables
                 if debug: logger.debug(f"Running Estimator for circuit '{circuit_to_estimate.name}' (no parameters)...")
            else:
                 # If parameters exist, repeat the ordered list for each observable
                 parameter_values_for_estimator = [param_values_ordered_list] * num_observables
                 if debug: logger.debug(f"Running Estimator for circuit '{circuit_to_estimate.name}' with {len(param_values_ordered_list)} param values...")

            # Run the job
            job = self.estimator.run(circuits=[circuit_to_estimate] * num_observables, # Use measurement-free copy
                                     observables=observables,
                                     parameter_values=parameter_values_for_estimator)

            result = job.result()
            expectation_values = result.values.tolist() # Get list of expectation values <Z_0>,<X_0>,<Y_0>,<Z_1>,<X_1>,<Y_1>,...
            if debug: logger.debug(f"Estimator job completed. Got {len(expectation_values)} expectation values.")
            feature_vector.extend(expectation_values)

        except QiskitError as qe:
             logger.error(f"QiskitError in get_enhanced_circuit_features (Estimator method): {qe}", exc_info=True)
             return fallback_features
        except Exception as e:
            logger.error(f"Unexpected error in get_enhanced_circuit_features (Estimator method): {e}", exc_info=True)
            return fallback_features # Return fallback on error

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
    def extract_complex_linguistic_features_v2(self, tokens: List[str], analyses: List[Dict], structure: str, roles: Dict) -> np.ndarray:
        """ Extracts more complex linguistic features using Stanza & CAMeL analyses. V2 """
        features = np.zeros(self.embedding_dim) # Ensure enough space
        num_features_added = 0
        max_features = self.embedding_dim // 2 # Allocate roughly half to linguistic features

        def add_feature(value):
            nonlocal num_features_added
            if num_features_added < max_features:
                # Normalize value roughly to [0, 1] or [-1, 1]
                norm_value = max(0.0, min(1.0, float(value))) # Simple clamp/scale
                features[num_features_added] = norm_value
                num_features_added += 1

        # --- Basic Features (from previous version) ---
        structure_map = {'VSO': 0, 'SVO': 1, 'NOMINAL': 2, 'COMPLEX': 3, 'OTHER': 4, 'VERBAL_OTHER': 5} # Added VERBAL_OTHER
        structure_idx = structure_map.get(structure.split('_')[0], 4)
        add_feature(structure_idx / (len(structure_map) - 1))

        pos_counts = Counter(a.get('pos', 'UNK_POS') for a in analyses) # Use .get() for safety
        total_tokens = max(1, len(tokens))
        add_feature(pos_counts.get('VERB', 0) / total_tokens)
        add_feature(pos_counts.get('NOUN', 0) / total_tokens)
        add_feature(pos_counts.get('ADJ', 0) / total_tokens)
        add_feature(pos_counts.get('ADP', 0) / total_tokens) # Added Prepositions
        add_feature(pos_counts.get('PRON', 0) / total_tokens) # Added Pronouns
        add_feature(1.0 if roles.get('verb') is not None else 0.0)
        add_feature(1.0 if roles.get('subject') is not None else 0.0)
        add_feature(1.0 if roles.get('object') is not None else 0.0)
        has_negation = any(a.get('lemma') in ['لا', 'ليس', 'غير', 'لم', 'لن'] for a in analyses) # Use .get()
        add_feature(1.0 if has_negation else 0.0)

        # --- More Complex Features ---
        # Dependency depth (approximate)
        max_depth = 0
        if roles.get('dependency_graph'):
            try: # Use networkx if available for proper depth calculation
                import networkx as nx
                G = nx.DiGraph()
                for i, analysis in enumerate(analyses):
                    head_idx = analysis.get('head', -1) # Use .get()
                    if head_idx != -1: G.add_edge(head_idx, i)
                root_node = roles.get('root', 0)
                if root_node in G:
                    max_depth = max(nx.shortest_path_length(G, source=root_node).values()) if G.nodes else 0
            except ImportError: max_depth = 0 # Fallback if networkx not installed
            except Exception: max_depth = 0 # Catch other graph errors
        add_feature(min(max_depth / 10.0, 1.0)) # Normalize max depth (e.g., up to 10)

        # Subordination / Complexity markers
        subordinate_markers = ['الذي', 'التي', 'الذين', 'اللواتي', 'عندما', 'حيث', 'لأن', 'كي', 'أنّ', 'حتى', 'بينما']
        has_subordinate = any(t in subordinate_markers for t in tokens)
        add_feature(1.0 if has_subordinate else 0.0)
        verb_count = pos_counts.get('VERB', 0)
        add_feature(min(verb_count / 3.0, 1.0)) # Normalize verb count

        # --- Morphological Features (Aggregated) ---
        num_fem = 0; num_pl = 0; num_past = 0; num_def = 0; num_valid_morph = 0
        for analysis_dict in analyses:
            morph = analysis_dict.get('morph')
            if morph:
                num_valid_morph += 1
                if morph.get('gen') == 'f': num_fem += 1
                if morph.get('num') == 'p': num_pl += 1
                if morph.get('asp') == 'p': num_past += 1
                if morph.get('stt') == 'd': num_def += 1

        total_valid_morph = max(1, num_valid_morph)
        add_feature(num_fem / total_valid_morph)
        add_feature(num_pl / total_valid_morph)
        add_feature(num_past / total_valid_morph)
        add_feature(num_def / total_valid_morph)

        # Final normalization of the linguistic feature vector part
        if num_features_added > 0:
            linguistic_part = features[:num_features_added]
            norm = np.linalg.norm(linguistic_part)
            if norm > 1e-9: features[:num_features_added] = linguistic_part / norm

        logger.debug(f"Extracted {num_features_added} complex linguistic features.")
        return features
        # --- End of extract_complex_linguistic_features_v2 implementation ---


    # --- NEW: Combine Features (Re-added from v6 logic) ---
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


    # --- Training (Modified to generate BOTH embedding types) ---
    def train(self, sentences: List[str], circuits: List[QuantumCircuit], tokens_list: List[List[str]], analyses_list: List[List[Dict]], structures: List[str], roles_list: List[Dict]):
        """
        Processes sentences, generates BOTH quantum and combined embeddings, stores them.
        Optionally performs clustering on quantum embeddings for visualization. V8.
        """
        self.reference_sentences = {i: sentences[i] for i in range(len(sentences))}
        self.circuit_embeddings = {} # Reset quantum embeddings
        self.sentence_embeddings = {} # Reset combined embeddings
        logger.info(f"\n--- Generating Embeddings V8 for {len(sentences)} sentences ---")

        min_len = len(sentences)
        # ... (Input validation as in v7) ...
        if not all(len(lst) == min_len for lst in [circuits, tokens_list, analyses_list, structures, roles_list]): logger.error("Input list length mismatch."); return self
        if min_len == 0: logger.error("Empty dataset."); return self

        # Extract features and create embeddings
        for i in range(min_len):
            try:
                current_circuit = circuits[i]
                current_tokens = tokens_list[i]
                current_analyses = analyses_list[i]
                current_structure = structures[i]
                current_roles = roles_list[i]

                if not isinstance(current_circuit, QuantumCircuit):
                     logger.warning(f"Skipping sentence {i}: Invalid circuit type {type(current_circuit)}")
                     continue

                # 1. Get Quantum Features
                quantum_features = self.get_enhanced_circuit_features(
                    current_circuit, current_tokens, current_analyses
                )
                self.circuit_embeddings[i] = quantum_features # Store quantum

                # 2. Get Linguistic Features
                linguistic_features = self.extract_complex_linguistic_features_v2(
                    current_tokens, current_analyses, current_structure, current_roles
                )

                # 3. Combine Features
                combined_embedding = self.combine_features_with_attention(
                    quantum_features, linguistic_features, current_structure
                )
                self.sentence_embeddings[i] = combined_embedding # Store combined

                logger.debug(f"Generated embeddings for sentence {i} (Q: {quantum_features.shape}, C: {combined_embedding.shape})")

            except Exception as e:
                logger.error(f"Error processing sentence {i} ('{sentences[i][:30]}...') during embedding generation: {e}", exc_info=True)

        if not self.circuit_embeddings:
            logger.error("No quantum embeddings generated.")
        else:
            logger.info(f"--- Quantum Embedding Generation Complete ({len(self.circuit_embeddings)} embeddings) ---")
        if not self.sentence_embeddings:
             logger.warning("No combined embeddings generated.")
        else:
             logger.info(f"--- Combined Embedding Generation Complete ({len(self.sentence_embeddings)} embeddings) ---")


        # --- Clustering on Quantum Embeddings (for Visualization/Description) ---
        self.learn_meaning_clusters_from_quantum()
        self.assign_meaning_to_clusters(sentences, structures, roles_list, analyses_list)

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
            self.cluster_labels = None; self.meaning_clusters = None
        # --- End of learn_meaning_clusters_from_quantum ---


    # --- Meaning Assignment (Unchanged - Describes clusters) ---
    def assign_meaning_to_clusters(self, sentences: List[str], structures: List[str], roles_list: List[Dict], analyses_list: List[List[Dict]]) -> Dict:
        """
        Assign meaning templates to clusters based on linguistic analysis.
        V6.3: Correctly handles list of analysis dictionaries.
        """
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            logger.warning("Cluster labels not found. Cannot assign meanings.")
            return self.meaning_map

        cluster_data_grouped = {label: [] for label in set(self.cluster_labels)}
        min_len = min(len(sentences), len(structures), len(roles_list), len(analyses_list), len(self.cluster_labels))

        # Group data by cluster label
        for i in range(min_len):
            label = self.cluster_labels[i]
            cluster_data_grouped[label].append({
                'sentence': sentences[i],
                'structure': structures[i],
                'roles': roles_list[i],
                'analyses': analyses_list[i], # This is the list of dicts for sentence i
                'index': i
            })

        # Process each cluster
        for cluster_id, cluster_data in cluster_data_grouped.items():
            if not cluster_data: continue

            # Counters for features within the cluster
            verb_lemmas = Counter(); subject_lemmas = Counter(); object_lemmas = Counter()
            common_preps = Counter(); verb_tenses = Counter(); verb_moods = Counter()
            structure_counts = Counter(); pos_counts = Counter() # Moved pos_counts here

            # Iterate through items (sentences) in the current cluster
            for item in cluster_data:
                structure_counts[item['structure']] += 1
                tokens = item['sentence'].split() # Get tokens if needed
                roles = item['roles']
                analyses = item['analyses'] # List of analysis dicts for this sentence

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
                    pos_counts[analysis_dict.get('pos', 'UNK_POS')] += 1
                    morph = analysis_dict.get('morph')
                    if morph: # Morphological features (if available)
                        verb_tenses[morph.get('asp', 'UNK')] += 1
                        verb_moods[morph.get('mod', 'UNK')] += 1

                    # Extract preposition info
                    if analysis_dict.get('pos') == 'ADP':
                        prep_lemma = analysis_dict.get('lemma', 'UNK_PREP')
                        common_preps[prep_lemma] += 1

                # Optional: Re-run CAMeL analysis if needed (consider if necessary)
                # This might be redundant if 'morph' is already populated correctly
                # if self.camel_analyzer:
                #     try:
                #         morph_analysis_list = self.camel_analyzer.analyze(item['sentence'])
                #         # ... (extract features from morph_analysis_list if needed) ...
                #     except Exception as e: pass

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
                    # *** FIX: Iterate through analysis dicts correctly ***
                    for item_inner in cluster_data:
                        subj_idx_inner = item_inner['roles'].get('subject')
                        if subj_idx_inner is not None:
                            # Iterate through the list of analysis dictionaries
                            for k, analysis_dict_k in enumerate(item_inner['analyses']):
                                # Access 'pos' and 'head' using dictionary keys
                                pos_k = analysis_dict_k.get('pos')
                                head_k = analysis_dict_k.get('head')
                                lemma_k = analysis_dict_k.get('lemma')
                                # Check condition
                                if head_k == subj_idx_inner and pos_k == 'ADJ':
                                    pred_lemmas[lemma_k] += 1
                                    break # Found predicate for this sentence
                    # *** END FIX ***
                    if pred_lemmas: predicate = pred_lemmas.most_common(1)[0][0]
                    deduced_template = f"{dominant_subj} is {predicate}"
                else:
                    deduced_template = f"Statement about {dominant_subj} involving {dominant_verb}"

            # Add tense information
            tense_map = {'p': ' (past)', 'i': ' (present)', 'c': ' (command)'}
            deduced_template += tense_map.get(dominant_tense, "")

            sentiment_label = None # Add sentiment logic if needed

            # Update the meaning map for the cluster
            if cluster_id not in self.meaning_map: self.meaning_map[cluster_id] = {}
            self.meaning_map[cluster_id].update({
                'structure': dominant_structure,
                'deduced_template': deduced_template,
                'dominant_verb': dominant_verb,
                'dominant_subject': dominant_subj,
                'dominant_object': dominant_obj,
                'common_prep_phrase': top_prep, # Store the lemma directly
                'sentiment': sentiment_label,
                'examples': [item['sentence'] for item in cluster_data[:3]],
                'original_templates': self.semantic_templates.get(dominant_structure.split('_')[0], self.semantic_templates.get('OTHER', {}))
            })
            # Ensure topics entry exists
            if 'topics' not in self.meaning_map[cluster_id]: self.meaning_map[cluster_id]['topics'] = {}

        return self.meaning_map
        # --- End of assign_meaning_to_clusters ---


    # --- Interpretation (Modified to return BOTH embeddings & handle context) ---
    def interpret_sentence(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Dict], structure: str, roles: Dict, previous_analysis_dict=None) -> Dict:
        """
        Generates quantum and combined embeddings, handles context. V8.
        """
        base_result = {
            'sentence': ' '.join(tokens), 'structure': structure, 'roles': roles,
            'quantum_embedding': None, 'combined_embedding': None, 'error': None,
            'discourse_relations': [] # Added for context
        }
        if not isinstance(circuit, QuantumCircuit):
            base_result['error'] = f'Invalid circuit type: {type(circuit)}'
            logger.error(base_result['error']); return base_result

        # --- Generate Quantum Embedding ---
        quantum_features = self.get_enhanced_circuit_features(circuit, tokens, analyses)
        if not np.all(np.isfinite(quantum_features)):
             base_result['error'] = 'Failed to generate valid quantum features (NaN/Inf).'
             logger.error(base_result['error']); return base_result
        base_result['quantum_embedding'] = quantum_features

        # --- Generate Linguistic & Combined Embedding ---
        try:
            linguistic_features = self.extract_complex_linguistic_features_v2(
                tokens, analyses, structure, roles
            )
            combined_embedding = self.combine_features_with_attention(
                quantum_features, linguistic_features, structure
            )
            base_result['combined_embedding'] = combined_embedding
        except Exception as e_comb:
             logger.error(f"Error generating combined embedding: {e_comb}")
             # Continue without combined embedding if it fails

        # --- Handle Context (Call separate method) ---
        if previous_analysis_dict is not None:
             return self.analyze_sentence_in_context(circuit, tokens, analyses, structure, roles, previous_analysis_dict)
        else:
             # Add nearest cluster info if no context needed (optional)
             if self.kmeans_model is not None:
                 try:
                     cluster_pred = self.kmeans_model.predict(quantum_features.reshape(1, -1))
                     base_result['nearest_cluster_id'] = int(cluster_pred[0])
                     # base_result['nearest_cluster_prob'] = ... # Similarity calc needed
                     base_result['nearest_cluster_desc'] = self.meaning_map.get(base_result['nearest_cluster_id'], {}).get('deduced_template', 'N/A')
                 except Exception as e_pred:
                     logger.warning(f"Could not predict nearest cluster: {e_pred}")
             return base_result


    # --- NEW: Discourse Analysis (Re-added from v6 logic) ---
    def analyze_sentence_in_context(self, current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analysis_dict=None):
        """ Analyze a sentence considering the previous sentence's context. """
        # Get base interpretation (passing tokens/analyses)
        base_interpretation = self.interpret_sentence(current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analysis_dict=None)

        if previous_analysis_dict is None:
            return base_interpretation
        if base_interpretation.get('error'): # Propagate error
             return base_interpretation

        previous_tokens = previous_analysis_dict.get('tokens')
        discourse_info = self.find_discourse_relations(current_tokens, previous_tokens)
        base_interpretation['discourse_relations'] = discourse_info

        # Optional: Contextual Embedding Adjustment
        previous_embedding = previous_analysis_dict.get('interpretation', {}).get('embedding')
        if previous_embedding is not None and base_interpretation.get('embedding') is not None:
            current_embedding = base_interpretation['embedding']
            context_influence = 0.2
            if isinstance(current_embedding, np.ndarray) and isinstance(previous_embedding, np.ndarray) and current_embedding.shape == previous_embedding.shape:
                context_aware_embedding = (1 - context_influence) * current_embedding + context_influence * previous_embedding
                norm = np.linalg.norm(context_aware_embedding); context_aware_embedding = context_aware_embedding / norm if norm > 0 else context_aware_embedding
                base_interpretation['context_aware_embedding'] = context_aware_embedding
            else:
                logger.warning("Embedding dimension/type mismatch. Skipping context blending.")
        return base_interpretation

        return current_interpretation_dict

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
                                         embedding_type: str = 'quantum', # 'quantum' or 'combined'
                                         test_size: float = 0.3,
                                         random_state: int = 42) -> Optional[Dict[str, Dict]]:
        """
        Evaluates classification accuracy using specified embeddings as features.
        Tests both GaussianNB and SVC classifiers.

        Args:
            labels (List[Any]): A list of labels corresponding to the stored embeddings.
                                Must be the same length as reference_sentences.
            embedding_type (str): Which embeddings to use ('quantum' or 'combined').
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.

        Returns:
            Optional[Dict[str, Dict]]: A dictionary containing results for each classifier,
                                       or None on error. Keys are classifier names (e.g., 'GaussianNB').
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot evaluate classification: scikit-learn not available.")
            return None

        logger.info(f"\n--- Evaluating Classification Accuracy using '{embedding_type}' Embeddings ---")

        # --- Select Embeddings ---
        if embedding_type == 'quantum':
            embeddings_dict = self.circuit_embeddings
            if not embeddings_dict: logger.error("No quantum embeddings available."); return None
        elif embedding_type == 'combined':
            embeddings_dict = self.sentence_embeddings
            if not embeddings_dict: logger.error("No combined embeddings available."); return None
        else:
            logger.error(f"Invalid embedding_type '{embedding_type}'. Choose 'quantum' or 'combined'.")
            return None

        if len(labels) != len(self.reference_sentences):
            logger.error(f"Label length mismatch: Got {len(labels)} labels, expected {len(self.reference_sentences)}.")
            return None

        # --- Prepare data ---
        indices = list(embeddings_dict.keys())
        X_list = []
        y_list = []

        for idx in indices:
            embedding = embeddings_dict.get(idx)
            # Ensure embedding exists, is finite, and a corresponding label exists
            if embedding is not None and np.all(np.isfinite(embedding)) and idx < len(labels):
                X_list.append(embedding)
                y_list.append(labels[idx])
            else:
                 logger.warning(f"Skipping index {idx} due to missing/invalid embedding or label for '{embedding_type}' type.")

        if len(X_list) < 2:
             logger.error(f"Not enough valid samples ({len(X_list)}) for classification evaluation using '{embedding_type}' embeddings.")
             return None

        X = np.array(X_list)
        y = np.array(y_list)
        num_classes = len(set(y))
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes.")

        if num_classes < 2:
            logger.error("Need at least two distinct classes for classification.")
            return None

        # --- Split data ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if num_classes > 1 else None
            )
            logger.info(f"Split data: Train={len(X_train)}, Test={len(X_test)}")
        except ValueError as e:
             logger.warning(f"Stratified train/test split failed (possibly too few samples per class): {e}. Trying without stratification.")
             try:
                 X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_size, random_state=random_state
                 )
                 logger.info("Performed train/test split without stratification as fallback.")
             except Exception as split_e:
                  logger.error(f"Fallback train/test split also failed: {split_e}")
                  return None
        except Exception as split_e_outer:
             logger.error(f"Error during train/test split: {split_e_outer}")
             return None

        # --- Scale Data ---
        # Fit scaler ONLY on training data, then transform both train and test
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logger.info("Applied StandardScaler to features.")
        except Exception as scale_e:
            logger.error(f"Error scaling data: {scale_e}. Proceeding without scaling.")
            X_train_scaled = X_train
            X_test_scaled = X_test


        # --- Classifiers to Evaluate ---
        classifiers = {
            "GaussianNB": GaussianNB(),
            "SVC": SVC(random_state=random_state) # Add SVC
            # Add other classifiers here if desired, e.g.:
            # "RandomForest": RandomForestClassifier(random_state=random_state),
        }

        results_summary = {}

        # --- Train and Evaluate Each Classifier ---
        for name, classifier in classifiers.items():
            logger.info(f"--- Training and Evaluating: {name} ---")
            try:
                # Train on scaled data
                classifier.fit(X_train_scaled, y_train)
                # Predict on scaled data
                y_pred = classifier.predict(X_test_scaled)

                # --- Evaluation ---
                accuracy = accuracy_score(y_test, y_pred)
                # Use target_names if you have a mapping from labels to readable names
                # report = classification_report(y_test, y_pred, zero_division=0, target_names=...)
                report = classification_report(y_test, y_pred, zero_division=0)

                logger.info(f"  Accuracy ({name}): {accuracy:.4f}")
                logger.info(f"  Classification Report ({name}):\n{report}")

                # Store results for this classifier
                results_summary[name] = {
                    'classifier': name,
                    'accuracy': accuracy,
                    'report': report,
                    'num_samples_train': len(X_train),
                    'num_samples_test': len(X_test),
                    'num_classes': num_classes,
                    'embedding_type_used': embedding_type # Record which embeddings were used
                }

            except Exception as e:
                logger.error(f"Error during {name} training/evaluation: {e}", exc_info=True)
                results_summary[name] = {
                    'classifier': name, 'error': str(e), 'embedding_type_used': embedding_type
                }

        return results_summary if results_summary else None
        # --- End of evaluate_classification_accuracy ---


    # --- Visualization (Modified to plot quantum embeddings by default) ---
    def visualize_meaning_space(self,
                                embedding_type: str = 'quantum',
                                highlight_indices=None,
                                labels=None, # External labels for coloring
                                save_path=None):
        """
        Visualize meaning space using PCA on specified embeddings ('quantum' or 'combined').
        Optionally color points based on provided labels (Handles string labels). V8.2.
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot visualize meaning space: scikit-learn not available.")
            return None

        logger.info(f"Visualizing meaning space using '{embedding_type}' embeddings...")

        if embedding_type == 'quantum':
            embeddings_dict = self.circuit_embeddings
            title_suffix = shape_arabic_text('(التضمينات الكمومية)')
        elif embedding_type == 'combined':
            embeddings_dict = self.sentence_embeddings
            title_suffix = shape_arabic_text('(التضمينات المدمجة)')
        else:
            logger.error("Invalid embedding_type specified for visualization. Choose 'quantum' or 'combined'.")
            return None

        if not embeddings_dict:
            logger.warning(f"No '{embedding_type}' embeddings available for visualization."); return None

        embeddings = list(embeddings_dict.values())
        original_indices = list(embeddings_dict.keys())

        if not embeddings: logger.warning("Embeddings list is empty."); return None

        X = np.array(embeddings)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
             logger.warning(f"Non-finite values found in {embedding_type} embeddings. Removing problematic rows for PCA.")
             X = X[finite_mask]
             viz_indices = [idx for i, idx in enumerate(original_indices) if finite_mask[i]]
             if X.shape[0] == 0: logger.error(f"All {embedding_type} embeddings contained non-finite values."); return None
             logger.info(f"Removed {len(finite_mask) - X.shape[0]} non-finite rows before PCA.")
        else:
             viz_indices = original_indices

        if X.shape[1] < 2: logger.warning("Need at least 2 embedding dimensions for PCA."); return None
        if X.shape[0] < 2: logger.warning("Need at least 2 samples for PCA."); return None

        try:
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(X)

            plt.figure(figsize=(12, 10))
            cmap_name = 'viridis' # Default colormap name

            # --- Determine Colors (Handles string labels) ---
            point_colors = 'blue' # Default color if no labels/clusters
            color_values = None   # Will hold numerical values for coloring
            legend_title = None
            unique_labels = None
            label_map = None      # Dictionary to map string labels to numbers

            if labels is not None and len(labels) == len(self.reference_sentences):
                # Filter labels to match the filtered embeddings used in PCA
                filtered_labels = [labels[idx] for idx in viz_indices if idx < len(labels)]

                if len(filtered_labels) == len(reduced_embeddings):
                    unique_labels = sorted(list(set(filtered_labels)))
                    # Create a mapping from unique string labels to integers 0, 1, 2...
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    # Convert the list of string labels to a list of integers
                    color_values = [label_map.get(lbl, -1) for lbl in filtered_labels] # Use -1 for potential errors
                    legend_title = shape_arabic_text("التصنيفات اللغوية")
                    logger.info("Coloring PCA plot by provided linguistic labels (mapped to numbers).")
                    cmap_name = 'tab20' if len(unique_labels) > 10 else 'viridis' # Use a categorical map if many labels
                else:
                     logger.warning("Length mismatch between provided labels and filtered embeddings. Using default colors.")
            elif self.cluster_labels is not None and len(self.cluster_labels) == len(self.reference_sentences):
                 # Fallback to K-Means cluster labels (already numerical)
                 filtered_cluster_labels = [self.cluster_labels[idx] for idx in viz_indices if idx < len(self.cluster_labels) and self.cluster_labels[idx] != -1]
                 if len(filtered_cluster_labels) == len(reduced_embeddings):
                      color_values = filtered_cluster_labels
                      legend_title = shape_arabic_text("عناقيد كمومية (K-Means)")
                      unique_labels = sorted([l for l in np.unique(filtered_cluster_labels) if l != -1]) # Use numbers as labels here
                      logger.info("Coloring PCA plot by K-Means cluster labels (on quantum embeddings).")
                 else:
                      logger.warning("Length mismatch between K-Means labels and filtered embeddings. Using default colors.")

            # --- Create Scatter Plot ---
            # Use color_values (the numbers) for 'c', and specify the colormap
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                                  c=color_values if color_values is not None else point_colors,
                                  cmap=cmap_name if color_values is not None else None,
                                  alpha=0.7, s=100)

            # --- Add Legend (Using original string labels if available) ---
            if legend_title and unique_labels is not None:
                try:
                    # Create legend handles based on the numerical mapping
                    num_unique = len(unique_labels)
                    cmap = cm.get_cmap(cmap_name, num_unique) # Get the colormap instance
                    handles = [plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=cmap(i), markersize=10)
                               for i in range(num_unique)]
                    # Get the original string labels corresponding to the numbers 0..N-1
                    legend_labels_str = [shape_arabic_text(str(lbl)) for lbl in unique_labels] # Shape labels for legend
                    plt.legend(handles, legend_labels_str, title=legend_title)
                except Exception as leg_e:
                    logger.warning(f"Could not create custom legend: {leg_e}")
                    # Fallback legend if custom fails
                    if color_values is not None:
                        try: plt.colorbar(scatter, label=legend_title) # Use colorbar as fallback
                        except: pass # Ignore colorbar error

            # --- Highlight specific points ---
            # (Highlighting logic remains the same)
            if highlight_indices is not None:
                 highlight_plot_indices = [i for i, original_idx in enumerate(viz_indices) if original_idx in highlight_indices]
                 if highlight_plot_indices:
                     plt.scatter(reduced_embeddings[highlight_plot_indices, 0],
                                 reduced_embeddings[highlight_plot_indices, 1],
                                 c='red', s=150, edgecolor='white', zorder=10, label='Highlighted')

            # --- Plot Cluster Centers ---
            # (Cluster center plotting logic remains the same)
            if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
                 if np.all(np.isfinite(self.meaning_clusters)):
                     try:
                         if hasattr(pca, 'components_') and self.meaning_clusters.shape[1] == pca.components_.shape[1]:
                              cluster_centers_2d = pca.transform(self.meaning_clusters)
                              plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker='*', s=350, c='white', edgecolor='black', label=shape_arabic_text('مراكز العناقيد (K-Means)'), zorder=15)
                         else: logger.warning("PCA not fitted correctly or dimension mismatch, cannot transform cluster centers.")
                     except Exception as cc_e: logger.error(f"Error plotting cluster centers: {cc_e}")
                 else: logger.warning("Non-finite values found in cluster centers. Skipping plotting centers.")


            # Shape plot title and axis labels
            main_title = shape_arabic_text('فضاء المعنى (PCA على {})').format(title_suffix)
            plt.title(main_title);
            plt.xlabel(shape_arabic_text(f'المكون الرئيسي 1 ({pca.explained_variance_ratio_[0]:.1%} تباين)'));
            plt.ylabel(shape_arabic_text(f'المكون الرئيسي 2 ({pca.explained_variance_ratio_[1]:.1%} تباين)'))
            plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            if save_path:
                try:
                    plt.savefig(save_path, dpi=150)
                    logger.info(f"Visualization saved to {save_path}")
                except Exception as save_e:
                    logger.error(f"Failed to save PCA plot: {save_e}")
            fig = plt.gcf() # Get current figure
            plt.close(fig) # Close the figure to prevent display issues in some environments
            return fig # Return the figure object
        except Exception as e:
            logger.error(f"Error during PCA visualization: {e}", exc_info=True)
            plt.close() # Ensure plot is closed on error
            return None
        # --- End of visualize_meaning_space ---


    # --- Quantum State Visualization (Unchanged) ---
    def analyze_quantum_states(self, circuits_dict, tokens_dict, analyses_dict, save_path_prefix=None):
        """ Analyze and visualize the quantum states of circuits. """
        # (Keep implementation from v7)
        # ... (Full implementation omitted for brevity - see v7 artifact) ...
        # --- Start of analyze_quantum_states ---
        if not circuits_dict: logger.warning("No circuits for state analysis."); return
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
                parameter_binds = self._bind_parameters(circuit, tokens, analyses); bindings_list = [parameter_binds] if parameter_binds else None
                job = self.simulator.run(circuit, parameter_binds=bindings_list); result = job.result()
                if not result.success: logger.error(f"Statevector sim failed for {circuit_id}. Status: {getattr(result, 'status', 'N/A')}"); i += 1; continue
                if hasattr(result, 'get_statevector'):
                    statevector = result.get_statevector();
                    if plot_state_city: plot_state_city(statevector, title=sentence_label, ax=ax); ax.tick_params(axis='both', which='major', labelsize=8); ax.title.set_size(10)
                    else: ax.text(0.5, 0.5, shape_arabic_text("العرض معطل"), ha='center', va='center', fontsize=9)
                    if save_path_prefix:
                         try:
                              ind_fig = plt.figure(figsize=(8, 6));
                              if plot_state_city: plot_state_city(statevector, title=f"{shape_arabic_text('الحالة:')} {sentence_label}", fig=ind_fig)
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
        return fig
        # --- End of analyze_quantum_states ---


    # --- Reporting (Modified to show both embeddings potentially) ---
    def generate_html_report(self, analysis_results_list):
        """
        Generate an HTML report for the combined approach.
        V8.1 - Displays multi-classifier results and embedding type used.
        """
        report_title = shape_arabic_text('تحليل QNLP للغة العربية V8.1 (تصنيف وتصور)')
        html = f"""<!DOCTYPE html>
        <html dir="rtl" lang="ar"><head><meta charset="UTF-8"><title>{report_title}</title><style>
        body{{font-family:'Tahoma','Arial',sans-serif;margin:20px;line-height:1.6;background-color:#f9f9f9;color:#333}} h1,h2,h3{{color:#0056b3}} h1{{text-align:center;border-bottom:2px solid}} h2{{margin-top:30px;border-bottom:1px solid #eee}}
        .sentence-block{{background-color:#fff;padding:15px;margin-bottom:15px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
        .sentence-text{{font-weight:bold;font-size:1.1em;color:#333;margin-bottom:10px}} .analysis-section{{margin-bottom:10px;padding-left:10px;border-left:3px solid #eee}}
        strong{{color:#0056b3}} code{{background-color:#e9ecef;padding:2px 5px;border-radius:4px;font-family:monospace;font-size:0.8em;word-break:break-all;direction:ltr;text-align:left;display:inline-block}}
        pre{{background-color:#f0f0f0;padding:10px;border-radius:5px;white-space:pre-wrap;word-wrap:break-word;direction:ltr;text-align:left}} .error{{color:#dc3545;border-left-color:#dc3545}}
        .label{{background-color:#d1ecf1;color:#0c5460;padding:2px 6px;border-radius:4px;font-size:0.9em;display:inline-block;margin-left:10px}}
        .classifier-result {{ border: 1px solid #ddd; padding: 10px; margin-top: 10px; border-radius: 5px; background-color: #f8f9fa; }}
        </style></head><body><h1>{report_title}</h1>"""

        # --- Classification Results ---
        # analysis_results_list[0] might not exist if pipeline failed early
        classification_results = analysis_results_list[0].get('classification_summary', {}) if analysis_results_list else {}

        if classification_results: # Check if the dictionary is not empty
            # Determine embedding type used from the first result (assuming consistency)
            first_classifier_name = next(iter(classification_results)) # Get the name of the first classifier
            embedding_type_used = classification_results[first_classifier_name].get('embedding_type_used', 'unknown')
            embedding_type_ar = shape_arabic_text('التضمينات الكمومية') if embedding_type_used == 'quantum' else shape_arabic_text('التضمينات المدمجة')
            class_results_title = shape_arabic_text(f'نتائج التصنيف (على {embedding_type_ar})')

            html += f"<h2>{class_results_title}</h2><div class='sentence-block'>"
            classifier_label_ar = shape_arabic_text('المصنف:')
            accuracy_label_ar = shape_arabic_text('الدقة:')
            report_label_ar = shape_arabic_text('تقرير:')

            # Loop through results for each classifier
            for classifier_name, summary in classification_results.items():
                html += f"<div class='classifier-result'>"
                html += f"<h3>{shape_arabic_text('المصنف:')} {classifier_name}</h3>" # Use shaped label
                if 'error' in summary:
                    html += f"<p><strong>{shape_arabic_text('خطأ:')}</strong> {shape_arabic_text(summary['error'])}</p>"
                else:
                    accuracy_value = summary.get('accuracy', 'N/A')
                    accuracy_formatted = f"{accuracy_value:.4f}" if isinstance(accuracy_value, float) else accuracy_value
                    html += f"<p><strong>{accuracy_label_ar}</strong> {accuracy_formatted}</p>"
                    html += f"<h4>{report_label_ar}</h4><pre>{summary.get('report', 'N/A')}</pre>"
                html += f"</div>" # Close classifier-result div
            html += "</div>" # Close sentence-block for classification

        # --- Individual Sentences ---
        individual_analysis_title = shape_arabic_text('تحليل الجمل الفردية')
        html += f"<h2>{individual_analysis_title}</h2>"

        for i, analysis_result in enumerate(analysis_results_list):
            # (Rest of the individual sentence loop remains the same as the previous version)
            # Shape dynamic Arabic content (sentence, label)
            sentence = shape_arabic_text(analysis_result.get('sentence', 'N/A'))
            label = shape_arabic_text(analysis_result.get('label', 'N/A'))
            interpretation_data = analysis_result.get('interpretation', {})

            # Shape static labels used within the loop
            sentence_index_label = shape_arabic_text(f"الجملة {i+1}")
            classification_label_text = shape_arabic_text("التصنيف:")
            error_label = shape_arabic_text("خطأ:")
            structure_label = shape_arabic_text("البنية:")
            q_embedding_label = shape_arabic_text("التضمين الكمومي:")
            c_embedding_label = shape_arabic_text("التضمين المدمج:")
            cluster_label = shape_arabic_text("أقرب عنقود (كمومي):")
            discourse_label = shape_arabic_text("علاقات الخطاب:")

            html += f'<div class="sentence-block">'
            html += f'<div class="sentence-text">{sentence_index_label}: {sentence}'
            html += f'<span class="label">{classification_label_text} {label}</span>'
            html += f'</div>'

            error_msg = interpretation_data.get('error')
            if error_msg:
                html += f'<div class="analysis-section error"><h3>{error_label}</h3>{shape_arabic_text(str(error_msg))}</div></div>'
                continue

            html += f'<div class="analysis-section"><strong>{structure_label}</strong> {interpretation_data.get("structure", "N/A")}</div>'

            q_emb = interpretation_data.get('quantum_embedding')
            if q_emb is not None:
                q_emb_str = np.array2string(q_emb, precision=3, separator=", ", threshold=20, edgeitems=5)
                html += f'<div class="analysis-section"><strong>{q_embedding_label}</strong><br><code>{q_emb_str}</code></div>'

            c_emb = interpretation_data.get('combined_embedding')
            if c_emb is not None:
                c_emb_str = np.array2string(c_emb, precision=3, separator=", ", threshold=20, edgeitems=5)
                html += f'<div class="analysis-section"><strong>{c_embedding_label}</strong><br><code>{c_emb_str}</code></div>'

            cluster_id = interpretation_data.get('nearest_cluster_id')
            if cluster_id is not None:
                cluster_desc = shape_arabic_text(interpretation_data.get("nearest_cluster_desc", "N/A"))
                html += f'<div class="analysis-section"><strong>{cluster_label}</strong> {cluster_id} - {cluster_desc}</div>'

            disc_rels = interpretation_data.get('discourse_relations')
            if disc_rels:
                formatted_rels = self.format_discourse_relations(disc_rels)
                html += f'<div class="analysis-section"><strong>{discourse_label}</strong><br>{formatted_rels}</div>'

            html += '</div>'

        html += """</body></html>"""
        return html

    def generate_similarity_heatmap(embeddings: Dict[int, np.ndarray],
                                sentences: Dict[int, str], # Use kernel's reference_sentences dict
                                title: str,
                                save_path: str):
        """
        Generates and saves a cosine similarity heatmap for given embeddings.
        Uses kernel's reference_sentences for labels.
        """
        logger.info(f"Generating similarity heatmap: {title}")
        if not SKLEARN_AVAILABLE: logger.error("Skipping heatmap: scikit-learn not available."); return
        if not embeddings or len(embeddings) < 2:
            logger.warning(f"Skipping heatmap '{title}': Not enough valid embeddings ({len(embeddings)}).")
            return

        # Get embeddings and corresponding labels in a consistent order
        processed_indices = sorted(embeddings.keys())
        embeddings_list = [embeddings[idx] for idx in processed_indices if isinstance(embeddings[idx], np.ndarray)]
        # Get sentence labels using the index from the kernel's reference dict
        labels_list = [sentences.get(idx, f"Sent_{idx}") for idx in processed_indices]

        # Filter out non-finite values AFTER aligning
        X_full = np.array(embeddings_list)
        finite_mask = np.all(np.isfinite(X_full), axis=1)
        if not np.all(finite_mask):
            num_removed = np.sum(~finite_mask)
            logger.warning(f"Removing {num_removed} non-finite rows for heatmap '{title}'.")
            X = X_full[finite_mask]
            labels_list = [label for i, label in enumerate(labels_list) if finite_mask[i]]
            if X.shape[0] < 2:
                logger.warning(f"Skipping heatmap '{title}': Not enough finite samples after filtering.")
                return
        else:
            X = X_full # Use all if they are finite

        if X.shape[0] < 2: # Check again after filtering
            logger.warning(f"Skipping heatmap '{title}': Not enough samples ({X.shape[0]}).")
            return

        # Calculate cosine similarity
        try:
            similarity_matrix = pairwise.cosine_similarity(X)
        except Exception as e_sim:
            logger.error(f"Error calculating similarity matrix for '{title}': {e_sim}")
            return

        # Plot heatmap
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(max(8, X.shape[0]*0.5), max(6, X.shape[0]*0.4)))
            im = ax.imshow(similarity_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
            ax.set_title(title)
            short_labels = [shape_arabic_text(s[:20] + ('...' if len(s) > 20 else '')) for s in labels_list]
            ax.set_xticks(np.arange(len(short_labels)))
            ax.set_yticks(np.arange(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=8)
            ax.set_yticklabels(short_labels, fontsize=8)
            fig.tight_layout()
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved heatmap: {save_path}")
        except Exception as e_plot:
            logger.error(f"Error plotting/saving heatmap '{title}': {e_plot}", exc_info=True)
        finally:
            if fig: plt.close(fig)

    # --- Utility Methods (Save/Load - Modified to handle both embeddings) ---
    def save_model(self, filename: str = 'arabic_quantum_kernel_v8.pkl'):
        """ Saves the kernel state (embeddings, config) to a file. V8 """
        logger.info(f"Saving kernel state (v8) to {filename}...")
        model_data = {
            'embedding_dim': self.embedding_dim, 'num_clusters': self.num_clusters,
            'params_per_word': self.params_per_word, 'morph_param_weight': self.morph_param_weight,
            'combine_feature_weight': self.combine_feature_weight, 'shots': self.shots,
            '_embedding_model_path': getattr(self, '_embedding_model_path', None),
            'reference_sentences': self.reference_sentences,
            'circuit_embeddings': self.circuit_embeddings, # Save quantum embeddings
            'sentence_embeddings': self.sentence_embeddings, # Save combined embeddings
            'kmeans_model': self.kmeans_model, # Save fitted KMeans model
            'cluster_labels': self.cluster_labels, 'meaning_map': self.meaning_map,
        }
        try:
            with open(filename, 'wb') as f: pickle.dump(model_data, f)
            logger.info(f"Kernel state saved successfully.")
        except Exception as e: logger.error(f"Error saving kernel state: {e}", exc_info=True)

    def load_model(self, filename: str = 'arabic_quantum_kernel_v8.pkl'):
        """ Loads a kernel state from a file. V8 """
        if not os.path.exists(filename): logger.error(f"Error: Kernel state file {filename} not found."); return self
        logger.info(f"Loading kernel state (v8) from {filename}...")
        try:
            with open(filename, 'rb') as f: model_data = pickle.load(f)
            saved_model_path = model_data.get('_embedding_model_path')
            self.__init__(
                embedding_dim=model_data.get('embedding_dim', 30),
                num_clusters=model_data.get('num_clusters', 5),
                embedding_model_path=saved_model_path,
                params_per_word=model_data.get('params_per_word', 3),
                shots=model_data.get('shots', 8192),
                morph_param_weight=model_data.get('morph_param_weight', 0.3),
                combine_feature_weight=model_data.get('combine_feature_weight', 0.5)
            )
            self.reference_sentences = model_data.get('reference_sentences', {})
            self.circuit_embeddings = model_data.get('circuit_embeddings', {})
            self.sentence_embeddings = model_data.get('sentence_embeddings', {})
            self.kmeans_model = model_data.get('kmeans_model')
            self.cluster_labels = model_data.get('cluster_labels')
            self.meaning_map = model_data.get('meaning_map', {})
            logger.info(f"Kernel state loaded successfully.")
        except Exception as e: logger.error(f"Error loading kernel state: {e}", exc_info=True)
        return self
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
    def format_discourse_relations(self, discourse_relations):
        """ Creates a user-friendly description of discourse relations. """
        if not discourse_relations: return shape_arabic_text("لم يتم اكتشاف علاقات خطاب محددة.") # Shape default msg
        formatted_output = []
        # Descriptions are already in Arabic, just need shaping before returning
        descriptions = {
            'CONTINUATION': "تواصل هذه الجملة الفكرة السابقة باستخدام '{}'", 'CAUSE': "تظهر هذه الجملة نتيجة أو عاقبة للجملة السابقة باستخدام '{}'",
            'CONTRAST': "تتناقض هذه الجملة مع المعلومات السابقة باستخدام '{}'", 'ELABORATION': "توضح هذه الجملة المعلومات السابقة باستخدام '{}'",
            'EXAMPLE': "تقدم هذه الجملة مثالاً على المفهوم السابق باستخدام '{}'", 'CONDITION': "تحدد هذه الجملة شرطًا متعلقًا بالجملة السابقة باستخدام '{}'",
            'TEMPORAL': "تحدد هذه الجملة علاقة زمنية مع الجملة السابقة باستخدام '{}'", 'REFERENCE': "تشير هذه الجملة إلى المحتوى السابق باستخدام '{}'"
        }
        for relation in discourse_relations:
            rel_type = relation.get('type', 'UNKNOWN'); marker = relation.get('marker', '')
            desc_template = descriptions.get(rel_type, f"تم اكتشاف علاقة من نوع {rel_type} باستخدام '{marker}'")
            # Shape the marker AND the template
            formatted_output.append(shape_arabic_text(desc_template.format(shape_arabic_text(marker))))
        return "\n".join(formatted_output) # Already shaped        
# ============================================
# Example Pipeline Function (V8 - Combined Approach)
# ============================================
def prepare_quantum_nlp_pipeline_v8(
    experiment_sets: Dict[str, List[Dict[str, str]]],
    max_sentences_per_set: int = 50,
    embedding_model_path: Optional[str] = None,
    ansatz_choice: str = 'IQP',
    n_layers_strong: int = 2, cnot_ranges: Optional[List[int]] = None,
    n_layers_iqp: int = 1, n_single_qubit_params_iqp: int = 3,
    run_clustering_viz: bool = True,
    run_state_viz: bool = False,
    run_classification: bool = True,
    classification_embedding_type: str = 'quantum', # <-- New parameter ('quantum' or 'combined')
    run_heatmap_viz: bool = True, # Added heatmap flag based on exp4.py usage
    output_dir_base: str = "qnlp_pipeline_v8_output"
    ) -> Tuple[Optional[ArabicQuantumMeaningKernel], List[Dict], Optional[Dict[str, Dict]]]:
    """
    Pipeline using ArabicQuantumMeaningKernel V8.
    Generates quantum & combined embeddings, performs classification on quantum,
    and visualizes quantum embeddings. Handles labeling based on detailed input structure.

    Args:
        experiment_sets (Dict[str, List[Dict[str, str]]]):
            Dictionary mapping a set name to a list of dictionaries.
            Each inner dictionary must have keys 'sentence' (str) and 'label' (str).
            The 'label' should be specific to the feature being tested.
            Example:
            {
                "WordOrder": [
                    {"sentence": "الولد يقرأ الكتاب.", "label": "WordOrder_SVO"},
                    {"sentence": "يقرأ الولد الكتاب.", "label": "WordOrder_VSO"}
                    # ... more pairs
                ],
                "LexicalAmbiguity": [
                    {"sentence": "جاء الرجل الطويل.", "label": "Ambiguity_Man"},
                    {"sentence": "انكسرت رجل الكرسي.", "label": "Ambiguity_Leg"}
                    # ... more pairs
                ],
                "Morphology": [
                    {"sentence": "المهندس يعمل.", "label": "Morph_Sg"},
                    {"sentence": "المهندسون يعملون.", "label": "Morph_Pl"},
                    {"sentence": "كتب الطالب.", "label": "Morph_Past"},
                    {"sentence": "يكتب الطالب.", "label": "Morph_Pres"}
                    # ... more pairs
                ]
            }
        max_sentences_per_set (int): Max sentences to process from each set list.
        embedding_model_path (str, optional): Path to Word2Vec model.
        ansatz_choice (str): Lambeq ansatz type ('IQP', 'STRONGLY_ENTANGLING').
        n_layers_strong, cnot_ranges: Params for StronglyEntanglingAnsatz.
        n_layers_iqp, n_single_qubit_params_iqp: Params for IQPAnsatz.
        run_clustering_viz, run_state_viz, run_classification: Flags.
        output_dir_base (str): Base directory for saving outputs.


    Returns:
        Tuple: (kernel, List[Dict], Optional[Dict])
               - kernel: The trained ArabicQuantumMeaningKernel instance.
               - analysis_results_list: List of dicts, one per sentence.
               - classification_summary: Results from the classification task, or None.
    """
    # --- Initial Checks ---
    # Assume necessary imports like logging, os, Counter, lambeq, sklearn, etc. are present above
    # Assume ArabicQuantumMeaningKernel class definition is present above
    # Assume arabic_to_quantum_enhanced function is imported correctly
    # --- Initial Checks ---
    if not LAMBEQ_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.critical("Exiting pipeline: Lambeq or scikit-learn not found.")
        return None, [], None
    if classification_embedding_type not in ['quantum', 'combined']:
        logger.error("Invalid classification_embedding_type. Defaulting to 'quantum'.")
        classification_embedding_type = 'quantum'

    # --- Sentence Processing Loop ---
    # (Sentence processing logic remains the same as before - omitted for brevity)
    all_sentences = []
    all_labels = []
    all_circuits = []
    all_tokens = []
    all_analyses = []
    all_structures = []
    all_roles = []
    processed_indices = []
    circuits_for_viz_dict: Dict[int, QuantumCircuit] = {}
    logger.info("\n--- Loading Sentences and Specific Labels ---")
    global_idx = 0
    current_overall_sentence_index = 0
    successfully_added_to_lists_count = 0
    for set_name, sentence_data_list in experiment_sets.items():
        logger.info(f"Processing set '{set_name}'...")
        set_output_dir = os.path.join(output_dir_base, set_name)
        os.makedirs(set_output_dir, exist_ok=True)
        count = 0
        for sentence_data in sentence_data_list:
            if count >= max_sentences_per_set: break
            sentence = sentence_data.get("sentence")
            specific_label = sentence_data.get("label")
            if not sentence or not specific_label:
                logger.warning(f"Skipping entry in set '{set_name}': {sentence_data}")
                global_idx += 1; continue

            logger.debug(f" Processing sentence {global_idx}: '{current_overall_sentence_index}...' (Label: {specific_label})")
            try:
                returned_values = arabic_to_quantum_enhanced(
                    sentence, ansatz_choice=ansatz_choice,
                    n_layers_ent=n_layers_strong, cnot_ranges=cnot_ranges,
                    n_layers_iqp=n_layers_iqp, n_single_qubit_params_iqp=n_single_qubit_params_iqp,
                    debug=False, output_dir=set_output_dir
                )
                circuit, _, structure, tokens, analyses_dicts, roles = returned_values

                if circuit is None:
                    logger.warning(f"  Skipping sentence {global_idx}: Invalid circuit."); global_idx += 1; continue
                    logger.error(f"V8_PIPELINE: Circuit is None for sentence (Overall Index {current_overall_sentence_index}): '{sentence}'. Skipping add to lists.")
                else:
                    all_sentences.append(sentence); all_labels.append(specific_label)
                    all_circuits.append(circuit); all_tokens.append(tokens)
                    all_analyses.append(analyses_dicts); all_structures.append(structure)
                    all_roles.append(roles); processed_indices.append(global_idx)
                    circuits_for_viz_dict[global_idx] = circuit
                    successfully_added_to_lists_count += 1
                count += 1; global_idx += 1
            except Exception as e:
                logger.error(f"Error processing sentence {global_idx}: {e}", exc_info=False); global_idx += 1
            current_overall_sentence_index += 1
    logger.info(f"Successfully processed {successfully_added_to_lists_count} sentences into circuits out of {current_overall_sentence_index} total input items.")
    if not all_circuits: logger.error("No circuits processed."); return None, [], None
    logger.info(f"Successfully processed {len(all_circuits)} sentences into circuits.")
    logger.info(f"Specific labels collected: {Counter(all_labels)}")

    # --- Instantiate Kernel ---
    kernel_embedding_dim = 30; kernel_clusters = min(5, max(1, len(all_circuits) // 3))
    kernel = ArabicQuantumMeaningKernel(embedding_dim=kernel_embedding_dim, num_clusters=kernel_clusters, embedding_model_path=embedding_model_path, morph_param_weight=0.3)

    # --- Generate Embeddings ---
    kernel.train(sentences=all_sentences, circuits=all_circuits, tokens_list=all_tokens, analyses_list=all_analyses, structures=all_structures, roles_list=all_roles)

    # --- Create final analysis list ---
    analysis_results_list = []
    previous_analysis = None
    for i in range(len(all_sentences)):
         # Ensure interpret_sentence exists and handles context correctly
         interpretation_dict = kernel.interpret_sentence(all_circuits[i], all_tokens[i], all_analyses[i], all_structures[i], all_roles[i], previous_analysis_dict=previous_analysis)
         current_analysis = {
             'sentence': all_sentences[i], 'label': all_labels[i],
             'original_index': processed_indices[i], 'structure': all_structures[i],
             'roles': all_roles[i], 'tokens': all_tokens[i],
             'interpretation': interpretation_dict, 'circuit': all_circuits[i]
         }
         analysis_results_list.append(current_analysis)
         previous_analysis = current_analysis # Update context for next iteration

    # --- Classification Evaluation ---
    classification_summary = None # Now expects a dict of dicts
    if run_classification:
        if len(set(all_labels)) < 2:
            logger.warning("Skipping classification: Need >= 2 distinct labels.")
        else:
             # Pass the new embedding_type parameter
             classification_summary = kernel.evaluate_classification_accuracy(
                 labels=all_labels,
                 embedding_type=classification_embedding_type # <-- Pass the type
             )
             if classification_summary:
                 logger.info(f"Classification evaluation complete using '{classification_embedding_type}' embeddings.")
                 # Add summary to the first result item for reporting convenience
                 if analysis_results_list:
                     analysis_results_list[0]['classification_summary'] = classification_summary
             else:
                 logger.error("Classification evaluation failed.")

    # --- Visualizations ---
    # (Clustering/PCA Viz logic remains the same, typically uses quantum embeddings)
    if run_clustering_viz:
        logger.info("\n--- Visualizing Meaning Space (Quantum Embeddings, Colored by Specific Label) ---")
        try:
            pca_path = os.path.join(output_dir_base, f"combined_pca_quantum_by_specific_label_{ansatz_choice}.png")
            meaning_space_fig = kernel.visualize_meaning_space(embedding_type='quantum', labels=all_labels, save_path=pca_path)
            if meaning_space_fig: plt.close(meaning_space_fig)
        except Exception as vis_e: logger.error(f"Error during PCA visualization: {vis_e}", exc_info=True)

    # --- Heatmap Visualization Call ---
    # (Heatmap logic remains the same, uses quantum embeddings)
    if run_heatmap_viz:
        logger.info("\n--- Generating Similarity Heatmap (Quantum Embeddings) ---")
        try:
            heatmap_path = os.path.join(output_dir_base, f"combined_heatmap_quantum_{ansatz_choice}.png")
            if kernel.circuit_embeddings and kernel.reference_sentences:
                # Assuming generate_similarity_heatmap helper exists elsewhere or is imported
                # generate_similarity_heatmap(...) # Call the helper
                logger.info(f"Heatmap generation skipped (helper function call commented out). Path: {heatmap_path}") # Placeholder log
            else:
                 logger.warning("Skipping heatmap: Missing embeddings or reference sentences in kernel.")
        except NameError:
             logger.warning("Skipping heatmap: generate_similarity_heatmap function not found.")
        except Exception as heat_e:
            logger.error(f"Error generating heatmap: {heat_e}", exc_info=True)


    # --- State Visualization ---
    # (State Viz logic remains the same)
    if run_state_viz:
         logger.info("\n--- Visualizing Quantum States ---")
         # (Code as before to prepare dicts and call kernel.analyze_quantum_states)
         logger.info("State visualization skipped (implementation details omitted in this update).") # Placeholder log


    # --- Reporting ---
    logger.info("\n--- Generating HTML Report ---")
    try:
        report_path = os.path.join(output_dir_base, f'combined_report_{ansatz_choice}.html')
        # generate_html_report now handles the new classification_summary structure
        html_report = kernel.generate_html_report(analysis_results_list)
        with open(report_path, 'w', encoding='utf-8') as f: f.write(html_report)
        logger.info(f"HTML report saved to {report_path}")
    except Exception as report_e:
        # Log the specific error during HTML generation
        logger.error(f"Error generating HTML report: {report_e}", exc_info=True)
        # Also log if the kernel object itself is None
        if kernel is None:
             logger.error("HTML report generation failed because the kernel object is None.")


    # --- Save Final Kernel State ---
    if kernel: # Check if kernel exists before saving
        kernel_save_path = os.path.join(output_dir_base, f'arabic_quantum_kernel_v8_{ansatz_choice}.pkl')
        kernel.save_model(kernel_save_path)
    else:
        logger.error("Skipping kernel save because kernel object is None.")


    logger.info("\n--- Pipeline Finished ---")
    return kernel, analysis_results_list, classification_summary

# ============================================
# Main execution block (Example)
# ============================================
if __name__ == "__main__":
    # --- Configuration ---
    # Define experiment sets directly (as in exp3.py)
    experiment_sets_data = {
        "WordOrder": [
            "الولدُ يقرأُ الكتابَ .", "البنتُ تشربُ الحليبَ .", "الطالبُ كتبَ الدرسَ .",
            "المعلمُ شرحَ القاعدةَ .", "الطبيبُ فحصَ المريضَ .", "المهندسُ بنى البيتَ .",
            "القطةُ أكلت السمكَ .", "الطفلُ يلعبُ بالكرةِ .", "السيارةُ سريعةٌ .",
            "السماءُ صافيةٌ .", "يقرأُ الولدُ الكتابَ .", "تشربُ البنتُ الحليبَ .",
            "كتبَ الطالبُ الدرسَ .", "شرحَ المعلمُ القاعدةَ .", "فحصَ الطبيبُ المريضَ .",
            "بنى المهندسُ البيتَ .", "أكلت القطةُ السمكَ .", "يلعبُ الطفلُ بالكرةِ .",
        ],
        "LexicalAmbiguity": [
            "جاء الرجلُ الطويلُ .", "الرجلُ القويُ يعملُ .", "تحدثَ الرجلُ الحكيمُ .",
            "هذا الرجلُ طبيبٌ .", "رأيتُ رجلاً في السوقِ .", "الرجلُ العجوزُ جلسَ .",
            "سألَ الرجلُ سؤالاً .", "أعطى الرجلُ المالَ .", "ابتسمَ الرجلُ السعيدُ .",
            "يسافرُ الرجلُ غداً .", "انكسرتْ رجلُ الكرسيِّ .", "للطاولةِ أربعُ أرجلٍ .",
            "عالجَ الطبيبُ رجلَ المريضِ .", "وضعَ الكتابَ على رجلِهِ .", "رجلُ السريرِ مكسورةٌ .",
            "أصيبتْ رجلُ اللاعبِ .", "لا تلمسْ رجلَ المكتبِ .", "تحتاجُ الطاولةُ إلى رجلٍ جديدةٍ .",
            "ألمُ الرجلِ شديدٌ .", "طولُ رجلِ الزرافةِ كبيرٌ ."
        ],
        "Morphology": [
            "المهندسُ يعملُ .", "المهندسونَ يعملونَ .", "الطالبةُ تدرسُ .",
            "الطالباتُ يدرسنَ .", "الكتابُ جديدٌ .", "الكتبُ جديدةٌ .",
            "السيارةُ سريعةٌ .", "السياراتُ سريعةٌ .", "البيتُ كبيرٌ .",
            "الغرفةُ كبيرةٌ .", "الولدُ ذكيٌّ .", "البنتُ ذكيةٌ .",
            "كتبَ الطالبُ .", "يكتبُ الطالبُ .", "قرأت البنتُ .",
            "تقرأُ البنتُ .", "هذا كتابي .", "هذا كتابكَ .",
            "هذا قلمهُ .", "هذا قلمها ."
        ]
    }


    logger.info("Transforming experiment data...")
    transformed_experiment_data = transform_data(experiment_sets_data)
    # Path to optional Word2Vec model
    model_path = None # Example: Use hash-based params

    # Choose Ansatz and parameters
    # ansatz = 'IQP'
    ansatz = 'STRONGLY_ENTANGLING'
    iqp_layers = 3
    strong_layers = 2 # Keep this lower for feasibility
    strong_ranges = None # Use default ranges

    # --- Run the Pipeline ---
    kernel_v8, final_results, classification_info = prepare_quantum_nlp_pipeline_v8(
        experiment_sets=transformed_experiment_sets_data,
        max_sentences_per_set=50, # Process all sentences in this example
        embedding_model_path=model_path,
        ansatz_choice=ansatz,
        n_layers_iqp=iqp_layers,
        n_layers_strong=strong_layers,
        cnot_ranges=strong_ranges,
        run_clustering_viz=True,
        run_state_viz=False, # Usually keep False unless debugging states
        run_classification=True,
        output_dir_base=f"qnlp_pipeline_v8_output_{ansatz}" # Separate output per ansatz
    )

    # --- Output Summary ---
    if kernel_v8 is None:
        logger.error("\nPipeline execution failed.")
    else:
        logger.info(f"\nPipeline completed. Processed {len(final_results)} sentences.")
        if classification_info:
            logger.info("\n--- Classification Summary ---")
            logger.info(f" Classifier: {classification_info.get('classifier')}")
            logger.info(f" Accuracy: {classification_info.get('accuracy'):.4f}")
            logger.info(f" Report:\n{classification_info.get('report')}")
        elif run_classification:
             logger.warning("\nClassification task was enabled but did not produce results.")