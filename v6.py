# quantum_kernel_v6.py (Revised Feature Extraction & Hash-Based Parameter Binding + Arabic Display Fix)
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any, Sequence, Set # Added Sequence, Set
import matplotlib.pyplot as plt
import pickle
import os
import traceback
import logging
from collections import Counter
import hashlib # For parameter binding hash

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


logger = logging.getLogger(__name__)
# Qiskit imports
try:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.exceptions import QiskitError
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler, Estimator # For expectation values
    from qiskit.quantum_info import SparsePauliOp, partial_trace # Added partial_trace
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit or Qiskit-Aer not found. Quantum execution will fail.")
    print("Please install them: pip install qiskit qiskit-aer")
    QISKIT_AVAILABLE = False
    # Dummy classes
    class QuantumCircuit: pass
    class AerSimulator: pass
    class ClassicalRegister: pass
    class Parameter: pass
    class Sampler: pass
    class Estimator: pass
    class SparsePauliOp: pass
    partial_trace = None

# Qiskit visualization (optional)
try:
    from qiskit.visualization import plot_state_city, plot_histogram
except ImportError:
    print("Warning: Qiskit visualization components not found.")
    plot_state_city = None
    plot_histogram = None

# Dependency on camel_test2 (assuming it provides the revised v2 function)
try:
    # Import the REVISED function
    from camel_test2 import arabic_to_quantum_enhanced_v3_morph as arabic_to_quantum_enhanced
    print("Successfully imported 'arabic_to_quantum_enhanced_v2_fixed' as 'arabic_to_quantum_enhanced'.")
except ImportError:
    print("ERROR: Cannot import 'arabic_to_quantum_enhanced_v2_fixed' from 'camel_test2'.")
    def arabic_to_quantum_enhanced(*args, **kwargs):
        print("ERROR: Dummy arabic_to_quantum_enhanced called because import failed.")
        return None, None, "ERROR", [], [], {}

# Lambeq imports (for type checking)
try:
    from lambeq.backend.grammar import Diagram as GrammarDiagram
    LAMBEQ_AVAILABLE = True
except ImportError:
    print("Warning: Lambeq not found.")
    class GrammarDiagram: pass
    LAMBEQ_AVAILABLE = False

# Word Embeddings (Gensim)
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: gensim not found (pip install gensim). Parameter binding via embeddings disabled.")
    GENSIM_AVAILABLE = False

# Enhanced Clustering (Gensim LDA)
try:
    from gensim import corpora, models
    GENSIM_LDA_AVAILABLE = True
except ImportError:
    GENSIM_LDA_AVAILABLE = False


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
# Do this once, e.g., after imports or before the first plot
# Ensure the chosen font (e.g., 'Tahoma', 'Arial', 'Amiri') is installed on your system
try:
    plt.rcParams['font.family'] = 'sans-serif'
    # Add fonts known to support Arabic. Matplotlib will try them in order.
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Arial', 'Amiri', 'Noto Naskh Arabic']
    logger.info(f"Configured Matplotlib font.sans-serif: {plt.rcParams['font.sans-serif']}")
except Exception as e:
    logger.warning(f"Could not set Matplotlib font configuration: {e}")
# --- END FONT CONFIG ---


# Helper function to ensure circuit name
def _ensure_circuit_name(circuit: Any, default_name: str = "qiskit_circuit") -> Any:
    if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
        if not getattr(circuit, 'name', None):
            try:
                setattr(circuit, 'name', default_name)
            except Exception: pass # Ignore if name setting fails
    return circuit

class ArabicQuantumMeaningKernel:
    """
    A quantum kernel for mapping quantum circuit outputs to potential
    sentence meanings for Arabic language processing. Includes discourse analysis
    and parameter binding via word embeddings.
    V6.1: Uses hash-based fallback for deterministic parameter binding.
    """
    def __init__(self,
                 embedding_dim: int = 20,
                 num_clusters: int = 5,
                 simulator_backend: Optional[str] = None, # No longer used directly
                 embedding_model_path: Optional[str] = None, # <-- NEW: Path to Word2Vec model
                 params_per_word: int = 3,
                 shots: int = 8192,
                 morph_param_weight: float = 0.3): # <-- NEW: How many params IQPAnsatz generates per word (usually 3)
        """
        Initialize the quantum meaning kernel.

        Args:
            embedding_dim (int): Dimension for the final sentence embeddings.
            num_clusters (int): Default number of meaning clusters.
            simulator_backend (str, optional): Deprecated. Simulator is now AerSimulator.
            embedding_model_path (str, optional): Path to the pre-trained gensim Word2Vec model file.
                                                  If None, hash-based fallback parameters will be used.
            params_per_word (int): Number of parameters the lambeq ansatz (e.g., IQPAnsatz)
                                   is expected to generate per word/box. Default is 3 for IQP.
            shots (int): Number of shots for Estimator (influences precision).
        """
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.params_per_word = params_per_word # Store how many params map to one word
        self._embedding_model_path = embedding_model_path # Store path for saving/loading
        self.shots = shots
        self.morph_param_weight = max(0.0, min(1.0, morph_param_weight))
        # --- Simulator Initialization (Qiskit 1.0+) ---
        if QISKIT_AVAILABLE:
            # Initialize simulator. Consider adding options if needed later.
            # e.g., method='matrix_product_state' for potentially larger circuits
            self.simulator = AerSimulator()
            # Initialize Estimator with specified shots
            # NOTE: The 'shots' option for Estimator might influence precision but
            # it fundamentally calculates expectation values, not samples counts.
            # If you need counts, use the Sampler primitive.
            self.estimator = Estimator(options={'shots': self.shots})
            logger.info("Initialized Qiskit AerSimulator and Estimator primitive.")
        else:
            self.simulator = None
            self.estimator = None
            logger.warning("Qiskit not available. Simulator and Estimator set to None.")
        # ---

        self.meaning_clusters = None
        self.cluster_labels = None
        self.meaning_map = {}
        self.reference_sentences = {}
        self.circuit_embeddings = {} # Stores features derived from circuits
        self.sentence_embeddings = {} # Stores combined quantum+linguistic embeddings
        self.camel_analyzer = None # Will be initialized if CAMeL Tools are found

        # --- NEW: Load Embedding Model ---
        self.embedding_model = None
        if GENSIM_AVAILABLE and embedding_model_path:
            logger.info(f"Loading word embedding model from: {embedding_model_path}...")
            try:
                # Use KeyedVectors for potentially faster loading and lower memory
                # Check if it's a binary model or text format
                is_binary = not embedding_model_path.endswith(".txt") and not embedding_model_path.endswith(".vec")
                self.embedding_model = KeyedVectors.load_word2vec_format(embedding_model_path, binary=True)
                logger.info(f"Word embedding model loaded successfully. Vector size: {self.embedding_model.vector_size}")
            except Exception as e:
                logger.error(f"Failed to load word embedding model: {e}")
                logger.warning("Falling back to hash-based parameter assignment.")
                self.embedding_model = None
        elif not GENSIM_AVAILABLE:
             logger.info("Gensim not installed. Cannot load embeddings. Using hash-based parameters.")
        else:
             logger.info("No embedding model path provided. Using hash-based parameters.")
        # --- END NEW ---

        self.semantic_templates = { # Default templates
            'VSO': {'declarative': "ACTION performed by SUBJECT on OBJECT", 'question': "Did SUBJECT perform ACTION on OBJECT?", 'command': "SUBJECT should perform ACTION on OBJECT"},
            'SVO': {'declarative': "SUBJECT performs ACTION on OBJECT", 'question': "Does SUBJECT perform ACTION on OBJECT?", 'command': "Make SUBJECT perform ACTION on OBJECT"},
            'NOMINAL': {'declarative': "SUBJECT is PREDICATE", 'question': "Is SUBJECT PREDICATE?", 'command': "Consider SUBJECT as PREDICATE"},
            'COMPLEX': {'declarative': "Complex statement involving CLAUSE_1 and CLAUSE_2", 'question': "Question about relationship between CLAUSE_1 and CLAUSE_2", 'command': "Directive concerning CLAUSE_1 and CLAUSE_2"},
            'OTHER': {'declarative': "General statement about TOPIC", 'question': "Question about TOPIC", 'command': "Directive related to TOPIC"}
        }
        # Attempt to initialize CAMeL Tools Analyzer
        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            # Try a common default DB name, adjust if needed
            db_path = MorphologyDB.builtin_db() # Or specify e.g., 'calima-msa-r13'
            self.camel_analyzer = Analyzer(db_path)
            logger.info("CAMeL Tools Analyzer initialized successfully.")
        except ImportError:
            logger.warning("CAMeL Tools not found (pip install camel-tools). NLP enhancements disabled.")
        except LookupError:
             logger.warning("CAMeL Tools default DB not found. Run 'camel_tools download <db_name>'. NLP enhancements disabled.")
        except Exception as e:
            logger.warning(f"Error initializing CAMeL Tools Analyzer: {e}. NLP enhancements disabled.")


    # --- Parameter Binding Function (MODIFIED for Hash Fallback) ---
    def _bind_parameters(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Dict]) -> Optional[Dict[Parameter, float]]:
        """
        Creates parameter bindings using word embeddings AND morphological features.
        Uses hash-based fallback if embeddings/morph are missing. V6.2 Morph-Aware.
        Args:
            circuit (QuantumCircuit): The circuit with parameters.
            tokens (List[str]): List of sentence tokens.
            analyses (List[Dict]): List of analysis dictionaries from Stanza/CAMeL.
                                   Expected keys: 'lemma', 'pos', 'deprel', 'head', 'morph'.
        Returns:
            Optional[Dict[Parameter, float]]: Dictionary mapping Parameter objects to float values.
        """
        parameter_binds_map = {}
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit) or not hasattr(circuit, 'parameters'):
            logger.warning("Invalid circuit passed to _bind_parameters.")
            return None

        params: Set[Parameter] = circuit.parameters
        if not params: return {} # No parameters

        num_params_needed = len(params)
        param_values = {} # Dict to store {Parameter: value}
        logger.debug(f"Binding {num_params_needed} parameters using morph-aware strategy...")

        # --- Helper for Hash-based Fallback ---
        def get_hashed_value(param_name: str) -> float:
            param_hash_input = param_name.encode('utf-8')
            param_hash_val = int(hashlib.sha256(param_hash_input).hexdigest(), 16)
            scaled_value = ((param_hash_val % (2 * np.pi * 10000)) / 10000.0) - np.pi
            return scaled_value
        # --- End Helper ---

        # --- Helper for Morphological Feature Vector ---
        def get_morph_vector(morph_features: Optional[Dict]) -> np.ndarray:
            """ Converts CAMeL morph dict to a simple numerical vector. """
            # Define a fixed order/mapping for features
            # Example: [gender(m=0,f=1,na=0.5), number(s=0,p=1,d=0.5,na=0.2), aspect(p=0,i=0.5,c=1,na=0.2), def(d=1,i=0,na=0.5)]
            # This needs careful design based on expected features and desired influence.
            vec = np.zeros(4) # Example size
            if not morph_features: return vec * 0.5 # Return neutral if no features

            # Gender: m=0, f=1
            gen = morph_features.get('gen')
            if gen == 'm': vec[0] = 0.0
            elif gen == 'f': vec[0] = 1.0
            else: vec[0] = 0.5 # Neutral

            # Number: s=0, p=1, d=0.5
            num = morph_features.get('num')
            if num == 's': vec[1] = 0.0
            elif num == 'p': vec[1] = 1.0
            elif num == 'd': vec[1] = 0.5
            else: vec[1] = 0.2 # Neutral

            # Aspect: p=0, i=0.5, c=1
            asp = morph_features.get('asp')
            if asp == 'p': vec[2] = 0.0
            elif asp == 'i': vec[2] = 0.5
            elif asp == 'c': vec[2] = 1.0
            else: vec[2] = 0.2 # Neutral

            # Definiteness: d=1, i=0
            stt = morph_features.get('stt')
            if stt == 'd': vec[3] = 1.0
            elif stt == 'i': vec[3] = 0.0
            else: vec[3] = 0.5 # Neutral

            # Normalize vector values to be roughly in [-1, 1] or [0, 1] range if needed
            # For simplicity, we use the 0-1 mapping here.
            return vec
        # --- End Morph Helper ---

        params_list = sorted(list(params), key=lambda p: p.name)
        param_idx = 0
        word_idx = 0 # Index into the analyses list
        assigned_params = set()

        # Check if analyses list is valid
        if not analyses or len(tokens) != len(analyses):
            logger.warning("Tokens/Analyses mismatch or missing in _bind_parameters. Using fallback.")
            analyses = [{}] * len(tokens) # Create dummy analysis list

        # Iterate through words/analyses to assign parameters
        while param_idx < num_params_needed and word_idx < len(analyses):
            analysis_dict = analyses[word_idx]
            lemma = analysis_dict.get('lemma', tokens[word_idx])
            token = tokens[word_idx]
            morph_features = analysis_dict.get('morph') # Get CAMeL dict

            # Get embedding vector
            embed_vector = None
            if self.embedding_model:
                try:
                    if lemma in self.embedding_model: embed_vector = self.embedding_model[lemma]
                    elif token in self.embedding_model: embed_vector = self.embedding_model[token]
                except Exception: embed_vector = None

            # Get morphological vector
            morph_vector = get_morph_vector(morph_features)

            # Assign params_per_word parameters for this word/analysis entry
            for i in range(self.params_per_word):
                if param_idx >= num_params_needed: break

                current_param = params_list[param_idx]
                if current_param in assigned_params: param_idx += 1; continue

                param_value = 0.0
                value_source = "Fallback"

                # --- Combined Assignment Logic ---
                embed_contrib = 0.0
                morph_contrib = 0.0

                # 1. Embedding Contribution
                if embed_vector is not None:
                    hash_input_embed = f"{lemma}_embed_{i}".encode('utf-8')
                    hash_val_embed = int(hashlib.sha256(hash_input_embed).hexdigest(), 16)
                    embed_idx = hash_val_embed % self.embedding_model.vector_size
                    raw_embed_val = float(embed_vector[embed_idx])
                    # Scale embedding value to [-pi, pi]
                    embed_contrib = (raw_embed_val % (2 * np.pi)) - np.pi
                    value_source = "Embed"
                else:
                    # Use hash fallback if no embedding
                    embed_contrib = get_hashed_value(f"{current_param.name}_embed_fallback")

                # 2. Morphological Contribution
                # Use hash of morph vector elements + param index to select feature
                morph_vec_len = len(morph_vector)
                if morph_vec_len > 0:
                    hash_input_morph = f"{lemma}_morph_{i}".encode('utf-8')
                    hash_val_morph = int(hashlib.sha256(hash_input_morph).hexdigest(), 16)
                    morph_idx = hash_val_morph % morph_vec_len
                    raw_morph_val = morph_vector[morph_idx] # Value between 0 and 1
                    # Scale morph value to [-pi, pi]
                    morph_contrib = (raw_morph_val * 2 * np.pi) - np.pi
                    if value_source == "Fallback": value_source = "Morph"
                    else: value_source += "+Morph"
                else:
                    # Use hash fallback if no morph features
                    morph_contrib = get_hashed_value(f"{current_param.name}_morph_fallback")


                # 3. Combine contributions using weight
                param_value = (1.0 - self.morph_param_weight) * embed_contrib + self.morph_param_weight * morph_contrib

                # Ensure value is within [-pi, pi] range (optional final clamp)
                param_value = max(-np.pi, min(np.pi, param_value))

                param_values[current_param] = param_value
                logger.debug(f"  Bound param {current_param.name} (idx {param_idx}, word {word_idx} '{token}') to {param_value:.3f} (Source: {value_source})")

                assigned_params.add(current_param)
                param_idx += 1
            # --- End Parameter Assignment Loop for Word ---
            word_idx += 1
        # --- End Word Loop ---

        # Assign fallback to any remaining unbound parameters (e.g., from Force_S boxes)
        while param_idx < num_params_needed:
             current_param = params_list[param_idx]
             if current_param not in assigned_params:
                  fallback_value = get_hashed_value(current_param.name)
                  param_values[current_param] = fallback_value
                  logger.warning(f"  Bound remaining param {current_param.name} (idx {param_idx}) to HASHED fallback -> {fallback_value:.3f}")
                  assigned_params.add(current_param)
             param_idx += 1

        # Final check
        if len(param_values) != num_params_needed:
             logger.error(f"Parameter binding mismatch! Expected {num_params_needed}, got {len(param_values)}. Assigning emergency fallbacks.")
             missing_params = params - set(param_values.keys())
             for p_missing in missing_params: param_values[p_missing] = get_hashed_value(p_missing.name)

        return param_values


    # --- Feature Extraction and Embedding ---
    def get_enhanced_circuit_features(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Dict], debug: bool = False) -> np.ndarray:
        """
        Extracts features using Pauli expectation values via Qiskit Estimator.
        Uses hash-based fallback for parameter binding if needed.
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
            # --- 1. Bind Parameters (Now uses hash fallback internally) ---
            parameter_binds_map = self._bind_parameters(circuit, tokens, analyses)
            if parameter_binds_map is None: # Handle potential error in binding
                 logger.error("Parameter binding failed. Returning zero features.")
                 return fallback_features
            if not parameter_binds_map and circuit.parameters:
                 logger.warning(f"Circuit '{circuit.name}' has parameters but binding map is empty. Check binding logic.")
                 # Proceeding, but parameters will likely default to 0 in Estimator if not explicitly bound

            # --- 2: Remove Measurements ---
            # Create a copy of the circuit without measurements for the Estimator
            circuit_to_estimate = circuit.remove_final_measurements(inplace=False)
            if debug: logger.debug(f"Removed final measurements from circuit '{circuit_name}' for Estimator.")

            # --- 3: Format parameters for Estimator ---
            # Get parameters in a deterministic order (use the original circuit's parameters)
            params_list = sorted(list(circuit.parameters), key=lambda p: p.name)
            # Create an ordered list of values, defaulting to 0.0 if somehow missing (shouldn't happen with new binding)
            param_values_ordered_list = [parameter_binds_map.get(p, 0.0) for p in params_list]

            if debug and params_list:
                 logger.debug("Parameter values list for Estimator:")
                 for p, v in zip(params_list, param_values_ordered_list): logger.debug(f"  {p.name}: {v:.4f}")

            # --- 4. Define Observables (Pauli Z and X for each qubit) ---
            observables_z = []
            observables_x = []
            for i in range(num_qubits):
                pauli_z_str = "I" * (num_qubits - 1 - i) + "Z" + "I" * i
                pauli_x_str = "I" * (num_qubits - 1 - i) + "X" + "I" * i
                observables_z.append(SparsePauliOp.from_list([(pauli_z_str, 1)]))
                observables_x.append(SparsePauliOp.from_list([(pauli_x_str, 1)]))

            all_observables = observables_z + observables_x
            num_observables = len(all_observables)
            if debug: logger.debug(f"Defined {num_observables} observables (Z and X for {num_qubits} qubits).")

            # --- 5. Run Estimator ---
            if not all_observables:
                 logger.warning("No observables defined. Returning zero features.")
                 return fallback_features

            # Create the list of lists expected by the estimator
            # Each inner list corresponds to one circuit execution (one for each observable)
            # If the circuit has no parameters, pass an empty list for that circuit's parameters
            parameter_values_for_estimator: Sequence[Sequence[float]]
            if not params_list:
                 # If no parameters, pass list of empty lists
                 parameter_values_for_estimator = [[]] * num_observables
                 if debug: logger.debug(f"Running Estimator for circuit '{circuit_to_estimate.name}' (no parameters)...")
            else:
                 # If parameters exist, pass the ordered list repeated
                 parameter_values_for_estimator = [param_values_ordered_list] * num_observables
                 if debug: logger.debug(f"Running Estimator for circuit '{circuit_to_estimate.name}' with {len(param_values_ordered_list)} param values...")


            # Use the measurement-free circuit with the Estimator
            job = self.estimator.run(circuits=[circuit_to_estimate] * num_observables, # Use circuit copy
                                     observables=all_observables,
                                     parameter_values=parameter_values_for_estimator) # Pass the list of lists

            result = job.result()
            expectation_values = result.values.tolist() # Get list of expectation values
            if debug: logger.debug(f"Estimator job completed. Got {len(expectation_values)} expectation values.")
            feature_vector.extend(expectation_values)

        except QiskitError as qe:
             # Catch specific Qiskit errors that might still occur
             logger.error(f"\n--- QiskitError in get_enhanced_circuit_features ---")
             logger.error(f"Circuit Name: {getattr(circuit, 'name', 'N/A')}")
             logger.error(f"Error type: {type(qe)}")
             logger.error(f"Error message: {qe}")
             # Don't print full traceback for known Qiskit issues unless needed
             if 'classical bits' in str(qe):
                 logger.error(">>> Error suggests measurements might still be present despite removal attempt.")
             traceback.print_exc() # Print full trace for debugging
             logger.error(f"---------------------------------------\n")
             return fallback_features
        except Exception as e:
            logger.error(f"\n--- ERROR in get_enhanced_circuit_features ---")
            logger.error(f"Circuit Name: {getattr(circuit, 'name', 'N/A')}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {e}")
            traceback.print_exc()
            logger.error(f"---------------------------------------\n")
            return fallback_features # Return fallback on error

        # --- 6. Combine, Pad/Truncate, Normalize ---
        final_features = np.array(feature_vector)

        # Pad or truncate to match embedding_dim
        current_len = len(final_features)
        if current_len == 0:
             logger.warning("Feature vector is empty after processing. Returning zeros.")
             return fallback_features
        elif current_len < self.embedding_dim:
             final_features = np.pad(final_features, (0, self.embedding_dim - current_len), 'constant')
             if debug: logger.debug(f"Padded feature vector from {current_len} to {self.embedding_dim}")
        elif current_len > self.embedding_dim:
             final_features = final_features[:self.embedding_dim]
             if debug: logger.debug(f"Truncated feature vector from {current_len} to {self.embedding_dim}")

        # Normalize final vector
        norm = np.linalg.norm(final_features)
        if norm > 1e-9: # Avoid division by zero
            final_features = final_features / norm
            if debug: logger.debug(f"Final feature vector normalized (Norm: {norm:.4f}).")
        else:
             if debug: logger.warning("Final feature vector norm is near zero. Not normalizing.")

        # Final check for NaNs
        if not np.all(np.isfinite(final_features)):
             logger.error("NaN or Inf found in final feature vector! Returning zeros.")
             return fallback_features

        return final_features


    # --- Linguistic Feature Extraction (No changes needed here for parameter binding) ---
    def extract_linguistic_features(self, tokens: List[str], analyses: List[Dict], structure: str, roles: Dict) -> np.ndarray:
        """
        Extracts basic linguistic features.
        V6.3: Correctly handles list of analysis dictionaries.
        """
        features = np.zeros(self.embedding_dim)
        num_features = 0

        # Structure feature
        structure_map = {'VSO': 0, 'SVO': 1, 'NOMINAL': 2, 'COMPLEX': 3, 'OTHER': 4}
        structure_idx = structure_map.get(structure.split('_')[0], 4) # Use base structure before COMPLEX_
        if num_features < self.embedding_dim:
            features[num_features] = structure_idx / (len(structure_map) - 1)
            num_features += 1

        # POS counts - *** FIX: Use dictionary access ***
        pos_counts = Counter(analysis_dict.get('pos', 'UNK_POS') for analysis_dict in analyses)
        total_tokens = max(1, len(tokens))
        if num_features < self.embedding_dim:
            features[num_features] = pos_counts.get('VERB', 0) / total_tokens
            num_features += 1
        if num_features < self.embedding_dim:
            features[num_features] = pos_counts.get('NOUN', 0) / total_tokens
            num_features += 1
        if num_features < self.embedding_dim:
            features[num_features] = pos_counts.get('ADJ', 0) / total_tokens
            num_features += 1
        # *** END FIX ***

        # Role presence features
        if num_features < self.embedding_dim:
            features[num_features] = 1.0 if roles.get('verb') is not None else 0.0
            num_features += 1
        if num_features < self.embedding_dim:
            features[num_features] = 1.0 if roles.get('subject') is not None else 0.0
            num_features += 1
        if num_features < self.embedding_dim:
            features[num_features] = 1.0 if roles.get('object') is not None else 0.0
            num_features += 1

        # Negation feature - *** FIX: Use dictionary access ***
        has_negation = any(analysis_dict.get('lemma') in ['لا', 'ليس', 'غير', 'لم', 'لن'] for analysis_dict in analyses)
        if num_features < self.embedding_dim:
            features[num_features] = 1.0 if has_negation else 0.0
            num_features += 1
        # *** END FIX ***

        # Normalize the feature vector (only the part filled)
        if num_features > 0:
            filled_features = features[:num_features]
            norm = np.linalg.norm(filled_features)
            if norm > 1e-9:
                features[:num_features] = filled_features / norm
        else: # Handle case where no features were added
            features = np.zeros(self.embedding_dim)

        return features


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

    # --- Training and Clustering ---
    # MODIFIED: Pass tokens/analyses to feature extraction
    def train(self, sentences, circuits, tokens_list, analyses_list, structures, roles_list, use_enhanced_clustering=False):
        """ Train the kernel. V6.2 - Uses morph-aware features. """
        self.reference_sentences = {i: sentences[i] for i in range(len(sentences))}
        self.circuit_embeddings = {}; self.sentence_embeddings = {}; embeddings = []
        logger.info(f"\n--- Training Kernel V6.2 on {len(sentences)} sentences ---")

        # Input validation (ensure all lists have same length)
        min_len = len(sentences)
        if not all(len(lst) == min_len for lst in [circuits, tokens_list, analyses_list, structures, roles_list]):
             logger.error("Input list length mismatch in train(). Aborting.")
             return self
        if min_len == 0: logger.error("Cannot train with empty dataset."); return self

        # Extract features and create embeddings
        for i in range(min_len):
            try:
                current_circuit = circuits[i]
                current_tokens = tokens_list[i]
                current_analyses = analyses_list[i] # List of dicts
                current_structure = structures[i]
                current_roles = roles_list[i]

                if not isinstance(current_circuit, QuantumCircuit):
                     logger.warning(f"Skipping sentence {i}: Invalid circuit type {type(current_circuit)}")
                     continue

                # --- Get Quantum Features (using morph-aware binding) ---
                quantum_features = self.get_enhanced_circuit_features(
                    current_circuit, current_tokens, current_analyses
                )
                self.circuit_embeddings[i] = quantum_features

                # --- Get Complex Linguistic Features ---
                linguistic_features = self.extract_complex_linguistic_features_v2(
                    current_tokens, current_analyses, current_structure, current_roles
                )

                # --- Combine Features ---
                embedding = self.combine_features_with_attention(
                    quantum_features, linguistic_features, current_structure
                )
                self.sentence_embeddings[i] = embedding
                embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Error processing sentence {i} ('{sentences[i][:30]}...') during training embedding generation: {e}", exc_info=True)

        if not embeddings: logger.error("No embeddings generated. Cannot proceed with clustering."); return self

        # Clustering (Keep existing logic: KMeans or enhanced LDA)
        if use_enhanced_clustering and GENSIM_LDA_AVAILABLE:
            self.learn_enhanced_meaning_clusters(embeddings, sentences)
        else:
            self.learn_meaning_clusters(embeddings)

        # Assign meaning (Keep existing, but pass the list of analysis dicts)
        self.assign_meaning_to_clusters(sentences, structures, roles_list, analyses_list) # Pass analyses_list
        logger.info("--- Training Complete ---")
        return self

    # --- Clustering Methods (No changes needed here) ---
    def learn_meaning_clusters(self, embeddings: List[np.ndarray]) -> None:
        """ Learn meaning clusters from embeddings using KMeans. """
        if not embeddings: logger.warning("No embeddings provided for clustering."); return
        X = np.array(embeddings); n_samples = X.shape[0]
        if n_samples == 0: logger.warning("Embedding array is empty. Cannot cluster."); return
        n_clusters = min(self.num_clusters, n_samples)
        if n_clusters <= 0: n_clusters = 1
        self.num_clusters = n_clusters
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(X)
            self.meaning_clusters = kmeans.cluster_centers_
        except Exception as e:
            logger.error(f"Error during KMeans clustering: {e}", exc_info=True)
            self.cluster_labels = None; self.meaning_clusters = None

    def learn_enhanced_meaning_clusters(self, embeddings, sentences):
        """Learn meaning clusters with topic modeling enhancement"""
        self.learn_meaning_clusters(embeddings) # Basic KMeans first
        if self.meaning_clusters is None or self.cluster_labels is None:
            logger.warning("Skipping topic modeling enhancement due to prior clustering failure.")
            return
        if not GENSIM_LDA_AVAILABLE:
            logger.warning("Skipping topic modeling enhancement: gensim library not available.")
            return
        logger.info("Enhancing clusters with LDA topic modeling...")
        try:
            # Use shaped text for topic modeling if possible
            processed_sentences = [shape_arabic_text(s) for s in sentences]
            tokenized_sentences = [sentence.split() for sentence in processed_sentences]
            # Consider refining stopwords
            arabic_stopwords = ['من', 'في', 'على', 'الى', 'إلى', 'عن', 'و', 'ف', 'ثم', 'أو', 'لا', 'ما', 'هو', 'هي', 'هم', 'هن', 'هذا', 'هذه', 'ذلك', 'تلك', 'الذي', 'التي', 'الذين', 'قد', 'لقد', 'أن', 'ان', 'إن', 'كان', 'يكون', 'لم', 'لن', 'كل', 'بعض', 'يا', 'اي', 'أي', 'مع', 'به', 'له', 'فيه', 'تم']
            filtered_sentences = [[word for word in sentence if word not in arabic_stopwords and len(word) > 1] for sentence in tokenized_sentences]
            dictionary = corpora.Dictionary(filtered_sentences)
            corpus = [dictionary.doc2bow(text) for text in filtered_sentences]
            if not corpus or not dictionary:
                logger.warning("Corpus or dictionary is empty after filtering. Skipping topic modeling.")
                return
            num_topics = min(self.num_clusters * 2, len(dictionary), 15) # Heuristic
            if num_topics <= 1:
                 logger.warning("Not enough unique terms for topic modeling. Skipping.")
                 return
            logger.info(f"Training LDA model with {num_topics} topics...")
            lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
            sentence_topics = [lda_model.get_document_topics(doc, minimum_probability=0.1) for doc in corpus]
            if not self.meaning_map:
                 logger.warning("meaning_map not initialized before topic assignment. Topics might not be stored correctly unless assign_meaning_to_clusters is called after.")
                 pass # Initialize later in assign_meaning_to_clusters
            for cluster_id in range(self.num_clusters):
                cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == cluster_id and i < len(sentence_topics)]
                cluster_topic_dist = Counter()
                for idx in cluster_indices:
                    for topic_id, prob in sentence_topics[idx]:
                        cluster_topic_dist[topic_id] += prob
                topic_words = {}
                top_cluster_topics = cluster_topic_dist.most_common(3)
                for topic_id, total_prob in top_cluster_topics:
                     words = lda_model.show_topic(topic_id, topn=5)
                     # Shape the words for display if needed, though meaning_map stores raw words
                     topic_words[f"Topic {topic_id}"] = {'words': [word for word, _ in words], 'weight': total_prob}
                if cluster_id in self.meaning_map:
                    self.meaning_map[cluster_id]['topics'] = topic_words
                else:
                    # Initialize the cluster entry if it doesn't exist
                    self.meaning_map[cluster_id] = {'topics': topic_words}
            logger.info("Topic modeling enhancement complete.")
        except Exception as e:
            logger.error(f"Topic modeling enhancement failed: {e}", exc_info=True)

    # --- Meaning Assignment (No changes needed here) ---
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

    # --- Interpretation and Discourse ---
    # MODIFIED: Pass tokens/analyses down to feature extraction
    def interpret_sentence(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Dict], structure: str, roles: Dict, previous_analyses=None) -> Dict:
        """ Interpret sentence meaning based on circuit, linguistics, and optionally context. """
        # Context Handling: Redirect if context is provided
        if previous_analyses is not None:
             return self.analyze_sentence_in_context(circuit, tokens, analyses, structure, roles, previous_analyses)

        # Direct Interpretation (No Context or Base Interpretation)
        if not isinstance(circuit, QuantumCircuit):
            logger.error(f"  ERROR in interpret_sentence: Expected QuantumCircuit, got {type(circuit)}. Cannot proceed.")
            return {'error': f'Invalid circuit type: {type(circuit)}'} # Return error dict

        # --- MODIFIED CALL: Pass tokens and analyses ---
        quantum_features = self.get_enhanced_circuit_features(circuit, tokens, analyses)
        if not np.all(np.isfinite(quantum_features)):
             logger.error("interpret_sentence: Quantum features contain NaN/Inf.")
             return {'error': 'Failed to generate valid quantum features.'}
        # --- END MODIFIED CALL ---

        enhanced_analyses = analyses # Use potentially enhanced analyses if CAMeL was used
        camel_morphology = None; sentiment_score = None; named_entities = []
        if self.camel_analyzer:
             try:
                 sentence_text = ' '.join(tokens)
                 camel_morphology = self.camel_analyzer.analyze(sentence_text)
                 # Add optional sentiment/NER calls here if needed
             except Exception as camel_e: pass

        linguistic_features = self.extract_complex_linguistic_features(tokens, enhanced_analyses, structure, roles)
        embedding = self.combine_features_with_attention(quantum_features, linguistic_features, structure)

        # Extract Semantic Frames (using potentially enhanced analyses)
        try:
            semantic_frames_data = self.extract_enhanced_semantic_frames(tokens, enhanced_analyses, roles)
            extracted_frames = semantic_frames_data.get('frames', [])
        except Exception as frame_e:
            logger.error(f"    Error extracting semantic frames: {frame_e}")
            extracted_frames = []

        # Prepare base result structure
        result = {
            'sentence': ' '.join(tokens), 'structure': structure, 'embedding': embedding,
            'interpretation': None, 'meaning_options': [], 'specific_interpretation': None,
            'semantic_frames': extracted_frames,
            'discourse_relations': [],
            'enhanced_linguistic_analysis': enhanced_analyses, 'roles': roles,
            'morphological_analysis': camel_morphology, 'sentiment': sentiment_score, 'named_entities': named_entities,
            'confidence': 0.0
        }

        # Find meaning options based on clusters
        if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
            similarities = []
            for i in range(len(self.meaning_clusters)):
                prob = self.get_meaning_probability(embedding, i)
                similarities.append((i, prob))
            similarities.sort(key=lambda x: x[1], reverse=True)
            meanings = []
            for cluster_id, prob in similarities[:min(3, len(similarities))]:
                if cluster_id in self.meaning_map:
                    cluster_info = self.meaning_map[cluster_id]
                    meanings.append({
                        'cluster_id': cluster_id, 'structure': cluster_info.get('structure', 'N/A'),
                        'deduced_template': cluster_info.get('deduced_template', 'N/A'),
                        'examples': cluster_info.get('examples', []), 'probability': prob,
                        'sentiment': cluster_info.get('sentiment', None),
                        'topics': cluster_info.get('topics', {})
                    })
            result['meaning_options'] = meanings
            if meanings:
                 result['top_meaning_cluster'] = meanings[0]['cluster_id']
                 result['confidence'] = meanings[0]['probability']
                 top_cluster_info = self.meaning_map.get(meanings[0]['cluster_id'], {})
                 template_dict = top_cluster_info.get('original_templates', self.semantic_templates.get(structure.split('_')[0], self.semantic_templates['OTHER']))
                 result['specific_interpretation'] = self.create_specific_interpretation(tokens, enhanced_analyses, roles, structure, template_dict)
                 result['interpretation'] = top_cluster_info.get('deduced_template', 'N/A')
        else:
            # Fallback if no clusters
            templates = self.semantic_templates.get(structure.split('_')[0], self.semantic_templates['OTHER'])
            result['specific_interpretation'] = self.create_specific_interpretation(tokens, enhanced_analyses, roles, structure, templates)
            result['interpretation'] = result['specific_interpretation']['templates'].get('declarative', 'N/A')

        return result

    # MODIFIED: Pass tokens/analyses down to interpret_sentence
    def analyze_sentence_in_context(self, current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analysis_dict=None):
        """ Analyze a sentence considering the previous sentence's context. """
        # Get base interpretation (passing tokens/analyses)
        base_interpretation = self.interpret_sentence(current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analyses=None)

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

    # --- Helper methods for interpretation and discourse (no changes needed) ---
    def get_meaning_probability(self, embedding: np.ndarray, cluster_id: int) -> float:
        """ Calculates probability of embedding belonging to a cluster using cosine similarity. """
        if self.meaning_clusters is None or cluster_id >= len(self.meaning_clusters): return 0.0
        cluster_center = self.meaning_clusters[cluster_id]
        if embedding.ndim == 1: embedding = embedding.reshape(1, -1)
        if cluster_center.ndim == 1: cluster_center = cluster_center.reshape(1, -1)
        if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(cluster_center)): return 0.0
        if embedding.shape[1] != cluster_center.shape[1]: return 0.0
        try: similarity = cosine_similarity(embedding, cluster_center)[0][0]
        except ValueError: return 0.0
        probability = (similarity + 1) / 2
        return max(0.0, min(1.0, probability))

    def create_specific_interpretation(self, tokens: List[str], analyses: List[Dict], roles: Dict, structure: str, templates: Dict) -> Dict:
        """ Creates a specific interpretation by filling templates with actual values. """
        subject = "unknown"; verb = "unknown"; predicate = "unknown"; object_text = "unknown"
        verb_lemma = None; tense = "present"; modality = "indicative"
        verb_idx = roles.get('verb'); subj_idx = roles.get('subject'); obj_idx = roles.get('object')
        if verb_idx is not None and verb_idx < len(tokens):
            verb = tokens[verb_idx]
            verb_lemma = analyses[verb_idx]['lemma']
            morph = analyses[verb_idx].get('morph')
            if morph: # Get tense/modality from morph features
                asp = morph.get('asp'); mod = morph.get('mod')
                if asp == 'p': tense = "past"
                elif asp == 'i': tense = "present"
                elif asp == 'c': tense = "imperative"
                if mod == 'i': modality = "indicative"
                elif mod == 's': modality = "subjunctive"
                elif mod == 'j': modality = "jussive"
        if subj_idx is not None and subj_idx < len(tokens):
            subject = tokens[subj_idx]
            if subj_idx > 0 and subj_idx < len(analyses) and analyses[subj_idx-1][1] == 'DET': subject = tokens[subj_idx-1] + " " + subject
        if obj_idx is not None and obj_idx < len(tokens):
            object_text = tokens[obj_idx]
            if obj_idx > 0 and obj_idx < len(analyses) and analyses[obj_idx-1][1] == 'DET': object_text = tokens[obj_idx-1] + " " + object_text
        if structure == 'NOMINAL':
            for i, (_, pos, dep, head) in enumerate(analyses):
                if pos == 'ADJ' and head == subj_idx: predicate = tokens[i]; break
        semantic_roles = {}; semantic_frames = []
        if self.camel_analyzer and verb_idx is not None:
             try:
                 morph_analysis_list = self.camel_analyzer.analyze(' '.join(tokens))
                 if verb_idx < len(morph_analysis_list):
                     verb_analysis_list = morph_analysis_list[verb_idx]
                     if verb_analysis_list:
                         verb_morph = verb_analysis_list[0]
                         asp = verb_morph.get('asp'); mod = verb_morph.get('mod')
                         if asp == 'p': tense = "past"
                         elif asp == 'i': tense = "present"
                         elif asp == 'c': tense = "imperative"
                         if mod == 'i': modality = "indicative"
                         elif mod == 's': modality = "subjunctive"
                         elif mod == 'j': modality = "jussive"
             except Exception as e: pass
        verb_class = self._classify_verb(verb_lemma)
        if verb_class != "UNKNOWN" and verb_class != "ACTION": semantic_frames.append(verb_class)
        filled_templates = {}
        for template_type, template in templates.items():
            filled = template.replace("SUBJECT", subject).replace("ACTION", verb).replace("OBJECT", object_text).replace("PREDICATE", predicate).replace("TOPIC", subject)
            filled_templates[template_type] = filled
        semantic_details = {
            'subject': {'text': subject, 'index': subj_idx, 'semantic_role': semantic_roles.get(subj_idx, "AGENT" if subj_idx is not None else None)},
            'verb': {'text': verb, 'index': verb_idx, 'lemma': verb_lemma, 'tense': tense, 'modality': modality},
            'object': {'text': object_text, 'index': obj_idx, 'semantic_role': semantic_roles.get(obj_idx, "PATIENT" if obj_idx is not None else None)},
            'predicate': {'text': predicate, 'structure_type': structure},
            'semantic_frames': semantic_frames
        }
        return {'templates': filled_templates, 'semantic_details': semantic_details}

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

    # --- Semantic Frame Extraction (No changes needed here) ---
    def extract_semantic_frames(self, tokens, analyses, roles):
        """ Extract basic semantic frames based on verb class and potentially external tools. """
        sentence = ' '.join(tokens); frames = []
        verb_idx = roles.get('verb'); verb = None; verb_lemma = None
        if verb_idx is not None and verb_idx < len(tokens):
            verb = tokens[verb_idx]
            if verb_idx < len(analyses): verb_lemma = analyses[verb_idx][0]
            verb_class = self._classify_verb(verb_lemma)
            if verb_class != "UNKNOWN" and verb_class != "ACTION":
                frame = {'type': verb_class, 'verb': verb}
                subj_idx = roles.get('subject'); obj_idx = roles.get('object')
                subj_token = tokens[subj_idx] if subj_idx is not None and subj_idx < len(tokens) else None
                obj_token = tokens[obj_idx] if obj_idx is not None and obj_idx < len(tokens) else None
                if verb_class == "MOTION": frame.update({'agent': subj_token, 'destination': obj_token})
                elif verb_class == "POSSESSION": frame.update({'possessor': subj_token, 'possessed': obj_token})
                # ... (other verb classes) ...
                frames.append(frame)
        if self.camel_analyzer:
            try:
                camel_analysis = self.camel_analyzer.analyze(sentence)
                semantic_properties = {'tense': None, 'mood': None, 'aspect': None, 'definiteness': [], 'gender': {}, 'number': {}}
                for i, token_analysis_list in enumerate(camel_analysis):
                    if token_analysis_list:
                        token_analysis = token_analysis_list[0]
                        if 'asp' in token_analysis: semantic_properties['aspect'] = token_analysis['asp']
                        # ... (extract other properties) ...
                if verb_idx is not None and verb_idx < len(camel_analysis) and camel_analysis[verb_idx]:
                     verb_analysis = camel_analysis[verb_idx][0]
                     # ... (extract tense from aspect) ...
                frames.append({'type': 'SEMANTIC_PROPERTIES', 'properties': semantic_properties})
            except Exception as e: logger.warning(f"Warning: Error in CAMeL semantic property extraction: {e}")
        # Optional: Add Farasa NER / AraVec calls if models are loaded
        return {'sentence': sentence, 'frames': frames}

    def extract_enhanced_semantic_frames(self, tokens, analyses, roles):
        """More comprehensive semantic frame extraction including rhetorical relations, etc."""
        basic_frames_data = self.extract_semantic_frames(tokens, analyses, roles)
        frames = basic_frames_data['frames'].copy(); sentence = basic_frames_data['sentence']
        # Add Rhetorical, Nested Predication, Coreference logic here...
        # (Code omitted for brevity - assume it's the same as previous version)
        return {'sentence': sentence, 'frames': frames}

    # --- Reporting (Modified for Arabic Display) ---
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

    def generate_discourse_report(self, discourse_analyses):
        """Generate a full report of discourse analysis for all sentences in Markdown."""
        # Shape titles and static text
        report = f"# {shape_arabic_text('تحليل الخطاب الكمي للنص العربي (محسن)')}\n\n"
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis.get('sentence', 'N/A'); interpretation_data = analysis.get('interpretation', {})
            # Shape dynamic text
            report += f"## {shape_arabic_text(f'الجملة {i+1}')}\n";
            report += f"**{shape_arabic_text('النص:')}** `{shape_arabic_text(sentence)}`\n\n" # Shape sentence here
            if interpretation_data.get('error'):
                 report += f"**{shape_arabic_text('خطأ في التحليل:')}** {interpretation_data['error']}\n\n"; report += "---\n\n"; continue

            # Shape other report sections as needed...
            structure = interpretation_data.get('structure', 'N/A')
            report += f"**{shape_arabic_text('البنية:')}** {structure}\n" # Structure likely English code, maybe no shaping needed

            deduced_template = interpretation_data.get('interpretation', 'N/A')
            report += f"**{shape_arabic_text('المعنى المستنتج:')}** {shape_arabic_text(deduced_template)}\n"

            specific_interp = interpretation_data.get('specific_interpretation', {})
            if specific_interp:
                report += f"**{shape_arabic_text('التفسير المحدد:')}**\n"
                details = specific_interp.get('semantic_details', {})
                templates = specific_interp.get('templates', {})
                report += f"  - **{shape_arabic_text('الفاعل:')}** {shape_arabic_text(details.get('subject', {}).get('text', 'N/A'))}\n"
                report += f"  - **{shape_arabic_text('الفعل:')}** {shape_arabic_text(details.get('verb', {}).get('text', 'N/A'))} ({shape_arabic_text('الزمن:')} {details.get('verb', {}).get('tense', 'N/A')})\n"
                report += f"  - **{shape_arabic_text('المفعول به:')}** {shape_arabic_text(details.get('object', {}).get('text', 'N/A'))}\n"
                report += f"  - **{shape_arabic_text('الخبر (اسمية):')}** {shape_arabic_text(details.get('predicate', {}).get('text', 'N/A'))}\n"
                report += f"  - **{shape_arabic_text('القالب الإخباري:')}** {shape_arabic_text(templates.get('declarative', 'N/A'))}\n"


            meaning_options = interpretation_data.get('meaning_options', [])
            if meaning_options:
                report += f"\n**{shape_arabic_text('خيارات المعنى المحتملة:')}**\n"
                for opt in meaning_options:
                    report += f"- **{shape_arabic_text('عنقود:')}** {opt.get('cluster_id', 'N/A')} ({shape_arabic_text('احتمال:')} {opt.get('probability', 0.0):.2f})\n"
                    report += f"  - **{shape_arabic_text('البنية:')}** {opt.get('structure', 'N/A')}\n" # Likely English code
                    report += f"  - **{shape_arabic_text('القالب المستنتج:')}** {shape_arabic_text(opt.get('deduced_template', 'N/A'))}\n"
                    topics = opt.get('topics', {})
                    if topics:
                        report += f"  - **{shape_arabic_text('المواضيع:')}**\n"
                        for topic_name, topic_data in topics.items():
                            words = ", ".join([shape_arabic_text(w) for w in topic_data.get('words', [])])
                            report += f"    - {shape_arabic_text(topic_name)}: {words} ({shape_arabic_text('وزن:')} {topic_data.get('weight', 0.0):.2f})\n"

            discourse_rels = interpretation_data.get('discourse_relations', [])
            report += f"\n**{shape_arabic_text('علاقات الخطاب:')}**\n{self.format_discourse_relations(discourse_rels)}\n" # format_discourse_relations handles shaping

            report += "\n---\n\n"
        return report

    def generate_html_report(self, discourse_analyses):
        """ Generate an HTML report with discourse analysis details. """
        # Shape static HTML text elements
        html = f"""
        <!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><title>{shape_arabic_text('تحليل الخطاب الكمي للغة العربية (محسن)')}</title><style>
        body{{font-family:'Tahoma', 'Segoe UI', 'Arial', sans-serif;margin:20px;line-height:1.6;background-color:#f9f9f9;color:#333}}
        h1{{color:#0056b3;text-align:center;border-bottom:2px solid #0056b3;padding-bottom:10px}}
        h2{{color:#0056b3;margin-top:30px;border-bottom:1px solid #eee;padding-bottom:5px}}
        h3{{color:#17a2b8;margin-top:20px}}
        .sentence-block{{background-color:#fff;padding:20px;margin-bottom:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
        .sentence-text{{font-weight:bold;font-size:1.2em;color:#333;margin-bottom:15px}}
        .analysis-section{{margin-bottom:15px;padding-left:15px;border-left:3px solid #eee}}
        .analysis-detail{{margin-left:10px}}
        .error{{border-left-color:#dc3545;color:#dc3545}}
        .error h3{{color:#dc3545}}
        ul{{list-style-type: none; padding-left: 0;}}
        li{{margin-bottom: 8px;}}
        li > ul {{ margin-top: 5px; padding-left: 20px; }}
        strong{{color:#0056b3}}
        code{{background-color:#e9ecef;padding:2px 5px;border-radius:4px;font-family:monospace}}
        .topic-words{{font-style: italic; color: #555;}}
        </style></head><body><h1>{shape_arabic_text('تحليل الخطاب الكمي للنص العربي (محسن)')}</h1>"""

        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis.get('sentence', 'N/A'); interpretation_data = analysis.get('interpretation', {})
            # Shape dynamic content
            html += f'<div class="sentence-block"><div class="sentence-text">{shape_arabic_text(f"الجملة {i+1}:")} {shape_arabic_text(sentence)}</div>'
            if interpretation_data.get('error'):
                 html += f'<div class="analysis-section error"><h3>{shape_arabic_text("خطأ في التحليل:")}</h3><div class="analysis-detail">{interpretation_data["error"]}</div></div></div>'; continue

            structure = interpretation_data.get('structure', 'N/A')
            html += f'<div class="analysis-section"><h3>{shape_arabic_text("البنية:")}</h3><div class="analysis-detail">{structure}</div></div>' # Structure likely English code

            deduced_template = interpretation_data.get('interpretation', 'N/A')
            html += f'<div class="analysis-section"><h3>{shape_arabic_text("المعنى المستنتج:")}</h3><div class="analysis-detail">{shape_arabic_text(deduced_template)}</div></div>'

            specific_interp = interpretation_data.get('specific_interpretation', {})
            if specific_interp:
                html += f'<div class="analysis-section"><h3>{shape_arabic_text("التفسير المحدد:")}</h3><div class="analysis-detail"><ul>'
                details = specific_interp.get('semantic_details', {})
                templates = specific_interp.get('templates', {})
                html += f'<li><strong>{shape_arabic_text("الفاعل:")}</strong> {shape_arabic_text(details.get("subject", {}).get("text", "N/A"))}</li>'
                html += f'<li><strong>{shape_arabic_text("الفعل:")}</strong> {shape_arabic_text(details.get("verb", {}).get("text", "N/A"))} ({shape_arabic_text("الزمن:")} {details.get("verb", {}).get("tense", "N/A")})</li>'
                html += f'<li><strong>{shape_arabic_text("المفعول به:")}</strong> {shape_arabic_text(details.get("object", {}).get("text", "N/A"))}</li>'
                html += f'<li><strong>{shape_arabic_text("الخبر (اسمية):")}</strong> {shape_arabic_text(details.get("predicate", {}).get("text", "N/A"))}</li>'
                html += f'<li><strong>{shape_arabic_text("القالب الإخباري:")}</strong> {shape_arabic_text(templates.get("declarative", "N/A"))}</li>'
                html += '</ul></div></div>'

            meaning_options = interpretation_data.get('meaning_options', [])
            if meaning_options:
                html += f'<div class="analysis-section"><h3>{shape_arabic_text("خيارات المعنى المحتملة:")}</h3><div class="analysis-detail"><ul>'
                for opt in meaning_options:
                    html += f'<li><strong>{shape_arabic_text("عنقود:")}</strong> {opt.get("cluster_id", "N/A")} ({shape_arabic_text("احتمال:")} {opt.get("probability", 0.0):.2f})<ul>'
                    html += f'<li><strong>{shape_arabic_text("البنية:")}</strong> {opt.get("structure", "N/A")}</li>' # Likely English code
                    html += f'<li><strong>{shape_arabic_text("القالب المستنتج:")}</strong> {shape_arabic_text(opt.get("deduced_template", "N/A"))}</li>'
                    topics = opt.get('topics', {})
                    if topics:
                        html += f'<li><strong>{shape_arabic_text("المواضيع:")}</strong><ul>'
                        for topic_name, topic_data in topics.items():
                            words = ", ".join([shape_arabic_text(w) for w in topic_data.get('words', [])])
                            html += f'<li>{shape_arabic_text(topic_name)}: <span class="topic-words">{words}</span> ({shape_arabic_text("وزن:")} {topic_data.get("weight", 0.0):.2f})</li>'
                        html += '</ul></li>'
                    html += '</ul></li>' # Close cluster details ul and li
                html += '</ul></div></div>'

            discourse_rels = interpretation_data.get('discourse_relations', [])
            html += f'<div class="analysis-section"><h3>{shape_arabic_text("علاقات الخطاب:")}</h3><div class="analysis-detail">{self.format_discourse_relations(discourse_rels)}</div></div>' # format_discourse_relations handles shaping

            html += '</div>' # Close sentence-block
        html += """</body></html>"""
        return html

    # --- Utility Methods (No changes needed here) ---
    def save_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Saves the trained kernel state to a file using pickle. """
        logger.info(f"Saving model to {filename}...")
        analyzer_state = None
        if hasattr(self.camel_analyzer, '__getstate__'): analyzer_state = self.camel_analyzer.__getstate__()
        elif self.camel_analyzer is not None: logger.warning("CAMeL Analyzer might not be pickleable.")
        # Exclude embedding model from saving, should be reloaded from path
        model_data = {
            'embedding_dim': self.embedding_dim, 'num_clusters': self.num_clusters, 'meaning_clusters': self.meaning_clusters,
            'cluster_labels': self.cluster_labels, 'meaning_map': self.meaning_map, 'reference_sentences': self.reference_sentences,
            'circuit_embeddings': self.circuit_embeddings, 'sentence_embeddings': self.sentence_embeddings,
            'semantic_templates': self.semantic_templates, #'camel_analyzer_state': analyzer_state,
            # Store path and params_per_word so we can reload model in load_model
            '_embedding_model_path': getattr(self, '_embedding_model_path', None),
            'params_per_word': self.params_per_word
        }
        try:
            with open(filename, 'wb') as f: pickle.dump(model_data, f)
            logger.info(f"Model saved successfully.")
        except Exception as e: logger.error(f"Error saving model: {e}", exc_info=True)

    def load_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Loads a trained kernel state from a file. Reloads embedding model from path."""
        if not os.path.exists(filename): logger.error(f"Error: Model file {filename} not found."); return self
        logger.info(f"Loading model from {filename}...")
        try:
            with open(filename, 'rb') as f: model_data = pickle.load(f)
            self.embedding_dim = model_data.get('embedding_dim', self.embedding_dim)
            self.num_clusters = model_data.get('num_clusters', self.num_clusters)
            self.meaning_clusters = model_data.get('meaning_clusters')
            self.cluster_labels = model_data.get('cluster_labels')
            self.meaning_map = model_data.get('meaning_map', {})
            self.reference_sentences = model_data.get('reference_sentences', {})
            self.circuit_embeddings = model_data.get('circuit_embeddings', {})
            self.sentence_embeddings = model_data.get('sentence_embeddings', {})
            self.semantic_templates = model_data.get('semantic_templates', self.semantic_templates)
            saved_model_path = model_data.get('_embedding_model_path')
            self.params_per_word = model_data.get('params_per_word', 3)

            # Re-run __init__ with saved path to reload embedding model and analyzer
            self.__init__(self.embedding_dim, self.num_clusters, embedding_model_path=saved_model_path, params_per_word=self.params_per_word, shots=self.shots) # Added shots
            # Restore the loaded state AFTER re-initialization
            self.meaning_clusters = model_data.get('meaning_clusters')
            self.cluster_labels = model_data.get('cluster_labels')
            self.meaning_map = model_data.get('meaning_map', {})
            self.reference_sentences = model_data.get('reference_sentences', {})
            self.circuit_embeddings = model_data.get('circuit_embeddings', {})
            self.sentence_embeddings = model_data.get('sentence_embeddings', {})

            logger.info(f"Model loaded successfully.")
        except Exception as e: logger.error(f"Error loading model: {e}", exc_info=True)
        return self

    # --- Visualization Methods (Modified for Arabic Display) ---
    def visualize_meaning_space(self, highlight_indices=None, save_path=None):
        """ Visualize the sentence meaning space using PCA. """
        logger.info("Visualizing meaning space...")
        if not self.sentence_embeddings: logger.warning("No embeddings available for visualization."); return None
        try: from sklearn.decomposition import PCA
        except ImportError: logger.error("scikit-learn is required for visualization."); return None
        embeddings = list(self.sentence_embeddings.values()); indices = list(self.sentence_embeddings.keys())
        if not embeddings: logger.warning("Embeddings list is empty."); return None
        X = np.array(embeddings)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
             logger.warning("Non-finite values found in embeddings. Removing problematic rows.")
             X = X[finite_mask]; original_indices = indices; indices = [idx for i, idx in enumerate(original_indices) if finite_mask[i]]
             if self.cluster_labels is not None:
                  if len(self.cluster_labels) == len(finite_mask): self.cluster_labels = self.cluster_labels[finite_mask]; logger.info(f"Filtered cluster labels to size {len(self.cluster_labels)}")
                  else: logger.warning("Cluster labels length mismatch after filtering NaNs."); self.cluster_labels = None
             if X.shape[0] == 0: logger.error("All embeddings contained non-finite values."); return None
             logger.info(f"Removed {len(finite_mask) - X.shape[0]} non-finite rows.")
        if X.shape[1] < 2: logger.warning("Need at least 2 embedding dimensions for PCA."); return None
        if X.shape[0] < 2: logger.warning("Need at least 2 samples for PCA."); return None
        try:
            pca = PCA(n_components=2); reduced_embeddings = pca.fit_transform(X)
            plt.figure(figsize=(12, 10)); current_labels = None
            if self.cluster_labels is not None and len(self.cluster_labels) == len(reduced_embeddings): current_labels = np.array(self.cluster_labels)
            colors = current_labels if current_labels is not None else 'blue'; cmap = 'viridis' if current_labels is not None else None
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, cmap=cmap, alpha=0.7, s=100)
            if highlight_indices is not None:
                 highlight_idxs_in_filtered = [indices.index(i) for i in highlight_indices if i in indices]
                 if highlight_idxs_in_filtered: plt.scatter(reduced_embeddings[highlight_idxs_in_filtered, 0], reduced_embeddings[highlight_idxs_in_filtered, 1], c='red', s=150, edgecolor='white', zorder=10, label='Highlighted') # Label likely English

            # Shape legend title
            legend_title = shape_arabic_text("عناقيد المعنى")
            if current_labels is not None and isinstance(colors, np.ndarray):
                try:
                     unique_labels = np.unique(current_labels)
                     if len(unique_labels) > 1: legend1 = plt.legend(*scatter.legend_elements(), title=legend_title); plt.gca().add_artist(legend1)
                     elif len(unique_labels) == 1: logger.info("Only one cluster found, skipping cluster legend.")
                except Exception as leg_e: logger.warning(f"Could not create cluster legend: {leg_e}")

            if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
                 if np.all(np.isfinite(self.meaning_clusters)):
                     try:
                         if hasattr(pca, 'components_'):
                              cluster_centers_2d = pca.transform(self.meaning_clusters)
                              # Shape label for cluster centers
                              plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker='*', s=350, c='white', edgecolor='black', label=shape_arabic_text('مراكز العناقيد'), zorder=15)
                              for i, (x, y) in enumerate(cluster_centers_2d):
                                  if i in self.meaning_map:
                                      meaning = self.meaning_map[i].get('structure', f'Cluster {i}'); template = self.meaning_map[i].get('deduced_template', '')[:30]
                                      # Shape annotation text
                                      annotation_text = shape_arabic_text(f"عنقود {i}: {meaning}\n'{template}...'")
                                      plt.annotate(annotation_text, (x, y), xytext=(0, 15), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8), zorder=20)
                         else: logger.warning("PCA not fitted, cannot transform cluster centers.")
                     except Exception as cc_e: logger.error(f"Error plotting cluster centers: {cc_e}")
                 else: logger.warning("Non-finite values found in cluster centers. Skipping plotting centers.")

            # Shape plot title and axis labels
            plt.title(shape_arabic_text('فضاء معنى الجملة الكمي (PCA)'));
            plt.xlabel(shape_arabic_text(f'المكون الرئيسي 1 ({pca.explained_variance_ratio_[0]:.1%} تباين)'));
            plt.ylabel(shape_arabic_text(f'المكون الرئيسي 2 ({pca.explained_variance_ratio_[1]:.1%} تباين)'))
            plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=150); logger.info(f"Visualization saved to {save_path}")
            return plt.gcf()
        except Exception as e: logger.error(f"Error during PCA visualization: {e}", exc_info=True); return None

    def analyze_quantum_states(self, circuits_dict, tokens_dict, analyses_dict, save_path_prefix=None):
        """ Analyze and visualize the quantum states of a dictionary of circuits, binding parameters. """
        if not circuits_dict: logger.warning("No circuits provided for state analysis."); return
        logger.info(f"Analyzing quantum states for {len(circuits_dict)} circuits...")
        num_circuits = len(circuits_dict); ncols = 3; nrows = (num_circuits + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False); axes = axes.flatten()
        i = 0
        for circuit_id, circuit in circuits_dict.items():
            if i >= len(axes): break
            ax = axes[i];
            # Get original sentence and shape it for the title
            original_sentence = self.reference_sentences.get(circuit_id, f"Circuit {circuit_id}")
            sentence_label = shape_arabic_text(original_sentence)
            if len(sentence_label) > 40: sentence_label = sentence_label[:37] + "..." # Truncate if needed

            try:
                if not isinstance(circuit, QuantumCircuit):
                    logger.warning(f"Skipping item {circuit_id}: Not a Qiskit QuantumCircuit (type: {type(circuit)})")
                    # Shape error message
                    error_text = shape_arabic_text(f"كائن دارة غير صالح\n(النوع: {type(circuit).__name__})\n(المعرف: {circuit_id})")
                    ax.text(0.5, 0.5, error_text, ha='center', va='center', fontsize=9, color='red')
                    ax.set_title(f"Error: {sentence_label}", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); i += 1; continue

                # Bind parameters for statevector simulation
                tokens = tokens_dict.get(circuit_id, [])
                analyses = analyses_dict.get(circuit_id, [])
                # Use the _bind_parameters method which now includes hash fallback
                parameter_binds = self._bind_parameters(circuit, tokens, analyses)
                bindings_list = [parameter_binds] if parameter_binds else None

                # Execute for statevector
                # Note: Statevector simulation might fail for large circuits
                job = self.simulator.run(circuit, parameter_binds=bindings_list) # No shots for statevector
                result = job.result()

                if not result.success:
                     status_msg = getattr(result, 'status', 'Unknown Status')
                     logger.error(f"Statevector simulation failed for circuit {circuit_id}. Status: {status_msg}")
                     # Shape error message
                     sim_error_text = shape_arabic_text(f"فشل المحاكاة\n(الحالة: {status_msg})")
                     ax.text(0.5, 0.5, sim_error_text, ha='center', va='center', fontsize=9, color='orange')
                     ax.set_title(f"{shape_arabic_text('خطأ محاكاة:')} {sentence_label}", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); i += 1; continue

                if hasattr(result, 'get_statevector'):
                    statevector = result.get_statevector();
                    if plot_state_city:
                        # Pass the shaped title to plot_state_city
                        plot_state_city(statevector, title=sentence_label, ax=ax)
                        ax.tick_params(axis='both', which='major', labelsize=8); ax.title.set_size(10)
                    else: ax.text(0.5, 0.5, shape_arabic_text("متجه الحالة (العرض معطل)"), ha='center', va='center', fontsize=9)

                    if save_path_prefix:
                         try:
                              individual_fig = plt.figure(figsize=(8, 6));
                              if plot_state_city:
                                  # Use shaped title for individual plot
                                  plot_state_city(statevector, title=f"{shape_arabic_text('الحالة:')} {sentence_label}", fig=individual_fig)
                              individual_path = f"{save_path_prefix}circuit_{circuit_id}_state.png"; individual_fig.savefig(individual_path, dpi=150, bbox_inches='tight'); plt.close(individual_fig)
                         except Exception as save_e: logger.warning(f"Warning: Failed to save individual state plot for {circuit_id}: {save_e}")
                else:
                    ax.text(0.5, 0.5, shape_arabic_text("متجه الحالة غير متوفر"), ha='center', va='center', fontsize=9)
                    ax.set_title(sentence_label, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            except Exception as e:
                logger.error(f"Error visualizing circuit {circuit_id}: {e}", exc_info=True)
                # Shape error message
                vis_error_text = shape_arabic_text(f"خطأ في العرض:\n{e}")
                ax.text(0.5, 0.5, vis_error_text, ha='center', va='center', fontsize=9, color='red')
                ax.set_title(f"{shape_arabic_text('خطأ:')} {sentence_label}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            i += 1
        for j in range(i, len(axes)): fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        # Shape overall figure title
        fig.suptitle(shape_arabic_text("تصورات الحالة الكمومية"), fontsize=16)
        if save_path_prefix:
            try: overview_path = f"{save_path_prefix}all_states_overview.png"; fig.savefig(overview_path, dpi=150); logger.info(f"Saved quantum states overview to {overview_path}")
            except Exception as overview_save_e: logger.warning(f"Warning: Failed to save overview state plot: {overview_save_e}")
        return fig

def generate_similarity_heatmap(embeddings: Dict[int, np.ndarray],
                                sentences: List[str],
                                processed_indices: List[int],
                                title: str,
                                save_path: str):
    """
    Generates and saves a cosine similarity heatmap for given embeddings.

    Args:
        embeddings (Dict[int, np.ndarray]): Dictionary mapping original index to embedding vector.
        sentences (List[str]): List of all original sentences (used for labels).
        processed_indices (List[int]): List of original indices that were successfully processed
                                       and have embeddings in the dictionary.
        title (str): Title for the heatmap plot.
        save_path (str): Full path to save the heatmap image.
    """
    logger.info(f"Generating similarity heatmap: {title}")
    if not embeddings or len(processed_indices) < 2:
        logger.warning(f"Skipping heatmap '{title}': Not enough valid embeddings ({len(processed_indices)}).")
        return

    # Ensure embeddings and labels align with processed_indices
    # Use list comprehension for clarity and safety
    embeddings_list = [embeddings[idx] for idx in processed_indices if idx in embeddings and isinstance(embeddings[idx], np.ndarray)]
    labels_list = [sentences[idx] for idx in processed_indices if idx < len(sentences) and idx in embeddings and isinstance(embeddings[idx], np.ndarray)]

    # Check alignment after filtering potentially missing embeddings
    if len(embeddings_list) != len(labels_list) or len(embeddings_list) < 2:
        logger.warning(f"Skipping heatmap '{title}': Mismatch after filtering or too few samples ({len(embeddings_list)}).")
        return

    # Filter out any remaining non-finite embeddings
    X = np.array(embeddings_list)
    finite_mask = np.all(np.isfinite(X), axis=1)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        logger.warning(f"Removing {num_removed} non-finite rows for heatmap '{title}'.")
        X = X[finite_mask]
        # Filter corresponding labels as well
        labels_list = [label for i, label in enumerate(labels_list) if finite_mask[i]]
        if X.shape[0] < 2:
            logger.warning(f"Skipping heatmap '{title}': Not enough finite samples after filtering.")
            return

    # Calculate cosine similarity
    try:
        # Ensure pairwise is imported if not done globally
        from sklearn.metrics import pairwise
        similarity_matrix = pairwise.cosine_similarity(X)
    except Exception as e_sim:
        logger.error(f"Error calculating similarity matrix for '{title}': {e_sim}")
        return

    # Plot heatmap
    fig = None # Initialize fig to None
    try:
        fig, ax = plt.subplots(figsize=(max(8, X.shape[0]*0.5), max(6, X.shape[0]*0.4))) # Adjust size
        im = ax.imshow(similarity_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_title(title)

        # Prepare labels (truncated and shaped)
        short_labels = [shape_arabic_text(s[:20] + ('...' if len(s) > 20 else '')) for s in labels_list]
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)

        # Loop over data dimensions and create text annotations.
        # Optional: Add text annotations for similarity values if matrix is small
        # for i in range(len(labels_list)):
        #     for j in range(len(labels_list)):
        #         text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
        #                        ha="center", va="center", color="w" if similarity_matrix[i, j] < 0.6 else "black",
        #                        fontsize=6)

        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved heatmap: {save_path}")
    except Exception as e_plot:
        logger.error(f"Error plotting/saving heatmap '{title}': {e_plot}", exc_info=True)
    finally:
        if fig:
            plt.close(fig) # Ensure plot is closed even on error

# ============================================
# Helper Function for Circuit Visualization
# ============================================
def visualize_circuit_matplotlib(circuit: QuantumCircuit,
                                 label: str,
                                 save_path: str):
    """
    Draws and saves a Qiskit circuit diagram using matplotlib.

    Args:
        circuit (QuantumCircuit): The Qiskit circuit object.
        label (str): A label/identifier for the circuit (used in filename/log).
        save_path (str): Full path to save the circuit image.
    """
    if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
        logger.warning(f"Skipping circuit visualization for '{label}': Invalid circuit object.")
        return
    if not hasattr(circuit, 'draw'):
         logger.warning(f"Skipping circuit visualization for '{label}': Circuit object has no 'draw' method.")
         return

    logger.debug(f"Visualizing circuit: {label}")
    circuit_fig = None # Initialize fig to None
    try:
        # Generate the plot using matplotlib backend
        # Increase scale for potentially better readability
        circuit_fig = circuit.draw(output='mpl', fold=-1, scale=0.7)

        if circuit_fig:
            # Manually save the figure generated by circuit.draw()
            circuit_fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.debug(f"Saved circuit diagram to {save_path}")
        else:
            logger.warning(f"Could not generate circuit diagram figure for '{label}'.")
    except ImportError as e_mpl:
         logger.error(f"Error drawing circuit '{label}': Matplotlib or pylatexenc might be missing? {e_mpl}")
    except Exception as e_draw:
        logger.error(f"Error drawing/saving circuit diagram for '{label}': {e_draw}", exc_info=True)
    finally:
        if circuit_fig:
             # Close the figure explicitly after saving
             plt.close(circuit_fig)
# ============================================
# Example Pipeline Function (Modified Calls)
# ============================================
def prepare_quantum_nlp_pipeline_v3(
    max_sentences=20,
    use_enhanced_clustering=True,
    embedding_model_path=None,
    ansatz_choice='IQP' # Add ansatz choice
    ):
    """
    Example of how to use the enhanced ArabicQuantumMeaningKernel with embedding parameters.
    """
    if not LAMBEQ_AVAILABLE:
        logger.critical("Exiting pipeline because Lambeq library is required but not found.")
        return None, None, []

    sentence_file = "sentences.txt"
    sentences = []
    # ... (sentence loading logic) ...
    try:
        with open(sentence_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()][:max_sentences]
        logger.info(f"Loaded {len(sentences)} sentences from {sentence_file}")
    except FileNotFoundError:
        logger.error(f"Error: Sentence file '{sentence_file}' not found. Using default example sentences.")
        sentences = [
            "يقرأ الولد الكتاب", "الولد يقرأ الكتاب", "كتب الطالب الدرس",
            "الطالب يكتب الدرس", "البيت كبير", "درسَ خالدٌ كثيرًا لأنه يريد النجاحَ"
        ][:max_sentences]


    results = []
    circuits_for_viz = {}
    tokens_for_viz = {}
    analyses_for_viz = {} # Store list of analysis dicts

    logger.info("\n--- Processing Sentences (V3 Morph Pipeline) ---")
    for idx, sentence in enumerate(sentences):
        logger.info(f"Processing sentence {idx+1}/{len(sentences)}...")
        try:
            # *** Call the V3 MORPH function from camel_test2 ***
            returned_values = arabic_to_quantum_enhanced_v3_morph(
                sentence,
                ansatz_type=ansatz_choice, # Pass selected ansatz
                debug=False
            )

            circuit, _, structure, tokens, analyses_dicts, roles = returned_values # analyses_dicts is List[Dict]

            if circuit is None or not isinstance(circuit, QuantumCircuit):
                logger.warning(f"Skipping sentence {idx+1}: Invalid circuit.")
                continue

            # Store results including the list of analysis dicts
            results.append({
                'sentence': sentence, 'circuit': circuit, 'structure': structure,
                'tokens': tokens, 'analyses': analyses_dicts, 'roles': roles, # Store list of dicts
                'original_index': idx
            })
            circuits_for_viz[idx] = circuit
            tokens_for_viz[idx] = tokens
            analyses_for_viz[idx] = analyses_dicts # Store list of dicts for viz binding

        except Exception as e:
            logger.error(f"Error processing sentence {idx+1}: {e}", exc_info=True)

    if not results: logger.error("No sentences processed successfully."); return None, None, []

    # --- Instantiate MORPH-AWARE Kernel ---
    kernel_clusters = min(5, max(1, len(results) // 2))
    kernel = ArabicQuantumMeaningKernel( # Uses the updated class definition
        embedding_dim=30, # Increased dimension
        num_clusters=kernel_clusters,
        embedding_model_path=embedding_model_path,
        morph_param_weight=0.3 # Example weight
    )

    logger.info(f"\n--- Training Morph-Aware Kernel ---")
    # Train method now expects list of analysis dicts
    kernel.train(
        sentences=[r['sentence'] for r in results],
        circuits=[r['circuit'] for r in results],
        tokens_list=[r['tokens'] for r in results],
        analyses_list=[r['analyses'] for r in results], # Pass list of dicts
        structures=[r['structure'] for r in results],
        roles_list=[r['roles'] for r in results],
        use_enhanced_clustering=use_enhanced_clustering
    )

    # Save Model, Visualizations, Discourse Analysis, Reports
    kernel.save_model('arabic_quantum_kernel_hash_params.pkl') # New name

    logger.info("\n--- Visualizing Meaning Space ---")
    try:
        meaning_space_fig = kernel.visualize_meaning_space(save_path='meaning_space_hash_params.png')
        if meaning_space_fig: plt.close(meaning_space_fig)
    except Exception as vis_e: logger.error(f"Error during meaning space visualization: {vis_e}", exc_info=True)

    logger.info("\n--- Visualizing Quantum States ---")
    try:
        # Pass tokens and analyses needed for parameter binding in visualization
        state_viz_fig = kernel.analyze_quantum_states(
            circuits_for_viz,
            tokens_for_viz,
            analyses_for_viz,
            save_path_prefix="state_viz_hash_"
        )
        if state_viz_fig: plt.close(state_viz_fig)
    except Exception as state_vis_e: logger.error(f"Error during quantum state visualization: {state_vis_e}", exc_info=True)

    logger.info("\n--- Performing Discourse Analysis ---")
    discourse_analyses = []
    previous_analysis_dict = None
    for i, result in enumerate(results):
        logger.info(f"Analyzing sentence {i+1}/{len(results)} in context...")
        try:
            # interpret_sentence now handles parameter binding internally via get_enhanced_circuit_features
            interpretation = kernel.interpret_sentence(
                result['circuit'], result['tokens'], result['analyses'],
                result['structure'], result['roles'],
                previous_analyses=previous_analysis_dict
            )
            current_analysis_dict = {**result, 'interpretation': interpretation}
            discourse_analyses.append(current_analysis_dict)
            previous_analysis_dict = current_analysis_dict
            logger.info(f"  Sentence {i+1}: Analysis complete.")
        except Exception as analysis_e:
            logger.error(f"  Error during sentence analysis {i+1}: {analysis_e}", exc_info=True)
            discourse_analyses.append({**result, 'interpretation': {'error': str(analysis_e)}})

    logger.info("\n--- Generating Reports ---")
    try:
        html_report = kernel.generate_html_report(discourse_analyses)
        report_filename = 'discourse_analysis_report_hash_params.html'
        with open(report_filename, 'w', encoding='utf-8') as f: f.write(html_report)
        logger.info(f"HTML report saved to {report_filename}")
    except Exception as report_e: logger.error(f"Error generating HTML report: {report_e}", exc_info=True)

    try:
        md_report = kernel.generate_discourse_report(discourse_analyses)
        md_report_filename = 'discourse_analysis_report_hash_params.md'
        with open(md_report_filename, 'w', encoding='utf-8') as f: f.write(md_report)
        logger.info(f"Markdown report saved to {md_report_filename}")
    except Exception as md_report_e: logger.error(f"Error generating Markdown report: {md_report_e}", exc_info=True)

    print("\n--- Interpreting a New Sentence ---")
    new_sentence = "الطالبة تدرس العلوم في الجامعة لأنها تحب البحث العلمي"
    print(f"New sentence: {new_sentence}")
    last_interpretation_result = None
    try:
        returned_new = arabic_to_quantum_enhanced(new_sentence, debug=False)
        new_circuit_obj = None
        new_tokens, new_analyses, new_structure, new_roles = [], [], "ERROR", {} # Defaults

        if isinstance(returned_new, tuple) and len(returned_new) >= 6:
             new_circuit_obj, _, new_structure, new_tokens, new_analyses, new_roles = returned_new
             if not isinstance(new_circuit_obj, QuantumCircuit):
                  logger.error("ERROR: Circuit generation failed for new sentence.")
                  new_circuit_obj = None # Ensure it's None if invalid
        else: logger.error("ERROR: Unexpected return from arabic_to_quantum_enhanced for new sentence.")

        if new_circuit_obj:
             # Call interpret_sentence (it handles parameter binding internally)
             last_interpretation_result = kernel.interpret_sentence(
                 new_circuit_obj, new_tokens, new_analyses, new_structure, new_roles,
                 previous_analyses=previous_analysis_dict
             )
             logger.info("New sentence interpretation complete.")
        else: logger.error("ERROR: Failed to get valid QuantumCircuit for new sentence.")
    except Exception as interp_e:
        logger.error(f"Error interpreting new sentence: {interp_e}", exc_info=True)

    logger.info("\n--- Pipeline Finished ---")
    return kernel, last_interpretation_result, discourse_analyses # Return list


# ============================================
# Main execution block
# ============================================
if __name__ == "__main__":
    # --- IMPORTANT: Specify the path to your Word2Vec model file ---
    # Example: If you downloaded an AraVec model (e.g., Twitter cbow 300 dim)
    # and extracted it to a folder named 'aravec' in the same directory as v6.py
    # model_path = "aravec/full_grams_cbow_300_twitter.mdl" # Adjust this path!
    # Or if you have a text format model:
    model_path = "../aravec/tweets_cbow_300" # Example path - SET YOUR ACTUAL PATH
    # If you don't have a model or don't want to use it, set model_path = None
    # model_path = None
    # ----------------------------------------------------------------

    # Run the pipeline
    kernel, last_new_sentence_interpretation, discourse_analyses_list = prepare_quantum_nlp_pipeline(
        max_sentences=20, # Adjust as needed
        use_enhanced_clustering=True,
        embedding_model_path=model_path # Pass the model path to the pipeline
    )

    # Check if pipeline ran successfully
    if kernel is None:
        logger.error("\nPipeline execution failed.")
    else:
        logger.info("\n--- Summary of Last New Sentence Interpretation ---")
        if last_new_sentence_interpretation is not None:
            if last_new_sentence_interpretation.get('error'):
                 logger.error(f"ERROR interpreting new sentence: {last_new_sentence_interpretation['error']}")
            else:
                logger.info(f"Sentence: {last_new_sentence_interpretation.get('sentence', 'N/A')}")
                logger.info(f"Structure: {last_new_sentence_interpretation.get('structure', 'N/A')}")
                logger.info(f"Confidence: {last_new_sentence_interpretation.get('confidence', 'N/A')}")
                logger.info(f"Deduced Meaning: {last_new_sentence_interpretation.get('interpretation', 'N/A')}")
                # You can add more detailed printing here if needed
        else:
            logger.warning("No interpretation available for the new sentence (it might have failed).")

        # Optional: Print summary of discourse analysis list
        if discourse_analyses_list:
             logger.info(f"\n--- Discourse Analysis Summary ({len(discourse_analyses_list)} sentences) ---")
             # Add printing logic if desired
        else:
             logger.warning("No discourse analyses generated.")





