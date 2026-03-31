# quantum_kernel_v4.py (Enhanced + Embedding Parameter Binding)
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
# Qiskit imports (ensure qiskit and qiskit-aer are installed)
try:
    from qiskit import QuantumCircuit, ClassicalRegister
    from qiskit_aer import AerSimulator # For Qiskit 1.0+
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit or Qiskit-Aer not found. Quantum execution will fail.")
    print("Please install them: pip install qiskit qiskit-aer")
    QISKIT_AVAILABLE = False
    # Define dummy classes if Qiskit not available to avoid NameErrors later
    class QuantumCircuit: pass
    class AerSimulator: pass
    class ClassicalRegister: pass

# Qiskit visualization and info (optional, might require qiskit-ibm-runtime or other extras)
try:
    from qiskit.visualization import plot_state_city
    from qiskit.quantum_info import partial_trace
except ImportError:
    print("Warning: Qiskit visualization/info components not found.")
    plot_state_city = None
    partial_trace = None

import pickle
import os
import traceback
from collections import Counter

# Dependency on camel_test (assuming it returns Qiskit circuit first now)
try:
    from camel_test2 import arabic_to_quantum_enhanced
except ImportError:
    print("ERROR: Cannot import 'arabic_to_quantum_enhanced' from 'camel_test'. Ensure camel_test.py is in the same directory or Python path.")
    # Define a dummy function to avoid NameErrors
    def arabic_to_quantum_enhanced(*args, **kwargs):
        print("ERROR: Dummy arabic_to_quantum_enhanced called because import failed.")
        return None, None, "ERROR", [], [], {}

# Lambeq imports (needed for type checking if diagram is passed)
try:
    from lambeq.backend.grammar import Diagram as GrammarDiagram
    from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram
    LAMBEQ_AVAILABLE = True
except ImportError:
    print("Warning: Lambeq not found. Type checking for diagrams might fail.")
    # Define dummy classes
    class GrammarDiagram: pass
    class LambeqQuantumDiagram: pass
    LAMBEQ_AVAILABLE = False

# --- NEW: Import for Word Embeddings ---
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: gensim not found (pip install gensim). Parameter binding via embeddings disabled.")
    GENSIM_AVAILABLE = False
# --- END NEW ---

# --- NEW: Import for Enhanced Clustering (if using LDA) ---
try:
    from gensim import corpora, models
    GENSIM_LDA_AVAILABLE = True
except ImportError:
    GENSIM_LDA_AVAILABLE = False
    # print("Warning: gensim not found (pip install gensim). Enhanced clustering with LDA disabled.") # Optional warning
# --- END NEW ---


# Helper function to ensure circuit name (applied to Qiskit circuits)
def _ensure_circuit_name(circuit: Any, default_name: str = "qiskit_circuit") -> Any:
    """Checks if a Qiskit circuit object has a name and assigns one if not."""
    if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
        has_name_attr = hasattr(circuit, 'name')
        name_is_set = has_name_attr and getattr(circuit, 'name', None)
        if not name_is_set:
            try:
                # Qiskit circuits might be immutable or have restricted attribute setting
                # Depending on the version. Let's try setting it if possible.
                if hasattr(circuit, '__dict__'): # Check if attributes can be set
                     setattr(circuit, 'name', default_name)
                # print(f"Debug: Assigned default name '{default_name}' to Qiskit circuit.") # DEBUG
            except Exception as e:
                # Silently ignore if name setting fails, as it's not critical for execution
                # print(f"Debug: Could not assign name to Qiskit circuit: {e}.") # DEBUG
                pass
    return circuit

class ArabicQuantumMeaningKernel:
    """
    A quantum kernel for mapping quantum circuit outputs to potential
    sentence meanings for Arabic language processing. Includes discourse analysis
    and parameter binding via word embeddings.
    """
    def __init__(self,
                 embedding_dim: int = 14,
                 num_clusters: int = 5,
                 simulator_backend: Optional[str] = None, # No longer used directly
                 embedding_model_path: Optional[str] = None, # <-- NEW: Path to Word2Vec model
                 params_per_word: int = 3): # <-- NEW: How many params IQPAnsatz generates per word (usually 3)
        """
        Initialize the quantum meaning kernel.

        Args:
            embedding_dim (int): Dimension for the final sentence embeddings.
            num_clusters (int): Default number of meaning clusters.
            simulator_backend (str, optional): Deprecated. Simulator is now AerSimulator.
            embedding_model_path (str, optional): Path to the pre-trained gensim Word2Vec model file.
                                                  If None, random parameters will be used.
            params_per_word (int): Number of parameters the lambeq ansatz (e.g., IQPAnsatz)
                                   is expected to generate per word/box. Default is 3 for IQP.
        """
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.params_per_word = params_per_word # Store how many params map to one word
        self._embedding_model_path = embedding_model_path # Store path for saving/loading

        # --- Simulator Initialization (Qiskit 1.0+) ---
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
            print("Initialized Qiskit AerSimulator.")
        else:
            self.simulator = None
            print("Warning: Qiskit not available. Simulator set to None.")
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
            print(f"Loading word embedding model from: {embedding_model_path}...")
            try:
                # Use KeyedVectors for potentially faster loading and lower memory
                self.embedding_model = KeyedVectors.load_word2vec_format(embedding_model_path, binary=False) # Adjust binary=True if your model is binary
                print(f"Word embedding model loaded successfully. Vector size: {self.embedding_model.vector_size}")
            except Exception as e:
                print(f"ERROR: Failed to load word embedding model: {e}")
                print("       Falling back to random parameter assignment.")
                self.embedding_model = None
        elif not GENSIM_AVAILABLE:
             print("Info: Gensim not installed. Cannot load embeddings. Using random parameters.")
        else:
             print("Info: No embedding model path provided. Using random parameters.")
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
            db_path = MorphologyDB.builtin_db('calima-msa-r13') # Example DB
            self.camel_analyzer = Analyzer(db_path)
            print("CAMeL Tools Analyzer initialized successfully.")
        except ImportError:
            print("Warning: CAMeL Tools not found (pip install camel-tools). NLP enhancements disabled.")
        except LookupError:
             print("Warning: CAMeL Tools default DB not found. Run 'camel_tools download calima-msa-r13' or specify DB path. NLP enhancements disabled.")
        except Exception as e:
            print(f"Warning: Error initializing CAMeL Tools Analyzer: {e}. NLP enhancements disabled.")


    # --- NEW: Parameter Binding Function ---
    def _bind_parameters(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Tuple]) -> Dict:
        """
        Creates parameter bindings for a circuit based on word embeddings or random values.

        Args:
            circuit: The parameterized Qiskit QuantumCircuit.
            tokens: The list of tokens for the sentence.
            analyses: The list of (lemma, pos, dep, head) tuples.

        Returns:
            A dictionary mapping Parameter objects to numerical values (wrapped in a list),
            or an empty dictionary if no parameters or binding fails.
        """
        parameter_binds = {}
        # Ensure circuit is valid and has parameters attribute
        if not isinstance(circuit, QuantumCircuit) or not hasattr(circuit, 'parameters'):
             print("Warning: Invalid circuit passed to _bind_parameters.")
             return {}

        params = circuit.parameters
        if not params:
            return {} # No parameters to bind

        num_params = len(params)
        param_values = [] # This will hold the final numerical values

        if self.embedding_model:
            # --- Bind using Word Embeddings ---
            print(f"  Binding {num_params} parameters using word embeddings...")
            params_bound_count = 0
            word_idx = 0
            # Simple sequential mapping: Assume params map to words in order
            # IQPAnsatz default: 3 params per word/box
            while params_bound_count < num_params and word_idx < len(analyses):
                lemma = analyses[word_idx][0] # Use lemma for lookup
                token = tokens[word_idx]

                try:
                    # Get embedding vector
                    if lemma in self.embedding_model:
                        vector = self.embedding_model[lemma]
                    elif token in self.embedding_model: # Fallback to token if lemma not found
                         vector = self.embedding_model[token]
                         # print(f"  Warning: Lemma '{lemma}' not in vocab, using token '{token}'.") # Debug
                    else:
                         # print(f"  Warning: Word '{token}' (lemma '{lemma}') not in embedding model. Using default values.") # Debug
                         vector = np.zeros(self.embedding_model.vector_size) # Default to zero vector

                    # Extract and scale components for the parameters associated with this word
                    num_params_for_this_word = min(self.params_per_word, num_params - params_bound_count)
                    for i in range(num_params_for_this_word):
                        if i < len(vector):
                            # Simple Scaling: Use modulo 2*pi
                            scaled_value = float(vector[i]) % (2 * np.pi) # Ensure float
                            param_values.append(scaled_value)
                        else:
                            # Not enough embedding dimensions, use default
                            # print(f"  Warning: Not enough embedding dimensions for word '{token}'. Using 0.0 for param {i+1}.") # Debug
                            param_values.append(0.0)
                        params_bound_count += 1

                except Exception as e_embed:
                    print(f"  Error processing embedding for word '{token}': {e_embed}. Using defaults.")
                    # Add default values if error occurs
                    num_params_to_add = min(self.params_per_word, num_params - params_bound_count)
                    param_values.extend([0.0] * num_params_to_add)
                    params_bound_count += num_params_to_add

                word_idx += 1

            # Check if enough values were generated
            if len(param_values) < num_params:
                print(f"Warning: Parameter/Word count mismatch. Needed {num_params}, generated {len(param_values)}. Padding with 0.0.")
                param_values.extend([0.0] * (num_params - len(param_values)))
            elif len(param_values) > num_params:
                 print(f"Warning: Generated more parameter values ({len(param_values)}) than needed ({num_params}). Truncating.")
                 param_values = param_values[:num_params]

        else:
            # --- Bind using Random Values (Fallback) ---
            print(f"  Binding {num_params} parameters randomly (no embedding model).")
            param_values = (np.random.rand(num_params) * 2 * np.pi).tolist() # Ensure list of floats

        # --- FIX: Wrap each value in a list for qiskit-aer compatibility ---
        # Create the final binding dictionary mapping Parameter -> [value]
        parameter_binds = {param: [value] for param, value in zip(params, param_values)}
        # --- END FIX ---

        return parameter_binds
    # --- END NEW ---


    # --- Feature Extraction and Embedding ---
    # MODIFIED: Accept tokens and analyses for parameter binding
    def get_circuit_features(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Tuple], shots: int = 1024, debug: bool = False) -> np.ndarray:
        """
        Extracts features from a Qiskit quantum circuit by executing it,
        binding parameters using embeddings or randomly if needed.

        Args:
            circuit: The Qiskit QuantumCircuit (potentially parameterized).
            tokens: List of tokens for the sentence.
            analyses: List of (lemma, pos, dep, head) tuples for the sentence.
            shots: Number of shots for simulation (if not using statevector).
            debug: Print debug info.

        Returns:
            A numpy array of features derived from the circuit execution.
        """
        fallback_features = np.zeros(self.embedding_dim) # Use zero vector as fallback

        if not QISKIT_AVAILABLE or self.simulator is None:
            print("Error: Qiskit or simulator not available in get_circuit_features.")
            return fallback_features
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_circuit_features received non-Qiskit object: {type(circuit)}")
             return fallback_features

        try:
            circuit = _ensure_circuit_name(circuit, "circuit_for_features")

            # --- MODIFIED: Parameter Binding ---
            parameter_binds = self._bind_parameters(circuit, tokens, analyses)
            # run() expects a list of bind dictionaries, one for each parameter set/circuit.
            # Since we run one circuit with one set of params, pass a list containing our dict.
            bindings_list = [parameter_binds] if parameter_binds else None
            # --- END MODIFIED ---

            if debug: print(f"  Executing circuit '{getattr(circuit, 'name', 'N/A')}'...")
            # Use run() method of the simulator instance
            job = self.simulator.run(circuit, shots=shots, parameter_binds=bindings_list)
            result = job.result() # Call result() to get the Result object
            if debug: print("  Execution successful.")

            # Extract features (Statevector preferred, fallback to counts)
            # Check for statevector method using attribute access on result object
            if hasattr(result, 'get_statevector'):
                try:
                    statevector = result.get_statevector(circuit) # Pass circuit for statevector method
                    if debug: print("  Extracting features from statevector...")
                    amplitudes = np.abs(statevector)
                    phases = np.angle(statevector)
                    # Simple feature construction (can be improved)
                    half_dim = self.embedding_dim // 2
                    amp_slice = amplitudes[:min(half_dim, len(amplitudes))]
                    phase_slice = phases[:min(half_dim, len(phases))]
                    features = np.concatenate([amp_slice, phase_slice])
                    # Pad or truncate features to match embedding_dim
                    if len(features) < self.embedding_dim: features = np.pad(features, (0, self.embedding_dim - len(features)), 'constant')
                    elif len(features) > self.embedding_dim: features = features[:self.embedding_dim]
                    # Normalize
                    norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
                    return features
                except Exception as e_sv:
                     print(f"  Error getting/processing statevector: {e_sv}. Falling back to counts.")
                     # Fall through to counts extraction if statevector fails

            # Fallback to counts if statevector not available or failed
            if debug: print("  Extracting features from counts...")
            # Ensure get_counts is called on the Result object
            counts = result.get_counts(circuit) # Pass circuit to get_counts
            total_shots_actual = sum(counts.values())
            feature_vector = np.zeros(self.embedding_dim)
            if total_shots_actual > 0:
                 for outcome, count in counts.items():
                     try:
                         # Simple hashing for outcome -> feature index mapping
                         idx_base = int(outcome, 2) if isinstance(outcome, str) and all(c in '01' for c in outcome) else hash(outcome)
                         idx = idx_base % self.embedding_dim
                         feature_vector[idx] += count / total_shots_actual
                     except Exception as e_idx: print(f"  Warning: Could not process outcome '{outcome}': {e_idx}")
            # Normalize
            norm = np.linalg.norm(feature_vector); feature_vector = feature_vector / norm if norm > 0 else feature_vector
            return feature_vector

        except Exception as e:
            print(f"\n--- ERROR in get_circuit_features ---")
            print(f"Error type: {type(e)}")
            print(f"Error message: {e}")
            traceback.print_exc()
            print(f"---------------------------------------\n")
            return fallback_features # Return fallback on error

    def _get_reduced_density_matrix(self, statevector, subsystem_qubits):
        """ Calculates reduced density matrix using Qiskit's partial_trace if available. """
        if partial_trace is None:
             print("Warning: qiskit.quantum_info.partial_trace not available. Cannot calculate reduced density matrix.")
             return None
        try:
            n_qubits = int(np.log2(len(statevector)))
            if not subsystem_qubits or max(subsystem_qubits) >= n_qubits or min(subsystem_qubits) < 0:
                 raise ValueError("Invalid subsystem qubits specified.")
            # Qiskit statevector is |psi>, density matrix rho = |psi><psi|
            rho = np.outer(statevector, np.conj(statevector))
            trace_out_qubits = [i for i in range(n_qubits) if i not in subsystem_qubits]
            reduced_rho_qiskit = partial_trace(rho, trace_out_qubits).data # .data gets numpy array
            return reduced_rho_qiskit
        except ValueError as ve:
             print(f"Error in _get_reduced_density_matrix (ValueError): {ve}")
        except Exception as e:
            print(f"Error using qiskit.partial_trace: {e}. Falling back.")
        # Fallback: Return None or a default state if calculation fails
        return None

    # MODIFIED: Accept tokens and analyses for parameter binding
    def get_enhanced_circuit_features(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Tuple], shots: int = 1024, debug: bool = False) -> np.ndarray:
        """ Enhanced feature extraction including basic features, entanglement, and Pauli expectations. """
        fallback_features = np.zeros(self.embedding_dim)

        if not QISKIT_AVAILABLE or self.simulator is None:
            print("Error: Qiskit or simulator not available in get_enhanced_circuit_features.")
            return fallback_features
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_enhanced_circuit_features received non-Qiskit object: {type(circuit)}")
             return fallback_features

        basic_features = fallback_features # Initialize basic_features
        try:
            # Get basic features (this already includes parameter binding)
            basic_features = self.get_circuit_features(circuit, tokens, analyses, shots, debug)

            # --- Additional features (Entanglement, Pauli) ---
            entanglement_features = []
            pauli_expectations = []

            # Bind parameters again (needed for statevector simulation if not done before)
            parameter_binds = self._bind_parameters(circuit, tokens, analyses)
            bindings_list = [parameter_binds] if parameter_binds else None

            # Execute specifically for statevector
            job_sv = self.simulator.run(circuit, parameter_binds=bindings_list, shots=None) # Request statevector
            result_sv = job_sv.result()

            if hasattr(result_sv, 'get_statevector'):
                statevector = result_sv.get_statevector(circuit)
                num_qubits = circuit.num_qubits

                # 1. Entanglement (Von Neumann Entropy of reduced states)
                if num_qubits > 1:
                    num_entanglement_checks = min(num_qubits - 1, 3) # Check first few qubits
                    for i in range(num_entanglement_checks):
                        reduced_density = self._get_reduced_density_matrix(statevector, [i])
                        if reduced_density is not None:
                            try:
                                eigenvalues = np.linalg.eigvalsh(reduced_density)
                                valid_eigenvalues = eigenvalues[eigenvalues > 1e-12] # Threshold for numerical stability
                                entropy = -np.sum(valid_eigenvalues * np.log2(valid_eigenvalues)) if len(valid_eigenvalues) > 0 else 0.0
                                entanglement_features.append(np.real(entropy))
                            except np.linalg.LinAlgError:
                                print(f"  Warning: LinAlgError calculating eigenvalues for qubit {i}. Appending 0 entropy.")
                                entanglement_features.append(0.0)
                        else:
                             entanglement_features.append(0.0) # Append 0 if reduced density failed

                # 2. Pauli Expectations (Example: Pauli X on first few qubits)
                num_pauli_checks = min(num_qubits, 3)
                for i in range(num_pauli_checks):
                    try:
                        # Create circuit to measure Pauli X on qubit i
                        meas_circuit = circuit.copy(name=f"{getattr(circuit, 'name', 'unnamed')}_pauli_x_q{i}")
                        meas_circuit.h(i) # Apply Hadamard for X basis measurement
                        # Ensure classical register exists and is large enough
                        cr_name = f'c_pauli_{i}'
                        # Find or create a suitable classical register
                        cr = None
                        for reg in meas_circuit.cregs:
                             if reg.size >= num_qubits: # Found a large enough register
                                  cr = reg
                                  break
                        if cr is None: # If no suitable register found, create one
                             cr = ClassicalRegister(num_qubits, name=cr_name)
                             meas_circuit.add_register(cr)

                        # Measure qubit i into classical bit i
                        if i < cr.size:
                             meas_circuit.measure(i, i) # Measure qubit i to classical bit i
                        else:
                             print(f"Warning: Classical register '{cr.name}' too small for qubit {i}. Skipping Pauli X measure.")
                             pauli_expectations.append(0.0)
                             continue

                        # Bind parameters and run measurement circuit
                        # Re-bind parameters for the copied circuit
                        pauli_binds = self._bind_parameters(meas_circuit, tokens, analyses)
                        pauli_bindings_list = [pauli_binds] if pauli_binds else None
                        pauli_job = self.simulator.run(meas_circuit, shots=shots, parameter_binds=pauli_bindings_list)
                        pauli_result = pauli_job.result()
                        pauli_counts = pauli_result.get_counts(meas_circuit)

                        # Calculate expectation value <X> = P(0) - P(1)
                        exp_val = 0.0
                        total_pauli_shots = sum(pauli_counts.values())
                        if total_pauli_shots > 0:
                             prob_0, prob_1 = 0.0, 0.0
                             for bitstring, count in pauli_counts.items():
                                  # Qiskit bitstrings are little-endian (rightmost is qubit 0)
                                  bit_index = i
                                  if bit_index < len(bitstring):
                                       measured_bit = bitstring[len(bitstring) - 1 - bit_index]
                                       if measured_bit == '0':
                                           prob_0 += count / total_pauli_shots
                                       else:
                                           prob_1 += count / total_pauli_shots
                             exp_val = prob_0 - prob_1
                        pauli_expectations.append(exp_val)
                    except Exception as pauli_e:
                        print(f"  Error calculating Pauli X expectation for qubit {i}: {type(pauli_e).__name__} - {pauli_e}")
                        pauli_expectations.append(0.0) # Append default on error

            # Combine all features
            feature_list = [basic_features]
            if entanglement_features: feature_list.append(np.array(entanglement_features))
            if pauli_expectations: feature_list.append(np.array(pauli_expectations))

            # Concatenate if multiple feature types exist
            all_features = np.concatenate(feature_list) if len(feature_list) > 1 else basic_features

            # Pad or truncate to match embedding_dim
            current_len = len(all_features)
            if current_len < self.embedding_dim: all_features = np.pad(all_features, (0, self.embedding_dim - current_len), 'constant')
            elif current_len > self.embedding_dim: all_features = all_features[:self.embedding_dim]

            # Normalize final vector
            norm = np.linalg.norm(all_features); all_features = all_features / norm if norm > 0 else all_features
            return all_features

        except Exception as e:
            print(f"\n--- ERROR in get_enhanced_circuit_features ---")
            print(f"Error type: {type(e)}")
            print(f"Error message: {e}")
            traceback.print_exc()
            print(f"----------------------------------------------\n")
            # Fallback to basic features if enhancement fails
            return basic_features # Return basic_features computed at the start

    # --- Linguistic Feature Extraction (No changes needed here for parameter binding) ---
    def extract_linguistic_features(self, tokens: List[str], analyses: List[Tuple], structure: str, roles: Dict) -> np.ndarray:
        """ Extracts basic linguistic features. """
        features = np.zeros(self.embedding_dim)
        num_features = 0
        structure_map = {'VSO': 0, 'SVO': 1, 'NOMINAL': 2, 'COMPLEX': 3, 'OTHER': 4}
        structure_idx = structure_map.get(structure.split('_')[0], 4) # Use base structure before COMPLEX_
        if num_features < self.embedding_dim: features[num_features] = structure_idx / (len(structure_map) -1); num_features += 1
        pos_counts = Counter(pos for _, pos, _, _ in analyses)
        total_tokens = max(1, len(tokens))
        if num_features < self.embedding_dim: features[num_features] = pos_counts.get('VERB', 0) / total_tokens; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = pos_counts.get('NOUN', 0) / total_tokens; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = pos_counts.get('ADJ', 0) / total_tokens; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = 1.0 if roles.get('verb') is not None else 0.0; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = 1.0 if roles.get('subject') is not None else 0.0; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = 1.0 if roles.get('object') is not None else 0.0; num_features += 1
        has_negation = any(lemma in ['لا', 'ليس', 'غير', 'لم', 'لن'] for lemma, _, _, _ in analyses)
        if num_features < self.embedding_dim: features[num_features] = 1.0 if has_negation else 0.0; num_features += 1
        norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
        return features

    def extract_complex_linguistic_features(self, tokens, analyses, structure, roles):
        """ Enhanced linguistic feature extraction including complexity metrics. """
        features = self.extract_linguistic_features(tokens, analyses, structure, roles)
        feature_idx = 8 # Start adding complex features after basic ones
        if feature_idx < self.embedding_dim:
            subordinate_markers = ['الذي', 'التي', 'الذين', 'اللواتي', 'عندما', 'حيث', 'لأن', 'كي', 'أنّ']
            has_subordinate = any(token in subordinate_markers for token in tokens)
            features[feature_idx] = 1.0 if has_subordinate else 0.0; feature_idx += 1
        if feature_idx < self.embedding_dim:
            verb_count = sum(1 for _, pos, _, _ in analyses if pos == 'VERB')
            features[feature_idx] = min(verb_count / 3.0, 1.0); feature_idx += 1 # Normalize, cap at 3
        if feature_idx < self.embedding_dim:
            conditional_markers = ['إذا', 'لو', 'إن']
            has_conditional = any(token in conditional_markers for token in tokens)
            features[feature_idx] = 1.0 if has_conditional else 0.0; feature_idx += 1
        if feature_idx < self.embedding_dim:
            quotation_markers = ['قال', 'صرح', 'أعلن', 'ذكر', 'أضاف']
            has_quotation = any(lemma in quotation_markers for lemma, _, _, _ in analyses)
            features[feature_idx] = 1.0 if has_quotation else 0.0; feature_idx += 1
        if feature_idx < self.embedding_dim:
            avg_word_length = np.mean([len(token) for token in tokens]) if tokens else 0
            features[feature_idx] = min(avg_word_length / 10.0, 1.0); feature_idx += 1 # Normalize
        if feature_idx < self.embedding_dim:
             features[feature_idx] = min(len(tokens) / 50.0, 1.0); feature_idx += 1 # Normalize based on max 50 tokens
        norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
        return features

    def combine_features_with_attention(self, quantum_features, linguistic_features, structure):
        """ Combine features with attention mechanism based on sentence structure. """
        base_structure = structure.split('_')[0] # Use base structure (VSO, SVO, NOMINAL) for weighting
        if base_structure == 'NOMINAL': quantum_weight, linguistic_weight = 0.3, 0.7
        elif base_structure in ['VSO', 'SVO']: quantum_weight, linguistic_weight = 0.5, 0.5
        elif base_structure == 'COMPLEX': quantum_weight, linguistic_weight = 0.6, 0.4 # Higher weight for quantum if complex
        else: quantum_weight, linguistic_weight = 0.5, 0.5 # Default equal weighting
        # Ensure consistent dimensions
        q_len = len(quantum_features); l_len = len(linguistic_features)
        if q_len < self.embedding_dim: quantum_features = np.pad(quantum_features, (0, self.embedding_dim - q_len), 'constant')
        elif q_len > self.embedding_dim: quantum_features = quantum_features[:self.embedding_dim]
        if l_len < self.embedding_dim: linguistic_features = np.pad(linguistic_features, (0, self.embedding_dim - l_len), 'constant')
        elif l_len > self.embedding_dim: linguistic_features = linguistic_features[:self.embedding_dim]
        combined = quantum_weight * quantum_features + linguistic_weight * linguistic_features
        norm = np.linalg.norm(combined); combined = combined / norm if norm > 0 else combined
        return combined

    # --- Training and Clustering ---
    # MODIFIED: Pass tokens/analyses to feature extraction
    def train(self, sentences, circuits, tokens_list, analyses_list, structures, roles_list, use_enhanced_clustering=False):
        """ Train the kernel on a set of sentences and their Qiskit circuits. """
        self.reference_sentences = {i: sentences[i] for i in range(len(sentences))}
        self.circuit_embeddings = {}; self.sentence_embeddings = {}; embeddings = []
        print(f"\n--- Training Kernel on {len(sentences)} sentences ---")
        # Check for mismatched lengths
        if not (len(circuits) == len(tokens_list) == len(analyses_list) == len(structures) == len(roles_list) == len(sentences)):
            print("Error: Input lists to train method have mismatched lengths.")
            min_len = min(len(sentences), len(circuits), len(tokens_list), len(analyses_list), len(structures), len(roles_list))
            print(f"Warning: Training with reduced dataset size: {min_len}")
            if min_len == 0: print("Error: Cannot train with empty dataset."); return self
            sentences, circuits, tokens_list, analyses_list, structures, roles_list = (lst[:min_len] for lst in [sentences, circuits, tokens_list, analyses_list, structures, roles_list])
        else: min_len = len(sentences)

        # Extract features and create embeddings
        for i in range(min_len):
            try:
                current_circuit = circuits[i]
                if not isinstance(current_circuit, QuantumCircuit):
                     print(f"  WARNING: Skipping embedding for sentence {i+1}. Expected QuantumCircuit, got {type(current_circuit)}")
                     continue

                # --- MODIFIED CALL: Pass tokens and analyses ---
                quantum_features = self.get_enhanced_circuit_features(
                    current_circuit, tokens_list[i], analyses_list[i]
                )
                # --- END MODIFIED CALL ---

                self.circuit_embeddings[i] = quantum_features # Store quantum features if needed separately
                linguistic_features = self.extract_complex_linguistic_features(
                    tokens_list[i], analyses_list[i], structures[i], roles_list[i]
                )
                embedding = self.combine_features_with_attention(
                    quantum_features, linguistic_features, structures[i]
                )
                self.sentence_embeddings[i] = embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"  ERROR processing sentence {i+1} during training embedding generation: {sentences[i]}")
                print(f"  Error type: {type(e).__name__}, Message: {e}")
                traceback.print_exc()
                print("  Skipping embedding for this sentence.")

        if not embeddings: print("Error: No embeddings generated. Cannot proceed."); return self

        # Perform clustering
        if use_enhanced_clustering and GENSIM_LDA_AVAILABLE:
            print("Learning meaning clusters (enhanced with topic modeling)...")
            self.learn_enhanced_meaning_clusters(embeddings, sentences)
        else:
            if use_enhanced_clustering and not GENSIM_LDA_AVAILABLE:
                print("Warning: Enhanced clustering requested but gensim not available. Falling back to basic clustering.")
            print("Learning meaning clusters (basic)...")
            self.learn_meaning_clusters(embeddings)

        print("Assigning meaning to clusters...")
        self.assign_meaning_to_clusters(sentences, structures, roles_list, analyses_list)
        print("--- Training Complete ---")
        return self

    # --- Clustering Methods (No changes needed here) ---
    def learn_meaning_clusters(self, embeddings: List[np.ndarray]) -> None:
        """ Learn meaning clusters from embeddings using KMeans. """
        if not embeddings: print("Warning: No embeddings provided for clustering."); return
        X = np.array(embeddings); n_samples = X.shape[0]
        if n_samples == 0: print("Warning: Embedding array is empty. Cannot cluster."); return
        n_clusters = min(self.num_clusters, n_samples)
        if n_clusters <= 0: n_clusters = 1
        self.num_clusters = n_clusters
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(X)
            self.meaning_clusters = kmeans.cluster_centers_
        except Exception as e:
            print(f"Error during KMeans clustering: {e}"); traceback.print_exc()
            self.cluster_labels = None; self.meaning_clusters = None

    def learn_enhanced_meaning_clusters(self, embeddings, sentences):
        """Learn meaning clusters with topic modeling enhancement"""
        self.learn_meaning_clusters(embeddings) # Basic KMeans first
        if self.meaning_clusters is None or self.cluster_labels is None:
            print("Skipping topic modeling enhancement due to prior clustering failure.")
            return
        if not GENSIM_LDA_AVAILABLE:
            print("Skipping topic modeling enhancement: gensim library not available.")
            return
        print("Enhancing clusters with LDA topic modeling...")
        try:
            tokenized_sentences = [sentence.split() for sentence in sentences]
            arabic_stopwords = ['من', 'في', 'على', 'الى', 'إلى', 'عن', 'و', 'ف', 'ثم', 'أو', 'لا', 'ما', 'هو', 'هي', 'هم', 'هن', 'هذا', 'هذه', 'ذلك', 'تلك', 'الذي', 'التي', 'الذين', 'قد', 'لقد', 'أن', 'ان', 'إن', 'كان', 'يكون', 'لم', 'لن', 'كل', 'بعض', 'يا', 'اي', 'أي', 'مع', 'به', 'له', 'فيه', 'تم']
            filtered_sentences = [[word for word in sentence if word not in arabic_stopwords and len(word) > 1] for sentence in tokenized_sentences]
            dictionary = corpora.Dictionary(filtered_sentences)
            corpus = [dictionary.doc2bow(text) for text in filtered_sentences]
            if not corpus or not dictionary:
                print("Warning: Corpus or dictionary is empty after filtering. Skipping topic modeling.")
                return
            num_topics = min(self.num_clusters * 2, len(dictionary), 15) # Heuristic
            if num_topics <= 1:
                 print("Warning: Not enough unique terms for topic modeling. Skipping.")
                 return
            print(f"Training LDA model with {num_topics} topics...")
            lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
            sentence_topics = [lda_model.get_document_topics(doc, minimum_probability=0.1) for doc in corpus]
            if not self.meaning_map:
                 print("Warning: meaning_map not initialized before topic assignment. Topics might not be stored correctly unless assign_meaning_to_clusters is called after.")
                 pass
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
                     topic_words[f"Topic {topic_id}"] = {'words': [word for word, _ in words], 'weight': total_prob}
                if cluster_id in self.meaning_map:
                    self.meaning_map[cluster_id]['topics'] = topic_words
                else:
                    self.meaning_map[cluster_id] = {'topics': topic_words}
            print("Topic modeling enhancement complete.")
        except Exception as e:
            print(f"Topic modeling enhancement failed: {e}")
            traceback.print_exc()

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

    def assign_meaning_to_clusters(self, sentences: List[str], structures: List[str], roles_list: List[Dict], analyses_list: List[List[Tuple]]) -> Dict:
        """ Assign meaning templates to clusters based on linguistic analysis. """
        if self.cluster_labels is None or len(self.cluster_labels) == 0: print("Warning: Cluster labels not found. Cannot assign meanings."); return self.meaning_map
        cluster_data_grouped = {label: [] for label in set(self.cluster_labels)}
        min_len = min(len(sentences), len(structures), len(roles_list), len(analyses_list), len(self.cluster_labels))
        for i in range(min_len):
            label = self.cluster_labels[i]
            cluster_data_grouped[label].append({'sentence': sentences[i], 'structure': structures[i], 'roles': roles_list[i], 'analyses': analyses_list[i], 'index': i})
        for cluster_id, cluster_data in cluster_data_grouped.items():
            if not cluster_data: continue
            verb_lemmas = Counter(); subject_lemmas = Counter(); object_lemmas = Counter(); common_preps = Counter(); verb_tenses = Counter(); verb_moods = Counter(); structure_counts = Counter()
            for item in cluster_data:
                structure_counts[item['structure']] += 1; tokens = item['sentence'].split(); roles = item['roles']; analyses = item['analyses']
                verb_idx = roles.get('verb'); subj_idx = roles.get('subject'); obj_idx = roles.get('object')
                if verb_idx is not None and verb_idx < len(analyses): verb_lemmas[analyses[verb_idx][0]] += 1
                if subj_idx is not None and subj_idx < len(analyses): subject_lemmas[analyses[subj_idx][0]] += 1
                if obj_idx is not None and obj_idx < len(analyses): object_lemmas[analyses[obj_idx][0]] += 1
                for i, (lemma, pos, dep, head) in enumerate(analyses):
                    if pos == 'ADP' and (head == verb_idx or head == obj_idx):
                         prep_obj_lemma = "THING"; found_prep_obj = False
                         for j, (_, _, dep_obj, head_obj) in enumerate(analyses):
                              if head_obj == i and dep_obj == 'obj': prep_obj_lemma = analyses[j][0]; found_prep_obj = True; break
                         if not found_prep_obj:
                             for j, (_, _, dep_obj, head_obj) in enumerate(analyses):
                                 if head_obj == i and dep_obj == 'nmod': prep_obj_lemma = analyses[j][0]; break
                         common_preps[(lemma, prep_obj_lemma)] += 1
                if self.camel_analyzer:
                    try:
                        morph_analysis_list = self.camel_analyzer.analyze(item['sentence'])
                        if verb_idx is not None and verb_idx < len(morph_analysis_list):
                            verb_morph_analyses = morph_analysis_list[verb_idx]
                            if verb_morph_analyses:
                                verb_morph = verb_morph_analyses[0]
                                verb_tenses[verb_morph.get('asp', 'UNK')] += 1
                                verb_moods[verb_morph.get('mod', 'UNK')] += 1
                    except Exception as e: pass
            dominant_structure = structure_counts.most_common(1)[0][0] if structure_counts else 'OTHER'
            dominant_verb = verb_lemmas.most_common(1)[0][0] if verb_lemmas else None
            dominant_subj = subject_lemmas.most_common(1)[0][0] if subject_lemmas else "SUBJECT"
            dominant_obj = object_lemmas.most_common(1)[0][0] if object_lemmas else "OBJECT"
            top_prep = common_preps.most_common(1)[0][0] if common_preps else None
            dominant_tense = verb_tenses.most_common(1)[0][0] if verb_tenses else None
            dominant_mood = verb_moods.most_common(1)[0][0] if verb_moods else None
            deduced_template = f"{dominant_subj} (did something involving) {dominant_obj}"; verb_class = self._classify_verb(dominant_verb)
            if verb_class == "MOTION": dest = top_prep[1] if top_prep and top_prep[0] in ['إلى', 'ل'] else "DESTINATION"; deduced_template = f"{dominant_subj} went to {dest}"
            elif verb_class == "COMMUNICATION": msg = dominant_obj if dominant_obj != "OBJECT" else "MESSAGE"; deduced_template = f"{dominant_subj} said {msg}"
            elif verb_class == "POSSESSION": item = dominant_obj if dominant_obj != "OBJECT" else "ITEM"; deduced_template = f"{dominant_subj} has/got {item}"
            elif verb_class == "COGNITION": thought = dominant_obj if dominant_obj != "OBJECT" else "IDEA"; deduced_template = f"{dominant_subj} thinks about {thought}"
            elif verb_class == "EMOTION": stimulus = dominant_obj if dominant_obj != "OBJECT" else "SOMETHING"; deduced_template = f"{dominant_subj} feels emotion about {stimulus}"
            elif dominant_verb:
                 base_structure = dominant_structure.split('_')[0]
                 if base_structure == 'VSO': action_desc = f"{dominant_verb} performed by {dominant_subj}"; action_desc += f" on {dominant_obj}" if dominant_obj != "OBJECT" else ""; deduced_template = action_desc
                 elif base_structure == 'SVO': action_desc = f"{dominant_subj} performs {dominant_verb}"; action_desc += f" on {dominant_obj}" if dominant_obj != "OBJECT" else ""; deduced_template = action_desc
                 elif base_structure == 'NOMINAL':
                      predicate = "ATTRIBUTE"; pred_lemmas = Counter()
                      for item_inner in cluster_data:
                          subj_idx_inner = item_inner['roles'].get('subject')
                          if subj_idx_inner is not None:
                              for k, (_, pos_k, _, head_k) in enumerate(item_inner['analyses']):
                                  if head_k == subj_idx_inner and pos_k == 'ADJ': pred_lemmas[item_inner['analyses'][k][0]] += 1; break
                      if pred_lemmas: predicate = pred_lemmas.most_common(1)[0][0]
                      deduced_template = f"{dominant_subj} is {predicate}"
                 else: deduced_template = f"Statement about {dominant_subj} involving {dominant_verb}"
            tense_map = {'p': ' (past)', 'i': ' (present)', 'c': ' (command)'}; deduced_template += tense_map.get(dominant_tense, "")
            sentiment_label = None # Add sentiment logic if needed
            if cluster_id not in self.meaning_map: self.meaning_map[cluster_id] = {}
            self.meaning_map[cluster_id].update({
                'structure': dominant_structure, 'deduced_template': deduced_template, 'dominant_verb': dominant_verb, 'dominant_subject': dominant_subj,
                'dominant_object': dominant_obj, 'common_prep_phrase': top_prep, 'sentiment': sentiment_label, 'examples': [item['sentence'] for item in cluster_data[:3]],
                'original_templates': self.semantic_templates.get(dominant_structure.split('_')[0], self.semantic_templates.get('OTHER', {}))
            })
            if 'topics' not in self.meaning_map[cluster_id]: self.meaning_map[cluster_id]['topics'] = {}
        return self.meaning_map

    # --- Interpretation and Discourse ---
    # MODIFIED: Pass tokens/analyses to feature extraction
    def interpret_sentence(self, circuit: QuantumCircuit, tokens: List[str], analyses: List[Tuple], structure: str, roles: Dict, previous_analyses=None) -> Dict:
        """ Interpret sentence meaning based on circuit, linguistics, and optionally context. """
        # Context Handling: Redirect if context is provided
        if previous_analyses is not None:
             return self.analyze_sentence_in_context(circuit, tokens, analyses, structure, roles, previous_analyses)

        # Direct Interpretation (No Context or Base Interpretation)
        if not isinstance(circuit, QuantumCircuit):
            print(f"  ERROR in interpret_sentence: Expected QuantumCircuit, got {type(circuit)}. Cannot proceed.")
            return {'error': f'Invalid circuit type: {type(circuit)}'} # Return error dict

        # --- MODIFIED CALL: Pass tokens and analyses ---
        quantum_features = self.get_enhanced_circuit_features(circuit, tokens, analyses)
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
            print(f"    Error extracting semantic frames: {frame_e}")
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
                print("Warning: Embedding dimension/type mismatch. Skipping context blending.")
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

    def create_specific_interpretation(self, tokens: List[str], analyses: List[Tuple], roles: Dict, structure: str, templates: Dict) -> Dict:
        """ Creates a specific interpretation by filling templates with actual values. """
        subject = "unknown"; verb = "unknown"; predicate = "unknown"; object_text = "unknown"
        verb_lemma = None; tense = "present"; modality = "indicative"
        verb_idx = roles.get('verb'); subj_idx = roles.get('subject'); obj_idx = roles.get('object')
        if verb_idx is not None and verb_idx < len(tokens): verb = tokens[verb_idx]; verb_lemma = analyses[verb_idx][0] if verb_idx < len(analyses) else verb
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
            except Exception as e: print(f"Warning: Error in CAMeL semantic property extraction: {e}")
        # Optional: Add Farasa NER / AraVec calls if models are loaded
        return {'sentence': sentence, 'frames': frames}

    def extract_enhanced_semantic_frames(self, tokens, analyses, roles):
        """More comprehensive semantic frame extraction including rhetorical relations, etc."""
        basic_frames_data = self.extract_semantic_frames(tokens, analyses, roles)
        frames = basic_frames_data['frames'].copy(); sentence = basic_frames_data['sentence']
        # Add Rhetorical, Nested Predication, Coreference logic here...
        # (Code omitted for brevity - assume it's the same as previous version)
        return {'sentence': sentence, 'frames': frames}

    # --- Reporting (No changes needed here) ---
    def format_discourse_relations(self, discourse_relations):
        """ Creates a user-friendly description of discourse relations. """
        if not discourse_relations: return "No specific discourse relations detected."
        formatted_output = []
        descriptions = { # Arabic descriptions
            'CONTINUATION': "تواصل هذه الجملة الفكرة السابقة باستخدام '{}'", 'CAUSE': "تظهر هذه الجملة نتيجة أو عاقبة للجملة السابقة باستخدام '{}'",
            'CONTRAST': "تتناقض هذه الجملة مع المعلومات السابقة باستخدام '{}'", 'ELABORATION': "توضح هذه الجملة المعلومات السابقة باستخدام '{}'",
            'EXAMPLE': "تقدم هذه الجملة مثالاً على المفهوم السابق باستخدام '{}'", 'CONDITION': "تحدد هذه الجملة شرطًا متعلقًا بالجملة السابقة باستخدام '{}'",
            'TEMPORAL': "تحدد هذه الجملة علاقة زمنية مع الجملة السابقة باستخدام '{}'", 'REFERENCE': "تشير هذه الجملة إلى المحتوى السابق باستخدام '{}'"
        }
        for relation in discourse_relations:
            rel_type = relation.get('type', 'UNKNOWN'); marker = relation.get('marker', '')
            desc_template = descriptions.get(rel_type, f"تم اكتشاف علاقة من نوع {rel_type} باستخدام '{marker}'")
            formatted_output.append(desc_template.format(marker))
        return "\n".join(formatted_output) if formatted_output else "لم يتم اكتشاف علاقات خطاب محددة."

    def generate_discourse_report(self, discourse_analyses):
        """Generate a full report of discourse analysis for all sentences in Markdown."""
        report = "# Arabic Text Discourse Analysis (Quantum Enhanced)\n\n"
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis.get('sentence', 'N/A'); interpretation_data = analysis.get('interpretation', {})
            report += f"## Sentence {i+1}\n"; report += f"**Text:** `{sentence}`\n\n"
            if interpretation_data.get('error'):
                 report += f"**ERROR during interpretation:** {interpretation_data['error']}\n\n"; report += "---\n\n"; continue
            # ... (rest of report generation) ...
        return report

    def generate_html_report(self, discourse_analyses):
        """ Generate an HTML report with discourse analysis details. """
        html = """
        <!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><title>تحليل الخطاب الكمي للغة العربية (محسن)</title><style>
        body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:20px;line-height:1.6;background-color:#f9f9f9;color:#333}
        h1{color:#0056b3;text-align:center;border-bottom:2px solid #0056b3;padding-bottom:10px}
        /* ... other styles ... */
        </style></head><body><h1>تحليل الخطاب الكمي للنص العربي (محسن)</h1>"""
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis.get('sentence', 'N/A'); interpretation_data = analysis.get('interpretation', {})
            html += f'<div class="sentence-block"><div class="sentence-text">الجملة {i+1}: {sentence}</div>'
            if interpretation_data.get('error'):
                 html += f'<div class="analysis-section error"><h3>خطأ في التحليل:</h3><div class="analysis-detail">{interpretation_data["error"]}</div></div></div>'; continue
            # ... (rest of HTML generation) ...
        html += """</body></html>"""
        return html

    # --- Utility Methods (No changes needed here) ---
    def save_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Saves the trained kernel state to a file using pickle. """
        print(f"Saving model to {filename}...")
        analyzer_state = None
        if hasattr(self.camel_analyzer, '__getstate__'): analyzer_state = self.camel_analyzer.__getstate__()
        elif self.camel_analyzer is not None: print("Warning: CAMeL Analyzer might not be pickleable.")
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
            print(f"Model saved successfully.")
        except Exception as e: print(f"Error saving model: {e}"); traceback.print_exc()

    def load_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Loads a trained kernel state from a file. Reloads embedding model from path."""
        if not os.path.exists(filename): print(f"Error: Model file {filename} not found."); return self
        print(f"Loading model from {filename}...")
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
            self.__init__(self.embedding_dim, self.num_clusters, embedding_model_path=saved_model_path, params_per_word=self.params_per_word)
            print(f"Model loaded successfully.")
        except Exception as e: print(f"Error loading model: {e}"); traceback.print_exc()
        return self

    def visualize_meaning_space(self, highlight_indices=None, save_path=None):
        """ Visualize the sentence meaning space using PCA. """
        print("Visualizing meaning space...")
        if not self.sentence_embeddings: print("Warning: No embeddings available for visualization."); return None
        try: from sklearn.decomposition import PCA
        except ImportError: print("Error: scikit-learn is required for visualization."); return None
        embeddings = list(self.sentence_embeddings.values()); indices = list(self.sentence_embeddings.keys())
        if not embeddings: print("Warning: Embeddings list is empty."); return None
        X = np.array(embeddings)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
             print("Warning: Non-finite values found in embeddings. Removing problematic rows.")
             X = X[finite_mask]; original_indices = indices; indices = [idx for i, idx in enumerate(original_indices) if finite_mask[i]]
             if self.cluster_labels is not None:
                  if len(self.cluster_labels) == len(finite_mask): self.cluster_labels = self.cluster_labels[finite_mask]; print(f"Filtered cluster labels to size {len(self.cluster_labels)}")
                  else: print("Warning: Cluster labels length mismatch after filtering NaNs."); self.cluster_labels = None
             if X.shape[0] == 0: print("Error: All embeddings contained non-finite values."); return None
             print(f"Removed {len(finite_mask) - X.shape[0]} non-finite rows.")
        if X.shape[1] < 2: print("Warning: Need at least 2 embedding dimensions for PCA."); return None
        if X.shape[0] < 2: print("Warning: Need at least 2 samples for PCA."); return None
        try:
            pca = PCA(n_components=2); reduced_embeddings = pca.fit_transform(X)
            plt.figure(figsize=(12, 10)); current_labels = None
            if self.cluster_labels is not None and len(self.cluster_labels) == len(reduced_embeddings): current_labels = np.array(self.cluster_labels)
            colors = current_labels if current_labels is not None else 'blue'; cmap = 'viridis' if current_labels is not None else None
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, cmap=cmap, alpha=0.7, s=100)
            if highlight_indices is not None:
                 highlight_idxs_in_filtered = [indices.index(i) for i in highlight_indices if i in indices]
                 if highlight_idxs_in_filtered: plt.scatter(reduced_embeddings[highlight_idxs_in_filtered, 0], reduced_embeddings[highlight_idxs_in_filtered, 1], c='red', s=150, edgecolor='white', zorder=10, label='Highlighted')
            if current_labels is not None and isinstance(colors, np.ndarray):
                try:
                     unique_labels = np.unique(current_labels)
                     if len(unique_labels) > 1: legend1 = plt.legend(*scatter.legend_elements(), title="Meaning Clusters"); plt.gca().add_artist(legend1)
                     elif len(unique_labels) == 1: print("Only one cluster found, skipping cluster legend.")
                except Exception as leg_e: print(f"Warning: Could not create cluster legend: {leg_e}")
            if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
                 if np.all(np.isfinite(self.meaning_clusters)):
                     try:
                         if hasattr(pca, 'components_'):
                              cluster_centers_2d = pca.transform(self.meaning_clusters)
                              plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker='*', s=350, c='white', edgecolor='black', label='Cluster Centers', zorder=15)
                              for i, (x, y) in enumerate(cluster_centers_2d):
                                  if i in self.meaning_map:
                                      meaning = self.meaning_map[i].get('structure', f'Cluster {i}'); template = self.meaning_map[i].get('deduced_template', '')[:30]
                                      plt.annotate(f"Cluster {i}: {meaning}\n'{template}...'", (x, y), xytext=(0, 15), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8), zorder=20)
                         else: print("Warning: PCA not fitted, cannot transform cluster centers.")
                     except Exception as cc_e: print(f"Error plotting cluster centers: {cc_e}")
                 else: print("Warning: Non-finite values found in cluster centers. Skipping plotting centers.")
            plt.title('Quantum Sentence Meaning Space (PCA)'); plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)'); plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=150); print(f"Visualization saved to {save_path}")
            return plt.gcf()
        except Exception as e: print(f"Error during PCA visualization: {e}"); traceback.print_exc(); return None

    def analyze_quantum_states(self, circuits_dict, save_path_prefix=None):
        """ Analyze and visualize the quantum states of a dictionary of circuits. """
        if not circuits_dict: print("No circuits provided for state analysis."); return
        print(f"Analyzing quantum states for {len(circuits_dict)} circuits...")
        num_circuits = len(circuits_dict); ncols = 3; nrows = (num_circuits + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False); axes = axes.flatten()
        i = 0
        for circuit_id, circuit in circuits_dict.items():
            if i >= len(axes): break
            ax = axes[i]; sentence_label = self.reference_sentences.get(circuit_id, f"Circuit {circuit_id}")
            if len(sentence_label) > 40: sentence_label = sentence_label[:37] + "..."
            try:
                if not isinstance(circuit, QuantumCircuit):
                    print(f"Skipping item {circuit_id}: Not a Qiskit QuantumCircuit (type: {type(circuit)})")
                    ax.text(0.5, 0.5, f"Invalid Circuit Object\n(Type: {type(circuit).__name__})\n(ID: {circuit_id})", ha='center', va='center', fontsize=9, color='red')
                    ax.set_title(f"Error: {sentence_label}", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); i += 1; continue
                result = execute(circuit, self.simulator).result()
                if hasattr(result, 'get_statevector'):
                    statevector = result.get_statevector(); plot_state_city(statevector, title=sentence_label, ax=ax)
                    ax.tick_params(axis='both', which='major', labelsize=8); ax.title.set_size(10)
                    if save_path_prefix:
                         try:
                              individual_fig = plt.figure(figsize=(8, 6)); plot_state_city(statevector, title=f"State: {sentence_label}")
                              individual_path = f"{save_path_prefix}circuit_{circuit_id}.png"; individual_fig.savefig(individual_path, dpi=150, bbox_inches='tight'); plt.close(individual_fig)
                         except Exception as save_e: print(f"Warning: Failed to save individual state plot for {circuit_id}: {save_e}")
                else:
                    ax.text(0.5, 0.5, "Statevector not available", ha='center', va='center', fontsize=9)
                    ax.set_title(sentence_label, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            except Exception as e:
                print(f"Error visualizing circuit {circuit_id}: {e}")
                ax.text(0.5, 0.5, f"Error visualizing:\n{e}", ha='center', va='center', fontsize=9, color='red')
                ax.set_title(f"Error: {sentence_label}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            i += 1
        for j in range(i, len(axes)): fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.suptitle("Quantum State Visualizations", fontsize=16)
        if save_path_prefix:
            try: overview_path = f"{save_path_prefix}all_states_overview.png"; fig.savefig(overview_path, dpi=150); print(f"Saved quantum states overview to {overview_path}")
            except Exception as overview_save_e: print(f"Warning: Failed to save overview state plot: {overview_save_e}")
        return fig


# ============================================
# Example Pipeline Function (Modified Calls)
# ============================================
def prepare_quantum_nlp_pipeline(max_sentences=20, use_enhanced_clustering=True, embedding_model_path=None):
    """
    Example of how to use the enhanced ArabicQuantumMeaningKernel with embedding parameters.
    """
    if not LAMBEQ_AVAILABLE:
        print("Exiting pipeline because Lambeq library is required but not found.")
        return None, None, []

    sentence_file = "sentences.txt"
    sentences = []
    # ... (sentence loading logic) ...
    try:
        with open(sentence_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()][:max_sentences]
        print(f"Loaded {len(sentences)} sentences from {sentence_file}")
    except FileNotFoundError:
        print(f"Error: Sentence file '{sentence_file}' not found. Using default example sentences.")
        sentences = [
            "يقرأ الولد الكتاب", "الولد يقرأ الكتاب", "كتب الطالب الدرس",
            "الطالب يكتب الدرس", "البيت كبير", "درسَ خالدٌ كثيرًا لأنه يريد النجاحَ"
        ][:max_sentences]


    results = []
    circuits_for_viz = {}
    tokens_for_viz = {}   # Store tokens for state viz parameter binding
    analyses_for_viz = {} # Store analyses for state viz parameter binding

    print("\n--- Processing Sentences ---")
    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx+1}/{len(sentences)}: '{sentence[:40]}...'")
        try:
            # Call the function from camel_test.py
            returned_values = arabic_to_quantum_enhanced(sentence, debug=False)

            circuit = None
            grammar_diagram = None
            structure = "ERROR"
            tokens = []
            analyses = []
            roles = {}

            if isinstance(returned_values, tuple) and len(returned_values) >= 6:
                circuit, grammar_diagram, structure, tokens, analyses, roles = returned_values
            else:
                print(f"  ERROR: Unexpected return value or length from arabic_to_quantum_enhanced: {returned_values}")
                continue # Skip this sentence

            if circuit is None or not isinstance(circuit, QuantumCircuit):
                print(f"  INFO: Skipping sentence {idx+1} due to missing or invalid QuantumCircuit.")
                continue # Skip to the next sentence

            # Store results
            results.append({
                'sentence': sentence, 'circuit': circuit, 'structure': structure,
                'tokens': tokens, 'analyses': analyses, 'roles': roles,
                'original_index': idx, 'diagram': grammar_diagram
            })
            circuits_for_viz[idx] = circuit
            tokens_for_viz[idx] = tokens     # Store tokens
            analyses_for_viz[idx] = analyses # Store analyses
            print(f"  Sentence {idx+1}: Processed successfully. Structure: {structure}, Circuit type: {type(circuit)}")

        except Exception as e:
            print(f"  ERROR processing sentence {idx+1}: {sentence}")
            print(f"  Actual Error Type: {type(e).__name__}, Message: {e}")
            traceback.print_exc()
            print("  Skipping this sentence.")

    if not results:
        print("\nError: No sentences were processed successfully. Exiting pipeline.")
        return None, None, []

    # --- Kernel Instantiation: Pass embedding model path ---
    num_processed = len(results)
    kernel_clusters = min(5, max(1, num_processed // 2))
    kernel = ArabicQuantumMeaningKernel(
        embedding_dim=14,
        num_clusters=kernel_clusters,
        embedding_model_path=embedding_model_path # Pass the path here
    )
    # ---

    print(f"\n--- Training Kernel ({kernel.num_clusters} clusters) ---")
    # Train method now passes tokens/analyses needed for parameter binding
    kernel.train(
        sentences=[r['sentence'] for r in results],
        circuits=[r['circuit'] for r in results],
        tokens_list=[r['tokens'] for r in results],     # Pass tokens
        analyses_list=[r['analyses'] for r in results], # Pass analyses
        structures=[r['structure'] for r in results],
        roles_list=[r['roles'] for r in results],
        use_enhanced_clustering=use_enhanced_clustering
    )

    # Save Model, Visualizations, Discourse Analysis, Reports
    kernel.save_model('arabic_quantum_kernel_embed_params.pkl') # New name

    print("\n--- Visualizing Meaning Space ---")
    try:
        meaning_space_fig = kernel.visualize_meaning_space(save_path='meaning_space_embed_params.png')
        if meaning_space_fig: plt.close(meaning_space_fig)
    except Exception as vis_e: print(f"Error during meaning space visualization: {vis_e}")

    print("\n--- Visualizing Quantum States ---")
    try:
        # Pass tokens and analyses needed for parameter binding in visualization
        state_viz_fig = kernel.analyze_quantum_states(
            circuits_for_viz,
            tokens_for_viz,
            analyses_for_viz,
            save_path_prefix="state_viz_embed_"
        )
        if state_viz_fig: plt.close(state_viz_fig)
    except Exception as state_vis_e: print(f"Error during quantum state visualization: {state_vis_e}")

    print("\n--- Performing Discourse Analysis ---")
    discourse_analyses = []
    previous_analysis_dict = None
    for i, result in enumerate(results):
        print(f"Analyzing sentence {i+1}/{len(results)} in context...")
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
            print(f"  Sentence {i+1}: Analysis complete.")
        except Exception as analysis_e:
            print(f"  Error during sentence analysis {i+1}: {analysis_e}")
            traceback.print_exc()
            discourse_analyses.append({**result, 'interpretation': {'error': str(analysis_e)}})

    print("\n--- Generating Reports ---")
    try:
        html_report = kernel.generate_html_report(discourse_analyses)
        report_filename = 'discourse_analysis_report_embed_params.html'
        with open(report_filename, 'w', encoding='utf-8') as f: f.write(html_report)
        print(f"HTML report saved to {report_filename}")
    except Exception as report_e: print(f"Error generating HTML report: {report_e}")

    try:
        md_report = kernel.generate_discourse_report(discourse_analyses)
        md_report_filename = 'discourse_analysis_report_embed_params.md'
        with open(md_report_filename, 'w', encoding='utf-8') as f: f.write(md_report)
        print(f"Markdown report saved to {md_report_filename}")
    except Exception as md_report_e: print(f"Error generating Markdown report: {md_report_e}")

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
                  print("ERROR: Circuit generation failed for new sentence.")
                  new_circuit_obj = None # Ensure it's None if invalid
        else: print("ERROR: Unexpected return from arabic_to_quantum_enhanced for new sentence.")

        if new_circuit_obj:
             # Call interpret_sentence (it handles parameter binding internally)
             last_interpretation_result = kernel.interpret_sentence(
                 new_circuit_obj, new_tokens, new_analyses, new_structure, new_roles,
                 previous_analyses=previous_analysis_dict
             )
             print("New sentence interpretation complete.")
        else: print("ERROR: Failed to get valid QuantumCircuit for new sentence.")
    except Exception as interp_e:
        print(f"Error interpreting new sentence: {interp_e}")
        traceback.print_exc()

    print("\n--- Pipeline Finished ---")
    return kernel, last_interpretation_result, discourse_analyses



# ============================================
# Main execution block
# ============================================
if __name__ == "__main__":
    # --- IMPORTANT: Specify the path to your Word2Vec model file ---
    # Example: If you downloaded an AraVec model (e.g., Twitter cbow 300 dim)
    # and extracted it to a folder named 'aravec' in the same directory as v4.py
    # model_path = "aravec/full_grams_cbow_300_twitter.mdl" # Adjust this path!
    # Or if you have a text format model:
    model_path = "../aravec/tweet_cbow_300" # Example path - SET YOUR ACTUAL PATH
    # If you don't have a model or don't want to use it, set model_path = None
    # model_path = None
    # ----------------------------------------------------------------

    # Run the pipeline
    kernel, last_new_sentence_interpretation, discourse_analyses_list = prepare_quantum_nlp_pipeline(
        max_sentences=20,
        use_enhanced_clustering=True,
        embedding_model_path=model_path # Pass the model path to the pipeline
    )

    # Check if pipeline ran successfully
    if kernel is None:
        print("\nPipeline execution failed.")
    else:
        print("\n--- Summary of Last New Sentence Interpretation ---")
        if last_new_sentence_interpretation is not None:
            if last_new_sentence_interpretation.get('error'):
                 print(f"ERROR interpreting new sentence: {last_new_sentence_interpretation['error']}")
            else:
                print(f"Sentence: {last_new_sentence_interpretation.get('sentence', 'N/A')}")
                print(f"Structure: {last_new_sentence_interpretation.get('structure', 'N/A')}")
                print(f"Confidence: {last_new_sentence_interpretation.get('confidence', 'N/A')}")
                print(f"Deduced Meaning: {last_new_sentence_interpretation.get('interpretation', 'N/A')}")
                # ... (print semantic frames, discourse relations etc. as before) ...
        else:
            print("No interpretation available for the new sentence (it might have failed).")
