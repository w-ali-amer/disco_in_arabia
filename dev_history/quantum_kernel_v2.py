import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from qiskit import Aer, execute, QuantumCircuit # Ensure QuantumCircuit is imported
from qiskit.visualization import plot_state_city
# Remove lambeq imports if no longer needed directly in kernel
# from lambeq import AtomicType, IQPAnsatz
import pickle
import os
import traceback
from collections import Counter
from camel_test import arabic_to_quantum_enhanced
from qiskit.quantum_info import partial_trace # For reduced density matrix

# Helper function to ensure circuit name (applied to Qiskit circuits)
def _ensure_circuit_name(circuit: Any, default_name: str = "qiskit_circuit") -> Any:
    """Checks if a Qiskit circuit object has a name and assigns one if not."""
    if isinstance(circuit, QuantumCircuit):
        has_name_attr = hasattr(circuit, 'name')
        name_is_set = has_name_attr and getattr(circuit, 'name', None)
        if not name_is_set:
            try:
                setattr(circuit, 'name', default_name)
                print(f"Debug: Assigned default name '{default_name}' to Qiskit circuit.") # DEBUG
            except Exception as e:
                print(f"Debug: Error assigning name to Qiskit circuit: {e}.") # DEBUG
    return circuit

class ArabicQuantumMeaningKernel:
    """
    A quantum kernel for mapping quantum circuit outputs to potential
    sentence meanings for Arabic language processing. Includes discourse analysis.
    (Full Implementation)
    """
    def __init__(self,
                 embedding_dim: int = 14,
                 num_clusters: int = 5,
                 simulator_backend: str = 'statevector_simulator'):
        """ Initialize the quantum meaning kernel. """
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.simulator = Aer.get_backend(simulator_backend)
        self.meaning_clusters = None
        self.cluster_labels = None
        self.meaning_map = {}
        self.reference_sentences = {}
        self.circuit_embeddings = {} # Stores features derived from circuits
        self.sentence_embeddings = {} # Stores combined quantum+linguistic embeddings
        self.camel_analyzer = None # Will be initialized if CAMeL Tools are found
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
            # Ensure DB is available (e.g., 'camel_tools download default_light')
            db = MorphologyDB.builtin_db('calima-msa-r13') # Example DB, adjust if needed
            self.camel_analyzer = Analyzer(db)
            print("CAMeL Tools Analyzer initialized successfully.")
        except ImportError:
            print("Warning: CAMeL Tools not found (pip install camel-tools). NLP enhancements disabled.")
        except LookupError:
             print("Warning: CAMeL Tools default DB not found. Run 'camel_tools download default_light' or specify DB path. NLP enhancements disabled.")
        except Exception as e:
            print(f"Warning: Error initializing CAMeL Tools Analyzer: {e}. NLP enhancements disabled.")


    # --- Feature Extraction and Embedding ---

    def get_circuit_features(self, circuit, shots: int = 1024) -> np.ndarray:
        """ Extracts features from a Qiskit quantum circuit by executing it. """
        print(f"\n--- Debugging get_circuit_features ---") # DEBUG
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_circuit_features received non-Qiskit object: {type(circuit)}")
             fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback
        print(f"Input type: {type(circuit)}") # DEBUG
        try:
            circuit = _ensure_circuit_name(circuit, "circuit_for_features")
            print(f"Qiskit circuit name: {getattr(circuit, 'name', 'N/A')}") # DEBUG
            print("Executing Qiskit circuit...") # DEBUG
            result = execute(circuit, self.simulator).result()
            print(f"Execution successful.") # DEBUG
            if hasattr(result, 'get_statevector'):
                print("Statevector found.") # DEBUG
                statevector = result.get_statevector()
                amplitudes = np.abs(statevector); phases = np.angle(statevector)
                half_dim = self.embedding_dim // 2
                amp_slice = amplitudes[:min(half_dim, len(amplitudes))]
                phase_slice = phases[:min(half_dim, len(phases))]
                features = np.concatenate([amp_slice, phase_slice])
                if len(features) < self.embedding_dim: features = np.pad(features, (0, self.embedding_dim - len(features)), 'constant')
                elif len(features) > self.embedding_dim: features = features[:self.embedding_dim]
                norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
                print(f"Returning statevector-based features. Shape: {features.shape}") # DEBUG
                return features
            else:
                 print("Statevector not found. Falling back to counts.") # DEBUG
                 counts = result.get_counts(); total_shots = sum(counts.values())
                 feature_vector = np.zeros(self.embedding_dim); num_outcomes = 0
                 if total_shots > 0:
                     for outcome, count in counts.items():
                         num_outcomes += 1
                         try:
                             idx_base = int(outcome, 2) if isinstance(outcome, str) and all(c in '01' for c in outcome) else hash(outcome)
                             idx = idx_base % self.embedding_dim; feature_vector[idx] += count / total_shots
                         except Exception as e_idx: print(f"  Warning: Could not process outcome '{outcome}': {e_idx}") # DEBUG
                 print(f"Processed {num_outcomes} outcomes.") # DEBUG
                 norm = np.linalg.norm(feature_vector); feature_vector = feature_vector / norm if norm > 0 else feature_vector
                 print(f"Returning counts-based features. Shape: {feature_vector.shape}") # DEBUG
                 return feature_vector
        except Exception as e:
            print(f"\n--- ERROR in get_circuit_features ---"); print(f"Error type: {type(e)}"); print(f"Error message: {e}"); traceback.print_exc(); print(f"---------------------------------------\n") # DEBUG
        print("Returning fallback random features due to error.") # DEBUG
        fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback

    def _get_reduced_density_matrix(self, statevector, subsystem_qubits):
        """ Calculates reduced density matrix using Qiskit's partial_trace. """
        try:
            n_qubits = int(np.log2(len(statevector)))
            if not subsystem_qubits or max(subsystem_qubits) >= n_qubits or min(subsystem_qubits) < 0:
                 raise ValueError("Invalid subsystem qubits specified.")
            rho = np.outer(statevector, np.conj(statevector))
            trace_out_qubits = [i for i in range(n_qubits) if i not in subsystem_qubits]
            reduced_rho_qiskit = partial_trace(rho, trace_out_qubits).data
            return reduced_rho_qiskit
        except ValueError as ve:
             print(f"Error in _get_reduced_density_matrix (ValueError): {ve}")
        except Exception as e:
            print(f"Error using qiskit.partial_trace: {e}. Falling back to maximally mixed state.")
        # Fallback: Return maximally mixed state for the subsystem size
        reduced_dims = 2 ** len(subsystem_qubits)
        return np.eye(reduced_dims, dtype=complex) / reduced_dims

    def get_enhanced_circuit_features(self, circuit, shots: int = 1024) -> np.ndarray:
        """ Enhanced feature extraction from a Qiskit circuit. """
        print(f"\n--- Debugging get_enhanced_circuit_features ---") # DEBUG
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_enhanced_circuit_features received non-Qiskit object: {type(circuit)}")
             fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback
        print(f"Input type: {type(circuit)}") # DEBUG
        try:
            print("Calling get_circuit_features for basic features...") # DEBUG
            basic_features = self.get_circuit_features(circuit, shots)
            print(f"Basic features obtained. Shape: {basic_features.shape}") # DEBUG
            print("Ensuring Qiskit circuit name for enhanced analysis...") # DEBUG
            circuit = _ensure_circuit_name(circuit, "circuit_for_enhanced")
            print(f"Qiskit circuit name: {getattr(circuit, 'name', 'N/A')}") # DEBUG
            print("Executing Qiskit circuit for enhanced analysis...") # DEBUG
            result = execute(circuit, self.simulator).result()
            print(f"Execution successful.") # DEBUG
            entanglement_features = []; pauli_expectations = []
            print("Checking for statevector for enhanced analysis...") # DEBUG
            if hasattr(result, 'get_statevector'):
                print("Statevector found.") # DEBUG
                statevector = result.get_statevector(); num_qubits = circuit.num_qubits
                print(f"Statevector obtained. Length: {len(statevector)}, Num Qubits: {num_qubits}") # DEBUG
                print("Calculating entanglement features...") # DEBUG
                if hasattr(self, '_get_reduced_density_matrix') and num_qubits > 1:
                    num_entanglement_checks = min(num_qubits -1, 3)
                    for i in range(num_entanglement_checks):
                        try:
                            reduced_density = self._get_reduced_density_matrix(statevector, [i])
                            eigenvalues = np.linalg.eigvalsh(reduced_density)
                            valid_eigenvalues = eigenvalues[eigenvalues > 1e-12]
                            entropy = -np.sum(valid_eigenvalues * np.log2(valid_eigenvalues)) if len(valid_eigenvalues) > 0 else 0.0
                            entanglement_features.append(np.real(entropy))
                            print(f"  Von Neumann Entropy for qubit {i}: {np.real(entropy):.4f}") # DEBUG
                        except Exception as inner_e: print(f"  Error calculating entanglement for qubit {i}: {type(inner_e).__name__} - {inner_e}"); entanglement_features.append(0.0) # DEBUG
                else: print(f"  Skipping entanglement calculation (num_qubits={num_qubits}).") #DEBUG
                print("Calculating Pauli X expectations...") # DEBUG
                if isinstance(circuit, QuantumCircuit) and hasattr(circuit, 'copy'):
                    try:
                        num_pauli_checks = min(num_qubits, 3)
                        for i in range(num_pauli_checks):
                            meas_circuit_name = f"{getattr(circuit, 'name', 'unnamed')}_pauli_x_q{i}"
                            meas_circuit = circuit.copy(name=meas_circuit_name); meas_circuit.h(i)
                            cr = None
                            if not meas_circuit.cregs:
                                cr = ClassicalRegister(num_qubits, name=f'c_pauli_{i}')
                                meas_circuit.add_register(cr)
                            elif meas_circuit.cregs[0].size < num_qubits:
                                # If existing register is too small, add a new one (might cause issues if names conflict)
                                print(f"Warning: Existing classical register too small ({meas_circuit.cregs[0].size} < {num_qubits}). Adding new one.")
                                cr = ClassicalRegister(num_qubits, name=f'c_pauli_{i}')
                                meas_circuit.add_register(cr)
                            else: cr = meas_circuit.cregs[0] # Use the first one if sufficient size
                            if i < cr.size: meas_circuit.measure(i, i)
                            else: print(f"Warning: Classical register '{cr.name}' too small for qubit {i}. Skipping measure."); pauli_expectations.append(0.0); continue
                            meas_result = execute(meas_circuit, self.simulator, shots=shots).result(); counts = meas_result.get_counts()
                            exp_val = 0; total = sum(counts.values())
                            if total > 0:
                                 prob_0, prob_1 = 0, 0
                                 for bitstring, count in counts.items():
                                      bit_index = len(bitstring) - 1 - i
                                      if 0 <= bit_index < len(bitstring):
                                          if int(bitstring[bit_index]) == 0: prob_0 += count / total
                                          else: prob_1 += count / total
                                 exp_val = prob_0 - prob_1
                            pauli_expectations.append(exp_val); print(f"  Pauli X expectation for qubit {i}: {exp_val:.4f}") # DEBUG
                    except Exception as pauli_e: print(f"  Error in Pauli expectation calculation: {type(pauli_e).__name__} - {pauli_e}"); traceback.print_exc(); num_pauli_checks = min(num_qubits, 3); pauli_expectations.extend([0.0] * (num_pauli_checks - len(pauli_expectations))) # DEBUG
                else: print("  Skipping Pauli expectation (input not a Qiskit circuit or no .copy).") # DEBUG
            else: print("Statevector not found. Skipping enhanced feature calculation.") # DEBUG
            print("Combining basic, entanglement, and Pauli features...") # DEBUG
            feature_list = [basic_features]
            if entanglement_features: feature_list.append(np.array(entanglement_features))
            if pauli_expectations: feature_list.append(np.array(pauli_expectations))
            all_features = np.concatenate(feature_list) if len(feature_list) > 1 else basic_features
            print(f"Combined features shape before padding/truncation: {all_features.shape}") # DEBUG
            current_len = len(all_features)
            if current_len < self.embedding_dim: all_features = np.pad(all_features, (0, self.embedding_dim - current_len), 'constant')
            elif current_len > self.embedding_dim: all_features = all_features[:self.embedding_dim]
            norm = np.linalg.norm(all_features); all_features = all_features / norm if norm > 0 else all_features
            print(f"Returning enhanced features. Final shape: {all_features.shape}") # DEBUG
            return all_features
        except Exception as e:
            print(f"\n--- ERROR in get_enhanced_circuit_features ---"); print(f"Error type: {type(e)}"); print(f"Error message: {e}"); traceback.print_exc(); print(f"----------------------------------------------\n") # DEBUG
        print("Falling back to basic features due to error in enhanced calculation.") # DEBUG
        if 'basic_features' in locals() and isinstance(basic_features, np.ndarray):
             current_len = len(basic_features)
             if current_len < self.embedding_dim: basic_features = np.pad(basic_features, (0, self.embedding_dim - current_len), 'constant')
             elif current_len > self.embedding_dim: basic_features = basic_features[:self.embedding_dim]
             norm = np.linalg.norm(basic_features); basic_features = basic_features / norm if norm > 0 else basic_features
             return basic_features
        else: fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback

    def extract_linguistic_features(self, tokens: List[str], analyses: List[Tuple], structure: str, roles: Dict) -> np.ndarray:
        """ Extracts basic linguistic features. """
        print("  Extracting basic linguistic features...") # DEBUG
        features = np.zeros(self.embedding_dim)
        num_features = 0
        # Feature 0: Structure type
        structure_map = {'VSO': 0, 'SVO': 1, 'NOMINAL': 2, 'COMPLEX': 3, 'OTHER': 4}
        structure_idx = structure_map.get(structure, 4)
        if num_features < self.embedding_dim: features[num_features] = structure_idx / (len(structure_map) -1); num_features += 1
        # Features 1-3: POS counts (normalized)
        pos_counts = Counter(pos for _, pos, _, _ in analyses)
        total_tokens = max(1, len(tokens))
        if num_features < self.embedding_dim: features[num_features] = pos_counts.get('VERB', 0) / total_tokens; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = pos_counts.get('NOUN', 0) / total_tokens; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = pos_counts.get('ADJ', 0) / total_tokens; num_features += 1
        # Features 4-6: Role presence
        if num_features < self.embedding_dim: features[num_features] = 1.0 if roles.get('verb') is not None else 0.0; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = 1.0 if roles.get('subject') is not None else 0.0; num_features += 1
        if num_features < self.embedding_dim: features[num_features] = 1.0 if roles.get('object') is not None else 0.0; num_features += 1
        # Feature 7: Negation presence
        has_negation = any(lemma in ['لا', 'ليس', 'غير', 'لم', 'لن'] for lemma, _, _, _ in analyses)
        if num_features < self.embedding_dim: features[num_features] = 1.0 if has_negation else 0.0; num_features += 1
        # Normalize final vector
        norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
        return features

    def extract_complex_linguistic_features(self, tokens, analyses, structure, roles):
        """ Enhanced linguistic feature extraction including complexity metrics. """
        print("  Extracting complex linguistic features...") # DEBUG
        # Start with basic features
        features = self.extract_linguistic_features(tokens, analyses, structure, roles)
        # Find the index of the next available feature slot (first zero after normalization might be tricky)
        # Let's restart count assuming basic features used ~8 slots
        feature_idx = 8 # Start adding complex features from index 8 (adjust if basic uses more/less)

        # Feature 8: Subordinate clause markers
        if feature_idx < self.embedding_dim:
            subordinate_markers = ['الذي', 'التي', 'الذين', 'اللواتي', 'عندما', 'حيث', 'لأن', 'كي', 'أنّ']
            has_subordinate = any(token in subordinate_markers for token in tokens)
            features[feature_idx] = 1.0 if has_subordinate else 0.0; feature_idx += 1
        # Feature 9: Verb count (clause count proxy)
        if feature_idx < self.embedding_dim:
            verb_count = sum(1 for _, pos, _, _ in analyses if pos == 'VERB')
            features[feature_idx] = min(verb_count / 3.0, 1.0); feature_idx += 1 # Normalize, cap at 3
        # Feature 10: Conditional markers
        if feature_idx < self.embedding_dim:
            conditional_markers = ['إذا', 'لو', 'إن']
            has_conditional = any(token in conditional_markers for token in tokens)
            features[feature_idx] = 1.0 if has_conditional else 0.0; feature_idx += 1
        # Feature 11: Quotation markers
        if feature_idx < self.embedding_dim:
            quotation_markers = ['قال', 'صرح', 'أعلن', 'ذكر', 'أضاف']
            has_quotation = any(lemma in quotation_markers for lemma, _, _, _ in analyses)
            features[feature_idx] = 1.0 if has_quotation else 0.0; feature_idx += 1
        # Feature 12: Average word length
        if feature_idx < self.embedding_dim:
            avg_word_length = np.mean([len(token) for token in tokens]) if tokens else 0
            features[feature_idx] = min(avg_word_length / 10.0, 1.0); feature_idx += 1 # Normalize
        # Feature 13: Sentence length (token count)
        if feature_idx < self.embedding_dim:
             features[feature_idx] = min(len(tokens) / 50.0, 1.0); feature_idx += 1 # Normalize based on max 50 tokens

        # Re-normalize the final vector including complex features
        norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
        return features

    def combine_features_with_attention(self, quantum_features, linguistic_features, structure):
        """ Combine features with attention mechanism based on sentence structure. """
        print("  Combining features with attention...") # DEBUG
        if structure == 'NOMINAL': quantum_weight, linguistic_weight = 0.3, 0.7
        elif structure in ['VSO', 'SVO']: quantum_weight, linguistic_weight = 0.5, 0.5
        elif 'COMPLEX' in structure or structure == 'COMPLEX': quantum_weight, linguistic_weight = 0.6, 0.4
        else: quantum_weight, linguistic_weight = 0.5, 0.5 # Default equal weighting
        # Ensure consistent dimensions
        q_len = len(quantum_features); l_len = len(linguistic_features)
        if q_len < self.embedding_dim: quantum_features = np.pad(quantum_features, (0, self.embedding_dim - q_len), 'constant')
        elif q_len > self.embedding_dim: quantum_features = quantum_features[:self.embedding_dim]
        if l_len < self.embedding_dim: linguistic_features = np.pad(linguistic_features, (0, self.embedding_dim - l_len), 'constant')
        elif l_len > self.embedding_dim: linguistic_features = linguistic_features[:self.embedding_dim]
        # Combine with weighted attention
        combined = quantum_weight * quantum_features + linguistic_weight * linguistic_features
        # Normalize
        norm = np.linalg.norm(combined); combined = combined / norm if norm > 0 else combined
        return combined

    # --- Training and Clustering ---

    def train(self, sentences, circuits, tokens_list, analyses_list, structures, roles_list):
        """ Train the kernel on a set of sentences and their Qiskit circuits. """
        self.reference_sentences = {i: sentences[i] for i in range(len(sentences))}
        self.circuit_embeddings = {}; self.sentence_embeddings = {}; embeddings = []
        print(f"\n--- Training Kernel on {len(sentences)} sentences ---") # DEBUG
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
            print(f"Processing sentence {i+1}/{min_len}: '{sentences[i][:30]}...'") # DEBUG
            try:
                quantum_features = self.get_enhanced_circuit_features(circuits[i])
                self.circuit_embeddings[i] = quantum_features
                linguistic_features = self.extract_complex_linguistic_features(tokens_list[i], analyses_list[i], structures[i], roles_list[i])
                embedding = self.combine_features_with_attention(quantum_features, linguistic_features, structures[i])
                self.sentence_embeddings[i] = embedding; embeddings.append(embedding)
                print(f"  Sentence {i+1}: Embedding generated. Shape: {embedding.shape}") # DEBUG
            except Exception as e: print(f"  ERROR processing sentence {i+1} during training: {sentences[i]}"); print(f"  Error type: {type(e).__name__}, Message: {e}"); traceback.print_exc(); print("  Skipping embedding for this sentence.") # DEBUG
        if not embeddings: print("Error: No embeddings generated. Cannot proceed."); return self
        print("Learning meaning clusters...") # DEBUG
        self.learn_meaning_clusters(embeddings)
        print("Assigning meaning to clusters...") # DEBUG
        self.assign_meaning_to_clusters(sentences, structures, roles_list, analyses_list)
        print("--- Training Complete ---") # DEBUG
        return self

    def learn_meaning_clusters(self, embeddings: List[np.ndarray]) -> None:
        """ Learn meaning clusters from embeddings using KMeans. """
        if not embeddings: print("Warning: No embeddings provided for clustering."); return
        X = np.array(embeddings); n_samples = X.shape[0]
        if n_samples == 0: print("Warning: Embedding array is empty. Cannot cluster."); return
        # Adjust num_clusters if fewer samples than requested clusters
        n_clusters = min(self.num_clusters, n_samples)
        if n_clusters <= 0: n_clusters = 1 # Ensure at least one cluster
        self.num_clusters = n_clusters # Update actual number of clusters used
        print(f"Attempting KMeans with {self.num_clusters} clusters on {n_samples} samples.") # DEBUG
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10) # Set n_init explicitly
            self.cluster_labels = kmeans.fit_predict(X)
            self.meaning_clusters = kmeans.cluster_centers_
            print(f"KMeans clustering complete. Found {len(self.meaning_clusters)} cluster centers.") # DEBUG
        except Exception as e:
            print(f"Error during KMeans clustering: {e}"); traceback.print_exc() # DEBUG
            self.cluster_labels = None; self.meaning_clusters = None

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
        return "ACTION" # Default

    def assign_meaning_to_clusters(self, sentences: List[str], structures: List[str], roles_list: List[Dict], analyses_list: List[List[Tuple]]) -> Dict:
        """ Assign meaning templates to clusters based on linguistic analysis. """
        self.meaning_map = {}
        if self.cluster_labels is None or len(self.cluster_labels) == 0: print("Warning: Cluster labels not found. Cannot assign meanings."); return self.meaning_map
        sentiment_analyzer = None # Optional: Initialize CAMeL Sentiment Analyzer if needed/available
        # --- Group sentences and analyses by cluster ---
        cluster_data_grouped = {label: [] for label in set(self.cluster_labels)}
        for i, label in enumerate(self.cluster_labels):
            if i < len(sentences) and i < len(structures) and i < len(roles_list) and i < len(analyses_list):
                 cluster_data_grouped[label].append({'sentence': sentences[i], 'structure': structures[i], 'roles': roles_list[i], 'analyses': analyses_list[i], 'index': i})
            else: print(f"Warning: Skipping sentence index {i} due to missing data for cluster {label}.")
        # --- Assign meaning to each cluster ---
        for cluster_id, cluster_data in cluster_data_grouped.items():
            if not cluster_data: print(f"Skipping empty cluster {cluster_id}"); continue
            print(f"Assigning meaning to cluster {cluster_id} ({len(cluster_data)} sentences)") # DEBUG
            verb_lemmas = Counter(); subject_lemmas = Counter(); object_lemmas = Counter(); common_preps = Counter(); verb_tenses = Counter(); verb_moods = Counter(); sentiments = []; structure_counts = Counter()
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
                if self.camel_analyzer: # Use CAMeL if available
                    try:
                        morph_analysis_list = self.camel_analyzer.analyze(item['sentence'])
                        if verb_idx is not None and verb_idx < len(morph_analysis_list):
                            verb_morph_analyses = morph_analysis_list[verb_idx]
                            if verb_morph_analyses: verb_morph = verb_morph_analyses[0]; verb_tenses[verb_morph.get('asp', 'UNK')] += 1; verb_moods[verb_morph.get('mod', 'UNK')] += 1
                    except Exception as e: print(f"CAMeL analysis failed for cluster {cluster_id}, sentence '{item['sentence'][:30]}...': {e}") # DEBUG
                # Add sentiment analysis call here if sentiment_analyzer is initialized
            # --- Determine dominant features ---
            dominant_structure = structure_counts.most_common(1)[0][0] if structure_counts else 'OTHER'
            dominant_verb = verb_lemmas.most_common(1)[0][0] if verb_lemmas else None
            dominant_subj = subject_lemmas.most_common(1)[0][0] if subject_lemmas else "SUBJECT"
            dominant_obj = object_lemmas.most_common(1)[0][0] if object_lemmas else "OBJECT"
            top_prep = common_preps.most_common(1)[0][0] if common_preps else None
            dominant_tense = verb_tenses.most_common(1)[0][0] if verb_tenses else None
            dominant_mood = verb_moods.most_common(1)[0][0] if verb_moods else None
            # --- Deduce Semantic Template ---
            deduced_template = f"{dominant_subj} (did something involving) {dominant_obj}"; verb_class = self._classify_verb(dominant_verb)
            if verb_class == "MOTION": dest = top_prep[1] if top_prep and top_prep[0] in ['إلى', 'ل'] else "DESTINATION"; deduced_template = f"{dominant_subj} went to {dest}"
            elif verb_class == "COMMUNICATION": msg = dominant_obj if dominant_obj != "OBJECT" else "MESSAGE"; deduced_template = f"{dominant_subj} said {msg}"
            elif verb_class == "POSSESSION": item = dominant_obj if dominant_obj != "OBJECT" else "ITEM"; deduced_template = f"{dominant_subj} has/got {item}"
            elif verb_class == "COGNITION": thought = dominant_obj if dominant_obj != "OBJECT" else "IDEA"; deduced_template = f"{dominant_subj} thinks about {thought}"
            elif verb_class == "EMOTION": stimulus = dominant_obj if dominant_obj != "OBJECT" else "SOMETHING"; deduced_template = f"{dominant_subj} feels emotion about {stimulus}"
            elif dominant_verb:
                 if dominant_structure == 'VSO': action_desc = f"{dominant_verb} performed by {dominant_subj}"; action_desc += f" on {dominant_obj}" if dominant_obj != "OBJECT" else ""; deduced_template = action_desc
                 elif dominant_structure == 'SVO': action_desc = f"{dominant_subj} performs {dominant_verb}"; action_desc += f" on {dominant_obj}" if dominant_obj != "OBJECT" else ""; deduced_template = action_desc
                 elif dominant_structure == 'NOMINAL':
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
            # --- Determine Sentiment ---
            sentiment_label = None # Add sentiment logic if analyzer used
            # --- Store results ---
            self.meaning_map[cluster_id] = {
                'structure': dominant_structure, 'deduced_template': deduced_template, 'dominant_verb': dominant_verb, 'dominant_subject': dominant_subj,
                'dominant_object': dominant_obj, 'common_prep_phrase': top_prep, 'sentiment': sentiment_label, 'examples': [item['sentence'] for item in cluster_data[:3]],
                'original_templates': self.semantic_templates.get(dominant_structure, self.semantic_templates.get('OTHER', {}))
            }
            print(f"  Assigned meaning for cluster {cluster_id}: Struct={dominant_structure}, Template='{deduced_template}'") # DEBUG
        return self.meaning_map

    # --- Interpretation and Discourse ---

    def get_meaning_probability(self, embedding: np.ndarray, cluster_id: int) -> float:
        """ Calculates probability of embedding belonging to a cluster using cosine similarity. """
        if self.meaning_clusters is None or cluster_id >= len(self.meaning_clusters): return 0.0
        cluster_center = self.meaning_clusters[cluster_id]
        # Ensure embedding is 2D for cosine_similarity
        if embedding.ndim == 1: embedding = embedding.reshape(1, -1)
        if cluster_center.ndim == 1: cluster_center = cluster_center.reshape(1, -1)
        similarity = cosine_similarity(embedding, cluster_center)[0][0]
        probability = (similarity + 1) / 2 # Scale similarity from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, probability)) # Clamp to [0, 1]

    def create_specific_interpretation(self, tokens: List[str], analyses: List[Tuple], roles: Dict, structure: str, templates: Dict) -> Dict:
        """ Creates a specific interpretation by filling templates with actual values. """
        print("    Creating specific interpretation...") # DEBUG
        subject = "unknown"; verb = "unknown"; predicate = "unknown"; object_text = "unknown"
        verb_lemma = None; tense = "present"; modality = "indicative"
        verb_idx = roles.get('verb'); subj_idx = roles.get('subject'); obj_idx = roles.get('object')
        if verb_idx is not None and verb_idx < len(tokens): verb = tokens[verb_idx]; verb_lemma = analyses[verb_idx][0] if verb_idx < len(analyses) else verb
        if subj_idx is not None and subj_idx < len(tokens):
            subject = tokens[subj_idx]
            # Include determiner if present (simple check)
            if subj_idx > 0 and analyses[subj_idx-1][1] == 'DET': subject = tokens[subj_idx-1] + " " + subject
        if obj_idx is not None and obj_idx < len(tokens):
            object_text = tokens[obj_idx]
            if obj_idx > 0 and analyses[obj_idx-1][1] == 'DET': object_text = tokens[obj_idx-1] + " " + object_text
        if structure == 'NOMINAL':
            for i, (_, pos, dep, head) in enumerate(analyses):
                if pos == 'ADJ' and head == subj_idx: predicate = tokens[i]; break
        # Enhanced analysis with CAMeL Tools if available
        semantic_roles = {}; semantic_frames = []
        if self.camel_analyzer and verb_idx is not None: # Check if analyzer exists
             try:
                 morph_analysis = self.camel_analyzer.analyze(' '.join(tokens))
                 if verb_idx < len(morph_analysis):
                     verb_analysis_list = morph_analysis[verb_idx]
                     if verb_analysis_list: # Check if list is not empty
                         verb_morph = verb_analysis_list[0] # Take first analysis
                         asp = verb_morph.get('asp'); mod = verb_morph.get('mod')
                         if asp == 'p': tense = "past"
                         elif asp == 'i': tense = "present"
                         elif asp == 'c': tense = "imperative"
                         if mod == 'i': modality = "indicative"
                         elif mod == 's': modality = "subjunctive"
                         elif mod == 'j': modality = "jussive"
                 # Add semantic role extraction based on case if needed
             except Exception as e: print(f"      CAMeL analysis error in specific interpretation: {e}") # DEBUG
        # Simple rule-based semantic frame detection
        verb_class = self._classify_verb(verb_lemma)
        if verb_class != "UNKNOWN" and verb_class != "ACTION": semantic_frames.append(verb_class)
        # Fill templates
        filled_templates = {}
        for template_type, template in templates.items():
            filled = template.replace("SUBJECT", subject).replace("ACTION", verb).replace("OBJECT", object_text).replace("PREDICATE", predicate).replace("TOPIC", subject)
            filled_templates[template_type] = filled
        # Prepare semantic details
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
        print("      Finding discourse relations...") # DEBUG
        discourse_relations = []
        if not current_tokens or not previous_tokens: return discourse_relations # Need both sentences
        # Look for discourse markers at the beginning of the current sentence
        discourse_markers = {
            'CONTINUATION': ['و', 'ثم', 'ف', 'بعد ذلك', 'بعدها'], # Added 'ف'
            'CAUSE': ['لذلك', 'وبالتالي', 'لهذا السبب', 'بسبب', 'نتيجة'],
            'CONTRAST': ['لكن', 'غير أن', 'ومع ذلك', 'بالرغم', 'بينما', 'إلا أن'],
            'ELABORATION': ['أي', 'يعني', 'بمعنى'],
            'EXAMPLE': ['مثل', 'على سبيل المثال', 'مثلا'],
            'CONDITION': ['إذا', 'لو', 'إن'], # Added conditional
            'TEMPORAL': ['عندما', 'حين', 'قبل', 'بعد'], # Added temporal
        }
        first_token = current_tokens[0]
        for relation_type, markers in discourse_markers.items():
            if first_token in markers:
                discourse_relations.append({'type': relation_type, 'marker': first_token})
                print(f"        Found relation: {relation_type} (marker: {first_token})") # DEBUG
                break # Assume only one marker at the beginning for simplicity
        # Check for pronouns referring back (simple heuristic)
        pronouns = ['هذا', 'ذلك', 'تلك', 'هذه']
        if first_token in pronouns and len(current_tokens) > 1 and current_tokens[1] in ['الأمر', 'الشيء', 'الحدث', 'الفكرة', 'القول']:
             relation = {'type': 'REFERENCE', 'marker': f"{first_token} {current_tokens[1]}"}
             if relation not in discourse_relations: # Avoid duplicates if marker was also found
                 discourse_relations.append(relation)
                 print(f"        Found relation: {relation['type']} (marker: {relation['marker']})") # DEBUG
        if not discourse_relations: print("        No specific discourse markers found.") # DEBUG
        return discourse_relations

    def analyze_sentence_in_context(self, current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analysis_dict=None):
        """ Analyze a sentence considering the previous sentence's context. """
        print(f"    Analyzing sentence in context: {' '.join(current_tokens[:5])}...") # DEBUG
        # Get base interpretation (without context first)
        base_interpretation = self.interpret_sentence(current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analyses=None) # Call interpret_sentence without context first

        # If no context provided, return base interpretation
        if previous_analysis_dict is None:
            print("      No previous context provided.") # DEBUG
            return base_interpretation

        print("      Processing previous context...") # DEBUG
        previous_tokens = previous_analysis_dict.get('tokens')
        # previous_embedding = previous_analysis_dict.get('interpretation', {}).get('embedding') # Get embedding from previous interpretation

        # Find discourse relations between current and previous tokens
        discourse_info = self.find_discourse_relations(current_tokens, previous_tokens)
        base_interpretation['discourse_relations'] = discourse_info # Add relations to the result

        # --- Optional: Contextual Embedding Adjustment ---
        # This part requires a strategy for combining embeddings. Simple averaging is one option.
        # if previous_embedding is not None and base_interpretation.get('embedding') is not None:
        #     current_embedding = base_interpretation['embedding']
        #     context_influence = 0.2 # Weight for context
        #     context_aware_embedding = (1 - context_influence) * current_embedding + context_influence * previous_embedding
        #     norm = np.linalg.norm(context_aware_embedding); context_aware_embedding = context_aware_embedding / norm if norm > 0 else context_aware_embedding
        #     base_interpretation['context_aware_embedding'] = context_aware_embedding
        #     # Optional: Recalculate meaning probabilities based on context_aware_embedding
        #     # ... (logic to recalculate and update 'meaning_options') ...
        #     print("      Contextual embedding calculated.") # DEBUG

        return base_interpretation # Return the interpretation possibly augmented with discourse relations

    def interpret_sentence(self, circuit, tokens: List[str], analyses: List[Tuple], structure: str, roles: Dict, previous_analyses=None) -> Dict:
        """ Interpret sentence meaning based on circuit, linguistics, and optionally context. """
        # Note: This is the main interpretation function.
        # If context is provided, it will call analyze_sentence_in_context.
        # If not, it proceeds directly.

        # --- Context Handling ---
        # If previous_analyses is provided, call the context-aware function
        # This avoids redundant feature extraction if called from prepare_pipeline loop
        if previous_analyses is not None:
             # We assume this call happens within the main loop in prepare_quantum_nlp_pipeline
             # which handles iterating through sentences.
             # analyze_sentence_in_context will call this function again with previous_analyses=None
             # to get the base interpretation before adding context.
             print("  Redirecting to analyze_sentence_in_context...") # DEBUG
             return self.analyze_sentence_in_context(circuit, tokens, analyses, structure, roles, previous_analyses)

        # --- Direct Interpretation (No Context or Base Interpretation) ---
        print(f"\n--- Interpreting sentence (base): {' '.join(tokens)} ---") # DEBUG
        print("  Getting enhanced circuit features...") # DEBUG
        quantum_features = self.get_enhanced_circuit_features(circuit)
        # Enhance analyses with CAMeL morphology if available
        enhanced_analyses = analyses # Start with basic analyses
        camel_morphology = None; sentiment_score = None; named_entities = []
        if self.camel_analyzer:
             print("  Using CAMeL Tools for enhanced linguistic analysis...") # DEBUG
             try:
                 sentence_text = ' '.join(tokens)
                 camel_morphology = self.camel_analyzer.analyze(sentence_text)
                 # Add sentiment/NER calls here if needed and available
                 # Example (requires CAMeL sentiment/NER components):
                 # try:
                 #     from camel_tools.sentiment.factory import SentimentAnalyzer
                 #     sentiment_analyzer = SentimentAnalyzer.pretrained()
                 #     sentiment_score = sentiment_analyzer.predict([sentence_text])[0]
                 # except Exception as sent_e: print(f"    Sentiment analysis failed: {sent_e}")
                 # try:
                 #     from camel_tools.ner.ner import NERecognizer
                 #     ner = NERecognizer.pretrained()
                 #     ner_tags = ner.predict_sentence(tokens)
                 #     named_entities = list(zip(tokens, ner_tags))
                 # except Exception as ner_e: print(f"    NER failed: {ner_e}")

                 # Optional: Enhance analyses list using camel_morphology (e.g., update lemmas/POS)
                 # ... (logic to update enhanced_analyses based on camel_morphology) ...

             except Exception as camel_e: print(f"    Error during CAMeL analysis: {camel_e}") # DEBUG
        else: print("  CAMeL Tools analyzer not available.") # DEBUG

        print("  Extracting complex linguistic features...") # DEBUG
        linguistic_features = self.extract_complex_linguistic_features(tokens, enhanced_analyses, structure, roles)
        print("  Combining features with attention...") # DEBUG
        embedding = self.combine_features_with_attention(quantum_features, linguistic_features, structure)

        # Prepare base result structure
        result = {
            'sentence': ' '.join(tokens), 'structure': structure, 'embedding': embedding,
            'interpretation': None, 'meaning_options': [], 'specific_interpretation': None,
            'semantic_frames': [], 'discourse_relations': [], # Discourse relations added by context analysis wrapper
            'enhanced_linguistic_analysis': enhanced_analyses, 'roles': roles,
            'morphological_analysis': camel_morphology, 'sentiment': sentiment_score, 'named_entities': named_entities,
            'confidence': 0.0 # Default confidence
        }

        # If meaning clusters exist, find the best match
        if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
            print("  Calculating similarities to meaning clusters...") # DEBUG
            similarities = []
            for i in range(len(self.meaning_clusters)):
                prob = self.get_meaning_probability(embedding, i)
                similarities.append((i, prob))
            similarities.sort(key=lambda x: x[1], reverse=True)
            print(f"    Top cluster: {similarities[0][0]} (Prob: {similarities[0][1]:.4f})") # DEBUG
            meanings = []
            for cluster_id, prob in similarities[:min(3, len(similarities))]:
                if cluster_id in self.meaning_map:
                    cluster_info = self.meaning_map[cluster_id]
                    meanings.append({
                        'cluster_id': cluster_id, 'structure': cluster_info.get('structure', 'N/A'),
                        'deduced_template': cluster_info.get('deduced_template', 'N/A'), # Use deduced template
                        'examples': cluster_info.get('examples', []), 'probability': prob,
                        'sentiment': cluster_info.get('sentiment', None), 'topics': cluster_info.get('topics', {})
                    })
            result['meaning_options'] = meanings
            if meanings:
                 result['top_meaning_cluster'] = meanings[0]['cluster_id']
                 result['confidence'] = meanings[0]['probability']
                 print("  Creating specific interpretation for top cluster...") # DEBUG
                 # Use the deduced template from the best matching cluster
                 top_cluster_info = self.meaning_map.get(meanings[0]['cluster_id'], {})
                 # Need to reconstruct the template dict format expected by create_specific_interpretation
                 # For now, just pass the deduced template string (needs refinement)
                 # Ideally, assign_meaning_to_clusters stores a template dict per cluster
                 template_dict = top_cluster_info.get('original_templates', self.semantic_templates.get(structure, self.semantic_templates['OTHER'])) # Fallback
                 result['specific_interpretation'] = self.create_specific_interpretation(tokens, enhanced_analyses, roles, structure, template_dict)
                 # Set primary interpretation based on top match's declarative deduced template
                 result['interpretation'] = top_cluster_info.get('deduced_template', 'N/A') # Use deduced template here
        else:
            # Fallback if no clusters
            print("  No meaning clusters found or available. Using basic template.") # DEBUG
            templates = self.semantic_templates.get(structure, self.semantic_templates['OTHER'])
            result['specific_interpretation'] = self.create_specific_interpretation(tokens, enhanced_analyses, roles, structure, templates)
            result['interpretation'] = result['specific_interpretation']['templates'].get('declarative', 'N/A')

        # Add semantic frame extraction call here if needed
        # result['semantic_frames'] = self.extract_enhanced_semantic_frames(...)

        print(f"--- Interpretation complete for: {' '.join(tokens)} ---") # DEBUG
        return result

    # --- Reporting ---

    def format_discourse_relations(self, discourse_relations):
        """ Creates a user-friendly description of discourse relations. """
        if not discourse_relations: return "No specific discourse relations detected."
        formatted_output = []
        descriptions = { # Arabic descriptions
            'CONTINUATION': "تواصل هذه الجملة الفكرة السابقة باستخدام '{}'",
            'CAUSE': "تظهر هذه الجملة نتيجة أو عاقبة للجملة السابقة باستخدام '{}'",
            'CONTRAST': "تتناقض هذه الجملة مع المعلومات السابقة باستخدام '{}'",
            'ELABORATION': "توضح هذه الجملة المعلومات السابقة باستخدام '{}'",
            'EXAMPLE': "تقدم هذه الجملة مثالاً على المفهوم السابق باستخدام '{}'",
            'CONDITION': "تحدد هذه الجملة شرطًا متعلقًا بالجملة السابقة باستخدام '{}'",
            'TEMPORAL': "تحدد هذه الجملة علاقة زمنية مع الجملة السابقة باستخدام '{}'",
            'REFERENCE': "تشير هذه الجملة إلى المحتوى السابق باستخدام '{}'"
        }
        for relation in discourse_relations:
            rel_type = relation.get('type', 'UNKNOWN')
            marker = relation.get('marker', '')
            desc_template = descriptions.get(rel_type, f"تم اكتشاف علاقة من نوع {rel_type} باستخدام '{marker}'")
            formatted_output.append(desc_template.format(marker))
        return "\n".join(formatted_output) if formatted_output else "لم يتم اكتشاف علاقات خطاب محددة."

    def generate_html_report(self, discourse_analyses):
        """ Generate an HTML report with discourse analysis details. """
        print("Generating HTML report...") # DEBUG
        html = """
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>تحليل الخطاب الكمي للغة العربية</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; background-color: #f9f9f9; color: #333; }
                h1 { color: #0056b3; text-align: center; border-bottom: 2px solid #0056b3; padding-bottom: 10px;}
                .sentence-block { background-color: #fff; margin-bottom: 25px; border: 1px solid #ddd; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .sentence-text { font-size: 1.3em; font-weight: bold; color: #003d7a; margin-bottom: 15px; }
                .analysis-section { margin-top: 15px; padding-top: 15px; border-top: 1px dashed #eee; }
                .analysis-section h3 { color: #0056b3; font-size: 1.1em; margin-bottom: 8px; }
                .analysis-detail { margin-left: 20px; margin-bottom: 5px; font-size: 0.95em; }
                .detail-label { font-weight: bold; color: #555; }
                .relation { color: #006699; margin-top: 8px; background-color: #e7f3fe; padding: 8px; border-radius: 4px; border-left: 3px solid #006699; }
                .relation-marker { font-weight: bold; color: #cc0000; }
                .no-relation { color: #777; font-style: italic; margin-top: 8px; }
                .interpretation { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>تحليل الخطاب الكمي للنص العربي</h1>
        """
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis.get('sentence', 'N/A')
            interpretation_data = analysis.get('interpretation', {}) # This is the dict returned by interpret_sentence

            html += f'<div class="sentence-block">'
            html += f'<div class="sentence-text">الجملة {i+1}: {sentence}</div>'

            # --- Basic Interpretation Details ---
            html += '<div class="analysis-section">'
            html += '<h3>التحليل الأساسي:</h3>'
            structure = interpretation_data.get('structure', 'غير محدد')
            confidence = interpretation_data.get('confidence', 0.0)
            html += f'<div class="analysis-detail"><span class="detail-label">البنية:</span> {structure}</div>'
            html += f'<div class="analysis-detail"><span class="detail-label">الثقة (التشابه مع العنقود):</span> {confidence:.2f}</div>'
            # Add more basic details if available in interpretation_data (e.g., sentiment, entities)
            if interpretation_data.get('sentiment') is not None:
                 html += f'<div class="analysis-detail"><span class="detail-label">الشعور:</span> {interpretation_data["sentiment"]}</div>'
            if interpretation_data.get('named_entities'):
                 html += f'<div class="analysis-detail"><span class="detail-label">الكيانات المسماة:</span> {interpretation_data["named_entities"]}</div>'
            html += '</div>'

            # --- Specific Interpretation ---
            specific_interp = interpretation_data.get('specific_interpretation')
            if specific_interp and isinstance(specific_interp, dict):
                html += '<div class="analysis-section interpretation">'
                html += '<h3>التفسير المحدد:</h3>'
                # Display the main interpretation template (e.g., declarative)
                main_interp_text = specific_interp.get('templates', {}).get('declarative', 'لا يوجد تفسير محدد.')
                html += f'<div class="analysis-detail">{main_interp_text}</div>'
                # Optionally add more details from semantic_details
                sem_details = specific_interp.get('semantic_details', {})
                if sem_details:
                     verb_info = sem_details.get('verb', {})
                     html += f'<div class="analysis-detail" style="font-size: 0.9em; color: #444;">'
                     html += f' <span class="detail-label">الفعل:</span> {verb_info.get("text", "-")} ({verb_info.get("lemma", "-")}),'
                     html += f' <span class="detail-label">الزمن:</span> {verb_info.get("tense", "-")}'
                     html += f'</div>'
                html += '</div>'
            else:
                 # Display the general interpretation if specific is missing
                 general_interp = interpretation_data.get('interpretation', 'لا يوجد تفسير.')
                 html += '<div class="analysis-section interpretation">'
                 html += '<h3>التفسير العام:</h3>'
                 html += f'<div class="analysis-detail">{general_interp}</div>'
                 html += '</div>'


            # --- Discourse Relations ---
            if i > 0: # Only show relations for sentences after the first one
                html += '<div class="analysis-section">'
                html += '<h3>العلاقة مع الجملة السابقة:</h3>'
                discourse_relations = interpretation_data.get('discourse_relations', [])
                relation_text = self.format_discourse_relations(discourse_relations)
                # Wrap each relation in a div for better spacing if multiple exist
                if discourse_relations:
                     for line in relation_text.split('\n'):
                          html += f'<div class="relation">{line}</div>'
                else:
                     html += f'<div class="no-relation">{relation_text}</div>'
                html += '</div>'

            html += '</div>' # Close sentence-block

        html += """
        </body>
        </html>
        """
        print("HTML report generation complete.") # DEBUG
        return html

    # --- Utility Methods ---

    def save_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Saves the trained kernel state to a file using pickle. """
        print(f"Saving model to {filename}...") # DEBUG
        model_data = {
            'embedding_dim': self.embedding_dim, 'num_clusters': self.num_clusters,
            'meaning_clusters': self.meaning_clusters, 'cluster_labels': self.cluster_labels,
            'meaning_map': self.meaning_map, 'reference_sentences': self.reference_sentences,
            'circuit_embeddings': self.circuit_embeddings, 'sentence_embeddings': self.sentence_embeddings,
            'semantic_templates': self.semantic_templates
        }
        try:
            with open(filename, 'wb') as f: pickle.dump(model_data, f)
            print(f"Model saved successfully.") # DEBUG
        except Exception as e: print(f"Error saving model: {e}"); traceback.print_exc() # DEBUG

    def load_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Loads a trained kernel state from a file. """
        if not os.path.exists(filename): print(f"Error: Model file {filename} not found."); return self
        print(f"Loading model from {filename}...") # DEBUG
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
            print(f"Model loaded successfully.") # DEBUG
        except Exception as e: print(f"Error loading model: {e}"); traceback.print_exc() # DEBUG
        return self

    def visualize_meaning_space(self, highlight_indices=None, save_path=None):
        """ Visualize the sentence meaning space using PCA. """
        print("Visualizing meaning space...") # DEBUG
        if not self.sentence_embeddings: print("Warning: No embeddings available for visualization."); return None
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("Error: scikit-learn is required for visualization. Please install it (`pip install scikit-learn`).")
            return None

        embeddings = list(self.sentence_embeddings.values())
        indices = list(self.sentence_embeddings.keys())
        if not embeddings: print("Warning: Embeddings list is empty."); return None
        X = np.array(embeddings)
        if X.shape[1] < 2: print("Warning: Need at least 2 embedding dimensions for PCA visualization."); return None

        try:
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(X)
            plt.figure(figsize=(12, 10))
            # Use cluster labels for color if available, otherwise default color
            colors = self.cluster_labels if self.cluster_labels is not None and len(self.cluster_labels) == len(reduced_embeddings) else 'blue'
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, cmap='viridis', alpha=0.7, s=100)
            # Highlight specific sentences
            if highlight_indices is not None:
                highlight_idxs = [indices.index(i) for i in highlight_indices if i in indices]
                if highlight_idxs:
                    plt.scatter(reduced_embeddings[highlight_idxs, 0], reduced_embeddings[highlight_idxs, 1], c='red', s=150, edgecolor='white', zorder=10, label='Highlighted')
                    # Add labels (optional, can get crowded)
                    # for idx in highlight_idxs: plt.annotate(self.reference_sentences.get(indices[idx], f"Sent {indices[idx]}"), (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]), xytext=(5, 5), textcoords='offset points')
            # Add legend for clusters if labels were used
            if self.cluster_labels is not None and isinstance(colors, np.ndarray):
                try: # legend_elements can fail if only one cluster exists
                     legend1 = plt.legend(*scatter.legend_elements(), title="Meaning Clusters")
                     plt.gca().add_artist(legend1)
                except Exception as leg_e: print(f"Warning: Could not create cluster legend: {leg_e}")
            # Add cluster centers and labels
            if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
                try:
                    cluster_centers_2d = pca.transform(self.meaning_clusters)
                    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker='*', s=350, c='white', edgecolor='black', label='Cluster Centers')
                    for i, (x, y) in enumerate(cluster_centers_2d):
                        if i in self.meaning_map:
                            meaning = self.meaning_map[i].get('structure', f'Cluster {i}')
                            template = self.meaning_map[i].get('deduced_template', '')[:30] # Show structure and start of template
                            plt.annotate(f"Cluster {i}: {meaning}\n'{template}...'", (x, y), xytext=(0, 15), textcoords='offset points', ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))
                except Exception as cc_e: print(f"Error plotting cluster centers: {cc_e}")
            plt.title('Quantum Sentence Meaning Space')
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=150); print(f"Visualization saved to {save_path}") # DEBUG
            return plt.gcf() # Return figure object
        except Exception as e:
            print(f"Error during PCA visualization: {e}"); traceback.print_exc(); return None # DEBUG
def prepare_quantum_nlp_pipeline():
    """
    Example of how to use the ArabicQuantumMeaningKernel in a complete pipeline.
    Processes sentences, trains the kernel, and performs discourse analysis.
    (Updated to pass generated Qiskit circuits to the kernel)
    """
    sentence_file = "sentences.txt"
    sentences = []
    try:
        with open(sentence_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()][:20] # Limit to 20 sentences
        print(f"Loaded {len(sentences)} sentences from {sentence_file}")
    except FileNotFoundError:
        print(f"Error: Sentence file '{sentence_file}' not found. Using default example sentences.")
        sentences = [
            "يقرأ الولد الكتاب", "الولد يقرأ الكتاب", "كتب الطالب الدرس",
            "الطالب يكتب الدرس", "البيت كبير", "درسَ خالدٌ كثيرًا لأنه يريد النجاحَ"
        ]

    # Process sentences to get Qiskit circuits and linguistic info
    results = []
    print("\n--- Processing Sentences ---")
    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx+1}/{len(sentences)}: '{sentence[:40]}...'")
        try:
            # arabic_to_quantum_enhanced returns the generated Qiskit circuit first
            circuit, diagram, structure, tokens, analyses, roles = arabic_to_quantum_enhanced(sentence, debug=False)

            # --- Store the generated Qiskit circuit ---
            results.append({
                'sentence': sentence,
                'circuit': circuit, # Store the Qiskit Circuit object
                'diagram': diagram, # Keep diagram for reference if needed
                'structure': structure,
                'tokens': tokens,
                'analyses': analyses,
                'roles': roles
            })
            print(f"  Sentence {idx+1}: Processed successfully. Structure: {structure}, Circuit type: {type(circuit)}")
        except Exception as e:
            print(f"  ERROR processing sentence {idx+1}: {sentence}")
            print(f"  Error type: {type(e).__name__}, Message: {e}")
            traceback.print_exc()
            print("  Skipping this sentence.")

    if not results:
        print("Error: No sentences were processed successfully. Exiting pipeline.")
        return None, None, []

    # Create and train the kernel
    num_processed = len(results)
    kernel_clusters = min(5, max(1, num_processed // 2))
    kernel = ArabicQuantumMeaningKernel(embedding_dim=14, num_clusters=kernel_clusters)

    print(f"\n--- Training Kernel ({kernel.num_clusters} clusters) ---")
    # --- Pass the generated Qiskit circuits to train ---
    kernel.train(
        sentences=[r['sentence'] for r in results],
        circuits=[r['circuit'] for r in results], # Pass the circuits
        tokens_list=[r['tokens'] for r in results],
        analyses_list=[r['analyses'] for r in results],
        structures=[r['structure'] for r in results],
        roles_list=[r['roles'] for r in results]
    )

    # Save the model
    kernel.save_model('arabic_quantum_kernel.pkl')

    # Visualize meaning space
    print("\n--- Visualizing Meaning Space ---")
    try:
        kernel.visualize_meaning_space(save_path='meaning_space.png')
    except AttributeError:
        print("Warning: visualize_meaning_space method not found in Kernel.")
    except Exception as vis_e:
        print(f"Error during meaning space visualization: {vis_e}")

    # Analyze sequence of sentences with discourse relations
    print("\n--- Performing Discourse Analysis ---")
    discourse_analyses = []
    previous_analysis_dict = None

    for i, result in enumerate(results):
        print(f"Analyzing sentence {i+1}/{len(results)} in context...")
        circuit = result['circuit'] # Use the stored Qiskit circuit
        tokens = result['tokens']
        analyses = result['analyses']
        structure = result['structure']
        roles = result['roles']

        try:
            # --- Pass the Qiskit circuit to interpret_sentence ---
            interpretation = kernel.interpret_sentence(
                circuit, tokens, analyses, structure, roles,
                previous_analyses=previous_analysis_dict
            )

            current_analysis_dict = {
                'sentence': result['sentence'], 'tokens': tokens, 'analyses': analyses,
                'structure': structure, 'roles': roles, 'interpretation': interpretation
            }
            discourse_analyses.append(current_analysis_dict)
            previous_analysis_dict = current_analysis_dict
            print(f"  Sentence {i+1}: Analysis complete.")
        except AttributeError as ae:
             print(f"  AttributeError during sentence analysis {i+1}: {ae}. Method might be missing from Kernel.")
             discourse_analyses.append({'sentence': result['sentence'], 'interpretation': {'error': str(ae)}})
        except Exception as analysis_e:
             print(f"  Error during sentence analysis {i+1}: {analysis_e}")
             discourse_analyses.append({'sentence': result['sentence'], 'interpretation': {'error': str(analysis_e)}})

    # Generate and save HTML report
    print("\n--- Generating HTML Report ---")
    try:
        html_report = kernel.generate_html_report(discourse_analyses)
        report_filename = 'discourse_analysis_report.html'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML report saved to {report_filename}")
    except AttributeError:
        print("Warning: generate_html_report method not found in Kernel.")
    except Exception as report_e:
        print(f"Error generating HTML report: {report_e}")

    # --- Example: Interpret a new sentence ---
    print("\n--- Interpreting a New Sentence ---")
    new_sentence = "الطالبة تدرس العلوم في الجامعة"
    print(f"New sentence: {new_sentence}")
    last_interpretation_result = None
    try:
        # Process the new sentence to get the Qiskit circuit
        new_circuit, _, new_structure, new_tokens, new_analyses, new_roles = arabic_to_quantum_enhanced(new_sentence, debug=False)

        # --- Interpret using the new Qiskit circuit ---
        last_interpretation_result = kernel.interpret_sentence(
            new_circuit, new_tokens, new_analyses, new_structure, new_roles,
            previous_analyses=previous_analysis_dict
        )
        print("New sentence interpretation complete.")

    except AttributeError as ae:
         print(f"AttributeError interpreting new sentence: {ae}. Check Kernel methods or arabic_to_quantum_enhanced import.")
    except Exception as interp_e:
        print(f"Error interpreting new sentence: {interp_e}")
        traceback.print_exc()

    print("\n--- Pipeline Finished ---")
    return kernel, last_interpretation_result, discourse_analyses
# End of ArabicQuantumMeaningKernel class
if __name__ == "__main__":
    # Correctly unpack all three return values
    kernel, last_new_sentence_interpretation, discourse_analyses_list = prepare_quantum_nlp_pipeline()

    # Check if pipeline ran successfully
    if kernel is None:
        print("\nPipeline execution failed.")
    else:
        print("\n--- Full Discourse Analysis Summary ---")
        # Print basic info for each sentence in the discourse analysis
        if discourse_analyses_list:
            for i, analysis in enumerate(discourse_analyses_list):
                print(f"\nSentence {i+1}: {analysis.get('sentence', 'N/A')}")
                interp_details = analysis.get('interpretation', {})
                specific_interp = interp_details.get('specific_interpretation', {})
                if specific_interp:
                     print(f"  Interpretation: {specific_interp.get('templates', {}).get('declarative', 'N/A')}")
                if 'discourse_relations' in interp_details and interp_details['discourse_relations']:
                     print(f"  Relations to previous: {interp_details['discourse_relations']}")
                print(f"  Structure: {interp_details.get('structure', 'N/A')}")
                print(f"  Confidence: {interp_details.get('confidence', 0.0):.2f}") # Default to 0.0 if missing
        else:
            print("No discourse analysis results to display.")


        print("\n--- Interpretation of Last New Sentence Processed ---")
        # Use the 'last_new_sentence_interpretation' variable unpacked earlier
        if last_new_sentence_interpretation is not None:
            print(f"Sentence: {last_new_sentence_interpretation.get('sentence', 'Not available')}")
            print(f"Structure: {last_new_sentence_interpretation.get('structure', 'N/A')}")
            print(f"Confidence: {last_new_sentence_interpretation.get('confidence', 'N/A')}")

            # Print detailed interpretation of the last sentence
            if 'specific_interpretation' in last_new_sentence_interpretation:
                print("\nSpecific Interpretation Details:")
                sem_details = last_new_sentence_interpretation['specific_interpretation'].get('semantic_details', {})
                print(f"  Subject: {sem_details.get('subject', {}).get('text', 'N/A')}")
                print(f"  Verb: {sem_details.get('verb', {}).get('text', 'N/A')} (Lemma: {sem_details.get('verb', {}).get('lemma', 'N/A')}, Tense: {sem_details.get('verb', {}).get('tense', 'N/A')})")
                print(f"  Object: {sem_details.get('object', {}).get('text', 'N/A')}")
                if 'named_entities' in last_new_sentence_interpretation and last_new_sentence_interpretation['named_entities']:
                    print(f"  Named Entities: {last_new_sentence_interpretation['named_entities']}")
                if 'sentiment' in last_new_sentence_interpretation and last_new_sentence_interpretation['sentiment'] is not None:
                    # Format sentiment score if it's numeric
                    sentiment_val = last_new_sentence_interpretation['sentiment']
                    if isinstance(sentiment_val, (int, float)):
                         print(f"  Sentiment Score: {sentiment_val:.4f}")
                    else:
                         print(f"  Sentiment Label: {sentiment_val}")


            # Optionally print meaning options
            # print("\nMeaning Options:")
            # for option in last_new_sentence_interpretation.get('meaning_options', []):
            #    print(f"  - Cluster: {option.get('cluster_id')}, Structure: {option.get('structure')}, Probability: {option.get('probability'):.2f}")

        else:
            print("No interpretation available for the new sentence (it might have failed).")

