# quantum_kernel_v2.py (Enhanced Version - Reverted Unpacking Fix)
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister # Ensure ClassicalRegister is imported
from qiskit.visualization import plot_state_city # Import for state visualization
from qiskit.quantum_info import partial_trace # For reduced density matrix
import pickle
import os
import traceback
from collections import Counter
from qiskit_aer import AerSimulator
# Import the function that returns circuit, diagram, ...
from camel_test2 import arabic_to_quantum_enhanced # Dependency
# Also import the specific Diagram class for type checking if needed
from lambeq.backend.grammar import Diagram as GrammarDiagram
#from lambeq.backend.quantum import QiskitBackend
#from lambeq.backend.qiskit import QiskitBackend
from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram

# --- NEW: Import for Enhanced Clustering (if using LDA) ---
try:
    from gensim import corpora, models
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: gensim not found (pip install gensim). Enhanced clustering with LDA disabled.")

# --- NEW: Imports/Setup for Semantic Frame Extraction (if using Farasa/AraVec) ---
# You might need to install and configure these separately
ARAVEC_MODEL = None
FARASA_NER = None
try:
    # Example: Load AraVec (adjust path as needed)
    # from gensim.models import Word2Vec
    # aravec_path = "../aravec/tweet_cbow_300" # Example path
    # if os.path.exists(aravec_path):
    #     ARAVEC_MODEL = Word2Vec.load(aravec_path)
    #     print("AraVec model loaded.")
    pass # Placeholder for AraVec loading
except Exception as e:
    print(f"Warning: Could not load AraVec model: {e}")

try:
    # Example: Initialize Farasa NER
    # from farasa.ner import FarasaNamedEntityRecognizer
    # FARASA_NER = FarasaNamedEntityRecognizer()
    # print("Farasa NER initialized.")
    pass # Placeholder for Farasa NER initialization
except Exception as e:
    print(f"Warning: Could not initialize Farasa NER: {e}")
# --- END NEW IMPORTS ---


# Helper function to ensure circuit name (applied to Qiskit circuits)
def _ensure_circuit_name(circuit: Any, default_name: str = "qiskit_circuit") -> Any:
    """Checks if a Qiskit circuit object has a name and assigns one if not."""
    if isinstance(circuit, QuantumCircuit):
        has_name_attr = hasattr(circuit, 'name')
        name_is_set = has_name_attr and getattr(circuit, 'name', None)
        if not name_is_set:
            try:
                setattr(circuit, 'name', default_name)
                # print(f"Debug: Assigned default name '{default_name}' to Qiskit circuit.") # DEBUG
            except Exception as e:
                print(f"Debug: Error assigning name to Qiskit circuit: {e}.") # DEBUG
    return circuit

class ArabicQuantumMeaningKernel:
    """
    A quantum kernel for mapping quantum circuit outputs to potential
    sentence meanings for Arabic language processing. Includes discourse analysis.
    (Enhanced Implementation incorporating features from quantum_kernel.py)
    """
    def __init__(self,
                 embedding_dim: int = 14,
                 num_clusters: int = 5,
                 simulator_backend: str = 'statevector_simulator'):
        """ Initialize the quantum meaning kernel. """
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.simulator = AerSimulator()
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
            # Using 'calima-msa-r13' as an example, adjust if needed
            db_path = MorphologyDB.builtin_db('calima-msa-r13')
            self.camel_analyzer = Analyzer(db_path)
            print("CAMeL Tools Analyzer initialized successfully.")
        except ImportError:
            print("Warning: CAMeL Tools not found (pip install camel-tools). NLP enhancements disabled.")
        except LookupError:
             print("Warning: CAMeL Tools default DB not found. Run 'camel_tools download calima-msa-r13' or specify DB path. NLP enhancements disabled.")
        except Exception as e:
            print(f"Warning: Error initializing CAMeL Tools Analyzer: {e}. NLP enhancements disabled.")


    # --- Feature Extraction and Embedding ---
    # [get_circuit_features, _get_reduced_density_matrix, get_enhanced_circuit_features]
    # [extract_linguistic_features, extract_complex_linguistic_features]
    # [combine_features_with_attention]
    # (These functions remain the same as in the previous version - they expect a QuantumCircuit)
    # --- (Code for these functions omitted for brevity, assume they are correct from previous version) ---
    def get_circuit_features(self, circuit, shots: int = 1024) -> np.ndarray:
        """ Extracts features from a Qiskit quantum circuit by executing it. """
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_circuit_features received non-Qiskit object: {type(circuit)}")
             fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback
        try:
            params = circuit.parameters
            if params: # Check if there are any parameters
                num_params = len(params)
                # Generate random values (or use fixed values, e.g., np.pi/2)
                param_values = np.random.rand(num_params) * 2 * np.pi # Random angles between 0 and 2*pi
                # Create the binding dictionary
                parameter_binds = {param: value for param, value in zip(params, param_values)}
                if debug: print(f"  Binding {num_params} parameters.") # Optional debug print
                # Run with the parameter bindings (note: parameter_binds expects a list)
                job = self.simulator.run(circuit, parameter_binds=[parameter_binds])
            else: # No parameters, run directly
                if debug: print("  No parameters found in circuit. Running directly.") # Optional debug print
                job = self.simulator.run(circuit)
                circuit = _ensure_circuit_name(circuit, "circuit_for_features")
                job = self.simulator.run(circuit)
                result = job.result()
            if hasattr(result, 'get_statevector'):
                statevector = result.get_statevector()
                amplitudes = np.abs(statevector); phases = np.angle(statevector)
                half_dim = self.embedding_dim // 2
                amp_slice = amplitudes[:min(half_dim, len(amplitudes))]
                phase_slice = phases[:min(half_dim, len(phases))]
                features = np.concatenate([amp_slice, phase_slice])
                if len(features) < self.embedding_dim: features = np.pad(features, (0, self.embedding_dim - len(features)), 'constant')
                elif len(features) > self.embedding_dim: features = features[:self.embedding_dim]
                norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
                return features
            else:
                 counts = result.get_counts(); total_shots = sum(counts.values())
                 feature_vector = np.zeros(self.embedding_dim); num_outcomes = 0
                 if total_shots > 0:
                     for outcome, count in counts.items():
                         num_outcomes += 1
                         try:
                             idx_base = int(outcome, 2) if isinstance(outcome, str) and all(c in '01' for c in outcome) else hash(outcome)
                             idx = idx_base % self.embedding_dim; feature_vector[idx] += count / total_shots
                         except Exception as e_idx: print(f"  Warning: Could not process outcome '{outcome}': {e_idx}")
                 norm = np.linalg.norm(feature_vector); feature_vector = feature_vector / norm if norm > 0 else feature_vector
                 return feature_vector
        except Exception as e:
            print(f"\n--- ERROR in get_circuit_features ---"); print(f"Error type: {type(e)}"); print(f"Error message: {e}"); traceback.print_exc(); print(f"---------------------------------------\n")
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
        reduced_dims = 2 ** len(subsystem_qubits)
        return np.eye(reduced_dims, dtype=complex) / reduced_dims

    def get_enhanced_circuit_features(self, circuit, shots: int = 1024) -> np.ndarray:
        """ Enhanced feature extraction from a Qiskit circuit. """
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_enhanced_circuit_features received non-Qiskit object: {type(circuit)}")
             fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback
        try:
            basic_features = self.get_circuit_features(circuit, shots)
            circuit = _ensure_circuit_name(circuit, "circuit_for_enhanced")
            job = self.simulator.run(circuit)
            result = job.result()
            entanglement_features = []; pauli_expectations = []
            if hasattr(result, 'get_statevector'):
                statevector = result.get_statevector(); num_qubits = circuit.num_qubits
                if hasattr(self, '_get_reduced_density_matrix') and num_qubits > 1:
                    num_entanglement_checks = min(num_qubits -1, 3)
                    for i in range(num_entanglement_checks):
                        try:
                            reduced_density = self._get_reduced_density_matrix(statevector, [i])
                            eigenvalues = np.linalg.eigvalsh(reduced_density)
                            valid_eigenvalues = eigenvalues[eigenvalues > 1e-12]
                            entropy = -np.sum(valid_eigenvalues * np.log2(valid_eigenvalues)) if len(valid_eigenvalues) > 0 else 0.0
                            entanglement_features.append(np.real(entropy))
                        except Exception as inner_e: print(f"  Error calculating entanglement for qubit {i}: {type(inner_e).__name__} - {inner_e}"); entanglement_features.append(0.0)
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
                                cr = ClassicalRegister(num_qubits, name=f'c_pauli_{i}')
                                meas_circuit.add_register(cr)
                            else: cr = meas_circuit.cregs[0]
                            if i < cr.size: meas_circuit.measure(i, cr[i])
                            else: print(f"Warning: Classical register '{cr.name}' too small for qubit {i}. Skipping measure."); pauli_expectations.append(0.0); continue
                            meas_result = execute(meas_circuit, self.simulator, shots=shots).result(); counts = meas_result.get_counts()
                            exp_val = 0; total = sum(counts.values())
                            if total > 0:
                                 prob_0, prob_1 = 0, 0
                                 for bitstring, count in counts.items():
                                      bit_index = i
                                      if 0 <= bit_index < len(bitstring):
                                          if bitstring[len(bitstring) - 1 - bit_index] == '0':
                                              prob_0 += count / total
                                          else:
                                              prob_1 += count / total
                                 exp_val = prob_0 - prob_1
                            pauli_expectations.append(exp_val);
                    except Exception as pauli_e: print(f"  Error in Pauli expectation calculation: {type(pauli_e).__name__} - {pauli_e}"); traceback.print_exc(); num_pauli_checks = min(num_qubits, 3); pauli_expectations.extend([0.0] * (num_pauli_checks - len(pauli_expectations)))
            feature_list = [basic_features]
            if entanglement_features: feature_list.append(np.array(entanglement_features))
            if pauli_expectations: feature_list.append(np.array(pauli_expectations))
            all_features = np.concatenate(feature_list) if len(feature_list) > 1 else basic_features
            current_len = len(all_features)
            if current_len < self.embedding_dim: all_features = np.pad(all_features, (0, self.embedding_dim - current_len), 'constant')
            elif current_len > self.embedding_dim: all_features = all_features[:self.embedding_dim]
            norm = np.linalg.norm(all_features); all_features = all_features / norm if norm > 0 else all_features
            return all_features
        except Exception as e:
            print(f"\n--- ERROR in get_enhanced_circuit_features ---"); print(f"Error type: {type(e)}"); print(f"Error message: {e}"); traceback.print_exc(); print(f"----------------------------------------------\n")
        if 'basic_features' in locals() and isinstance(basic_features, np.ndarray):
             current_len = len(basic_features)
             if current_len < self.embedding_dim: basic_features = np.pad(basic_features, (0, self.embedding_dim - current_len), 'constant')
             elif current_len > self.embedding_dim: basic_features = basic_features[:self.embedding_dim]
             norm = np.linalg.norm(basic_features); basic_features = basic_features / norm if norm > 0 else basic_features
             return basic_features
        else: fallback = np.random.rand(self.embedding_dim); norm = np.linalg.norm(fallback); return fallback / norm if norm > 0 else fallback

    def extract_linguistic_features(self, tokens: List[str], analyses: List[Tuple], structure: str, roles: Dict) -> np.ndarray:
        """ Extracts basic linguistic features. """
        features = np.zeros(self.embedding_dim)
        num_features = 0
        structure_map = {'VSO': 0, 'SVO': 1, 'NOMINAL': 2, 'COMPLEX': 3, 'OTHER': 4}
        structure_idx = structure_map.get(structure, 4)
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
        feature_idx = 8
        if feature_idx < self.embedding_dim:
            subordinate_markers = ['الذي', 'التي', 'الذين', 'اللواتي', 'عندما', 'حيث', 'لأن', 'كي', 'أنّ']
            has_subordinate = any(token in subordinate_markers for token in tokens)
            features[feature_idx] = 1.0 if has_subordinate else 0.0; feature_idx += 1
        if feature_idx < self.embedding_dim:
            verb_count = sum(1 for _, pos, _, _ in analyses if pos == 'VERB')
            features[feature_idx] = min(verb_count / 3.0, 1.0); feature_idx += 1
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
            features[feature_idx] = min(avg_word_length / 10.0, 1.0); feature_idx += 1
        if feature_idx < self.embedding_dim:
             features[feature_idx] = min(len(tokens) / 50.0, 1.0); feature_idx += 1
        norm = np.linalg.norm(features); features = features / norm if norm > 0 else features
        return features

    def combine_features_with_attention(self, quantum_features, linguistic_features, structure):
        """ Combine features with attention mechanism based on sentence structure. """
        if structure == 'NOMINAL': quantum_weight, linguistic_weight = 0.3, 0.7
        elif structure in ['VSO', 'SVO']: quantum_weight, linguistic_weight = 0.5, 0.5
        elif 'COMPLEX' in structure or structure == 'COMPLEX': quantum_weight, linguistic_weight = 0.6, 0.4
        else: quantum_weight, linguistic_weight = 0.5, 0.5
        q_len = len(quantum_features); l_len = len(linguistic_features)
        if q_len < self.embedding_dim: quantum_features = np.pad(quantum_features, (0, self.embedding_dim - q_len), 'constant')
        elif q_len > self.embedding_dim: quantum_features = quantum_features[:self.embedding_dim]
        if l_len < self.embedding_dim: linguistic_features = np.pad(linguistic_features, (0, self.embedding_dim - l_len), 'constant')
        elif l_len > self.embedding_dim: linguistic_features = linguistic_features[:self.embedding_dim]
        combined = quantum_weight * quantum_features + linguistic_weight * linguistic_features
        norm = np.linalg.norm(combined); combined = combined / norm if norm > 0 else combined
        return combined

    # --- Training and Clustering ---
    # [train, learn_meaning_clusters, learn_enhanced_meaning_clusters]
    # [_classify_verb, assign_meaning_to_clusters]
    # (These functions remain the same as in the previous version)
    # --- (Code for these functions omitted for brevity, assume they are correct from previous version) ---
    def train(self, sentences, circuits, tokens_list, analyses_list, structures, roles_list, use_enhanced_clustering=False):
        """ Train the kernel on a set of sentences and their Qiskit circuits. """
        self.reference_sentences = {i: sentences[i] for i in range(len(sentences))}
        self.circuit_embeddings = {}; self.sentence_embeddings = {}; embeddings = []
        print(f"\n--- Training Kernel on {len(sentences)} sentences ---")
        if not (len(circuits) == len(tokens_list) == len(analyses_list) == len(structures) == len(roles_list) == len(sentences)):
            print("Error: Input lists to train method have mismatched lengths.")
            min_len = min(len(sentences), len(circuits), len(tokens_list), len(analyses_list), len(structures), len(roles_list))
            print(f"Warning: Training with reduced dataset size: {min_len}")
            if min_len == 0: print("Error: Cannot train with empty dataset."); return self
            sentences, circuits, tokens_list, analyses_list, structures, roles_list = (lst[:min_len] for lst in [sentences, circuits, tokens_list, analyses_list, structures, roles_list])
        else: min_len = len(sentences)
        for i in range(min_len):
            try:
                current_circuit = circuits[i]
                if not isinstance(current_circuit, QuantumCircuit):
                     print(f"  WARNING: Skipping embedding for sentence {i+1}. Expected QuantumCircuit, got {type(current_circuit)}")
                     continue
                quantum_features = self.get_enhanced_circuit_features(current_circuit)
                self.circuit_embeddings[i] = quantum_features
                linguistic_features = self.extract_complex_linguistic_features(tokens_list[i], analyses_list[i], structures[i], roles_list[i])
                embedding = self.combine_features_with_attention(quantum_features, linguistic_features, structures[i])
                self.sentence_embeddings[i] = embedding; embeddings.append(embedding)
            except Exception as e: print(f"  ERROR processing sentence {i+1} during training: {sentences[i]}"); print(f"  Error type: {type(e).__name__}, Message: {e}"); traceback.print_exc(); print("  Skipping embedding for this sentence.")
        if not embeddings: print("Error: No embeddings generated. Cannot proceed."); return self
        if use_enhanced_clustering and GENSIM_AVAILABLE:
            print("Learning meaning clusters (enhanced with topic modeling)...")
            self.learn_enhanced_meaning_clusters(embeddings, sentences)
        else:
            if use_enhanced_clustering and not GENSIM_AVAILABLE:
                print("Warning: Enhanced clustering requested but gensim not available. Falling back to basic clustering.")
            print("Learning meaning clusters (basic)...")
            self.learn_meaning_clusters(embeddings)
        print("Assigning meaning to clusters...")
        self.assign_meaning_to_clusters(sentences, structures, roles_list, analyses_list)
        print("--- Training Complete ---")
        return self

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
        self.learn_meaning_clusters(embeddings)
        if self.meaning_clusters is None or self.cluster_labels is None:
            print("Skipping topic modeling enhancement due to prior clustering failure.")
            return
        if not GENSIM_AVAILABLE:
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
            num_topics = min(self.num_clusters * 2, len(dictionary), 15)
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
            sentiment_label = None
            if cluster_id not in self.meaning_map:
                 self.meaning_map[cluster_id] = {}
            self.meaning_map[cluster_id].update({
                'structure': dominant_structure, 'deduced_template': deduced_template, 'dominant_verb': dominant_verb, 'dominant_subject': dominant_subj,
                'dominant_object': dominant_obj, 'common_prep_phrase': top_prep, 'sentiment': sentiment_label, 'examples': [item['sentence'] for item in cluster_data[:3]],
                'original_templates': self.semantic_templates.get(dominant_structure, self.semantic_templates.get('OTHER', {}))
            })
            if 'topics' not in self.meaning_map[cluster_id]:
                 self.meaning_map[cluster_id]['topics'] = {}
        return self.meaning_map

    # --- Interpretation and Discourse ---
    # [get_meaning_probability, create_specific_interpretation]
    # [find_discourse_relations, analyze_sentence_in_context]
    # (These functions remain the same as in the previous version)
    # --- (Code for these functions omitted for brevity, assume they are correct from previous version) ---
    def get_meaning_probability(self, embedding: np.ndarray, cluster_id: int) -> float:
        """ Calculates probability of embedding belonging to a cluster using cosine similarity. """
        if self.meaning_clusters is None or cluster_id >= len(self.meaning_clusters): return 0.0
        cluster_center = self.meaning_clusters[cluster_id]
        if embedding.ndim == 1: embedding = embedding.reshape(1, -1)
        if cluster_center.ndim == 1: cluster_center = cluster_center.reshape(1, -1)
        if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(cluster_center)):
             print(f"Warning: NaN or Inf found in embedding or cluster center for cluster {cluster_id}. Returning 0 probability.")
             return 0.0
        if embedding.shape[1] != cluster_center.shape[1]:
             print(f"Warning: Dimension mismatch between embedding ({embedding.shape}) and cluster center ({cluster_center.shape}) for cluster {cluster_id}. Returning 0 probability.")
             return 0.0
        try:
             similarity = cosine_similarity(embedding, cluster_center)[0][0]
        except ValueError as ve:
             print(f"Error calculating cosine similarity for cluster {cluster_id}: {ve}")
             return 0.0
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
             if relation not in discourse_relations:
                 discourse_relations.append(relation)
        return discourse_relations

    def analyze_sentence_in_context(self, current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analysis_dict=None):
        """ Analyze a sentence considering the previous sentence's context. """
        base_interpretation = self.interpret_sentence(current_circuit, current_tokens, current_analyses, current_structure, current_roles, previous_analyses=None)
        if previous_analysis_dict is None:
            return base_interpretation
        previous_tokens = previous_analysis_dict.get('tokens')
        discourse_info = self.find_discourse_relations(current_tokens, previous_tokens)
        base_interpretation['discourse_relations'] = discourse_info
        previous_embedding = previous_analysis_dict.get('interpretation', {}).get('embedding')
        if previous_embedding is not None and base_interpretation.get('embedding') is not None:
            current_embedding = base_interpretation['embedding']
            context_influence = 0.2
            if isinstance(current_embedding, np.ndarray) and isinstance(previous_embedding, np.ndarray) and current_embedding.shape == previous_embedding.shape:
                context_aware_embedding = (1 - context_influence) * current_embedding + context_influence * previous_embedding
                norm = np.linalg.norm(context_aware_embedding); context_aware_embedding = context_aware_embedding / norm if norm > 0 else context_aware_embedding
                base_interpretation['context_aware_embedding'] = context_aware_embedding
            else:
                print("Warning: Embedding dimension/type mismatch between current and previous sentence. Skipping context blending.")
        return base_interpretation

    # --- NEW: Semantic Frame Extraction Functions (from quantum_kernel.py) ---
    # [extract_semantic_frames, extract_enhanced_semantic_frames]
    # (These functions remain the same as in the previous version)
    # --- (Code for these functions omitted for brevity, assume they are correct from previous version) ---
    def extract_semantic_frames(self, tokens, analyses, roles):
        """ Extract basic semantic frames based on verb class and potentially external tools. """
        sentence = ' '.join(tokens)
        frames = []
        verb_idx = roles.get('verb')
        verb = None; verb_lemma = None
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
                elif verb_class == "COMMUNICATION": frame.update({'speaker': subj_token, 'message': obj_token})
                elif verb_class == "COGNITION": frame.update({'thinker': subj_token, 'thought': obj_token})
                elif verb_class == "EMOTION": frame.update({'experiencer': subj_token, 'stimulus': obj_token})
                elif verb_class == "CREATION": frame.update({'creator': subj_token, 'created': obj_token})
                elif verb_class == "PERCEPTION": frame.update({'perceiver': subj_token, 'perceived': obj_token})
                frames.append(frame)
        if self.camel_analyzer:
            try:
                camel_analysis = self.camel_analyzer.analyze(sentence)
                semantic_properties = {'tense': None, 'mood': None, 'aspect': None, 'definiteness': [], 'gender': {}, 'number': {}}
                for i, token_analysis_list in enumerate(camel_analysis):
                    if token_analysis_list:
                        token_analysis = token_analysis_list[0]
                        if 'asp' in token_analysis: semantic_properties['aspect'] = token_analysis['asp']
                        if 'mod' in token_analysis: semantic_properties['mood'] = token_analysis['mod']
                        if 'gen' in token_analysis: semantic_properties['gender'][tokens[i]] = token_analysis['gen']
                        if 'num' in token_analysis: semantic_properties['number'][tokens[i]] = token_analysis['num']
                        if 'def' in token_analysis and token_analysis['def'] == 'd': semantic_properties['definiteness'].append(tokens[i])
                if verb_idx is not None and verb_idx < len(camel_analysis) and camel_analysis[verb_idx]:
                    verb_analysis = camel_analysis[verb_idx][0]
                    if 'asp' in verb_analysis:
                        if verb_analysis['asp'] == 'p': semantic_properties['tense'] = 'past'
                        elif verb_analysis['asp'] == 'i': semantic_properties['tense'] = 'present'
                        elif verb_analysis['asp'] == 'c': semantic_properties['tense'] = 'imperative'
                frames.append({'type': 'SEMANTIC_PROPERTIES', 'properties': semantic_properties})
            except Exception as e: print(f"Warning: Error in CAMeL semantic property extraction: {e}")
        if FARASA_NER:
            try:
                entities = FARASA_NER.recognize(sentence)
                if entities: frames.append({'type': 'NAMED_ENTITIES', 'entities': entities})
            except Exception as e: print(f"Warning: Error using Farasa NER: {e}")
        if ARAVEC_MODEL:
            try:
                similar_words = {}
                for i, (lemma, pos, _, _) in enumerate(analyses):
                    if pos in ['NOUN', 'VERB', 'ADJ'] and lemma in ARAVEC_MODEL.wv:
                        similar = ARAVEC_MODEL.wv.most_similar(lemma, topn=3)
                        similar_words[lemma] = [word for word, _ in similar]
                if similar_words: frames.append({'type': 'WORD_EMBEDDINGS', 'similar_words': similar_words})
            except Exception as e: print(f"Warning: Error using AraVec model: {e}")
        return {'sentence': sentence, 'frames': frames}

    def extract_enhanced_semantic_frames(self, tokens, analyses, roles):
        """More comprehensive semantic frame extraction including rhetorical relations, etc."""
        basic_frames_data = self.extract_semantic_frames(tokens, analyses, roles)
        frames = basic_frames_data['frames'].copy()
        sentence = basic_frames_data['sentence']
        rhetorical_markers = {
            'CAUSE': ['لأن', 'بسبب', 'نتيجة', 'لذلك'], 'CONTRAST': ['لكن', 'بينما', 'رغم', 'مع ذلك', 'على الرغم', 'إلا أن', 'غير أن'],
            'CONDITION': ['إذا', 'لو', 'إن', 'شرط', 'في حال'], 'TEMPORAL': ['قبل', 'بعد', 'أثناء', 'خلال', 'عندما', 'حين'],
            'PURPOSE': ['كي', 'ل', 'من أجل', 'بهدف', 'حتى'], 'CONJUNCTION': ['و', 'ف', 'ثم']
        }
        for rel_type, markers in rhetorical_markers.items():
            for i, token in enumerate(tokens):
                if token in markers:
                    span1_tokens = tokens[:i]; span2_tokens = tokens[i+1:]
                    if span1_tokens and span2_tokens:
                        frames.append({'type': f'RHETORICAL_{rel_type}', 'marker': token, 'span1': ' '.join(span1_tokens), 'span2': ' '.join(span2_tokens)})
        verb_indices = [i for i, (_, pos, _, _) in enumerate(analyses) if pos == 'VERB']
        if len(verb_indices) > 1:
            root_idx = roles.get('root'); main_verb_idx = roles.get('verb', root_idx)
            if main_verb_idx is not None:
                 subordinate_clauses = []
                 for i, (_, pos, dep, head) in enumerate(analyses):
                      if pos == 'VERB' and i != main_verb_idx and (dep in ['ccomp', 'xcomp', 'advcl'] or head == main_verb_idx):
                           clause_span = ' '.join(tokens[i:])
                           subordinate_clauses.append({'verb': tokens[i], 'span': clause_span, 'index': i, 'relation_to_main': dep})
                 if subordinate_clauses:
                      frames.append({'type': 'NESTED_PREDICATION', 'main_verb': tokens[main_verb_idx] if main_verb_idx < len(tokens) else '?', 'main_index': main_verb_idx, 'subordinate_clauses': subordinate_clauses})
        pronouns = ['هو', 'هي', 'هم', 'هن', 'ه', 'ها', 'هما', 'ك', 'كم', 'نا']
        pronoun_indices = [(i, token) for i, token in enumerate(tokens) if token in pronouns]
        coref_chains = []; processed_pronouns = set()
        for i, (pronoun_idx, pronoun) in enumerate(pronoun_indices):
            if pronoun_idx in processed_pronouns: continue
            potential_antecedents = []
            for k in range(pronoun_idx - 1, -1, -1):
                if k < len(analyses):
                    lemma, pos, _, _ = analyses[k]
                    if pos in ['NOUN', 'PROPN']:
                        potential_antecedents.append({'text': tokens[k], 'index': k, 'lemma': lemma})
                        break
            if potential_antecedents:
                antecedent = potential_antecedents[0]
                coref_chains.append({'pronoun': pronoun, 'pronoun_index': pronoun_idx, 'antecedent': antecedent['text'], 'antecedent_index': antecedent['index']})
                processed_pronouns.add(pronoun_idx)
        if coref_chains: frames.append({'type': 'COREFERENCE', 'chains': coref_chains})
        return {'sentence': sentence, 'frames': frames}

    # --- END NEW FUNCTIONS ---


    def interpret_sentence(self, circuit, tokens: List[str], analyses: List[Tuple], structure: str, roles: Dict, previous_analyses=None) -> Dict:
        """ Interpret sentence meaning based on circuit, linguistics, and optionally context. """
        # --- Context Handling ---
        if previous_analyses is not None:
             # Redirect to context-aware function to avoid redundant feature extraction if called from pipeline loop
             return self.analyze_sentence_in_context(circuit, tokens, analyses, structure, roles, previous_analyses)

        # --- Direct Interpretation (No Context or Base Interpretation) ---

        # *** Add check here to ensure circuit is the correct type ***
        if not isinstance(circuit, QuantumCircuit):
            print(f"  ERROR in interpret_sentence: Expected QuantumCircuit, got {type(circuit)}. Cannot proceed.")
            # Return a minimal error dictionary
            return {
                'sentence': ' '.join(tokens), 'structure': structure, 'embedding': None,
                'interpretation': 'Error: Invalid circuit type received.', 'meaning_options': [],
                'specific_interpretation': None, 'semantic_frames': [], 'discourse_relations': [],
                'enhanced_linguistic_analysis': analyses, 'roles': roles,
                'morphological_analysis': None, 'sentiment': None, 'named_entities': [],
                'confidence': 0.0, 'error': f'Invalid circuit type: {type(circuit)}'
            }

        quantum_features = self.get_enhanced_circuit_features(circuit) # This should now work
        enhanced_analyses = analyses
        camel_morphology = None; sentiment_score = None; named_entities = []
        if self.camel_analyzer:
             try:
                 sentence_text = ' '.join(tokens)
                 camel_morphology = self.camel_analyzer.analyze(sentence_text)
                 # Add optional sentiment/NER calls here
             except Exception as camel_e: pass

        linguistic_features = self.extract_complex_linguistic_features(tokens, enhanced_analyses, structure, roles)
        embedding = self.combine_features_with_attention(quantum_features, linguistic_features, structure)

        # --- Extract Semantic Frames ---
        try:
            semantic_frames_data = self.extract_enhanced_semantic_frames(tokens, enhanced_analyses, roles)
            extracted_frames = semantic_frames_data.get('frames', [])
        except Exception as frame_e:
            print(f"    Error extracting semantic frames: {frame_e}")
            extracted_frames = []

        # --- Prepare base result structure ---
        result = {
            'sentence': ' '.join(tokens), 'structure': structure, 'embedding': embedding,
            'interpretation': None, 'meaning_options': [], 'specific_interpretation': None,
            'semantic_frames': extracted_frames, # Store extracted frames
            'discourse_relations': [],
            'enhanced_linguistic_analysis': enhanced_analyses, 'roles': roles,
            'morphological_analysis': camel_morphology, 'sentiment': sentiment_score, 'named_entities': named_entities,
            'confidence': 0.0
        }

        # --- Find meaning options based on clusters ---
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
                 template_dict = top_cluster_info.get('original_templates', self.semantic_templates.get(structure, self.semantic_templates['OTHER']))
                 result['specific_interpretation'] = self.create_specific_interpretation(tokens, enhanced_analyses, roles, structure, template_dict)
                 result['interpretation'] = top_cluster_info.get('deduced_template', 'N/A')
        else:
            # Fallback if no clusters
            templates = self.semantic_templates.get(structure, self.semantic_templates['OTHER'])
            result['specific_interpretation'] = self.create_specific_interpretation(tokens, enhanced_analyses, roles, structure, templates)
            result['interpretation'] = result['specific_interpretation']['templates'].get('declarative', 'N/A')

        return result


    # --- Reporting ---
    # [format_discourse_relations, generate_discourse_report, generate_html_report]
    # (These functions remain the same as in the previous version)
     # --- (Code for these functions omitted for brevity, assume they are correct from previous version) ---
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
            report += "**Basic Analysis:**\n"; report += f"- Structure: {interpretation_data.get('structure', 'N/A')}\n"
            report += f"- Confidence (Cluster Similarity): {interpretation_data.get('confidence', 0.0):.2f}\n"
            if interpretation_data.get('sentiment') is not None: report += f"- Sentiment: {interpretation_data['sentiment']}\n"
            if interpretation_data.get('named_entities'): report += f"- Named Entities: `{interpretation_data['named_entities']}`\n"
            report += "\n"; report += "**Interpretation:**\n"
            specific_interp = interpretation_data.get('specific_interpretation')
            if specific_interp and isinstance(specific_interp, dict):
                main_interp_text = specific_interp.get('templates', {}).get('declarative', 'N/A')
                report += f"- Deduced Meaning: {main_interp_text}\n"
                sem_details = specific_interp.get('semantic_details', {})
                if sem_details:
                     verb_info = sem_details.get('verb', {}); report += f"  - Verb: `{verb_info.get('text', '-')}` (Lemma: `{verb_info.get('lemma', '-')}`, Tense: {verb_info.get('tense', '-')})\n"
                     report += f"  - Subject: `{sem_details.get('subject', {}).get('text', '-')}`\n"; report += f"  - Object: `{sem_details.get('object', {}).get('text', '-')}`\n"
            else: report += f"- General Interpretation: {interpretation_data.get('interpretation', 'N/A')}\n"
            report += "\n"; report += "**Semantic Frames:**\n"
            frames = interpretation_data.get('semantic_frames', [])
            if frames:
                for frame in frames:
                     frame_type = frame.get('type', 'UNKNOWN_FRAME'); details = ', '.join([f"{k}: {v}" for k, v in frame.items() if k != 'type'])
                     report += f"- Type: **{frame_type}**\n"
                     if frame_type == 'SEMANTIC_PROPERTIES': props = frame.get('properties', {}); report += f"  - Properties: Tense=`{props.get('tense')}`, Aspect=`{props.get('aspect')}`, Mood=`{props.get('mood')}`\n"
                     elif frame_type == 'NAMED_ENTITIES': ents = frame.get('entities', []); report += f"  - Entities: `{ents}`\n"
                     elif frame_type == 'COREFERENCE': chains = frame.get('chains', []); report += f"  - Coref Chains: `{chains}`\n"
                     elif frame_type.startswith('RHETORICAL_'): report += f"  - Marker: `{frame.get('marker')}`, Span1: `{frame.get('span1')}`, Span2: `{frame.get('span2')}`\n"
                     else: report += f"  - Details: {details}\n"
            else: report += "- No specific semantic frames extracted.\n"
            report += "\n"
            if i > 0:
                report += "**Relationship to Previous Sentence:**\n"; discourse_relations = interpretation_data.get('discourse_relations', [])
                relation_text = self.format_discourse_relations(discourse_relations); report += f"{relation_text}\n\n"
            else: report += "**Relationship to Previous Sentence:**\n- N/A (First Sentence)\n\n"
            report += "---\n\n"
        return report

    def generate_html_report(self, discourse_analyses):
        """ Generate an HTML report with discourse analysis details. """
        html = """
        <!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><title>تحليل الخطاب الكمي للغة العربية (محسن)</title><style>
        body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:20px;line-height:1.6;background-color:#f9f9f9;color:#333}
        h1{color:#0056b3;text-align:center;border-bottom:2px solid #0056b3;padding-bottom:10px}
        .sentence-block{background-color:#fff;margin-bottom:25px;border:1px solid #ddd;border-radius:8px;padding:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
        .sentence-text{font-size:1.3em;font-weight:bold;color:#003d7a;margin-bottom:15px}
        .analysis-section{margin-top:15px;padding-top:15px;border-top:1px dashed #eee}
        .analysis-section h3{color:#0056b3;font-size:1.1em;margin-bottom:8px}
        .analysis-detail{margin-left:20px;margin-bottom:5px;font-size:0.95em}
        .detail-label{font-weight:bold;color:#555}
        .relation{color:#006699;margin-top:8px;background-color:#e7f3fe;padding:8px;border-radius:4px;border-left:3px solid #006699}
        .relation-marker{font-weight:bold;color:#cc0000}
        .no-relation{color:#777;font-style:italic;margin-top:8px}
        .interpretation{background-color:#f0f0f0;padding:10px;border-radius:5px;margin-top:10px}
        .frames{background-color:#e8f5e9;padding:10px;border-radius:5px;margin-top:10px;border-left:3px solid #4caf50}
        .frame-item{margin-bottom:8px}
        .frame-type{font-weight:bold;color:#2e7d32}
        code{background-color:#eee;padding:2px 4px;border-radius:3px;font-family:monospace}
        .error{color:red;font-weight:bold}
        </style></head><body><h1>تحليل الخطاب الكمي للنص العربي (محسن)</h1>"""
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis.get('sentence', 'N/A'); interpretation_data = analysis.get('interpretation', {})
            html += f'<div class="sentence-block"><div class="sentence-text">الجملة {i+1}: {sentence}</div>'
            if interpretation_data.get('error'):
                 html += f'<div class="analysis-section error"><h3>خطأ في التحليل:</h3><div class="analysis-detail">{interpretation_data["error"]}</div></div></div>'; continue
            html += '<div class="analysis-section"><h3>التحليل الأساسي:</h3>'
            structure = interpretation_data.get('structure', 'غير محدد'); confidence = interpretation_data.get('confidence', 0.0)
            html += f'<div class="analysis-detail"><span class="detail-label">البنية:</span> {structure}</div>'
            html += f'<div class="analysis-detail"><span class="detail-label">الثقة (التشابه مع العنقود):</span> {confidence:.2f}</div>'
            if interpretation_data.get('sentiment') is not None: html += f'<div class="analysis-detail"><span class="detail-label">الشعور:</span> {interpretation_data["sentiment"]}</div>'
            if interpretation_data.get('named_entities'): ne_html = ", ".join([f"<code>{token} ({tag})</code>" for token, tag in interpretation_data["named_entities"]]); html += f'<div class="analysis-detail"><span class="detail-label">الكيانات المسماة:</span> {ne_html}</div>'
            html += '</div>'
            specific_interp = interpretation_data.get('specific_interpretation')
            if specific_interp and isinstance(specific_interp, dict):
                html += '<div class="analysis-section interpretation"><h3>التفسير المحدد:</h3>'
                main_interp_text = specific_interp.get('templates', {}).get('declarative', 'لا يوجد تفسير محدد.'); html += f'<div class="analysis-detail">{main_interp_text}</div>'
                sem_details = specific_interp.get('semantic_details', {})
                if sem_details:
                     verb_info = sem_details.get('verb', {}); subj_info = sem_details.get('subject', {}); obj_info = sem_details.get('object', {})
                     html += f'<div class="analysis-detail" style="font-size: 0.9em; color: #444;"> <span class="detail-label">الفعل:</span> <code>{verb_info.get("text", "-")}</code> (Lemma: <code>{verb_info.get("lemma", "-")}</code>, الزمن: {verb_info.get("tense", "-")}) | <span class="detail-label">الفاعل:</span> <code>{subj_info.get("text", "-")}</code> | <span class="detail-label">المفعول:</span> <code>{obj_info.get("text", "-")}</code></div>'
                html += '</div>'
            else:
                 general_interp = interpretation_data.get('interpretation', 'لا يوجد تفسير.'); html += f'<div class="analysis-section interpretation"><h3>التفسير العام:</h3><div class="analysis-detail">{general_interp}</div></div>'
            frames = interpretation_data.get('semantic_frames', [])
            if frames:
                html += '<div class="analysis-section frames"><h3>الإطارات الدلالية:</h3>'
                for frame in frames:
                    frame_type = frame.get('type', 'UNKNOWN_FRAME'); html += f'<div class="frame-item"><span class="frame-type">{frame_type}:</span> '
                    details = []
                    for k, v in frame.items():
                        if k != 'type': v_str = str(v); v_str = v_str[:100] + "..." if len(v_str) > 100 else v_str; details.append(f'<span class="detail-label">{k}=</span><code>{v_str}</code>')
                    html += ", ".join(details); html += '</div>'
                html += '</div>'
            if i > 0:
                html += '<div class="analysis-section"><h3>العلاقة مع الجملة السابقة:</h3>'
                discourse_relations = interpretation_data.get('discourse_relations', []); relation_text = self.format_discourse_relations(discourse_relations)
                if discourse_relations:
                     for line in relation_text.split('\n'): line_html = line.replace("'", "<code>").replace("'", "</code>", 1); html += f'<div class="relation">{line_html}</div>'
                else: html += f'<div class="no-relation">{relation_text}</div>'
                html += '</div>'
            html += '</div>'
        html += """</body></html>"""
        return html

    # --- Utility Methods ---
    # [save_model, load_model, visualize_meaning_space]
    # [analyze_quantum_states]
    # (These functions remain the same as in the previous version)
    # --- (Code for these functions omitted for brevity, assume they are correct from previous version) ---
    def save_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Saves the trained kernel state to a file using pickle. """
        print(f"Saving model to {filename}...")
        analyzer_state = None
        if hasattr(self.camel_analyzer, '__getstate__'):
             analyzer_state = self.camel_analyzer.__getstate__()
        elif self.camel_analyzer is not None:
             print("Warning: CAMeL Analyzer might not be pickleable. Saving without it.")
        model_data = {
            'embedding_dim': self.embedding_dim, 'num_clusters': self.num_clusters, 'meaning_clusters': self.meaning_clusters,
            'cluster_labels': self.cluster_labels, 'meaning_map': self.meaning_map, 'reference_sentences': self.reference_sentences,
            'circuit_embeddings': self.circuit_embeddings, 'sentence_embeddings': self.sentence_embeddings,
            'semantic_templates': self.semantic_templates, #'camel_analyzer_state': analyzer_state
        }
        try:
            with open(filename, 'wb') as f: pickle.dump(model_data, f)
            print(f"Model saved successfully.")
        except Exception as e: print(f"Error saving model: {e}"); traceback.print_exc()

    def load_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """ Loads a trained kernel state from a file. """
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
            self.__init__(self.embedding_dim, self.num_clusters, self.simulator.name) # Re-run init to get analyzer
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


# Example pipeline function (modified to include new options)
# Required imports (ensure these are at the top of your v4.py file)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.visualization import plot_state_city
from qiskit.quantum_info import partial_trace
import pickle
import os
import traceback
from collections import Counter
from camel_test2 import arabic_to_quantum_enhanced # Assumes this is importable

# --- Required Lambeq imports for conversion attempt ---
try:
    from lambeq import IQPAnsatz, AtomicType
    # Explicitly import the Diagram types for checking
    from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram
    from lambeq.backend.grammar import Diagram as LambeqGrammarDiagram
    LAMBEQ_AVAILABLE = True
except ImportError:
    print("ERROR: Lambeq library not found. Cannot perform conversion in v4.py. Please install lambeq.")
    LAMBEQ_AVAILABLE = False
# --- GENSIM imports (if used for enhanced clustering) ---
try:
    from gensim import corpora, models
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    # print("Warning: gensim not found. Enhanced clustering with LDA disabled.") # Optional warning

# This function assumes the ArabicQuantumMeaningKernel class is defined above it in the file

def prepare_quantum_nlp_pipeline(max_sentences=20, use_enhanced_clustering=True):
    """
    Example of how to use the enhanced ArabicQuantumMeaningKernel.
    (Modified to cleanly handle None return from camel_test.py)
    """
    if not LAMBEQ_AVAILABLE: # Ensure Lambeq check remains
        print("Exiting pipeline because Lambeq library is required but not found.")
        return None, None, []

    sentence_file = "sentences.txt"
    # ... (rest of sentence loading logic) ...
    sentences = []
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
    print("\n--- Processing Sentences ---")
    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx+1}/{len(sentences)}: '{sentence[:40]}...'")
        try:
            # Call the function from camel_test.py (which now returns None first on failure)
            returned_values = arabic_to_quantum_enhanced(sentence, debug=False)

            # --- REVISED UNPACKING LOGIC ---
            circuit = None # Initialize circuit variable
            grammar_diagram = None # To store the grammar diagram

            if isinstance(returned_values, tuple) and len(returned_values) >= 6:
                item0 = returned_values[0]
                item1 = returned_values[1]
                item0_type = type(item0)
                # item1_type = type(item1) # We might not need item1_type check now

                # Unpack remaining values first
                structure, tokens, analyses, roles = returned_values[2:6]

                # --- Explicitly check for None first ---
                if item0 is None:
                    # Case 0: camel_test.py signaled conversion failure
                    print(f"INFO: Sentence {idx+1} - Circuit generation failed in camel_test. Skipping.")
                    # circuit remains None, loop will continue
                elif item0_type is QuantumCircuit:
                    # Case 1: Successful return (Circuit first)
                    print(f"DEBUG: Sentence {idx+1} - Received QuantumCircuit first.")
                    circuit = item0
                    grammar_diagram = item1 if isinstance(item1, LambeqGrammarDiagram) else None
                # --- Remove/Comment out previous workarounds if camel_test_fix is applied ---
                # elif item0_type is LambeqQuantumDiagram:
                #     # Case 2: No longer expected if camel_test_fix works (returns None on failure)
                #     print(f"WARNING: Sentence {idx+1} - Received LambeqQuantumDiagram first (unexpected if camel_test_fix applied). Attempting conversion...")
                #     # ... (conversion logic from v4_pipeline_fix_ty_qubit) ...
                # elif item1_type is QuantumCircuit:
                #      # Case 3: No longer expected if camel_test_fix works
                #      print(f"WARNING: Sentence {idx+1} - Received Diagram first, Circuit second (unexpected if camel_test_fix applied).")
                #      # ... (unpacking logic) ...
                # --- End Remove/Comment ---
                else:
                    # Case 4: Unexpected return type (neither None nor QuantumCircuit first)
                    print(f"  ERROR: Unexpected return type from camel_test. Item 0: {item0_type}. Skipping.")
                    # circuit remains None, loop will continue

            else:
                # Handle cases where the return is not a tuple or incorrect length
                print(f"  ERROR: Unexpected return value or length from arabic_to_quantum_enhanced: {returned_values}")
                # circuit remains None, loop will continue

            # --- Check if we have a valid circuit before proceeding ---
            if circuit is None or not isinstance(circuit, QuantumCircuit):
                # This check is now the main way to skip failed sentences
                # print(f"  INFO: Skipping sentence {idx+1} due to missing or invalid QuantumCircuit.") # Optional print
                continue # Skip to the next sentence

            # --- Store results only if circuit is valid ---
            results.append({
                'sentence': sentence,
                'circuit': circuit, # Store the actual Qiskit circuit
                'structure': structure,
                'tokens': tokens,
                'analyses': analyses,
                'roles': roles,
                'original_index': idx,
                'diagram': grammar_diagram # Store the grammar diagram (might be None)
            })
            circuits_for_viz[idx] = circuit
            print(f"  Sentence {idx+1}: Processed successfully. Structure: {structure}, Circuit type: {type(circuit)}")

        except Exception as e:
            # Catch any other unexpected errors during processing
            print(f"  ERROR processing sentence {idx+1}: {sentence}")
            print(f"  Actual Error Type: {type(e).__name__}, Message: {e}")
            traceback.print_exc()
            print("  Skipping this sentence.")
        # --- END REVISED UNPACKING LOGIC ---

    # --- Check if any sentences were processed ---
    if not results:
        print("\nError: No sentences were processed successfully. Exiting pipeline.")
        return None, None, []
    # ---

    # --- Kernel Instantiation and Training (No changes needed) ---
    num_processed = len(results)
    kernel_clusters = min(5, max(1, num_processed // 2))
    kernel = ArabicQuantumMeaningKernel(embedding_dim=14, num_clusters=kernel_clusters)

    print(f"\n--- Training Kernel ({kernel.num_clusters} clusters) ---")
    kernel.train(
        sentences=[r['sentence'] for r in results],
        circuits=[r['circuit'] for r in results],
        tokens_list=[r['tokens'] for r in results],
        analyses_list=[r['analyses'] for r in results],
        structures=[r['structure'] for r in results],
        roles_list=[r['roles'] for r in results],
        use_enhanced_clustering=use_enhanced_clustering
    )

    # --- Save Model, Visualizations, Discourse Analysis, Reports, New Sentence (No changes needed) ---
    # ... (rest of the function remains the same as in v4_pipeline_fix_ty_qubit) ...
    kernel.save_model('arabic_quantum_kernel_enhanced.pkl')
    print("\n--- Visualizing Meaning Space ---")
    try:
        meaning_space_fig = kernel.visualize_meaning_space(save_path='meaning_space_enhanced.png')
        if meaning_space_fig: plt.close(meaning_space_fig)
    except Exception as vis_e: print(f"Error during meaning space visualization: {vis_e}")
    print("\n--- Visualizing Quantum States ---")
    try:
        state_viz_fig = kernel.analyze_quantum_states(circuits_for_viz, save_path_prefix="state_viz_")
        if state_viz_fig: plt.close(state_viz_fig)
    except Exception as state_vis_e: print(f"Error during quantum state visualization: {state_vis_e}")
    print("\n--- Performing Discourse Analysis ---")
    discourse_analyses = []
    previous_analysis_dict = None
    for i, result in enumerate(results):
        print(f"Analyzing sentence {i+1}/{len(results)} in context...")
        try:
            interpretation = kernel.interpret_sentence(result['circuit'], result['tokens'], result['analyses'], result['structure'], result['roles'], previous_analyses=previous_analysis_dict)
            current_analysis_dict = {**result, 'interpretation': interpretation}
            discourse_analyses.append(current_analysis_dict)
            previous_analysis_dict = current_analysis_dict
            print(f"  Sentence {i+1}: Analysis complete.")
        except Exception as analysis_e: print(f"  Error during sentence analysis {i+1}: {analysis_e}"); traceback.print_exc(); discourse_analyses.append({**result, 'interpretation': {'error': str(analysis_e)}})
    print("\n--- Generating Reports ---")
    try:
        html_report = kernel.generate_html_report(discourse_analyses)
        report_filename = 'discourse_analysis_report_enhanced.html'
        with open(report_filename, 'w', encoding='utf-8') as f: f.write(html_report)
        print(f"HTML report saved to {report_filename}")
    except Exception as report_e: print(f"Error generating HTML report: {report_e}")
    try:
        md_report = kernel.generate_discourse_report(discourse_analyses)
        md_report_filename = 'discourse_analysis_report_enhanced.md'
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
        if isinstance(returned_new, tuple) and len(returned_new) >= 6:
             item0_new = returned_new[0]
             new_structure, new_tokens, new_analyses, new_roles = returned_new[2:6]
             if item0_new is None: print("ERROR: Circuit generation failed for new sentence in camel_test.")
             elif isinstance(item0_new, QuantumCircuit): new_circuit_obj = item0_new
             else: print(f"ERROR: Unexpected type {type(item0_new)} returned first for new sentence.")

             if isinstance(new_circuit_obj, QuantumCircuit):
                  last_interpretation_result = kernel.interpret_sentence(new_circuit_obj, new_tokens, new_analyses, new_structure, new_roles, previous_analyses=previous_analysis_dict)
                  print("New sentence interpretation complete.")
             else: print("ERROR: Failed to get valid QuantumCircuit for new sentence.")
        else: print("ERROR: Unexpected return from arabic_to_quantum_enhanced for new sentence.")
    except Exception as interp_e: print(f"Error interpreting new sentence: {interp_e}"); traceback.print_exc()

    print("\n--- Pipeline Finished ---")
    return kernel, last_interpretation_result, discourse_analyses



# Main execution block
if __name__ == "__main__":
    # Run the pipeline
    kernel, last_new_sentence_interpretation, discourse_analyses_list = prepare_quantum_nlp_pipeline(
        max_sentences=20, # Process 20 sentences from the file
        use_enhanced_clustering=True # Try using LDA clustering
    )

    # Check if pipeline ran successfully
    if kernel is None:
        print("\nPipeline execution failed.")
    else:
        print("\n--- Summary of Last New Sentence Interpretation ---")
        if last_new_sentence_interpretation is not None:
            # Check for error during interpretation
            if last_new_sentence_interpretation.get('error'):
                 print(f"ERROR interpreting new sentence: {last_new_sentence_interpretation['error']}")
            else:
                print(f"Sentence: {last_new_sentence_interpretation.get('sentence', 'N/A')}")
                print(f"Structure: {last_new_sentence_interpretation.get('structure', 'N/A')}")
                print(f"Confidence: {last_new_sentence_interpretation.get('confidence', 'N/A')}")
                print(f"Deduced Meaning: {last_new_sentence_interpretation.get('interpretation', 'N/A')}")

                # Print semantic frames for the new sentence
                print("\nSemantic Frames (New Sentence):")
                frames = last_new_sentence_interpretation.get('semantic_frames', [])
                if frames:
                    for frame in frames:
                        print(f"  - Type: {frame.get('type', 'N/A')}, Details: { {k:v for k,v in frame.items() if k!='type'} }")
                else:
                    print("  - None extracted.")

                # Print discourse relations for the new sentence
                print("\nDiscourse Relations (New Sentence):")
                relations = last_new_sentence_interpretation.get('discourse_relations', [])
                if relations:
                     print(f"  - {kernel.format_discourse_relations(relations)}")
                else:
                     print("  - None detected.")

        else:
            print("No interpretation available for the new sentence (it might have failed).")

        # You can add more summary prints for the discourse_analyses_list if needed
        # print("\n--- Full Discourse Analysis Summary ---")
        # ... (loop through discourse_analyses_list and print details) ...
