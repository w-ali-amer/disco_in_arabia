import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from qiskit import Aer, execute, QuantumCircuit
from qiskit.visualization import plot_state_city
from lambeq import AtomicType, IQPAnsatz
import pickle
import os
from collections import Counter
from camel_test import arabic_to_quantum_enhanced



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
# --- End FIX ---

class ArabicQuantumMeaningKernel:
    """
    A quantum kernel for mapping quantum circuit outputs to potential
    sentence meanings for Arabic language processing.
    
    This kernel works with the output of DisCoCat diagrams converted to
    quantum circuits and helps identify semantic patterns and meanings.
    """
    
    def __init__(self,
                 embedding_dim: int = 14, # Increased default based on complex features
                 num_clusters: int = 5,
                 simulator_backend: str = 'statevector_simulator'):
        """
        Initialize the quantum meaning kernel.
        # ... (rest of docstring)
        """
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.simulator = Aer.get_backend(simulator_backend)
        self.meaning_clusters = None
        self.cluster_labels = None
        self.meaning_map = {}
        self.reference_sentences = {}
        self.circuit_embeddings = {}
        self.sentence_embeddings = {}
        # CAMeL Tools Analyzer (optional, initialize if needed)
        self.camel_analyzer = None
        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            db = MorphologyDB.builtin_db()
            self.camel_analyzer = Analyzer(db)
            print("CAMeL Tools Analyzer initialized.")
        except ImportError:
            print("Warning: CAMeL Tools not found. Some NLP enhancements will be disabled.")
        except Exception as e:
            print(f"Warning: Error initializing CAMeL Tools Analyzer: {e}")

        # Semantic templates (can be expanded)
        self.semantic_templates = {
            'VSO': {
                'declarative': "ACTION performed by SUBJECT on OBJECT",
                'question': "Did SUBJECT perform ACTION on OBJECT?",
                'command': "SUBJECT should perform ACTION on OBJECT"
            },
            'SVO': {
                'declarative': "SUBJECT performs ACTION on OBJECT",
                'question': "Does SUBJECT perform ACTION on OBJECT?",
                'command': "Make SUBJECT perform ACTION on OBJECT"
            },
            'NOMINAL': {
                'declarative': "SUBJECT is PREDICATE",
                'question': "Is SUBJECT PREDICATE?",
                'command': "Consider SUBJECT as PREDICATE"
            },
            'COMPLEX': { # Added for complex sentences
                 'declarative': "Complex statement involving CLAUSE_1 and CLAUSE_2",
                 'question': "Question about relationship between CLAUSE_1 and CLAUSE_2",
                 'command': "Directive concerning CLAUSE_1 and CLAUSE_2"
            },
            'OTHER': {
                'declarative': "General statement about TOPIC",
                'question': "Question about TOPIC",
                'command': "Directive related to TOPIC"
            }
        }
    
    def get_circuit_features(self, circuit, shots: int = 1024) -> np.ndarray:
        """
        Extract features from a Qiskit quantum circuit by executing it.
        (Input 'circuit' is now expected to be a Qiskit QuantumCircuit)
        """
        print(f"\n--- Debugging get_circuit_features ---") # DEBUG
        # --- Input is now expected to be a Qiskit circuit ---
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_circuit_features received non-Qiskit object: {type(circuit)}")
             # Return fallback random features
             fallback = np.random.rand(self.embedding_dim)
             norm = np.linalg.norm(fallback)
             if norm > 0: fallback = fallback / norm
             return fallback

        print(f"Input type: {type(circuit)}") # DEBUG

        try:
            # Ensure the Qiskit circuit has a name
            circuit = _ensure_circuit_name(circuit, "circuit_for_features")
            print(f"Qiskit circuit name: {getattr(circuit, 'name', 'N/A')}") # DEBUG

            print("Executing Qiskit circuit...") # DEBUG
            # Execute the Qiskit circuit directly
            result = execute(circuit, self.simulator).result()
            print(f"Execution successful. Result type: {type(result)}") # DEBUG

            # (Statevector/Counts extraction logic remains the same)
            print(f"Checking for statevector...") # DEBUG
            if hasattr(result, 'get_statevector'):
                print("Statevector found.") # DEBUG
                statevector = result.get_statevector()
                amplitudes = np.abs(statevector)
                phases = np.angle(statevector)
                half_dim = self.embedding_dim // 2
                amp_slice = amplitudes[:min(half_dim, len(amplitudes))]
                phase_slice = phases[:min(half_dim, len(phases))]
                features = np.concatenate([amp_slice, phase_slice])
                if len(features) < self.embedding_dim: features = np.pad(features, (0, self.embedding_dim - len(features)), 'constant')
                elif len(features) > self.embedding_dim: features = features[:self.embedding_dim]
                norm = np.linalg.norm(features)
                if norm > 0: features = features / norm
                print(f"Returning statevector-based features. Shape: {features.shape}") # DEBUG
                return features
            else:
                 print("Statevector not found. Falling back to counts.") # DEBUG
                 counts = result.get_counts()
                 total_shots = sum(counts.values())
                 feature_vector = np.zeros(self.embedding_dim)
                 num_outcomes = 0
                 if total_shots > 0:
                     for outcome, count in counts.items():
                         num_outcomes += 1
                         try:
                             idx_base = int(outcome, 2) if isinstance(outcome, str) and all(c in '01' for c in outcome) else hash(outcome)
                             idx = idx_base % self.embedding_dim
                             feature_vector[idx] += count / total_shots
                         except Exception as e_idx: print(f"  Warning: Could not process outcome '{outcome}': {e_idx}") # DEBUG
                 print(f"Processed {num_outcomes} outcomes.") # DEBUG
                 norm = np.linalg.norm(feature_vector)
                 if norm > 0: feature_vector = feature_vector / norm
                 print(f"Returning counts-based features. Shape: {feature_vector.shape}") # DEBUG
                 return feature_vector

        except Exception as e:
            print(f"\n--- ERROR in get_circuit_features ---") # DEBUG
            print(f"Error type: {type(e)}") # DEBUG
            print(f"Error message: {e}") # DEBUG
            traceback.print_exc() # DEBUG
            print(f"---------------------------------------\n") # DEBUG

        # Fallback for any error during execution
        print("Returning fallback random features due to error.") # DEBUG
        fallback = np.random.rand(self.embedding_dim)
        norm = np.linalg.norm(fallback)
        if norm > 0: fallback = fallback / norm
        return fallback
    
    def extract_linguistic_features(self, tokens: List[str], 
                                  analyses: List[Tuple], 
                                  structure: str, 
                                  roles: Dict) -> np.ndarray:
        """
        Extract linguistic features from the sentence analysis.
        
        Args:
            tokens: List of tokens in the sentence
            analyses: List of (lemma, pos, dep_type, head_idx) tuples
            structure: The sentence structure (VSO, SVO, NOMINAL, etc.)
            roles: Dictionary with grammatical roles
            
        Returns:
            features: A vector of linguistic features
        """
        # Initialize feature vector
        features = np.zeros(self.embedding_dim)
        
        # Record structure type as a feature
        structure_map = {'VSO': 0, 'SVO': 1, 'NOMINAL': 2, 'OTHER': 3}
        structure_idx = structure_map.get(structure, 3)
        features[0] = structure_idx / 3  # Normalize
        
        # Count POS tags as features
        pos_counts = {'VERB': 0, 'NOUN': 0, 'ADJ': 0, 'ADV': 0, 'DET': 0, 'PRON': 0, 'ADP': 0}
        for _, pos, _, _ in analyses:
            if pos in pos_counts:
                pos_counts[pos] += 1
        
        # Normalize and add POS counts to features
        total_tokens = max(1, len(tokens))
        features[1] = pos_counts['VERB'] / total_tokens
        features[2] = pos_counts['NOUN'] / total_tokens
        features[3] = pos_counts['ADJ'] / total_tokens
        
        # Add role presence information
        features[4] = 1.0 if roles.get('verb') is not None else 0.0
        features[5] = 1.0 if roles.get('subject') is not None else 0.0
        features[6] = 1.0 if roles.get('object') is not None else 0.0
        
        # Add negation feature
        has_negation = any(lemma == 'لا' or lemma == 'ليس' or lemma == 'غير' 
                         for lemma, _, _, _ in analyses)
        features[7] = 1.0 if has_negation else 0.0
        
        return features
    
    def combine_features(self, quantum_features: np.ndarray, 
                        linguistic_features: np.ndarray) -> np.ndarray:
        """
        Combine quantum and linguistic features into a single embedding.
        
        Args:
            quantum_features: Features extracted from the quantum circuit
            linguistic_features: Features extracted from linguistic analysis
            
        Returns:
            combined_features: Combined feature vector
        """
        # Ensure consistent dimensions
        q_len = len(quantum_features)
        l_len = len(linguistic_features)
        
        # Pad if necessary
        if q_len < self.embedding_dim:
            quantum_features = np.pad(quantum_features, 
                                     (0, self.embedding_dim - q_len),
                                     'constant')
        elif q_len > self.embedding_dim:
            quantum_features = quantum_features[:self.embedding_dim]
            
        if l_len < self.embedding_dim:
            linguistic_features = np.pad(linguistic_features,
                                        (0, self.embedding_dim - l_len),
                                        'constant')
        elif l_len > self.embedding_dim:
            linguistic_features = linguistic_features[:self.embedding_dim]
        
        # Combine with equal weighting (can be adjusted based on performance)
        combined = 0.5 * quantum_features + 0.5 * linguistic_features
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined
    
    def learn_meaning_clusters(self, embeddings: List[np.ndarray]) -> None:
        """
        Learn meaning clusters from a set of embeddings.
        
        Args:
            embeddings: List of embedding vectors to cluster
        """
        if len(embeddings) < self.num_clusters:
            # Adjust number of clusters if we have too few examples
            self.num_clusters = max(1, len(embeddings))
            
        # Convert to numpy array
        X = np.array(embeddings)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(X)
        self.meaning_clusters = kmeans.cluster_centers_

    def combine_features_with_attention(self, quantum_features, linguistic_features, structure):
        """Combine features with attention mechanism based on sentence structure"""
        # Create attention weights based on sentence structure
        if structure == 'NOMINAL':
            # For nominal sentences, emphasize linguistic features
            quantum_weight = 0.3
            linguistic_weight = 0.7
        elif structure == 'VSO' or structure == 'SVO':
            # For verbal sentences, balance features
            quantum_weight = 0.5
            linguistic_weight = 0.5
        elif 'COMPLEX' in structure:  # If we've detected a complex sentence
            # For complex sentences, emphasize quantum features that capture entanglement
            quantum_weight = 0.6
            linguistic_weight = 0.4
        else:
            # Default equal weighting
            quantum_weight = 0.5
            linguistic_weight = 0.5
        
        # Ensure consistent dimensions (as in original)
        q_len = len(quantum_features)
        l_len = len(linguistic_features)
        
        # Pad if necessary
        if q_len < self.embedding_dim:
            quantum_features = np.pad(quantum_features, 
                                    (0, self.embedding_dim - q_len),
                                    'constant')
        elif q_len > self.embedding_dim:
            quantum_features = quantum_features[:self.embedding_dim]
            
        if l_len < self.embedding_dim:
            linguistic_features = np.pad(linguistic_features,
                                    (0, self.embedding_dim - l_len),
                                    'constant')
        elif l_len > self.embedding_dim:
            linguistic_features = linguistic_features[:self.embedding_dim]
        
        # Combine with weighted attention
        combined = quantum_weight * quantum_features + linguistic_weight * linguistic_features
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined


    def learn_enhanced_meaning_clusters(self, embeddings, sentences):
        """Learn meaning clusters with topic modeling enhancement"""
        # Basic KMeans clustering as in original
        self.learn_meaning_clusters(embeddings)
        
        # Add topic modeling for deeper semantic understanding
        try:
            # Try to use gensim for LDA topic modeling
            from gensim import corpora, models
            
            # Tokenize and prepare for topic modeling
            tokenized_sentences = [sentence.split() for sentence in sentences]
            
            # Remove stopwords (Arabic)
            arabic_stopwords = ['من', 'إلى', 'عن', 'على', 'في', 'مع', 'و', 'ا', 'أن', 'ال']
            filtered_sentences = [[word for word in sentence if word not in arabic_stopwords] 
                                for sentence in tokenized_sentences]
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(filtered_sentences)
            corpus = [dictionary.doc2bow(text) for text in filtered_sentences]
            
            # Number of topics - can be adjusted
            num_topics = min(10, len(sentences) // 5) if len(sentences) > 5 else 2
            
            # Train LDA model
            lda_model = models.LdaModel(corpus=corpus, 
                                    id2word=dictionary, 
                                    num_topics=num_topics, 
                                    passes=10)
            
            # Get topics for each sentence
            sentence_topics = []
            for i, doc in enumerate(corpus):
                topics = lda_model.get_document_topics(doc)
                sentence_topics.append(topics)
            
            # Enhance clusters with topic information
            for cluster_id in self.meaning_map:
                # Find sentences in this cluster
                cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == cluster_id]
                
                # Get topics for sentences in this cluster
                cluster_topics = {}
                for idx in cluster_indices:
                    for topic_id, prob in sentence_topics[idx]:
                        if topic_id not in cluster_topics:
                            cluster_topics[topic_id] = 0
                        cluster_topics[topic_id] += prob
                
                # Normalize topic weights
                total_weight = sum(cluster_topics.values())
                if total_weight > 0:
                    for topic_id in cluster_topics:
                        cluster_topics[topic_id] /= total_weight
                
                # Get top words for each significant topic
                topic_words = {}
                for topic_id, weight in cluster_topics.items():
                    if weight > 0.1:  # Only significant topics
                        words = lda_model.show_topic(topic_id, topn=5)
                        topic_words[topic_id] = [word for word, _ in words]
                
                # Add topic information to meaning map
                self.meaning_map[cluster_id]['topics'] = {
                    'distribution': cluster_topics,
                    'key_words': topic_words
                }
        except Exception as e:
            print(f"Topic modeling enhancement failed: {e}")


    def analyze_sentence_in_context(self, current_circuit, current_tokens, current_analyses, 
                           current_structure, current_roles,
                           previous_circuits=None, previous_analyses=None):
        """Analyze a sentence with consideration of previous context"""
        # Get base interpretation
        base_interpretation = self.interpret_sentence(
            current_circuit, current_tokens, current_analyses, current_structure, current_roles)
        
        # If no context, return base interpretation
        if previous_circuits is None or previous_analyses is None:
            return base_interpretation
        
        # Process context
        context_embeddings = []
        for i, circuit in enumerate(previous_circuits):
            # Extract features from previous sentences
            prev_tokens = previous_analyses[i]['tokens']
            prev_struct = previous_analyses[i]['structure']
            prev_roles = previous_analyses[i]['roles']
            prev_analyses_tuples = previous_analyses[i]['analyses']
            
            # Get quantum features
            quantum_features = self.get_circuit_features(circuit)
            
            # Get linguistic features
            linguistic_features = self.extract_linguistic_features(
                prev_tokens, prev_analyses_tuples, prev_struct, prev_roles)
            
            # Combine features
            embedding = self.combine_features(quantum_features, linguistic_features)
            context_embeddings.append(embedding)
        
        # Calculate average context embedding
        if context_embeddings:
            avg_context = np.mean(context_embeddings, axis=0)
            
            # Add context effect to current interpretation
            current_embedding = base_interpretation.get('embedding', None)
            if current_embedding is not None:
                # Blend with context (weighted sum)
                context_influence = 0.2  # How much context influences current interpretation
                context_aware_embedding = (1 - context_influence) * current_embedding + context_influence * avg_context
                
                # Normalize
                norm = np.linalg.norm(context_aware_embedding)
                if norm > 0:
                    context_aware_embedding = context_aware_embedding / norm
                
                # Update interpretation with context-aware embedding
                base_interpretation['context_aware_embedding'] = context_aware_embedding
                
                # Recalculate meaning probabilities with context-aware embedding
                if self.meaning_clusters is not None:
                    context_aware_similarities = []
                    for i, cluster in enumerate(self.meaning_clusters):
                        prob = self.get_meaning_probability(context_aware_embedding, i)
                        context_aware_similarities.append((i, prob))
                    
                    # Sort by probability
                    context_aware_similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add context-aware meaning probabilities
                    base_interpretation['context_aware_meanings'] = context_aware_similarities[:3]
        
        # Find discourse relations with previous sentences
        discourse_relations = self.find_discourse_relations(
            current_tokens, previous_analyses[-1]['tokens'] if previous_analyses else [])
        if discourse_relations:
            base_interpretation['discourse_relations'] = discourse_relations
        
        return base_interpretation

    def find_discourse_relations(self, current_tokens, previous_tokens):
        """Find discourse relations between current and previous sentence"""
        discourse_relations = []
        
        # Look for discourse markers
        discourse_markers = {
            'CONTINUATION': ['و', 'ثم', 'بعد ذلك', 'بعدها'],
            'CAUSE': ['لذلك', 'وبالتالي', 'لهذا السبب'],
            'CONTRAST': ['لكن', 'غير أن', 'ومع ذلك', 'بالرغم'],
            'ELABORATION': ['أي', 'يعني', 'بمعنى'],
            'EXAMPLE': ['مثل', 'على سبيل المثال'],
        }
        
        # Check for markers at the beginning of current sentence
        if current_tokens:
            first_token = current_tokens[0]
            for relation_type, markers in discourse_markers.items():
                if first_token in markers:
                    discourse_relations.append({
                        'type': relation_type,
                        'marker': first_token
                    })
        
        # Check for pronouns referring to previous sentence
        pronouns = ['هذا', 'ذلك', 'تلك', 'هذه']
        for i, token in enumerate(current_tokens):
            if token in pronouns and i < len(current_tokens) - 1:
                # Check if followed by sentence reference
                if current_tokens[i+1] in ['الأمر', 'الشيء', 'الحدث', 'الفكرة', 'القول']:
                    discourse_relations.append({
                        'type': 'REFERENCE',
                        'marker': f"{token} {current_tokens[i+1]}"
                    })
        
        return discourse_relations

    def _get_reduced_density_matrix(self, statevector, subsystem_qubits):
        """
        Calculate reduced density matrix by taking partial trace.
        
        Args:
            statevector: Full quantum state vector
            subsystem_qubits: List of qubit indices to keep
            
        Returns:
            reduced_density_matrix: Density matrix for the subsystem
        """
        n_qubits = int(np.log2(len(statevector)))
        
        # Reshape statevector to tensor form
        tensor_form = statevector.reshape([2] * n_qubits)
        
        # Determine qubits to trace out
        trace_out_qubits = [i for i in range(n_qubits) if i not in subsystem_qubits]
        
        # Build reduced density matrix through partial trace
        # This is a simplified implementation
        rho = np.outer(statevector, np.conj(statevector))
        
        # Reshape to multi-index form
        rho_tensor = rho.reshape([2] * (2 * n_qubits))
        
        # Trace out unwanted qubits
        # This is a simplified approach - a full implementation would involve tensor contractions
        reduced_dims = 2 ** len(subsystem_qubits)
        reduced_rho = np.zeros((reduced_dims, reduced_dims), dtype=complex)
        
        # Convert to matrix with subsystem as rows/cols
        # For each basis state of the qubits to trace out
        for i in range(2 ** len(trace_out_qubits)):
            # Convert to binary representation
            bin_i = format(i, f'0{len(trace_out_qubits)}b')
            
            # For each element in reduced density matrix
            for row in range(reduced_dims):
                for col in range(reduced_dims):
                    # Convert row/col to binary
                    bin_row = format(row, f'0{len(subsystem_qubits)}b')
                    bin_col = format(col, f'0{len(subsystem_qubits)}b')
                    
                    # Build full indices for original density matrix
                    full_row_idx = ['0'] * n_qubits
                    full_col_idx = ['0'] * n_qubits
                    
                    # Fill in subsystem indices
                    r_idx = 0
                    for q in subsystem_qubits:
                        full_row_idx[q] = bin_row[r_idx]
                        full_col_idx[q] = bin_col[r_idx]
                        r_idx += 1
                    
                    # Fill in traced-out indices (same for row and col for trace)
                    t_idx = 0
                    for q in trace_out_qubits:
                        full_row_idx[q] = bin_i[t_idx]
                        full_col_idx[q] = bin_i[t_idx]
                        t_idx += 1
                    
                    # Convert binary indices to integer indices
                    full_row = int(''.join(full_row_idx), 2)
                    full_col = int(''.join(full_col_idx), 2)
                    
                    # Add contribution to reduced density matrix
                    reduced_rho[row, col] += rho[full_row, full_col]
        
        return reduced_rho


    def get_enhanced_circuit_features(self, circuit, shots: int = 1024) -> np.ndarray:
        """
        Enhanced feature extraction from a Qiskit circuit.
        (Input 'circuit' is now expected to be a Qiskit QuantumCircuit)
        """
        print(f"\n--- Debugging get_enhanced_circuit_features ---") # DEBUG
         # --- Input is now expected to be a Qiskit circuit ---
        if not isinstance(circuit, QuantumCircuit):
             print(f"Error: get_enhanced_circuit_features received non-Qiskit object: {type(circuit)}")
             # Return fallback random features
             fallback = np.random.rand(self.embedding_dim)
             norm = np.linalg.norm(fallback)
             if norm > 0: fallback = fallback / norm
             return fallback

        print(f"Input type: {type(circuit)}") # DEBUG

        try:
            # First, get the basic circuit features
            print("Calling get_circuit_features for basic features...") # DEBUG
            # Pass the Qiskit circuit directly
            basic_features = self.get_circuit_features(circuit, shots)
            print(f"Basic features obtained. Shape: {basic_features.shape}") # DEBUG

            # --- No conversion needed here, use the input circuit directly ---
            print("Ensuring Qiskit circuit name for enhanced analysis...") # DEBUG
            circuit = _ensure_circuit_name(circuit, "circuit_for_enhanced")
            print(f"Qiskit circuit name: {getattr(circuit, 'name', 'N/A')}") # DEBUG

            # Execute the Qiskit circuit again for enhanced analysis
            print("Executing Qiskit circuit for enhanced analysis...") # DEBUG
            result = execute(circuit, self.simulator).result()
            print(f"Execution successful. Result type: {type(result)}") # DEBUG

            # (Entanglement/Pauli extraction logic remains the same, uses the input 'circuit')
            entanglement_features = []
            pauli_expectations = []

            print("Checking for statevector for enhanced analysis...") # DEBUG
            if hasattr(result, 'get_statevector'):
                print("Statevector found.") # DEBUG
                statevector = result.get_statevector()
                num_qubits = circuit.num_qubits # Get from Qiskit circuit
                print(f"Statevector obtained. Length: {len(statevector)}, Num Qubits: {num_qubits}") # DEBUG

                # 1. Calculate entanglement features
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
                        except Exception as inner_e:
                            print(f"  Error calculating entanglement for qubit {i}: {type(inner_e).__name__} - {inner_e}") # DEBUG
                            entanglement_features.append(0.0)
                else: print(f"  Skipping entanglement calculation (num_qubits={num_qubits}).") #DEBUG

                # 3. Simplified Pauli expectations (uses the input Qiskit 'circuit')
                print("Calculating Pauli X expectations...") # DEBUG
                if isinstance(circuit, QuantumCircuit) and hasattr(circuit, 'copy'):
                    try:
                        num_pauli_checks = min(num_qubits, 3)
                        for i in range(num_pauli_checks):
                            meas_circuit_name = f"{getattr(circuit, 'name', 'unnamed')}_pauli_x_q{i}"
                            meas_circuit = circuit.copy(name=meas_circuit_name)
                            meas_circuit.h(i)
                            if not meas_circuit.cregs or meas_circuit.cregs[0].size < num_qubits:
                                 from qiskit import ClassicalRegister
                                 cr_name = f'c_{i}'
                                 existing_cr = next((cr for cr in meas_circuit.cregs if cr.name == cr_name), None)
                                 if not existing_cr: cr = ClassicalRegister(num_qubits, name=cr_name); meas_circuit.add_register(cr)
                                 else: cr = existing_cr
                            else: cr = meas_circuit.cregs[0]
                            if i < cr.size: meas_circuit.measure(i, i)
                            else: print(f"Warning: Classical register '{cr.name}' too small for qubit {i}. Skipping measure."); pauli_expectations.append(0.0); continue
                            meas_result = execute(meas_circuit, self.simulator, shots=shots).result()
                            counts = meas_result.get_counts()
                            exp_val = 0; total = sum(counts.values())
                            if total > 0:
                                 prob_0, prob_1 = 0, 0
                                 for bitstring, count in counts.items():
                                      bit_index = len(bitstring) - 1 - i
                                      if 0 <= bit_index < len(bitstring):
                                          if int(bitstring[bit_index]) == 0: prob_0 += count / total
                                          else: prob_1 += count / total
                                 exp_val = prob_0 - prob_1
                            pauli_expectations.append(exp_val)
                            print(f"  Pauli X expectation for qubit {i}: {exp_val:.4f}") # DEBUG
                    except Exception as pauli_e:
                        print(f"  Error in Pauli expectation calculation: {type(pauli_e).__name__} - {pauli_e}") # DEBUG
                        traceback.print_exc() # DEBUG
                        num_pauli_checks = min(num_qubits, 3)
                        pauli_expectations.extend([0.0] * (num_pauli_checks - len(pauli_expectations)))
                else: print("  Skipping Pauli expectation (input not a Qiskit circuit or no .copy).") # DEBUG
            else: print("Statevector not found. Skipping enhanced feature calculation.") # DEBUG

            # Combine features
            print("Combining basic, entanglement, and Pauli features...") # DEBUG
            feature_list = [basic_features]
            if entanglement_features: feature_list.append(np.array(entanglement_features))
            if pauli_expectations: feature_list.append(np.array(pauli_expectations))
            all_features = np.concatenate(feature_list) if len(feature_list) > 1 else basic_features
            print(f"Combined features shape before padding/truncation: {all_features.shape}") # DEBUG
            current_len = len(all_features)
            if current_len < self.embedding_dim: all_features = np.pad(all_features, (0, self.embedding_dim - current_len), 'constant')
            elif current_len > self.embedding_dim: all_features = all_features[:self.embedding_dim]
            norm = np.linalg.norm(all_features)
            if norm > 0: all_features = all_features / norm
            print(f"Returning enhanced features. Final shape: {all_features.shape}") # DEBUG
            return all_features

        except Exception as e:
            print(f"\n--- ERROR in get_enhanced_circuit_features ---") # DEBUG
            print(f"Error type: {type(e)}") # DEBUG
            print(f"Error message: {e}") # DEBUG
            traceback.print_exc() # DEBUG
            print(f"----------------------------------------------\n") # DEBUG

        # Fallback uses the already computed basic_features if enhanced fails
        print("Falling back to basic features due to error in enhanced calculation.") # DEBUG
        if 'basic_features' in locals() and isinstance(basic_features, np.ndarray):
             current_len = len(basic_features)
             if current_len < self.embedding_dim: basic_features = np.pad(basic_features, (0, self.embedding_dim - current_len), 'constant')
             elif current_len > self.embedding_dim: basic_features = basic_features[:self.embedding_dim]
             norm = np.linalg.norm(basic_features);
             if norm > 0: basic_features = basic_features / norm
             return basic_features
        else:
             fallback = np.random.rand(self.embedding_dim)
             norm = np.linalg.norm(fallback);
             if norm > 0: fallback = fallback / norm
             return fallback


    def _classify_verb(self, verb_lemma: Optional[str]) -> str:
        """Classify verb lemma into semantic categories."""
        if not verb_lemma:
            return "UNKNOWN"

        # Define verb classes (expand these lists)
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

        # Default class if not found
        return "ACTION"

    def assign_meaning_to_clusters(self,
                              sentences: List[str],
                              structures: List[str],
                              roles_list: List[Dict],
                              analyses_list: List[List[Tuple]], # Added analyses_list
                              camel_analyzer=None) -> Dict:
        """
        Assign meaning templates to the identified clusters, attempting to deduce
        templates based on linguistic analysis of cluster members.

        Args:
            sentences: List of sentences corresponding to the embeddings
            structures: List of sentence structures
            roles_list: List of role dictionaries
            analyses_list: List of analysis tuples [(lemma, pos, dep, head), ...] for each sentence
            camel_analyzer: Optional CAMeL Tools analyzer for deeper analysis

        Returns:
            meaning_map: Dictionary mapping cluster IDs to meaning templates
        """
        # Initialize meaning map
        self.meaning_map = {}

        if self.cluster_labels is None:
             print("Warning: Cluster labels not found. Cannot assign meanings.")
             return self.meaning_map

        # --- Optional: Initialize CAMeL Tools if needed (keep your existing logic) ---
        sentiment_analyzer = None
        # (Your existing logic to initialize camel_analyzer, sentiment_analyzer, farasa_segmenter if needed)
        try:
            from camel_tools.sentiment.factory import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer.pretrained()
        except ImportError:
            print("CAMeL Sentiment Analyzer not found, skipping sentiment analysis.")
            sentiment_analyzer = None
        except Exception as e:
            print(f"Error initializing CAMeL Sentiment Analyzer: {e}")
            sentiment_analyzer = None

        # --- Group sentences and analyses by cluster ---
        cluster_data_grouped = {}
        for i, label in enumerate(self.cluster_labels):
            if label not in cluster_data_grouped:
                cluster_data_grouped[label] = []

            # Ensure all necessary data is available for the index
            if i < len(sentences) and i < len(structures) and i < len(roles_list) and i < len(analyses_list):
                 cluster_data_grouped[label].append({
                     'sentence': sentences[i],
                     'structure': structures[i],
                     'roles': roles_list[i],
                     'analyses': analyses_list[i], # Include analyses
                     'index': i # Keep track of original index if needed
                 })
            else:
                 print(f"Warning: Skipping sentence index {i} due to missing data.")


        # --- Assign meaning to each cluster ---
        for cluster_id, cluster_data in cluster_data_grouped.items():
            if not cluster_data: # Skip empty clusters
                continue

            # --- Aggregate linguistic info from the cluster ---
            verb_lemmas = Counter()
            subject_lemmas = Counter()
            object_lemmas = Counter()
            common_preps = Counter() # Counter for prepositional objects like ('to', 'location')
            verb_tenses = Counter() # Counter for tense/aspect
            verb_moods = Counter() # Counter for mood
            sentiments = []
            structure_counts = Counter()

            for item in cluster_data:
                structure_counts[item['structure']] += 1
                tokens = item['sentence'].split() # Simple tokenization
                roles = item['roles']
                analyses = item['analyses']

                # Extract verb, subject, object lemmas
                verb_idx = roles.get('verb')
                subj_idx = roles.get('subject')
                obj_idx = roles.get('object')

                if verb_idx is not None and verb_idx < len(analyses):
                    verb_lemmas[analyses[verb_idx][0]] += 1 # Use lemma (index 0)

                if subj_idx is not None and subj_idx < len(analyses):
                    subject_lemmas[analyses[subj_idx][0]] += 1

                if obj_idx is not None and obj_idx < len(analyses):
                    object_lemmas[analyses[obj_idx][0]] += 1

                # Extract common prepositions associated with the verb or object
                for i, (lemma, pos, dep, head) in enumerate(analyses):
                    # Look for prepositions modifying the verb or object
                    if pos == 'ADP' and (head == verb_idx or head == obj_idx):
                         # Find the object of the preposition (often the next word or its dependent)
                         prep_obj_lemma = "THING" # Default
                         for j, (_, _, dep_obj, head_obj) in enumerate(analyses):
                              if head_obj == i and dep_obj == 'obj': # ADP -> obj
                                   prep_obj_lemma = analyses[j][0]
                                   break
                         common_preps[(lemma, prep_obj_lemma)] += 1 # Store ('to', 'location_lemma')

                # --- Optional: Deeper analysis with CAMeL Tools ---
                if camel_analyzer:
                    try:
                        # Analyze the sentence using the provided analyzer
                        morph_analysis = camel_analyzer.analyze(item['sentence'])

                        # Extract tense/aspect/mood for the verb
                        if verb_idx is not None and verb_idx < len(morph_analysis):
                            verb_morph = morph_analysis[verb_idx]
                            if 'asp' in verb_morph: verb_tenses[verb_morph['asp']] += 1
                            if 'mod' in verb_morph: verb_moods[verb_morph['mod']] += 1
                        # Add more CAMeL feature extraction here (e.g., case for roles)

                    except Exception as e:
                        # print(f"CAMeL analysis failed for cluster {cluster_id}: {e}")
                        pass # Silently ignore CAMeL errors for now

                # --- Optional: Sentiment Analysis ---
                if sentiment_analyzer:
                    try:
                        sentiment = sentiment_analyzer.predict(item['sentence'])
                        sentiments.append(sentiment)
                    except Exception as e:
                        # print(f"Sentiment analysis failed for cluster {cluster_id}: {e}")
                        pass


            # --- Determine dominant features for the cluster ---
            dominant_structure = structure_counts.most_common(1)[0][0] if structure_counts else 'OTHER'
            dominant_verb = verb_lemmas.most_common(1)[0][0] if verb_lemmas else None
            dominant_subj = subject_lemmas.most_common(1)[0][0] if subject_lemmas else "SUBJECT"
            dominant_obj = object_lemmas.most_common(1)[0][0] if object_lemmas else "OBJECT"
            top_prep = common_preps.most_common(1)[0][0] if common_preps else None # e.g., ('to', 'place')
            dominant_tense = verb_tenses.most_common(1)[0][0] if verb_tenses else None # e.g., 'i' (imperfective)
            dominant_mood = verb_moods.most_common(1)[0][0] if verb_moods else None # e.g., 'i' (indicative)

            # --- Deduce Semantic Template ---
            deduced_template = f"{dominant_subj} (did something involving) {dominant_obj}" # Basic fallback
            verb_class = self._classify_verb(dominant_verb) # Helper to classify verb

            # Generate template based on verb class and common arguments
            if verb_class == "MOTION":
                dest = "DESTINATION"
                if top_prep and top_prep[0] in ['إلى', 'ل']: # 'to'
                     dest = top_prep[1] # Use the object of 'to'
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
            elif dominant_verb: # Generic verb case
                 # Refine based on structure and objects
                 if dominant_structure == 'VSO':
                      action_desc = f"{dominant_verb} performed by {dominant_subj}"
                      if dominant_obj != "OBJECT": action_desc += f" on {dominant_obj}"
                      deduced_template = action_desc
                 elif dominant_structure == 'SVO':
                      action_desc = f"{dominant_subj} performs {dominant_verb}"
                      if dominant_obj != "OBJECT": action_desc += f" on {dominant_obj}"
                      deduced_template = action_desc
                 elif dominant_structure == 'NOMINAL':
                      # Find common predicate adjective/noun
                      predicate = "ATTRIBUTE" # Default
                      # (Add logic here to find common predicate based on analyses)
                      deduced_template = f"{dominant_subj} is {predicate}"
                 else: # OTHER structure
                      deduced_template = f"Statement about {dominant_subj} involving {dominant_verb}"


            # Refine template with tense (Example)
            tense_map = {'p': ' (past)', 'i': ' (present)', 'c': ' (command)'}
            if dominant_tense in tense_map:
                 deduced_template += tense_map[dominant_tense]

            # --- Determine Sentiment ---
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
            sentiment_label = None
            if avg_sentiment is not None:
                # Adjust thresholds as needed
                if avg_sentiment > 0.6: sentiment_label = "positive"
                elif avg_sentiment < 0.4: sentiment_label = "negative"
                else: sentiment_label = "neutral"

            # --- Store results in meaning_map ---
            self.meaning_map[cluster_id] = {
                'structure': dominant_structure,
                'deduced_template': deduced_template, # Store the new template
                'dominant_verb': dominant_verb,
                'dominant_subject': dominant_subj,
                'dominant_object': dominant_obj,
                'common_prep_phrase': top_prep,
                'sentiment': sentiment_label,
                'examples': [item['sentence'] for item in cluster_data[:3]], # Keep examples
                # Keep original templates if desired for comparison
                'original_templates': self.semantic_templates.get(dominant_structure, self.semantic_templates['OTHER'])
            }

        return self.meaning_map

    # --- Helper function to classify verbs (add this to the class) ---
    def _classify_verb(self, verb_lemma: Optional[str]) -> str:
        """Classify verb lemma into semantic categories."""
        if not verb_lemma:
            return "UNKNOWN"

        # Define verb classes (expand these lists)
        motion_verbs = ["ذهب", "جاء", "مشى", "سار", "رجع", "دخل", "خرج", "وصل", "سافر"]
        possession_verbs = ["أخذ", "أعطى", "ملك", "وهب", "منح", "سلم", "اشترى", "باع", "امتلك"]
        communication_verbs = ["قال", "تكلم", "صرح", "أخبر", "سأل", "أجاب", "حدث", "نادى"]
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

        # Default class if not found
        return "ACTION"

        
    def get_meaning_probability(self, 
                              embedding: np.ndarray, 
                              cluster_id: int) -> float:
        """
        Calculate the probability that an embedding belongs to a meaning cluster.
        
        Args:
            embedding: The embedding vector to evaluate
            cluster_id: The cluster ID to compare against
            
        Returns:
            probability: A value between 0 and 1 indicating the probability
        """
        if self.meaning_clusters is None:
            return 0.0
            
        # Get the cluster center
        cluster_center = self.meaning_clusters[cluster_id]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding], [cluster_center])[0][0]
        
        # Convert to a probability (0 to 1)
        probability = (similarity + 1) / 2
        
        return probability
    def extract_complex_linguistic_features(self, tokens, analyses, structure, roles):
        """Enhanced linguistic feature extraction for complex sentences"""

        features = np.zeros(14)
        # Get basic features
        basic_features = self.extract_linguistic_features(tokens, analyses, structure, roles)
        
        # Add features for complex sentences
        features = np.zeros(self.embedding_dim)
        features[:len(basic_features)] = basic_features
        
        # 1. Detect subordinate clauses
        subordinate_markers = ['الذي', 'التي', 'الذين', 'اللواتي', 'عندما', 'حيث', 'لأن', 'كي']
        has_subordinate = any(token in subordinate_markers for token in tokens)
        features[8] = 1.0 if has_subordinate else 0.0
        
        # 2. Count clauses by counting verbs
        verb_count = sum(1 for _, pos, _, _ in analyses if pos == 'VERB')
        features[9] = min(verb_count / 3, 1.0)  # Normalize, cap at 3 verbs
        
        # 3. Detect conditional structures
        conditional_markers = ['إذا', 'لو', 'إن']
        has_conditional = any(token in conditional_markers for token in tokens)
        features[10] = 1.0 if has_conditional else 0.0
        
        # 4. Detect quotations/reported speech
        quotation_markers = ['قال', 'صرح', 'أعلن', 'ذكر']
        has_quotation = any(token in quotation_markers for token in tokens)
        features[11] = 1.0 if has_quotation else 0.0
        
        # 5. Extract sentence complexity metrics
        avg_word_length = np.mean([len(token) for token in tokens])
        features[12] = min(avg_word_length / 10, 1.0)  # Normalize
        
        # 6. Identify clause boundaries
        clause_boundaries = []
        for i, (_, pos, dep, _) in enumerate(analyses):
            if pos == 'VERB' or tokens[i] in subordinate_markers:
                clause_boundaries.append(i)
        
        # 7. Feature for clause density
        if len(tokens) > 0:
            clause_density = len(clause_boundaries) / len(tokens)
            features[13] = min(clause_density * 10, 1.0)  # Normalize
        
        return features

    def interpret_sentence(self,
                      circuit,
                      tokens: List[str],
                      analyses: List[Tuple],
                      structure: str,
                      roles: Dict,
                      # Removed camel_analyzer from args, use self.camel_analyzer
                      previous_circuits=None,
                      previous_analyses=None # Added previous_analyses for context
                      ) -> Dict:
        """
        Interpret the meaning of a sentence based on its quantum circuit,
        linguistic features, context, and enhanced semantic analysis.

        Args:
            circuit: The quantum circuit for the sentence
            tokens: List of tokens in the sentence
            analyses: List of (lemma, pos, dep_type, head_idx) tuples
            structure: The sentence structure (VSO, SVO, NOMINAL, COMPLEX, etc.)
            roles: Dictionary with grammatical roles
            previous_circuits: List of circuits from previous sentences (for context)
            previous_analyses: List of full analysis dicts from previous sentences

        Returns:
            interpretation: Dictionary with interpretation details
        """
        print(f"\n--- Interpreting sentence: {' '.join(tokens)} ---") # DEBUG
        # Extract features from quantum circuit (using enhanced version)
        print("Getting enhanced circuit features...") # DEBUG
        quantum_features = self.get_enhanced_circuit_features(circuit)

        # Initialize enhanced analyses with existing data
        enhanced_analyses = analyses
        enhanced_roles = roles.copy()

        # --- Integration: Use self.camel_analyzer if available ---
        camel_morphology = None
        sentiment_score = None # Use None as default, not 0.0
        named_entities = []
        if self.camel_analyzer is not None:
             print("Using CAMeL Tools for enhanced analysis...") # DEBUG
             # ... (CAMeL tools logic as before, using self.camel_analyzer) ...
             try:
                # Import CAMeL Tools components if not already imported
                from camel_tools.morphology.database import MorphologyDB
                from camel_tools.morphology.analyzer import Analyzer
                from camel_tools.sentiment.factory import SentimentAnalyzer
                from camel_tools.ner.ner import NERecognizer

                sentence_text = ' '.join(tokens)

                # Perform morphological analysis
                print("  Performing morphological analysis...") # DEBUG
                camel_morphology = self.camel_analyzer.analyze(sentence_text)

                # Try to perform sentiment analysis
                try:
                    print("  Performing sentiment analysis...") # DEBUG
                    sentiment_analyzer = SentimentAnalyzer.pretrained()
                    # Predict returns a list, get the first element
                    sentiment_result = sentiment_analyzer.predict([sentence_text])
                    sentiment_score = sentiment_result[0] # Assuming single sentence prediction
                    print(f"  Sentiment score: {sentiment_score}") # DEBUG
                except Exception as sent_e:
                    print(f"  Sentiment analysis failed: {sent_e}")
                    pass

                # Try to perform named entity recognition
                try:
                    print("  Performing NER...") # DEBUG
                    ner = NERecognizer.pretrained()
                    # predict_sentence returns list of tags, not entities directly
                    # We might need to process this further if needed
                    ner_tags = ner.predict_sentence(tokens) # Predict on tokens
                    named_entities = list(zip(tokens, ner_tags)) # Combine tokens and tags
                    print(f"  NER tags: {named_entities}") # DEBUG
                except Exception as ner_e:
                    print(f"  NER failed: {ner_e}")
                    pass

                # Enhance analyses with CAMeL morphology (if successful)
                if camel_morphology:
                    print("  Enhancing linguistic analyses with CAMeL morphology...") # DEBUG
                    temp_analyses = []
                    for i, token in enumerate(tokens):
                        if i < len(analyses):
                            lemma, pos, dep_type, head_idx = analyses[i]
                            if i < len(camel_morphology):
                                morpho_analysis = camel_morphology[i]
                                if 'lex' in morpho_analysis and morpho_analysis['lex']:
                                    lemma = morpho_analysis['lex']
                                if 'pos' in morpho_analysis and morpho_analysis['pos']:
                                    camel_pos = morpho_analysis['pos'].upper() # Ensure uppercase
                                    # More robust mapping needed potentially
                                    pos_mapping = {
                                        'VERB': 'VERB', 'NOUN': 'NOUN', 'ADJ': 'ADJ',
                                        'ADV': 'ADV', 'PRON': 'PRON', 'DET': 'DET',
                                        'PART': 'PART', 'PREP': 'ADP', 'CONJ': 'CONJ',
                                        'PUNC': 'PUNCT', 'NUM': 'NUM', 'ABBREV': 'X',
                                        'FOREIGN': 'X' # Map others as needed
                                    }
                                    pos = pos_mapping.get(camel_pos, pos) # Keep original if no map
                            temp_analyses.append((lemma, pos, dep_type, head_idx))
                        else:
                             temp_analyses.append(analyses[i]) # Should not happen if lengths match
                    enhanced_analyses = temp_analyses # Update with enhanced data

             except ImportError:
                print("  CAMeL Tools import failed, skipping enhancement.") # DEBUG
                pass
             except Exception as camel_e:
                 print(f"  Error during CAMeL analysis: {camel_e}") # DEBUG
                 pass
        else:
             print("CAMeL Tools analyzer not available.") # DEBUG
        # --- End Integration ---

        # Extract linguistic features using possibly enhanced analyses
        print("Extracting complex linguistic features...") # DEBUG
        linguistic_features = self.extract_complex_linguistic_features(
            tokens, enhanced_analyses, structure, enhanced_roles
        )

        # Combine features
        print("Combining features with attention...") # DEBUG
        embedding = self.combine_features_with_attention(
            quantum_features, linguistic_features, structure
        )

        # --- Integration: Contextual Analysis ---
        context_result = None
        discourse_info = []
        if previous_circuits is not None and previous_analyses is not None:
            print("Analyzing sentence in context...") # DEBUG
            # Note: analyze_sentence_in_context internally calls feature extraction again
            # This could be optimized by passing features/embeddings if needed.
            context_result = self.analyze_sentence_in_context(
                circuit, tokens, enhanced_analyses, structure, enhanced_roles,
                previous_circuits, previous_analyses
            )
            # Use the context-aware embedding if available
            embedding = context_result.get('context_aware_embedding', embedding)
            discourse_info = context_result.get('discourse_relations', [])
            print(f"  Context analysis complete. Discourse relations found: {len(discourse_info)}") # DEBUG
        # --- End Integration ---

        # --- Integration: Semantic Frame Extraction ---
        print("Extracting enhanced semantic frames...") # DEBUG
        semantic_frames_data = self.extract_enhanced_semantic_frames(
            tokens, enhanced_analyses, enhanced_roles, self.camel_analyzer
        )
        print(f"  Semantic frames extracted: {len(semantic_frames_data.get('frames', []))}") # DEBUG
        # --- End Integration ---

        # Prepare base result structure
        result = {
            'sentence': ' '.join(tokens),
            'structure': structure,
            'embedding': embedding, # Store the final embedding
            'interpretation': None, # Placeholder for primary interpretation
            'meaning_options': [],
            'specific_interpretation': None,
            'semantic_frames': semantic_frames_data.get('frames', []), # Add frames
            'discourse_relations': discourse_info, # Add discourse relations
            'enhanced_linguistic_analysis': enhanced_analyses, # Store enhanced analysis
            'roles': enhanced_roles # Store roles used
        }

        # Add CAMeL specific info if available
        if camel_morphology:
            result['morphological_analysis'] = camel_morphology
        if sentiment_score is not None: # Check for None explicitly
            result['sentiment'] = sentiment_score
        if named_entities:
            result['named_entities'] = named_entities


        # If we have learned meaning clusters, use them for interpretation
        if self.meaning_clusters is not None and len(self.meaning_clusters) > 0:
            print("Calculating similarities to meaning clusters...") # DEBUG
            similarities = []
            for i, cluster in enumerate(self.meaning_clusters):
                prob = self.get_meaning_probability(embedding, i)
                similarities.append((i, prob))

            # Sort by probability
            similarities.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top cluster: {similarities[0][0]} (Prob: {similarities[0][1]:.4f})") # DEBUG

            # Get the top 3 meaning interpretations
            meanings = []
            for cluster_id, prob in similarities[:min(3, len(similarities))]: # Ensure we don't exceed bounds
                if cluster_id in self.meaning_map:
                    cluster_info = self.meaning_map[cluster_id]
                    meanings.append({
                        'cluster_id': cluster_id,
                        'structure': cluster_info.get('structure', 'N/A'),
                        'templates': cluster_info.get('enhanced_templates', cluster_info.get('templates', {})), # Prefer enhanced
                        'examples': cluster_info.get('examples', []),
                        'probability': prob,
                        'sentiment': cluster_info.get('sentiment', None),
                        'topics': cluster_info.get('topics', {}) # Include topics if available
                    })

            result['meaning_options'] = meanings
            if meanings:
                 result['top_meaning_cluster'] = meanings[0]['cluster_id']
                 result['confidence'] = meanings[0]['probability']
                 # Create specific interpretation for the top match
                 print("Creating specific interpretation for top cluster...") # DEBUG
                 result['specific_interpretation'] = self.create_specific_interpretation(
                     tokens, enhanced_analyses, enhanced_roles,
                     meanings[0]['structure'], meanings[0]['templates']
                 )
                 # Set primary interpretation based on top match's declarative template
                 result['interpretation'] = result['specific_interpretation']['templates'].get('declarative', 'N/A')

        else:
            # No learned clusters or clusters are empty, fall back to basic interpretation
            print("No meaning clusters found or available. Falling back to basic template.") # DEBUG
            templates = self.semantic_templates.get(structure, self.semantic_templates['OTHER'])
            result['specific_interpretation'] = self.create_specific_interpretation(
                tokens, enhanced_analyses, enhanced_roles, structure, templates)
            result['interpretation'] = result['specific_interpretation']['templates'].get('declarative', 'N/A')
            result['confidence'] = 0.0 # No confidence score without clusters

        print(f"--- Interpretation complete for: {' '.join(tokens)} ---") # DEBUG
        return result

    
    def create_specific_interpretation(self, 
                                tokens: List[str], 
                                analyses: List[Tuple], 
                                roles: Dict,
                                structure: str,
                                templates: Dict,
                                camel_analyzer=None) -> Dict:
        """
        Create a specific interpretation by filling in templates with actual values,
        enhanced with NLP analysis.
        
        Args:
            tokens: List of tokens in the sentence
            analyses: List of (lemma, pos, dep_type, head_idx) tuples
            roles: Dictionary with grammatical roles
            structure: The sentence structure (VSO, SVO, etc.)
            templates: Meaning templates to fill in
            camel_analyzer: Optional CAMeL Tools analyzer
            
        Returns:
            interpretation: Dictionary with filled templates and semantic details
        """
        # Extract the subject, verb, and object if present
        subject = "unknown"
        verb = "unknown"
        predicate = "unknown"
        object_text = "unknown"
        
        # Get the verb
        verb_idx = roles.get('verb')
        verb_lemma = None
        if verb_idx is not None and verb_idx < len(tokens):
            verb = tokens[verb_idx]
            # Try to get the lemma from analyses
            if verb_idx < len(analyses):
                verb_lemma = analyses[verb_idx][0]
        
        # Get the subject
        subj_idx = roles.get('subject')
        if subj_idx is not None and subj_idx < len(tokens):
            subject = tokens[subj_idx]
            
            # For subjects with determiners, include the determiner
            if subj_idx > 0 and subj_idx < len(analyses) and analyses[subj_idx-1][1] == 'DET':
                subject = tokens[subj_idx-1] + " " + subject
        
        # Get the object
        obj_idx = roles.get('object')
        if obj_idx is not None and obj_idx < len(tokens):
            object_text = tokens[obj_idx]
            
            # For objects with determiners, include the determiner
            if obj_idx > 0 and obj_idx < len(analyses) and analyses[obj_idx-1][1] == 'DET':
                object_text = tokens[obj_idx-1] + " " + object_text
        
        # For nominal sentences, find the predicate (usually an adjective)
        if structure == 'NOMINAL':
            for i, (_, pos, dep, head) in enumerate(analyses):
                if pos == 'ADJ' and head == subj_idx:
                    predicate = tokens[i]
                    break
        
        # Perform enhanced analysis with CAMeL Tools if available
        semantic_roles = {}
        semantic_frames = []
        tense = "present"  # Default tense
        modality = "indicative"  # Default modality
        
        if camel_analyzer:
            try:
                # Get full sentence text
                sentence = ' '.join(tokens)
                
                # Analyze with CAMeL Tools
                morph_analysis = camel_analyzer.analyze(sentence)
                
                # Extract advanced features
                for i, token_analysis in enumerate(morph_analysis):
                    # Check for verb features
                    if i == verb_idx and 'pos' in token_analysis and token_analysis['pos'] == 'VERB':
                        # Extract tense
                        if 'asp' in token_analysis:
                            asp = token_analysis['asp']
                            if asp == 'p':  # perfective
                                tense = "past"
                            elif asp == 'i':  # imperfective
                                tense = "present"
                            elif asp == 'c':  # command
                                tense = "imperative"
                                
                        # Extract modality
                        if 'mod' in token_analysis:
                            mod = token_analysis['mod']
                            if mod == 'i':
                                modality = "indicative"
                            elif mod == 's':
                                modality = "subjunctive"
                            elif mod == 'j':
                                modality = "jussive"
                    
                    # Try to extract semantic roles
                    if 'pos' in token_analysis:
                        pos = token_analysis['pos']
                        if pos == 'NOUN' or pos == 'PRON':
                            # Try to determine semantic role from case marking
                            if 'cas' in token_analysis:
                                case = token_analysis['cas']
                                if case == 'n':  # nominative
                                    semantic_roles[i] = "AGENT"
                                elif case == 'a':  # accusative
                                    semantic_roles[i] = "PATIENT"
                                elif case == 'g':  # genitive
                                    semantic_roles[i] = "POSSESSOR"
            except Exception as e:
                print(f"CAMeL Tools analysis error: {e}")
        
        # Try using a simple rule-based semantic frame detection based on verb
        if verb_lemma:
            # Motion verbs
            if verb_lemma in ["ذهب", "جاء", "مشى", "سار", "عاد"]:
                semantic_frames.append("MOTION")
            # Communication verbs
            elif verb_lemma in ["قال", "تكلم", "صرح", "أعلن", "خاطب"]:
                semantic_frames.append("COMMUNICATION") 
            # Cognition verbs
            elif verb_lemma in ["فكر", "اعتقد", "ظن", "علم", "فهم"]:
                semantic_frames.append("COGNITION")
            # Possession verbs
            elif verb_lemma in ["ملك", "حاز", "اشترى", "باع", "أعطى"]:
                semantic_frames.append("POSSESSION")
        
        # Fill in the templates
        filled_templates = {}
        for template_type, template in templates.items():
            filled = template
            filled = filled.replace("SUBJECT", subject)
            filled = filled.replace("ACTION", verb)
            filled = filled.replace("OBJECT", object_text)
            filled = filled.replace("PREDICATE", predicate)
            filled = filled.replace("TOPIC", subject)  # Default topic to subject
            
            filled_templates[template_type] = filled
        
        # Prepare detailed semantic information
        semantic_details = {
            'subject': {
                'text': subject,
                'index': subj_idx,
                'semantic_role': semantic_roles.get(subj_idx, "AGENT" if subj_idx is not None else None)
            },
            'verb': {
                'text': verb,
                'index': verb_idx,
                'lemma': verb_lemma,
                'tense': tense,
                'modality': modality
            },
            'object': {
                'text': object_text,
                'index': obj_idx,
                'semantic_role': semantic_roles.get(obj_idx, "PATIENT" if obj_idx is not None else None)
            },
            'predicate': {
                'text': predicate,
                'structure_type': structure
            },
            'semantic_frames': semantic_frames
        }
            
        return {
            'templates': filled_templates,
            'semantic_details': semantic_details
        }
        
    def train(self, sentences, circuits, tokens_list, analyses_list, structures, roles_list):
        """
        Train the kernel on a set of sentences and their Qiskit circuits.
        (Input 'circuits' is now expected to be a list of Qiskit QuantumCircuit objects)
        """
        self.reference_sentences = {i: sentences[i] for i in range(len(sentences))}
        self.circuit_embeddings = {}
        self.sentence_embeddings = {}
        embeddings = []
        print(f"\n--- Training Kernel on {len(sentences)} sentences ---") # DEBUG

        # Check for mismatched lengths
        if not (len(circuits) == len(tokens_list) == len(analyses_list) == len(structures) == len(roles_list) == len(sentences)):
            print("Error: Input lists to train method have mismatched lengths.")
            min_len = min(len(sentences), len(circuits), len(tokens_list), len(analyses_list), len(structures), len(roles_list))
            print(f"Warning: Training with reduced dataset size: {min_len}")
            if min_len == 0: print("Error: Cannot train with empty dataset."); return self
            sentences, circuits, tokens_list, analyses_list, structures, roles_list = (lst[:min_len] for lst in
                [sentences, circuits, tokens_list, analyses_list, structures, roles_list])
        else:
            min_len = len(sentences)

        # Extract features and create embeddings
        for i in range(min_len):
            print(f"Processing sentence {i+1}/{min_len}: '{sentences[i][:30]}...'") # DEBUG
            try:
                # --- Pass the Qiskit circuit to get_enhanced_circuit_features ---
                quantum_features = self.get_enhanced_circuit_features(circuits[i])
                self.circuit_embeddings[i] = quantum_features

                linguistic_features = self.extract_complex_linguistic_features(
                    tokens_list[i], analyses_list[i], structures[i], roles_list[i]
                )
                embedding = self.combine_features_with_attention(
                    quantum_features, linguistic_features, structures[i]
                )
                self.sentence_embeddings[i] = embedding
                embeddings.append(embedding)
                print(f"  Sentence {i+1}: Embedding generated. Shape: {embedding.shape}") # DEBUG
            except Exception as e:
                 print(f"  ERROR processing sentence {i+1} during training: {sentences[i]}")
                 print(f"  Error type: {type(e).__name__}, Message: {e}")
                 traceback.print_exc()
                 print("  Skipping embedding for this sentence.")

        if not embeddings:
             print("Error: No embeddings generated. Cannot proceed."); return self

        print("Learning meaning clusters...") # DEBUG
        self.learn_meaning_clusters(embeddings)
        print("Assigning meaning to clusters...") # DEBUG
        self.assign_meaning_to_clusters(sentences, structures, roles_list, analyses_list)
        print("--- Training Complete ---") # DEBUG
        return self

    def extract_semantic_frames(self, tokens, analyses, roles, camel_analyzer=None):
        """
        Extract semantic frames and deeper meaning from a sentence using various NLP tools.
        
        Args:
            tokens: List of tokens in the sentence
            analyses: List of (lemma, pos, dep_type, head_idx) tuples
            roles: Dictionary with grammatical roles
            camel_analyzer: Optional CAMeL Tools analyzer
            
        Returns:
            dict: Semantic frames and relationships
        """
        sentence = ' '.join(tokens)
        frames = []
        
        # Try to use AraVec if available for word embeddings
        aravec_model = None
        try:
            from gensim.models import Word2Vec
            from gensim.models.word2vec import Word2Vec
            # Try to load a pre-trained AraVec model if available
            model_path = "../aravec/tweet_cbow_300"  # User would need to set this
            if os.path.exists(model_path):
                aravec_model = Word2Vec.load(model_path)
        except:
            pass

        # Try to use Farasa NER if available
        farasa_ner = None
        try:
            from farasa.ner import FarasaNamedEntityRecognizer
            farasa_ner = FarasaNamedEntityRecognizer()
        except:
            pass
        
        # Extract verb information
        verb_idx = roles.get('verb')
        verb = None
        if verb_idx is not None and verb_idx < len(tokens):
            verb = tokens[verb_idx]
            
            # Define semantic frames based on verb classes
            motion_verbs = ["ذهب", "جاء", "مشى", "سار", "رجع", "دخل", "خرج"]
            possession_verbs = ["أخذ", "أعطى", "ملك", "وهب", "منح", "سلم"]
            communication_verbs = ["قال", "تكلم", "صرح", "أخبر", "سأل", "أجاب"]
            cognition_verbs = ["فكر", "اعتقد", "ظن", "علم", "فهم", "نسي"]
            emotion_verbs = ["أحب", "كره", "خاف", "فرح", "حزن"]
            
            # Get verb lemma if available
            verb_lemma = None
            if verb_idx < len(analyses):
                verb_lemma = analyses[verb_idx][0]
            
            # Add semantic frame based on verb class
            if verb_lemma:
                if verb_lemma in motion_verbs:
                    frames.append({
                        'type': 'MOTION',
                        'verb': verb,
                        'agent': tokens[roles.get('subject')] if roles.get('subject') is not None else None,
                        'destination': tokens[roles.get('object')] if roles.get('object') is not None else None
                    })
                elif verb_lemma in possession_verbs:
                    frames.append({
                        'type': 'POSSESSION',
                        'verb': verb,
                        'possessor': tokens[roles.get('subject')] if roles.get('subject') is not None else None,
                        'possessed': tokens[roles.get('object')] if roles.get('object') is not None else None
                    })
                elif verb_lemma in communication_verbs:
                    frames.append({
                        'type': 'COMMUNICATION',
                        'verb': verb,
                        'speaker': tokens[roles.get('subject')] if roles.get('subject') is not None else None,
                        'message': tokens[roles.get('object')] if roles.get('object') is not None else None
                    })
                elif verb_lemma in cognition_verbs:
                    frames.append({
                        'type': 'COGNITION',
                        'verb': verb,
                        'thinker': tokens[roles.get('subject')] if roles.get('subject') is not None else None,
                        'thought': tokens[roles.get('object')] if roles.get('object') is not None else None
                    })
                elif verb_lemma in emotion_verbs:
                    frames.append({
                        'type': 'EMOTION',
                        'verb': verb,
                        'experiencer': tokens[roles.get('subject')] if roles.get('subject') is not None else None,
                        'stimulus': tokens[roles.get('object')] if roles.get('object') is not None else None
                    })
        
        # Use CAMeL morphological analysis for additional semantic info
        if camel_analyzer:
            try:
                camel_analysis = camel_analyzer.analyze(sentence)
                
                # Extract deep semantic features
                semantic_properties = {
                    'tense': None,
                    'mood': None,
                    'aspect': None,
                    'definiteness': [],
                    'gender': {},
                    'number': {}
                }
                
                for i, token_analysis in enumerate(camel_analysis):
                    # Extract morphological features
                    if 'asp' in token_analysis:
                        semantic_properties['aspect'] = token_analysis['asp']
                    if 'mod' in token_analysis:
                        semantic_properties['mood'] = token_analysis['mod']
                    if 'gen' in token_analysis:
                        semantic_properties['gender'][tokens[i]] = token_analysis['gen']
                    if 'num' in token_analysis:
                        semantic_properties['number'][tokens[i]] = token_analysis['num']
                    if 'def' in token_analysis and token_analysis['def'] == 'D':
                        semantic_properties['definiteness'].append(tokens[i])
                
                # Find predicate-argument structure
                if verb_idx is not None:
                    verb_analysis = camel_analysis[verb_idx] if verb_idx < len(camel_analysis) else None
                    if verb_analysis and 'asp' in verb_analysis:
                        # Determine tense from aspect
                        if verb_analysis['asp'] == 'p':  # perfective
                            semantic_properties['tense'] = 'past'
                        elif verb_analysis['asp'] == 'i':  # imperfective
                            semantic_properties['tense'] = 'present'
                        elif verb_analysis['asp'] == 'c':  # command
                            semantic_properties['tense'] = 'imperative'
                
                # Add frame with semantic properties
                frames.append({
                    'type': 'SEMANTIC_PROPERTIES',
                    'properties': semantic_properties
                })
                
            except Exception as e:
                print(f"Error in CAMeL semantic analysis: {e}")
        
        # Use Farasa NER if available
        if farasa_ner:
            try:
                entities = farasa_ner.recognize(sentence)
                if entities:
                    frames.append({
                        'type': 'NAMED_ENTITIES',
                        'entities': entities
                    })
            except:
                pass
        
        # Use AraVec for word embeddings if available
        if aravec_model:
            try:
                # Get most similar words to content words
                similar_words = {}
                for i, (lemma, pos, _, _) in enumerate(analyses):
                    if pos in ['NOUN', 'VERB', 'ADJ'] and lemma in aravec_model:
                        similar = aravec_model.most_similar(lemma, topn=3)
                        similar_words[lemma] = [word for word, _ in similar]
                
                if similar_words:
                    frames.append({
                        'type': 'WORD_EMBEDDINGS',
                        'similar_words': similar_words
                    })
            except:
                pass
        
        return {
            'sentence': sentence,
            'frames': frames
        }
    def extract_enhanced_semantic_frames(self, tokens, analyses, roles, camel_analyzer=None):
        """More comprehensive semantic frame extraction"""
        # Get basic frames
        basic_frames = self.extract_semantic_frames(tokens, analyses, roles, camel_analyzer)
        
        # Add additional frames for complex sentences
        sentence = ' '.join(tokens)
        frames = basic_frames['frames'].copy()
        
        # 1. Extract rhetorical relations (cause, contrast, condition)
        rhetorical_markers = {
            'CAUSE': ['لأن', 'بسبب', 'نتيجة', 'لذلك'],
            'CONTRAST': ['لكن', 'بينما', 'رغم', 'مع ذلك', 'على الرغم'],
            'CONDITION': ['إذا', 'لو', 'شرط', 'في حال'],
            'TEMPORAL': ['قبل', 'بعد', 'أثناء', 'خلال', 'عندما'],
            'PURPOSE': ['كي', 'ل', 'من أجل', 'بهدف'],
        }
        
        for relation_type, markers in rhetorical_markers.items():
            for i, token in enumerate(tokens):
                if token in markers:
                    # Identify spans before and after the marker
                    before_span = ' '.join(tokens[:i])
                    after_span = ' '.join(tokens[i+1:])
                    
                    frames.append({
                        'type': f'RHETORICAL_{relation_type}',
                        'marker': token,
                        'span1': before_span,
                        'span2': after_span
                    })
        
        # 2. Extract nested predications (clauses within clauses)
        verb_indices = [i for i, (_, pos, _, _) in enumerate(analyses) if pos == 'VERB']
        if len(verb_indices) > 1:
            # Sort by dependency to find main and subordinate verbs
            verb_deps = [(i, analyses[i][2], analyses[i][3]) for i in verb_indices]
            main_verbs = []
            subordinate_verbs = []
            
            # Classify verbs as main or subordinate
            for i, dep_type, head_idx in verb_deps:
                if dep_type == 'root' or head_idx == -1:
                    main_verbs.append(i)
                else:
                    subordinate_verbs.append((i, head_idx))
            
            # Create nested predication frames
            for verb_idx in main_verbs:
                # Find all subordinate verbs that depend on this main verb
                related_subs = [sub_idx for sub_idx, head in subordinate_verbs if head == verb_idx]
                
                if related_subs:
                    verb_token = tokens[verb_idx]
                    sub_clauses = []
                    
                    for sub_idx in related_subs:
                        # Find span of the subordinate clause
                        sub_token = tokens[sub_idx]
                        # Simple heuristic: take from subordinate verb to next main verb or end
                        next_main = min([i for i in main_verbs if i > sub_idx] + [len(tokens)])
                        sub_span = ' '.join(tokens[sub_idx:next_main])
                        
                        sub_clauses.append({
                            'verb': sub_token,
                            'span': sub_span,
                            'index': sub_idx
                        })
                    
                    frames.append({
                        'type': 'NESTED_PREDICATION',
                        'main_verb': verb_token,
                        'main_index': verb_idx,
                        'subordinate_clauses': sub_clauses
                    })
        
        # 3. Extract coreference chains
        # Simple pronoun resolution for 3rd person pronouns
        pronouns = ['هو', 'هي', 'هم', 'هن', 'ه', 'ها', 'هما', 'هن']
        pronoun_indices = [i for i, token in enumerate(tokens) if token in pronouns]
        
        coref_chains = []
        for pronoun_idx in pronoun_indices:
            pronoun = tokens[pronoun_idx]
            
            # Find potential antecedents (nouns before the pronoun)
            potential_antecedents = []
            for i, (_, pos, _, _) in enumerate(analyses[:pronoun_idx]):
                if pos == 'NOUN' or pos == 'PROPN':
                    potential_antecedents.append((i, tokens[i]))
            
            if potential_antecedents:
                # Simple heuristic: closest matching noun by gender
                # More sophisticated approach would use CAMeL for gender matching
                antecedent = potential_antecedents[-1]
                
                coref_chains.append({
                    'pronoun': pronoun,
                    'pronoun_index': pronoun_idx,
                    'antecedent': antecedent[1],
                    'antecedent_index': antecedent[0]
                })
        
        if coref_chains:
            frames.append({
                'type': 'COREFERENCE',
                'chains': coref_chains
            })
        
        return {
            'sentence': sentence,
            'frames': frames
        }
    def save_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """
        Save the trained kernel to a file.
        
        Args:
            filename: File path to save to
        """
        # Create a dictionary with all the data to save
        model_data = {
            'embedding_dim': self.embedding_dim,
            'num_clusters': self.num_clusters,
            'meaning_clusters': self.meaning_clusters,
            'cluster_labels': self.cluster_labels,
            'meaning_map': self.meaning_map,
            'reference_sentences': self.reference_sentences,
            'circuit_embeddings': self.circuit_embeddings,
            'sentence_embeddings': self.sentence_embeddings,
            'semantic_templates': self.semantic_templates
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str = 'arabic_quantum_kernel.pkl'):
        """
        Load a trained kernel from a file.
        
        Args:
            filename: File path to load from
        """
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return self
            
        try:
            # Load from file
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model attributes
            self.embedding_dim = model_data.get('embedding_dim', self.embedding_dim)
            self.num_clusters = model_data.get('num_clusters', self.num_clusters)
            self.meaning_clusters = model_data.get('meaning_clusters')
            self.cluster_labels = model_data.get('cluster_labels')
            self.meaning_map = model_data.get('meaning_map', {})
            self.reference_sentences = model_data.get('reference_sentences', {})
            self.circuit_embeddings = model_data.get('circuit_embeddings', {})
            self.sentence_embeddings = model_data.get('sentence_embeddings', {})
            
            # Only update templates if they exist in the saved model
            if 'semantic_templates' in model_data:
                self.semantic_templates = model_data['semantic_templates']
                
            print(f"Model loaded from {filename}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            
        return self
    
    def visualize_meaning_space(self, highlight_indices=None, save_path=None):
        """
        Visualize the sentence meaning space using dimensionality reduction.
        
        Args:
            highlight_indices: Indices of sentences to highlight
            save_path: Path to save the visualization
        """
        from sklearn.decomposition import PCA
        
        if not self.sentence_embeddings:
            print("No embeddings available for visualization")
            return
            
        # Collect embeddings
        embeddings = list(self.sentence_embeddings.values())
        indices = list(self.sentence_embeddings.keys())
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot all points
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                 c=self.cluster_labels if self.cluster_labels is not None else 'blue',
                 cmap='viridis', alpha=0.6, s=100)
        
        # Highlight specific sentences if requested
        if highlight_indices is not None:
            highlight_idxs = [indices.index(i) for i in highlight_indices if i in indices]
            if highlight_idxs:
                plt.scatter(reduced_embeddings[highlight_idxs, 0], 
                           reduced_embeddings[highlight_idxs, 1],
                           c='red', s=150, edgecolor='white', zorder=10)
                
                # Add labels for highlighted points
                for idx in highlight_idxs:
                    sentence = self.reference_sentences.get(indices[idx], f"Sentence {indices[idx]}")
                    plt.annotate(sentence, 
                                (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        # Add legend
        if self.cluster_labels is not None:
            legend1 = plt.legend(*scatter.legend_elements(),
                                title="Meaning Clusters")
            plt.gca().add_artist(legend1)
        
        # Add labels for meaning clusters
        if self.meaning_clusters is not None:
            # Project cluster centers to 2D
            cluster_centers_2d = pca.transform(self.meaning_clusters)
            
            # Add cluster centers
            plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                       marker='*', s=300, c='white', edgecolor='black')
            
            # Add meaning labels
            for i, (x, y) in enumerate(cluster_centers_2d):
                if i in self.meaning_map:
                    meaning = self.meaning_map[i]['structure']
                    plt.annotate(f"Cluster {i}: {meaning}",
                                (x, y), xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))
        
        plt.title('Quantum Sentence Meaning Space')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Visualization saved to {save_path}")
        
        return plt.gcf()

    def analyze_quantum_states(self, circuits, tokens_list, save_path=None):
        """
        Analyze and visualize the quantum states of a set of circuits.
        
        Args:
            circuits: List of quantum circuits to analyze
            tokens_list: List of token lists for labels
            save_path: Base path to save visualizations
        """
        # Create a figure to compare quantum states
        fig, axes = plt.subplots(len(circuits), 1, figsize=(12, 5*len(circuits)))
        if len(circuits) == 1:
            axes = [axes]
            
        for i, (circuit, tokens) in enumerate(zip(circuits, tokens_list)):
            try:
                # Execute the circuit
                result = execute(circuit, self.simulator).result()
                
                # Get statevector if available
                if hasattr(result, 'get_statevector'):
                    statevector = result.get_statevector()
                    
                    # Plot state
                    sentence = ' '.join(tokens)
                    if len(sentence) > 50:
                        sentence = sentence[:47] + "..."
                    
                    # Create city plot in the current axis
                    plot_state_city(statevector, title=f"Quantum State for: {sentence}", 
                                   ax=axes[i])
                    
                    # Save individual plot if requested
                    if save_path:
                        plt.figure(figsize=(12, 8))
                        plot_state_city(statevector, 
                                      title=f"Quantum State for: {sentence}")
                        individual_path = f"{save_path}_circuit_{i}.png"
                        plt.savefig(individual_path, dpi=150)
                        plt.close()
                        
                else:
                    axes[i].text(0.5, 0.5, "Statevector not available for this circuit",
                                ha='center', va='center')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error visualizing circuit: {e}",
                            ha='center', va='center')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_all_circuits.png", dpi=150)
            
        return fig

    def format_discourse_relations(self, sentence, previous_sentence, discourse_relations):
        """Create a user-friendly description of discourse relations"""
        if not discourse_relations:
            return "No specific discourse relations detected."
        
        formatted_output = []
        for relation in discourse_relations:
            relation_type = relation['type']
            marker = relation['marker']
            
            # Create human-readable descriptions
            descriptions = {
                'CONTINUATION': f"This sentence continues the previous thought using '{marker}'",
                'CAUSE': f"This sentence shows a result or consequence of the previous sentence using '{marker}'",
                'CONTRAST': f"This sentence contrasts with the previous information using '{marker}'",
                'ELABORATION': f"This sentence elaborates on the previous information using '{marker}'",
                'EXAMPLE': f"This sentence provides an example of the previous concept using '{marker}'",
                'REFERENCE': f"This sentence refers back to the previous content using '{marker}'"
            }
            
            if relation_type in descriptions:
                formatted_output.append(descriptions[relation_type])
        
        return "\n".join(formatted_output)

    def generate_discourse_report(self, discourse_analyses):
        """Generate a full report of discourse analysis for all sentences"""
        report = "# Arabic Text Discourse Analysis\n\n"
        
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis['sentence']
            interpretation = analysis['interpretation']
            discourse_relations = interpretation.get('discourse_relations', [])
            
            report += f"## Sentence {i+1}\n"
            report += f"**Text:** {sentence}\n\n"
            
            # Add basic interpretation (whatever your current system provides)
            basic_interp = interpretation.get('basic_interpretation', {})
            if basic_interp:
                report += "**Basic Analysis:**\n"
                # Format your basic interpretation here
                # This will depend on what your _interpret_sentence method returns
                # Example:
                if 'sentence_type' in basic_interp:
                    report += f"- Sentence Type: {basic_interp['sentence_type']}\n"
                if 'main_meaning' in basic_interp:
                    report += f"- Main Meaning: {basic_interp['main_meaning']}\n"
                report += "\n"
            
            # Add discourse relations if not the first sentence
            if i > 0:
                previous_sentence = discourse_analyses[i-1]['sentence']
                report += "**Relationship to Previous Sentence:**\n"
                relation_text = self.format_discourse_relations(sentence, previous_sentence, discourse_relations)
                report += relation_text + "\n\n"
            
            # Visual separator
            report += "---\n\n"
        
        return report



    def generate_html_report(self, discourse_analyses):
        """Generate an HTML report with interactive elements"""
        html = """
        <!DOCTYPE html>
        <html dir="rtl">
        <head>
            <meta charset="UTF-8">
            <title>Arabic Quantum NLP Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .sentence { margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; }
                .sentence-text { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
                .analysis { margin-top: 10px; }
                .relation { color: #006699; margin-top: 5px; }
                .relation-marker { font-weight: bold; color: #cc0000; }
                .no-relation { color: #666; font-style: italic; }
            </style>
        </head>
        <body>
            <h1>تحليل النص العربي بالكم</h1>
        """
        
        for i, analysis in enumerate(discourse_analyses):
            sentence = analysis['sentence']
            interpretation = analysis['interpretation']
            discourse_relations = interpretation.get('discourse_relations', [])
            basic_interp = interpretation.get('basic_interpretation', {})
            
            html += f'<div class="sentence">'
            html += f'<div class="sentence-text">{sentence}</div>'
            
            # Basic interpretation
            html += '<div class="analysis">'
            if basic_interp:
                # Format your basic interpretation here
                if 'sentence_type' in basic_interp:
                    html += f'<div>نوع الجملة: {basic_interp["sentence_type"]}</div>'
                if 'main_meaning' in basic_interp:
                    html += f'<div>المعنى الرئيسي: {basic_interp["main_meaning"]}</div>'
            html += '</div>'
            
            # Discourse relations
            if i > 0:
                html += '<div class="discourse-section">'
                html += '<h3>العلاقة مع الجملة السابقة:</h3>'
                
                if discourse_relations:
                    for relation in discourse_relations:
                        relation_type = relation['type']
                        marker = relation['marker']
                        
                        # Arabic descriptions
                        descriptions = {
                            'CONTINUATION': f"تواصل هذه الجملة الفكرة السابقة باستخدام '<span class='relation-marker'>{marker}</span>'",
                            'CAUSE': f"تظهر هذه الجملة نتيجة أو عاقبة للجملة السابقة باستخدام '<span class='relation-marker'>{marker}</span>'",
                            'CONTRAST': f"تتناقض هذه الجملة مع المعلومات السابقة باستخدام '<span class='relation-marker'>{marker}</span>'",
                            'ELABORATION': f"توضح هذه الجملة المعلومات السابقة باستخدام '<span class='relation-marker'>{marker}</span>'",
                            'EXAMPLE': f"تقدم هذه الجملة مثالاً على المفهوم السابق باستخدام '<span class='relation-marker'>{marker}</span>'",
                            'REFERENCE': f"تشير هذه الجملة إلى المحتوى السابق باستخدام '<span class='relation-marker'>{marker}</span>'"
                        }
                        
                        if relation_type in descriptions:
                            html += f'<div class="relation">{descriptions[relation_type]}</div>'
                else:
                    html += '<div class="no-relation">لم يتم اكتشاف علاقات خطاب محددة.</div>'
                
                html += '</div>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html

# Example usage in a complete pipeline
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

# ============================================================
# Example Usage Block
# ============================================================
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
