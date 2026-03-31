# -*- coding: utf-8 -*-
import stanza
# Lambeq imports likely needed by camel_test2 functions
from lambeq import AtomicType, IQPAnsatz
from lambeq.backend.grammar import Ty, Box, Cup, Id, Spider, Swap, Diagram as GrammarDiagram
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import traceback
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator # Use AerSimulator directly
from qiskit.visualization import plot_histogram
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import logging # Import logging

# --- CENTRALIZED LOGGING CONFIGURATION ---
# Configure logging level and format ONCE here
logging.basicConfig(
    level=logging.INFO, # Set to INFO for less verbose output, DEBUG for more
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Optionally silence overly verbose libraries if needed
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING) # Silence Stanza info logs if desired

logger = logging.getLogger(__name__) # Logger for this script
logger.info("*****************************************")
logger.info("exp3.py starting - Logging configured.")
logger.info("*****************************************")
# --- END LOGGING CONFIGURATION ---


# --- IMPORT YOUR EXISTING FUNCTIONS ---
try:
    # Import the REVISED function from camel_test2.py
    from camel_test1.9.1 import arabic_to_quantum_enhanced_v3_morph as arabic_to_quantum_enhanced
    from camel_test1.9.1 import analyze_arabic_sentence # Keep if needed separately
    logger.info("Successfully imported functions from camel_test2.py")
except ImportError as e:
    logger.error(f"Could not import from camel_test2.py: {e}", exc_info=True)
    logger.critical("CRITICAL: Cannot proceed without camel_test2.py functions.")
    exit() # Exit if core functions can't be imported

# --- IMPORT KERNEL CLASS FROM v6.py ---
KERNEL_AVAILABLE = False
KERNEL_CLASS = None
try:
    from v6 import ArabicQuantumMeaningKernel
    KERNEL_CLASS = ArabicQuantumMeaningKernel # Store the class itself
    logger.info("Successfully imported ArabicQuantumMeaningKernel class from v6.py")
    KERNEL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Could not import ArabicQuantumMeaningKernel from v6.py: {e}", exc_info=True)
except Exception as e_kernel_import:
    logger.error(f"Error during import/setup related to v6.py: {e_kernel_import}", exc_info=True)

# Define dummy kernel if import failed
if not KERNEL_AVAILABLE:
    logger.warning("Defining dummy kernel methods as fallback.")
    class DummyKernel:
        def __init__(self, *args, **kwargs):
            self.embedding_dim = kwargs.get('embedding_dim', 20)
            self.cluster_labels = None
            self.sentence_embeddings = {}
            logger.warning("Using DummyKernel instance.")

        def _bind_parameters(self, circuit, tokens, analyses):
            logger.warning("Using dummy _bind_parameters.")
            if not hasattr(circuit, 'parameters') or not circuit.parameters: return None
            param_values = np.random.rand(len(circuit.parameters)) * 2 * np.pi
            return {p: v for p, v in zip(circuit.parameters, param_values)} # v6 expects dict[Param, float]

        def get_enhanced_circuit_features(self, circuit, tokens, analyses, debug=False):
            logger.warning("Using dummy get_enhanced_circuit_features. Returns random vector.")
            features = np.random.rand(self.embedding_dim)
            norm = np.linalg.norm(features)
            return features / norm if norm > 1e-9 else features

        def train(self, sentences, circuits, tokens_list, analyses_list, structures, roles_list, use_enhanced_clustering=False):
            logger.warning("Using dummy train method.")
            # Simulate embedding generation and clustering
            num_sentences = len(sentences)
            embeddings = []
            for i in range(num_sentences):
                 # Try to get features, handle potential errors from dummy circuit
                 try:
                     q_features = self.get_enhanced_circuit_features(circuits[i], tokens_list[i], analyses_list[i])
                     self.sentence_embeddings[i] = q_features # Store dummy embedding
                     embeddings.append(q_features)
                 except Exception:
                     logger.warning(f"Dummy feature extraction failed for sentence {i}. Skipping.")
                     self.sentence_embeddings[i] = np.zeros(self.embedding_dim) # Store zeros
                     embeddings.append(np.zeros(self.embedding_dim))

            if embeddings and len(embeddings) > 1:
                num_clusters = min(5, len(embeddings)) # Example cluster count
                self.cluster_labels = np.random.randint(0, num_clusters, size=len(embeddings))
                logger.info(f"Dummy kernel assigned random cluster labels (k={num_clusters}).")
            else:
                self.cluster_labels = np.zeros(len(embeddings), dtype=int) if embeddings else None
            return self

        def visualize_meaning_space(self, highlight_indices=None, save_path=None):
             logger.warning("Using dummy visualize_meaning_space. No plot generated.")
             return None

        # Add dummy methods for other functions called if needed
        def save_model(self, filename): logger.warning(f"Dummy save_model called for {filename}.")
        def analyze_quantum_states(self, circuits_dict, tokens_dict, analyses_dict, save_path_prefix=None): logger.warning("Dummy analyze_quantum_states called.")
        def generate_html_report(self, analyses): logger.warning("Dummy generate_html_report called."); return "<html><body>Dummy Report</body></html>"
        def generate_discourse_report(self, analyses): logger.warning("Dummy generate_discourse_report called."); return "# Dummy Report"

    KERNEL_CLASS = DummyKernel # Use the dummy class if import failed


# --- Analysis Helper Functions (Keep as they were in original exp3.py) ---
def analyze_diagram_structure(diagram, diagram_path=None):
    """Analyzes and optionally saves the diagram."""
    if diagram is None: return {"error": "No diagram generated"}
    # Ensure lambeq is available for type check if possible
    try: from lambeq.backend.grammar import Diagram as GrammarDiagram
    except ImportError: GrammarDiagram = None

    if GrammarDiagram and not isinstance(diagram, GrammarDiagram):
        logger.warning(f"Analysis: Expected GrammarDiagram, got {type(diagram)}")
        return {"error": f"Invalid diagram type: {type(diagram)}"}
    elif not GrammarDiagram and not hasattr(diagram, 'boxes'): # Basic check if lambeq unavailable
         logger.warning(f"Analysis: Diagram object seems invalid (type: {type(diagram)})")
         return {"error": f"Invalid diagram object type: {type(diagram)}"}

    try:
        metrics = {
            'n_boxes': len(diagram.boxes) if hasattr(diagram, 'boxes') else 'N/A',
            'dom': str(diagram.dom) if hasattr(diagram, 'dom') else 'N/A',
            'cod': str(diagram.cod) if hasattr(diagram, 'cod') else 'N/A',
            'is_sentence': (hasattr(diagram, 'cod') and diagram.cod == AtomicType.SENTENCE) if 'AtomicType' in globals() else 'N/A'
        }
        if diagram_path and hasattr(diagram, 'draw'):
            try:
                ax = diagram.draw(figsize=(12, 7))
                if ax and hasattr(ax, 'figure'):
                    fig = ax.figure
                    fig.savefig(diagram_path, bbox_inches='tight', dpi=150)
                    plt.close(fig) # Close the figure
                    logger.debug(f"Saved diagram to {diagram_path}")
                else:
                    logger.warning("Diagram draw() did not return valid axes.")
                    metrics["draw_error"] = "Draw failed to return axes"
            except Exception as e_draw:
                logger.warning(f"Could not draw/save diagram: {e_draw}")
                metrics["draw_error"] = str(e_draw)
        elif not hasattr(diagram, 'draw'):
             metrics["draw_error"] = "Diagram object has no draw method"

        return metrics
    except Exception as e:
        logger.error(f"Error analyzing diagram: {e}", exc_info=True)
        return {"error": str(e)}

def analyze_circuit_structure(circuit, circuit_path=None):
    """Analyzes and optionally saves the circuit structure."""
    if circuit is None: return {"error": "No circuit generated"}
    # Ensure qiskit is available for type check if possible
    try: from qiskit import QuantumCircuit
    except ImportError: QuantumCircuit = None

    if QuantumCircuit and not isinstance(circuit, QuantumCircuit):
        logger.warning(f"Analysis: Expected QuantumCircuit, got {type(circuit)}")
        return {"error": f"Invalid circuit type: {type(circuit)}"}
    elif not QuantumCircuit and not hasattr(circuit, 'num_qubits'): # Basic check
         logger.warning(f"Analysis: Circuit object seems invalid (type: {type(circuit)})")
         return {"error": f"Invalid circuit object type: {type(circuit)}"}

    try:
        metrics = {
            'num_qubits': circuit.num_qubits if hasattr(circuit, 'num_qubits') else 'N/A',
            'depth': circuit.depth() if hasattr(circuit, 'depth') else 'N/A',
            'ops': dict(circuit.count_ops()) if hasattr(circuit, 'count_ops') else {},
            'num_parameters': len(circuit.parameters) if hasattr(circuit, 'parameters') else 0,
        }
        if circuit_path and hasattr(circuit, 'draw'):
            try:
                # Use 'mpl' output, save manually
                fig = circuit.draw(output='mpl', fold=-1)
                if fig:
                     fig.savefig(circuit_path, bbox_inches='tight', dpi=150)
                     plt.close(fig) # Close the figure manually
                     logger.debug(f"Saved circuit plot to {circuit_path}")
                else:
                     logger.warning("Circuit draw() did not return a figure.")
                     metrics["draw_error"] = "Draw failed to return figure"

            except Exception as e_draw:
                logger.warning(f"Could not draw/save circuit: {e_draw}")
                metrics["draw_error"] = str(e_draw)
        elif not hasattr(circuit, 'draw'):
             metrics["draw_error"] = "Circuit object has no draw method"
        return metrics
    except Exception as e:
        logger.error(f"Error analyzing circuit: {e}", exc_info=True)
        return {"error": str(e)}

def visualize_set_results(set_results, set_name, kernel_instance, output_dir):
    """Visualize results for a specific experiment set."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"--- Starting Visualization for Set: {set_name} ---")

    embeddings = {}
    valid_indices = [] # Track indices *within the set* with valid embeddings
    original_sentences = []

    for i, result in enumerate(set_results):
        original_sentences.append(result["sentence"])
        # Check if embedding exists and is valid
        if "quantum_embedding" in result and isinstance(result["quantum_embedding"], list):
            emb_array = np.array(result["quantum_embedding"])
            if np.all(np.isfinite(emb_array)): # Check for NaN/Inf
                embeddings[i] = emb_array # Use index within set as key
                valid_indices.append(i)
            else: logger.warning(f"Skipping non-finite embedding for sentence {i} in set {set_name}")
        else: logger.debug(f"No valid embedding for sentence {i} in set {set_name}")

    # 1. Similarity heatmap
    if len(valid_indices) > 1:
        sim_matrix_data = np.zeros((len(valid_indices), len(valid_indices)))
        valid_sentences = [original_sentences[i] for i in valid_indices]

        for i1_idx, set_idx1 in enumerate(valid_indices):
            for i2_idx, set_idx2 in enumerate(valid_indices):
                if i1_idx == i2_idx:
                    sim_matrix_data[i1_idx, i2_idx] = 1.0
                elif i1_idx < i2_idx:
                    emb1 = embeddings[set_idx1]
                    emb2 = embeddings[set_idx2]
                    try:
                        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
                        sim_matrix_data[i1_idx, i2_idx] = sim
                        sim_matrix_data[i2_idx, i1_idx] = sim # Symmetric matrix
                    except Exception as e_sim:
                        logger.error(f"Error comparing embeddings for indices {set_idx1}, {set_idx2} in set {set_name}: {e_sim}")
                        sim_matrix_data[i1_idx, i2_idx] = np.nan # Mark error as NaN
                        sim_matrix_data[i2_idx, i1_idx] = np.nan

        # Plot heatmap
        plt.figure(figsize=(max(8, len(valid_indices)*0.7), max(6, len(valid_indices)*0.6))) # Adjusted size
        plt.imshow(sim_matrix_data, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title(f"Embedding Similarity Heatmap - {set_name}")
        # Use helper for potentially shaping Arabic labels if needed
        try: from camel_test2 import shape_arabic_text
        except ImportError: shape_arabic_text = lambda x: x # Dummy function if import fails
        short_labels = [shape_arabic_text(s[:20] + '...' if len(s) > 20 else s) for s in valid_sentences]
        plt.xticks(range(len(short_labels)), short_labels, rotation=55, ha='right', fontsize=8) # Adjusted rotation/size
        plt.yticks(range(len(short_labels)), short_labels, fontsize=8)
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f"{set_name}_similarity_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        logger.info(f"Saved heatmap: {heatmap_path}")
    elif len(valid_indices) <= 1:
        logger.warning(f"Skipping heatmap for set '{set_name}': Not enough valid embeddings ({len(valid_indices)}).")

    # 2. PCA Meaning Space Visualization (using the kernel instance for this set)
    if kernel_instance and hasattr(kernel_instance, 'visualize_meaning_space'):
        pca_path = os.path.join(output_dir, f"{set_name}_meaning_space_pca.png")
        logger.info(f"Generating PCA plot for set '{set_name}'...")
        # The kernel's visualize_meaning_space uses its internal sentence_embeddings and cluster_labels
        # which were set during kernel.train() for this specific set.
        try:
            pca_fig = kernel_instance.visualize_meaning_space(save_path=pca_path)
            if pca_fig:
                plt.close(pca_fig) # Close the figure returned by the method
                logger.info(f"Saved PCA plot: {pca_path}")
            else:
                 logger.warning(f"PCA visualization for set '{set_name}' did not return a figure.")
        except Exception as e_pca:
            logger.error(f"Error generating PCA plot for set '{set_name}': {e_pca}", exc_info=True)

    logger.info(f"--- Visualization finished for Set: {set_name} ---")


# --- Main Experiment Runner ---

if __name__ == "__main__":
    logger.info("Running Revised QNLP Experiment Script (exp3.py with per-set clustering)...")

    # --- Define Test Cases ---
    experiment_sets = {
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

    # --- Output Directory ---
    output_dir = "qnlp_experiments_output_revised_v3_per_set_iqp_l3" # New output dir name
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Path to Word Embedding Model (Required by Kernel) ---
    # !!! USER: SET THE CORRECT PATH TO YOUR MODEL FILE HERE !!!
    embedding_model_path_for_kernel = "aravec/tweets_cbow_300"
    # embedding_model_path_for_kernel = None # Set to None to use hash-fallback only
    if embedding_model_path_for_kernel and not os.path.exists(embedding_model_path_for_kernel):
        logger.warning(f"Embedding model path specified but not found: {embedding_model_path_for_kernel}")
        logger.warning("Kernel will use hash-based parameter fallback.")
        embedding_model_path_for_kernel = None # Force fallback
    elif embedding_model_path_for_kernel:
         logger.info(f"Using embedding model: {embedding_model_path_for_kernel}")
    else:
         logger.info("No embedding model path specified. Using hash-based parameter fallback.")

    # --- Run Experiments ---
    choose_ansatz = 'iqp'
    all_results = {}
    total_sentences = sum(len(s) for s in experiment_sets.values())
    processed_count = 0

    for set_name, sentences in experiment_sets.items():
        logger.info(f"\n{'='*15} Running Experiment Set: {set_name} {'='*15}")
        set_results_list = [] # Store dicts for this set's results
        set_circuits = []
        set_tokens_list = []
        set_analyses_list = []
        set_structures = []
        set_roles_list = []
        set_indices_map = {} # Map original sentence index to index within this set

        set_output_dir = os.path.join(output_dir, set_name) # Subdirectory for set outputs
        os.makedirs(set_output_dir, exist_ok=True)

        for i, sentence in enumerate(sentences):
            processed_count += 1
            logger.info(f"--- Processing Sentence {processed_count}/{total_sentences} (Set: {set_name}, Idx: {i}): '{sentence[:50]}...' ---")
            # Use a unique prefix including the set name for file outputs
            sentence_prefix = f"{set_name}_{i}"
            result_data = {"sentence": sentence, "set_index": i} # Store index within the set

            try:
                # 1. Generate diagram and circuit using REVISED pipeline from camel_test2
                circuit, diagram, structure, tokens, analyses, roles = \
                    arabic_to_quantum_enhanced(sentence, ansatz_choice = choose_ansatz, debug=False, output_dir=set_output_dir) # Save intermediate diagrams/circuits per set

                result_data.update({
                    "structure": structure,
                    "tokens": tokens,
                    # Omitting analyses/roles from JSON for brevity, but they are available if needed
                })

                # 2. Analyze Diagram Structure
                diagram_path = os.path.join(set_output_dir, f"{sentence_prefix}_diagram_final.png")
                result_data["diagram_metrics"] = analyze_diagram_structure(diagram, diagram_path)

                # 3. Analyze Circuit Structure
                circuit_path = os.path.join(set_output_dir, f"{sentence_prefix}_circuit_final.png")
                result_data["circuit_metrics"] = analyze_circuit_structure(circuit, circuit_path)

                # 4. Prepare data for kernel training (only if circuit generation succeeded)
                circuit_metrics = result_data["circuit_metrics"]
                if circuit is not None and isinstance(circuit_metrics, dict) and "error" not in circuit_metrics:
                    current_set_index = len(set_circuits) # Index within this set's valid data
                    set_indices_map[i] = current_set_index # Map original index 'i' to set index

                    set_circuits.append(circuit)
                    set_tokens_list.append(tokens)
                    set_analyses_list.append(analyses)
                    set_structures.append(structure)
                    set_roles_list.append(roles)
                    result_data["set_processing_status"] = "Success"
                else:
                    logger.warning(f"  Skipping kernel processing for sentence {i} due to circuit generation failure.")
                    result_data["set_processing_status"] = "Circuit Error"
                    result_data["quantum_embedding"] = None
                    result_data["cluster_label"] = None

            except Exception as e_main:
                logger.error(f"!!! Top-level ERROR processing sentence: '{sentence}' !!!", exc_info=True)
                result_data["error"] = str(e_main)
                result_data["quantum_embedding"] = None
                result_data["cluster_label"] = None

            set_results_list.append(result_data) # Add result dict to the list for this set

        # --- Process this set with its own Kernel instance ---
        if not set_circuits:
            logger.warning(f"No valid circuits generated for set '{set_name}'. Skipping kernel training and analysis for this set.")
            all_results[set_name] = set_results_list # Store results even if kernel part failed
            continue # Move to the next experiment set

        logger.info(f"\n--- Instantiating and Training Kernel for Set: {set_name} ---")
        num_valid_sentences = len(set_circuits)
        kernel_clusters = min(5, max(1, num_valid_sentences // 2)) # Adjust cluster count based on set size

        # Ensure KERNEL_CLASS is callable (either the real one or the dummy)
        if not callable(KERNEL_CLASS):
             logger.error(f"KERNEL_CLASS is not callable (type: {type(KERNEL_CLASS)}). Cannot proceed.")
             all_results[set_name] = set_results_list # Store results
             continue

        try:
            kernel_for_set = KERNEL_CLASS(
                embedding_dim=30,
                num_clusters=kernel_clusters,
                embedding_model_path=embedding_model_path_for_kernel,
                params_per_word=3,
                shots=8192
            )

            # Train the kernel instance specifically on this set's data
            kernel_for_set.train(
                sentences=[s['sentence'] for i, s in enumerate(set_results_list) if i in set_indices_map], # Only pass sentences with valid circuits
                circuits=set_circuits,
                tokens_list=set_tokens_list,
                analyses_list=set_analyses_list,
                structures=set_structures,
                roles_list=set_roles_list,
                use_enhanced_clustering=True # Or False, depending on your preference/needs
            )

            # Extract embeddings and cluster labels from the trained kernel
            logger.info(f"  Extracting embeddings and cluster labels for set '{set_name}'...")
            for original_idx, set_idx in set_indices_map.items():
                if set_idx < len(kernel_for_set.sentence_embeddings):
                    embedding = kernel_for_set.sentence_embeddings.get(set_idx)
                    if embedding is not None and np.all(np.isfinite(embedding)):
                        set_results_list[original_idx]["quantum_embedding"] = embedding.tolist()
                    else:
                        logger.warning(f"  Invalid embedding found for set index {set_idx} (original index {original_idx}).")
                        set_results_list[original_idx]["quantum_embedding"] = None # Mark as None if invalid

                    if kernel_for_set.cluster_labels is not None and set_idx < len(kernel_for_set.cluster_labels):
                        set_results_list[original_idx]["cluster_label"] = int(kernel_for_set.cluster_labels[set_idx]) # Add cluster label
                    else:
                        set_results_list[original_idx]["cluster_label"] = None
                else:
                     logger.warning(f"  Set index {set_idx} out of bounds for embeddings/labels.")
                     set_results_list[original_idx]["quantum_embedding"] = None
                     set_results_list[original_idx]["cluster_label"] = None

            # Generate visualizations for this set using its dedicated kernel
            visualize_set_results(set_results_list, set_name, kernel_for_set, set_output_dir)

            # Optional: Save the kernel instance for this set if needed
            # kernel_for_set.save_model(os.path.join(set_output_dir, f'{set_name}_kernel.pkl'))

        except Exception as e_kernel_set:
            logger.error(f"Error during kernel processing for set '{set_name}': {e_kernel_set}", exc_info=True)
            # Mark remaining results in the set as having a kernel error
            for i in range(len(set_results_list)):
                 if "set_processing_status" not in set_results_list[i]: # Avoid overwriting previous errors
                      set_results_list[i]["set_processing_status"] = "Kernel Error"
                      set_results_list[i]["quantum_embedding"] = None
                      set_results_list[i]["cluster_label"] = None

        # Store the results for this set
        all_results[set_name] = set_results_list


    # --- Final Summary (Aggregated across sets) ---
    logger.info(f"\n{'='*10} Aggregated Experiment Metrics Summary {'='*10}")
    circuits_generated_count = 0
    embeddings_extracted_count = 0
    total_processed_for_summary = 0
    sentences_with_clusters = 0

    for set_name, results in all_results.items():
        logger.info(f"\n--- Summary for Set: {set_name} ---")
        set_circ_ok = 0
        set_embed_ok = 0
        set_clust_ok = 0
        for i, res in enumerate(results):
            total_processed_for_summary += 1
            logger.debug(f"  Sentence {i}: {res['sentence']}") # Use debug level for less clutter
            if res.get("error"):
                logger.info(f"    Status: Top-Level ERROR ({res['error']})")
                continue
            if res.get("set_processing_status") == "Kernel Error":
                 logger.info(f"    Status: Kernel Processing ERROR")
                 # Check if circuit was okay before kernel error
                 circuit_metrics = res.get('circuit_metrics', {})
                 circuit_ok_before_kernel = isinstance(circuit_metrics, dict) and "error" not in circuit_metrics
                 if circuit_ok_before_kernel: set_circ_ok += 1
                 continue # Skip further checks if kernel failed

            circuit_metrics = res.get('circuit_metrics', {})
            circuit_ok = isinstance(circuit_metrics, dict) and "error" not in circuit_metrics
            if circuit_ok:
                 set_circ_ok += 1
                 logger.debug(f"    Circuit OK: Qubits={circuit_metrics.get('num_qubits', 'N/A')}, Depth={circuit_metrics.get('depth', 'N/A')}")
            else:
                 logger.info(f"    Circuit Generation Failed: {circuit_metrics.get('error', 'Unknown')}")

            if "quantum_embedding" in res and isinstance(res["quantum_embedding"], list):
                set_embed_ok += 1
                logger.debug(f"    Embedding Extracted: Yes")
                if "cluster_label" in res and res["cluster_label"] is not None:
                    set_clust_ok += 1
                    logger.debug(f"    Cluster Assigned: {res['cluster_label']}")
                else:
                    logger.debug(f"    Cluster Assigned: No / Failed")
            elif res.get("set_processing_status") == "Circuit Error":
                 logger.debug(f"    Embedding Extracted: No (Circuit Error)")
            else: # Includes cases where embedding was None or invalid
                 logger.debug(f"    Embedding Extracted: No / Failed")

        logger.info(f"  Set '{set_name}' Summary: Processed={len(results)}, Circuits OK={set_circ_ok}, Embeddings OK={set_embed_ok}, Clustered OK={set_clust_ok}")
        circuits_generated_count += set_circ_ok
        embeddings_extracted_count += set_embed_ok
        sentences_with_clusters += set_clust_ok


    logger.info(f"\n--- Overall Aggregated Summary ---")
    logger.info(f"Total sentences processed: {total_processed_for_summary}")
    logger.info(f"Successfully generated circuits: {circuits_generated_count}")
    logger.info(f"Successfully extracted embeddings: {embeddings_extracted_count}")
    logger.info(f"Successfully assigned cluster labels: {sentences_with_clusters}")


    logger.info("\n--- Experiments Finished ---")

    # Save detailed results (now includes cluster labels per set)
    results_path = os.path.join(output_dir, 'experiment_results_revised_v3_clustered.json') # New filename
    logger.info(f"Saving detailed results to {results_path}...")
    try:
        # Custom encoder to handle numpy arrays if necessary
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # Handle Qiskit Parameter objects if they sneak in (shouldn't normally)
                if 'Parameter' in str(type(obj)): return str(obj)
                # Handle basic numpy types
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32,
                                      np.float64)):
                    # Check for NaN/Inf before converting
                    if not np.isfinite(obj): return None # Replace non-finite floats with null
                    return float(obj)
                elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                    # Check for NaN/Inf in complex numbers
                    if not np.isfinite(obj.real) or not np.isfinite(obj.imag): return None
                    return {'real': obj.real, 'imag': obj.imag}
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.void)):
                    return None
                return json.JSONEncoder.default(self, obj)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"Detailed results saved successfully.")
    except Exception as e_json:
        logger.error(f"Error saving detailed results to JSON: {e_json}", exc_info=True)

    logger.info("Script finished.")

