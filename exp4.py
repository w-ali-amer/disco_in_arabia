# -*- coding: utf-8 -*-
import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any, Union
import re
import copy

# --- CRITICAL: Import common_qnlp_types FIRST ---
# This ensures N_ARABIC, S_ARABIC, and global Box functors are defined
# before any other module in your project tries to use or define them.
try:
    import common_qnlp_types # This will execute common_qnlp_types.py
    if not common_qnlp_types.LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        # This log might appear before full logger setup if common_qnlp_types fails early.
        print("EXP4_CRITICAL_WARNING: common_qnlp_types reported that Lambeq types were NOT initialized successfully. Expect widespread issues.")
    else:
        # This log might also appear before full logger setup.
        print("EXP4_INFO: common_qnlp_types imported and hopefully initialized successfully.")
except ImportError as e_common_init:
    print(f"EXP4_CRITICAL_ERROR: Could not import common_qnlp_types.py: {e_common_init}. The system cannot proceed.")
    # Depending on your setup, you might want to sys.exit(1) here.
    # For now, define dummies to allow the rest of the file to be parsed without NameErrors,
    # but the pipeline will not function.
    class DummyArabicQuantumMeaningKernel:
        def __init__(self, *args, **kwargs): pass
    def dummy_prepare_quantum_nlp_pipeline_v8(*args, **kwargs): return None, [], None
    ArabicQuantumMeaningKernel = DummyArabicQuantumMeaningKernel # type: ignore
    prepare_quantum_nlp_pipeline_v8 = dummy_prepare_quantum_nlp_pipeline_v8 # type: ignore
    PIPELINE_AVAILABLE = False

# --- Import the pipeline function from v8.py ---
try:
    from v8 import prepare_quantum_nlp_pipeline_v8, ArabicQuantumMeaningKernel
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    logging.error(f"FATAL: Could not import pipeline from v8.py: {e}", exc_info=True)
    PIPELINE_AVAILABLE = False
    def prepare_quantum_nlp_pipeline_v8(*args, **kwargs): # type: ignore
        logging.error("Dummy prepare_quantum_nlp_pipeline_v8 called.")
        return None, [], None
    class ArabicQuantumMeaningKernel: # type: ignore
        def __init__(self, *args, **kwargs): self.reference_sentences = {}; self.circuit_embeddings = {}; self.sentence_embeddings = {} # type: ignore
        def visualize_meaning_space(self, *args, **kwargs): logging.warning("Dummy visualize_meaning_space called.") # type: ignore

# --- Qiskit/Sklearn/Arabic Helpers ---
try:
    from qiskit import QuantumCircuit # type: ignore
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    class QuantumCircuit: pass # type: ignore

try:
    from sklearn.metrics import pairwise # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
     SKLEARN_AVAILABLE = False
     class pairwise: # type: ignore
         @staticmethod
         def cosine_similarity(X): return np.identity(X.shape[0]) if isinstance(X, np.ndarray) else np.array([]) # type: ignore

try:
    import arabic_reshaper # type: ignore
    from bidi.algorithm import get_display # type: ignore
    ARABIC_DISPLAY_ENABLED = True
    def shape_arabic_text(text): # Defined here for use in this script's visualizations
        if not ARABIC_DISPLAY_ENABLED or not text or not isinstance(text, str): return text
        if any('\u0600' <= char <= '\u06FF' for char in text):
            try: return get_display(arabic_reshaper.reshape(text))
            except Exception as e: logging.warning(f"Arabic reshape error: {e}"); return text
        return text
except ImportError:
    ARABIC_DISPLAY_ENABLED = False
    def shape_arabic_text(text): return text


# --- Configure Logging ---
# Ensure this is configured. If exp4.py is the main entry point, this is crucial.
# If v8.py or another script is the entry point and configures logging, this might be redundant
# but it's safer to have it here if exp4.py can be run standalone.
log_filename = 'qnlp_experiment.txt' # Centralized log file name
log_level = logging.DEBUG # Set to DEBUG for detailed logs, INFO for less

# Clears all handlers from the root logger to prevent duplicate outputs if script is re-run
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'), # Overwrite log file each run
        logging.StreamHandler() # Output to console
    ]
)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Silence matplotlib's own INFO/DEBUG logs
logger = logging.getLogger("exp4_runner") # Logger for this specific script (exp4.py)

if PIPELINE_AVAILABLE: logger.info("Imported pipeline and kernel from v8.py")
else: logger.warning("Using dummy pipeline and kernel due to import error from v8.py.")
logger.info(f"Logging configured for exp4_runner. Level: {logging.getLevelName(logger.getEffectiveLevel())}. Output to console and '{log_filename}'")


# --- Matplotlib Font Configuration (copied from camel_test2.py for consistency) ---
try:
    plt.rcParams['font.family'] = 'sans-serif'
    # Add fonts known to support Arabic. Matplotlib will try them in order.
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Arial', 'Amiri', 'Noto Naskh Arabic']
    logger.info(f"Configured Matplotlib font.sans-serif: {plt.rcParams['font.sans-serif']}")
except Exception as e_font:
    logger.warning(f"Could not set Matplotlib font configuration: {e_font}")

# ============================================
# Helper Function: Sanitize Filename
# ============================================
def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a string to be safe for use as a filename.
    Removes or replaces potentially problematic characters.
    """
    sanitized = filename.strip()
    # Replace common problematic characters (whitespace, slashes, colons, etc.) with underscores
    sanitized = re.sub(r'[\s/\\:]+', '_', sanitized)
    # Remove any characters that are not alphanumeric, underscore, hyphen, or period
    sanitized = re.sub(r'[^\w\-_\.]', '', sanitized)
    # Replace multiple underscores with a single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized if sanitized else "sanitized_empty_filename"

# ============================================
# Helper Function: Numpy Encoder for JSON
# ============================================
class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy types and other non-serializable objects. """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)): # type: ignore
            return obj.item()
        elif isinstance(obj, np.ndarray): # type: ignore
            return obj.tolist()
        elif hasattr(obj, 'to_json') and callable(obj.to_json):
            # lambeq/discopy Diagram objects have to_json()
            try:
                return obj.to_json()
            except Exception:
                return f"<{type(obj).__name__}: serialization_failed>"
        elif hasattr(obj, 'boxes') and hasattr(obj, 'dom'):
            # Fallback for diagram-like objects without to_json
            return f"<Diagram: {str(obj)[:80]}>"
        else:
            # Last resort: stringify anything else rather than crashing
            try:
                return str(obj)
            except Exception:
                return f"<{type(obj).__name__}: unserializable>"

# ============================================
# Helper Function: Save Results to JSON
# ============================================
def save_results_to_json(data: Dict, output_directory: str, filename: str = "results.json"):
    """ Saves dictionary data to a JSON file with robust error handling. """
    os.makedirs(output_directory, exist_ok=True)
    filepath = os.path.join(output_directory, sanitize_filename(filename))
    logger.info(f"Attempting to save results to JSON: {filepath}")
    try:
        # Create a deep copy to avoid modifying the original data structure during serialization prep
        data_to_serialize = copy.deepcopy(data)

        # Specifically process 'results_data' if it exists and is a list (expected format)
        if 'results_data' in data_to_serialize and isinstance(data_to_serialize['results_data'], list):
            for item_idx, item in enumerate(data_to_serialize['results_data']):
                if isinstance(item, dict):
                    # Remove the 'circuit_object' as it's not directly JSON serializable
                    if 'circuit_object' in item:
                        del item['circuit_object']
                        logger.debug(f"Removed 'circuit_object' from item {item_idx} for JSON serialization.")
                    # 'circuit_qasm_for_json' should already be a string or None, suitable for JSON.
                else:
                    logger.warning(f"Item {item_idx} in 'results_data' is not a dict, skipping circuit object removal.")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_serialize, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"Successfully saved results to {filepath}")
    except TypeError as te:
        logger.error(f"TypeError during JSON serialization for {filepath}. Data might still contain non-serializable types. Error: {te}", exc_info=True)
    except Exception as e:
        logger.error(f"General error saving results to {filepath}: {e}", exc_info=True)

def analyze_circuit_metrics(
    circuits_data: List[Dict[str, Any]], # Each dict: {'id': str, 'label': str, 'circuit': QuantumCircuit}
    output_dir: str,
    filename_prefix: str = "circuit_metrics"
):
    """
    Analyzes a list of Qiskit circuits for common metrics and saves a plot/table.
    """
    if not QISKIT_AVAILABLE:
        logger.error("Circuit metrics: Qiskit not available.")
        return

    metrics_data = []
    for item in circuits_data:
        circuit_id = item.get('id', 'UnknownID')
        label = item.get('label', 'UnknownLabel')
        circuit = item.get('circuit')

        if not isinstance(circuit, QuantumCircuit):
            logger.warning(f"Skipping item {circuit_id} ('{label}'): Not a valid Qiskit circuit.")
            continue

        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        
        # Count CNOTs (cx gates)
        num_cnots = 0
        if hasattr(circuit, 'count_ops'): # Newer Qiskit
            ops = circuit.count_ops()
            num_cnots = ops.get('cx', 0)
        else: # Older Qiskit might need to iterate
            for instruction, _, _ in circuit.data:
                if instruction.name == 'cx':
                    num_cnots += 1
        
        num_params = len(circuit.parameters)
        
        # Identify variational params (those starting with _anc_ if you adopt that naming)
        # This is specific to your VariationalSenseIQPAnsatz naming,
        # adjust if ControlledSenseIQPAnsatz (fixed ancilla) is used.
        # For fixed ancilla, all params are feature-bound.
        num_variational_params = 0
        if hasattr(circuit, 'name') and "var_amb" in circuit.name: # Check if it's from VariationalSenseIQPAnsatz
             for param in circuit.parameters:
                 if "_anc_" in param.name:
                     num_variational_params +=1
        num_feature_bound_params = num_params - num_variational_params


        metrics_data.append({
            'ID': circuit_id,
            'Label': label,
            'Circuit Name': circuit.name,
            'Qubits': num_qubits,
            'Depth': depth,
            'CNOTs': num_cnots,
            'Total Params': num_params,
            'Feature-Bound Params': num_feature_bound_params, # For current H3.1, this is same as Total Params
            'Variational Params': num_variational_params # For current H3.1, this is 0
        })

    if not metrics_data:
        logger.info("No valid circuits found to analyze metrics.")
        return

    df_metrics = pd.DataFrame(metrics_data)
    logger.info("\nCircuit Metrics Summary:\n" + df_metrics.to_string())

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{filename_prefix}_summary.csv")
    df_metrics.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Circuit metrics saved to CSV: {csv_path}")

    # Create a plot (example: bar chart of qubits and depth)
    try:
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(df_metrics) * 0.5), 10), sharex=True)
        
        indices = np.arange(len(df_metrics))
        bar_width = 0.35

        axes[0].bar(indices, df_metrics['Qubits'], bar_width, label='Qubits', color='skyblue')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Circuit Qubits')
        axes[0].legend()

        axes[1].bar(indices, df_metrics['Depth'], bar_width, label='Depth', color='lightcoral')
        axes[1].bar(indices + bar_width, df_metrics['CNOTs'], bar_width, label='CNOTs', color='lightgreen')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Circuit Depth and CNOTs')
        axes[1].legend()
        
        # Use a shortened version of 'Label' or 'ID' for x-ticks if labels are too long
        tick_labels = [f"{row['ID']}_{row['Label'][:15]}" for index, row in df_metrics.iterrows()]
        plt.xticks(indices + bar_width / 2, tick_labels, rotation=45, ha="right", fontsize=8)
        
        fig.suptitle(f'Circuit Metrics Comparison: {filename_prefix}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        
        plot_path = os.path.join(output_dir, f"{filename_prefix}_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Circuit metrics plot saved to: {plot_path}")
    except Exception as e_plot:
        logger.error(f"Error generating metrics plot: {e_plot}", exc_info=True)
        if 'fig' in locals() and plt.gcf().get_axes(): plt.close(fig)

# ============================================
# Helper Function for Heatmap Visualization (Adapted from v8.py)
# ============================================
def generate_similarity_heatmap(embeddings: Dict[int, np.ndarray], # Dict mapping original_index to embedding
                                sentences_dict: Dict[int, str], # Dict mapping original_index to sentence string
                                title: str,
                                save_path: str):
    """Generates and saves a cosine similarity heatmap for given embeddings."""
    logger.info(f"Generating similarity heatmap: {title}")
    if not SKLEARN_AVAILABLE: logger.error("Skipping heatmap: scikit-learn not available."); return
    if not embeddings or len(embeddings) < 2: logger.warning(f"Skipping heatmap '{title}': Not enough embeddings provided (need at least 2)."); return

    # Ensure consistent ordering and filter out any missing pairs
    # Use original_index from the results as keys for embeddings and sentences
    valid_indices = sorted([idx for idx in embeddings if idx in sentences_dict and isinstance(embeddings[idx], np.ndarray) and np.all(np.isfinite(embeddings[idx]))])

    if len(valid_indices) < 2:
        logger.warning(f"Skipping heatmap '{title}': Not enough valid/paired embeddings after filtering ({len(valid_indices)})."); return

    embeddings_list = [embeddings[idx] for idx in valid_indices]
    labels_list = [sentences_dict[idx] for idx in valid_indices]

    X = np.array(embeddings_list)
    if X.ndim != 2 or X.shape[0] < 2:
        logger.error(f"Embeddings for heatmap '{title}' are not suitable for similarity calculation. Shape: {X.shape}"); return

    try:
        similarity_matrix = pairwise.cosine_similarity(X)
    except Exception as e_sim:
        logger.error(f"Error calculating cosine similarity for '{title}': {e_sim}", exc_info=True); return

    fig, ax = plt.subplots(figsize=(max(8, len(labels_list)*0.7), max(6, len(labels_list)*0.6)))
    im = ax.imshow(similarity_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title(shape_arabic_text(title), fontsize=12) # Shape title for Arabic

    # Prepare short, shaped labels for ticks
    short_labels = [shape_arabic_text(s[:25] + ('...' if len(s) > 25 else '')) for s in labels_list]
    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    fig.tight_layout(pad=1.5) # Add padding to ensure labels/title are not cut off
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved heatmap: {save_path}")
    except Exception as e_save_heatmap:
        logger.error(f"Failed to save heatmap '{save_path}': {e_save_heatmap}", exc_info=True)
    finally:
        plt.close(fig)


# ============================================
# Helper Function for Circuit Visualization (Adapted from camel_test2.py)
# ============================================
def visualize_circuit_matplotlib(circuit: Any, # Expected: Qiskit QuantumCircuit
                                 label: str, # Used for filename and potentially title elements
                                 save_path: str):
    """Visualizes a Qiskit QuantumCircuit using matplotlib and saves it."""
    if not QISKIT_AVAILABLE: logger.warning(f"Skipping circuit viz for '{label}': Qiskit not available."); return
    if not isinstance(circuit, QuantumCircuit): # Check if it's the Qiskit QuantumCircuit
        logger.warning(f"Skipping circuit viz for '{label}': Not a Qiskit QuantumCircuit (type: {type(circuit)})."); return
    if not hasattr(circuit, 'draw'):
        logger.warning(f"Skipping circuit viz for '{label}': Circuit object lacks 'draw' method."); return

    logger.debug(f"Visualizing circuit: {label}")
    fig = None
    try:
        # Use Qiskit's draw method with 'mpl' backend
        fig = circuit.draw(output='mpl', fold=-1, scale=0.7)
        if fig: # Check if draw returned a figure
            # Add a shaped title to the figure
            circuit_name_for_title = getattr(circuit, 'name', label) # Use circuit.name if available, else label
            fig.suptitle(shape_arabic_text(f"الدارة الكمومية: {circuit_name_for_title}"), fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.debug(f"Saved circuit diagram to {save_path}")
        else:
            logger.warning(f"Qiskit circuit.draw(output='mpl') returned None for '{label}'. Cannot save figure.")
    except Exception as e_draw:
        logger.error(f"Error drawing/saving circuit diagram for '{label}': {e_draw}", exc_info=True)
    finally:
        if fig: # Ensure the figure is closed if it was created
            plt.close(fig)


# --- Main Experiment Runner ---
if __name__ == "__main__":
    logger.info("="*20)
    logger.info("Starting QNLP Experiment Runner (using v8 Pipeline)")
    logger.info("="*20)

    if not PIPELINE_AVAILABLE:
        logger.critical("Cannot proceed without the pipeline function from v8.py. Exiting.")
        # exit() # Use sys.exit(1) for a non-zero exit code indicating error
        import sys
        sys.exit(1)

    # --- Define Test Cases ---
    # This structure seems correct: a dictionary where keys are set names
    # and values are lists of dictionaries, each with 'sentence' and 'label'.
    datasets_json_path = "sentences.json" # Or your specific path
    experiment_sets_data = {}
    try:
        with open(datasets_json_path, 'r', encoding='utf-8') as f:
            experiment_sets_data = json.load(f)
        logger.info(f"Successfully loaded {len(experiment_sets_data)} experiment sets from '{datasets_json_path}'")
    except FileNotFoundError:
        logger.error(f"Dataset JSON file not found: '{datasets_json_path}'. Please create it or check the path.")

    logger.info(f"Loaded {len(experiment_sets_data)} experiment sets: {list(experiment_sets_data.keys())}")

    # --- Configuration ---
    # *** IMPORTANT: Set the correct path to your AraVec model file ***
    # Ensure this path is valid and accessible from where the script runs.
    # Example: embedding_model_path = "path/to/your/aravec/model.bin"
    # embedding_model_path = "aravec/full_uni_cbow_300_twitter.mdl" # Original example path
    embedding_model_path = "aravec/full_uni_cbow_300_twitter.mdl" # !!! USER: Ensure this path is correct !!!
    if not os.path.exists(embedding_model_path):
        logger.warning(f"Embedding model path does not exist: {embedding_model_path}. Kernel will use hash-based fallback.")
        embedding_model_path = None # Ensure it's None if not found

    # --- Output Directory ---
    base_output_dir = "qnlp_experiment_outputs_per_set_v2" # New output dir name for this version
    os.makedirs(base_output_dir, exist_ok=True)
    # --- Experiment Configuration ---
    experiment_sets_to_run = list(experiment_sets_data.keys())

    # Core module integration
    USE_CORE_MODULE_ENHANCEMENTS = True
    MORPHO_LEX_ENHANCED_SETS = ["LexicalAmbiguity", "Morphology"]
    CLASSICAL_FEATURE_DIM_CORE = 16
    MAX_SENSES_CORE = 1

    # -----------------------------------------------------------------------
    # Ansatz configurations to run (ablation study)
    # Each tuple: (ansatz_name, n_layers_iqp, n_layers_strong)
    # IQP(2)  – deeper diagonal circuits, good baseline
    # StronglyEntangling(1) – full SU(2)+CNOT, highest expressivity
    # Spider  – no learnable params, pure grammar structure baseline
    # -----------------------------------------------------------------------
    ANSATZ_CONFIGS = [
        ('IQP',               2, 1),   # primary baseline — diagonal circuits
        ('StronglyEntangling', 1, 1),  # full SU(2)+CNOT, highest expressivity
        ('Sim14',              2, 1),  # hardware-efficient circuit (from Sim2019 paper)
    ]

    max_sentences_override_per_set = None
    run_clustering_viz_per_set = True
    run_classification_per_set = True
    # Run classification on BOTH embedding types so we can compare directly
    # 'quantum' = Pauli expectation values only (captures grammar structure)
    # 'combined' = quantum || AraVec mean (captures both structure + lexical)
    classify_using_embeddings = 'quantum'  # primary; combined results also logged separately
    generate_per_category_pca_plots = True
    run_circuit_visualization = False   # circuit PNGs are slow; set True if needed
    run_heatmap_visualization = True
    debug_circuits_in_pipeline = False

    all_sets_pipeline_results = {}
    all_sets_classification_summaries = {}
    all_sets_kernel_instances = {}

    for ansatz_choice_config, n_layers_iqp, n_layers_strong in ANSATZ_CONFIGS:
        logger.info(f"\n{'='*25} ANSATZ: {ansatz_choice_config} (IQP layers={n_layers_iqp}, Strong layers={n_layers_strong}) {'='*25}")
        for set_name in experiment_sets_to_run:
            sentences_for_current_set = experiment_sets_data.get(set_name, [])
            if not sentences_for_current_set:
                logger.warning(f"Data for set '{set_name}' is empty or not found. Skipping."); continue

            logger.info(f"\n{'='*20} PROCESSING SET: {set_name} (Ansatz: {ansatz_choice_config}) {'='*20}")
            set_specific_output_dir = os.path.join(base_output_dir, sanitize_filename(ansatz_choice_config), sanitize_filename(set_name))
            os.makedirs(set_specific_output_dir, exist_ok=True)

            sentences_to_process_this_run = sentences_for_current_set
            if max_sentences_override_per_set and 0 < max_sentences_override_per_set < len(sentences_for_current_set):
                logger.info(f"Processing a subset of {max_sentences_override_per_set} sentences for set '{set_name}'.")
                sentences_to_process_this_run = sentences_for_current_set[:max_sentences_override_per_set]

            if not sentences_to_process_this_run:
                logger.warning(f"No sentences to process for set '{set_name}' after override. Skipping."); continue

            try:
                kernel_instance, final_results_for_set, classification_info_for_set = \
                    prepare_quantum_nlp_pipeline_v8(
                        sentences_for_current_set=sentences_to_process_this_run,
                        current_set_name=set_name,
                        max_sentences_to_process=len(sentences_to_process_this_run),
                        embedding_model_path=embedding_model_path,
                        ansatz_choice=ansatz_choice_config,
                        n_layers_iqp=n_layers_iqp,
                        n_layers_strong=n_layers_strong,
                        run_clustering_viz=run_clustering_viz_per_set,
                        run_classification=run_classification_per_set,
                        output_dir_set=set_specific_output_dir,
                        embedding_type_for_classification=classify_using_embeddings,
                        debug_circuits=debug_circuits_in_pipeline,
                        generate_per_category_pca_plots=generate_per_category_pca_plots,
                        classical_feature_dim_config_for_core=CLASSICAL_FEATURE_DIM_CORE,
                        max_senses_from_core=MAX_SENSES_CORE,
                        morpho_lex_enhanced_sets=MORPHO_LEX_ENHANCED_SETS if USE_CORE_MODULE_ENHANCEMENTS else []
                    )

                run_key = f"{ansatz_choice_config}/{set_name}"
                if kernel_instance and final_results_for_set is not None:
                    all_sets_pipeline_results[run_key] = final_results_for_set
                    all_sets_kernel_instances[run_key] = kernel_instance
                    # Also run classification on combined embedding for comparison
                    if run_classification_per_set and classify_using_embeddings == 'quantum':
                        combined_clf = kernel_instance.evaluate_classification_accuracy(
                            labels=[sentences_to_process_this_run[i]['label'] for i in range(len(sentences_to_process_this_run))
                                    if i < len(kernel_instance._kernel_original_indices)],
                            embedding_type='combined'
                        )
                        if combined_clf and classification_info_for_set:
                            for k, v in combined_clf.items():
                                classification_info_for_set[f'{k}_combined'] = v
                        elif combined_clf:
                            classification_info_for_set = {f'{k}_combined': v for k, v in combined_clf.items()}
                    if classification_info_for_set:
                        all_sets_classification_summaries[run_key] = classification_info_for_set

                    set_results_filename = f"results_metrics_{sanitize_filename(set_name)}.json"
                    save_results_to_json(
                        data={
                            "set_name": set_name, "ansatz_used": ansatz_choice_config,
                            "n_layers_iqp": n_layers_iqp, "n_layers_strong": n_layers_strong,
                            "num_sentences_processed": len(final_results_for_set),
                            "results_data": final_results_for_set,
                            "classification_summary": classification_info_for_set
                        },
                        output_directory=set_specific_output_dir,
                        filename=set_results_filename
                    )

                    if run_heatmap_visualization and kernel_instance:
                        embeddings_for_heatmap = {}
                        sentences_for_heatmap_viz = {}
                        for res_item in final_results_for_set:
                            orig_idx = res_item.get('original_index')
                            interp = res_item.get('interpretation', {}) or {}
                            emb_vec = interp.get('combined_embedding') or interp.get('quantum_embedding')
                            if orig_idx is not None and emb_vec is not None:
                                embeddings_for_heatmap[orig_idx] = np.array(emb_vec)
                                sentences_for_heatmap_viz[orig_idx] = res_item.get('sentence', f'S_{orig_idx}')
                        if len(embeddings_for_heatmap) > 1:
                            heatmap_path = os.path.join(set_specific_output_dir, f"heatmap_{sanitize_filename(set_name)}_combined.png")
                            generate_similarity_heatmap(embeddings_for_heatmap, sentences_for_heatmap_viz,
                                                        f"Similarity — {set_name} ({ansatz_choice_config})", heatmap_path)

                    if run_circuit_visualization:
                        circuits_output_dir = os.path.join(set_specific_output_dir, "circuit_diagrams")
                        os.makedirs(circuits_output_dir, exist_ok=True)
                        num_circuits_visualized = 0
                        for result_item in final_results_for_set:
                            circuit_obj = result_item.get('circuit_object')
                            orig_idx = result_item.get('original_index', 'unknown_idx')
                            label = result_item.get('label', 'unknown_label')
                            if circuit_obj and QISKIT_AVAILABLE and isinstance(circuit_obj, QuantumCircuit):
                                circuit_filename = f"circuit_{orig_idx}_{sanitize_filename(label)}.png"
                                visualize_circuit_matplotlib(circuit_obj, f"Set_{set_name}_Sent_{orig_idx}",
                                                             os.path.join(circuits_output_dir, circuit_filename))
                                num_circuits_visualized += 1
                        logger.info(f"Saved {num_circuits_visualized} circuit diagrams for set '{set_name}'.")
                else:
                    logger.error(f"Pipeline FAILED for set: {set_name} / ansatz: {ansatz_choice_config}.")
            except Exception as pipeline_error:
                logger.error(f"Unhandled exception for set '{set_name}' / ansatz '{ansatz_choice_config}': {pipeline_error}", exc_info=True)

    circuits_to_analyze_for_metrics = []
    for set_name, results_list in all_sets_pipeline_results.items():
      for item in results_list:
          if item and not item.get('error') and isinstance(item.get('circuit_object'), QuantumCircuit):
              circuits_to_analyze_for_metrics.append({
                  'id': f"{set_name}_{item.get('original_index')}",
                  'label': item.get('label', 'N/A'),
                  'circuit': item.get('circuit_object')
              })
#
    if circuits_to_analyze_for_metrics:
        metrics_output_dir = os.path.join(base_output_dir, "overall_circuit_metrics") # A general dir for this
        os.makedirs(metrics_output_dir, exist_ok=True)
        analyze_circuit_metrics(circuits_to_analyze_for_metrics, metrics_output_dir, "all_sets_comparison")
    # --- Final Consolidated Summary (all ansatzes × all sets) ---
    logger.info("\n" + "="*30 + "\nAll Ansatz Experiments Finished\n" + "="*30)
    ansatz_names = [a[0] for a in ANSATZ_CONFIGS]
    consolidated_summary_dir = base_output_dir
    save_results_to_json(
        data={
            "experiment_consolidated_summary": True,
            "ansatzes_configured": ansatz_names,
            "embedding_model_path_configured": embedding_model_path,
            "sets_processed": experiment_sets_to_run,
            "classification_summary_by_run": all_sets_classification_summaries,
            "num_runs_processed": len(all_sets_pipeline_results),
        },
        output_directory=consolidated_summary_dir,
        filename="run_summary_all_ansatzes.json"
    )
    logger.info(f"\nLog file saved at: {os.path.abspath(log_filename)}")
    logger.info(f"All outputs saved in base directory: {os.path.abspath(base_output_dir)}")
    logger.info("Experiment script (exp4.py) finished.")

