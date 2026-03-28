# -*- coding: utf-8 -*-
import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt # Needed for visualization functions
from typing import List, Dict, Optional, Any # Import necessary types

# --- Import the pipeline function from v8.py ---
try:
    from v8 import prepare_quantum_nlp_pipeline_v8, ArabicQuantumMeaningKernel # Import Kernel too if needed for type hints or direct access
    PIPELINE_AVAILABLE = True
    # Logger might not be configured yet, so initial info is logged later
except ImportError as e:
    # Basic config for critical import error
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    logging.error(f"FATAL: Could not import pipeline from v8.py: {e}", exc_info=True)
    PIPELINE_AVAILABLE = False
    def prepare_quantum_nlp_pipeline_v8(*args, **kwargs): return None, [], None
    class ArabicQuantumMeaningKernel: pass # Dummy class

# --- Qiskit/Sklearn/Arabic Helpers (Need to be available for helpers) ---
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    class QuantumCircuit: pass

try:
    from sklearn.metrics import pairwise
    SKLEARN_AVAILABLE = True
except ImportError:
     SKLEARN_AVAILABLE = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_DISPLAY_ENABLED = True
    def shape_arabic_text(text):
        if not ARABIC_DISPLAY_ENABLED or not text or not isinstance(text, str): return text
        if any('\u0600' <= char <= '\u06FF' for char in text):
            try: return get_display(arabic_reshaper.reshape(text))
            except Exception as e: logging.warning(f"Reshape error: {e}"); return text
        return text
except ImportError:
    ARABIC_DISPLAY_ENABLED = False
    def shape_arabic_text(text): return text

# --- Configure Logging ---

# Define the log filename
log_filename = 'qnlp_experiment.txt'
log_level = logging.DEBUG # Change to logging.DEBUG for more detail

# --- NEW Logging Configuration ---
# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level) # Set the minimum level for the root logger

# Define the format for log messages
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 1. File Handler - Writes logs to a file
# Use 'w' mode to overwrite the file each time the script runs
# Use 'a' mode to append to the file if it exists
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setLevel(log_level) # Set level for this handler
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# 2. Stream Handler - Writes logs to the console (stderr by default)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level) # Set level for this handler
stream_handler.setFormatter(log_formatter)
root_logger.addHandler(stream_handler)

# --- End NEW Logging Configuration ---


# Silence overly verbose libraries if needed
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Create a logger specific to this runner script (optional, inherits root config)
logger = logging.getLogger("exp4_runner")

# Now log the successful import if it happened
if PIPELINE_AVAILABLE:
    logger.info("Successfully imported pipeline function from v8.py")
logger.info(f"Logging configured. Output will be sent to console and '{log_filename}'")


# ============================================
# Helper Function for Heatmap Visualization (Defined in exp4.py)
# ============================================
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
    # Ensure embeddings are valid numpy arrays before adding
    embeddings_list = [embeddings[idx] for idx in processed_indices if isinstance(embeddings.get(idx), np.ndarray)]
    # Get sentence labels using the index from the kernel's reference dict, only for valid embeddings
    labels_list = [sentences.get(idx, f"Sent_{idx}") for idx in processed_indices if isinstance(embeddings.get(idx), np.ndarray)]

    if not embeddings_list: # Check if list is empty after filtering
        logger.warning(f"Skipping heatmap '{title}': No valid numpy array embeddings found.")
        return

    # Filter out non-finite values AFTER aligning
    X_full = np.array(embeddings_list)
    finite_mask = np.all(np.isfinite(X_full), axis=1)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        logger.warning(f"Removing {num_removed} non-finite rows for heatmap '{title}'.")
        X = X_full[finite_mask]
        # Filter labels corresponding to the finite rows
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
        ax.set_title(shape_arabic_text(title)) # Shape the title
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

# ============================================
# Helper Function for Circuit Visualization (Defined in exp4.py)
# ============================================
def visualize_circuit_matplotlib(circuit: QuantumCircuit,
                                 label: str,
                                 save_path: str):
    """
    Draws and saves a Qiskit circuit diagram using matplotlib.
    """
    if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit): logger.warning(f"Skipping circuit viz for '{label}': Invalid circuit."); return
    if not hasattr(circuit, 'draw'): logger.warning(f"Skipping circuit viz for '{label}': No 'draw' method."); return

    logger.debug(f"Visualizing circuit: {label}")
    circuit_fig = None
    try:
        # Draw the circuit using matplotlib backend
        circuit_fig = circuit.draw(output='mpl', fold=-1, scale=0.7)
        if circuit_fig:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the figure
            circuit_fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.debug(f"Saved circuit diagram to {save_path}")
        else: logger.warning(f"Could not generate circuit figure for '{label}'.")
    except ImportError as e_mpl:
        # Catch specific error if matplotlib or pylatexenc is missing
        logger.error(f"Error drawing circuit '{label}': Matplotlib/pylatexenc missing? {e_mpl}")
    except Exception as e_draw:
        # Catch any other drawing or saving errors
        logger.error(f"Error drawing/saving circuit diagram for '{label}': {e_draw}", exc_info=True)
    finally:
        # Ensure the figure is closed to free memory, even if saving failed
        if circuit_fig:
            plt.close(circuit_fig)


# --- Main Experiment Runner ---
if __name__ == "__main__":
    logger.info("="*20)
    logger.info("Starting QNLP Experiment Runner (using v8 Pipeline)")
    logger.info("="*20)

    if not PIPELINE_AVAILABLE:
        logger.critical("Cannot proceed without the pipeline function from v8.py. Exiting.")
        exit() # Stop execution if the core pipeline cannot be imported

    # --- Define Test Cases (Populate with your actual sentences) ---
    # Ensure these match the structure expected by the pipeline (dict with 'sentence' and 'label')
    experiment_sets_data = {
        "WordOrder": [
            {"sentence": "الولدُ يقرأُ الكتابَ .", "label": "WordOrder_SVO"},
            {"sentence": "البنتُ تشربُ الحليبَ .", "label": "WordOrder_SVO"},
            {"sentence": "الطالبُ كتبَ الدرسَ .", "label": "WordOrder_SVO"},
            {"sentence": "المعلمُ شرحَ القاعدةَ .", "label": "WordOrder_SVO"},
            {"sentence": "الطبيبُ فحصَ المريضَ .", "label": "WordOrder_SVO"},
            {"sentence": "المهندسُ بنى البيتَ .", "label": "WordOrder_SVO"},
            {"sentence": "القطةُ أكلت السمكَ .", "label": "WordOrder_SVO"},
            {"sentence": "الطفلُ يلعبُ بالكرةِ .", "label": "WordOrder_SVO"},
            {"sentence": "السيارةُ سريعةٌ .", "label": "WordOrder_Nominal"}, # Adjusted label
            {"sentence": "السماءُ صافيةٌ .", "label": "WordOrder_Nominal"}, # Adjusted label
            {"sentence": "يقرأُ الولدُ الكتابَ .", "label": "WordOrder_VSO"},
            {"sentence": "تشربُ البنتُ الحليبَ .", "label": "WordOrder_VSO"},
            {"sentence": "كتبَ الطالبُ الدرسَ .", "label": "WordOrder_VSO"},
            {"sentence": "شرحَ المعلمُ القاعدةَ .", "label": "WordOrder_VSO"},
            {"sentence": "فحصَ الطبيبُ المريضَ .", "label": "WordOrder_VSO"},
            {"sentence": "بنى المهندسُ البيتَ .", "label": "WordOrder_VSO"},
            {"sentence": "أكلت القطةُ السمكَ .", "label": "WordOrder_VSO"},
            {"sentence": "يلعبُ الطفلُ بالكرةِ .", "label": "WordOrder_VSO"},
            {"sentence": "الفلاحُ يزرعُ الأرضَ .", "label": "WordOrder_SVO"},
            {"sentence": "يزرعُ الفلاحُ الأرضَ .", "label": "WordOrder_VSO"},
            {"sentence": "العاملُ يبني الجدارَ .", "label": "WordOrder_SVO"},
            {"sentence": "يبني العاملُ الجدارَ .", "label": "WordOrder_VSO"},
            {"sentence": "الطائرةُ تطيرُ عالياً .", "label": "WordOrder_SVO"},
            {"sentence": "تطيرُ الطائرةُ عالياً .", "label": "WordOrder_VSO"},
            {"sentence": "الشمسُ مشرقةٌ .", "label": "WordOrder_Nominal"}, # Adjusted label
            {"sentence": "القمرُ منيرٌ .", "label": "WordOrder_Nominal"}  # Adjusted label
        ],
        "LexicalAmbiguity": [
            {"sentence": "جاء الرجلُ الطويلُ .", "label": "Ambiguity_Man"},
            {"sentence": "الرجلُ القويُ يعملُ .", "label": "Ambiguity_Man"},
            {"sentence": "تحدثَ الرجلُ الحكيمُ .", "label": "Ambiguity_Man"},
            {"sentence": "هذا الرجلُ طبيبٌ .", "label": "Ambiguity_Man"},
            {"sentence": "رأيتُ رجلاً في السوقِ .", "label": "Ambiguity_Man"},
            {"sentence": "الرجلُ العجوزُ جلسَ .", "label": "Ambiguity_Man"},
            {"sentence": "سألَ الرجلُ سؤالاً .", "label": "Ambiguity_Man"},
            {"sentence": "أعطى الرجلُ المالَ .", "label": "Ambiguity_Man"},
            {"sentence": "ابتسمَ الرجلُ السعيدُ .", "label": "Ambiguity_Man"},
            {"sentence": "يسافرُ الرجلُ غداً .", "label": "Ambiguity_Man"},
            {"sentence": "انكسرتْ رجلُ الكرسيِّ .", "label": "Ambiguity_Leg"},
            {"sentence": "للطاولةِ أربعُ أرجلٍ .", "label": "Ambiguity_Leg"},
            {"sentence": "عالجَ الطبيبُ رجلَ المريضِ .", "label": "Ambiguity_Leg"},
            {"sentence": "وضعَ الكتابَ على رجلِهِ .", "label": "Ambiguity_Leg"},
            {"sentence": "رجلُ السريرِ مكسورةٌ .", "label": "Ambiguity_Leg"},
            {"sentence": "أصيبتْ رجلُ اللاعبِ .", "label": "Ambiguity_Leg"},
            {"sentence": "لا تلمسْ رجلَ المكتبِ .", "label": "Ambiguity_Leg"},
            {"sentence": "تحتاجُ الطاولةُ إلى رجلٍ جديدةٍ .", "label": "Ambiguity_Leg"},
            {"sentence": "ألمُ الرجلِ شديدٌ .", "label": "Ambiguity_Leg"},
            {"sentence": "طولُ رجلِ الزرافةِ كبيرٌ .", "label": "Ambiguity_Leg"},
            {"sentence": "هذا رجلٌ كريمٌ .", "label": "Ambiguity_Man"},
            {"sentence": "رجلُ الطاولةِ تحتاجُ إلى إصلاحٍ .", "label": "Ambiguity_Leg"},
            {"sentence": "صافحتُ الرجلَ الغريبَ .", "label": "Ambiguity_Man"},
            {"sentence": "رجلُ الكنبةِ متينةٌ .", "label": "Ambiguity_Leg"},
            {"sentence": "شربتُ من عينِ الماءِ .", "label": "Ambiguity_Spring"},
            {"sentence": "عينُ الطفلِ زرقاءُ .", "label": "Ambiguity_Eye"},
            {"sentence": "هذه عينٌ صافيةٌ .", "label": "Ambiguity_Eye"},
            {"sentence": "في الصحراءِ عينُ ماءٍ نادرةٌ .", "label": "Ambiguity_Spring"},
            {"sentence": "رأيتُ بعيني اليمنى.", "label": "Ambiguity_Eye"},
            {"sentence": "تدفقت عينُ الماءِ بغزارةٍ.", "label": "Ambiguity_Spring"},
            {"sentence": "أُصيبَ اللاعبُ في عينهِ.", "label": "Ambiguity_Eye"},
            {"sentence": "وجدوا عينَ ماءٍ عذبةٍ في الجبلِ.", "label": "Ambiguity_Spring"},
            {"sentence": "عينُ الحسودِ فيها عودٌ.", "label": "Ambiguity_Eye"}
        ],
        "Morphology": [
            {"sentence": "المهندسُ يعملُ .", "label": "Morph_SgMasc"},
            {"sentence": "المهندسونَ يعملونَ .", "label": "Morph_PlMasc"},
            {"sentence": "الطالبةُ تدرسُ .", "label": "Morph_SgFem"},
            {"sentence": "الطالباتُ يدرسنَ .", "label": "Morph_PlFem"},
            {"sentence": "الكتابُ جديدٌ .", "label": "Morph_SgMasc"},
            {"sentence": "الكتبُ جديدةٌ .", "label": "Morph_PlBroken"},
            {"sentence": "السيارةُ سريعةٌ .", "label": "Morph_SgFem"},
            {"sentence": "السياراتُ سريعةٌ .", "label": "Morph_PlFem"},
            {"sentence": "البيتُ كبيرٌ .", "label": "Morph_SgMasc"},
            {"sentence": "الغرفةُ كبيرةٌ .", "label": "Morph_SgFem"},
            {"sentence": "الولدُ ذكيٌّ .", "label": "Morph_SgMasc"},
            {"sentence": "البنتُ ذكيةٌ .", "label": "Morph_SgFem"},
            {"sentence": "كتبَ الطالبُ .", "label": "Morph_Past"},
            {"sentence": "يكتبُ الطالبُ .", "label": "Morph_Pres"},
            {"sentence": "قرأت البنتُ .", "label": "Morph_Past"},
            {"sentence": "تقرأُ البنتُ .", "label": "Morph_Pres"},
            {"sentence": "هذا كتابي .", "label": "Morph_Poss1Sg"},
            {"sentence": "هذا كتابكَ .", "label": "Morph_Poss2MascSg"},
            {"sentence": "هذا قلمهُ .", "label": "Morph_Poss3MascSg"},
            {"sentence": "هذا قلمها .", "label": "Morph_Poss3FemSg"},
            {"sentence": "المعلمةُ تشرحُ .", "label": "Morph_SgFem"},
            {"sentence": "المعلماتُ يشرحنَ .", "label": "Morph_PlFem"},
            {"sentence": "اللاعبانِ ماهرانِ .", "label": "Morph_DualMasc"},
            {"sentence": "اللاعبتانِ ماهرتانِ .", "label": "Morph_DualFem"},
            {"sentence": "سافرَ الرجلُ .", "label": "Morph_Past"},
            {"sentence": "يسافرُ الرجلُ .", "label": "Morph_Pres"},
            {"sentence": "سافرت المرأةُ .", "label": "Morph_Past"},
            {"sentence": "تسافرُ المرأةُ .", "label": "Morph_Pres"},
            {"sentence": "هذه مدرستنا .", "label": "Morph_Poss1Pl"},
            {"sentence": "هذا بيتكم .", "label": "Morph_Poss2Pl"},
            {"sentence": "هؤلاء طلابٌ مجتهدونَ .", "label": "Morph_AdjPlMasc"},
            {"sentence": "هؤلاء طالباتٌ مجتهداتٌ .", "label": "Morph_AdjPlFem"}
        ]
    }

    logger.info(f"Loaded experiment sets: {list(experiment_sets_data.keys())}")

    # --- Configuration ---
    # *** IMPORTANT: Set the correct path to your AraVec model file ***
    # Example: If 'aravec' is a folder in the same directory as exp4.py
    # and the model file is named 'tweets_cbow_300.bin' (assuming binary)
    # embedding_model_path_for_kernel = "aravec/tweets_cbow_300.bin"
    # Example: If it's a text file named 'tweets_cbow_300.vec'
    # embedding_model_path_for_kernel = "aravec/tweets_cbow_300.vec"
    # Example: Using an absolute path
    # embedding_model_path_for_kernel = "/path/to/your/models/aravec/tweets_cbow_300.bin"
    # *** Set your actual path here: ***
    embedding_model_path_for_kernel = "../aravec/full_uni_cbow_300_twitter.mdl" # Keep original if no extension
    logger.info(f"Attempting to load embedding model from: {embedding_model_path_for_kernel}")

    classify_using = 'combined' # Choose 'quantum' or 'combined' for classification features
    ansatz = 'IQP'             # Choose 'IQP' or 'STRONGLY_ENTANGLING' or 'SPIDER'
    iqp_layers = 2
    iqp_params_per_qubit = 3
    strong_layers = 2
    strong_ranges = None
    # Create a unique output directory name based on config
    output_dir_name = f"qnlp_pipeline_v8_output_{ansatz}_{classify_using}"
    output_dir = os.path.abspath(output_dir_name) # Use absolute path for clarity
    os.makedirs(output_dir, exist_ok=True) # Ensure base output dir exists

    # --- Flags ---
    RUN_PCA_VIZ = True
    RUN_HEATMAP_VIZ = True
    RUN_CIRCUIT_VIZ = True # Set True to draw circuits (can be many files)
    RUN_STATE_VIZ = False
    RUN_CLASSIFICATION = True

    logger.info("--- Pipeline Configuration ---")
    logger.info(f"  Embedding Model Path: {embedding_model_path_for_kernel}")
    logger.info(f"  Classification Features: {classify_using}")
    logger.info(f"  Ansatz: {ansatz}")
    if ansatz == 'IQP':
        logger.info(f"    IQP Layers: {iqp_layers}")
        logger.info(f"    IQP Params/Qubit: {iqp_params_per_qubit}")
    elif ansatz == 'STRONGLY_ENTANGLING':
        logger.info(f"    Strongly Entangling Layers: {strong_layers}")
        logger.info(f"    Strongly Entangling Ranges: {strong_ranges if strong_ranges else 'Default'}")
    logger.info(f"  Run PCA Viz: {RUN_PCA_VIZ}")
    logger.info(f"  Run Heatmap Viz: {RUN_HEATMAP_VIZ}")
    logger.info(f"  Run Circuit Viz: {RUN_CIRCUIT_VIZ}")
    logger.info(f"  Run State Viz: {RUN_STATE_VIZ}")
    logger.info(f"  Run Classification: {RUN_CLASSIFICATION}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info("-----------------------------")

    # --- Call the Pipeline Function ---
    logger.info("Starting the combined QNLP pipeline (v8)...")
    kernel_instance, final_results, classification_info = prepare_quantum_nlp_pipeline_v8(
        experiment_sets=experiment_sets_data,
        max_sentences_per_set=50, # Adjust as needed
        embedding_model_path=embedding_model_path_for_kernel,
        ansatz_choice=ansatz,
        n_layers_iqp=iqp_layers,
        n_single_qubit_params_iqp=iqp_params_per_qubit, # Pass IQP param count
        n_layers_strong=strong_layers,
        cnot_ranges=strong_ranges,
        run_clustering_viz=RUN_PCA_VIZ, # Control internal PCA viz
        run_state_viz=RUN_STATE_VIZ,    # Control internal state viz
        run_classification=RUN_CLASSIFICATION,
        classification_embedding_type=classify_using,
        run_heatmap_viz=False, # Heatmap generation is handled locally below
        output_dir_base=output_dir # Pass the specific output dir
    )

    # --- Post-Pipeline Analysis & Visualization (Called from exp4.py) ---
    if kernel_instance and final_results:
        logger.info("\n--- Pipeline Execution Summary ---")
        logger.info(f"Pipeline completed. Processed {len(final_results)} sentences.")

        # --- Generate Heatmap (using helper function defined above) ---
        if RUN_HEATMAP_VIZ:
             logger.info("--- Generating Overall Similarity Heatmap ---")
             # Decide which embeddings to use for the heatmap
             heatmap_embedding_type = 'quantum' # Or 'combined'
             embeddings_for_heatmap = kernel_instance.circuit_embeddings if heatmap_embedding_type == 'quantum' else kernel_instance.sentence_embeddings
             heatmap_title_suffix = shape_arabic_text('(تضمينات كمومية)') if heatmap_embedding_type == 'quantum' else shape_arabic_text('(تضمينات مدمجة)')

             if embeddings_for_heatmap and kernel_instance.reference_sentences:
                 heatmap_path = os.path.join(output_dir, f"overall_heatmap_{heatmap_embedding_type}_{ansatz}.png")
                 generate_similarity_heatmap(
                     embeddings=embeddings_for_heatmap,
                     sentences=kernel_instance.reference_sentences,
                     title=f"{shape_arabic_text('تشابه التضمينات الكلي')} {heatmap_title_suffix} ({ansatz})",
                     save_path=heatmap_path
                 )
             else:
                 logger.warning(f"Skipping heatmap generation for '{heatmap_embedding_type}': Missing embeddings or reference sentences in kernel.")

        # --- Circuit Visualization (using helper function defined above) ---
        if RUN_CIRCUIT_VIZ:
             logger.info("--- Generating Circuit Diagrams ---")
             circuits_drawn = 0
             # Retrieve circuits from the final_results list
             for result in final_results:
                  circuit_obj = result.get('circuit') # Assuming circuit object is stored here by the pipeline
                  sent_idx = result.get('original_index', 'unknown')
                  label = result.get('label', 'unknown')
                  # Try to determine the set name from the label for subdirectories
                  set_name = label.split('_')[0] if '_' in label else 'General'

                  if circuit_obj and isinstance(circuit_obj, QuantumCircuit):
                      # Create subdirectory for the set if it doesn't exist
                      set_output_dir = os.path.join(output_dir, "circuits", set_name)
                      os.makedirs(set_output_dir, exist_ok=True)
                      # Create a safe filename from the label
                      safe_label = "".join(c if c.isalnum() else "_" for c in label)
                      circuit_filename = f"circuit_{sent_idx}_{safe_label}.png"
                      circuit_path = os.path.join(set_output_dir, circuit_filename)
                      # Call the visualization helper
                      visualize_circuit_matplotlib(circuit_obj, f"Sent_{sent_idx}", circuit_path)
                      circuits_drawn += 1
             logger.info(f"Attempted to draw {circuits_drawn} circuit diagrams.")


        # --- Final Summary Logging ---
        # The classification_info dictionary might contain results for multiple classifiers
        if classification_info:
            logger.info(f"\n--- Classification Results (using '{classify_using}' Embeddings) ---")
            for classifier_name, summary in classification_info.items():
                 logger.info(f"  Classifier: {classifier_name}")
                 if 'error' in summary:
                      logger.error(f"    Error: {summary['error']}")
                 else:
                      accuracy = summary.get('accuracy', 'N/A')
                      accuracy_formatted = f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy
                      logger.info(f"    Accuracy: {accuracy_formatted}")
                      # Log the full report here if desired, or mention it's in the HTML report
                      # logger.info(f"    Report:\n{summary.get('report', 'N/A')}")
                      logger.info(f"    (See HTML report or log file for full classification details)")
        elif RUN_CLASSIFICATION:
            logger.warning("\nClassification task was enabled but did not produce results or failed.")

        logger.info(f"\nOutputs and report saved in directory: {output_dir}")
        logger.info(f"Log file saved as: {os.path.join(os.getcwd(), log_filename)}") # Show log file location
    else:
        logger.error("Pipeline execution failed or produced no results.")

    logger.info("\nExperiment script finished.")
    logger.info("="*30)