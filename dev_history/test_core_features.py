import json
import logging
import os
import numpy as np

# Attempt to import from arabic_morpho_lex_core for its components if needed directly
# However, v8.py will be the primary interface to it in this modified script.
try:
    from arabic_morpho_lex_core import N, S, ROOT_TYPE, IQPAnsatz, SpiderAnsatz, StronglyEntanglingAnsatz
    from lambeq.backend.grammar import Ty
    CORE_MODULE_COMPONENTS_LOADED = True
except ImportError:
    CORE_MODULE_COMPONENTS_LOADED = False
    # Define dummies if direct import fails, v8 import might still work or fail later
    class Ty: pass
    class IQPAnsatz: pass
    class SpiderAnsatz: pass
    class StronglyEntanglingAnsatz: pass
    N, S, ROOT_TYPE = None, None, None


# Attempt to import the main pipeline function from v8.py
try:
    from v8 import prepare_quantum_nlp_pipeline_v8, ArabicQuantumMeaningKernel
    # We also need QISKIT_AVAILABLE from v8 to know if circuit objects can be expected
    from v8 import QISKIT_AVAILABLE as V8_QISKIT_AVAILABLE
    if V8_QISKIT_AVAILABLE is None: # If v8 didn't define it (older version)
        try:
            from qiskit import QuantumCircuit
            V8_QISKIT_AVAILABLE = True
            class QuantumCircuit_test: pass # Local dummy to avoid conflict if v8.QuantumCircuit is different
        except ImportError:
            V8_QISKIT_AVAILABLE = False
            class QuantumCircuit_test: pass


    V8_PIPELINE_LOADED = True
except ImportError as e_v8:
    V8_PIPELINE_LOADED = False
    V8_QISKIT_AVAILABLE = False
    logging.basicConfig(level=logging.ERROR) # BasicConfig if logger not yet set up
    logging.error(f"FATAL: Could not import from v8.py: {e_v8}", exc_info=True)
    def prepare_quantum_nlp_pipeline_v8(*args, **kwargs):
        logging.error("v8.prepare_quantum_nlp_pipeline_v8 not loaded.")
        return None, [], None
    class ArabicQuantumMeaningKernel: pass


# --- Configure Logging for this Test Script ---
log_filename_test = 'test_v8_integration_features.txt' # New log file name
log_level_test = logging.DEBUG
root_logger_test = logging.getLogger()
if root_logger_test.hasHandlers():
    for handler in root_logger_test.handlers[:]:
        root_logger_test.removeHandler(handler)
        handler.close()
logging.basicConfig(
    level=log_level_test,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename_test, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger_core_tester = logging.getLogger("V8_IntegrationTester")

if V8_PIPELINE_LOADED: logger_core_tester.info("v8.py pipeline components loaded.")
else: logger_core_tester.error("Failed to load v8.py pipeline components.")
if CORE_MODULE_COMPONENTS_LOADED: logger_core_tester.info("Direct components from arabic_morpho_lex_core.py loaded (for type checking/ansatz).")
else: logger_core_tester.warning("Direct components from arabic_morpho_lex_core.py failed to load (v8 might still work if its internal imports are fine).")


# --- Test Configuration ---
SENTENCES_JSON_PATH = "sentences.json"
# *** IMPORTANT: CHOOSE A SMALL, FOCUSED TEST SET FOR DEBUGGING ***
# TEST_SET_CATEGORY = "Morphology" # This has 192 sentences, maybe too many for initial debug
TEST_SET_CATEGORY = "LexicalAmbiguity"  # Or a custom small set you create in sentences.json
# Example: Create a "DebugStreamIssues" category in sentences.json with 2-3 problematic sentences

# Configs that will be passed to prepare_quantum_nlp_pipeline_v8
# These should align with how you run exp4.py for the problematic cases
PIPELINE_ANSATZ_CHOICE = 'IQP' # This is the ansatz for camel_test2 via v8.py
PIPELINE_N_LAYERS_IQP = 1
PIPELINE_N_LAYERS_STRONG = 1
PIPELINE_DEBUG_CIRCUITS = True # Enables debug logs from camel_test2 and core_module via v8
PIPELINE_OUTPUT_DIR_SET = "v8_integration_test_outputs"
os.makedirs(PIPELINE_OUTPUT_DIR_SET, exist_ok=True)

# Configs for the core module part WITHIN v8.py's call
# v8.py uses these when it calls process_sentence_for_qnlp_core
CORE_CLASSICAL_FEATURE_DIM = 16
CORE_MAX_SENSES = 1
# This list in v8.py determines if the core module is used for a given set.
# Make sure TEST_SET_CATEGORY is in this list if you want v8 to attempt core processing.
V8_MORPHO_LEX_ENHANCED_SETS = ["Morphology", "LexicalAmbiguity", "DebugStreamIssues"]


# Global flag for detailed CAMeL dumps in the lookup function
debug_print_full_camel_data_test = False


def load_test_sentences(json_path, category_key) -> list[dict[str, str]]:
    # (Your existing load_test_sentences function - unchanged)
    if not os.path.exists(json_path):
        logger_core_tester.error(f"Sentences JSON file not found: '{json_path}'. Please create it.")
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f: all_datasets = json.load(f)
        if category_key in all_datasets:
            logger_core_tester.info(f"Loaded {len(all_datasets[category_key])} sentences from category '{category_key}'.")
            return all_datasets[category_key]
        else:
            logger_core_tester.error(f"Category '{category_key}' not found in '{json_path}'. Available categories: {list(all_datasets.keys())}")
            return []
    except Exception as e:
        logger_core_tester.error(f"Error loading sentences from '{json_path}': {e}", exc_info=True)
        return []

def print_v8_results_summary(sentence_data_from_v8_pipeline: dict[str, any]):
    """Prints a summary of a single sentence result from v8.prepare_quantum_nlp_pipeline."""
    original_idx = sentence_data_from_v8_pipeline.get('original_index', 'N/A')
    sentence_text = sentence_data_from_v8_pipeline.get('sentence', 'N/A')
    logger_core_tester.info(f"\n--- V8 Pipeline Summary for Sentence Original Idx: {original_idx} --- '{sentence_text}' ---")

    if sentence_data_from_v8_pipeline.get('error'):
        logger_core_tester.error(f"  Pipeline Error for this sentence: {sentence_data_from_v8_pipeline['error']}")
        return

    # Main processor (camel_test2) outputs
    logger_core_tester.info(f"  Structure (from main_sentence_processor/camel_test2): {sentence_data_from_v8_pipeline.get('structure_from_main_processor', 'N/A')}")
    tokens_main = sentence_data_from_v8_pipeline.get('tokens_from_main_processor', [])
    logger_core_tester.info(f"  Tokens (from main_sentence_processor): {tokens_main}")

    # Core module linguistic stream (this is what we want to inspect)
    linguistic_stream_core = sentence_data_from_v8_pipeline.get('linguistic_stream_of_words_core')
    if linguistic_stream_core is not None: # It could be an empty list if core processing succeeded but found no words, or None if it failed to be added
        logger_core_tester.info(f"  Linguistic Word Stream from Core Module (count: {len(linguistic_stream_core)}):")
        if not linguistic_stream_core:
            logger_core_tester.warning("    Core linguistic stream is EMPTY.")
        for k, word_data in enumerate(linguistic_stream_core[:3]): # Print first few
            surface_text_stream = word_data.get('surface_text', 'N/A')
            root_stream = word_data.get('extracted_root', 'N/A')
            camel_analysis_surf = word_data.get('camel_analysis_of_surface_word', {})
            pos_surf = camel_analysis_surf.get('pos', 'N/A')
            classical_feats_surf = word_data.get('classical_features_of_surface_word')
            classical_feats_preview_str = f"Shape: {classical_feats_surf.shape}, First 3: {np.round(classical_feats_surf[:3], 3)}" if isinstance(classical_feats_surf, np.ndarray) else "N/A"
            logger_core_tester.info(f"    Word {k+1}: '{surface_text_stream}' (Root: '{root_stream}', CAMeL POS: {pos_surf})")
            logger_core_tester.info(f"      Classical Features: {classical_feats_preview_str}")
            if k == 0 and debug_print_full_camel_data_test:
                 logger_core_tester.debug(f"      Full CAMeL for Word {k+1} (Surface '{surface_text_stream}'): {json.dumps(camel_analysis_surf, ensure_ascii=False, indent=2)}")

    else:
        logger_core_tester.warning("  Linguistic Word Stream from Core Module: NOT FOUND in v8 results for this sentence.")

    # Kernel Embeddings
    interpretation = sentence_data_from_v8_pipeline.get('interpretation', {})
    if interpretation:
        q_emb = interpretation.get('quantum_embedding')
        c_emb = interpretation.get('combined_embedding')
        l_emb = interpretation.get('linguistic_features_vector')
        logger_core_tester.info(f"  Kernel Quantum Embedding: {'Present' if q_emb is not None else 'MISSING'} (Shape: {q_emb.shape if q_emb is not None else 'N/A'})")
        logger_core_tester.info(f"  Kernel Combined Embedding: {'Present' if c_emb is not None else 'MISSING'} (Shape: {c_emb.shape if c_emb is not None else 'N/A'})")
        logger_core_tester.info(f"  Kernel Linguistic Features: {'Present' if l_emb is not None else 'MISSING'} (Shape: {l_emb.shape if l_emb is not None else 'N/A'})")
        if q_emb is not None: logger_core_tester.debug(f"    Quantum Emb (first 5): {np.round(q_emb[:5],3)}")
    else:
        logger_core_tester.warning("  No 'interpretation' data from kernel in v8 results.")
    logger_core_tester.info("-" * 40)


def main_test_block_rt():
    global debug_print_full_camel_data_test
    if not V8_PIPELINE_LOADED:
        logger_core_tester.critical("v8.py pipeline function not loaded. Cannot run integrated tests.")
        return

    logger_core_tester.info("===== Starting Integrated Test Block (test_core_features.py calling v8.py pipeline) =====")
    debug_print_full_camel_data_test = False # Set to True for verbose CAMeL data

    test_sentences_data = load_test_sentences(SENTENCES_JSON_PATH, TEST_SET_CATEGORY)
    if not test_sentences_data:
        logger_core_tester.error(f"No sentences loaded for category '{TEST_SET_CATEGORY}'. Exiting test.")
        return

    # Make a unique output directory for this specific test run to avoid conflicts
    current_test_run_output_dir = os.path.join(PIPELINE_OUTPUT_DIR_SET, f"test_run_{TEST_SET_CATEGORY}")
    os.makedirs(current_test_run_output_dir, exist_ok=True)
    logger_core_tester.info(f"Test outputs will be in: {current_test_run_output_dir}")

    # Call v8.py's pipeline function
    # It processes a list of sentence data dictionaries
    kernel_instance, final_results_for_set, classification_info = \
        prepare_quantum_nlp_pipeline_v8(
            sentences_for_current_set=test_sentences_data, # Pass the list of dicts
            current_set_name=TEST_SET_CATEGORY,
            max_sentences_to_process=len(test_sentences_data), # Process all loaded test sentences
            embedding_model_path=None, # Set path if you want to test embedding-based param binding
            ansatz_choice=PIPELINE_ANSATZ_CHOICE,
            n_layers_iqp=PIPELINE_N_LAYERS_IQP,
            n_layers_strong=PIPELINE_N_LAYERS_STRONG,
            run_clustering_viz=False, # Disable heavy viz for this focused test
            run_state_viz=False,
            run_classification=False, # Disable classification for this focused test
            output_dir_set=current_test_run_output_dir,
            debug_circuits=PIPELINE_DEBUG_CIRCUITS,
            generate_per_category_pca_plots=False,
            classical_feature_dim_config_for_core=CORE_CLASSICAL_FEATURE_DIM,
            max_senses_from_core=CORE_MAX_SENSES,
            morpho_lex_enhanced_sets=V8_MORPHO_LEX_ENHANCED_SETS
        )

    if kernel_instance:
        logger_core_tester.info(f"Kernel instance created by v8 pipeline. Number of circuit embeddings in kernel: {len(kernel_instance.circuit_embeddings)}")
    else:
        logger_core_tester.warning("v8 pipeline did not return a kernel instance.")

    if final_results_for_set:
        logger_core_tester.info(f"v8 pipeline returned {len(final_results_for_set)} result items for the set.")
        for sentence_result_from_v8 in final_results_for_set:
            print_v8_results_summary(sentence_result_from_v8) # New summary function
            # You can add more detailed checks here, e.g., assert that
            # 'linguistic_stream_of_words_core' exists if TEST_SET_CATEGORY is in V8_MORPHO_LEX_ENHANCED_SETS
            if TEST_SET_CATEGORY in V8_MORPHO_LEX_ENHANCED_SETS:
                if 'linguistic_stream_of_words_core' not in sentence_result_from_v8:
                    logger_core_tester.error(f"  MISSING 'linguistic_stream_of_words_core' for sentence original_idx {sentence_result_from_v8.get('original_index')} in an enhanced set!")
                elif not sentence_result_from_v8['linguistic_stream_of_words_core']: # Check if it's empty
                     logger_core_tester.warning(f"  EMPTY 'linguistic_stream_of_words_core' for sentence original_idx {sentence_result_from_v8.get('original_index')}.")


    else:
        logger_core_tester.error("v8 pipeline returned no final_results_for_set.")

    logger_core_tester.info("\n" + "="*20 + " Integrated Test Block Finished " + "="*20)
    logger_core_tester.info(f"Test log saved to: {os.path.abspath(log_filename_test)}")

if __name__ == "__main__":
    if not V8_PIPELINE_LOADED:
        print("CRITICAL: v8.py could not be loaded. Integrated tests cannot run.")
    else:
        logger_core_tester = logging.getLogger("V8_IntegrationTester")
        main_test_block_rt()

