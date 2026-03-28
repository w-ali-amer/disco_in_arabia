import logging
import sys
import numpy as np
import types

# --- Setup sys.path if your modules are not directly importable ---
# This allows importing from your project structure. Adjust as needed.
# Example:
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# --- Configure Logging (Minimal for test script) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UnifiedBindingTest")

# --- Imports (MUST match your project structure) ---
try:
    # Core types and custom classes
    from common_qnlp_types import (
        N_ARABIC, S_ARABIC, Ty, Word, Box, AtomicType, GrammarDiagram,
        AmbiguousLexicalBox, ControlledSenseFunctor, PatchedLambeqTketCircuit
    )
    # Analysis and type assignment
    from camel_test2 import analyze_arabic_sentence_with_morph, assign_discocat_types_v2_2
    
    # --- MODIFICATION START ---
    # Import the v8 module first, then the class from it
    import v8
    from v8 import ArabicQuantumMeaningKernel
    
    # Qiskit for type checking and to confirm it's available in this scope
    from qiskit.circuit import QuantumCircuit, Parameter

    # --- FIX: Force QISKIT_AVAILABLE to be True in the v8 module's namespace ---
    # This circumvents environment issues where the test script can find qiskit 
    # but the v8 module fails on its own initial import.
    v8.QISKIT_AVAILABLE = True
    logger.info("Monkey-patched v8.QISKIT_AVAILABLE to True for testing.")
    # --- MODIFICATION END ---

except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}. Check sys.path and module integrity.", exc_info=True)
    sys.exit(1)

# --- Mock Components for Standalone Testing ---

class MockEmbeddingModel:
    """A mock Word2Vec model for deterministic testing."""
    def __init__(self, vector_size=5):
        self.vector_size = vector_size
        self.vocab = {
            "وَلَد": np.linspace(0.1, 0.5, vector_size),
            "رَجُل": np.linspace(0.6, 1.0, vector_size),
            "رجل_ambiguous_man": np.linspace(0.8, 1.2, vector_size), # Sense-specific
            "رجل_ambiguous_leg": np.linspace(-0.4, 0.0, vector_size), # Sense-specific
        }
        logger.info("Initialized MockEmbeddingModel.")

    def has_index_for(self, key: str) -> bool:
        return key in self.vocab

    def get_vector(self, key: str) -> np.ndarray:
        # Match based on the start of the key for flexibility
        for vocab_key, vector in self.vocab.items():
            if key.startswith(vocab_key):
                return vector
        return np.zeros(self.vector_size)

class TestKernel(ArabicQuantumMeaningKernel):
    """A minimal kernel for testing the binding function."""
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        # Use a real get_morph_vector if available, otherwise mock it
        if not hasattr(self, 'get_morph_vector'):
            self.get_morph_vector = lambda x: np.zeros(16)
        self.params_per_word_fallback = 3
        self.ansatz_choice = "IQP" # Needed for _bind_parameters
        logger.info("Initialized TestKernel for binding test.")

    # In a real scenario, these methods would be inherited from ArabicQuantumMeaningKernel
    _bind_parameters = ArabicQuantumMeaningKernel._bind_parameters
    _build_token_mapping = ArabicQuantumMeaningKernel._build_token_mapping
    _categorize_parameters = ArabicQuantumMeaningKernel._categorize_parameters
    _parse_parameter_name = ArabicQuantumMeaningKernel._parse_parameter_name
    _find_token_info_for_prefix = ArabicQuantumMeaningKernel._find_token_info_for_prefix
    _bind_feature_parameters = ArabicQuantumMeaningKernel._bind_feature_parameters
    _bind_variational_parameters = ArabicQuantumMeaningKernel._bind_variational_parameters
    _bind_unknown_parameters = ArabicQuantumMeaningKernel._bind_unknown_parameters
    _get_feature_vector = ArabicQuantumMeaningKernel._get_feature_vector
    _hash_to_angle = ArabicQuantumMeaningKernel._hash_to_angle
    _find_analysis_by_idx = ArabicQuantumMeaningKernel._find_analysis_by_idx
    _find_core_data_by_idx = ArabicQuantumMeaningKernel._find_core_data_by_idx
    _validate_parameter_binding = ArabicQuantumMeaningKernel._validate_parameter_binding


def debug_quantum_result(quantum_result, logger):
    """Debug helper to understand why conversion fails."""
    if quantum_result is None:
        logger.error("=== DEBUG ABORTED: quantum_result is None ===")
        return
        
    logger.info(f"=== DEBUGGING QUANTUM RESULT ===")
    logger.info(f"Type: {type(quantum_result)}")
    
    # Check conversion capabilities
    conversion_methods = ['to_tk', 'to_qiskit']
    for method in conversion_methods:
        if hasattr(quantum_result, method):
            logger.info(f"Has method: {method}")
            try:
                result = getattr(quantum_result, method)()
                logger.info(f"  {method}() -> {type(result)}")
            except Exception as e:
                logger.error(f"  {method}() FAILED: {e}", exc_info=True)
        else:
            logger.warning(f"Missing method: {method}")
    logger.info(f"=== END DEBUG ===")


def run_binding_test():
    logger.info("--- Starting Unified Parameter Binding Test ---")

    # --- Setup ---
    mock_model = MockEmbeddingModel()
    test_kernel = TestKernel(embedding_model=mock_model)
    sentence = "جاء رجل ولد"

    # --- 1. Full Analysis and Diagram Creation ---
    logger.info(f"\n1. Analyzing sentence: '{sentence}'")
    tokens, analyses, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug=False)
    
    diagram_elements = []
    ambiguous_map_for_sentence = {} # To collect ambiguity info
    for ana in analyses:
        entity = assign_discocat_types_v2_2(ana, roles, debug=False, handle_lexical_ambiguity=True)
        
        if isinstance(entity, AmbiguousLexicalBox):
             ambiguous_map_for_sentence[ana['original_idx']] = {'senses': entity.data.get('senses', [])}

        if isinstance(entity, (Ty, AtomicType)):
            new_word = Word(ana.get('lemma', ana['text']), entity)
            new_word.data = {'original_stanza_idx': ana['original_idx']}
            diagram_elements.append(new_word)
            logger.debug(f"Created Word '{new_word.name}' for token '{ana['text']}'")
        elif isinstance(entity, (Box, Word)):
            if not hasattr(entity, 'data') or entity.data is None: entity.data = {}
            if isinstance(entity.data, dict): entity.data['original_stanza_idx'] = ana['original_idx']
            diagram_elements.append(entity)
            logger.debug(f"Added Box/Word '{entity.name}' for token '{ana['text']}'")
        else:
            logger.warning(f"Skipping entity of unexpected type {type(entity)} for token '{ana['text']}'")

    if not diagram_elements:
        logger.error("Failed to generate diagrammatic elements. Aborting.")
        return

    # This is the original grammar diagram, which is needed for parameter binding
    sentence_diagram = diagram_elements[0]
    for el in diagram_elements[1:]:
        sentence_diagram = sentence_diagram @ el
    
    logger.info(f"Created composite sentence diagram: {sentence_diagram}")

    # --- 2. Create Circuit using ControlledSenseFunctor ---
    logger.info("\n2. Creating circuit with ControlledSenseFunctor")
    ob_map = {N_ARABIC: 1, S_ARABIC: 1}
    functor = ControlledSenseFunctor(ob_map, n_layers=1, n_single_qubit_params=2)

    # --- Test Scenario A: Fixed Ancilla ---
    logger.info("\n--- Scenario A: Testing with Fixed Ancilla ---")
    quantum_diagram_fixed = functor(sentence_diagram, use_variational_ancilla=False)
    
    circuit_fixed = None
    if hasattr(quantum_diagram_fixed, 'to_qiskit'):
        circuit_fixed = quantum_diagram_fixed.to_qiskit()
    else:
        logger.error("Result of functor application cannot be converted to Qiskit circuit.")
        debug_quantum_result(quantum_diagram_fixed, logger)
        return
        
    logger.info(f"Circuit (Fixed Ancilla) created. Num Qubits: {circuit_fixed.num_qubits}, Num Params: {len(circuit_fixed.parameters)}")
    logger.debug(f"Parameters: {[p.name for p in circuit_fixed.parameters]}")

    # --- 3. Bind Parameters for Fixed Ancilla Circuit ---
    logger.info("\n3. Binding parameters for Fixed Ancilla circuit...")
    bound_params_fixed = test_kernel._bind_parameters(
        circuit=circuit_fixed,
        diagram_main=sentence_diagram, 
        analyses_from_main_processor=analyses,
        tokens_from_main_processor=[a['text'] for a in analyses],
        linguistic_stream_core=None,
        ambiguous_word_info_map=ambiguous_map_for_sentence
    )

    if bound_params_fixed is None:
        logger.error("Parameter binding returned None for fixed ancilla case.")
    else:
        logger.info(f"Binding complete. {len(bound_params_fixed)}/{len(circuit_fixed.parameters)} parameters were bound.")
        unbound_params = circuit_fixed.parameters - set(bound_params_fixed.keys())
        if unbound_params:
            logger.warning(f"Found {len(unbound_params)} unbound parameters: {[p.name for p in unbound_params]}")
        else:
            logger.info("SUCCESS: All non-variational parameters were successfully bound.")

    # --- Test Scenario B: Variational Ancilla ---
    logger.info("\n--- Scenario B: Testing with Variational Ancilla ---")
    
    quantum_diagram_variational = functor(sentence_diagram, use_variational_ancilla=True)
    
    circuit_variational = None
    if hasattr(quantum_diagram_variational, 'to_qiskit'):
        circuit_variational = quantum_diagram_variational.to_qiskit()
    else:
        logger.error("Result of variational functor application cannot be converted to Qiskit circuit.")
        debug_quantum_result(quantum_diagram_variational, logger)
        return

    logger.info(f"Circuit (Variational Ancilla) created. Num Qubits: {circuit_variational.num_qubits}, Num Params: {len(circuit_variational.parameters)}")
    logger.debug(f"Parameters: {[p.name for p in circuit_variational.parameters]}")

    # --- 4. Bind Parameters for Variational Ancilla Circuit ---
    logger.info("\n4. Binding parameters for Variational Ancilla circuit...")
    bound_params_variational = test_kernel._bind_parameters(
        circuit=circuit_variational,
        diagram_main=sentence_diagram,
        analyses_from_main_processor=analyses,
        tokens_from_main_processor=[a['text'] for a in analyses],
        linguistic_stream_core=None,
        ambiguous_word_info_map=ambiguous_map_for_sentence
    )

    if bound_params_variational is None:
        logger.error("Parameter binding returned None for variational ancilla case.")
    else:
        logger.info(f"Binding complete. {len(bound_params_variational)}/{len(circuit_variational.parameters)} parameters were bound.")
        unbound_params_var = circuit_variational.parameters - set(bound_params_variational.keys())
        if unbound_params_var:
            logger.info(f"SUCCESS: Found {len(unbound_params_var)} unbound (variational) parameters as expected: {[p.name for p in unbound_params_var]}")
            ancilla_params_found = [p.name for p in unbound_params_var if '_ancilla_' in p.name]
            if len(ancilla_params_found) > 0:
                logger.info(f"  Correctly identified ancilla params: {ancilla_params_found}")
            else:
                logger.error("  ERROR: Unbound parameters did not seem to include the expected ancilla parameters.")
        else:
            logger.error("  ERROR: No unbound parameters found. The variational parameters were incorrectly bound.")

    logger.info("\n--- Unified Parameter Binding Test Finished ---")


if __name__ == '__main__':
    if not N_ARABIC or not isinstance(N_ARABIC, Ty):
        logger.critical("N_ARABIC from common_qnlp_types is not a valid Ty. Aborting test.")
    else:
        run_binding_test()
