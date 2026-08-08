import logging
import sys

# --- Configure Logging (Minimal for test script) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AmbiguityH3Test")

# --- Imports (MUST match your project structure) ---
try:
    from common_qnlp_types import N_ARABIC, S_ARABIC, Ty, Word, Box, AtomicType, GrammarDiagram
    from common_qnlp_types import AmbiguousLexicalBox, ControlledSenseFunctor
    from camel_test2 import analyze_arabic_sentence_with_morph, assign_discocat_types_v2_2
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}. Check sys.path and module integrity.")
    logger.error("Ensure common_qnlp_types.py defines N_ARABIC, AmbiguousLexicalBox, and ControlledSenseIQPAnsatz correctly.")
    logger.error("Ensure camel_test2.py is importable and assign_discocat_types_v2_2 is correct.")
    sys.exit(1)

logger.info(f"TEST_SCRIPT_DEBUG: AmbiguousLexicalBox_TEST imported. ID: {id(AmbiguousLexicalBox)}, Module: {AmbiguousLexicalBox.__module__}")
# Lambeq Diagram import (might be different based on your setup)
try:
    Diagram = GrammarDiagram
    logger.info("Lambeq imported from common_qnlp_types")
except ImportError:
    try: # Older lambeq might use this path
        from discopy.rigid import Diagram
        logger.debug("DiscoPy being used")
    except ImportError as e_diag:
        logger.error(f"Could not import Diagram class: {e_diag}")
        sys.exit(1)


def run_ambiguity_test():
    logger.info("--- Starting H3 Ambiguity Circuit Test ---")

    # Test sentences
    sentence_unambiguous = "جاء ولد"  # An unambiguous noun "ولد" (boy)
    sentence_ambiguous   = "جاء رجل"  # "رجل" can be man or leg

    # --- 1. Process sentences to get types/boxes ---
    logger.info(f"\nProcessing Unambiguous: '{sentence_unambiguous}'")
    tokens_u, analyses_u, structure_u, roles_u = analyze_arabic_sentence_with_morph(sentence_unambiguous, debug=True)
    
    word_box_unambiguous = None
    if analyses_u:
        for ana in analyses_u:
            if ana['text'] == 'ولد':
                # assign_discocat_types_v2_2 returns Ty('n') for "ولد"
                type_for_unambiguous = assign_discocat_types_v2_2(ana, roles_u, debug=True)
                if isinstance(type_for_unambiguous, Ty):
                    # For this test, to ensure it goes through box_to_circuit if it's not a Word,
                    # let's explicitly make it a simple Box.
                    word_entity_unambiguous = Box("ولد_std", Ty(), type_for_unambiguous)
                    logger.info(f"Created standard Box for 'ولد': {word_entity_unambiguous}")
                else: # Should ideally be Ty('n')
                    logger.error(f"Unexpected type for 'ولد': {type_for_unambiguous}")
                    word_entity_unambiguous = Box("ولد_fallback_err", Ty(), N_ARABIC)
                break
    if not word_entity_unambiguous: word_entity_unambiguous = Box("ولد_ultimate_fallback", Ty(), N_ARABIC)
    if word_box_unambiguous is None:
        logger.error("Could not get type for unambiguous word 'ولد'. Test cannot proceed as expected.")
        # As a fallback, create a standard Box for comparison
        word_box_unambiguous = Box("ولد_fallback", Ty(), N_ARABIC)
        logger.info(f"Using fallback Box for 'ولد': {word_box_unambiguous}")


    logger.info(f"\nProcessing Ambiguous: '{sentence_ambiguous}'")
    tokens_a, analyses_a, structure_a, roles_a = analyze_arabic_sentence_with_morph(sentence_ambiguous, debug=True)
    
    word_box_ambiguous = None
    if analyses_a:
        # Find 'رجل' - assume it's the second token
        for ana in analyses_a:
            if ana['text'] == 'رجل':
                word_box_ambiguous = assign_discocat_types_v2_2(ana, roles_a, debug=True, handle_lexical_ambiguity=True)
                logger.info(f"Assigned type for 'رجل': {word_box_ambiguous} (Type: {type(word_box_ambiguous)})")
                break
    if not isinstance(word_box_ambiguous, AmbiguousLexicalBox):
        logger.error(f"'رجل' was not assigned an AmbiguousLexicalBox. Got {type(word_box_ambiguous)} instead. Check camel_test2.py modification.")
        return
    if word_box_ambiguous:
        logger.info(f"TEST_SCRIPT_DEBUG: Instance 'word_entity_ambiguous' is type: {type(word_box_ambiguous)}")
        logger.info(f"TEST_SCRIPT_DEBUG: ID of type(word_entity_ambiguous): {id(type(word_box_ambiguous))}")
        logger.info(f"TEST_SCRIPT_DEBUG: Module of type(word_entity_ambiguous): {type(word_box_ambiguous).__module__}")
    # ...
    # --- 2. Create simple diagrams ---
    # Unambiguous diagram: Ty() -> N_ARABIC
    # Ensure word_box_unambiguous is a Box or Word suitable for Diagram
    if isinstance(word_box_unambiguous, Ty): # If assign_types returned just N_ARABIC
        
        diag_unambiguous = Word("ولد", word_box_unambiguous)
    elif isinstance(word_box_unambiguous, Box):
        diag_unambiguous =  word_box_unambiguous
    else:
        logger.error(f"word_box_unambiguous is of unexpected type {type(word_box_unambiguous)}. Cannot create diagram.")
        return

    # Ambiguous diagram: Ty() -> N_ARABIC (but via AmbiguousLexicalBox)
    diag_ambiguous = word_box_ambiguous # POSITIONAL ARGS
    diag_unambiguous = word_box_unambiguous
    
    # --- 3. Instantiate and apply Custom Ansatz ---
    # Define ob_map for the ansatz. N_ARABIC must be a valid key.
    if N_ARABIC is None or not hasattr(N_ARABIC, '__hash__'):
        logger.error("N_ARABIC is not properly defined or not hashable for ob_map. Exiting.")
        return
        
    ob_map_for_test = {N_ARABIC: 1} # Nouns (and their ambiguous variants) get 1 data qubit
    # The ControlledSenseIQPAnsatz will add an ancilla for ambiguous boxes.
    
    n_layers_test = 1
    n_params_per_qubit_test = 2 # e.g., for Ry, Rz

    try:
        custom_functor = ControlledSenseFunctor(ob_map_for_test, n_layers=n_layers_test, n_single_qubit_params=n_params_per_qubit_test) # NEW
        logger.info(f"Initialized ControlledSenseFunctor.")
    except Exception as e_functor_init:
        logger.error(f"Error initializing ControlledSenseFunctor: {e_functor_init}", exc_info=True)
        return

    # <<< END OF THE BLOCK >>>

    logger.info("\n--- Applying Ansatz to Unambiguous Diagram ---")
    try:
        # When an ansatz is called on a diagram, it returns a new diagram
        # where boxes are replaced by lambeq.backend.quantum.Circuit objects.
        # The final qiskit circuit is then obtained by calling .eval().circuit or similar on this new diagram.
        if diag_unambiguous:
            print(f"DEBUG: diag_unambiguous = {diag_unambiguous}")
            print(f"DEBUG: diag_unambiguous.dom = {diag_unambiguous.dom}, type = {type(diag_unambiguous.dom)}")
            print(f"DEBUG: diag_unambiguous.cod = {diag_unambiguous.cod}, type = {type(diag_unambiguous.cod)}")
            #print(f"DEBUG: diag_unambiguous.boxes = {diag_unambiguous.boxes}")

#            print(f"DEBUG: diag_unambiguous.offsets = {diag_unambiguous.offsets}")
            print(f"DEBUG: N_ARABIC used for ob_map key: {N_ARABIC}, id: {id(N_ARABIC)}")
            print(f"DEBUG: N_ARABIC in diagram cod: {diag_unambiguous.cod}, id: {id(diag_unambiguous.cod)}")
#            print(f"DEBUG: N_ARABIC in box cod: {diag_unambiguous.boxes[0].cod if diag_unambiguous.boxes else 'N/A'}, id: {id(diag_unambiguous.boxes[0].cod) if diag_unambiguous.boxes else 'N/A'}")
        quantum_diag_unambiguous = custom_functor(diag_unambiguous)
        from pytket.extensions.qiskit import tk_to_qiskit
        from lambeq.backend.converters.tk import Circuit as LambeqCircuit
        logger.info(f"Applied ansatz to unambiguous. Result type: {type(quantum_diag_unambiguous.to_tk())}")
        # To get the qiskit circuit for the *single box* in our simple diagram:
        # The quantum_diag_unambiguous.boxes[0] should be a LambeqCircuit wrapper.
        if not isinstance(quantum_diag_unambiguous.to_tk(), LambeqCircuit):
             logger.error("Ansatz application on unambiguous diagram did not produce expected LambeqCircuit box.")
             return
        circuit_unambiguous = tk_to_qiskit(quantum_diag_unambiguous.to_tk()) # Get the Qiskit circuit
        
        logger.info("Unambiguous Circuit:")
        try:
            logger.info(f"\n{circuit_unambiguous.draw(output='text')}")
        except Exception as e_draw:
            logger.warning(f"Could not draw circuit with text output: {e_draw}. Trying qasm.")
            logger.info(f"\n{circuit_unambiguous.qasm()}")

        logger.info(f"  Num Qubits: {circuit_unambiguous.num_qubits}")
        logger.info(f"  Depth: {circuit_unambiguous.depth()}")
        logger.info(f"  Parameters: {[p.name for p in circuit_unambiguous.parameters]}")
        # Expected: N_ARABIC maps to 1 qubit (from ob_map). No ancilla.
        # Expected params: e.g., ['ولد_fallback_q0_p0', 'ولد_fallback_q0_p1'] if it used fallback naming.
        # Or if assign_discocat_types_v2_2 returned a standard Box("ولد",Ty(),N), params like ['ولد_q0_p0', 'ولد_q0_p1']

    except Exception as e:
        logger.error(f"Error applying ansatz to unambiguous diagram: {e}", exc_info=True)
        return

    logger.info("\n--- Applying Ansatz to Ambiguous Diagram ---")
    try:
        quantum_diag_ambiguous = custom_functor(diag_ambiguous)
        if quantum_diag_ambiguous is None:
            logger.error("custom_functor returned None for ambiguous diagram! This is unexpected.")
            return
        
        logger.info(f"Functor returned for ambiguous: type={type(quantum_diag_ambiguous)}, name={getattr(quantum_diag_ambiguous, 'name', 'N/A')}")

        # Check that we received a circuit wrapper as expected.
        if not isinstance(quantum_diag_ambiguous, LambeqCircuit):
            logger.error(f"Functor did not return a LambeqCircuit or its subclass. Got {type(quantum_diag_ambiguous)}")
            return
        
        # FIX: The erroneous check is removed. We directly convert the result of .to_tk() to a Qiskit circuit.
        logger.info(f"  Ambiguous wrapper discopy_box: name='{quantum_diag_ambiguous.discopy_box.name}', type={type(quantum_diag_ambiguous.discopy_box)}")
        pytket_circ_ambiguous = quantum_diag_ambiguous.to_tk()
        circuit_ambiguous = tk_to_qiskit(pytket_circ_ambiguous)
        
        logger.info("Ambiguous Circuit ('رجل'):")
        try:
            logger.info(f"\n{circuit_ambiguous.draw(output='text')}")
        except Exception as e_draw:
            logger.warning(f"Could not draw circuit with text output: {e_draw}. Trying qasm.")
            logger.info(f"\n{circuit_ambiguous.qasm()}")

        logger.info(f"  Num Qubits: {circuit_ambiguous.num_qubits}")
        logger.info(f"  Depth: {circuit_ambiguous.depth()}")
        logger.info(f"  Parameters: {[p.name for p in circuit_ambiguous.parameters]}")
    
    except Exception as e:
        logger.error(f"Error applying ansatz to ambiguous diagram: {e}", exc_info=True)
        return

    logger.info("\n--- Test Parameter Binding (Conceptual) ---")
    if circuit_ambiguous and circuit_ambiguous.parameters:
        param_map_amb = {p.name: p for p in circuit_ambiguous.parameters}
        
        # Dummy feature values for senses
        features_man = {"val1": 0.5, "val2": -0.2} # e.g. from embedding("رجل_man")
        features_leg = {"val1": 0.1, "val2": 0.8} # e.g. from embedding("رجل_leg")

        bindings = {}
        # Example: binding first param of 'man' sense and 'leg' sense
        man_param_name_q0_p0 = f"{word_box_ambiguous.name}_man_q0_p0" # Match how ControlledSenseIQPAnsatz names them
        leg_param_name_q0_p0 = f"{word_box_ambiguous.name}_leg_q0_p0"
        
        if man_param_name_q0_p0 in param_map_amb:
            bindings[param_map_amb[man_param_name_q0_p0]] = features_man["val1"] * 3.14159 # Scale to angle
            logger.info(f"Binding {man_param_name_q0_p0} to {bindings[param_map_amb[man_param_name_q0_p0]]}")
        else:
            logger.warning(f"Parameter {man_param_name_q0_p0} not found in ambiguous circuit.")

        if leg_param_name_q0_p0 in param_map_amb:
            bindings[param_map_amb[leg_param_name_q0_p0]] = features_leg["val1"] * 3.14159 # Scale to angle
            logger.info(f"Binding {leg_param_name_q0_p0} to {bindings[param_map_amb[leg_param_name_q0_p0]]}")
        else:
             logger.warning(f"Parameter {leg_param_name_q0_p0} not found in ambiguous circuit.")
        
        # To actually run with these bindings:
        # from qiskit_aer import AerSimulator
        # simulator = AerSimulator()
        # bound_circuit = circuit_ambiguous.assign_parameters(bindings)
        # result = simulator.run(bound_circuit.measure_all(inplace=False), shots=100).result()
        # print(result.get_counts())
    else:
        logger.warning("No parameters in ambiguous circuit to test binding for.")

    logger.info("\n--- H3 Ambiguity Circuit Test Finished ---")

if __name__ == '__main__':
    run_ambiguity_test()
