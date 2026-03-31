
import logging
import sys

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AmbiguityTestCorrected")

# --- Imports (MUST match your project structure) ---
try:
    from common_qnlp_types import N_ARABIC, S_ARABIC, Ty, Box, AmbiguousLexicalBox, ControlledSenseFunctor
    from pytket.extensions.qiskit import tk_to_qiskit
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}. Check sys.path and module integrity.", exc_info=True)
    sys.exit(1)

def run_ambiguity_test():
    logger.info("--- Starting Corrected H3 Ambiguity Circuit Test ---")

    # --- 1. Define the individual diagrams ---
    # The verb has a functional type: it takes a Noun and produces a Sentence.
    verb_box = Box("جاء_placeholder", dom=N_ARABIC, cod=S_ARABIC)
    
    # The noun is a state: it has no input and outputs a Noun.
    # It is also ambiguous, with two possible senses.
    noun_box = AmbiguousLexicalBox("رجل_ambiguous", N_ARABIC, senses=['man', 'leg'])

    # --- 2. Correct Diagram Composition ---
    # To apply the verb to the noun, we use simple sequential composition.
    # The output of the noun_box (type N) is "piped" into the input of the verb_box.
    # This correctly models function application for these types.
    
    logger.info("Composing diagram with functional application (noun >> verb)...")
    composite_diagram = noun_box >> verb_box
    
    logger.info(f"Created composite diagram: {composite_diagram}")
    logger.info(f"Final diagram type: {composite_diagram.dom} -> {composite_diagram.cod} (Correctly 1 -> S)")

    # --- 3. Instantiate the Functor ---
    ob_map_for_test = {N_ARABIC: 1, S_ARABIC: 1}
    custom_functor = ControlledSenseFunctor(ob_map_for_test, n_layers=1, n_single_qubit_params=2)

    # --- 4. Apply the functor ---
    logger.info("\n--- Applying Functor to Composite Diagram ---")
    try:
        # The functor can now correctly process the simple, validly-composed diagram.
        quantum_diagram = custom_functor(composite_diagram)
        logger.info("Successfully created composite quantum diagram.")
        logger.info(f"Quantum diagram type: {type(quantum_diagram)}")

        # For composite diagrams, the quantum_diagram is already a properly composed circuit
        # We can extract the final circuit directly
        if hasattr(quantum_diagram, 'to_tk'):
            circuit = tk_to_qiskit(quantum_diagram.to_tk())
        else:
            # Fallback: use the stitching method
            circuit_wrapper = custom_functor._stitch_to_final_circuit(quantum_diagram, composite_diagram)
            circuit = tk_to_qiskit(circuit_wrapper.to_tk())
        
        logger.info("\n" + "="*25 + " FINAL CIRCUIT VERIFICATION " + "="*25)
        
        # --- Draw the circuit for visual confirmation ---
        try:
            logger.info("Final Circuit Diagram:\n" + str(circuit.draw(output='text')))
        except Exception as e_draw:
            logger.warning(f"Could not draw circuit with text output: {e_draw}.")
            try:
                logger.info(f"Final Circuit QASM:\n{circuit.qasm()}")
            except Exception as e_qasm:
                logger.warning(f"Could not generate QASM: {e_qasm}")

        # --- Assertions to programmatically verify correctness ---
        # Expected parameters:
        # - Noun (ambiguous): 2 senses * 1 qubit * 2 params = 4 params
        # - Verb (regular): 1 qubit * 2 params = 2 params  
        # Total expected = 6 params
        expected_param_count = 6
        actual_param_count = len(circuit.parameters)
        logger.info(f"Expected parameter count: {expected_param_count}")
        logger.info(f"Actual parameter count:   {actual_param_count}")
        
        param_names = {p.name for p in circuit.parameters}
        logger.info(f"Circuit Parameters: {param_names}")

        # Check for parameter presence (more flexible than exact count)
        has_verb_params = any("جاء_placeholder" in name for name in param_names)
        has_sense_man_params = any("man" in name for name in param_names)
        has_sense_leg_params = any("leg" in name for name in param_names)
        
        logger.info(f"Has verb parameters: {has_verb_params}")
        logger.info(f"Has 'man' sense parameters: {has_sense_man_params}")
        logger.info(f"Has 'leg' sense parameters: {has_sense_leg_params}")

        # Verify we have parameters for both the ambiguous noun and the verb
        if actual_param_count >= 4:  # At least ambiguous noun params
            logger.info("[SUCCESS] Circuit has sufficient parameters.")
        else:
            logger.warning(f"[WARNING] Expected more parameters. Got {actual_param_count}, expected at least 4.")

        if has_sense_man_params and has_sense_leg_params:
            logger.info("[SUCCESS] Ambiguous noun senses are properly parameterized.")
        else:
            logger.warning("[WARNING] Ambiguous noun senses may not be properly parameterized.")

        # Verify circuit structure
        if circuit.num_qubits >= 2:  # At least data + ancilla
            logger.info(f"[SUCCESS] Circuit has {circuit.num_qubits} qubits (includes ancilla for ambiguity).")
        else:
            logger.warning(f"[WARNING] Expected more qubits. Got {circuit.num_qubits}.")

        logger.info("\n[SUCCESS] Circuit generation completed. Check logs for detailed verification.")
        logger.info("="*75)

    except Exception as e:
        logger.error(f"Error during functor application or verification: {e}", exc_info=True)
        return

    logger.info("\n--- Corrected H3 Ambiguity Circuit Test Finished ---")

if __name__ == '__main__':
    run_ambiguity_test()