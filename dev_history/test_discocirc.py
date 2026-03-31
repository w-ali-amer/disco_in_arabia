# test_arabic_discocirc_pipeline.py
# A script to test the output of the arabic_discocirc_pipeline.py module.

import logging
import os
from typing import Optional, List, Dict, Any

# Lambeq imports (primarily for type hinting and potentially drawing)
from lambeq.backend.grammar import Box, Diagram as GrammarDiagram
from lambeq import AtomicType # For type checking and potentially creating dummy types if needed

# --- Import the main function from your pipeline ---
try:
    from arabic_discocirc_pipeline import generate_discocirc_ready_diagram
    ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = True
    # The pipeline module should configure its own logger,
    # but we can get it if we want to adjust its level for testing.
    # pipeline_logger = logging.getLogger('arabic_discocirc_pipeline')
    # pipeline_logger.setLevel(logging.DEBUG) # Example: set to DEBUG for this test
except ImportError as e:
    logging.error(f"Failed to import 'generate_discocirc_ready_diagram' from 'arabic_discocirc_pipeline.py': {e}")
    ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = False
    # Define a dummy function if import fails, so the script can still run basic checks or report failure
    def generate_discocirc_ready_diagram(*args, **kwargs) -> Optional[GrammarDiagram]: # type: ignore
        logging.error("Called dummy 'generate_discocirc_ready_diagram' due to import failure.")
        return None

# --- Configure Logging for this Test Script ---
# This ensures that if this script is the entry point, logging is set up.
# If another script (like exp4.py) imports this, its logging config might take precedence.
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
    logging.basicConfig(
        level=logging.DEBUG, # Set to DEBUG for detailed test output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("test_discocirc_pipeline_output.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Silence matplotlib's verbose logs

# --- Helper to print diagram box data ---
def print_box_data_summary(diagram: GrammarDiagram, num_boxes_to_inspect: int = 3):
    """Prints a summary of the .data attribute for a few boxes in the diagram."""
    if not diagram or not diagram.boxes:
        logger.info("  Diagram has no boxes to inspect.")
        return

    logger.info(f"  Inspecting data for up to {num_boxes_to_inspect} boxes:")
    for i, box in enumerate(diagram.boxes[:num_boxes_to_inspect]):
        logger.info(f"    Box {i}: '{box.name}' (Type: {type(box).__name__}, Dom: {box.dom}, Cod: {box.cod})")
        if hasattr(box, 'data') and box.data:
            logger.info(f"      Data keys: {list(box.data.keys())}")
            # Print a few key-value pairs from the data to check enrichment
            enrichment_source = box.data.get('discocirc_enrichment_source')
            if enrichment_source:
                logger.info(f"        Enrichment Source: {enrichment_source}")
            
            extracted_root = box.data.get('extracted_root')
            if extracted_root:
                logger.info(f"        Extracted Root: {extracted_root}")

            classical_features = box.data.get('classical_features_of_surface_word')
            if classical_features is not None: # Check for None explicitly
                 logger.info(f"        Classical Features Vector (exists): True, Shape: {getattr(classical_features, 'shape', 'N/A')}")
            else:
                 logger.info(f"        Classical Features Vector (exists): False")

            camel_analysis = box.data.get('camel_analysis_of_surface_word')
            if camel_analysis and isinstance(camel_analysis, dict):
                logger.info(f"        CAMeL POS (surface): {camel_analysis.get('pos', 'N/A')}")
        else:
            logger.info("      Box has no .data attribute or .data is empty.")

# --- Main Test Function ---
def run_pipeline_tests():
    """Runs tests on the Arabic DisCoCirc pipeline."""
    if not ARABIC_DISCOCIRC_PIPELINE_AVAILABLE:
        logger.critical("Arabic DisCoCirc pipeline is not available due to import errors. Cannot run tests.")
        return

    logger.info("===================================================")
    logger.info("  STARTING ARABIC DISCOCIRC PIPELINE TEST SUITE  ")
    logger.info("===================================================")

    test_sentences: List[Dict[str, Any]] = [
        {"id": "sent_01", "text": "يقرأ الولد الكتاب بتمعن"},
        {"id": "sent_02", "text": "البيت الكبير جميل جدا"},
        {"id": "sent_03", "text": "هذا هو الرجل الذي ساعدني"},
        {"id": "sent_04", "text": "القطة السوداء الصغيرة نائمة بهدوء فوق الأريكة"}, # More complex
        {"id": "sent_05", "text": "سافر الرجل إلى المدينة"}, # Simple SVO
        {"id": "sent_06", "text": "لم يأت"}, # Very short, potentially problematic
        {"id": "sent_07", "text": "مهندس"}, # Single word
    ]

    output_dir_for_test_diagrams = "test_pipeline_diagram_outputs"
    os.makedirs(output_dir_for_test_diagrams, exist_ok=True)
    logger.info(f"Diagrams will be saved (if successful) in: {os.path.abspath(output_dir_for_test_diagrams)}")

    overall_success_count = 0
    overall_failure_count = 0

    for i, sentence_data in enumerate(test_sentences):
        sentence_id = sentence_data["id"]
        sentence_text = sentence_data["text"]
        logger.info(f"\n--- Test Case {i+1}/{len(test_sentences)} (ID: {sentence_id}) ---")
        logger.info(f"Input Sentence: \"{sentence_text}\"")

        enriched_diagram: Optional[GrammarDiagram] = None
        try:
            # Call the main function from your pipeline
            # Pass debug=True to get more verbose logging from the pipeline itself
            enriched_diagram = generate_discocirc_ready_diagram(
                sentence_str=sentence_text,
                debug=True, # Enable debug for the pipeline call
                classical_feature_dim_for_enrichment=16 # Example, adjust if needed
            )

            if enriched_diagram is not None:
                logger.info(f"SUCCESS: Diagram generated for '{sentence_text}'.")
                logger.info(f"  Diagram Type: {type(enriched_diagram).__name__}")
                logger.info(f"  Is instance of GrammarDiagram: {isinstance(enriched_diagram, GrammarDiagram)}")
                logger.info(f"  Number of Boxes: {len(enriched_diagram.boxes)}")
                logger.info(f"  Number of Wires in Domain: {len(enriched_diagram.dom)}")
                logger.info(f"  Number of Wires in Codomain: {len(enriched_diagram.cod)}")
                logger.info(f"  Domain: {enriched_diagram.dom}")
                logger.info(f"  Codomain: {enriched_diagram.cod}")

                # Inspect data of some boxes
                print_box_data_summary(enriched_diagram, num_boxes_to_inspect=5)

                # Attempt to save the diagram visualization
                diagram_filename = f"{sentence_id}_enriched_diagram.png"
                diagram_save_path = os.path.join(output_dir_for_test_diagrams, diagram_filename)
                try:
                    # Ensure the diagram object has a draw method
                    if hasattr(enriched_diagram, 'draw'):
                        enriched_diagram.draw(path=diagram_save_path, show=False, figsize=(12, 8), aspect='auto', margins=(0.1,0.1))
                        logger.info(f"  Diagram image saved to: {diagram_save_path}")
                    else:
                        logger.warning("  Diagram object does not have a 'draw' method. Cannot save image.")
                except Exception as e_draw:
                    logger.error(f"  Error saving diagram image for '{sentence_text}': {e_draw}", exc_info=False) # Set exc_info to True for full traceback
                overall_success_count +=1
            else:
                logger.error(f"FAILURE: No diagram returned for '{sentence_text}'.")
                overall_failure_count += 1

        except Exception as e:
            logger.error(f"EXCEPTION during pipeline processing for '{sentence_text}': {e}", exc_info=True)
            overall_failure_count += 1
        
        logger.info("--- End Test Case ---")

    logger.info("\n===================================================")
    logger.info("            ARABIC DISCOCIRC PIPELINE TEST SUMMARY            ")
    logger.info(f"  Total Sentences Tested: {len(test_sentences)}")
    logger.info(f"  Successful Diagram Generations: {overall_success_count}")
    logger.info(f"  Failed Diagram Generations: {overall_failure_count}")
    logger.info("===================================================")
    logger.info(f"Test log saved to: {os.path.abspath('test_discocirc_pipeline_output.log')}")


if __name__ == '__main__':
    # This check ensures that if camel_test2.py (or other modules) also try to set up
    # basicConfig, it doesn't cause issues. The first basicConfig call usually wins.
    # However, the logger for *this* module (__name__) is configured above.
    
    # Before running tests, ensure that the dependencies of arabic_discocirc_pipeline.py
    # (like camel_test2.py and arabic_morpho_lex_core.py) can initialize their
    # Stanza/CAMeL tools instances correctly. If they rely on global variables for these,
    # those need to be available. The pipeline module itself tries to initialize them.

    # A common pattern for modules that use global-like resources (e.g., NLP models):
    #
    # In camel_test2.py (and arabic_morpho_lex_core.py if similar):
    #
    # N = None # Or some default
    # S = None
    # nlp_pipeline_camel = None
    #
    # def initialize_resources():
    #     global N, S, nlp_pipeline_camel
    #     if N is None: N = AtomicType.NOUN
    #     if S is None: S = AtomicType.SENTENCE
    #     if nlp_pipeline_camel is None:
    #         nlp_pipeline_camel = stanza.Pipeline(...)
    #     # ... initialize other global-like resources ...
    #
    # # Call at the end of the module if it can be imported,
    # # or explicitly call it from the main script (like exp4.py or this test script)
    # # initialize_resources()
    #
    # The `arabic_discocirc_pipeline.py` already attempts its own initializations
    # if it can't find pre-initialized ones from imported modules, which is good.

    run_pipeline_tests()
