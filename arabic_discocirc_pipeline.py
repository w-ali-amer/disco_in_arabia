import logging
logger = logging.getLogger(__name__)

# --- Import shared Lambeq types ---
try:
    from common_qnlp_types import (
        N_ARABIC, S_ARABIC, ROOT_TYPE_ARABIC as ROOT_TYPE_COMMON, 
        LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY,
        AtomicType, Ty, Box as CommonBox, 
        GrammarDiagram as CommonGrammarDiagram, Functor as CommonFunctor, Word as CommonWord,
        IQPAnsatz as CommonIQPAnsatz, 
        SpiderAnsatz as CommonSpiderAnsatz, 
        StronglyEntanglingAnsatz as CommonStronglyEntanglingAnsatz
    )
    if not LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY:
        logger.warning("arabic_discocirc_pipeline: common_qnlp_types reported that Lambeq types were NOT initialized successfully. Expect issues.")
    else:
        logger.info("arabic_discocirc_pipeline: Successfully imported types from common_qnlp_types.")
    
    # Make Lambeq classes available directly
    AtomicType = AtomicType
    Ty = Ty
    Box = CommonBox
    GrammarDiagram = CommonGrammarDiagram
    Functor = CommonFunctor
    Word = CommonWord
    IQPAnsatz = CommonIQPAnsatz
    SpiderAnsatz = CommonSpiderAnsatz
    StronglyEntanglingAnsatz = CommonStronglyEntanglingAnsatz

except ImportError as e_common_types_adp:
    logger.critical(f"arabic_discocirc_pipeline: CRITICAL - Failed to import from common_qnlp_types.py: {e_common_types_adp}. This module will not function correctly.", exc_info=True)
    # Define dummies
    class FallbackTyPlaceholderADP:
        def __init__(self, name): self.name = name
        def __str__(self): return self.name
    N_ARABIC = FallbackTyPlaceholderADP('n_adp_dummy_NI') # type: ignore
    S_ARABIC = FallbackTyPlaceholderADP('s_adp_dummy_NI') # type: ignore
    ROOT_TYPE_COMMON = N_ARABIC # type: ignore
    if 'AtomicType' not in globals(): 
        class AtomicType:
            pass # type: ignore
    if 'Ty' not in globals():
        class Ty:
            pass # type: ignore
    if 'Box' not in globals():
        class Box:
            def __init__(self, *args, **kwargs):
                self.name="DummyBoxADP"
                self.dom=None
                self.cod=None
                self.data=None # type: ignore
    if 'Functor' not in globals():
        class Functor:
            pass # type: ignore
    if 'GrammarDiagram' not in globals():
        class GrammarDiagram:
            pass # type: ignore
    if 'Word' not in globals():
        class Word:
            pass # type: ignore
    if 'IQPAnsatz' not in globals():
        class IQPAnsatz:
            pass # type: ignore
    if 'SpiderAnsatz' not in globals():
        class SpiderAnsatz:
            pass # type: ignore
    if 'StronglyEntanglingAnsatz' not in globals():
        class StronglyEntanglingAnsatz:
            pass # type: ignore


import stanza
import os
import copy
from typing import List, Dict, Tuple, Optional, Any, Set, Union

# Imports from user's existing files

# From camel_test2.py:
try:
    from camel_test2 import (
        analyze_arabic_sentence_with_morph,
        assign_discocat_types_v2_2,
        create_verbal_sentence_diagram_v3_7, # Import specific diagram functions
        create_nominal_sentence_diagram_v2_7, # Import specific diagram functions
        N as N_camel, S as S_camel, 
        ADJ_MOD_TYPE as ADJ_MOD_TYPE_camel, 
        ADJ_PRED_TYPE as ADJ_PRED_TYPE_camel,
        DET_TYPE as DET_TYPE_camel,
        PREP_FUNCTOR_TYPE as PREP_FUNCTOR_TYPE_camel,
        VERB_INTRANS_TYPE as VERB_INTRANS_TYPE_camel,
        VERB_TRANS_TYPE as VERB_TRANS_TYPE_camel,
        S_MOD_BY_N as S_MOD_BY_N_camel,
        N_MOD_BY_N as N_MOD_BY_N_camel,
        ADV_FUNCTOR_TYPE as ADV_FUNCTOR_TYPE_camel,
        STANZA_AVAILABLE as CAMEL_STANZA_AVAILABLE,
        CAMEL_ANALYZER as CAMEL_TOOLS_ANALYZER_camel,
        nlp as stanza_pipeline_camel,
        build_np_diagram_v4 # Assuming this is used by diagram creation functions
        # REMOVED: create_discocat_diagram_v2_7_3 
    )
    CAMEL_TEST2_AVAILABLE = True
    logger_camel_test2 = logging.getLogger('camel_test2') # Ensure this logger exists or remove
    logger_camel_test2.info("Successfully imported components from camel_test2.py for DisCoCirc pipeline.")
except ImportError as e:
    logging.error(f"Failed to import from camel_test2.py: {e}")
    CAMEL_TEST2_AVAILABLE = False
    N_camel, S_camel = None, None 


# From arabic_morpho_lex_core.py:
ARABIC_MORPHO_LEX_CORE_AVAILABLE = False
try:
    import arabic_morpho_lex_core 
    from arabic_morpho_lex_core import (
        analyze_sentence_for_root_transform, # Renamed in core to return tuple
        STANZA_PIPELINE as stanza_pipeline_core,
        CAMEL_ANALYZER as CAMEL_TOOLS_ANALYZER_core,
        N as N_core,
        S as S_core,
        ROOT_TYPE as ROOT_TYPE_core
        # REMOVED: TransformationBox (as it's not a class to be imported)
    )
    ARABIC_MORPHO_LEX_CORE_AVAILABLE = True
    logger_core = logging.getLogger('arabic_morpho_lex_core') # Ensure this logger exists or remove
    logger_core.info("Successfully imported components from arabic_morpho_lex_core.py for DisCoCirc pipeline.")
except ImportError as e:
    logging.error(f"Failed to import from arabic_morpho_lex_core.py: {e}")
    N_core, S_core, ROOT_TYPE_core = None, None, None


# Configure logging for this module
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers(): # Avoid duplicate handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not CAMEL_TEST2_AVAILABLE:
    # This is a critical failure, so raise an error to stop execution.
    raise ImportError("Critical components from camel_test2.py could not be loaded. DisCoCirc pipeline cannot function.")
if not ARABIC_MORPHO_LEX_CORE_AVAILABLE:
    logger.warning("Components from arabic_morpho_lex_core.py could not be loaded. Rich feature enrichment will be unavailable.")

# Stanza and CAMeL tools initialization (preferring instances from imported modules if available)
nlp_pipeline = None
if CAMEL_STANZA_AVAILABLE and stanza_pipeline_camel:
    nlp_pipeline = stanza_pipeline_camel
    logger.info("Using Stanza pipeline initialized in camel_test2.py.")
elif ARABIC_MORPHO_LEX_CORE_AVAILABLE and stanza_pipeline_core:
    nlp_pipeline = stanza_pipeline_core
    logger.info("Using Stanza pipeline initialized in arabic_morpho_lex_core.py.")
else:
    try:
        logger.info("Attempting to initialize a new Stanza pipeline for DisCoCirc module...")
        nlp_pipeline = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse,mwt', verbose=False, use_gpu=False, logging_level='WARN')
        logger.info("New Stanza pipeline initialized successfully for DisCoCirc module.")
    except Exception as e_stanza_init:
        logger.error(f"Failed to initialize Stanza pipeline in DisCoCirc module: {e_stanza_init}")
        # Depending on criticality, you might want to raise an error here too.
        # For now, just log, as the main check is for CAMEL_TEST2_AVAILABLE.

camel_analyzer_instance = None
if CAMEL_TOOLS_ANALYZER_camel:
    camel_analyzer_instance = CAMEL_TOOLS_ANALYZER_camel
    logger.info("Using CAMeL Analyzer initialized in camel_test2.py.")
elif ARABIC_MORPHO_LEX_CORE_AVAILABLE and CAMEL_TOOLS_ANALYZER_core:
    camel_analyzer_instance = CAMEL_TOOLS_ANALYZER_core
    logger.info("Using CAMeL Analyzer initialized in arabic_morpho_lex_core.py.")
else:
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        logger.info("Attempting to initialize a new CAMeL Analyzer for DisCoCirc module...")
        db_path = MorphologyDB.builtin_db()
        camel_analyzer_instance = Analyzer(db_path)
        logger.info("New CAMeL Analyzer initialized successfully for DisCoCirc module.")
    except Exception as e_camel_init:
        logger.error(f"Failed to initialize CAMeL Analyzer in DisCoCirc module: {e_camel_init}")

if not nlp_pipeline:
    # If Stanza is critical for all paths, raise error.
    # If only for core module, a warning might suffice if core module is optional.
    logger.error("Stanza pipeline is not available or failed to initialize. DisCoCirc pipeline functionality will be severely limited or fail.")
    # raise RuntimeError("Stanza pipeline is not available or failed to initialize.")

if not camel_analyzer_instance and ARABIC_MORPHO_LEX_CORE_AVAILABLE: 
    logger.warning("CAMeL Analyzer is not available for DisCoCirc pipeline. Morphological analysis by core module might be affected.")


def create_ansatz_functor_for_core_module(
    ansatz_config_from_kernel: Optional[Dict[str, Any]],
    default_n_qubits: int = 1
) -> Optional[Functor]:
    if not ansatz_config_from_kernel:
        logger.warning("ADP: No ansatz_config_from_kernel provided to create_ansatz_functor_for_core_module. Cannot create ansatz.")
        return None

    ansatz_choice = ansatz_config_from_kernel.get('name', 'IQP')
    
    N_core_type_from_config = ansatz_config_from_kernel.get('N_core')
    ROOT_TYPE_core_from_config = ansatz_config_from_kernel.get('ROOT_TYPE_core')
    S_core_type_from_config = ansatz_config_from_kernel.get('S_core') 

    if not N_core_type_from_config or not ROOT_TYPE_core_from_config: # S_core might be optional
        logger.error("ADP: Core atomic types (N_core, ROOT_TYPE_core) not properly provided in ansatz_config. Cannot create core ansatz functor.")
        return None

    ob_map = { N_core_type_from_config: default_n_qubits }
    if ROOT_TYPE_core_from_config is not N_core_type_from_config: # Only add if distinct
         ob_map[ROOT_TYPE_core_from_config] = default_n_qubits
    if S_core_type_from_config: # Add S_core if provided and distinct
        if S_core_type_from_config is not N_core_type_from_config and S_core_type_from_config is not ROOT_TYPE_core_from_config:
            ob_map[S_core_type_from_config] = default_n_qubits
        elif S_core_type_from_config in ob_map and ob_map[S_core_type_from_config] != default_n_qubits: # Check if already mapped differently
            logger.warning(f"ADP: S_core type was already in ob_map with a different qubit count. Overwriting with {default_n_qubits}.")
            ob_map[S_core_type_from_config] = default_n_qubits


    logger.debug(f"ADP: Creating core ansatz. Choice: {ansatz_choice}. Ob_map keys: {[str(k) for k in ob_map.keys()]}")

    try:
        ansatz_instance: Optional[Functor] = None
        if ansatz_choice == 'IQP':
            n_layers = ansatz_config_from_kernel.get('n_layers_iqp', 1)
            # n_single_qubit_params should be passed from kernel, or use default from IQPAnsatz
            n_single_qubit_params = ansatz_config_from_kernel.get('n_single_qubit_params_iqp', IQPAnsatz.DEFAULT_N_SINGLE_QUBIT_PARAMS) # Use Lambeq's default
            ansatz_instance = IQPAnsatz(ob_map, n_layers=n_layers, n_single_qubit_params=n_single_qubit_params)
        elif ansatz_choice == 'StronglyEntangling':
            n_layers = ansatz_config_from_kernel.get('n_layers_strong', 1)
            ansatz_instance = StronglyEntanglingAnsatz(ob_map, n_layers=n_layers)
        elif ansatz_choice == 'Spider':
            ansatz_instance = SpiderAnsatz(ob_map)
        else:
            logger.warning(f"ADP: Unknown or unsupported ansatz choice '{ansatz_choice}' for core module. Defaulting to IQP.")
            ansatz_instance = IQPAnsatz(ob_map, n_layers=1, n_single_qubit_params=IQPAnsatz.DEFAULT_N_SINGLE_QUBIT_PARAMS)
        
        logger.info(f"ADP: Successfully created '{ansatz_choice}' ansatz for core module.")
        return ansatz_instance
    except Exception as e:
        logger.error(f"ADP: Exception during ansatz instantiation for core: {e}", exc_info=True)
        return None

def enrich_diagram_with_core_qnlp_features(
    base_diagram: GrammarDiagram,
    linguistic_stream_from_core: List[Dict[str, Any]],
    ansatz_functor_for_core: Optional[Functor], 
    original_indices_in_base_diagram: List[int], # Not directly used here, mapping is by original_stanza_idx
    debug: bool = False
) -> GrammarDiagram:
    if not ARABIC_MORPHO_LEX_CORE_AVAILABLE or not linguistic_stream_from_core:
        logger.warning("ADP: Core module features not available or stream is empty. Skipping core enrichment.")
        return base_diagram

    enriched_diagram = copy.deepcopy(base_diagram) 

    core_data_map: Dict[int, Dict[str, Any]] = {}
    for core_entry in linguistic_stream_from_core:
        orig_idx = core_entry.get('original_stanza_idx') 
        if orig_idx is not None:
            core_data_map[orig_idx] = core_entry
        else:
            logger.warning(f"ADP: Core data entry missing 'original_stanza_idx': {core_entry.get('surface_text', 'Unknown token')}")

    new_boxes = []
    for i, box in enumerate(enriched_diagram.boxes):
        new_box_data = box.data.copy() if box.data else {}
        
        box_original_idx: Optional[int] = None
        # Try to extract original_stanza_idx from box.data if it was added by camel_test2
        # Or parse from box.name if it follows "lemma_originalidx" convention
        if 'original_stanza_idx' in new_box_data:
            box_original_idx = new_box_data['original_stanza_idx']
        elif '_' in box.name:
            parts = box.name.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                try: box_original_idx = int(parts[-1])
                except ValueError: pass
        
        if box_original_idx is not None and box_original_idx in core_data_map:
            word_core_data = core_data_map[box_original_idx]
            
            new_box_data['discocirc_enrichment_source'] = 'arabic_morpho_lex_core'
            # Add specific fields, not just a blind update, to avoid overwriting existing important data
            fields_to_add_from_core = [
                'extracted_root', 'core_linguistic_representation', 
                'classical_features_of_surface_word', 'camel_analysis_of_surface_word'
            ]
            for field in fields_to_add_from_core:
                if field in word_core_data:
                    new_box_data[field] = word_core_data[field]
            
            word_diagram_from_core = word_core_data.get('word_diagram_from_core') # This key needs to be added by core module
            if word_diagram_from_core and ansatz_functor_for_core:
                try:
                    ansatz_circuit_for_word = ansatz_functor_for_core(word_diagram_from_core)
                    new_box_data['core_word_ansatz_circuit'] = ansatz_circuit_for_word
                    if debug: logger.debug(f"ADP: Applied core ansatz to word diagram for '{word_core_data.get('surface_text', box.name)}'.")
                except Exception as e_ansatz_apply:
                    logger.error(f"ADP: Failed to apply core ansatz to word diagram for '{word_core_data.get('surface_text', box.name)}': {e_ansatz_apply}", exc_info=debug)
            
            new_boxes.append(Box(box.name, box.dom, box.cod, data=new_box_data, _dagger=box._dagger))
            if debug: logger.info(f"ADP: Enriched box '{box.name}' (orig_idx {box_original_idx}) with core features.")
        else:
            new_boxes.append(box) 
            if debug and box_original_idx is not None:
                 logger.debug(f"ADP: No core data found for box '{box.name}' (orig_idx {box_original_idx}).")
            elif debug:
                 logger.debug(f"ADP: Could not determine original_stanza_idx for box '{box.name}' to fetch core data.")

    return GrammarDiagram(dom=enriched_diagram.dom, cod=enriched_diagram.cod, boxes=new_boxes, offsets=enriched_diagram.offsets)


def generate_discocirc_ready_diagram(
    sentence_str: str,
    # Pass the actual functions from camel_test2, not strings
    sentence_analyzer_func: callable = analyze_arabic_sentence_with_morph,
    type_assigner_func: callable = assign_discocat_types_v2_2,
    # bobcat_parser_instance: Optional[BobcatParser] = None, # If using Bobcat
    debug: bool = False,
    # classical_feature_dim_for_enrichment: int = 16 # If you plan to use this
) -> Optional[GrammarDiagram]:
    """
    Generates a DisCoCirc-style diagram, processing types from camel_test2.
    V2: Converts Ty objects to Word objects.
    """
    logger.info(f"DisCoCirc Pipeline (ADP): Processing sentence: '{sentence_str}'")

    try:
        tokens_camel, analyses_details_camel, structure_camel, roles_camel = \
            sentence_analyzer_func(sentence_str, debug=debug)

        if not analyses_details_camel:
            logger.error("ADP: Sentence analysis by sentence_analyzer_func returned no analysis details.")
            return None
        # Ensure roles_camel has the analysis map
        if 'analysis_map_for_diagram_creation' not in roles_camel:
            roles_camel['analysis_map_for_diagram_creation'] = {a['original_idx']: a for a in analyses_details_camel}

    except Exception as e_analysis:
        logger.error(f"ADP: Exception during sentence_analyzer_func: {e_analysis}", exc_info=True)
        return None

    logger.info(f"ADP: Initial camel_test2 analysis complete. Structure: {structure_camel}, Tokens: {len(tokens_camel)}")

    # This list will hold Word or Box objects
    word_core_types_list_camel: List[Union[Word, Box]] = []
    original_indices_for_diagram_camel: List[int] = []
    filtered_tokens_for_diagram_camel: List[str] = []

    logger.debug(f"ADP: --- Assigning Core Types (via type_assigner_func) for DisCoCirc path for: '{sentence_str}' ---")
    for i, analysis_entry_camel in enumerate(analyses_details_camel):
        current_assigned_entity = type_assigner_func(
            analysis=analysis_entry_camel,
            roles=roles_camel,
            debug=debug
        )

        if current_assigned_entity is not None:
            final_entity_for_list: Optional[Union[Word, Box]] = None
            token_text = analysis_entry_camel.get('text', f"unk_{i}")
            token_lemma = analysis_entry_camel.get('lemma', token_text)

            if isinstance(current_assigned_entity, Ty):
                # Convert Ty to Word, assign data from analysis_entry_camel
                final_entity_for_list = Word(token_lemma, current_assigned_entity)
                # It's good practice for Word objects in DisCoCirc to carry original token info
                final_entity_for_list.data = { # type: ignore
                    'original_text': token_text,
                    'original_idx': analysis_entry_camel['original_idx'],
                    'upos': analysis_entry_camel.get('upos'),
                    'deprel': analysis_entry_camel.get('deprel')
                }
                logger.debug(f"ADP: Token '{token_text}' (idx {analysis_entry_camel['original_idx']}) assigned Ty '{current_assigned_entity}'. Converted to Word: '{final_entity_for_list.name}': {final_entity_for_list.cod}")

            elif isinstance(current_assigned_entity, Box):
                final_entity_for_list = current_assigned_entity
                # Ensure Box.data exists and add original_idx
                if not hasattr(final_entity_for_list, 'data') or final_entity_for_list.data is None:
                    final_entity_for_list.data = {} # type: ignore
                if isinstance(final_entity_for_list.data, dict): # type: ignore
                    final_entity_for_list.data['original_stanza_idx'] = analysis_entry_camel['original_idx'] # Using stanza_idx for consistency with ADP logs
                    final_entity_for_list.data['original_text'] = token_text # Add original text
                logger.debug(f"ADP: Token '{token_text}' (idx {analysis_entry_camel['original_idx']}) assigned Box: '{final_entity_for_list.name}': {final_entity_for_list.dom} >> {final_entity_for_list.cod}") # type: ignore
            else:
                logger.warning(f"ADP: Token '{token_text}' assigned unexpected type '{type(current_assigned_entity)}' by type_assigner_func. Skipping.")
                continue

            word_core_types_list_camel.append(final_entity_for_list)
            original_indices_for_diagram_camel.append(analysis_entry_camel['original_idx'])
            filtered_tokens_for_diagram_camel.append(token_text)
        else:
            logger.debug(f"ADP: Token '{analysis_entry_camel.get('text')}' (orig_idx {analysis_entry_camel.get('original_idx')}) assigned None core type, excluding from ADP.")

    logger.debug(f"ADP: word_core_types_list_camel for '{sentence_str}': {[str(t) for t in word_core_types_list_camel]}")
    logger.debug(f"ADP: Corresponding original indices: {original_indices_for_diagram_camel}")

    if not word_core_types_list_camel:
        logger.error("ADP: No tokens were converted to Word/Box objects. Cannot build diagram for enriched path.")
        return None

    # ------------------------------------------------------------------
    # Placeholder for ADP's actual diagram construction logic.
    # This logic needs to be able to take `word_core_types_list_camel`
    # (which is a List[Union[Word, Box]]) and construct a sentence diagram.
    # For the purpose of this fix, we'll assume that if word_core_types_list_camel is populated,
    # the rest of ADP *might* work or will have its own errors if its internal composition fails.
    # The key was to provide it with a non-empty list of diagrammatic elements.
    # ------------------------------------------------------------------
    logger.warning("ADP: Diagram construction from word_core_types_list_camel is a placeholder in this snippet. Returning None to trigger camel_test2 fallback.")
    # To allow camel_test2 to fallback, return None for now, as ADP's full logic isn't here.
    # If ADP has its own full diagram builder, it would return that diagram.
    return None



# --- Test Runner (Optional, for standalone testing of this pipeline) ---
if __name__ == '__main__':
    # Ensure loggers from imported modules don't propagate if not desired for standalone test
    # logging.getLogger('camel_test2').propagate = False 
    # logging.getLogger('arabic_morpho_lex_core').propagate = False

    logger.info("Running arabic_discocirc_pipeline.py directly for testing...")

    test_sentences = [
        "الولد يقرا الكتاب", # SVO
        "المهندس بنى البيت", # VSO (Stanza might see SVO)
        "عين الرجل جميلة",   # Nominal
        "شربت من عين الماء", # Verbal with PP
        "المهندسون يعملون بجد", 
        "المهندسة تعمل بجد",   
    ]
    
    test_ansatz_config = None
    if ARABIC_MORPHO_LEX_CORE_AVAILABLE and N_core and ROOT_TYPE_core and S_core: 
        test_ansatz_config = {
            'name': 'IQP', 'n_layers_iqp': 1, 'n_single_qubit_params_iqp': 2,
            'N_core': N_core, 'S_core': S_core, 'ROOT_TYPE_core': ROOT_TYPE_core
        }
        logger.info(f"Using test_ansatz_config with core types: N={N_core}, S={S_core}, ROOT={ROOT_TYPE_core}")
    else:
        logger.warning("Core atomic types from arabic_morpho_lex_core not available for test_ansatz_config. Core ansatz might not work.")

    output_dir_test = "adp_test_outputs_import_fix"
    os.makedirs(output_dir_test, exist_ok=True)

    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n--- Testing Sentence {i+1}: '{sentence}' ---")
        try:
            result_tuple = generate_discocirc_ready_diagram(
                sentence_str=sentence, debug=True,
                ansatz_config=test_ansatz_config,
                use_core_enhancements_locally_in_adp=True 
            )
            if result_tuple:
                enriched_diagram, core_stream, structure, _, _, _ = result_tuple
                logger.info(f"Successfully generated diagram for: {sentence}. Structure: {structure}, Final Diagram Cod: {enriched_diagram.cod}")
                logger.info(f"  Number of boxes in final diagram: {len(enriched_diagram.boxes)}")
                if core_stream: logger.info(f"  Core stream generated with {len(core_stream)} entries.")
                
                for box_idx, box in enumerate(enriched_diagram.boxes):
                    logger.info(f"  Box {box_idx}: Name='{box.name}' (Dom: {box.dom}, Cod: {box.cod})")
                    if hasattr(box, 'data') and box.data:
                        logger.info(f"    Data: {box.data.get('discocirc_enrichment_source', 'No enrichment source')}")
                        if 'extracted_root' in box.data: logger.info(f"      Extracted Root: {box.data['extracted_root']}")
                        if 'classical_features_of_surface_word' in box.data and box.data['classical_features_of_surface_word'] is not None:
                             logger.info(f"      Classical Feats (shape): {box.data['classical_features_of_surface_word'].shape}")
                try:
                    diag_path = os.path.join(output_dir_test, f"diagram_sent_{i+1}.png")
                    enriched_diagram.draw(path=diag_path, show=False, figsize=(max(10, len(enriched_diagram.boxes)), max(6, len(enriched_diagram.cod)))) # type: ignore
                    logger.info(f"Saved diagram image to {diag_path}")
                except Exception as e_draw: logger.error(f"Could not draw/save diagram for sentence {i+1}: {e_draw}")
            else: logger.error(f"Failed to generate diagram for: {sentence}")
        except Exception as e_test_run:
            logger.error(f"Unhandled exception during test run for sentence '{sentence}': {e_test_run}", exc_info=True)

