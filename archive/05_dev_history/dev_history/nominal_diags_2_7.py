def create_nominal_sentence_diagram_v2_7( # Renamed
    tokens: List[str], analyses_details: List[Dict[str, Any]], roles: Dict,
    word_core_types: List[Union[Ty, GrammarDiagram, None]],
    original_indices: List[int], # Indices of tokens included in word_core_types
    debug: bool = True, output_dir: Optional[str] = None, sentence_prefix: str = "diag_nominal",
    hint_predicate_original_idx: Optional[int] = None # ***** NEW PARAMETER *****
) -> Optional[GrammarDiagram]:
    """
    Creates a DisCoCat diagram for nominal sentences (Subject-Predicate).
    V2.7: Hybrid approach - attempts NP build for subject, falls back to simple box.
    """
    logger.info(f"Creating nominal diagram (V2.7 - Hybrid/Fallback) for: {' '.join(tokens)}")
    # --- Check essential types ---
    if ADJ_PRED_TYPE is None: # Assuming NounPred is created dynamically
         logger.error("Cannot create nominal diagram: ADJ_PRED_TYPE is not defined.")
         return None

    # --- Map data and create initial boxes ---
    analysis_map = {a['original_idx']: a for a in analyses_details}
    core_type_map = {orig_idx: word_core_types[i] for i, orig_idx in enumerate(original_indices)}
    arg_producer_boxes: Dict[int, Box] = {}
    functor_boxes: Dict[int, Box] = {}
    processed_indices: Set[int] = set()

    for orig_idx, core_entity in core_type_map.items():
        analysis = analysis_map.get(orig_idx)
        if not analysis or core_entity is None: continue
        box_name = f"{analysis.get('lemma', analysis.get('text','unk'))}_{orig_idx}"
        if isinstance(core_entity, Box):
            functor_boxes[orig_idx] = Box(box_name, core_entity.dom, core_entity.cod)
        elif isinstance(core_entity, Ty) and core_entity == N:
            arg_producer_boxes[orig_idx] = Box(box_name, Ty(), core_entity)

    # --- Identify Subject and Predicate ---
    subj_idx = roles.get('subject', roles.get('root'))
    predicate_idx: Optional[int] = None
    predicate_functor_box: Optional[Box] = None
    subj_diag: Optional[GrammarDiagram] = None

    if subj_idx is not None:
        for idx, functor_box in functor_boxes.items():
            if idx == subj_idx: continue
            if functor_box.dom == N and functor_box.cod == S:
                pred_analysis = analysis_map.get(idx)
                if pred_analysis:
                    is_root_and_not_subj = (idx == roles.get('root') and idx != subj_idx)
                    is_headed_by_subj = (pred_analysis.get('head') == subj_idx)
                    # Additional check: Ensure the predicate wasn't already used in the subject NP build
                    if (is_root_and_not_subj or is_headed_by_subj) and idx not in processed_indices:
                        predicate_idx = idx
                        predicate_functor_box = functor_box
                        logger.info(f"  Found nominal predicate functor: '{predicate_functor_box.name}' (idx {predicate_idx})")
                        break

    if hint_predicate_original_idx is not None:
        logger.debug(f"  Using hint_predicate_original_idx: {hint_predicate_original_idx}")
        hinted_functor = core_type_map.get(hint_predicate_original_idx)
        if isinstance(hinted_functor, Box) and hinted_functor.dom == N and hinted_functor.cod == S:
            predicate_idx = hint_predicate_original_idx
            predicate_functor_box = hinted_functor
            logger.info(f"  Using HINTED predicate functor: '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")
        else:
            logger.warning(f"  Hinted predicate idx {hint_predicate_original_idx} does not correspond to a valid N->S functor in core_type_map. Ignoring hint.")
                
    if predicate_functor_box is None and subj_diag and subj_diag.cod == N:
        logger.debug("  No pre-assigned predicate functor. Searching for potential unassigned predicate...")
        # Try to find an unassigned N, ADJ, or NUM that could be the predicate
        potential_predicate_idx = None
        potential_predicate_analysis = None
        # Iterate over original_indices to find an unassigned token that could be a predicate
        # This requires passing the full 'analyses_details' and 'original_indices_for_diagram'
        # to this function, or a map of original_idx to its analysis and core_type.

        # Simplified: Assume 'roles' now contains 'analyses_details_for_context'
        all_analyses_details = roles.get('analyses_details_for_context', [])

        for token_analysis in all_analyses_details:
            idx = token_analysis['original_idx']
            # Ensure idx is in core_type_map and not already processed or the subject itself
            if idx in core_type_map and idx not in processed_indices and idx != subj_idx:
                core_type_of_token = core_type_map.get(idx)
                if isinstance(core_type_of_token, Ty) and core_type_of_token == N and \
                token_analysis['upos'] in ["ADJ", "NOUN", "PROPN", "NUM", "X"]:
                    # Plausibility check: is it the sentence root OR a direct dependent of the subject?
                    # Or, for SUBJ_NO_VERB_OTHER, it might be the root if subject is not.
                    is_plausible_predicate = (token_analysis['deprel'] == 'root') or \
                                            (token_analysis.get('head') == subj_idx) or \
                                            (sentence_structure == "SUBJ_NO_VERB_OTHER" and idx == roles.get('root'))

                    if is_plausible_predicate:
                        potential_predicate_idx = idx
                        potential_predicate_analysis = token_analysis
                        logger.info(f"  Found potential dynamic predicate: '{token_analysis['text']}' (idx {idx}, POS {token_analysis['upos']}, DepRel {token_analysis['deprel']})")
                        break

        if potential_predicate_idx is not None and potential_predicate_analysis is not None:
            pred_lemma = potential_predicate_analysis.get('lemma', potential_predicate_analysis.get('text', 'unk'))
            pred_pos = potential_predicate_analysis['upos']

            box_name_prefix = ""
            if pred_pos == "ADJ" and ADJ_PRED_TYPE is not None:
                # Create a unique Box instance for this specific adjective predicate
                box_name_prefix = "AdjPred"
                predicate_functor_box = Box(f"{box_name_prefix}_{pred_lemma}_{potential_predicate_idx}", N, S)
                logger.info(f"  Dynamically created Adjective Predicate Functor for '{pred_lemma}' (idx {potential_predicate_idx})")
            elif pred_pos in ["NOUN", "PROPN", "NUM", "X"]:
                box_name_prefix = "NounPred"
                predicate_functor_box = Box(f"{box_name_prefix}_{pred_lemma}_{potential_predicate_idx}", N, S)
                logger.info(f"  Dynamically created Noun/X Predicate Functor for '{pred_lemma}' (idx {potential_predicate_idx})")

            if predicate_functor_box:
                predicate_idx = potential_predicate_idx
                functor_boxes[predicate_idx] = predicate_functor_box

                # Ensure it's marked as processed if used
                # processed_indices.add(predicate_idx) # Will be added later if composition succeeds
    # --- Attempt to Build Subject NP, with Fallback ---
    if subj_idx is not None:
        logger.debug(f"Attempting to build/get subject NP for explicit subj_idx: {subj_idx} ('{analysis_map.get(subj_idx, {}).get('text', 'N/A')}')")
        # If subj_idx points to a DET that was typed as N
        for idx_loop, functor_box_iter in functor_boxes.items():
            if idx_loop == subj_idx: continue
            if isinstance(functor_box_iter, Box) and functor_box_iter.dom == N and functor_box_iter.cod == S:
                pred_analysis_loop = analysis_map.get(idx_loop)
                if pred_analysis_loop:
                    is_root_and_not_subj = (idx_loop == roles.get('root') and idx_loop != subj_idx)
                    is_headed_by_subj = (pred_analysis_loop.get('head') == subj_idx)
                    # Allow if it's the root of the sentence OR if its head is the subject
                    if (is_root_and_not_subj or is_headed_by_subj) and idx_loop not in processed_indices:
                        predicate_idx = idx_loop # Assign to function-scope variable
                        predicate_functor_box = functor_box_iter # Assign to function-scope variable
                        logger.info(f"  Found pre-assigned nominal predicate functor: '{get_diagram_repr(predicate_functor_box)}' (idx {predicate_idx})")
                        break 
        if analysis_map.get(subj_idx, {}).get('upos') == 'DET' and core_type_map.get(subj_idx) == N:
            subj_diag = arg_producer_boxes.get(subj_idx)
            if subj_diag:
                logger.info(f"Using DET '{analysis_map[subj_idx]['text']}' (idx {subj_idx}) directly as N-type subject diagram.")
                if subj_idx not in processed_indices: processed_indices.add(subj_idx) # Mark as processed
            else:
                logger.error(f"DET subject {subj_idx} was typed N but not found in arg_producer_boxes. subj_diag remains None.")
                # subj_diag is already None or remains None

        # If not a DET subject or DET as N failed, try full NP build
        if subj_diag is None: 
            logger.debug(f"Attempting build_np_diagram_v4 for subject index {subj_idx}.")
            subj_diag = build_np_diagram_v4(
                subj_idx, analysis_map, roles, core_type_map,
                arg_producer_boxes, functor_boxes, processed_indices, debug
            )

        # Fallback to simple box if NP build failed or returned non-N
        if subj_diag is None or subj_diag.cod != N:
            logger.warning(f"Subject NP build for explicit index {subj_idx} failed or yielded non-N type ({get_diagram_repr(subj_diag)}). Falling back to simple arg_producer_box.")
            subj_diag = arg_producer_boxes.get(subj_idx) # This might re-assign subj_diag
            if subj_diag:
                logger.info(f"Using fallback arg_producer_box for subject {subj_idx}: {get_diagram_repr(subj_diag)}")
                if subj_idx not in processed_indices: processed_indices.add(subj_idx)
            else:
                logger.error(f"Fallback failed: Simple argument box also missing for explicit subject {subj_idx}. subj_diag is None.")
                # subj_diag remains None
    else: # If subj_idx was None initially (e.g. for SUBJ_NO_VERB_OTHER where Stanza doesn't pick a clear subject role)
        logger.warning("No explicit subject_idx from roles for nominal. Attempting to find a default subject from arg_producer_boxes.")
        for idx_candidate, arg_box_candidate in arg_producer_boxes.items():
            if idx_candidate not in processed_indices and arg_box_candidate.cod == N:
                logger.info(f"  Trying idx {idx_candidate} ('{analysis_map.get(idx_candidate,{}).get('text')}') as default subject.")
                # Attempt to build an NP around this candidate
                temp_processed = processed_indices.copy() # Use a copy for trial
                subj_diag_candidate = build_np_diagram_v4(
                    idx_candidate, analysis_map, roles, core_type_map,
                    arg_producer_boxes, functor_boxes, temp_processed, debug
                )
                if subj_diag_candidate and subj_diag_candidate.cod == N:
                    subj_diag = subj_diag_candidate
                    subj_idx = idx_candidate # CRITICAL: Update subj_idx
                    processed_indices.update(temp_processed) # Commit processed indices from successful NP build
                    logger.info(f"  Found and built default subject NP: '{get_diagram_repr(subj_diag)}' (orig_idx {subj_idx})")
                    break 
                elif arg_box_candidate.cod == N: # Fallback to simple box if NP build fails for this candidate
                    subj_diag = arg_box_candidate
                    subj_idx = idx_candidate # CRITICAL: Update subj_idx
                    processed_indices.add(idx_candidate)
                    logger.info(f"  Found default subject (simple box): '{get_diagram_repr(subj_diag)}' (orig_idx {subj_idx})")
                    break
        if subj_diag is None:
            logger.error("Could not find or build any suitable subject diagram for nominal sentence. subj_diag is None.")
            # subj_diag remains None
        elif subj_idx not in processed_indices: processed_indices.add(subj_idx)
        # If subj_diag is still None, error out

    # Check if required components are valid
    subj_diag_repr = get_diagram_repr(subj_diag) # Use helper for logging
    pred_func_repr = get_diagram_repr(predicate_functor_box)

    if subj_diag is None or subj_diag.cod != N: # Check if subj_diag is valid N
        logger.error(f"Cannot form nominal diagram: Subject diagram is invalid or missing ({subj_diag_repr}).")
        return None
    if predicate_idx is None or predicate_functor_box is None:
        logger.error(f"Cannot form nominal diagram: Predicate functor is missing (Predicate idx: {predicate_idx}, Box: {pred_func_repr}).")
        return None
    if predicate_functor_box.dom != N or predicate_functor_box.cod != S:
        logger.error(f"Cannot form nominal diagram: Predicate functor '{pred_func_repr}' has incorrect type {predicate_functor_box.dom} >> {predicate_functor_box.cod}.")
        return None
    logger.debug(f"PRE-DYNAMIC-PRED-SEARCH: subj_diag is {get_diagram_repr(subj_diag)}, type: {type(subj_diag)}, predicate_functor_box is {get_diagram_repr(predicate_functor_box)}")
    if predicate_functor_box is None:
        if subj_diag is not None and hasattr(subj_diag, 'cod') and subj_diag.cod == N:
            logger.debug("  No pre-assigned predicate functor and subject is valid. Searching for potential unassigned predicate...")
            potential_predicate_head_idx = None
            # Iterate over original_indices to find an unassigned token that could be a predicate HEAD
            for token_analysis in all_analyses_details: # Ensure all_analyses_details is available
                idx = token_analysis['original_idx']
                # Not already processed, not the subject, and a potential predicate POS
                if idx not in processed_indices and idx != subj_idx and \
                token_analysis['upos'] in ["ADJ", "NOUN", "PROPN", "NUM", "X"]:
                    # Plausibility: is it the sentence root OR a direct dependent of the subject?
                    # OR for SUBJ_NO_VERB_OTHER, it might be the root if subject is not.
                    is_plausible_head = (idx == roles.get('root') and token_analysis['upos'] != 'PUNCT') or \
                                        (subj_idx is not None and token_analysis.get('head') == subj_idx)
                    if is_plausible_predicate_head:
                        potential_predicate_head_idx = idx
                        logger.info(f"  Found potential dynamic predicate HEAD: '{token_analysis['text']}' (idx {idx}, POS {token_analysis['upos']})")
                        break

            if potential_predicate_head_idx is not None:
                pred_head_analysis = analysis_map.get(potential_predicate_head_idx)
                if pred_head_analysis:
                    pred_lemma = pred_head_analysis.get('lemma', pred_head_analysis.get('text', 'unk'))
                    pred_pos = pred_head_analysis['upos']
                    temp_functor_box = None
                    # Attempt to build an NP for the predicate if it's a Noun/Num/X.
                    # This NP will be the argument to the dynamically created N->S functor.
                    # The functor itself is associated with the *head* of this NP.
                    #new_functor_name = ""
                    #predicate_argument_diag = None
                    if pred_pos == "ADJ" and ADJ_PRED_TYPE is not None: # ADJ_PRED_TYPE is N->S
                        #new_functor_name = f"DynAdjPred_{pred_lemma}_{potential_predicate_head_idx}"
                        temp_functor_box = Box(f"DynAdjPred_{pred_lemma}_{potential_predicate_head_idx}", N, S)
                    elif pred_pos in ["NOUN", "PROPN", "NUM", "X"]:
                        #new_functor_name = f"DynNounPred_{pred_lemma}_{potential_predicate_head_idx}"
                        temp_functor_box = Box(f"DynNounPred_{pred_lemma}_{potential_predicate_head_idx}", N, S)

                    if temp_functor_box:
                        predicate_idx = potential_predicate_head_idx
                        predicate_functor_box = temp_functor_box # Assign to the main variable

                        # Update core_type_map and functor_boxes (as in previous version)
                        core_type_map[predicate_idx] = predicate_functor_box 
                        functor_boxes[predicate_idx] = predicate_functor_box

                        logger.info(f"  Dynamically assigned predicate functor '{get_diagram_repr(predicate_functor_box)}' to idx {predicate_idx} ('{pred_head_analysis['text']}').")

                        # Mark the predicate head and its nmod dependents as processed.
                        if predicate_idx not in processed_indices:
                            processed_indices.add(predicate_idx)
                            pred_dependents = roles.get('dependency_graph', {}).get(predicate_idx, [])
                            for dep_idx, dep_rel_str in pred_dependents:
                                # For S37, "أرجلٍ" (idx 3) is nmod of "أربعُ" (idx 2).
                                if dep_rel_str == 'nmod' and dep_idx not in processed_indices and core_type_map.get(dep_idx) == N:
                                    logger.info(f"    Marking nmod '{analysis_map.get(dep_idx,{}).get('text')}' (idx {dep_idx}) of dynamic predicate head as processed.")
                                    processed_indices.add(dep_idx)
                    else:
                        logger.warning(f"  Dynamic predicate functor creation failed for head {potential_predicate_head_idx}.")
                else:
                    logger.warning(f"  Dynamic predicate search: Analysis not found for potential head {potential_predicate_head_idx}.")
            else:
                logger.warning("  Dynamic predicate search: No suitable unassigned token found as predicate head after all strategies")

    # Mark predicate as processed if it's a valid index
    elif predicate_idx is not None: # Ensure predicate_idx was actually set
        processed_indices.add(predicate_idx)
    logger.debug(f"  Nominal components: Subj(idx {subj_idx}): {subj_diag_repr}, Pred(idx {predicate_idx}): {pred_func_repr}")

    subj_diag_repr = get_diagram_repr(subj_diag)
    pred_func_repr = get_diagram_repr(predicate_functor_box)
    logger.debug(f"FINAL CHECK for nominal composition: subj_idx={subj_idx}, predicate_idx={predicate_idx}, subj_diag={subj_diag_repr}, predicate_functor_box={pred_func_repr}")

    if subj_diag is None or not (hasattr(subj_diag, 'cod') and subj_diag.cod == N):
        logger.error(f"Cannot form nominal diagram: Subject diagram is invalid or missing ({subj_diag_repr}).")
        return None
    if predicate_idx is None or predicate_functor_box is None: # This is the failing line for S37
        logger.error(f"Cannot form nominal diagram: Predicate functor is missing (Final check: Predicate idx: {predicate_idx}, Box: {pred_func_repr}).")
        return None
    # Mark predicate as processed
    processed_indices.add(predicate_idx)
    logger.debug(f"  Nominal components after NP build/fallback: Subj(idx {subj_idx}): {subj_diag}, Pred(idx {predicate_idx}): {predicate_functor_box.name}")
    if subj_idx is None and subj_diag is not None: # If default subject was found
    # Try to get original_idx from the diagram if it's a simple Box
        if hasattr(subj_diag, 'name') and '_' in subj_diag.name:
            try: subj_idx = int(subj_diag.name.split('_')[-1])
            except ValueError: pass
        if subj_idx is None: logger.warning("Could not infer subj_idx from default subj_diag name.")
    # --- Compose Basic Predication ---
    final_diagram: Optional[GrammarDiagram] = None
    if subj_diag.cod == N and predicate_functor_box.dom == N and predicate_functor_box.cod == S:
        try:
            final_diagram = subj_diag >> predicate_functor_box
            logger.info(f"Nominal composition successful for '{sentence_prefix}'. Cod: {final_diagram.cod}")
        except Exception as e:
            logger.error(f"Nominal composition error for '{sentence_prefix}': {e}", exc_info=True)
            return None
    else:
        logger.error(f"Nominal type mismatch: Subj diag cod={subj_diag.cod}, Pred dom={predicate_functor_box.dom}")
        return None

    # --- Attach Sentence-Level Modifiers (Simplified) ---
    if final_diagram and final_diagram.cod == S:
         logger.debug("--- Attaching Sentence-Level Modifiers (Nominal - Simplified) ---")
         # Add logic similar to verbal function if needed, checking head == predicate_idx

    # --- Final Normalization and Return ---
    if final_diagram:
        try:
            normalized_diagram = final_diagram.normal_form()
            if normalized_diagram.cod == S:
                logger.info(f"Nominal diagram (V2.7 Hybrid) normalization successful. Final cod: {normalized_diagram.cod}")
                return normalized_diagram
            else:
                logger.warning(f"Nominal diagram (V2.7 Hybrid) normalized, but final cod is {normalized_diagram.cod}, not S. Discarding.")
                return None
        except NotImplementedError as e_norm_ni:
             logger.error(f"Normalization failed (NotImplementedError) for diagram: {final_diagram}. Error: {e_norm_ni}")
             return None
        except Exception as e_norm:
            logger.error(f"Nominal diagram (V2.7 Hybrid) normal_form failed: {e_norm}", exc_info=True)
            return None

    logger.warning(f"Could not form a complete nominal diagram (V2.7 Hybrid) ending in S for sentence '{sentence_prefix}'")
    return None


# ==================================
# Main Conversion Function (V2.7 - Uses Hybrid Diagram Functions)
# ==================================
def arabic_to_quantum_enhanced_v2_7( # Keeping name for consistency, but using V2.7.3 logic
    sentence: str,
    debug: bool = True,
    output_dir: Optional[str] = None,
    ansatz_choice: str = "IQP",
    # Pass ansatz parameters explicitly
    n_layers_iqp: int = 2,
    n_single_qubit_params_iqp: int = 3,
    n_layers_strong: int = 1,
    cnot_ranges: Optional[List[Tuple[int, int]]] = None,
    discard_qubits_spider: bool = True,
    **kwargs # Catch-all for other/unexpected keyword arguments
) -> Tuple[Optional[QuantumCircuit], Optional[GrammarDiagram], str, List[str], List[Dict[str,Any]], Dict]:
    """
    Processes an Arabic sentence, creates a DisCoCat diagram, and converts it to a Qiskit QuantumCircuit.
    V2.7.3: Added targeted fallback for 'OTHER' structure based on assigned functor types.
            Uses V2.2.2 type assignment logic.
    """
    global ARABIC_DISCOCIRC_PIPELINE_AVAILABLE # Allow modification of the global flag

    # --- MOVED IMPORT INSIDE FUNCTION ---
    generate_discocirc_ready_diagram_func = None
    try:
        from arabic_discocirc_pipeline import generate_discocirc_ready_diagram
        generate_discocirc_ready_diagram_func = generate_discocirc_ready_diagram
        ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = True # Set to True on successful import
    except ImportError as e_discocirc_runtime:
        # Log this error if it occurs at runtime, even if the top-level one was removed
        logger.error(f"Runtime import of generate_discocirc_ready_diagram failed: {e_discocirc_runtime}. Enriched diagrams unavailable.")
        ARABIC_DISCOCIRC_PIPELINE_AVAILABLE = False # Ensure it's False

    if kwargs:
        logger.warning(f"Function arabic_to_quantum_enhanced_v2_7 received UNEXPECTED keyword arguments: {kwargs}")

    # --- 1. Analyze Sentence ---
    # (Keep analysis step the same)
    logger.info(f"Analyzing sentence: '{sentence}'")
    try:
        tokens, analyses_details, structure, roles = analyze_arabic_sentence_with_morph(sentence, debug)
        if analyses_details: roles['analyses_details_for_context'] = analyses_details
        analysis_map_for_diagram_creation = {a['original_idx']: a for a in analyses_details}
        # Make it accessible via roles if sub-functions expect it there, or pass directly
        roles['analysis_map_for_diagram_creation'] = analysis_map_for_diagram_creation
        if structure == "ERROR" or not tokens:
            logger.warning(f"Sentence analysis failed or returned empty for: '{sentence}'")
            return None, None, structure, tokens or [], analyses_details or [], roles or {}
        logger.info(f"Analysis complete. Detected structure: {structure}. Roles: {roles}")
    except Exception as e_analyze_main:
        logger.error(f"Sentence analysis failed unexpectedly: {e_analyze_main}", exc_info=True)
        return None, None, "ERROR", [], [], {}

    analysis_map_for_diagram_creation = {a['original_idx']: a for a in analyses_details}
    # --- 2. Assign Core DisCoCat Types (Using V2.2.2 Logic) ---
    diagram: Optional[GrammarDiagram] = None
    used_enriched_diagram_path = False

    if ARABIC_DISCOCIRC_PIPELINE_AVAILABLE:
        logger.info(f"Attempting to generate feature-enriched DisCoCirc diagram for: '{sentence}'")
        try:
            enriched_diagram = generate_discocirc_ready_diagram(
                sentence_str=sentence,
                debug=debug
                #classical_feature_dim_for_enrichment=classical_feature_dim_for_discocirc_enrichment
            )
            if enriched_diagram is not None:
                logger.info(f"Successfully generated feature-enriched diagram via arabic_discocirc_pipeline. Boxes: {len(enriched_diagram.boxes)}")
                # Log data of first few boxes if debug
                if debug and enriched_diagram.boxes:
                    for i, b in enumerate(enriched_diagram.boxes[:3]): # Log first 3 boxes
                        logger.debug(f"  Enriched Box {i} ('{b.name}') data keys: {list(getattr(b, 'data', {}).keys()) if getattr(b, 'data', {}) else 'No data'}")

                diagram = enriched_diagram # Use this diagram
                used_enriched_diagram_path = True
            else:
                logger.warning("Feature-enriched diagram generation (arabic_discocirc_pipeline) returned None. Proceeding with fallback diagram logic.")
        except Exception as e_discocirc_call:
            logger.error(f"Error calling generate_discocirc_ready_diagram: {e_discocirc_call}", exc_info=True)
            logger.warning("Proceeding with fallback diagram logic due to error in enriched pipeline.")
    else:
        logger.info("arabic_discocirc_pipeline not available. Using fallback diagram logic directly.")

    # --- 3. Fallback or Original Diagram Creation Logic (IF enriched diagram was not created) ---
    if not used_enriched_diagram_path:
        logger.info("Using internal camel_test2.py logic for diagram generation (fallback path).")
        # This part is your existing logic from arabic_to_quantum_enhanced_v2_7 for type assignment and diagram creation
        word_core_types_list = []
        original_indices_for_diagram = []
        filtered_tokens_for_diagram = []
        core_type_map_for_fallback: Dict[int, Union[Ty, GrammarDiagram, None]] = {}

        logger.debug(f"--- Assigning Core Types (Fallback Path) V2.2.2 for: '{sentence}' ---")
        for i, analysis_entry in enumerate(analyses_details):
            current_core_type = assign_discocat_types_v2_2(
                analysis=analysis_entry,
                roles=roles,
                debug=debug
            )
            core_type_map_for_fallback[analysis_entry['original_idx']] = current_core_type
            if current_core_type is not None:
                word_core_types_list.append(current_core_type)
                original_indices_for_diagram.append(analysis_entry['original_idx'])
                filtered_tokens_for_diagram.append(analysis_entry['text'])
            else:
                logger.debug(f"  Fallback Token '{analysis_entry['text']}' (orig_idx {analysis_entry['original_idx']}) assigned None core type, excluding.")

        if not filtered_tokens_for_diagram:
            logger.error(f"Fallback Path: No valid tokens with core types remained for diagram construction: '{sentence}'")
            return None, None, structure, tokens, analyses_details, roles
        
        logger.debug(f"Fallback Path Filtered Tokens: {filtered_tokens_for_diagram}")
        diagram_creation_error = None
        try:
            logger.info(f"Creating DisCoCat diagram (V2.7.3 - Structure/OTHER Handling) for structure: {structure}...")
            safe_prefix = "".join(c if c.isalnum() else "_" for c in sentence.split()[0]) if sentence else "empty"

            # --- Decision Logic for Diagram Type ---
            attempted_diagram_type = None # Track which type we attempted

            # 1. Explicit Nominal Check
            if structure in ["NOMINAL", "SUBJ_NO_VERB_OTHER"]:
                logger.info(f"Attempting NOMINAL diagram creation based on structure '{structure}'.")
                attempted_diagram_type = "Nominal"
                diagram = create_nominal_sentence_diagram_v2_7(
                    filtered_tokens_for_diagram, analyses_details, roles,
                    word_core_types_list, original_indices_for_diagram, debug,
                    output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_nominal"
                )
                if diagram is None:
                    logger.warning(f"Nominal diagram creation failed for structure '{structure}'.")
                    # Optional: Fallback to verbal if nominal fails AND a verb exists
                    if roles.get('verb') is not None:
                        logger.info("Nominal failed, attempting verbal as fallback...")
                        attempted_diagram_type = "Verbal (Fallback)"
                        diagram = create_verbal_sentence_diagram_v3_7(
                            filtered_tokens_for_diagram, analyses_details, roles,
                            word_core_types_list, original_indices_for_diagram, debug,
                            output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_fallback"
                        )

            # 2. Explicit Verbal Check (or if Nominal wasn't applicable/failed and verb exists)
            elif structure not in ["ERROR", "OTHER"] or roles.get('verb') is not None:
                logger.info(f"Attempting VERBAL diagram creation based on structure '{structure}' or identified verb role.")
                attempted_diagram_type = "Verbal"
                diagram = create_verbal_sentence_diagram_v3_7(
                    filtered_tokens_for_diagram, analyses_details, roles,
                    word_core_types_list, original_indices_for_diagram, debug,
                    output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal"
                )
                if diagram is None:
                    logger.warning(f"Verbal diagram creation failed for structure '{structure}'.")

            # 3. **MODIFIED:** Fallback for 'OTHER' structure using assigned types
            elif structure == "OTHER":
                logger.info(f"Structure is 'OTHER'. Checking assigned types for fallback strategy.")
                has_verb_functor = any(isinstance(ct, Box) and ct.name in ["VerbIntransFunctor", "VerbTransFunctor"] for ct in core_type_map_for_fallback.values())
                has_pred_functor = any(isinstance(ct, Box) and (ct.name == "AdjPredFunctor" or (hasattr(ct, 'name') and ct.name.startswith("NounPred_"))) for ct in core_type_map_for_fallback.values())
                found_verb_functor = False
                found_predicate_functor = False
                if has_verb_functor:
                    logger.info("  'OTHER' structure has an assigned Verb Functor. Attempting VERBAL diagram.")
                    attempted_diagram_type = "Verbal (OTHER - Assigned Verb Functor)"
                    # Ensure roles['verb'] is set if it was None but a verb functor exists
                    if roles.get('verb') is None:
                        for idx, core_type in core_type_map_for_fallback.items():
                            if isinstance(core_type, Box) and core_type.name in ["VerbIntransFunctor", "VerbTransFunctor"]:
                                roles['verb'] = idx
                                logger.warning(f"  Updated roles['verb'] to {idx} for 'OTHER' verbal attempt.")
                                break
                    diagram = create_verbal_sentence_diagram_v3_7(
                        filtered_tokens_for_diagram, analyses_details, roles,
                        word_core_types_list, original_indices_for_diagram, debug,
                        output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_fallback"
                    )
                    found_verb_functor = True
                elif has_pred_functor:
                    logger.info("  'OTHER' structure has an assigned Predicate Functor. Attempting NOMINAL diagram.")
                    attempted_diagram_type = "Nominal (OTHER - Assigned Pred Functor)"
                    # Ensure roles['subject'] and roles['root'] (for predicate) are sensible
                    if roles.get('subject') is None or roles.get('root') is None: # Or predicate_idx logic
                        # Heuristics to find subject and predicate for nominal 'OTHER'
                        potential_subj_idx, potential_pred_idx = None, None
                        for idx, core_type in core_type_map_for_fallback.items():
                            analysis = analysis_map_for_diagram_creation.get(idx)
                            if isinstance(core_type, Box) and (core_type.name == "AdjPredFunctor" or (hasattr(core_type, 'name') and core_type.name.startswith("NounPred_"))):
                                potential_pred_idx = idx
                            elif core_type == N and analysis and analysis.get('deprel') == 'nsubj': # A noun that is a subject
                                potential_subj_idx = idx

                        if potential_pred_idx is not None and roles.get('root') != potential_pred_idx : roles['root'] = potential_pred_idx # Predicate is often root
                        if potential_subj_idx is not None and roles.get('subject') is None: roles['subject'] = potential_subj_idx
                        logger.warning(f"  Updated roles for 'OTHER' nominal attempt: subject={roles.get('subject')}, root/predicate_anchor={roles.get('root')}")

                    diagram = create_nominal_sentence_diagram_v2_7(
                                        filtered_tokens_for_diagram, 
                                        analyses_details, 
                                        roles, 
                                        temp_word_core_types_list, 
                                        original_indices_for_diagram, debug
                                    )

                else:
                    # NEW: If 'OTHER' and NO functor, try to dynamically make one (e.g. root noun/adj as predicate)
                    logger.warning("  'OTHER' structure with NO pre-assigned functor. Attempting dynamic predicate identification.")
                    root_idx = roles.get('root')
                    if root_idx is not None and root_idx in core_type_map_for_fallback:
                        root_analysis = analysis_map_for_diagram_creation.get(root_idx)
                        root_core_type = core_type_map_for_fallback.get(root_idx)
                        # Try to find a subject for this root
                        current_subject_idx = roles.get('subject')
                        if current_subject_idx is None:
                            dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                            for dep_idx, d_rel in dependents_of_root:
                                dep_analysis = analysis_map_for_diagram_creation.get(dep_idx)
                                if d_rel == 'nsubj' and analyses_details and dep_idx < len(analyses_details) and analyses_details[dep_idx]['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                    roles['subject'] = dep_idx; current_subject_idx = dep_idx; break

                        if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["ADJ", "NOUN", "X", "PROPN", "NUM"]:
                            logger.info(f"  Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}) in 'OTHER' structure.")
                            # Temporarily update core_type_map for this attempt
                            temp_core_type_map = core_type_map.copy() # or word_core_types_list if that's what's passed
                            # This requires word_core_types_list to be a dict or to find the right index
                            original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                            list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                            if list_idx_of_root is not None:
                                # Create a new word_core_types_list for this attempt
                                temp_word_core_types_list = list(word_core_types_list) # Make a mutable copy
                                if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE:
                                    temp_word_core_types_list[list_idx_of_root] = ADJ_PRED_TYPE
                                else:
                                    temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicNounPred_{root_analysis['lemma']}_{root_idx}", N, S)

                                logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate at root {root_idx}.")
                                attempted_diagram_type = "Nominal (OTHER - Dynamic Root Predicate)"
                                diagram = create_nominal_sentence_diagram_v2_7(
                                    filtered_tokens_for_diagram, analyses_details, roles,
                                    temp_word_core_types_list, # Pass the modified list
                                    original_indices_for_diagram, debug, # ...
                                )
                            else:
                                logger.warning(f"  Root index {root_idx} not in original_indices_for_diagram, cannot dynamically assign predicate for 'OTHER'.")

                    if diagram is None: # If dynamic predicate also failed
                        logger.warning("  'OTHER' structure with NO pre-assigned functor. Attempting dynamic predicate identification.")
                        root_idx = roles.get('root')
                        if root_idx is not None and root_idx in core_type_map_for_fallback:
                            # Use the analysis_map_for_diagram_creation defined earlier
                            root_analysis = analysis_map_for_diagram_creation.get(root_idx) 

                            current_subject_idx = roles.get('subject')
                            if current_subject_idx is None and root_analysis: # Check root_analysis exists
                                dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                                for dep_idx, d_rel in dependents_of_root:
                                    # Use analysis_map_for_diagram_creation to get dep_analysis
                                    dep_analysis = analysis_map_for_diagram_creation.get(dep_idx)
                                    if d_rel == 'nsubj' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                        roles['subject'] = dep_idx; current_subject_idx = dep_idx
                                        logger.info(f"  Dynamic 'OTHER': Set subject to {dep_idx} ('{dep_analysis['text']}') for root predicate candidate '{root_analysis['text']}'.")
                                        break

                            if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["ADJ", "NOUN", "X", "PROPN", "NUM"]:
                                logger.info(f"  Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}) in 'OTHER' structure.")
                                # ... (rest of dynamic predicate assignment to temp_word_core_types_list) ...
                                # Ensure temp_word_core_types_list is correctly created and modified
                                original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                                list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                                if list_idx_of_root is not None and list_idx_of_root < len(word_core_types_list):
                                    temp_word_core_types_list = list(word_core_types_list) 
                                    if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE:
                                        temp_word_core_types_list[list_idx_of_root] = ADJ_PRED_TYPE
                                    else:
                                        temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicNounPred_{root_analysis.get('lemma', root_analysis.get('text','unk'))}_{root_idx}", N, S)

                                    logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate at root {root_idx}.")
                                    # Pass analyses_details so create_nominal_sentence_diagram_v2_7 can build its own analysis_map
                                    diagram = create_nominal_sentence_diagram_v2_7(
                                        filtered_tokens_for_diagram, 
                                        analyses_details, 
                                        roles, 
                                        temp_word_core_types_list, 
                                        original_indices_for_diagram, debug
                                    )

                # Check for any assigned verb functor
                for idx, core_type in core_type_map_for_fallback.items():
                    if isinstance(core_type, Box) and core_type.name in ["VerbIntransFunctor", "VerbTransFunctor"]:
                        logger.info(f"  Found assigned Verb Functor ('{core_type.name}' for token idx {idx}) in 'OTHER' structure. Attempting VERBAL diagram.")
                        attempted_diagram_type = "Verbal (OTHER Fallback - Assigned Type)"
                        # If main verb role wasn't set, set it to the first one found
                        if roles.get('verb') is None:
                            logger.warning(f"  Updating roles['verb'] heuristically to {idx} for diagram creation.")
                            roles['verb'] = idx
                            # Optional: Try to find subj/obj based on this verb? Risky.
                        diagram = create_verbal_sentence_diagram_v3_7(
                            filtered_tokens_for_diagram, analyses_details, roles,
                            word_core_types_list, original_indices_for_diagram, debug,
                            output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_verbal_other"
                        )
                        found_verb_functor = True
                        break

                # If no verb functor, check for any assigned predicate functor
                if not found_verb_functor:
                    for idx, core_type in core_type_map_for_fallback.items():
                        if isinstance(core_type, Box) and (core_type.name == "AdjPredFunctor" or core_type.name.startswith("NounPred_")):
                            logger.info(f"  Found assigned Predicate Functor ('{core_type.name}' for token idx {idx}) in 'OTHER' structure. Attempting NOMINAL diagram.")
                            attempted_diagram_type = "Nominal (OTHER Fallback - Assigned Type)"
                            # Need to ensure subject is identified for nominal path
                            if roles.get('subject') is None:
                                # Try to find nsubj dependent of this predicate
                                dep_graph = roles.get('dependency_graph', {})
                                potential_subj_idx = None
                                if isinstance(dep_graph, dict):
                                    dependents = dep_graph.get(idx, [])
                                    for dep_idx, rel in dependents:
                                            if rel == 'nsubj':
                                                potential_subj_idx = dep_idx
                                                break
                                if potential_subj_idx is not None:
                                    logger.warning(f"  Updating roles['subject'] heuristically to {potential_subj_idx} for nominal diagram creation.")
                                    roles['subject'] = potential_subj_idx
                                else:
                                    logger.warning(f"  Found predicate functor but no subject identified for 'OTHER' structure. Nominal diagram likely to fail.")
                            # Update root if necessary? Maybe not needed if predicate is found.
                            # if roles.get('root') != idx : roles['root'] = idx

                            diagram = create_nominal_sentence_diagram_v2_7(
                                filtered_tokens_for_diagram, analyses_details, roles,
                                word_core_types_list, original_indices_for_diagram, debug,
                                output_dir=output_dir, sentence_prefix=f"sent_{safe_prefix}_nominal_other"
                            )
                            found_predicate_functor = True
                            break

                # If neither found for 'OTHER'
                if not found_verb_functor and not found_predicate_functor:
                    # logger.warning(f"  Structure is 'OTHER', but no assigned Verb or Predicate functor found. Skipping diagram creation.")
                    attempted_diagram_type = "Skipped (OTHER - No Functor)"
                    logger.warning("  'OTHER' structure with NO pre-assigned and USED functor. Attempting dynamic predicate identification.")
                    root_idx = roles.get('root')
                    analysis_map_from_roles = roles.get('analysis_map_for_diagram_creation', {}) # Get the map from roles
                    potential_pred_idx = None
                    potential_subj_idx = None
                    if root_idx is not None:
                        root_analysis = analysis_map_from_roles.get(root_idx)
                        if root_analysis and root_analysis['upos'] in ["ADJ", "X", "NUM"]: # Potential predicate POS
                            dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                            for dep_idx, d_rel in dependents_of_root:
                                if d_rel == 'nsubj':
                                    potential_subj_idx = dep_idx
                                    potential_pred_idx = root_idx
                                    logger.info(f"  Dynamic 'OTHER' (Attempt 1): Root '{root_analysis['text']}' (ADJ/X/NUM) is predicate, its nsubj '{analysis_map_from_roles.get(dep_idx,{}).get('text')}' is subject.")
                                    break

                    # Attempt 2: If root is NOUN/X, look for an ADJ/X dependent that could be the predicate.
                    # Example: "عينُ الطفلِ زرقاءُ" (S51). Root="عينُ"(N/X). "زرقاءُ"(X) is nmod of "الطفلِ"(obj of "عينُ").
                    # This requires finding "زرقاءُ" as predicate and "عينُ الطفلِ" as subject.
                    if potential_pred_idx is None and root_analysis and root_analysis['upos'] in ["NOUN", "X"]:
                        # Iterate through all tokens to find a potential adjectival/nominal predicate
                        for token_idx, token_analysis_iter in analysis_map_from_roles.items():
                            if token_idx == root_idx: continue # Don't pick root as its own predicate here
                            if token_analysis_iter['upos'] in ["ADJ", "X", "NUM"]: # Potential predicate
                                # Check if this token is related to the root or its main arguments
                                # For S51: "زرقاءُ" (idx 2) head is "الطفلِ" (idx 1), head of "الطفلِ" is "عينُ" (idx 0, root)
                                head_of_candidate_pred = token_analysis_iter.get('head')
                                if head_of_candidate_pred is not None:
                                    head_of_head = analysis_map_from_roles.get(head_of_candidate_pred, {}).get('head')
                                    if head_of_candidate_pred == root_idx or head_of_head == root_idx or \
                                    (roles.get('object') is not None and head_of_candidate_pred == roles.get('object')): # Predicate modifies object of root

                                        # If this is our predicate, the root (or an NP around it) is the subject
                                        potential_pred_idx = token_idx
                                        potential_subj_idx = root_idx # Assume root is subject, or head of subject NP
                                        logger.info(f"  Dynamic 'OTHER' (Attempt 2): Found predicate '{token_analysis_iter['text']}' (idx {token_idx}) for subject (around) root '{root_analysis['text']}'.")
                                        break

                    # If we found a potential subject and predicate for nominal construction:
                    if potential_pred_idx is not None and potential_subj_idx is not None:
                        roles['subject'] = potential_subj_idx # Update roles
                        # The dynamic predicate functor will be assigned to potential_pred_idx
                        pred_cand_analysis = analysis_map_from_roles.get(potential_pred_idx)
                        if pred_cand_analysis:
                            logger.info(f"  Dynamically assigning N->S functor to '{pred_cand_analysis['text']}' (idx {potential_pred_idx}) in 'OTHER' structure.")
                            original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                            list_idx_of_pred_cand = original_idx_to_list_idx.get(potential_pred_idx)

                            if list_idx_of_pred_cand is not None and list_idx_of_pred_cand < len(word_core_types_list):
                                temp_word_core_types_list = list(word_core_types_list) 
                                new_functor_name_base = pred_cand_analysis.get('lemma', pred_cand_analysis.get('text','unk'))
                                if pred_cand_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE: # ADJ_PRED_TYPE is N->S Box
                                    temp_word_core_types_list[list_idx_of_pred_cand] = Box(f"DynamicAdjPred_{new_functor_name_base}_{potential_pred_idx}", N, S)
                                else: # NOUN, X, NUM as predicate
                                    temp_word_core_types_list[list_idx_of_pred_cand] = Box(f"DynamicNounPred_{new_functor_name_base}_{potential_pred_idx}", N, S)

                                logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate '{get_diagram_repr(temp_word_core_types_list[list_idx_of_pred_cand])}' and subject idx {potential_subj_idx}.")
                                attempted_diagram_type = "Nominal (OTHER - Dynamic Subj/Pred)"
                                diagram = create_nominal_sentence_diagram_v2_7(
                                    filtered_tokens_for_diagram, analyses_details, roles, 
                                    temp_word_core_types_list, original_indices_for_diagram, debug,
                                    hint_predicate_original_idx=potential_pred_idx # ***** NEW HINT *****
                                )
                            # ...
                    """ if root_idx is not None and root_idx in core_type_map_for_fallback:
                        root_analysis = analysis_map_from_roles.get(root_idx) 
                        current_subject_idx = roles.get('subject')
                        if current_subject_idx is None and root_analysis:
                            logger.debug(f"  Dynamic 'OTHER': Root '{root_analysis['text']}' is predicate candidate. Searching for its subject.")
                            # Strategy 1: Look for 'nsubj' dependent of the root.
                            dependents_of_root = roles.get('dependency_graph', {}).get(root_idx, [])
                            for dep_idx, d_rel in dependents_of_root:
                                dep_analysis = analysis_map_from_roles.get(dep_idx)
                                if d_rel == 'nsubj' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                    roles['subject'] = dep_idx; current_subject_idx = dep_idx
                                    logger.info(f"    Found subject (nsubj of root): '{dep_analysis['text']}' (idx {dep_idx}) for predicate '{root_analysis['text']}'.")
                                    break
                            # Strategy 2: If root is ADJ/X and has no nsubj, look for a preceding N/PRON/DET that might be the subject.
                            if current_subject_idx is None and root_analysis['upos'] in ["ADJ", "X"] :
                                for i in range(root_idx - 1, -1, -1): # Look backwards
                                    prev_token_analysis = analysis_map_from_roles.get(i)
                                    if prev_token_analysis and prev_token_analysis['upos'] in ["NOUN", "PRON", "DET", "X"] and prev_token_analysis.get('head') == root_idx: # Check if it's a dependent or just preceding
                                        roles['subject'] = i; current_subject_idx = i
                                        logger.info(f"    Found potential preceding subject: '{prev_token_analysis['text']}' (idx {i}) for predicate '{root_analysis['text']}'.")
                                        break
                            # Strategy 3: For S81 "اللاعبانِ ماهرانِ .", "ماهرانِ" is root, "اللاعبانِ" is its nmod. This is unusual.
                            # If root is X/ADJ and subject is still None, check nmod as last resort.
                            if current_subject_idx is None and root_analysis['upos'] in ["ADJ", "X"]:
                                for dep_idx, d_rel in dependents_of_root:
                                    dep_analysis = analysis_map_from_roles.get(dep_idx)
                                    if d_rel == 'nmod' and dep_analysis and dep_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X", "DET"]:
                                        roles['subject'] = dep_idx; current_subject_idx = dep_idx
                                        logger.info(f"    Found subject (nmod of root): '{dep_analysis['text']}' (idx {dep_idx}) for predicate '{root_analysis['text']}'.")
                                        break
                        if current_subject_idx is not None and root_analysis and root_analysis['upos'] in ["ADJ", "NOUN", "X", "PROPN", "NUM"]:
                            logger.info(f"  Dynamically assigning N->S functor to root '{root_analysis['text']}' (idx {root_idx}) in 'OTHER' structure.")
                            original_idx_to_list_idx = {orig_idx: i for i, orig_idx in enumerate(original_indices_for_diagram)}
                            list_idx_of_root = original_idx_to_list_idx.get(root_idx)

                            if list_idx_of_root is not None and list_idx_of_root < len(word_core_types_list):
                                temp_word_core_types_list = list(word_core_types_list) 
                                new_functor_name_base = root_analysis.get('lemma', root_analysis.get('text','unk'))
                                if root_analysis['upos'] == "ADJ" and ADJ_PRED_TYPE:
                                    temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicAdjPred_{new_functor_name_base}_{root_idx}", N, S)
                                else:
                                    temp_word_core_types_list[list_idx_of_root] = Box(f"DynamicNounPred_{new_functor_name_base}_{root_idx}", N, S)

                                logger.info(f"  Attempting NOMINAL diagram for 'OTHER' with dynamic predicate at root {root_idx} ('{get_diagram_repr(temp_word_core_types_list[list_idx_of_root])}').")
                                attempted_diagram_type = "Nominal (OTHER - Dynamic Root Predicate)"
                                diagram = create_nominal_sentence_diagram_v2_7(
                                    filtered_tokens_for_diagram, analyses_details, roles, 
                                    temp_word_core_types_list, original_indices_for_diagram, debug
                                )
                            else:
                                logger.warning(f"  Root index {root_idx} not in original_indices_for_diagram or word_core_types_list, cannot dynamically assign predicate for 'OTHER'.")
    """
                    if diagram is None: # If dynamic predicate also failed
                        logger.warning(f"  'OTHER' structure: All fallbacks failed. Skipping diagram creation.")
                        attempted_diagram_type = "Skipped (OTHER - All Fallbacks Failed)"


            # 4. Handle ERROR or unhandled cases
            else:
                logger.warning(f"Structure is '{structure}'. Cannot determine diagram type. Skipping diagram creation.")
                attempted_diagram_type = f"Skipped ({structure})"


            # --- Logging Outcome ---
            if diagram is None:
                # Log error only if we actually attempted a diagram type
                if attempted_diagram_type and not attempted_diagram_type.startswith("Skipped"):
                    logger.error(f"Diagram creation ({attempted_diagram_type}) returned None for sentence '{sentence}'.")
            else:
                logger.info(f"Diagram ({attempted_diagram_type}) created successfully for '{sentence}'. Final Cod: {diagram.cod}")

        except Exception as e_diagram:
            logger.error(f"Exception during diagram creation phase for '{sentence}': {e_diagram}", exc_info=True)
            diagram_creation_error = str(e_diagram)

    # --- Steps 4 & 5 (Circuit Conversion & Return) ---
    # (Keep the rest of the function the same as in camel_test2.py v3.3)
    # ... (circuit conversion logic using selected_ansatz) ...

    # --- 4. Convert Diagram to Quantum Circuit ---
    circuit: Optional[QuantumCircuit] = None
    # Ensure diagram exists before proceeding
    if diagram is None:
        logger.error("Diagram is None after creation attempt, cannot proceed to circuit conversion.")
        # Return existing info, circuit and diagram are None
        return None, None, structure, tokens, analyses_details, roles

    try:
        logger.info(f"Converting diagram to quantum circuit using ansatz: {ansatz_choice}")
        # Define object map (adjust if you use different atomic types like P, ADJ etc.)
        ob_map = {N: 1, S: 1} # Basic map, assumes N=1 qubit, S=1 qubit

        selected_ansatz = None
        # --- Ansatz Selection (Ensure names match Lambeq's classes) ---
        if ansatz_choice.upper() == "IQP":
            n_single_qubit_params_calculated = n_layers_iqp * 2 
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=n_layers_iqp, n_single_qubit_params=n_single_qubit_params_iqp)
            logger.debug(f"Created IQPAnsatz with {n_layers_iqp} layers and {n_single_qubit_params_calculated} params per qubit.")

        elif ansatz_choice.upper() == "STRONGLY_ENTANGLING":
             # Determine number of qubits needed for the StrongAnsatz
             # This depends on the input type of the diagram (diagram.dom)
             # If diagram.dom is Ty(), it might represent 0 qubits, handle appropriately.
             num_qubits_required = len(diagram.dom) if diagram.dom else 1 # Default to 1 if dom is Ty()
             logger.info(f"Diagram domain requires {num_qubits_required} qubits for StronglyEntanglingAnsatz.")
             # Note: Lambeq's StronglyEntanglingAnsatz might take ob_map directly now. Check documentation.
             # If it requires num_qubits explicitly:
             # selected_ansatz = StronglyEntanglingAnsatz(num_qubits=num_qubits_required, n_layers=n_layers_strong, ranges=cnot_ranges)
             # If it takes ob_map:
             selected_ansatz = StronglyEntanglingAnsatz(ob_map=ob_map, n_layers=n_layers_strong, ranges=cnot_ranges)
             logger.info(f"Using StronglyEntanglingAnsatz (layers={n_layers_strong}, qubits based on ob_map)")
        elif ansatz_choice.upper() == "SPIDER":
            nounS = AtomicType.NOUN
            sentS = AtomicType.SENTENCE
            ob_map = {nounS: 1, sentS: 1}
            selected_ansatz = SpiderAnsatz(ob_map=ob_map)
            logger.info(f"Using SpiderAnsatz (discard_qubits={discard_qubits_spider})")
        else:
            logger.warning(f"Unknown ansatz_choice: '{ansatz_choice}'. Defaulting to IQPAnsatz.")
            selected_ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3) # Default IQP

        if selected_ansatz is None:
            raise ValueError("Ansatz object could not be created.")
        logger.debug(f"Attempting to apply ansatz. Grammatical diagram for sentence '{sentence}':")
        logger.debug(f"Diagram: {diagram}") # This might be too verbose
        logger.debug(f"Boxes in diagram ({len(diagram.boxes)}):")
        for i, box in enumerate(diagram.boxes):
            logger.debug(f"  Box {i}: name='{box.name}', dom={box.dom}, cod={box.cod}, data={getattr(box, 'data', None)}") # Log box.data if it exists

        logger.info(f"--- SPIDER DEBUG for sentence: '{sentence}' ---")
        logger.info(f"Grammatical diagram: {diagram}")
        logger.info("Boxes in grammatical diagram:")
        for i, box in enumerate(diagram.boxes):
            logger.info(f"  GRMR_BOX {i}: name='{box.name}', dom={box.dom}, cod={box.cod}")

        if "spider" in str(type(selected_ansatz)).lower(): # Check if it's SpiderAnsatz
            logger.info("Applying SPIDER ansatz functor to each grammatical box individually:")
            problematic_box_found = False
            for i, box in enumerate(diagram.boxes):
                try:
                    # selected_ansatz is the SpiderAnsatz instance
                    quantum_box = selected_ansatz(box) # Apply functor to individual box
                    logger.info(f"  QUANTUM_BOX from GRMR_BOX {i} ('{box.name}'): q_dom={quantum_box.dom}, q_cod={quantum_box.cod}")
                    if quantum_box.dom == Ty():
                        logger.error(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        logger.error(f"    !!!! PROBLEM: GRMR_BOX {i} ('{box.name}', dom={box.dom}, cod={cod.cod})")
                        logger.error(f"    !!!!   resulted in QUANTUM_BOX with dom=Ty() and cod={quantum_box.cod} via SPIDER !!!!")
                        logger.error(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        problematic_box_found = True
                except Exception as e_box:
                    logger.error(f"  ERROR applying SPIDER to GRMR_BOX {i} ('{box.name}', dom={box.dom}, cod={box.cod}): {e_box}", exc_info=False)
                    problematic_box_found = True

            if problematic_box_found:
                logger.error("SPIDER Debug: Problematic box found. See details above. Full diagram conversion will likely fail.")
                # Consider returning a failure state here for the single sentence test
                # return None, None, None, None # (match your function's error return)

        # Now, let the original call happen, which will likely raise the error
        # if problematic_box_found was true
        try:
            quantum_diagram = selected_ansatz(diagram)
        except TypeError as e_diag:
            logger.error(f"SPIDER TypeError on full diagram conversion (as expected if box issues found): {e_diag}")
            # If not already done by the individual box check, re-log the failing diagram's boxes
            if not problematic_box_found: # Log only if individual check didn't catch it
                logger.error("Original grammatical diagram that failed on full conversion:")
                for i, box in enumerate(diagram.boxes):
                    logger.error(f"  GRMR_BOX {i}: name='{box.name}', dom={box.dom}, cod={box.cod}")
            raise

        quantum_diagram = selected_ansatz(diagram)
        logger.info(f"Applied {ansatz_choice} ansatz to the diagram.")

        # --- Convert to Qiskit Circuit ---
        if PYTKET_QISKIT_AVAILABLE:
            logger.debug("Attempting conversion via Pytket...")
            tket_circ = quantum_diagram.to_tk()
            circuit = tk_to_qiskit(tket_circ)
            logger.info("Circuit conversion via Tket successful.")
        elif hasattr(quantum_diagram, 'to_qiskit'):
             logger.debug("Attempting direct conversion using Lambeq's to_qiskit...")
             circuit = quantum_diagram.to_qiskit()
             logger.info("Direct circuit conversion to Qiskit successful.")
        else:
             logger.error("No available method (Tket or direct) to convert Lambeq diagram to Qiskit circuit.")
             circuit = None # Ensure circuit is None if conversion fails

    except NotImplementedError as e_nf_main:
        # This might occur if normal_form was called implicitly or if ansatz fails
        logger.error(f"Diagram normalization or ansatz application failed: {e_nf_main}", exc_info=True)
        # Return the diagram that was created, but circuit is None
        return None, diagram, structure, tokens, analyses_details, roles
    except Exception as e_circuit_outer:
        logger.error(f"Exception during circuit conversion: {e_circuit_outer}", exc_info=True)
        # Return the diagram, but circuit is None
        return None, diagram, structure, tokens, analyses_details, roles

    if circuit is None:
        logger.error("Circuit conversion resulted in None.")
        # Return diagram, circuit is None
        return None, diagram, structure, tokens, analyses_details, roles

    # --- 5. Return Results ---
    logger.info(f"Successfully processed sentence '{sentence}' into circuit.")
    return circuit, diagram, structure, tokens, analyses_details, roles

def build_np_diagram_v4( # Renamed
    head_noun_idx: int,
    analysis_map: Dict[int, Dict[str, Any]],
    roles: Dict[str, Any],
    core_type_map: Dict[int, Union[Ty, GrammarDiagram, None]],
    arg_producer_boxes: Dict[int, Box],
    functor_boxes: Dict[int, Box],
    processed_indices: Set[int], # Keep track of indices used
    debug: bool = True
) -> Optional[GrammarDiagram]:
    """
    Recursively builds a diagram for a Noun Phrase centered around head_noun_idx.
    Handles DET, ADJ (amod), and PP (nmod) modifiers.
    V4: Allows 'X'/'DET' heads if Subj/Obj role. Simplified PP object handling on recursion fail.
    Returns a diagram of type Ty() -> N.
    """
    if head_noun_idx in processed_indices:
        logger.warning(f"NP Head Noun {head_noun_idx} already processed. Skipping NP build.")
        return arg_producer_boxes.get(head_noun_idx)

    head_analysis = analysis_map.get(head_noun_idx)
    if not head_analysis:
        logger.error(f"Cannot build NP: Analysis not found for head index {head_noun_idx}.")
        return None

    # Check POS, allowing 'X' or 'DET' if the token is identified as Subject or Object
    allowed_pos = ["NOUN", "PROPN", "PRON"]
    is_subj_or_obj = (head_noun_idx == roles.get('subject') or head_noun_idx == roles.get('object'))
    head_pos = head_analysis['upos']
    head_deprel = head_analysis['deprel']

    can_build = head_pos in allowed_pos or \
                (head_pos == 'X' and is_subj_or_obj) or \
                (head_pos == 'DET' and is_subj_or_obj and head_deprel == 'nsubj')

    if not can_build:
        logger.error(f"Cannot build NP: Head index {head_noun_idx} ('{head_analysis['text']}') has invalid POS/Role combination ('{head_pos}', Subj/Obj={is_subj_or_obj}, DepRel='{head_deprel}') for NP head.")
        return None

    # Start with the base noun box (Ty() -> N)
    np_diagram = arg_producer_boxes.get(head_noun_idx)
    if np_diagram is None:
        if core_type_map.get(head_noun_idx) == N:
             box_name = f"{head_analysis.get('lemma', head_analysis.get('text','unk'))}_{head_noun_idx}"
             np_diagram = Box(box_name, Ty(), N)
             arg_producer_boxes[head_noun_idx] = np_diagram
             logger.info(f"Created missing Argument Producer Box for '{box_name}': Ty() -> N")
        else:
             logger.error(f"Cannot build NP: Argument producer box not found or incorrect type for head noun {head_noun_idx}.")
             return None

    logger.info(f"Building NP diagram V4 for head noun: '{head_analysis['text']}' (idx {head_noun_idx})")
    logger.debug(f"  Initial NP diagram: {np_diagram.name} ({np_diagram.dom} -> {np_diagram.cod})")
    processed_indices.add(head_noun_idx)

    # Find dependents
    dep_graph = roles.get('dependency_graph', {})
    dependents = dep_graph.get(head_noun_idx, []) if isinstance(dep_graph, dict) else []
    logger.debug(f"  Dependents of {head_noun_idx}: {dependents}")

    # --- Apply Modifiers ---
    modifiers_to_apply = []
    for dep_idx, dep_rel in dependents:
        if dep_idx in processed_indices: continue

        modifier_box = functor_boxes.get(dep_idx)
        modifier_analysis = analysis_map.get(dep_idx)

        if modifier_box and modifier_analysis:
            # Determiner
            if dep_rel == 'det' and head_pos != 'DET' and modifier_box.dom == N and modifier_box.cod == N:
                modifiers_to_apply.append({'idx': dep_idx, 'box': modifier_box, 'rel': 'det', 'order': 0})
            # Adjective Modifier
            elif dep_rel == 'amod' and modifier_box.dom == N and modifier_box.cod == N:
                 modifiers_to_apply.append({'idx': dep_idx, 'box': modifier_box, 'rel': 'amod', 'order': 1})
            # Prepositional Phrase Modifier
            elif dep_rel == 'case' and modifier_analysis['upos'] == 'ADP' and PREP_FUNCTOR_TYPE and N_MOD_BY_N:
                 prep_box = modifier_box
                 prep_dependents = dep_graph.get(dep_idx, []) if isinstance(dep_graph, dict) else []
                 pp_obj_idx = None
                 for pp_dep_idx, pp_dep_rel in prep_dependents:
                      pp_obj_analysis = analysis_map.get(pp_dep_idx)
                      if pp_obj_analysis and pp_dep_idx not in processed_indices and pp_obj_analysis['upos'] in ["NOUN", "PROPN", "PRON", "X"]:
                           pp_obj_idx = pp_dep_idx
                           break
                 if pp_obj_idx is not None:
                      pp_obj_np_diagram = None
                      # Attempt recursive build for PP object
                      logger.debug(f"  Attempting recursive NP build for PP object {pp_obj_idx} ('{analysis_map[pp_obj_idx]['text']}')")
                      pp_obj_np_diagram = build_np_diagram_v4( # Recursive call
                          pp_obj_idx, analysis_map, roles, core_type_map,
                          arg_producer_boxes, functor_boxes, processed_indices, debug
                      )

                      # Fallback if recursive build fails
                      if pp_obj_np_diagram is None or pp_obj_np_diagram.cod != N:
                          logger.warning(f"  Recursive NP build failed for PP object {pp_obj_idx}. Falling back to simple box for PP composition.")
                          pp_obj_np_diagram = arg_producer_boxes.get(pp_obj_idx) # Use simple box
                          if pp_obj_np_diagram and pp_obj_idx not in processed_indices:
                               processed_indices.add(pp_obj_idx) # Mark simple box as used

                      # Compose PP if object diagram is valid
                      if pp_obj_np_diagram and pp_obj_np_diagram.cod == N:
                           try:
                               composed_pp = pp_obj_np_diagram >> prep_box
                               if composed_pp.cod == N:
                                    modifiers_to_apply.append({'idx': dep_idx, 'box': composed_pp, 'rel': 'pp_nmod', 'order': 2, 'pp_obj_idx': pp_obj_idx})
                                    logger.info(f"  Identified NP-PP modifier: Prep='{modifier_analysis['text']}' (idx {dep_idx}), Obj='{analysis_map[pp_obj_idx]['text']}' (idx {pp_obj_idx})")
                               else: logger.warning(f"  PP composition failed for prep {dep_idx}.")
                           except Exception as e_pp_compose: logger.error(f"  Error composing PP for prep {dep_idx}: {e_pp_compose}")
                      else: logger.warning(f"  Could not get valid diagram (even fallback) for PP object {pp_obj_idx}.")
                 else: logger.debug(f"  Could not find valid, unprocessed object for preposition {dep_idx}.")

    # Sort and apply modifiers
    modifiers_to_apply.sort(key=lambda m: m['order'])
    for mod_info in modifiers_to_apply:
        mod_idx = mod_info['idx']
        mod_box = mod_info['box']
        dep_rel = mod_info['rel']

        try:
            if dep_rel in ['det', 'amod']:
                 logger.info(f"  Applying {dep_rel.upper()} modifier: '{analysis_map[mod_idx]['text']}' (idx {mod_idx}) to NP '{head_analysis['text']}'")
                 if np_diagram.cod == mod_box.dom: # type: ignore
                     np_diagram = np_diagram >> mod_box
                     processed_indices.add(mod_idx)
                     logger.debug(f"    NP diagram after {dep_rel.upper()}: {np_diagram}")
                 else: logger.warning(f"    Type mismatch applying {dep_rel.upper()} {mod_idx}.")

            elif dep_rel == 'pp_nmod':
                 composed_pp_diag = mod_box
                 pp_obj_idx = mod_info['pp_obj_idx'] # Already marked processed if simple box used
                 if np_diagram.cod == N and composed_pp_diag.cod == N and N_MOD_BY_N is not None: # type: ignore
                     logger.info(f"  Applying NP-PP ({analysis_map[mod_idx]['text']}) to NP '{head_analysis['text']}' using N_MOD_BY_N")
                     np_diagram = (np_diagram @ composed_pp_diag) >> N_MOD_BY_N
                     processed_indices.add(mod_idx) # Mark preposition as used
                     logger.debug(f"    NP diagram after PP: {np_diagram}")
                 else: logger.warning(f"    Type mismatch applying PP {mod_idx} or N_MOD_BY_N is None.")

        except Exception as e_mod_apply:
            logger.error(f"  Error applying modifier {mod_idx} ({dep_rel}): {e_mod_apply}", exc_info=True)

    logger.info(f"Finished building NP V4 for '{head_analysis['text']}'. Final diagram cod: {np_diagram.cod if np_diagram else 'None'}. Consumed indices: {processed_indices}") # type: ignore
    if np_diagram and np_diagram.cod == N:
        return np_diagram
    else:
        logger.error(f"NP diagram build V4 for {head_noun_idx} resulted in invalid type: {np_diagram.cod if np_diagram else 'None'}. Returning None.")
        return None