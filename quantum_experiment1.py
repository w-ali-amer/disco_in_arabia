# -*- coding: utf-8 -*-
import stanza
from lambeq import AtomicType, IQPAnsatz  # Import necessary lambeq components
from lambeq.backend.grammar import Ty, Box, Cup, Id, Spider, Swap, Diagram as GrammarDiagram
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import traceback
from qiskit import QuantumCircuit
from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram
import os

# --- Imports for TKET/Qiskit Conversion ---
from pytket.extensions.qiskit.tket_backend import TketBackend
from pytket.extensions.qiskit import tk_to_qiskit # Lambeq's TKET backend
try:
    # pytket-qiskit provides the AerBackend and conversion methods
    from pytket.extensions.qiskit import AerBackend
    PYTKET_QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: pytket-qiskit extension not found.")
    print("Please install it: pip install pytket-qiskit")
    PYTKET_QISKIT_AVAILABLE = False
# --- END Imports ---


# Initialize Stanza with Arabic models
try:
    nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse', verbose=False) # Use verbose=False for cleaner output
    STANZA_AVAILABLE = True
except Exception as e:
    print(f"Error initializing Stanza: {e}")
    print("Please ensure Stanza Arabic models are downloaded: stanza.download('ar')")
    STANZA_AVAILABLE = False

# Define DisCoCat types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# ==================================
# Linguistic Analysis Function
# ==================================
def analyze_arabic_sentence(sentence, debug=True):
    """
    Analyzes an Arabic sentence using Stanza for dependency parsing.

    Args:
        sentence: The Arabic sentence to analyze
        debug: Whether to print debug information

    Returns:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        structure: Detected sentence structure ("VSO", "SVO", "NOMINAL", "COMPLEX", "OTHER")
        roles: Dictionary with indices of verb, subject, object and other dependencies
    """
    if not STANZA_AVAILABLE:
        raise RuntimeError("Stanza is not available or failed to initialize.")

    if not sentence or not sentence.strip():
        print("Warning: Received empty sentence for analysis.")
        return [], [], "OTHER", {}

    # Use Stanza for dependency parsing
    try:
        doc = nlp(sentence)
    except Exception as e_nlp:
        print(f"ERROR: Stanza processing failed for sentence: '{sentence}'")
        print(f"       Error: {e_nlp}")
        return [], [], "ERROR", {}

    # Extract tokens, POS tags, and dependency information
    tokens = []
    pos_tags = []
    dependencies = []
    heads = []
    lemmas = []

    # We'll work with the first sentence in the document
    if not doc.sentences:
         print(f"Warning: Stanza did not find any sentences in: '{sentence}'")
         return [], [], "OTHER", {} # Return empty/default values

    sent = doc.sentences[0]

    for word in sent.words:
        tokens.append(word.text)
        pos_tags.append(word.upos)       # Universal POS tag
        dependencies.append(word.deprel)  # Dependency relation
        heads.append(word.head)           # Head token ID (1-indexed)
        lemmas.append(word.lemma if word.lemma else word.text)

    # Combine information into analyses
    analyses = []
    for i in range(len(tokens)):
        # Convert from 1-indexed to 0-indexed for head
        head_idx = heads[i] - 1 if heads[i] > 0 else -1
        analyses.append((
            lemmas[i],      # Lemma
            pos_tags[i],    # POS tag
            dependencies[i], # Dependency relation
            head_idx        # Head index (0-indexed)
        ))

    if debug:
        print("\nParsed sentence with dependencies:")
        for i, (token, (lemma, pos, dep, head)) in enumerate(zip(tokens, analyses)):
            print(f"  {i}: Token='{token}', Lemma='{lemma}', POS='{pos}', Dep='{dep}', Head={head}")

    # Determine sentence structure and roles
    roles = {"verb": None, "subject": None, "object": None, "root": None}

    # Find the root of the dependency tree
    for i, (_, _, dep, _) in enumerate(analyses):
        if dep == "root":
            roles["root"] = i
            break

    # Find verb, subject, and object based on dependencies relative to the root or other verbs
    potential_verbs = [i for i, (_, pos, _, _) in enumerate(analyses) if pos == "VERB"]
    if not potential_verbs and roles["root"] is not None and analyses[roles["root"]][1] == "VERB":
         potential_verbs.append(roles["root"]) # Root is the only verb

    # Identify main verb (often the root, or the highest verb in the tree)
    if roles["root"] is not None and analyses[roles["root"]][1] == "VERB":
         roles["verb"] = roles["root"]
    elif potential_verbs:
         # Simple heuristic: pick the first verb found if root isn't a verb
         roles["verb"] = potential_verbs[0]
         # Could be refined by checking dependencies

    # Find subject and object based on standard dependency relations
    for i, (_, _, dep, head) in enumerate(analyses):
        # Standard subject relations
        if dep in ["nsubj", "csubj"]:
            # Check if it attaches to the identified main verb or the root
            if head == roles["verb"] or head == roles["root"]:
                 roles["subject"] = i
        # Standard object relations
        elif dep in ["obj", "iobj", "dobj", "ccomp", "xcomp"]:
             # Check if it attaches to the identified main verb
             if head == roles["verb"]:
                  roles["object"] = i

    # Determine structure based on relative order of V, S, O
    verb_idx = roles.get("verb")
    subj_idx = roles.get("subject")
    obj_idx = roles.get("object") # Object might be None for intransitive

    structure = "OTHER" # Default
    num_verbs = len(potential_verbs)

    if verb_idx is not None:
         if subj_idx is not None:
              if verb_idx < subj_idx: # Verb appears before subject
                   structure = "VSO"
              elif subj_idx < verb_idx: # Subject appears before verb
                   structure = "SVO"
              # If verb_idx == subj_idx, it's likely an error or unusual structure, keep OTHER
         else: # Verb exists but no subject found (e.g., imperative, passive?)
              structure = "VERBAL_OTHER" # More specific than OTHER
         if num_verbs > 1: # If multiple verbs detected, mark as complex
              structure = "COMPLEX_" + structure # e.g., COMPLEX_SVO
    elif subj_idx is not None:
         # If there's a subject but no clear main verb (root might be noun/adj), it's likely nominal
         if roles["root"] is not None and analyses[roles["root"]][1] in ["NOUN", "PROPN", "ADJ", "PRON"]:
              structure = "NOMINAL"
    elif num_verbs > 1:
         structure = "COMPLEX_OTHER" # Multiple verbs but no clear SVO/VSO


    if debug:
        print(f"\nDetected structure: {structure}")
        print(f"  Verb index: {verb_idx} ({tokens[verb_idx] if verb_idx is not None else 'None'})")
        print(f"  Subject index: {subj_idx} ({tokens[subj_idx] if subj_idx is not None else 'None'})")
        print(f"  Object index: {obj_idx} ({tokens[obj_idx] if obj_idx is not None else 'None'})")
        print(f"  Root index: {roles.get('root')} ({tokens[roles.get('root')] if roles.get('root') is not None else 'None'})")

    # Build complete dependency graph
    dependency_graph = {i: [] for i in range(len(tokens))}
    for i, (_, _, dep, head) in enumerate(analyses):
        if head >= 0:
            if head < len(tokens):
                 dependency_graph[head].append((i, dep))
            elif debug:
                 print(f"Warning: Invalid head index {head} for token {i} ('{tokens[i]}'). Skipping dependency edge.")

    if debug:
        print("\nDependency graph (Head -> [(Dependent Index, Relation)]):")
        for head, dependents in dependency_graph.items():
            if dependents:
                head_token = tokens[head] if head < len(tokens) else "INVALID_HEAD"
                dep_info = [(f"{idx}:{tokens[idx]}" if idx < len(tokens) else f"{idx}:INVALID_DEP", rel) for idx, rel in dependents]
                print(f"  {head} ('{head_token}'): {dep_info}")

    roles["dependency_graph"] = dependency_graph
    roles["structure"] = structure # Store final structure

    return tokens, analyses, structure, roles

# ==================================
# DisCoCat Type Assignment
# ==================================
def assign_discocat_types(pos, dep_rel, is_verb=False, verb_takes_subject=False, verb_takes_object=False, is_nominal_pred=False, debug=True):
    """
    Assigns DisCoCat types based on POS tag and role.
    Verb type depends on whether it takes subject/object.
    Nominal predicates get functional type n.l >> s.
    """
    n = AtomicType.NOUN
    s = AtomicType.SENTENCE
    assigned_type = None

    if is_verb:
        type_ = s
        if verb_takes_subject: type_ = n.r @ type_
        if verb_takes_object: type_ = type_ @ n.l
        assigned_type = type_
    elif is_nominal_pred:
        # Predicate in nominal sentence (often ADJ or NOUN)
        assigned_type = n.l >> s # Takes subject N on left, produces S
    elif pos in ["NOUN", "PROPN", "PRON"]:
        assigned_type = n
    elif pos == "ADJ":
        # Adjective modifying a noun
        assigned_type = n @ n.l # Takes N on left, produces N
    elif pos == "DET":
        assigned_type = n @ n.l # Determiner modifying noun
    elif pos == "ADV":
        assigned_type = s @ s.l # Adverb modifying sentence/verb (simplified)
    elif pos == "ADP":
        assigned_type = n.r @ n @ n.l # Preposition linking two Nouns
    elif pos == "CCONJ":
        assigned_type = s.r @ s @ s.l # Coordinating conjunction linking Sentences
    elif pos == "SCONJ":
        assigned_type = s.r @ s # Subordinating conjunction linking Sentences
    else:
        # Default fallback for PART, AUX, PUNCT, NUM, etc.
        assigned_type = n # Treat as Noun for simplicity, may need removal later

    if debug: print(f"DEBUG assign_discocat_types: POS='{pos}', Dep='{dep_rel}', is_verb={is_verb}, is_nom_pred={is_nominal_pred} -> Type={assigned_type}")
    return assigned_type

# ==================================
# Diagram Creation Functions
# ==================================

def create_nominal_sentence_diagram(tokens, analyses, roles, debug=True):
    """
    Create DisCoCat diagram specifically for nominal sentences.
    Attempts Cup composition between subject and the first valid predicate.
    """
    if debug: print("\nAttempting to create diagram for NOMINAL sentence...")
    dependency_graph = roles.get("dependency_graph", {})
    root_idx = roles.get("root")

    # Identify subject (often the root noun/pronoun)
    subject_idx = None
    if root_idx is not None and analyses[root_idx][1] in ["NOUN", "PROPN", "PRON"]:
        subject_idx = root_idx
    else: # Look for nsubj dependent
        for i, (_, _, dep, head) in enumerate(analyses):
            if head == root_idx and dep == "nsubj":
                subject_idx = i
                break

    # Identify potential predicates (dependents of subject/root, or root itself if not subject)
    predicate_indices = []
    check_indices = [subject_idx, root_idx] if subject_idx != root_idx else [root_idx]
    for head_idx in filter(None, check_indices):
         if head_idx in dependency_graph:
              for dep_idx, dep_rel in dependency_graph[head_idx]:
                   if analyses[dep_idx][1] in ["ADJ", "NOUN"] and dep_rel in ["amod", "nmod", "appos", "xcomp", "acl", "advcl", "conj", "root", "obl"]:
                        predicate_indices.append(dep_idx)
                   elif dep_rel == "cop": # Include copula dependents
                        predicate_indices.append(dep_idx)
    # If root is ADJ/NOUN and not subject, it can be predicate
    if root_idx is not None and root_idx != subject_idx and analyses[root_idx][1] in ["ADJ", "NOUN"]:
         predicate_indices.append(root_idx)

    predicate_indices = sorted(list(set(idx for idx in predicate_indices if idx is not None)))

    if debug:
         subj_token = tokens[subject_idx] if subject_idx is not None else "None"
         pred_tokens = [tokens[i] for i in predicate_indices]
         print(f"  Nominal Subject Index: {subject_idx} ('{subj_token}')")
         print(f"  Nominal Predicate Indices: {predicate_indices} ({pred_tokens})")

    # Create word boxes
    word_boxes = {}
    first_predicate_idx = -1
    for i, (token, (lemma, pos, dep_rel, head)) in enumerate(zip(tokens, analyses)):
        is_pred = i in predicate_indices and first_predicate_idx == -1 # Mark only the first predicate
        if is_pred: first_predicate_idx = i
        output_type = assign_discocat_types(pos, dep_rel, is_nominal_pred=is_pred, debug=debug)
        word_boxes[i] = Box(token, Ty(), output_type)
        if debug: print(f"  Nominal Box {i}: {token} ({pos}, {dep_rel}) -> Type: {output_type}")

    # --- Composition Logic ---
    diagram = None
    if subject_idx is not None and subject_idx in word_boxes:
        subj_box = word_boxes[subject_idx]
        # Try Cup composition with the first predicate found
        if first_predicate_idx != -1 and first_predicate_idx in word_boxes:
            pred_box = word_boxes[first_predicate_idx]
            # Check if types are compatible: N @ (N.l >> S)
            if subj_box.cod == N and pred_box.cod == (N.l >> S):
                try:
                    if debug: print(f"  Applying Cup composition: Subj={subj_box.cod}, Pred={pred_box.cod}")
                    diagram = subj_box @ pred_box >> Cup(N, N.l)
                    applied_cup_successfully = True
                except ValueError as e_cup:
                    print(f"ERROR: Nominal Cup composition failed: {e_cup}. Falling back.")
                    diagram = subj_box # Fallback to just subject
                    applied_cup_successfully = False
            else:
                print(f"WARNING: Type mismatch for nominal Cup: Subj={subj_box.cod}, Pred={pred_box.cod}. Falling back.")
                diagram = subj_box # Fallback to just subject
                applied_cup_successfully = False

            # Tensor remaining words (including other predicates and modifiers)
            remaining_indices = [i for i in range(len(tokens)) if i != subject_idx and i != first_predicate_idx]
            remaining_indices.sort()
            for r_idx in remaining_indices:
                if r_idx in word_boxes:
                    diagram = diagram @ word_boxes[r_idx] # Simple tensor product for others

        else: # No predicate found or box missing
            if debug: print("  No valid predicate found for Cup, using subject and tensoring others.")
            diagram = subj_box
            remaining_indices = [i for i in range(len(tokens)) if i != subject_idx]
            remaining_indices.sort()
            for r_idx in remaining_indices:
                 if r_idx in word_boxes: diagram = diagram @ word_boxes[r_idx]

    # Fallback if subject logic failed
    if diagram is None:
        if debug: print("  Nominal subject logic failed, using fallback tensor of all words.")
        indices = sorted(word_boxes.keys())
        if indices:
            diagram = Id(Ty()) # Start with empty domain
            for idx in indices: diagram = diagram @ word_boxes[idx]
        else: diagram = Id(S)

    # Final check: Ensure output type is Sentence
    if not hasattr(diagram, 'cod') or diagram.cod != S:
        if debug: print(f"  Final nominal diagram cod is {getattr(diagram, 'cod', 'None')}. Forcing to S.")
        try:
            # Add a box that takes the current codomain and outputs S
            diagram = diagram >> Box(f"Force_S", diagram.cod, S)
        except Exception as e_conv:
            print(f"ERROR: Could not force nominal diagram to S type: {e_conv}. Using Id(S).")
            diagram = Id(S)

    if debug: print(f"Final nominal diagram created.")
    return diagram


def create_verbal_sentence_diagram(tokens, analyses, roles, debug=True):
    """
    Creates a DisCoCat diagram for verbal sentences with enhanced grammatical nuance handling.
    Supports VSO/SVO structures with modifiers, prepositional phrases, and dependent clauses.
    """
    if debug: print("\nAttempting to create diagram for VERBAL sentence with enhanced nuance...")
    structure = roles.get("structure", "OTHER")
    verb_idx = roles.get("verb")
    subj_idx = roles.get("subject")
    obj_idx = roles.get("object")
    root_idx = roles.get("root")
    dependency_graph = roles.get("dependency_graph", {})
    
    # Handle case where main verb is not identified
    if verb_idx is None:
        if debug: print("No main verb identified. Trying to recover from dependencies...")
        # Try to find a VERB among the tokens
        for i, (_, (_, pos, _, _)) in enumerate(zip(tokens, analyses)):
            if pos == "VERB":
                verb_idx = i
                roles["verb"] = i
                if debug: print(f"Recovered verb at index {i}: '{tokens[i]}'")
                break
    
    if verb_idx is None:
        print("ERROR: Cannot create verbal diagram without a verb. Falling back.")
        # Fallback to tensor product
        word_boxes_fallback = {i: Box(token, Ty(), N) for i, token in enumerate(tokens)}
        indices = sorted(word_boxes_fallback.keys())
        diagram = Id(Ty()) if indices else Id(S)
        for idx in indices: diagram = diagram @ word_boxes_fallback[idx]
        return diagram >> Box("Force_S", diagram.cod, S)

    # Categorize all tokens by their dependency relationship
    token_categories = {
        "subject": set() if subj_idx is None else {subj_idx},
        "object": set() if obj_idx is None else {obj_idx},
        "verb": {verb_idx},
        "aux_verbs": set(),  # Auxiliary verbs
        "verb_modifiers": set(),  # Adverbs, negation, etc.
        "noun_modifiers": set(),  # Adjectives, determiners, etc.
        "prep_phrases": set(),  # Prepositions
        "conj": set(),  # Conjunctions
        "complements": set(),  # Various clause complements
        "other": set()  # Anything else
    }
    
    # Enhanced dependency analysis - populate categories
    for i, (_, (_, pos, dep_rel, head)) in enumerate(zip(tokens, analyses)):
        if i in token_categories["subject"] or i in token_categories["object"] or i in token_categories["verb"]:
            continue  # Skip already categorized core elements
            
        if head == verb_idx:
            # Dependents of the main verb
            if dep_rel in ["advmod", "neg"]:
                token_categories["verb_modifiers"].add(i)
            elif dep_rel in ["aux", "auxpass"]:
                token_categories["aux_verbs"].add(i)
            elif dep_rel in ["ccomp", "xcomp", "advcl"]:
                token_categories["complements"].add(i)
            elif dep_rel in ["prep", "case"]:
                token_categories["prep_phrases"].add(i)
            elif dep_rel in ["dobj", "iobj", "obj"] and obj_idx is None:
                # Found object not identified earlier
                token_categories["object"].add(i)
                obj_idx = i
                roles["object"] = i
            else:
                token_categories["other"].add(i)
        elif head == subj_idx or head == obj_idx:
            # Dependents of subject or object
            if dep_rel in ["amod", "det", "nummod", "nmod", "appos"]:
                token_categories["noun_modifiers"].add(i)
            elif dep_rel == "case":
                token_categories["prep_phrases"].add(i)
            else:
                token_categories["other"].add(i)
        elif pos in ["CCONJ", "SCONJ"]:
            token_categories["conj"].add(i)
        else:
            token_categories["other"].add(i)
    
    # Create word boxes with more nuanced type assignments
    word_boxes = {}
    for i, (token, (lemma, pos, dep_rel, head)) in enumerate(zip(tokens, analyses)):
        # Decide if this is verb and its valency (what it takes as arguments)
        is_verb = i == verb_idx
        verb_takes_subject = subj_idx is not None
        verb_takes_object = obj_idx is not None
        
        # Handle specific grammatical roles
        if is_verb:
            output_type = assign_discocat_types(pos, dep_rel, is_verb=True, 
                                               verb_takes_subject=verb_takes_subject,
                                               verb_takes_object=verb_takes_object,
                                               debug=debug)
        elif i in token_categories["subject"] or i in token_categories["object"]:
            output_type = N  # Noun type for subjects and objects
        elif i in token_categories["aux_verbs"]:
            # Auxiliary verbs modify the main verb: e.g., "كان يقرأ" (was reading)
            output_type = assign_discocat_types("VERB", "aux", is_verb=False, debug=debug)
        elif i in token_categories["verb_modifiers"]:
            if pos == "ADV":
                output_type = S @ S.l  # Adverb type: modifies sentences
            else:
                output_type = assign_discocat_types(pos, dep_rel, debug=debug)
        elif i in token_categories["noun_modifiers"]:
            if pos == "ADJ":
                output_type = N @ N.l  # Adjectives: modify noun and produce noun
            elif pos == "DET":
                output_type = N @ N.l  # Determiners modify nouns
            else:
                output_type = assign_discocat_types(pos, dep_rel, debug=debug)
        elif i in token_categories["prep_phrases"]:
            # Prepositions link nouns to other structures
            output_type = N.r @ N @ N.l
        elif i in token_categories["conj"]:
            if pos == "CCONJ":  # Coordinating conjunction
                output_type = S.r @ S @ S.l
            else:  # Subordinating conjunction
                output_type = S.r @ S
        elif i in token_categories["complements"]:
            # Complements can be complex - often another clause
            if dep_rel == "ccomp":  # Clausal complement
                output_type = S  # Treat as a sentence
            elif dep_rel == "xcomp":  # Open clausal complement
                output_type = N.l >> S  # Takes an implied subject
            else:
                output_type = assign_discocat_types(pos, dep_rel, debug=debug)
        else:
            # Default assignment for other tokens
            output_type = assign_discocat_types(pos, dep_rel, debug=debug)
        
        word_boxes[i] = Box(token, Ty(), output_type)
        if debug: print(f"  Verbal Box {i}: {token} ({pos}, {dep_rel}) -> Type: {output_type}")
    
    # --- Enhanced Composition Logic ---
    diagram = None
    try:
        # Get core boxes
        verb_box = word_boxes.get(verb_idx)
        subj_box = word_boxes.get(subj_idx)
        obj_box = word_boxes.get(obj_idx)
        
        # Choose composition strategy based on sentence structure
        if structure.endswith("SVO"):
            if debug: print("  Enhanced composition for SVO structure...")
            
            # Order matters: Subject @ Verb @ Object
            # Start with the subject
            current_diagram = subj_box if subj_box else Id(Ty())
            
            # Add verb modifiers that should appear before the verb
            verb_mod_before = sorted([i for i in token_categories["verb_modifiers"] 
                                     if i < verb_idx])
            for mod_idx in verb_mod_before:
                if mod_idx in word_boxes:
                    current_diagram = current_diagram @ word_boxes[mod_idx]
            
            # Add the verb
            current_diagram = current_diagram @ verb_box
            
            # Add verb modifiers that should appear after the verb
            verb_mod_after = sorted([i for i in token_categories["verb_modifiers"] 
                                    if i > verb_idx and i < obj_idx]) if obj_box else []
            for mod_idx in verb_mod_after:
                if mod_idx in word_boxes:
                    current_diagram = current_diagram @ word_boxes[mod_idx]
            
            # Add the object if it exists
            if obj_box:
                current_diagram = current_diagram @ obj_box
            
            # Add remaining elements in order
            remaining_indices = sorted(set(word_boxes.keys()) - 
                                      {subj_idx, verb_idx, obj_idx} - 
                                      set(verb_mod_before) - 
                                      set(verb_mod_after))
            
            # Track which ones we'll process with cup compositions later
            processed_indices = {subj_idx, verb_idx, obj_idx}
            
            # --- Apply Cup compositions between core elements ---
            # Calculate the current diagram's type
            current_type = current_diagram.cod
            
            # Apply cups between subject, verb, and object
            # For SVO: (N @ (N.r @ S @ N.l) @ N) >> Cup(N, N.r) >> Swap(S @ N.l, N) >> Id(S) @ Cup(N.l, N)
            composition_steps = Id(current_type)
            wires_to_cup = 0
            
            if verb_takes_subject:
                # Cup subject with verb's subject expectation
                composition_steps = composition_steps >> Cup(N, N.r) @ Id(current_type[len(N @ N.r):])
                wires_to_cup += 2
            
            if verb_takes_object and obj_box:
                # More complex swapping to bring together verb's object expectation and object
                remaining_type = current_type[wires_to_cup:]
                
                # This section needs careful handling of wire ordering
                if remaining_type == S @ N.l @ N:  # Common case after subject cup
                    # Need to swap N.l past N to cup them
                    composition_steps = composition_steps >> Id(S) @ Swap(N.l, N)
                    # Now cup N.l and N
                    composition_steps = composition_steps >> Id(S) @ Cup(N.l, N)
                elif remaining_type.count(N.l) == 1 and remaining_type.count(N) >= 1:
                    # Try to find and cup the first N.l with the next available N
                    nl_index = list(remaining_type).index(N.l)
                    n_indices = [i for i, t in enumerate(remaining_type) if t == N]
                    if n_indices:
                        n_index = min([i for i in n_indices if i > nl_index], default=n_indices[0])
                        # Need complex swapping to bring them together
                        between_types = remaining_type[nl_index+1:n_index]
                        
                        # Swap N.l rightward until it meets N
                        swap_steps = Id(remaining_type[:nl_index])
                        current_pos = nl_index
                        
                        for between_type in between_types:
                            swap_steps = swap_steps @ Swap(N.l, between_type) @ Id(remaining_type[current_pos+2:])
                            current_pos += 1
                        
                        composition_steps = composition_steps >> swap_steps
                        
                        # Now cup N.l and N
                        cup_step = Id(remaining_type[:nl_index]) @ Cup(N.l, N) @ Id(remaining_type[n_index+1:])
                        composition_steps = composition_steps >> cup_step
                else:
                    print(f"Warning: Complex type arrangement for object cup in SVO: {remaining_type}")
                    # Fallback to simpler approach if swapping gets too complex
            
            # Apply the composition steps to the diagram
            diagram = current_diagram >> composition_steps
            
            # Process remaining elements
            for idx in remaining_indices:
                if idx in word_boxes and idx not in processed_indices:
                    # Handle modifiers, prepositional phrases, etc.
                    if idx in token_categories["noun_modifiers"]:
                        # Find which noun this modifies
                        head = analyses[idx][3]
                        if head in word_boxes:
                            # Could use Spider for proper attachment, but often just tensoring works
                            diagram = diagram @ word_boxes[idx]
                    else:
                        # For other elements, just tensor them for now
                        diagram = diagram @ word_boxes[idx]
                        
        elif structure.endswith("VSO"):
            if debug: print("  Enhanced composition for VSO structure...")
            
            # Order: Verb @ Subject @ Object
            # Start with the verb
            current_diagram = verb_box
            
            # Add subject
            if subj_box:
                current_diagram = current_diagram @ subj_box
            
            # Add object if it exists
            if obj_box:
                current_diagram = current_diagram @ obj_box
            
            # Add remaining elements in order
            remaining_indices = sorted(set(word_boxes.keys()) - {verb_idx, subj_idx, obj_idx})
            
            # Calculate current diagram's type
            current_type = current_diagram.cod
            
            # Apply cups between verb, subject, and object
            # For VSO: (N.r @ S @ N.l) @ N @ N >> Id(N.r) @ Swap(S @ N.l, N) >> Id(N.r @ N) @ Cup(S @ N.l, N)
            composition_steps = Id(current_type)
            
            # For VSO, often easier to cup object first, then subject
            if verb_takes_object and obj_box:
                # Find N.l in verb type and cup with object N
                if current_type.count(N.l) >= 1 and current_type.count(N) >= 2:  # Need at least one N.l and two Ns
                    try:
                        # Cup N.l with the second N (object)
                        nl_index = list(current_type).index(N.l)
                        n_indices = [i for i, t in enumerate(current_type) if t == N]
                        if len(n_indices) >= 2:  # Need at least two Ns (subject and object)
                            # Object N is typically the second N in the type
                            obj_n_index = n_indices[1]
                            
                            # Complex swapping to bring N.l and object N together
                            if obj_n_index > nl_index:
                                between_types = current_type[nl_index+1:obj_n_index]
                                
                                # Swap N.l rightward until it meets object N
                                swap_steps = Id(current_type[:nl_index])
                                current_pos = nl_index
                                
                                for between_type in between_types:
                                    swap_steps = swap_steps @ Swap(N.l, between_type) @ Id(current_type[current_pos+2:])
                                    current_pos += 1
                                
                                composition_steps = composition_steps >> swap_steps
                            elif nl_index > obj_n_index:
                                between_types = current_type[obj_n_index+1:nl_index]
                                
                                # Swap object N rightward until it meets N.l
                                swap_steps = Id(current_type[:obj_n_index])
                                current_pos = obj_n_index
                                
                                for between_type in between_types:
                                    swap_steps = swap_steps @ Swap(N, between_type) @ Id(current_type[current_pos+2:])
                                    current_pos += 1
                                
                                composition_steps = composition_steps >> swap_steps
                            
                            # Now cup N.l and object N
                            min_idx = min(nl_index, obj_n_index)
                            cup_step = Id(current_type[:min_idx]) @ Cup(N.l, N) @ Id(current_type[max(nl_index, obj_n_index)+1:])
                            composition_steps = composition_steps >> cup_step
                            
                            # Update current_type after cupping
                            new_type_list = list(current_type)
                            new_type_list.pop(max(nl_index, obj_n_index))
                            new_type_list.pop(min(nl_index, obj_n_index))
                            current_type = Ty(*new_type_list)
                    except Exception as e_obj_cup:
                        print(f"Warning: Error during VSO object cup: {e_obj_cup}")
            
            if verb_takes_subject and subj_box:
                # Find N.r in verb type and cup with subject N
                if current_type.count(N.r) >= 1 and current_type.count(N) >= 1:
                    try:
                        # Cup N.r with the first remaining N (subject)
                        nr_index = list(current_type).index(N.r)
                        n_indices = [i for i, t in enumerate(current_type) if t == N]
                        if n_indices:  # Need at least one remaining N
                            subj_n_index = n_indices[0]
                            
                            # Complex swapping to bring N.r and subject N together
                            if subj_n_index > nr_index:
                                between_types = current_type[nr_index+1:subj_n_index]
                                
                                # Swap N.r rightward until it meets subject N
                                swap_steps = Id(current_type[:nr_index])
                                current_pos = nr_index
                                
                                for between_type in between_types:
                                    swap_steps = swap_steps @ Swap(N.r, between_type) @ Id(current_type[current_pos+2:])
                                    current_pos += 1
                                
                                composition_steps = composition_steps >> swap_steps
                            elif nr_index > subj_n_index:
                                between_types = current_type[subj_n_index+1:nr_index]
                                
                                # Swap subject N rightward until it meets N.r
                                swap_steps = Id(current_type[:subj_n_index])
                                current_pos = subj_n_index
                                
                                for between_type in between_types:
                                    swap_steps = swap_steps @ Swap(N, between_type) @ Id(current_type[current_pos+2:])
                                    current_pos += 1
                                
                                composition_steps = composition_steps >> swap_steps
                            
                            # Now cup N.r and subject N
                            min_idx = min(nr_index, subj_n_index)
                            cup_step = Id(current_type[:min_idx]) @ Cup(N.r, N) @ Id(current_type[max(nr_index, subj_n_index)+1:])
                            composition_steps = composition_steps >> cup_step
                    except Exception as e_subj_cup:
                        print(f"Warning: Error during VSO subject cup: {e_subj_cup}")
            
            # Apply the composition steps to the diagram
            diagram = current_diagram >> composition_steps
            
            # Process remaining elements
            for idx in remaining_indices:
                if idx in word_boxes:
                    diagram = diagram @ word_boxes[idx]
                    
        else:  # Fallback for OTHER verbal structures
            if debug: print("  Using enhanced fallback for OTHER verbal structure.")
            
            # Create a more intelligent fallback based on dependency relations
            verb_first = verb_idx < subj_idx if subj_idx is not None else True
            
            core_indices = [i for i in [verb_idx, subj_idx, obj_idx] if i is not None]
            if verb_first:
                ordered_indices = sorted(core_indices)
            else:
                # Try to maintain SVO order even if we don't recognize the exact pattern
                ordered_indices = sorted([i for i in core_indices if i != verb_idx])
                if verb_idx is not None:
                    verb_pos = 0 if not ordered_indices else 1
                    ordered_indices.insert(verb_pos, verb_idx)
            
            # Add remaining indices in order
            all_indices = ordered_indices + sorted(set(word_boxes.keys()) - set(core_indices))
            
            # Create diagram by tensoring in proper order
            if not all_indices:
                diagram = Id(S)
            else:
                diagram = word_boxes[all_indices[0]]
                for idx in all_indices[1:]:
                    if idx in word_boxes:
                        diagram = diagram @ word_boxes[idx]

    except Exception as e_comp:
        print(f"ERROR: Enhanced verbal composition failed: {e_comp}")
        traceback.print_exc()
        # Fall back to simple tensor product
        indices = sorted(word_boxes.keys())
        if not indices:
            diagram = Id(S)
        else:
            diagram = word_boxes[indices[0]]
            for idx in indices[1:]:
                if idx in word_boxes:
                    diagram = diagram @ word_boxes[idx]
    
    # Final check: Ensure output type is Sentence
    if diagram is None:
        print("ERROR: Verbal diagram creation failed completely. Using Id(S).")
        diagram = Id(S)
    elif not hasattr(diagram, 'cod') or diagram.cod != S:
        if debug: print(f"  Final verbal diagram cod is {getattr(diagram, 'cod', 'None')}. Forcing to S.")
        try:
            diagram = diagram >> Box(f"Force_S", diagram.cod, S)
        except Exception as e_conv:
            print(f"ERROR: Could not force verbal diagram to S type: {e_conv}. Using Id(S).")
            diagram = Id(S)
    
    if debug: print(f"Final enhanced verbal diagram created.")
    return diagram


# ==================================
# Main Conversion Function
# ==================================
def arabic_to_quantum_enhanced(sentence, debug=True):
    """
    Processes an Arabic sentence, creates a DisCoCat diagram, and converts
    it to a Qiskit QuantumCircuit using TketBackend and AerBackend.

    Args:
        sentence (str): The Arabic sentence.
        debug (bool): Whether to print debugging information.

    Returns:
        tuple: (circuit, diagram, structure, tokens, analyses, roles)
               circuit is a qiskit.QuantumCircuit or None on failure.
               diagram is a lambeq GrammarDiagram or None.
    """
    if not PYTKET_QISKIT_AVAILABLE:
         print("ERROR: pytket-qiskit extension is required but not available. Cannot proceed.")
         try:
              tokens, analyses, structure, roles = analyze_arabic_sentence(sentence, debug=False)
              return None, None, structure, tokens, analyses, roles
         except Exception as e_analyze:
              print(f"Error during fallback analysis: {e_analyze}")
              return None, None, "ERROR", [], [], {}

    # 1. Analyze sentence
    try:
        tokens, analyses, structure, roles = analyze_arabic_sentence(sentence, debug)
        if structure == "ERROR": # Handle analysis failure
             return None, None, structure, tokens, analyses, roles
    except Exception as e_analyze_main:
         print(f"ERROR: Sentence analysis failed unexpectedly: {e_analyze_main}")
         traceback.print_exc()
         return None, None, "ERROR", [], [], {}


    # 2. Create DisCoCat Diagram
    diagram = None
    try:
        # Determine if nominal and call appropriate diagram function
        if structure == "NOMINAL":
            if debug: print("\nCreating NOMINAL sentence diagram...")
            diagram = create_nominal_sentence_diagram(tokens, analyses, roles, debug)
        elif structure != "OTHER" and structure != "ERROR": # Handle VSO, SVO, COMPLEX, VERBAL_OTHER
            if debug: print(f"\nCreating VERBAL/COMPLEX sentence diagram ({structure})...")
            diagram = create_verbal_sentence_diagram(tokens, analyses, roles, debug)
        else: # Fallback for OTHER or unhandled structures
             print(f"Warning: Unhandled sentence structure '{structure}'. Falling back to basic tensor diagram.")
             word_boxes_fallback = {i: Box(token, Ty(), N) for i, token in enumerate(tokens)}
             indices = sorted(word_boxes_fallback.keys())
             if not indices: diagram = Id(S)
             else:
                  diagram = Id(Ty())
                  for idx in indices: diagram = diagram @ word_boxes_fallback[idx]
                  diagram = diagram >> Box("Force_S", diagram.cod, S)


        if diagram is None:
            raise ValueError("Diagram creation returned None.")
        # Lambeq diagrams should inherit from GrammarDiagram
        if not isinstance(diagram, GrammarDiagram):
             if hasattr(diagram, 'cod') and hasattr(diagram, 'dom') and hasattr(diagram, 'normal_form'):
                  print(f"Warning: Diagram created is type {type(diagram)}, but seems compatible. Proceeding cautiously.")
             else:
                  raise TypeError(f"Diagram created is not a valid lambeq Diagram object: {type(diagram)}")

        if debug:
            print(f"\nDEBUG: Diagram created successfully. Type: {type(diagram)}")
            # print(f"Diagram details: {diagram}") # Verbose

    except Exception as e_diagram:
        print(f"ERROR: Exception during diagram creation for sentence: '{sentence}'")
        print(f"       Error: {e_diagram}")
        traceback.print_exc()
        return None, None, structure, tokens, analyses, roles

    # 3. Convert diagram to quantum circuit using TketBackend and AerBackend
    circuit = None
    try:
        # Define the object map explicitly for N and S types.
        ob_map = {N: 1, S: 1} # Simple 1 qubit per type
        ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3)

        if debug: print(f"\nDEBUG: Diagram before ansatz: dom={getattr(diagram, 'dom', 'N/A')}, cod={getattr(diagram, 'cod', 'N/A')}")

        if debug: print("DEBUG: Simplifying diagram...")
        simplified_diagram = diagram.normal_form()
        if debug: print(f"DEBUG: Simplified diagram: dom={simplified_diagram.dom}, cod={simplified_diagram.cod}")


        if debug: print("DEBUG: Applying ansatz...")
        quantum_diagram = ansatz(simplified_diagram) # This is a lambeq QuantumDiagram
        if debug: print(f"DEBUG: Ansatz applied. Quantum diagram type: {type(quantum_diagram)}")

        # --- Conversion using TKET ---
        if debug: print("DEBUG: Compiling lambeq quantum diagram using TketBackend...")
        tket_circuit = quantum_diagram.to_tk()
        from pytket.backends.backend import Backend
        from pytket.extensions.qiskit import AerStateBackend
        if debug: print("DEBUG: Initializing AerBackend for conversion...")
        # Create a backend instance first
        qiskit_backend = AerBackend()
        tket_backend = TketBackend(qiskit_backend)
        #tket_circuit = tket_backend.compile(quantum_diagram)
        if debug: print(f"DEBUG: Compiled to pytket circuit. Type: {type(tket_circuit)}")

        if debug: print("DEBUG: Initializing AerBackend for conversion...")
        qiskit_backend = AerBackend()

        if debug: print("DEBUG: Compiling pytket circuit for AerBackend...")
        #qiskit_backend.compile_circuit(tket_circuit)
        if debug: print("DEBUG: Pytket circuit compiled for AerBackend.")

        if debug: print("DEBUG: Converting pytket circuit to Qiskit QuantumCircuit...")
        circuit = tk_to_qiskit(tket_circuit)
        if debug: print("DEBUG: Conversion to Qiskit QuantumCircuit successful.")

    except Exception as e_circuit_outer:
        print(f"ERROR: Exception during circuit conversion setup for sentence: '{sentence}'")
        print(f"       Diagram: {diagram}") # Print diagram that caused error
        print(f"       Error: {e_circuit_outer}")
        traceback.print_exc()
        circuit = None # Ensure circuit is None

    # --- Handle failure ---
    if circuit is None:
        print(f"WARNING: Failed to convert lambeq diagram to Qiskit circuit for: '{sentence}'")
        return None, diagram, structure, tokens, analyses, roles

    # Final check: Ensure we have a Qiskit circuit object
    if not isinstance(circuit, QuantumCircuit):
         print(f"ERROR: Conversion resulted in unexpected type {type(circuit)}. Expected qiskit.QuantumCircuit. Returning None.")
         return None, diagram, structure, tokens, analyses, roles

    if debug: print(f"DEBUG: Circuit creation successful. Type: {type(circuit)}")
    # Return the Qiskit circuit object first, as expected by v4.py
    return circuit, diagram, structure, tokens, analyses, roles


# ==================================
# Visualization (Optional)
# ==================================
# Add visualization functions (visualize_dependency_tree, visualize_circuit, etc.) back here if needed.

# ==================================
# Main Execution / Testing (Optional)
# ==================================

def visualize_dependency_tree(tokens, analyses, dependency_structure=None, save_path=None):
    """
    Visualize the dependency tree using networkx and matplotlib.
    
    Args:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        dependency_structure: Parsed dependency structure
        save_path: Path to save the visualization
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, token in enumerate(tokens):
        # Get POS tag for coloring
        pos_tag = analyses[i][1] if i < len(analyses) else "UNK"
        
        # Determine node color based on grammatical role
        if dependency_structure:
            if i == dependency_structure.get("root"):
                node_color = "red"
            elif i == dependency_structure.get("verb"):
                node_color = "green"
            elif i == dependency_structure.get("subject"):
                node_color = "blue"
            elif i == dependency_structure.get("object"):
                node_color = "purple"
            else:
                node_color = "lightgray"
        else:
            node_color = "lightgray"
        
        G.add_node(i, label=token, pos=pos_tag, color=node_color)
    
    # Add edges
    for i, (_, _, dep, head) in enumerate(analyses):
        if head >= 0:  # Skip the root
            G.add_edge(head, i, label=dep)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility
    
    # Draw nodes
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, arrowsize=20)
    
    # Add node labels
    labels = {i: f"{i}: {tokens[i]}\n({G.nodes[i]['pos']})" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Add edge labels
    edge_labels = {(head, i): analyses[i][2] for i, (_, _, _, head) in enumerate(analyses) if head >= 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Dependency Parse Tree")
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    return plt.gcf()

def visualize_circuit(circuit, tokens=None, save_path=None):
    """
    Visualize the quantum circuit with improved layout.
    
    Args:
        circuit: The quantum circuit to visualize
        tokens: Optional list of tokens for the title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 8))
    
    title = "Quantum Circuit"
    if tokens:
        title += f" for: {' '.join(tokens)}"
    
    plt.title(title)
    
    # Draw the circuit with enhanced visibility
    try:
        circuit.draw(figsize=(14, 8))
    except Exception as e:
        print(f"Error drawing circuit: {e}")
        plt.text(0.5, 0.5, f"Could not render circuit: {e}", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()

def visualize_results(sentence, circuit, diagram, structure, tokens, analyses, dependency_structure, save_prefix=""):
    """
    Visualize the dependency tree, DisCoCat diagram and quantum circuit.
    
    Args:
        sentence: The original sentence
        circuit: The quantum circuit
        diagram: The DisCoCat diagram
        structure: The sentence structure
        tokens: The tokens of the sentence
        analyses: The linguistic analyses
        dependency_structure: The parsed dependency structure
        save_prefix: Prefix for saved files
    """
    # Create a readable representation of the tokens
    token_str = " ".join(tokens)
    
    # Print information
    print(f"\nSentence: {sentence}")
    print(f"Structure: {structure}")
    print(f"Tokens: {token_str}")
    print(f"\nDiagram type: {diagram.cod}")
    
    # Visualize dependency tree
    try:
        dep_tree_fig = plt.figure(figsize=(12, 8))
        dep_tree = visualize_dependency_tree(tokens, analyses, dependency_structure)
        filename = f"{save_prefix}dependency_tree.png" if save_prefix else "dependency_tree.png"
        dep_tree.savefig(filename)
        print(f"Saved dependency tree to {filename}")
        plt.close(dep_tree_fig)
    except Exception as e:
        print(f"Could not visualize dependency tree: {e}")
    
    # Visualize DisCoCat diagram with improved visualization
    try:
        diagram_fig = plt.figure(figsize=(14, 8))
        diagram_fig = visualize_discocat_diagram(diagram, tokens, structure)
        filename = f"{save_prefix}diagram_{structure}.png" if save_prefix else f"diagram_{structure}.png"
        diagram_fig.savefig(filename, dpi=150)
        print(f"Saved diagram to {filename}")
        plt.close(diagram_fig)
    except Exception as e:
        print(f"Could not visualize diagram: {e}")
    
    # Visualize the quantum circuit with enhanced visualization
    try:
        circuit_fig = plt.figure(figsize=(14, 8))
        circuit_fig = visualize_circuit(circuit, tokens)
        filename = f"{save_prefix}circuit_{structure}.png" if save_prefix else f"circuit_{structure}.png"
        circuit_fig.savefig(filename, dpi=150)
        print(f"Saved quantum circuit to {filename}")
        plt.close(circuit_fig)
    except Exception as e:
        print(f"Could not visualize quantum circuit: {e}")

# Test with examples
def test_examples():
    examples = [
        ("يقرأ الولد الكتاب", "VSO Example"),         # VSO: "reads the boy the book"
        ("الولد يقرأ الكتاب", "SVO Example"),         # SVO: "the boy reads the book"
        ("كتب الطالب الدرس", "VSO Past Tense"),      # VSO past: "wrote the student the lesson"
        ("الطالب يكتب الدرس", "SVO Present Tense"),   # SVO present: "the student writes the lesson"
        ("البيت كبير", "Nominal Sentence")            # Nominal: "the house is big"
    ]
    
    results = []
    
    for sentence, description in examples:
        print(f"\n{'-'*50}")
        print(f"{description}: {sentence}")
        print(f"{'-'*50}")
        
        circuit, diagram, structure, tokens, analyses, dependency_structure = arabic_to_quantum(sentence)
        visualize_results(sentence, circuit, diagram, structure, tokens, analyses, dependency_structure, f"{description.replace(' ', '_').lower()}_")
        
        results.append({
            "sentence": sentence,
            "description": description,
            "structure": structure,
            "diagram": diagram,
            "circuit": circuit,
            "tokens": tokens,
            "analyses": analyses,
            "dependency_structure": dependency_structure
        })
    
    return results



def analyze_diagram_structure(diagram, diagram_path=None):
    """Analyzes and optionally saves the diagram."""
    if diagram is None:
        print("  Analysis: No diagram provided.")
        return None
    try:
        metrics = {
            'n_boxes': len(diagram.boxes),
            'dom': str(diagram.dom),
            'cod': str(diagram.cod),
            # Add more metrics if needed (e.g., wire count, specific box counts)
        }
        print(f"  Diagram Metrics: {metrics}")
        if diagram_path:
            try:
                # Draw the diagram
                diagram.draw(figsize=(10, 6), path=diagram_path)
                print(f"  Diagram saved to: {diagram_path}")
            except Exception as e_draw:
                print(f"  Warning: Could not draw/save diagram: {e_draw}")
        return metrics
    except Exception as e:
        print(f"  Error analyzing diagram: {e}")
        return None

def analyze_circuit_structure(circuit, circuit_path=None):
    """Analyzes and optionally saves the circuit structure."""
    if circuit is None:
        print("  Analysis: No circuit provided.")
        return None
    if not isinstance(circuit, QuantumCircuit):
        print(f"  Analysis: Expected QuantumCircuit, got {type(circuit)}")
        return None
    try:
        metrics = {
            'num_qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'ops': dict(circuit.count_ops()),
            'num_parameters': len(circuit.parameters),
        }
        print(f"  Circuit Metrics: {metrics}")
        if circuit_path:
            try:
                # Draw the circuit
                circuit.draw(output='mpl', filename=circuit_path, fold=-1) # fold=-1 prevents wrapping
                print(f"  Circuit saved to: {circuit_path}")
                plt.close() # Close the matplotlib figure
            except Exception as e_draw:
                # Qiskit drawing can sometimes fail with complex circuits/matplotlib issues
                print(f"  Warning: Could not draw/save circuit: {e_draw}")
        return metrics
    except Exception as e:
        print(f"  Error analyzing circuit: {e}")
        return None

# --- Main Experiment Runner ---

if __name__ == "__main__":
    print("Running QNLP Experiment Script...")

    # --- Define Your Test Cases (Minimal Pairs/Sets) ---
    experiment_sets = {
        "WordOrder": [
            "الولدُ يقرأُ الكتابَ", # SVO
            "يقرأُ الولدُ الكتابَ", # VSO
        ],
        "LexicalAmbiguity": [
            "جاء الرجلُ القويُ",       # Context: Man
            "انكسرتْ رجلُ الطاولةِ",   # Context: Leg
            # Add a sentence where 'رجل' is ambiguous without broader context?
        ],
        "Morphology": [
             "يكتبُ", # He writes
             "يكتبونَ", # They write
        ],
        # Add more sets for other phenomena (e.g., coreference)
        # "Coreference": [
        #    "رأيتُ الطالبَ .",
        #    "هو ذكيٌ ." # Needs DiscoCirc approach ideally
        # ]
    }

    # --- Output Directory ---
    output_dir = "qnlp_experiments_output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Run Experiments ---
    all_results = {}

    for set_name, sentences in experiment_sets.items():
        print(f"\n{'='*10} Running Experiment Set: {set_name} {'='*10}")
        set_results = []
        for i, sentence in enumerate(sentences):
            print(f"\n--- Processing Sentence: '{sentence}' ---")
            sentence_prefix = f"{set_name}_{i}"
            result_data = {"sentence": sentence}
            try:
                # Generate diagram and circuit
                circuit, diagram, structure, tokens, analyses, roles = \
                    arabic_to_quantum_enhanced(sentence, debug=False) # Set debug=False for cleaner output

                result_data.update({
                    "structure": structure,
                    "tokens": tokens,
                    "analyses": analyses,
                    "roles": roles, # Contains dependency graph
                })

                # Analyze Diagram
                diagram_path = os.path.join(output_dir, f"{sentence_prefix}_diagram.png")
                result_data["diagram_metrics"] = analyze_diagram_structure(diagram, diagram_path)

                # Analyze Circuit
                circuit_path = os.path.join(output_dir, f"{sentence_prefix}_circuit.png")
                result_data["circuit_metrics"] = analyze_circuit_structure(circuit, circuit_path)

                 #--- Optional: Small-scale simulation (Use with caution!) ---
                if circuit is not None and circuit.num_qubits < 10:  #Example threshold
                    print("  Attempting small-scale simulation...")
                    try:
                         # You would need to import/adapt the kernel's feature extraction
                        from v6 import ArabicQuantumMeaningKernel
                        temp_kernel = ArabicQuantumMeaningKernel() # Need to initialize properly
                        features = temp_kernel.get_circuit_features(circuit, tokens, analyses, shots=8192)
                        print(f"  Simulated Features (Norm): {np.linalg.norm(features)}")
                        result_data["sim_features"] = features.tolist() # Store features if needed
                        pass # Placeholder
                    except Exception as e_sim:
                        print(f"  Simulation failed: {e_sim}")
                else:
                    print("  Skipping simulation (circuit None or too large).")


            except Exception as e_main:
                print(f"!!! ERROR processing sentence: '{sentence}' !!!")
                print(f"    Error: {e_main}")
                traceback.print_exc()
                result_data["error"] = str(e_main)

            set_results.append(result_data)
        all_results[set_name] = set_results

    # --- Basic Comparison (Example) ---
    print(f"\n{'='*10} Experiment Comparison {'='*10}")
    for set_name, results in all_results.items():
        print(f"\n--- Comparing Set: {set_name} ---")
        for i, res in enumerate(results):
            print(f"  Sentence {i}: {res['sentence']}")
            if "error" in res:
                print(f"    Status: ERROR ({res['error']})")
                continue
            print(f"    Structure: {res.get('structure')}")
            print(f"    Diagram Metrics: {res.get('diagram_metrics')}")
            print(f"    Circuit Metrics: {res.get('circuit_metrics')}")
            # Add more sophisticated comparison logic here
            # e.g., compare metrics between results[0] and results[1] for minimal pairs

    print("\n--- Experiments Finished ---")
    # You can save `all_results` to a file (e.g., JSON) for further analysis
    # import json
    # with open(os.path.join(output_dir, 'experiment_results.json'), 'w', encoding='utf-8') as f:
    #     # Convert numpy arrays in metrics if they exist, handle non-serializable roles if needed
    #     json.dump(all_results, f, ensure_ascii=False, indent=2, default=str) # default=str is a basic handler

