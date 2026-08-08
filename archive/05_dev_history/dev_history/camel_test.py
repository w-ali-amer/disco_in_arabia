import stanza
from lambeq import AtomicType, IQPAnsatz
from lambeq.backend.grammar import Ty, Box, Id, Cup, Spider
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import traceback
from qiskit import QuantumCircuit
from lambeq.backend.grammar import Diagram as GrammarDiagram
from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram
#from lambeq.backend.quantum import QiskitBackend
#from lambeq.backend.qiskit import QiskitBackend

# Initialize Stanza with Arabic models
# You'll need to run this once: stanza.download('ar')
nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma,depparse')

# Define DisCoCat types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

def analyze_arabic_sentence(sentence, debug=True):
    """
    Analyzes an Arabic sentence using Stanza for dependency parsing.
    
    Args:
        sentence: The Arabic sentence to analyze
        debug: Whether to print debug information
        
    Returns:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        structure: Detected sentence structure ("VSO", "SVO", or "OTHER")
        roles: Dictionary with indices of verb, subject, object and other dependencies
    """
    # Use Stanza for dependency parsing
    doc = nlp(sentence)
    
    # Extract tokens, POS tags, and dependency information
    tokens = []
    pos_tags = []
    dependencies = []
    heads = []
    lemmas = []
    
    # We'll work with the first sentence in the document
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
        print("Parsed sentence with dependencies:")
        for i, (token, (lemma, pos, dep, head)) in enumerate(zip(tokens, analyses)):
            print(f"{i}: Token: {token}, Lemma: {lemma}, POS: {pos}, Dep: {dep}, Head: {head}")
    
    # Determine sentence structure and roles
    roles = {}
    
    # Find the root of the dependency tree
    root_idx = None
    for i, (_, _, dep, _) in enumerate(analyses):
        if dep == "root":
            root_idx = i
            break
    
    if root_idx is not None:
        roles["root"] = root_idx
        
        # Find verb, subject, and object based on dependencies
        for i, (_, pos, dep, head) in enumerate(analyses):
            if dep in ["nsubj", "csubj"]:
                roles["subject"] = i
                roles["subject_head"] = head
            elif dep in ["obj", "iobj"]:
                roles["object"] = i
                roles["object_head"] = head
            
            # Also mark the main verb
            if pos == "VERB" and (dep == "root" or head == root_idx):
                roles["verb"] = i
    
    # Determine structure
    verb_idx = roles.get("verb")
    subj_idx = roles.get("subject")
    obj_idx = roles.get("object")
    
    if verb_idx is not None and subj_idx is not None:
        if verb_idx < subj_idx:
            structure = "VSO"
        elif subj_idx < verb_idx:
            structure = "SVO"
        else:
            structure = "OTHER"
    elif verb_idx is None and subj_idx is not None:
        structure = "NOMINAL"
    else:
        structure = "OTHER"
    
    if debug:
        print(f"Detected structure: {structure}")
        print(f"Verb index: {verb_idx}, Subject index: {subj_idx}, Object index: {obj_idx}")
        print(f"All roles: {roles}")
    
    # Build complete dependency graph
    dependency_graph = {i: [] for i in range(len(tokens))}
    for i, (_, _, dep, head) in enumerate(analyses):
        if head >= 0:
            dependency_graph[head].append((i, dep))
    
    if debug:
        print("Dependency graph:")
        for head, dependents in dependency_graph.items():
            if dependents:
                print(f"Token {head} ({tokens[head]}) has dependents: {dependents}")
    
    # Include dependency graph in roles
    roles["dependency_graph"] = dependency_graph
    
    return tokens, analyses, structure, roles

def assign_discocat_types(pos, dep_rel, has_subject=False, has_object=False, in_nominal=False, debug=True): # Added debug flag
    """
    Assigns DisCoCat types based on POS tag, dependency relation,
    and sentence type (nominal or verbal). Includes enhanced debugging.
    """
    n = AtomicType.NOUN
    s = AtomicType.SENTENCE
    assigned_type = None # Variable to track assigned type before return

    if in_nominal:
        # --- Logic for Nominal Sentences ---
        if pos in ["NOUN", "PROPN", "PRON"] and dep_rel == "root":
            assigned_type = n
            if debug: print(f"DEBUG assign_discocat_types (nominal): Assigning type '{assigned_type}' to {pos}/{dep_rel}")
            return assigned_type
        elif pos == "ADJ" and dep_rel in ["amod", "nsubj:pass", "flat", "nsubj", "root", "xcomp", "advmod", "obl", "acl", "advcl"]:
             assigned_type = n.l >> s
             if debug: print(f"DEBUG assign_discocat_types (nominal): Assigning type '{assigned_type}' to {pos}/{dep_rel}")
             return assigned_type
        elif pos == "NOUN" and dep_rel in ["nmod", "nsubj", "appos", "root", "obl", "obj", "flat", "acl", "advcl"]:
             assigned_type = n.l >> s
             if debug: print(f"DEBUG assign_discocat_types (nominal): Assigning type '{assigned_type}' to {pos}/{dep_rel}")
             return assigned_type
        elif pos == "ADP":
             assigned_type = n.r @ n @ n.l
             if debug: print(f"DEBUG assign_discocat_types (nominal): Assigning type '{assigned_type}' to {pos}/{dep_rel}")
             return assigned_type
        else:
             # Fallback for nominal cases not explicitly handled above
             assigned_type = n
             if debug: print(f"DEBUG assign_discocat_types (nominal): Assigning fallback type '{assigned_type}' to {pos}/{dep_rel}")
             return assigned_type # Prevent fall-through

    # --- Logic for Verbal Sentences (or if in_nominal=False) ---
    if pos == "VERB":
        type = s
        if has_subject: type = n.r @ type
        if has_object: type = type @ n.l
        assigned_type = type
    elif pos in ["NOUN", "PROPN", "PRON", "DET"]:
        assigned_type = n
    elif pos == "ADJ":
        # Check if it modifies a noun or acts predicatively (less common in verbal)
        assigned_type = n @ n.l # Default: modify noun
    elif pos == "ADV":
        assigned_type = s @ s.l
    elif pos == "ADP":
        assigned_type = n.r @ n @ n.l
    elif pos == "CCONJ":
        assigned_type = s.r @ s @ s.l
    elif pos == "SCONJ":
         assigned_type = s.r @ s
    else:
        # Default fallback type for verbal sentences
        assigned_type = n

    # Final debug print before returning for verbal/fallback cases
    if debug and not in_nominal: print(f"DEBUG assign_discocat_types (verbal/other): Assigning type '{assigned_type}' to {pos}/{dep_rel}")
    # Check specifically for n @ s assignment (which seems problematic)
    if assigned_type == (n @ s):
         if debug: print(f"ALERT: Assigning unusual type 'n @ s' to {pos}/{dep_rel}")

    return assigned_type

def create_discocat_diagram_from_dependencies(tokens, analyses, roles, debug=True):
    """
    Creates a DisCoCat diagram that properly reflects dependency relations.
    
    Args:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        roles: Dictionary with grammatical roles and dependency graph
        debug: Whether to print debug information
        
    Returns:
        diagram: The created DisCoCat diagram
    """
    dependency_graph = roles.get("dependency_graph", {})
    root_idx = roles.get("root")
    
    # First pass: identify which tokens have subject/object dependencies
    has_subject = {i: False for i in range(len(tokens))}
    has_object = {i: False for i in range(len(tokens))}
    
    for head, dependents in dependency_graph.items():
        for dep_idx, dep_rel in dependents:
            if dep_rel in ["nsubj", "csubj"]:
                has_subject[head] = True
            elif dep_rel in ["obj", "iobj", "dobj"]:
                has_object[head] = True
    
    # Create word boxes with appropriate types
    word_boxes = {}
    for i, (token, (_, pos, dep_rel, _)) in enumerate(zip(tokens, analyses)):
        # Assign DisCoCat types based on linguistic properties
        output_type = assign_discocat_types(pos, dep_rel, has_subject[i], has_object[i])
        word_boxes[i] = Box(token, Ty(), output_type)
    
    if debug:
        print("\nCreated DisCoCat boxes:")
        for i, box in word_boxes.items():
            print(f"{i}: {tokens[i]} - Type: {box.cod}")
    
    # Handle VSO and SVO structures differently
    structure = roles.get("structure", "OTHER")
    verb_idx = roles.get("verb")
    subj_idx = roles.get("subject")
    obj_idx = roles.get("object")
    
    try:
        # SIMPLIFIED APPROACH: Create a diagram based on dependency relations directly
        if root_idx is not None:
            # Start with the root as our base diagram
            diagram = word_boxes[root_idx]
            
            # For VSO structure with verb as root
            if structure == "VSO" and verb_idx is not None and verb_idx == root_idx:
                # Start with the verb
                diagram = word_boxes[verb_idx]
                
                # If there's a subject, add a simple connection
                if subj_idx is not None:
                    # Here we'll use a simplified approach that avoids complex tensor networks
                    subject_box = word_boxes[subj_idx]
                    # For VSO, we'll just tensor the verb and subject for now
                    diagram = diagram @ subject_box
                
                # If there's an object, add it too
                if obj_idx is not None:
                    object_box = word_boxes[obj_idx]
                    diagram = diagram @ object_box
                
            # For SVO structure with verb as root
            elif structure == "SVO" and verb_idx is not None and verb_idx == root_idx:
                # If there's a subject, start with it
                if subj_idx is not None:
                    diagram = word_boxes[subj_idx]
                    
                    # Add the verb
                    verb_box = word_boxes[verb_idx]
                    diagram = diagram @ verb_box
                    
                    # If there's an object, add it too
                    if obj_idx is not None:
                        object_box = word_boxes[obj_idx]
                        diagram = diagram @ object_box
                else:
                    # If no subject, start with the verb
                    diagram = word_boxes[verb_idx]
                    
                    # If there's an object, add it
                    if obj_idx is not None:
                        object_box = word_boxes[obj_idx]
                        diagram = diagram @ object_box
            
            # For nominal sentences or other structures
            elif structure == "NOMINAL" or (structure == "OTHER" and verb_idx is None):
                # Check if we have an adjective modifying a noun
                if root_idx in dependency_graph:
                    modifiers = []
                    
                    # Find all modifiers (amod, nmod, etc.)
                    for dep_idx, dep_rel in dependency_graph[root_idx]:
                        if dep_rel in ["amod", "nmod", "nummod", "advmod"]:
                            modifiers.append((dep_idx, dep_rel))
                    
                    # Add all modifiers as tensored elements
                    for mod_idx, _ in modifiers:
                        diagram = diagram @ word_boxes[mod_idx]
            
            # Make sure the diagram outputs a sentence type for consistency
            if diagram.cod != S:
                # Create a proper type conversion
                conversion_box = Box(f"convert_to_S", diagram.cod, S)
                diagram = diagram >> conversion_box
                
        else:
            # No root found, create a simple diagram from available boxes
            indices = sorted(word_boxes.keys())
            if indices:
                diagram = word_boxes[indices[0]]
                for idx in indices[1:]:
                    diagram = diagram @ word_boxes[idx]
                    
                # Convert to sentence type
                if diagram.cod != S:
                    conversion_box = Box(f"convert_to_S", diagram.cod, S)
                    diagram = diagram >> conversion_box
            else:
                # Empty diagram fallback
                diagram = Id(S)
        
    except Exception as e:
        if debug:
            print(f"Error in diagram construction: {e}")
        
        # Fallback to simpler, non-connected diagram
        try:
            # Just create a simple tensor product of all boxes
            indices = sorted(word_boxes.keys())
            if indices:
                diagram = word_boxes[indices[0]]
                for idx in indices[1:]:
                    diagram = diagram @ word_boxes[idx]
            else:
                diagram = Id(S)
                
            # Add a final identity to ensure S type output
            if diagram.cod != S:
                diagram = diagram >> Box("convert_to_S", diagram.cod, S)
                
        except Exception as e2:
            if debug:
                print(f"Fallback also failed: {e2}")
            # Last resort - minimal diagram
            diagram = Id(S)
    
    return diagram

# Add this function to improve handling of complex sentences with subordinate clauses
def create_enhanced_discocat_diagram(tokens, analyses, roles, debug=True):
    """
    Creates an enhanced DisCoCat diagram that can handle subordinate clauses.
    
    Args:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        roles: Dictionary with grammatical roles and dependency graph
        debug: Whether to print debug information
        
    Returns:
        diagram: The created DisCoCat diagram
    """
    dependency_graph = roles.get("dependency_graph", {})
    root_idx = roles.get("root")
    
    # First identify clauses and their boundaries
    clauses = identify_clauses(tokens, analyses, dependency_graph, debug)
    
    if debug:
        print("\nIdentified clauses:")
        for i, clause in enumerate(clauses):
            print(f"Clause {i}: tokens {clause['start']}-{clause['end']}, type: {clause['type']}")
    
    # Create word boxes with appropriate types
    word_boxes = {}
    for i, (token, (_, pos, dep_rel, _)) in enumerate(zip(tokens, analyses)):
        # Check if this token has subjects or objects
        has_subject = any(dep_idx for dep_idx, dep_type in dependency_graph.get(i, []) 
                         if dep_type in ["nsubj", "csubj"])
        has_object = any(dep_idx for dep_idx, dep_type in dependency_graph.get(i, []) 
                        if dep_type in ["obj", "iobj", "dobj"])
        
        # Assign DisCoCat types based on linguistic properties
        output_type = assign_discocat_types(pos, dep_rel, has_subject, has_object)
        word_boxes[i] = Box(token, Ty(), output_type)
    
    if debug:
        print("\nCreated DisCoCat boxes:")
        for i, box in word_boxes.items():
            print(f"{i}: {tokens[i]} - Type: {box.cod}")
    
    # Create diagrams for each clause
    clause_diagrams = []
    for clause in clauses:
        clause_tokens = tokens[clause['start']:clause['end']+1]
        clause_diagram = create_clause_diagram(
            clause, word_boxes, roles, dependency_graph, debug)
        clause_diagrams.append((clause, clause_diagram))
    
    # Connect the clause diagrams
    if clause_diagrams:
        # Start with the main clause (usually containing the root)
        main_clause = None
        for clause, diagram in clause_diagrams:
            if root_idx is not None and clause['start'] <= root_idx <= clause['end']:
                main_clause = (clause, diagram)
                break
        
        if main_clause is None:
            main_clause = clause_diagrams[0]
        
        final_diagram = main_clause[1]
        
        # Connect subordinate clauses
        for clause, diagram in clause_diagrams:
            if clause != main_clause[0]:
                # Find the connection point (usually a complementizer or conjunction)
                connection_idx = clause.get('connector')
                if connection_idx is not None:
                    # We'll use a simple tensor product to connect clauses for now
                    # In a more advanced version, we could use proper diagram composition
                    final_diagram = final_diagram @ diagram
                else:
                    # If no clear connection, just tensor the diagrams
                    final_diagram = final_diagram @ diagram
        
        # Make sure the diagram outputs a sentence type
        if final_diagram.cod != AtomicType.SENTENCE:
            # Create a proper type conversion
            conversion_box = Box(f"convert_to_S", final_diagram.cod, AtomicType.SENTENCE)
            final_diagram = final_diagram >> conversion_box
            
        return final_diagram
    
    # Fallback to the original method if clause identification fails
    return create_discocat_diagram_from_dependencies(tokens, analyses, roles, debug)

def identify_clauses(tokens, analyses, dependency_graph, debug=True):
    """
    Identify clauses in the sentence.
    
    Args:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        dependency_graph: The dependency graph
        debug: Whether to print debug information
        
    Returns:
        clauses: List of clause dictionaries with start/end indices and type
    """
    clauses = []
    
    # Find the root
    root_idx = None
    for i, (_, _, dep, _) in enumerate(analyses):
        if dep == "root":
            root_idx = i
            break
    
    if root_idx is not None:
        # The main clause containing the root
        main_clause = {
            'start': 0,
            'end': len(tokens) - 1,
            'type': 'main',
            'head': root_idx
        }
        
        # Find subordinate clauses by looking for complementizers and conjunctions
        subordinate_starts = []
        for i, (_, pos, dep, head) in enumerate(analyses):
            if (dep in ["mark", "cc", "conj"] or 
                pos in ["SCONJ", "CCONJ"] or 
                # Check for Arabic 'because' (لأن)
                (tokens[i].startswith('لأن') or tokens[i] == 'ل' and i+1 < len(tokens) and tokens[i+1] == 'أن')):
                
                # Identify the boundaries of the subordinate clause
                # For simplicity, we'll assume it extends from the marker to the end of the sentence
                # or until another marker is found
                clause_start = i
                clause_end = len(tokens) - 1
                
                # Find next marker if any
                for j in range(i+1, len(tokens)):
                    if (analyses[j][2] in ["mark", "cc", "conj"] or 
                        analyses[j][1] in ["SCONJ", "CCONJ"]):
                        clause_end = j - 1
                        break
                
                subordinate_starts.append((clause_start, clause_end))
        
        # Sort by start position
        subordinate_starts.sort()
        
        # Adjust main clause boundaries
        if subordinate_starts:
            main_clause['end'] = subordinate_starts[0][0] - 1
        
        # Add all clauses
        clauses.append(main_clause)
        
        for i, (start, end) in enumerate(subordinate_starts):
            connector = None
            if analyses[start][1] in ["SCONJ", "CCONJ"] or analyses[start][2] in ["mark", "cc", "conj"]:
                connector = start
            
            clauses.append({
                'start': start,
                'end': end,
                'type': 'subordinate',
                'connector': connector
            })
    else:
        # If no root, treat the whole sentence as one clause
        clauses.append({
            'start': 0,
            'end': len(tokens) - 1,
            'type': 'unknown'
        })
    
    # Adjust clause boundaries to make sure they don't overlap
    clauses.sort(key=lambda x: x['start'])
    for i in range(1, len(clauses)):
        if clauses[i]['start'] <= clauses[i-1]['end']:
            clauses[i]['start'] = clauses[i-1]['end'] + 1
    
    # Remove invalid clauses (start > end)
    clauses = [c for c in clauses if c['start'] <= c['end']]
    
    return clauses

def create_clause_diagram(clause, word_boxes, roles, dependency_graph, debug=True):
    """
    Create a DisCoCat diagram for a specific clause.
    
    Args:
        clause: Clause dictionary with start/end indices
        word_boxes: Dictionary of word boxes
        roles: Dictionary with grammatical roles
        dependency_graph: The dependency graph
        debug: Whether to print debug information
        
    Returns:
        diagram: The created DisCoCat diagram for the clause
    """
    start, end = clause['start'], clause['end']
    clause_indices = list(range(start, end + 1))
    
    # For very short clauses, just tensor the boxes
    if len(clause_indices) <= 2:
        if clause_indices:
            diagram = word_boxes[clause_indices[0]]
            for idx in clause_indices[1:]:
                diagram = diagram @ word_boxes[idx]
            return diagram
        return Id(AtomicType.SENTENCE)  # Empty clause fallback
    
    # Find the head of the clause
    clause_head = clause.get('head')
    
    # If no explicit head, try to find a verb or noun that could be the head
    if clause_head is None or clause_head < start or clause_head > end:
        for idx in clause_indices:
            if idx in dependency_graph and len(dependency_graph[idx]) > 0:
                clause_head = idx
                break
    
    # If still no head, use the first element
    if clause_head is None or clause_head < start or clause_head > end:
        clause_head = clause_indices[0]
    
    # Create the diagram starting with the head
    if clause_head in word_boxes:
        diagram = word_boxes[clause_head]
        
        # Add dependents as tensor products
        added_indices = {clause_head}
        
        # First add direct dependents of the head
        if clause_head in dependency_graph:
            for dep_idx, dep_rel in dependency_graph[clause_head]:
                if start <= dep_idx <= end and dep_idx not in added_indices:
                    diagram = diagram @ word_boxes[dep_idx]
                    added_indices.add(dep_idx)
        
        # Then add any remaining tokens
        for idx in clause_indices:
            if idx not in added_indices:
                diagram = diagram @ word_boxes[idx]
                added_indices.add(idx)
    else:
        # Fallback if head box doesn't exist
        if clause_indices:
            diagram = word_boxes[clause_indices[0]]
            for idx in clause_indices[1:]:
                if idx in word_boxes:
                    diagram = diagram @ word_boxes[idx]
        else:
            diagram = Id(AtomicType.SENTENCE)
    
    return diagram

def detect_arabic_nominal_features(tokens, analyses):
    """Detect special features in Arabic nominal sentences"""
    features = {
        "idafa": False,
        "demonstrative": False,
        "negation": False
    }
    
    # Look for idafa construction (possessive relationship)
    for i, (_, pos, dep, _) in enumerate(analyses):
        if dep == "nmod:poss" or (dep == "nmod" and i > 0):
            features["idafa"] = True
            break
    
    # Look for demonstratives (هذا, ذلك, etc.)
    for token in tokens:
        if token in ["هذا", "هذه", "ذلك", "تلك", "هؤلاء", "أولئك"]:
            features["demonstrative"] = True
            break
    
    # Look for negation (ليس, لا)
    for token in tokens:
        if token in ["ليس", "ليست", "لا", "ما", "غير"]:
            features["negation"] = True
            break
            
    return features

def create_nominal_sentence_diagram(tokens, analyses, roles, debug=True):
    """
    Create DisCoCat diagram specifically for nominal sentences.
    Attempts Cup composition with the FIRST valid predicate found.
    Falls back to simpler structures if composition fails.

    Args:
        tokens: List of tokens
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        roles: Dictionary with grammatical roles
        debug: Whether to print debug information

    Returns:
        diagram: DisCoCat diagram for the nominal sentence
    """
    if debug:
        print("\nAttempting to create diagram for NOMINAL sentence...")

    dependency_graph = roles.get("dependency_graph", {})
    root_idx = roles.get("root")

    # Create word boxes with types specifically for nominal sentence
    word_boxes = {}
    for i, (token, (_, pos, dep_rel, _)) in enumerate(zip(tokens, analyses)):
        output_type = assign_discocat_types(pos, dep_rel,
                                          has_subject=False,
                                          has_object=False,
                                          in_nominal=True)
        word_boxes[i] = Box(token, Ty(), output_type)
        if debug:
             print(f"  Nominal Box {i}: {token} ({pos}, {dep_rel}) -> Type: {output_type}")

    # Identify subject (usually the root) and predicate(s)
    subject_idx = None
    predicate_indices = []

    if root_idx is not None and root_idx < len(analyses): # Check index bounds
        if analyses[root_idx][1] in ["NOUN", "PROPN", "PRON"]:
             subject_idx = root_idx
             if root_idx in dependency_graph:
                  for dep_idx, dep_rel in dependency_graph[root_idx]:
                       if dep_idx < len(analyses): # Check index bounds
                            if analyses[dep_idx][1] in ["ADJ", "NOUN"] and dep_rel in ["amod", "nmod", "appos", "nsubj", "xcomp", "root", "obl", "acl", "advcl"]:
                                 predicate_indices.append(dep_idx)
                            elif dep_rel == "cop":
                                 predicate_indices.append(dep_idx)
        else:
             for i, (_, _, dep, head) in enumerate(analyses):
                  if head == root_idx and dep == "nsubj":
                       subject_idx = i
                       predicate_indices = [root_idx]
                       break

    if debug:
         print(f"  Nominal Subject Index: {subject_idx}")
         print(f"  Nominal Predicate Indices: {predicate_indices}")

    # --- Modified Composition Logic: Try Cup with First Predicate ---
    diagram = None
    if subject_idx is not None and subject_idx in word_boxes:
        current_subject_diagram = word_boxes[subject_idx]
        subj_cod = current_subject_diagram.cod # Get subject codomain type

        # Filter predicate indices to ensure they exist in word_boxes
        valid_predicate_indices = [idx for idx in predicate_indices if idx in word_boxes]

        applied_cup_successfully = False
        if valid_predicate_indices:
            try:
                # --- Try applying only the FIRST valid predicate using Cup ---
                first_pred_idx = valid_predicate_indices[0]
                predicate_box = word_boxes[first_pred_idx]
                pred_type = predicate_box.cod # Type of the first predicate box

                # Check if pred_type is functional (n.l >> s) and matches subject (n)
                is_functional_type = hasattr(pred_type, 'input') and hasattr(pred_type, 'output')
                types_match_for_cup = (subj_cod == AtomicType.NOUN and
                                       is_functional_type and
                                       pred_type.input == AtomicType.NOUN.l and
                                       pred_type.output == AtomicType.SENTENCE)

                if types_match_for_cup:
                    if debug: print(f"  Applying Cup composition with first predicate: Subj={subj_cod}, Pred={pred_type}")
                    diagram = current_subject_diagram @ predicate_box >> Cup(AtomicType.NOUN, AtomicType.NOUN.l) >> Id(AtomicType.SENTENCE)
                    applied_cup_successfully = True
                else:
                    # Type mismatch or predicate is not functional type n.l >> s
                    print(f"WARNING: Type mismatch or non-functional type for first predicate Cup composition.")
                    print(f"         Subject cod: {subj_cod}")
                    pred_type_repr = f"Type={type(pred_type)}"
                    if is_functional_type:
                         pred_type_repr = f"Input={pred_type.input}, Output={pred_type.output}"
                    else:
                         pred_type_repr = f"Type={pred_type}" # Show simple type like 'n'
                    print(f"         First Predicate diagram type info: {pred_type_repr}")
                    print(f"         Skipping Cup composition.")
                    # If Cup fails for the first predicate, fall back to just the subject for now
                    diagram = current_subject_diagram

            except ValueError as e_comp: # Cup composition failed
                 print(f"ERROR: Nominal Cup composition failed for first predicate: {e_comp}")
                 diagram = current_subject_diagram # Fallback to just subject
            except Exception as e_other: # Other unexpected errors
                 print(f"ERROR: Unexpected error during nominal composition: {e_other}")
                 traceback.print_exc()
                 diagram = current_subject_diagram # Fallback to just subject

            # --- Handle remaining predicates (if any) ---
            # If Cup was successful with the first predicate, we might ignore others for simplicity
            # Or attempt to tensor them onto the resulting sentence diagram (less ideal)
            if len(valid_predicate_indices) > 1:
                 if applied_cup_successfully:
                      print(f"WARNING: Multiple predicates found ({len(valid_predicate_indices)}). Only composed the first one via Cup. Diagram may be incomplete.")
                      # Optionally, tensor remaining predicates (treat as sentence modifiers?)
                      # for p_idx in valid_predicate_indices[1:]:
                      #     diagram = diagram @ word_boxes[p_idx] # This might create complex types again
                 else:
                      print(f"WARNING: Multiple predicates found but Cup composition failed/skipped for the first. Falling back.")
                      # Fallback: Tensor subject with all predicates if Cup failed initially
                      print(f"         Falling back to tensor product of subject and all predicates.")
                      diagram = current_subject_diagram
                      for p_idx in valid_predicate_indices:
                           diagram = diagram @ word_boxes[p_idx]

        else: # No valid predicates found
             if debug: print("  No valid predicates found, using only subject diagram.")
             diagram = current_subject_diagram # Only subject

    # --- Fallback if subject/predicate logic failed or wasn't applicable ---
    if diagram is None:
        if debug: print("  Nominal subject/predicate logic failed entirely, using fallback tensor.")
        # Tensor all boxes in the sentence order
        indices = sorted(word_boxes.keys())
        if indices:
            try:
                 diagram = word_boxes[indices[0]]
                 for idx in indices[1:]:
                      diagram = diagram @ word_boxes[idx]
            except Exception as e_tensor:
                 print(f"ERROR: Fallback tensor composition failed: {e_tensor}")
                 diagram = Id(AtomicType.SENTENCE) # Final fallback
        else:
            diagram = Id(AtomicType.SENTENCE) # Empty sentence

    # --- Final Check: Ensure output type is Sentence ---
    # Check if diagram exists and has a codomain attribute
    if diagram is None or not hasattr(diagram, 'cod'):
         print(f"ERROR: Diagram is None or invalid before final type check. Falling back to Id(S).")
         diagram = Id(AtomicType.SENTENCE)
    elif diagram.cod != AtomicType.SENTENCE:
        current_cod = diagram.cod
        if debug: print(f"  Final nominal diagram cod is {current_cod}. Attempting conversion to S.")
        try:
            # Use a generic conversion box
            conversion_box = Box(f"Force_to_S", diagram.cod, AtomicType.SENTENCE)
            diagram = diagram >> conversion_box
            if debug: print(f"  Conversion successful. New cod: {diagram.cod}")
        except Exception as e_conv:
            print(f"ERROR: Could not convert final nominal diagram to S type: {e_conv}. Falling back to Id(S).")
            diagram = Id(AtomicType.SENTENCE) # Last resort fallback

    if debug: print(f"Final nominal diagram: {diagram}")
    return diagram

def is_nominal_sentence(analyses, roles):
    """
    Determine if a sentence is nominal (has no main verb).
    
    Args:
        analyses: List of (lemma, pos, dep_type, head_idx) tuples
        roles: Dictionary with grammatical roles
        
    Returns:
        bool: True if the sentence is nominal
    """
    # Check if there's no verb at all
    has_verb = any(pos == "VERB" for _, pos, _, _ in analyses)
    
    # Check if the root is a noun or adjective
    root_idx = roles.get("root")
    if root_idx is not None and root_idx < len(analyses):
        _, root_pos, _, _ = analyses[root_idx]
        if root_pos in ["NOUN", "PROPN", "ADJ"] and not has_verb:
            return True
    
    # Check if structure is already detected as NOMINAL
    if roles.get("structure") == "NOMINAL":
        return True
        
    return False



# Modify the main function to use our enhanced diagram creation
def arabic_to_quantum_enhanced(sentence, debug=True):
    """
    Processes an Arabic sentence, creates a DisCoCat diagram, and converts
    it to a Qiskit QuantumCircuit using appropriate methods. Includes
    attempt to use lambeq backend for compilation.

    Args:
        sentence (str): The Arabic sentence.
        debug (bool): Whether to print debugging information.

    Returns:
        tuple: (circuit, diagram, structure, tokens, analyses, roles)
               circuit is a qiskit.QuantumCircuit or None on failure.
               diagram is a lambeq GrammarDiagram or None.
    """
    # 1. Analyze sentence
    tokens, analyses, structure, roles = analyze_arabic_sentence(sentence, debug)
    roles["structure"] = structure # Ensure structure is in roles dict

    # 2. Check if nominal and create diagram
    diagram = None
    try:
        is_nominal = is_nominal_sentence(analyses, roles)
        if is_nominal:
            roles["structure"] = "NOMINAL" # Update structure if determined nominal
            if debug: print("\nDetected NOMINAL sentence structure")
            diagram = create_nominal_sentence_diagram(tokens, analyses, roles, debug)
        else:
            if debug: print("\nDetected VERBAL sentence structure")
            # Use enhanced diagram creation for potentially complex verbal sentences
            if 'create_enhanced_discocat_diagram' in globals():
                 diagram = create_enhanced_discocat_diagram(tokens, analyses, roles, debug)
            else: # Fallback if enhanced function is missing
                 print("WARNING: create_enhanced_discocat_diagram not found, using basic dependency diagram.")
                 # Ensure create_discocat_diagram_from_dependencies exists if used here
                 if 'create_discocat_diagram_from_dependencies' in globals():
                      diagram = create_discocat_diagram_from_dependencies(tokens, analyses, roles, debug)
                 else:
                      print("ERROR: Basic diagram function create_discocat_diagram_from_dependencies not found.")
                      raise RuntimeError("Missing basic diagram creation function.")


        if diagram is None:
            raise ValueError("Diagram creation returned None.")
        if not isinstance(diagram, GrammarDiagram):
             if hasattr(diagram, 'cod') and hasattr(diagram, 'dom'):
                  print(f"Warning: Diagram created is type {type(diagram)}, attempting to use.")
             else:
                  raise TypeError(f"Diagram created is not a valid lambeq Diagram object: {type(diagram)}")

        if debug:
            print(f"\nDEBUG: Diagram created successfully. Type: {type(diagram)}")

    except Exception as e_diagram:
        print(f"ERROR: Exception during diagram creation for sentence: '{sentence}'")
        print(f"       Error: {e_diagram}")
        traceback.print_exc()
        return None, None, structure, tokens, analyses, roles

    # 3. Convert diagram to quantum circuit
    circuit = None
    try:
        # Define the object map explicitly for N and S types.
        N_ = AtomicType.NOUN
        S_ = AtomicType.SENTENCE
        ob_map = {N_: 1, S_: 1}
        ansatz = IQPAnsatz(ob_map=ob_map, n_layers=1, n_single_qubit_params=3)

        # Add debug print for diagram type before ansatz
        if debug: print(f"\nDEBUG: Final diagram before ansatz: dom={getattr(diagram, 'dom', 'N/A')}, cod={getattr(diagram, 'cod', 'N/A')}")

        if debug: print("DEBUG: Simplifying diagram...")
        simplified_diagram = diagram.normal_form()

        if debug: print("DEBUG: Applying ansatz...")
        quantum_diagram = ansatz(simplified_diagram)
        if debug: print(f"DEBUG: Ansatz applied. Quantum diagram type: {type(quantum_diagram)}")

        # --- Attempt Conversion using different methods ---
        # Method 1: Try specific Lambeq QuantumCircuit backend conversion first
        try:
            from lambeq.backend.quantum import QuantumCircuit as LambeqCircuit # Try specific import
            if isinstance(quantum_diagram, LambeqCircuit):
                 circuit = quantum_diagram.to_qiskit() # Convert backend circuit to qiskit
                 if debug: print("DEBUG: Converted LambeqCircuit to Qiskit circuit.")
        except ImportError:
             if debug: print("DEBUG: lambeq.backend.quantum.QuantumCircuit not found.")
        except Exception as e_lqc:
             if debug: print(f"DEBUG: Conversion from LambeqCircuit failed: {e_lqc}")
             circuit = None # Reset circuit if conversion failed

        # Method 2: Try direct .to_qiskit() if Method 1 didn't work or apply
        if circuit is None and hasattr(quantum_diagram, 'to_qiskit') and callable(quantum_diagram.to_qiskit):
            if debug: print("DEBUG: Attempting conversion using quantum_diagram.to_qiskit()...")
            try:
                circuit = quantum_diagram.to_qiskit()
                if debug: print(f"DEBUG: Conversion via .to_qiskit() successful.")
            except Exception as e_tq:
                print(f"DEBUG: quantum_diagram.to_qiskit() failed: {e_tq}")
                circuit = None

        # Method 3: Try compilation using QiskitBackend if still no circuit
        if circuit is None:
             print(f"DEBUG: Standard conversion methods failed for type {type(quantum_diagram)}. Trying compilation.")
             QiskitBackend = None # Initialize to None
             try:
                  # Try importing from standard location first
                  from lambeq.backend.quantum import QiskitBackend
                  if debug: print("DEBUG: Found QiskitBackend in lambeq.backend.quantum")
             except ImportError:
                  try:
                       # Try alternative location if first failed
                       from lambeq.backend.qiskit import QiskitBackend # <<< TRY THIS PATH
                       if debug: print("DEBUG: Found QiskitBackend in lambeq.backend.qiskit")
                  except ImportError:
                       print("ERROR: QiskitBackend not found in lambeq.backend.quantum or lambeq.backend.qiskit. Cannot compile.")
                       # Keep QiskitBackend as None

             if QiskitBackend: # Proceed only if backend class was found
                  try:
                       backend = QiskitBackend()
                       # Ensure the input to compile is the quantum_diagram from the ansatz
                       circuit = backend.compile(quantum_diagram)
                       if debug: print("DEBUG: Compiled using QiskitBackend.")
                  except Exception as e_compile:
                       print(f"ERROR: Compilation with QiskitBackend failed: {e_compile}")
                       traceback.print_exc()
                       circuit = None
             else:
                  circuit = None # Cannot compile if backend wasn't found

    except Exception as e_circuit_outer:
        print(f"ERROR: Exception during circuit conversion setup for sentence: '{sentence}'")
        print(f"       Error: {e_circuit_outer}")
        traceback.print_exc()
        circuit = None # Ensure circuit is None

    # --- Handle failure ---
    if circuit is None:
        print("WARNING: Failed to convert lambeq diagram to Qiskit circuit.")
        # Return None for circuit, but keep the diagram if it was created
        return None, diagram, structure, tokens, analyses, roles

    # Final check: Ensure we have a Qiskit circuit object
    if not isinstance(circuit, QuantumCircuit):
         print(f"ERROR: Conversion resulted in unexpected type {type(circuit)}. Expected qiskit.QuantumCircuit. Returning None.")
         return None, diagram, structure, tokens, analyses, roles

    if debug: print(f"DEBUG: Circuit creation successful. Type: {type(circuit)}")
    return circuit, diagram, structure, tokens, analyses, roles


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

# Run the tests
if __name__ == "__main__":
    # Test the examples from before
    results = test_examples()
    
    # Test our complex sentence
    complex_result = test_complex_sentence()
    
    print("\nSummary of Results:")
    print("==================")
    for result in results:
        print(f"{result['description']}: {result['sentence']}")
        print(f"  - Detected structure: {result['structure']}")
        print(f"  - Tokens: {' '.join(result['tokens'])}")
        print()
    
    print(f"Complex sentence: {complex_result['sentence']}")
    print(f"  - Detected structure: {complex_result['structure']}")
    print(f"  - Tokens: {' '.join(complex_result['tokens'])}")