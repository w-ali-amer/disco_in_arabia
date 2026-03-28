# lambeq_minimal_test.py
from lambeq import AtomicType
from lambeq.backend.grammar import Diagram, Box, Id, Ty
from typing import List, Dict, Optional
import traceback
# Define atomic types
# phrase_composer.py
# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

def create_svo_diagram(subject_word: str, verb_word: str, object_word: str) -> Optional[Diagram]:
    """
    Creates a simple Subject-Verb-Object diagram with consistent N types.
    - Subject word box: Ty() -> N
    - Object word box: Ty() -> N
    - Verb functor: N @ N -> S (first N is subject, second N is object)
    """
    try:
        # 1. Lexical entries (word boxes outputting their basic types)
        subj_box = Box(subject_word, Ty(), N)
        obj_box = Box(object_word, Ty(), N)
        
        # Verb functor: Takes two Nouns (Subject then Object) and produces S
        # The roles (subject, object) are determined by the order in the domain.
        verb_functor = Box(verb_word, N @ N, S) 

        print(f"Subject Box ('{subject_word}'): {subj_box.cod}")
        print(f"Verb Functor ('{verb_word}'): dom={verb_functor.dom}, cod={verb_functor.cod}")
        print(f"Object Box ('{object_word}'): {obj_box.cod}")

        # 2. Form the input for the verb functor: Subj @ Obj
        # (subj_box: Ty() -> N) @ (obj_box: Ty() -> N)
        # This combined diagram has dom: Ty() @ Ty() (simplifies to Ty())
        # And cod: N @ N
        subject_object_pair = subj_box @ obj_box
        print(f"Subject-Object Pair Diagram: dom={subject_object_pair.dom}, cod={subject_object_pair.cod}")

        # 3. Compose the subject-object pair with the verb functor
        # (subject_object_pair: Ty() -> N @ N) >> (verb_functor: N @ N -> S)
        # This is valid because cod(subject_object_pair) == dom(verb_functor)
        final_diag = subject_object_pair >> verb_functor
        
        print(f"Final SVO Diagram: {final_diag}")
        print(f"Final SVO DOM: {final_diag.dom}") # Should be Ty()
        print(f"Final SVO COD: {final_diag.cod}") # Should be S

        # Check connectivity
        final_diag.normal_form() 
        print("SVO Diagram created and is connected.")
        return final_diag

    except Exception as e:
        print(f"Error creating SVO diagram for '{subject_word} {verb_word} {object_word}': {e}")
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Test SVO
    subject = "الولد"  
    verb = "يقرأ"    
    object_ = "الكتاب" 

    svo_diagram = create_svo_diagram(subject, verb, object_)

    if svo_diagram:
        svo_diagram.draw(path="svo_diagram_corrected.png", show_types=True, figsize=(8,6))
        print("SVO diagram drawn to svo_diagram_corrected.png")

    print("\n--- Test with Intransitive Verb (Subject-Verb) ---")
    # SV: Subject (N), Verb (N -> S)
    verb_intrans_word = "نام" 
    # Intransitive verb functor: Takes one N (Subject) and produces S
    verb_intrans_functor = Box(verb_intrans_word, N, S) 
    
    subj_box_sv = Box(subject, Ty(), N) # Output: N
    
    # Compose subject with intransitive verb functor
    # (subj_box_sv: Ty() -> N) >> (verb_intrans_functor: N -> S)
    sv_diag = subj_box_sv >> verb_intrans_functor
    print(f"SV Diagram: {sv_diag}")
    print(f"SV DOM: {sv_diag.dom}, COD: {sv_diag.cod}")
    try:
        sv_diag.normal_form()
        sv_diag.draw(path="sv_diagram_corrected.png", show_types=True, figsize=(6,4))
        print("SV diagram connected and drawn to sv_diagram_corrected.png")
    except Exception as e_sv:
        print(f"SV diagram error: {e_sv}")
        traceback.print_exc()

    print("\n--- Test Nominal Sentence (Subject-Predicate) ---")
    # Subject (N), Predicate (Adjective as N -> S)
    # (e.g., "الولد كبير" - The boy is big)
    predicate_word = "كبير" 
    # Adjective as predicate functor: Takes N (Subject) and produces S
    adj_predicate_functor = Box(predicate_word, N, S) 
    
    subj_box_nom = Box(subject, Ty(), N) # Output: N

    # Compose subject with adjectival predicate functor
    # (subj_box_nom: Ty() -> N) >> (adj_predicate_functor: N -> S)
    nominal_diag = subj_box_nom >> adj_predicate_functor
    print(f"Nominal Diagram (Subj >> Predicate(N -> S)): {nominal_diag}")
    print(f"Nominal DOM: {nominal_diag.dom}, COD: {nominal_diag.cod}")
    try:
        nominal_diag.normal_form()
        nominal_diag.draw(path="nominal_diagram_corrected.png", show_types=True, figsize=(6,4))
        print("Nominal diagram connected and drawn to nominal_diagram_corrected.png")
    except Exception as e_nom:
        print(f"Nominal diagram error: {e_nom}")
        traceback.print_exc()
