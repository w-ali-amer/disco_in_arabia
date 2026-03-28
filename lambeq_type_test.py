# focused_diagram_creation_test_v3.py
import logging
import traceback
from typing import Optional, List, Dict, Tuple, Any, Set, Sequence # Comprehensive typing imports

# --- Lambeq Imports ---
from lambeq import AtomicType
from lambeq.backend.grammar import Ty, Box, Diagram as GrammarDiagram
from lambeq import IQPAnsatz, SpiderAnsatz, StronglyEntanglingAnsatz 
from lambeq.backend.quantum import Diagram as LambeqQuantumDiagram 

# --- Qiskit Imports ---
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.exceptions import QiskitError
try: from qiskit_aer import AerSimulator; AER_AVAILABLE = True
except ImportError: AerSimulator = None; AER_AVAILABLE = False # type: ignore
try: from qiskit.primitives import Sampler, Estimator; PRIMITIVES_AVAILABLE = True
except ImportError: Sampler, Estimator = None, None; PRIMITIVES_AVAILABLE = False # type: ignore
try: from qiskit.quantum_info import SparsePauliOp, partial_trace; QUANTUM_INFO_AVAILABLE = True
except ImportError: SparsePauliOp, partial_trace = None, None; QUANTUM_INFO_AVAILABLE = False # type: ignore

# --- Pytket Import (Crucial for the fix) ---
try:
    from pytket.extensions.qiskit import tk_to_qiskit
    # Optional: Import Pytket Circuit if needed for type hinting
    # from pytket.circuit import Circuit as PytketCircuit 
    PYTKET_QISKIT_AVAILABLE = True
except ImportError:
    tk_to_qiskit = None # type: ignore
    # PytketCircuit = None # type: ignore
    PYTKET_QISKIT_AVAILABLE = False
    
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Aer available: {AER_AVAILABLE}")
logger.info(f"Primitives available: {PRIMITIVES_AVAILABLE}")
logger.info(f"Quantum Info available: {QUANTUM_INFO_AVAILABLE}")
logger.info(f"Pytket-Qiskit available: {PYTKET_QISKIT_AVAILABLE}")

# --- 1. Define Atomic Types ---
N = AtomicType.NOUN
S = AtomicType.SENTENCE

logger.info(f"N type: {type(N)}, S type: {type(S)}")

# --- 2. Test SVO Diagram Creation ---
def test_svo_diagram(subj_lemma: str, verb_lemma: str, obj_lemma: str):
    logger.info(f"\n--- Testing SVO: {subj_lemma} {verb_lemma} {obj_lemma} ---")
    try:
        subj_box = Box(subj_lemma, Ty(), N)
        verb_box = Box(verb_lemma, N @ N, S) 
        obj_box = Box(obj_lemma, Ty(), N)

        logger.info(f"  Subject Box: {subj_box.name}, {subj_box.dom} -> {subj_box.cod}")
        logger.info(f"  Verb Box: {verb_box.name}, {verb_box.dom} -> {verb_box.cod}")
        logger.info(f"  Object Box: {obj_box.name}, {obj_box.dom} -> {obj_box.cod}")
        
        diagram: GrammarDiagram = (subj_box @ obj_box) >> verb_box 
        logger.info(f"  Composed Diagram: {diagram}")
        
        diagram.normal_form() 
        logger.info("  SVO Diagram created and is connected.")
        
        logger.info("  Attempting circuit conversion (IQPAnsatz) via Pytket...")
        # Check if Pytket converter is available before proceeding
        if not PYTKET_QISKIT_AVAILABLE or tk_to_qiskit is None:
            logger.error("  ERROR: pytket-qiskit extension not found. Cannot perform circuit conversion via Pytket.")
            return diagram # Return the grammar diagram, but no circuit

        try:
            ob_map = {N: 1, S: 1}
            ansatz = IQPAnsatz(ob_map, n_layers=1, n_single_qubit_params=3)
            quantum_diagram: LambeqQuantumDiagram = ansatz(diagram) 
            logger.debug(f"  Type of quantum_diagram: {type(quantum_diagram)}")
            
            # --- Pytket Conversion Steps ---
            logger.debug("  Converting Lambeq QuantumDiagram to Pytket circuit using .to_tk()...")
            tket_circuit = quantum_diagram.to_tk()
            logger.debug(f"  Type of tket_circuit: {type(tket_circuit)}")
            
            logger.debug("  Converting Pytket circuit to Qiskit circuit using tk_to_qiskit()...")
            circuit = tk_to_qiskit(tket_circuit)
            # --- End Pytket Conversion Steps ---

            if isinstance(circuit, QuantumCircuit):
                logger.info(f"  SUCCESS: Circuit conversion via Pytket successful. Qubits: {circuit.num_qubits}")
            else:
                logger.error(f"  ERROR: tk_to_qiskit() did not return a QuantumCircuit. Returned Type: {type(circuit)}")
        except AttributeError as e_attr:
             logger.error(f"  ATTRIBUTE ERROR during circuit conversion: {e_attr}", exc_info=True)
             logger.error("  This might indicate an issue with the .to_tk() or tk_to_qiskit() methods.")
        except Exception as e_circ:
            logger.error(f"  GENERAL ERROR during circuit conversion via Pytket: {e_circ}", exc_info=True)
            
        return diagram

    except Exception as e:
        logger.error(f"Error creating/checking SVO diagram: {e}", exc_info=True)
        return None

# --- 3. Test Nominal Diagram Creation ---
def test_nominal_diagram(subj_lemma: str, pred_adj_lemma: str):
    logger.info(f"\n--- Testing Nominal (Subj-PredAdj): {subj_lemma} {pred_adj_lemma} ---")
    try:
        subj_box = Box(subj_lemma, Ty(), N)
        pred_adj_box = Box(pred_adj_lemma, N, S) 

        logger.info(f"  Subject Box: {subj_box.name}, {subj_box.dom} -> {subj_box.cod}")
        logger.info(f"  Predicate Box: {pred_adj_box.name}, {pred_adj_box.dom} -> {pred_adj_box.cod}")

        diagram: GrammarDiagram = subj_box >> pred_adj_box 
        logger.info(f"  Composed Diagram: {diagram}")
        
        diagram.normal_form()
        logger.info("  Nominal Diagram created and is connected.")

        logger.info("  Attempting circuit conversion (IQPAnsatz) via Pytket...")
        if not PYTKET_QISKIT_AVAILABLE or tk_to_qiskit is None:
            logger.error("  ERROR: pytket-qiskit extension not found. Cannot perform circuit conversion via Pytket.")
            return diagram 

        try:
            ob_map = {N: 1, S: 1}
            ansatz = IQPAnsatz(ob_map, n_layers=1, n_single_qubit_params=3)
            quantum_diagram: LambeqQuantumDiagram = ansatz(diagram)
            logger.debug(f"  Type of quantum_diagram: {type(quantum_diagram)}")
            
            # --- Pytket Conversion Steps ---
            logger.debug("  Converting Lambeq QuantumDiagram to Pytket circuit using .to_tk()...")
            tket_circuit = quantum_diagram.to_tk()
            logger.debug(f"  Type of tket_circuit: {type(tket_circuit)}")
            
            logger.debug("  Converting Pytket circuit to Qiskit circuit using tk_to_qiskit()...")
            circuit = tk_to_qiskit(tket_circuit)
            # --- End Pytket Conversion Steps ---

            if isinstance(circuit, QuantumCircuit):
                logger.info(f"  SUCCESS: Circuit conversion via Pytket successful. Qubits: {circuit.num_qubits}")
            else:
                logger.error(f"  ERROR: tk_to_qiskit() did not return a QuantumCircuit. Returned Type: {type(circuit)}")
        except AttributeError as e_attr:
             logger.error(f"  ATTRIBUTE ERROR during circuit conversion: {e_attr}", exc_info=True)
             logger.error("  This might indicate an issue with the .to_tk() or tk_to_qiskit() methods.")
        except Exception as e_circ:
            logger.error(f"  GENERAL ERROR during circuit conversion via Pytket: {e_circ}", exc_info=True) 

        return diagram

    except Exception as e:
        logger.error(f"Error creating/checking Nominal diagram: {e}", exc_info=True)
        return None

# --- 4. Run Tests ---
if __name__ == "__main__":
    logger.info("==== Running Focused Diagram Creation & Conversion Test (Pytket Method) ====")
    
    # Test SVO
    svo_diagram = test_svo_diagram("الولد", "يقرأ", "الكتاب")
    if svo_diagram:
        logger.info("SVO test completed.")
        
    # Test Nominal
    nominal_diagram = test_nominal_diagram("البيت", "كبير")
    if nominal_diagram:
        logger.info("Nominal test completed.")

    logger.info("\n==== Test Finished ====")
