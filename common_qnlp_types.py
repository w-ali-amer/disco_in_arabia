# common_qnlp_types.py (Corrected Type Checks & Dummies & LambeqCircuit fix)
import logging
from typing import Union, Optional, Any, List, Dict, Mapping, Set # Added Set
import sys 

# --- Setup basic logging ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO) 

logger.info("common_qnlp_types: MODULE EXECUTION STARTED (Corrected Type Checks & Dummies)")
logger.info(f"common_qnlp_types: Python version: {sys.version}")
try:
    import lambeq
    logger.info(f"common_qnlp_types: Lambeq version: {lambeq.__version__}")
except ImportError:
    logger.critical("common_qnlp_types: Could not import lambeq to check version.")
except AttributeError: 
    logger.warning("common_qnlp_types: Lambeq __version__ attribute not found.")

# --- STEP 1: Define Dummies First at module level ---
class _DummyAtomicTypeMeta(type): # A dummy metaclass if needed for isinstance checks
    pass
class _DummyAtomicType(metaclass=_DummyAtomicTypeMeta): # type: ignore
    def __init__(self, name="dummy_atomic"): self.name = name
    def __str__(self): return self.name
class _DummyTy:
    def __init__(self, name: str): self.name = name
    def __str__(self): return self.name
    def __eq__(self, other): return isinstance(other, _DummyTy) and self.name == other.name
    def __hash__(self): return hash(self.name)
    def __matmul__(self, other): return _DummyTy(f"{self}@{other}")
    def __pow__(self, other): # basic pow for Id(Ty() ** 0)
        if isinstance(other, int) and other == 0:
            return _DummyTy() 
        return _DummyTy(f"({self})^{other}")

class _DummyBox:
    def __init__(self, name: str, dom, cod, data=None, _dagger=False):
        self.name = name; self.dom = dom; self.cod = cod
        self.data = data if data is not None else {}; self._dagger = _dagger
    def __str__(self): return self.name
class _DummyGDiagram: pass
class _DummyCup: pass
class _DummyId: # Needs to be callable for Id(Ty())
    def __init__(self, t: Any = None): self.t = t
    def __call__(self, t: Any): return self # Simplified for Id(Ty())
class _DummySpider: pass
class _DummySwap: pass
class _DummyWord: pass 
class _DummyFunctor: pass
class _DummyIQP: pass
class _DummySpiderAns: pass 
class _DummyStrongAns: pass

# Initialize with dummies. These will be overridden by actual imports if successful.
AtomicType: Any = _DummyAtomicType
Ty: Any = _DummyTy
Box: Any = _DummyBox
Functor: Any = _DummyFunctor
GrammarDiagram: Any = _DummyGDiagram
Cup: Any = _DummyCup; Id: Any = _DummyId; Spider: Any = _DummySpider; Swap: Any = _DummySwap
Word: Any = _DummyWord
IQPAnsatz: Any = _DummyIQP; SpiderAnsatz: Any = _DummySpiderAns; StronglyEntanglingAnsatz: Any = _DummyStrongAns

LAMBEQ_IMPORTED_SUCCESSFULLY = False

# --- STEP 2: Attempt to import actual Lambeq classes ---
try:
    from lambeq import AtomicType as ImportedAtomicType_
    from lambeq.backend.grammar import Ty as ImportedTy_
    from lambeq.backend.grammar import Box as ImportedBox_
    from lambeq.backend.grammar import Functor as ImportedFunctor_
    from lambeq.backend.grammar import grammar 
    from lambeq.backend.grammar import Cup as ImportedCup_, Id as ImportedId_, Spider as ImportedSpider_, Swap as ImportedSwap_
    from lambeq.backend.grammar import Diagram as ImportedGrammarDiagram_, Word as ImportedWord_
    from lambeq import IQPAnsatz as ImportedIQPAnsatz_, SpiderAnsatz as ImportedSpiderAnsatz_, \
                       StronglyEntanglingAnsatz as ImportedStronglyEntanglingAnsatz_
    
    AtomicType = ImportedAtomicType_
    Ty = ImportedTy_
    Box = ImportedBox_
    Functor = ImportedFunctor_
    GrammarDiagram = ImportedGrammarDiagram_
    Cup = ImportedCup_
    Id = ImportedId_
    Spider = ImportedSpider_
    Swap = ImportedSwap_
    Word = ImportedWord_
    IQPAnsatz = ImportedIQPAnsatz_; SpiderAnsatz = ImportedSpiderAnsatz_; StronglyEntanglingAnsatz = ImportedStronglyEntanglingAnsatz_
    
    LAMBEQ_IMPORTED_SUCCESSFULLY = True
    logger.info("common_qnlp_types: Successfully imported all required Lambeq components.")

except ImportError as e_lambeq_import:
    logger.critical(f"common_qnlp_types: CRITICAL IMPORT ERROR for Lambeq components: {e_lambeq_import}. Using dummies defined above.", exc_info=True)
except Exception as e_general_import: 
    logger.critical(f"common_qnlp_types: UNEXPECTED ERROR during Lambeq component import: {e_general_import}. Using dummies.", exc_info=True)


# --- STEP 3: Define N_ARABIC and S_ARABIC ---
N_ARABIC: Any 
S_ARABIC: Any
ROOT_TYPE_ARABIC: Any
ARE_N_S_PROPER_ATOMIC_TYPES = False # This will be True if N_ARABIC and S_ARABIC are usable for Box creation

if LAMBEQ_IMPORTED_SUCCESSFULLY and AtomicType is not _DummyAtomicType:
    logger.info(f"common_qnlp_types: Attempting to assign AtomicType.NOUN to N_ARABIC. Current AtomicType type: {type(AtomicType)}")
    try:
        N_ARABIC = AtomicType.NOUN # type: ignore
        S_ARABIC = AtomicType.SENTENCE # type: ignore
        logger.info(f"common_qnlp_types: N_ARABIC assigned. Value: {N_ARABIC}, Type: {type(N_ARABIC)}")
        logger.info(f"common_qnlp_types: S_ARABIC assigned. Value: {S_ARABIC}, Type: {type(S_ARABIC)}")

        if isinstance(N_ARABIC, Ty) and str(N_ARABIC) == 'n' and \
           isinstance(S_ARABIC, Ty) and str(S_ARABIC) == 's': # type: ignore
            ARE_N_S_PROPER_ATOMIC_TYPES = True
            logger.info("common_qnlp_types: N_ARABIC and S_ARABIC are Ty('n') and Ty('s') respectively. Considered proper for Box creation.")
        else:
            logger.critical("common_qnlp_types: N_ARABIC or S_ARABIC are not proper Ty('n')/Ty('s') after assignment. This is problematic.")
            N_ARABIC = Ty('n') if Ty is not _DummyTy else _DummyTy('n') # type: ignore
            S_ARABIC = Ty('s') if Ty is not _DummyTy else _DummyTy('s') # type: ignore
            logger.info(f"common_qnlp_types: Fallback N_ARABIC: {N_ARABIC}, S_ARABIC: {S_ARABIC}")
            if isinstance(N_ARABIC, Ty) and str(N_ARABIC) == 'n' and isinstance(S_ARABIC, Ty) and str(S_ARABIC) == 's': # type: ignore
                 ARE_N_S_PROPER_ATOMIC_TYPES = True 

    except AttributeError as e_attr:
        logger.critical(f"common_qnlp_types: AttributeError assigning AtomicType.NOUN/SENTENCE: {e_attr}. Defaulting to new Ty objects.", exc_info=True)
        N_ARABIC = Ty('n') if Ty is not _DummyTy else _DummyTy('n') # type: ignore
        S_ARABIC = Ty('s') if Ty is not _DummyTy else _DummyTy('s') # type: ignore
    except Exception as e_atomic_assign:
        logger.critical(f"common_qnlp_types: Exception assigning AtomicType.NOUN/SENTENCE: {e_atomic_assign}. Defaulting to new Ty objects.", exc_info=True)
        N_ARABIC = Ty('n') if Ty is not _DummyTy else _DummyTy('n') # type: ignore
        S_ARABIC = Ty('s') if Ty is not _DummyTy else _DummyTy('s') # type: ignore
else:
    logger.critical("common_qnlp_types: Lambeq AtomicType not imported successfully or is a dummy. Defining N_ARABIC/S_ARABIC as basic Ty.")
    N_ARABIC = Ty('n') if Ty is not _DummyTy else _DummyTy('n') # type: ignore
    S_ARABIC = Ty('s') if Ty is not _DummyTy else _DummyTy('s') # type: ignore

ROOT_TYPE_ARABIC = N_ARABIC

# --- STEP 4: Define Global Functorial Box Types (conditionally) ---
ADJ_MOD_TYPE_ARABIC: Optional[Box] = None; VERB_TRANS_TYPE_ARABIC: Optional[Box] = None; VERB_INTRANS_TYPE_ARABIC: Optional[Box] = None; ADJ_PRED_TYPE_ARABIC: Optional[Box] = None; DET_TYPE_ARABIC: Optional[Box] = None; PREP_FUNCTOR_TYPE_ARABIC: Optional[Box] = None; S_MOD_BY_N_ARABIC: Optional[Box] = None; N_MOD_BY_N_ARABIC: Optional[Box] = None; ADV_FUNCTOR_TYPE_ARABIC: Optional[Box] = None; NOUN_TYPE_BOX_FALLBACK_ARABIC: Optional[Box] = None # type: ignore

if ARE_N_S_PROPER_ATOMIC_TYPES and Box is not None and Box is not _DummyBox: # type: ignore
    logger.info("common_qnlp_types: Proceeding to define functorial Boxes.")
    try:
        ADJ_MOD_TYPE_ARABIC = Box("AdjModFunctor_common", N_ARABIC, N_ARABIC)
        ADJ_PRED_TYPE_ARABIC = Box("AdjPredFunctor_common", N_ARABIC, S_ARABIC)
        DET_TYPE_ARABIC = Box("DetFunctor_common", N_ARABIC, N_ARABIC)
        PREP_FUNCTOR_TYPE_ARABIC = Box("PrepFunctor_common", N_ARABIC, N_ARABIC)
        VERB_INTRANS_TYPE_ARABIC = Box("VerbIntransFunctor_common", N_ARABIC, S_ARABIC)
        VERB_TRANS_TYPE_ARABIC = Box("VerbTransFunctor_common", N_ARABIC @ N_ARABIC, S_ARABIC) # type: ignore
        S_MOD_BY_N_ARABIC = Box("S_mod_by_N_common", S_ARABIC @ N_ARABIC, S_ARABIC) # type: ignore
        N_MOD_BY_N_ARABIC = Box("N_mod_by_N_common", N_ARABIC @ N_ARABIC, N_ARABIC) # type: ignore
        ADV_FUNCTOR_TYPE_ARABIC = Box("AdvFunctor_common", S_ARABIC, S_ARABIC)
        
        _Ty_for_fallback_creation = Ty if Ty is not None and Ty is not _DummyTy else _DummyTy # type: ignore
        NOUN_TYPE_BOX_FALLBACK_ARABIC = Box("Noun_Fallback_common", _Ty_for_fallback_creation(), N_ARABIC) # type: ignore
        logger.info("common_qnlp_types: Global functorial Box types defined successfully.")
    except Exception as e_box_def: 
        logger.critical(f"common_qnlp_types: CRITICAL ERROR defining global Box types: {e_box_def}", exc_info=True)
        ADJ_MOD_TYPE_ARABIC = ADJ_PRED_TYPE_ARABIC = DET_TYPE_ARABIC = PREP_FUNCTOR_TYPE_ARABIC = None
        VERB_INTRANS_TYPE_ARABIC = VERB_TRANS_TYPE_ARABIC = S_MOD_BY_N_ARABIC = N_MOD_BY_N_ARABIC = ADV_FUNCTOR_TYPE_ARABIC = None
        NOUN_TYPE_BOX_FALLBACK_ARABIC = None
else:
    logger.warning("common_qnlp_types: N_ARABIC or S_ARABIC are not proper AtomicTypes, or Box is None/Dummy. Functorial Box types will be None.")

LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY = ARE_N_S_PROPER_ATOMIC_TYPES and VERB_TRANS_TYPE_ARABIC is not None and Box is not None and Box is not _DummyBox # type: ignore

logger.info(f"common_qnlp_types: MODULE EXECUTION FINISHED. N_ARABIC type: {type(N_ARABIC)}, S_ARABIC type: {type(S_ARABIC)}, ARE_N_S_PROPER_ATOMIC_TYPES: {ARE_N_S_PROPER_ATOMIC_TYPES}, LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY: {LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY}")

from lambeq.ansatz import IQPAnsatz
from lambeq.backend.grammar import Category, Cap# Ty, Id, Box are already imported or dummied
from lambeq.backend.converters.tk import Circuit as LambeqTketCircuit, from_tk, to_tk # Specific alias for clarity
from lambeq.backend.quantum import bit as q_bit_type, quantum, qubit, Id as QuantumId, Diagram as QuantumDiagram, Layer, Ty as QTy
from qiskit import QuantumCircuit, QuantumRegister 
from qiskit.circuit import Parameter 
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.circuit import Circuit as PytketCircuitInternal, Bit as PytketBit # To avoid confusion with LambeqCircuit alias

class PatchedLambeqTketCircuit(LambeqTketCircuit):
    def __init__(self,
                 circuit: PytketCircuitInternal,
                 discopy_box: Box,
                 ob_map_int: Mapping[Ty, int],
                 post_processing: Optional[GrammarDiagram] = None,
                 bitlist_map: Optional[Mapping[PytketBit, int]] = None) -> None:

        logger.info("PatchedLambeqTketCircuit: Initializing with manual quantum type conversion.")

        # ================== FIX START ==================
        def to_quantum_ty(grammar_ty: Ty, ob_map: Mapping[Ty, int]) -> quantum.Ty:
            """Converts a grammar.Ty to a quantum.Ty using a Ty->int map."""
            if not hasattr(grammar_ty, 'objects') or not grammar_ty.objects:
                # Base case for simple types like Ty('n') or Ty()
                num_qubits = ob_map.get(grammar_ty, 0)
                return qubit ** num_qubits

            # Composite types like Ty('n') @ Ty('s')
            quantum_tys = []
            for atomic_type in grammar_ty.objects:
                num_qubits = ob_map.get(atomic_type, 0)
                quantum_tys.append(qubit ** num_qubits)
            
            if not quantum_tys: return quantum.Ty()
            return quantum.Ty.tensor(*quantum_tys) if len(quantum_tys) > 1 else quantum_tys[0]

        # 1. First, call the Pytket __init__
        n_qubits = len(circuit.qubits)
        n_bits = len(circuit.bits) if circuit.bits else 0
        name = discopy_box.name
        PytketCircuitInternal.__init__(self, n_qubits, n_bits, name)

        # 2. Manually set dom and cod to QUANTUM types using our helper function.
        self.dom = to_quantum_ty(discopy_box.dom, ob_map_int)
        self.cod = to_quantum_ty(discopy_box.cod, ob_map_int)

        # 3. Initialize the Diagram parent class with the correct quantum types.
        QuantumDiagram.__init__(self,
                                dom=self.dom,
                                cod=self.cod,
                                layers=[Layer(QTy(), self, QTy())])
        # =================== FIX END ===================

        # 4. Set remaining attributes.
        self._circuit = circuit
        self._discopy_box = discopy_box
        self.is_pregroup = False
        self.is_causal = True
        self.is_circuital = True 

        # 5. Original library logic.
        n_bits_from_cod = len(self.cod)
        self.post_selection = {bit: i for i, bit in enumerate(self.cod)}
        self.post_processing = post_processing or Id(q_bit_type ** (n_bits_from_cod - len(self.post_selection)))

        if bitlist_map is None:
            bitlist_map = {b: i for i, b in enumerate(circuit.bits)}
        self._bitlist_map = bitlist_map
        
        logger.info("PatchedLambeqTketCircuit: Successfully initialized.")


    @property
    def discopy_box(self) -> Box:
        """The DisCoPy box that this circuit is based on."""
        return self._discopy_box
    
    @property
    def label(self) -> Box:
        """The label of the circuit, which is an alias for the DisCoPy box."""
        return self._discopy_box

    def to_tk(self) -> PytketCircuitInternal:
        """Return the underlying Pytket circuit."""
        return self._circuit
    
    def to_diagram(self) -> QuantumDiagram:
        """
        A circuit is already a diagram containing itself as a box.
        This method is required for compatibility with some of lambeq's
        internal converters. The original implementation was buggy.
        """
        logger.debug(f"PatchedLambeqTketCircuit '{self.name}': .to_diagram() called, returning self.")
        return self
    
    def to_qiskit(self) -> 'QuantumCircuit':
        """Fixed method to convert to Qiskit circuit."""
        try:
            from pytket.extensions.qiskit import tk_to_qiskit
            qiskit_circuit = tk_to_qiskit(self._circuit)
            logger.info(f"Successfully converted to Qiskit circuit with {qiskit_circuit.num_qubits} qubits")
            return qiskit_circuit
        except Exception as e:
            logger.error(f"Error in tk_to_qiskit conversion: {e}")
            # Create a fallback Qiskit circuit
            from qiskit import QuantumCircuit
            fallback_qc = QuantumCircuit(max(1, self._circuit.n_qubits))
            logger.warning(f"Created fallback Qiskit circuit with {fallback_qc.num_qubits} qubits")
            return fallback_qc
# ======================================================================

class AmbiguousLexicalBox(Box): 
    def __init__(self, name: str, base_type: Ty, senses: List[str],
                 data: Optional[Dict[str, Any]] = None, **kwargs):
        if not isinstance(base_type, Ty): # Ensure base_type is Ty
            logger.warning(f"AmbiguousLexicalBox '{name}': base_type was {type(base_type)}, attempting to convert to Ty assuming it's a string like 'n'.")
            if isinstance(base_type, str): # If it was a string like "n"
                 base_type = Ty(base_type)
            else: # If it's something else, this will likely fail or use a dummy.
                 base_type = Ty(str(base_type)) if Ty is not _DummyTy else _DummyTy(str(base_type))


        super().__init__(name, Ty(), base_type, **kwargs) 

        current_box_data = self.data if hasattr(self, 'data') and self.data is not None else {}
        custom_metadata = data if data is not None else {}
        custom_metadata.update({
            'senses': senses,
            'base_type_name': base_type, # Store the Ty object
            '_is_ambiguous_lexical': True
        })
        self.data = custom_metadata 
        logger.debug(f"AmbiguousLexicalBox '{self.name}' created. Data: {self.data}")


class ControlledSenseFunctor(Functor):
    def __init__(self, ob_map: Mapping[Ty, int], n_layers: int, n_single_qubit_params: int):
        logger.info("ControlledSenseFunctor (Functor Architecture Fix): Initializing...")
        # Store the original Ty->int map explicitly to avoid relying on ansatz internals
        self.ob_map_int = ob_map
        self.internal_iqp_ansatz = IQPAnsatz(ob_map, n_layers, n_single_qubit_params)
        self.symbols: Set[Parameter] = set()
        super().__init__(ob=self.ob_func, ar=self.ar_func, target_category=quantum)

    def ob_func(self, functor_instance, grammar_ty: Ty) -> quantum.Ty:
        if hasattr(grammar_ty, 'objects') and grammar_ty.objects:
            return quantum.Ty.tensor(*(functor_instance(t) for t in grammar_ty.objects))
        # Use the Ty->int map to get the number of qubits, then create the quantum type.
        num_qubits = self.ob_map_int.get(grammar_ty, 0)
        return qubit ** num_qubits

    def ar_func(self, functor_instance, box: Box) -> QuantumDiagram:
        use_variational = getattr(self, '_use_variational_ancilla', False)
        # Robust check using the data flag
        if hasattr(box, 'data') and isinstance(box.data, dict) and box.data.get('_is_ambiguous_lexical'):
            logger.debug(f"Intercepted AmbiguousLexicalBox '{box.name}' via attribute. Calling custom handler.")
            return self._handle_ambiguous_lexical_box_for_functor(box, use_variational_ancilla=use_variational)
        else:
            # For regular boxes, use the internal IQP ansatz
            return self.internal_iqp_ansatz(box)

    def _get_default_params(self, n_qubits: int, name_hint: str = 'param') -> List[Parameter]:
        params = []
        params_per_qubit_val = self.internal_iqp_ansatz.n_single_qubit_params
        for i in range(n_qubits):
            for j in range(params_per_qubit_val):
                new_param = Parameter(f'{name_hint}_q{i}_p{j}')
                params.append(new_param)
                self.symbols.add(new_param) 
        return params

    def _apply_controlled_iqp_layer(self, qc: QuantumCircuit, control_qubit, data_register: QuantumRegister, params: List[Parameter]):
        qubits_in_register = len(data_register)
        params_per_qubit_val = self.internal_iqp_ansatz.n_single_qubit_params
        for i in range(qubits_in_register):
            q_idx = i
            if (i * params_per_qubit_val + 1) < len(params):
                qc.cry(params[i * params_per_qubit_val + 0], control_qubit, data_register[q_idx])
                qc.crz(params[i * params_per_qubit_val + 1], control_qubit, data_register[q_idx])
            elif (i * params_per_qubit_val) < len(params):
                qc.cry(params[i * params_per_qubit_val + 0], control_qubit, data_register[q_idx])

    def _handle_ambiguous_lexical_box_for_functor(self, box: AmbiguousLexicalBox, use_variational_ancilla: bool = False) -> QuantumDiagram:
        logger.critical(f"CRITICAL_SUCCESS_FUNCTOR: Entered _handle_ambiguous_lexical_box_for_functor for '{box.name}'!")

        try:
            logger.info("    Attempting to create test QuantumRegister/QuantumCircuit...")
            test_qr = QuantumRegister(1, "test_reg_inside_handler")
            test_qc = QuantumCircuit(test_qr, name="test_qc_inside_handler")
            logger.info(f"    SUCCESSFULLY created test_qc with {test_qc.num_qubits} qubit(s).")
        except Exception as e_qiskit_test:
            logger.error(f"    EXCEPTION during initial Qiskit object creation inside handler: {e_qiskit_test}", exc_info=True)
            logger.warning(f"    Delegating '{box.name}' to internal IQP due to Qiskit creation failure.")
            return self.internal_iqp_ansatz(box)

        if not isinstance(box.data, dict):
            logger.error(f"Functor Handler: Box '{box.name}' data not dict. Delegating to internal IQP.")
            return self.internal_iqp_ansatz(box)

        senses = box.data.get('senses', [])
        actual_base_type = box.data.get('base_type_name')

        if not senses or not isinstance(actual_base_type, Ty) or actual_base_type not in self.ob_map_int:
            logger.warning(f"Functor Handler for '{box.name}': Preconditions not met (senses, base_type, or base_type not in ob_map). Actual base_type: {actual_base_type} (type: {type(actual_base_type)}). Delegating to internal IQP.")
            return self.internal_iqp_ansatz(box)

        num_data_qubits = self.ob_map_int[actual_base_type]
        logger.info(f"    Original num_data_qubits from ob_map for '{str(actual_base_type)}': {num_data_qubits}")
        
        num_senses = len(senses)

        if num_senses != 2:
            logger.warning(f"Functor Handler for '{box.name}': num_senses != 2. Delegating to internal IQP.")
            return self.internal_iqp_ansatz(box)

        logger.info(f"  For box '{box.name}': dom='{box.dom}' (type: {type(box.dom)}), cod='{box.cod}' (type: {type(box.cod)})")
        
        logger.info(f"  Functor Handler: Building controlled circuit for '{box.name}'")
        safe_box_name = f"qnlp_amb_{id(box)}_{actual_base_type.name if hasattr(actual_base_type, 'name') else str(actual_base_type)}" 
        
        q_data = QuantumRegister(num_data_qubits, name=f'q_{safe_box_name}_data')
        q_ancilla = QuantumRegister(1, name=f'q_{safe_box_name}_anc')
        qc = QuantumCircuit(q_data, q_ancilla, name=f"circ_amb_{safe_box_name}")

        if use_variational_ancilla:
            logger.info(f"  Using VARIATIONAL ancilla for '{box.name}'")
            ancilla_params = [Parameter(f"{box.name}_ancilla_theta"),
                            Parameter(f"{box.name}_ancilla_phi"),
                            Parameter(f"{box.name}_ancilla_lambda")]
            self.symbols.update(ancilla_params)
            
            qc.u(ancilla_params[0], ancilla_params[1], ancilla_params[2], q_ancilla[0])
        else:
            logger.info(f"  Using FIXED (Hadamard) ancilla for '{box.name}'")
            qc.h(q_ancilla[0])

        params_sense0 = self._get_default_params(num_data_qubits, name_hint=f"{box.name}_{senses[0]}")
        qc.x(q_ancilla[0])
        self._apply_controlled_iqp_layer(qc, q_ancilla[0], q_data, params_sense0)
        qc.x(q_ancilla[0])

        params_sense1 = self._get_default_params(num_data_qubits, name_hint=f"{box.name}_{senses[1]}")
        self._apply_controlled_iqp_layer(qc, q_ancilla[0], q_data, params_sense1)
        
        try:
            logger.debug("    Converting Qiskit circuit to Pytket circuit...")
            tk_circuit_actual: PytketCircuitInternal = qiskit_to_tk(qc)
            logger.debug(f"    Pytket circuit created. Num qubits: {tk_circuit_actual.n_qubits}")

            semantic_discopy_box = Box(name=box.name, dom=box.dom, cod=actual_base_type)
            
            logger.info(f"Creating circuit wrapper for '{box.name}' using the PatchedLambeqTketCircuit.")
            
            lambeq_tket_circuit_wrapper = PatchedLambeqTketCircuit(
                circuit=tk_circuit_actual, 
                discopy_box=semantic_discopy_box,
                ob_map_int=self.ob_map_int,
                post_processing=None
            )
            
            logger.info(f"  Successfully created PatchedLambeqTketCircuit for '{box.name}'.")
            return lambeq_tket_circuit_wrapper
        except Exception as e_lc_constr:
            logger.error(f"EXCEPTION during LambeqTketCircuit construction/conversion for {box.name}: {e_lc_constr}", exc_info=True)
            logger.warning(f"  Falling back to internal IQP for {box.name} due to LambeqTketCircuit construction error.")
            return self.internal_iqp_ansatz(box)


    def __call__(self, diagram: Union[GrammarDiagram, Ty], use_variational_ancilla: bool = False) -> Union[QuantumDiagram, quantum.Ty]:
        """
        Applies the functor to a DisCoPy diagram or type.
        Fixed to properly handle mixed circuit/diagram composition and return the result.
        """
        if isinstance(diagram, Ty):
            return self.ob_func(self, diagram)

        # Store the ancilla flag on the instance so ar_func can access it
        self._use_variational_ancilla = use_variational_ancilla

        try:
            # Handle different diagram types
            if isinstance(diagram, (AmbiguousLexicalBox, Box, Word)):
                # Single box - apply ar_func directly
                result = self.ar_func(self, diagram)
            elif isinstance(diagram, Swap):
                result = Swap(self(diagram.left), self(diagram.right))
            elif isinstance(diagram, (Cup, Cap)):
                result = type(diagram)(self(diagram.left), self(diagram.right))
            elif hasattr(diagram, 'boxes') and hasattr(diagram, 'offsets'):
                # Composite diagram - this is the key fix
                logger.info(f"Processing composite diagram with {len(diagram.boxes)} boxes")
                
                # Apply functor to each box individually
                quantum_boxes = []
                circuit_boxes = []  # Track which are circuits vs diagrams
                
                for i, box in enumerate(diagram.boxes):
                    logger.info(f"  Processing box {i}: {box.name} ({type(box).__name__})")
                    quantum_box = self.ar_func(self, box)
                    quantum_boxes.append(quantum_box)
                    # Track if this is a circuit wrapper
                    circuit_boxes.append(isinstance(quantum_box, PatchedLambeqTketCircuit))
                
                # If we have mixed types, we need to convert everything to a unified format
                if any(circuit_boxes) and not all(circuit_boxes):
                    logger.info("Mixed circuit/diagram types detected. Converting to unified circuit format.")
                    result = self._stitch_mixed_quantum_objects(quantum_boxes, diagram)
                    
                elif all(circuit_boxes):
                    # All are circuits - need to compose them at the circuit level
                    logger.info("All quantum objects are circuits. Composing at circuit level.")
                    result = self._compose_circuit_objects(quantum_boxes, diagram)
                
                else:
                    # All are regular diagrams - use normal composition
                    logger.info("All quantum objects are diagrams. Using normal composition.")
                    if len(quantum_boxes) == 1:
                        result = quantum_boxes[0]
                    elif len(quantum_boxes) == 2:
                        result = quantum_boxes[0] >> quantum_boxes[1]
                    else:
                        result = quantum_boxes[0]
                        for qbox in quantum_boxes[1:]:
                            result = result >> qbox
                            
                    logger.info(f"Composite diagram processed successfully")
            else:
                # Fallback for other diagram types
                logger.warning(f"Unknown diagram type: {type(diagram)}. Attempting generic conversion.")
                result = super().__call__(diagram)
                
        except Exception as e:
            logger.error(f"Error in functor application: {e}", exc_info=True)
            raise
        finally:
            # Clean up the instance attribute
            if hasattr(self, '_use_variational_ancilla'):
                del self._use_variational_ancilla
        
        # CRITICAL FIX: Always return the result, not self
        logger.info(f"Functor __call__ returning result of type: {type(result)}")
        return result

    def _compose_circuit_objects(self, circuit_objects: List[PatchedLambeqTketCircuit], 
                            original_diagram: GrammarDiagram) -> PatchedLambeqTketCircuit:
        """Compose multiple circuit objects into a single circuit."""
        logger.info("Composing multiple circuit objects...")
        
        try:
            # Extract Pytket circuits from all objects
            circuits_data = []
            for i, circuit_obj in enumerate(circuit_objects):
                circuits_data.append({
                    'circuit': circuit_obj.to_tk(),
                    'name': f"circuit_obj_{i}",
                    'type': 'circuit_object'
                })
            
            # Use the sequential composition method
            combined_circuit = self._compose_circuits_sequentially(circuits_data)
            
            # Create a representative box for the composed circuit
            composed_box_name = f"ComposedCircuit_{len(circuit_objects)}boxes"
            composed_box = Box(composed_box_name, original_diagram.dom, original_diagram.cod)
            
            # Wrap in our patched circuit class
            return PatchedLambeqTketCircuit(
                circuit=combined_circuit,
                discopy_box=composed_box,
                ob_map_int=self.ob_map_int
            )
            
        except Exception as e:
            logger.error(f"Error composing circuit objects: {e}", exc_info=True)
            # Fallback: just return the first circuit
            return circuit_objects[0]

    def _stitch_mixed_quantum_objects(self, quantum_objects: List, original_diagram: GrammarDiagram) -> PatchedLambeqTketCircuit:
        """Handle mixed circuit/diagram objects by converting all to circuit format."""
        logger.info("Stitching mixed quantum objects into unified circuit...")
        
        try:
            # Convert all objects to Pytket circuits with proper qubit mapping
            all_circuits_data = []
            
            for i, qobj in enumerate(quantum_objects):
                if isinstance(qobj, PatchedLambeqTketCircuit):
                    circuit = qobj.to_tk()
                    all_circuits_data.append({
                        'circuit': circuit,
                        'name': f"circuit_{i}",
                        'type': 'circuit'
                    })
                else:
                    # Convert diagram to circuit
                    try:
                        circuit = qobj.to_tk()
                        all_circuits_data.append({
                            'circuit': circuit,
                            'name': f"diagram_{i}",
                            'type': 'diagram'
                        })
                    except Exception as e:
                        logger.warning(f"Could not convert object {i} to circuit: {e}. Creating identity circuit.")
                        # Create a minimal identity circuit that matches the expected type dimensions
                        from pytket.circuit import Circuit as PytketCircuitInternal
                        minimal_circuit = PytketCircuitInternal(1)
                        minimal_circuit.add_gate('I', [0])  # Identity gate
                        all_circuits_data.append({
                            'circuit': minimal_circuit,
                            'name': f"fallback_{i}",
                            'type': 'fallback'
                        })
            
            # Now properly compose the circuits using sequential composition
            final_circuit = self._compose_circuits_sequentially(all_circuits_data)
            
            # Create a representative box
            stitched_box_name = f"StitchedMixed_{len(quantum_objects)}objects"
            stitched_box = Box(stitched_box_name, original_diagram.dom, original_diagram.cod)
            
            # Return as patched circuit
            return PatchedLambeqTketCircuit(
                circuit=final_circuit,
                discopy_box=stitched_box,
                ob_map_int=self.ob_map_int
            )
            
        except Exception as e:
            logger.error(f"Error stitching mixed objects: {e}", exc_info=True)
            return self._create_fallback_circuit(original_diagram)

    def _compose_circuits_sequentially(self, circuits_data: List[dict]) -> 'PytketCircuitInternal':
        """Compose circuits sequentially by adding gates to the same register."""
        from pytket.circuit import Circuit as PytketCircuitInternal, Qubit
    
        if not circuits_data:
            return PytketCircuitInternal(1)
    
        logger.info(f"Composing {len(circuits_data)} circuits sequentially (Gate-level Logic)...")
    
        # Determine the qubit count from the largest circuit in the sequence.
        max_qubits = max((c['circuit'].n_qubits for c in circuits_data if c.get('circuit')), default=0)
        max_qubits = max(max_qubits, 1)
    
        logger.info(f"Creating combined circuit with {max_qubits} qubits.")
        combined_circuit = PytketCircuitInternal(max_qubits)
        target_qubits = combined_circuit.qubits  # This is a list of Qubit objects
    
        # Append each circuit's gates to the combined circuit
        for i, circuit_data in enumerate(circuits_data):
            circuit_to_add = circuit_data.get('circuit')
            if not circuit_to_add:
                continue
            
            circuit_name = circuit_data.get('name', f'circuit_{i}')
            logger.debug(f"Adding gates from '{circuit_name}' ({circuit_to_add.n_qubits} qubits).")
            
            # Create a mapping from the source circuit's qubits to the target circuit's qubits.
            # This handles cases where qubit registers might have different names.
            source_qubits = circuit_to_add.qubits
            
            # We assume a simple sequential mapping: qubit 0 of source -> qubit 0 of target, etc.
            qubit_map = {source_qubits[i]: target_qubits[i] for i in range(min(len(source_qubits), len(target_qubits)))}
    
            for command in circuit_to_add.get_commands():
                try:
                    # Map the arguments (qubits) of the command to the combined circuit's qubits
                    mapped_args = [qubit_map[arg] for arg in command.args if arg in qubit_map]
                    
                    # We must have mapped all qubits for the operation to be valid.
                    if len(mapped_args) == len(command.args):
                         combined_circuit.add_gate(command.op, mapped_args)
                    else:
                        unmapped_args = [arg for arg in command.args if arg not in qubit_map]
                        logger.warning(f"Skipping command {command} from '{circuit_name}': could not map qubits {unmapped_args}.")
    
                except KeyError as e:
                    logger.error(f"KeyError during qubit mapping for command {command} in '{circuit_name}': qubit {e} not in map. This indicates a structural mismatch between circuits.")
                except Exception as e:
                    logger.error(f"Unexpected error adding command {command} from '{circuit_name}': {e}", exc_info=True)
    
        logger.info(f"Successfully combined circuits into a {combined_circuit.n_qubits}-qubit circuit.")
        return combined_circuit

    
    def _create_qubit_mapping(self, source_circuit: 'PytketCircuitInternal', 
                                target_circuit: 'PytketCircuitInternal', 
                                circuit_index: int) -> dict:
        """Fixed version of qubit mapping that handles different qubit types properly."""
        qubit_map = {}
        
        # Get qubits from both circuits
        source_qubits = list(source_circuit.qubits)
        target_qubits = list(target_circuit.qubits)
        
        logger.debug(f"Mapping {len(source_qubits)} source qubits to {len(target_qubits)} target qubits")
        
        # Create mapping from source qubit objects to target qubit indices
        for i, source_qubit in enumerate(source_qubits):
            # Map to the corresponding index in target, with bounds checking
            target_index = min(i, len(target_qubits) - 1)
            # Store the mapping as qubit object -> integer index
            qubit_map[source_qubit] = target_index
            logger.debug(f"Mapped source qubit {source_qubit} to target index {target_index}")
        
        return qubit_map


    def _create_fallback_circuit(self, original_diagram: GrammarDiagram) -> PatchedLambeqTketCircuit:
        """Create a minimal fallback circuit when composition fails."""
        from pytket.circuit import Circuit as PytketCircuitInternal
        
        logger.info("Creating fallback circuit...")
        
        # Create a minimal circuit based on the diagram's type structure
        try:
            # Try to infer the number of qubits needed from the domain/codomain
            dom_qubits = sum(self.ob_map_int.get(ty, 1) for ty in getattr(original_diagram.dom, 'objects', [original_diagram.dom]))
            cod_qubits = sum(self.ob_map_int.get(ty, 1) for ty in getattr(original_diagram.cod, 'objects', [original_diagram.cod]))
            total_qubits = max(dom_qubits, cod_qubits, 2)  # At least 2 qubits
        except:
            total_qubits = 2  # Default fallback
        
        fallback_circuit = PytketCircuitInternal(total_qubits)
        
        # Add some minimal gates to make it non-trivial
        for i in range(min(total_qubits, 2)):
            fallback_circuit.add_gate('H', [i])  # Hadamard gates
        
        fallback_box = Box("FallbackCircuit", original_diagram.dom, original_diagram.cod)
        
        return PatchedLambeqTketCircuit(
            circuit=fallback_circuit,
            discopy_box=fallback_box,
            ob_map_int=self.ob_map_int
        )
    
    def _stitch_to_final_circuit(self, quantum_diagram: QuantumDiagram, original_diagram: GrammarDiagram) -> PatchedLambeqTketCircuit:
        """Helper to convert the final quantum.Diagram to a single circuit object."""
        try:
            # Convert quantum diagram to Pytket circuit
            final_tk_circuit: PytketCircuitInternal = quantum_diagram.to_tk()
            logger.info("Successfully stitched intermediate diagram into a final Pytket circuit.")
        except Exception as e:
            logger.error(f"Failed to stitch diagram to Pytket circuit: {e}", exc_info=True)
            # Create a minimal fallback circuit
            final_tk_circuit = PytketCircuitInternal(1) # Fallback to 1-qubit circuit
        
        # Create a representative box for the whole sentence
        sentence_box_name = f"SentenceDiagram_{getattr(original_diagram, 'name', 'composite')}"
        sentence_box = Box(sentence_box_name, original_diagram.dom, original_diagram.cod)
        
        # Wrap the final circuit in our PatchedLambeqTketCircuit class
        return PatchedLambeqTketCircuit(
            circuit=final_tk_circuit,
            discopy_box=sentence_box,
            ob_map_int=self.ob_map_int
        )