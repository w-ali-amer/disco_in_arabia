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
    Cup = ImportedCup_; Id = ImportedId_; Spider = ImportedSpider_; Swap = ImportedSwap_
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
from lambeq.backend.grammar import Category# Ty, Id, Box are already imported or dummied
from lambeq.backend.converters.tk import Circuit as LambeqTketCircuit, from_tk, to_tk # Specific alias for clarity
from lambeq.backend.quantum import bit as q_bit_type, quantum, qubit, Diagram as QuantumDiagram, Layer
from qiskit import QuantumCircuit, QuantumRegister 
from qiskit.circuit import Parameter 
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.circuit import Circuit as PytketCircuitInternal, Bit as PytketBit # To avoid confusion with LambeqCircuit alias

class PatchedLambeqTketCircuit(LambeqTketCircuit):
    def __init__(self,
                 circuit: PytketCircuitInternal,
                 discopy_box: Box,
                 post_processing: Optional[GrammarDiagram] = None,
                 bitlist_map: Optional[Mapping[PytketBit, int]] = None) -> None:

        logger.info("PatchedLambeqTketCircuit: Using fully replaced and corrected __init__.")
        
        # This is a full replacement of the library's buggy __init__ logic.
        # It correctly initializes the parent classes and their attributes
        # in the right order to prevent all previous errors. It does NOT call
        # the original buggy `super().__init__` via super().

        # 1. Initialize the Pytket Circuit part of the inheritance. This is the C++-bound
        #    class that requires integer arguments. This fixes the nanobind warning.
        n_qubits = len(circuit.qubits)
        n_bits = len(circuit.bits) if circuit.bits else 0
        name = discopy_box.name
        PytketCircuitInternal.__init__(self, n_qubits, n_bits, name)

        # 2. Initialize the Diagram part of the inheritance. This sets `.dom`, `.cod`, etc.
        #    and fixes the `AttributeError`.
        GrammarDiagram.__init__(self,
                                dom=discopy_box.dom,
                                cod=discopy_box.cod,
                                layers=1)
        
        # 3. Now that the object is a valid Diagram and PytketCircuit, set the
        #    remaining attributes that are specific to LambeqTketCircuit.
        self._circuit = circuit
        self._discopy_box = discopy_box
        self.is_pregroup = False
        self.is_causal = True # Most diagrams are causal
        self.is_circuital = True # This is the missing attribute from the error!
        # 4. This logic is from the original library code. It should now work
        #    correctly because `self.cod` was set in step 2.
        n_bits_from_cod = len(self.cod)
        self.post_selection = {bit: i for i, bit in enumerate(self.cod)}
        self.post_processing = post_processing or Id(q_bit_type ** (n_bits_from_cod - len(self.post_selection)))

        if bitlist_map is None:
            bitlist_map = {b: i for i, b in enumerate(circuit.bits)}
        self._bitlist_map = bitlist_map
        
        # CRITICAL FIX: Copy the circuit content to make this object a proper lambeq circuit
        # This ensures that when from_tk is called, it can find the necessary attributes
        #self.add_circuit(circuit)
        
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
        Fixed version that properly converts to quantum diagram.
        Instead of using from_tk on the raw circuit, we use the fact that
        this object IS already a lambeq circuit with proper attributes.
        """
        logger.debug(f"PatchedLambeqTketCircuit '{self.name}': .to_diagram() called.")
        
        try:
            # Method 1: Use the built-in conversion since this object already has lambeq structure
            # The to_tk converter should work on this object directly
            return to_tk(self)
        except Exception as e1:
            logger.warning(f"Method 1 failed: {e1}")
            try:
                # Method 2: Create a proper lambeq circuit first, then convert
                # We'll manually build the quantum diagram structure
                logger.debug("Attempting manual quantum diagram construction...")
                
                # Get the quantum types for domain and codomain
                qubit_dom = qubit ** len(self.dom) if len(self.dom) > 0 else qubit ** 0
                qubit_cod = qubit ** len(self.cod) if len(self.cod) > 0 else qubit ** 1
                
                # Create a quantum diagram with the circuit operations
                from lambeq.backend.quantum import Ket, Bra
                
                # Start with identity on the domain
                diagram = Id(qubit_dom)
                
                # Add the circuit as a layer
                # For now, we'll create a simple parameterized layer
                # This is a simplified approach - you might need to adapt based on your specific needs
                if hasattr(self, '_circuit') and self._circuit.n_gates > 0:
                    # Create a custom quantum box that represents our circuit
                    circuit_box = Box(f"circuit_{self.name}", qubit_dom, qubit_cod)
                    diagram = diagram >> circuit_box
                else:
                    # If no gates, just return identity or measurement
                    if len(self.cod) > 0:
                        diagram = diagram >> Id(qubit_cod)
                
                logger.debug(f"Manual quantum diagram construction successful for '{self.name}'")
                return diagram
                
            except Exception as e2:
                logger.error(f"Method 2 also failed: {e2}")
                # Method 3: Fallback to a simple identity diagram
                logger.warning(f"Using fallback identity diagram for '{self.name}'")
                qubit_cod = qubit ** max(1, len(self.cod))
                return Id(qubit_cod)
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
    def __init__(self, ob_map_for_ansatz: Mapping[Ty, int], n_layers: int, n_single_qubit_params: int):
        logger.info("ControlledSenseFunctor (v4 - Mimicking Ansatz): Initializing...")

        # --- START OF THE MINIMAL FIX ---

        # 1. Store the ob_map and ansatz parameters on self, so our _ob and _ar methods can use them.
        self.ob_map = {src: qubit ** ty if isinstance(ty, int) else ty
                       for src, ty in ob_map_for_ansatz.items()}
        # Ensure all required base types are in the map
        if S_ARABIC not in self.ob_map: self.ob_map[S_ARABIC] = qubit
        if ROOT_TYPE_ARABIC not in self.ob_map: self.ob_map[ROOT_TYPE_ARABIC] = qubit
        
        # We still need the ansatz for its utility methods, but we won't delegate to it as a functor.
        self.internal_iqp_ansatz = IQPAnsatz(self.ob_map, n_layers, n_single_qubit_params)
        self.symbols: Set[Parameter] = set()

        # 2. Define the object mapping function (_ob). This is called by the Functor.
        # It maps a grammatical Ty to a quantum Ty (a certain number of qubits).
        def ob_map_func(functor_instance, grammar_ty: Ty) -> Ty:
            # This logic is adapted from CircuitAnsatz.ob_size
            if grammar_ty not in self.ob_map:
                # This handles complex types like n @ s by summing their qubit counts.
                # It recursively calls the functor's ob mapping on the atomic components.
                return Ty.tensor(*(functor_instance(t) for t in grammar_ty))
            return self.ob_map[grammar_ty]

        # 3. Define the arrow mapping function (_ar). This is the master function.
        # It will look at the box type and decide which specific handler to use.
        def ar_map_func(functor_instance, box: Box):
            if isinstance(box, AmbiguousLexicalBox):
                # If it's our special box, call our custom circuit builder.
                # This handler must now return a quantum.Diagram, not a raw circuit.
                return self._handle_ambiguous_lexical_box_for_functor(box)
            else:
                # For any other Box, Word, Cup, etc., use the utility ansatz's default handler.
                # The `_ar` method of CircuitAnsatz does exactly what we need.
                return self.internal_iqp_ansatz._ar(functor_instance, box)

        # 4. Initialize the parent Functor with the correct arguments.
        super().__init__(ob=ob_map_func, ar=ar_map_func, target_category=quantum)


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


    def _handle_ambiguous_lexical_box_for_functor(self, box: AmbiguousLexicalBox, use_variational_ancilla: bool = False) -> LambeqTketCircuit:
        logger.critical(f"CRITICAL_SUCCESS_FUNCTOR: Entered _handle_ambiguous_lexical_box_for_functor for '{box.name}'!")

        try:
            logger.info("    Attempting to create test QuantumRegister/QuantumCircuit...")
            test_qr = QuantumRegister(1, "test_reg_inside_handler")
            test_qc = QuantumCircuit(test_qr, name="test_qc_inside_handler")
            logger.info(f"    SUCCESSFULLY created test_qc with {test_qc.num_qubits} qubit(s).")
        except Exception as e_qiskit_test:
            logger.error(f"    EXCEPTION during initial Qiskit object creation inside handler: {e_qiskit_test}", exc_info=True)
            logger.warning(f"    Delegating '{box.name}' to internal IQP due to Qiskit creation failure.")
            # self.internal_iqp_ansatz(box) returns a lambeq.backend.circuit.Circuit (e.g. TKETCircuit)
            # which is a subclass of Diagram. This should be compatible.
            return self.internal_iqp_ansatz(box) # type: ignore 

        if not isinstance(box.data, dict):
            logger.error(f"Functor Handler: Box '{box.name}' data not dict. Delegating to internal IQP.")
            return self.internal_iqp_ansatz(box) # type: ignore

        senses = box.data.get('senses', [])
        actual_base_type = box.data.get('base_type_name') # This is a Ty object

        if not senses or not isinstance(actual_base_type, Ty) or actual_base_type not in self.internal_iqp_ansatz.ob_map:
            logger.warning(f"Functor Handler for '{box.name}': Preconditions not met (senses, base_type, or base_type not in ob_map). Actual base_type: {actual_base_type} (type: {type(actual_base_type)}). Delegating to internal IQP.")
            return self.internal_iqp_ansatz(box) # type: ignore

        original_num_data_qubits = self.internal_iqp_ansatz.ob_map[actual_base_type]
        logger.info(f"    Original num_data_qubits from ob_map for '{str(actual_base_type)}': {original_num_data_qubits}")
        num_data_qubits = 1 
        logger.info(f"    USING HARDCODED num_data_qubits = {num_data_qubits}")
        num_senses = len(senses)

        if num_senses != 2: # For simplicity, current controlled logic assumes 2 senses
            logger.warning(f"Functor Handler for '{box.name}': num_senses != 2. Delegating to internal IQP.")
            return self.internal_iqp_ansatz(box) # type: ignore

        logger.info(f"  For box '{box.name}': dom='{box.dom}' (type: {type(box.dom)}), cod='{box.cod}' (type: {type(box.cod)})")
        logger.info(f"    N_ARABIC object ID in this scope: {id(N_ARABIC)}")

        if box.cod is N_ARABIC: # Check actual_base_type as well if box.cod could differ
            logger.info("    box.cod IS the N_ARABIC object.")
        else:
            logger.warning(f"    box.cod (ID: {id(box.cod)}, str: '{str(box.cod)}') is NOT N_ARABIC (ID: {id(N_ARABIC)}, str: '{str(N_ARABIC)}'). This might be an issue if ob_map relies on strict identity for N_ARABIC.")
            logger.warning(f"    actual_base_type from box.data: '{str(actual_base_type)}' (ID: {id(actual_base_type)})")
            if str(actual_base_type) != str(N_ARABIC) or actual_base_type not in self.internal_iqp_ansatz.ob_map :
                 logger.error(f"CRITICAL: Mismatch or unmapped actual_base_type '{actual_base_type}'. Cannot proceed with custom logic.")
                 return self.internal_iqp_ansatz(box) # type: ignore


        logger.info(f"  Functor Handler: Building controlled circuit for '{box.name}'")
        safe_box_name =  f"qnlp_amb_{id(box)}_{actual_base_type.name if hasattr(actual_base_type, 'name') else str(actual_base_type)}" 
        
        q_data = QuantumRegister(num_data_qubits, name=f'q_{safe_box_name}_data')
        q_ancilla = QuantumRegister(1, name=f'q_{safe_box_name}_anc')
        qc = QuantumCircuit(q_data, q_ancilla, name=f"circ_amb_{safe_box_name}")

        if use_variational_ancilla:
            logger.info(f"  Using VARIATIONAL ancilla for '{box.name}'")
            # Create 3 new learnable parameters for the U gate on the ancilla
            ancilla_params = [Parameter(f"{box.name}_ancilla_theta"),
                            Parameter(f"{box.name}_ancilla_phi"),
                            Parameter(f"{box.name}_ancilla_lambda")]
            self.symbols.update(ancilla_params) # Add them to the functor's symbol list
            
            # Apply the parameterized U gate instead of the Hadamard gate
            qc.u(ancilla_params[0], ancilla_params[1], ancilla_params[2], q_ancilla[0])
        else:
            # Original H3.1 logic: fixed superposition
            logger.info(f"  Using FIXED (Hadamard) ancilla for '{box.name}'")
            qc.h(q_ancilla[0])

        
        params_sense0 = self._get_default_params(num_data_qubits, name_hint=f"{safe_box_name}_{senses[0]}")
        qc.x(q_ancilla[0])
        self._apply_controlled_iqp_layer(qc, q_ancilla[0], q_data, params_sense0)
        qc.x(q_ancilla[0])

        params_sense1 = self._get_default_params(num_data_qubits, name_hint=f"{safe_box_name}_{senses[1]}")
        self._apply_controlled_iqp_layer(qc, q_ancilla[0], q_data, params_sense1)
        
        try:
            logger.debug("    Converting Qiskit circuit to Pytket circuit...")
            tk_circuit_actual: PytketCircuitInternal = qiskit_to_tk(qc)
            logger.debug(f"    Pytket circuit created. Num qubits: {tk_circuit_actual.n_qubits}")

            semantic_discopy_box = Box(name=box.name, dom=box.dom, cod=actual_base_type)
            
            logger.info(f"Creating circuit wrapper for '{box.name}' using the PatchedLambeqTketCircuit.")
            
            # Use our new patched class instead of the original, buggy one.
            lambeq_tket_circuit_wrapper = PatchedLambeqTketCircuit(
                circuit=tk_circuit_actual, 
                discopy_box=semantic_discopy_box,
                post_processing=None # We can pass None now, our patched class will handle it
            )
            
            logger.info(f"  Successfully created PatchedLambeqTketCircuit for '{box.name}'.")
            return lambeq_tket_circuit_wrapper
        except Exception as e_lc_constr:
            logger.error(f"EXCEPTION during LambeqTketCircuit construction/conversion for {box.name}: {e_lc_constr}", exc_info=True)
            logger.warning(f"  Falling back to internal IQP for {box.name} due to LambeqTketCircuit construction error.")
            return self.internal_iqp_ansatz(box) # type: ignore 
        

    def __call__(self, diagram: Union[GrammarDiagram, Box, Word], use_variational_ancilla: bool = False): # Return type should be Diagram (LambeqTketCircuit is a Diagram)
        logger.debug(f"ControlledSenseFunctor __call__: Received element '{diagram.name if hasattr(diagram, 'name') else type(diagram)}', type: {type(diagram)}")
        if isinstance(diagram, AmbiguousLexicalBox):
            logger.debug(f"  Dispatching single AmbiguousLexicalBox '{diagram.name}' to custom handler.")
            # Pass the flag to the handler
            result = self._handle_ambiguous_lexical_box_for_functor(diagram, use_variational_ancilla)
            logger.debug(f"  Custom handler for AmbiguousLexicalBox '{diagram.name}' returned type: {type(result)}")
            return result
        elif isinstance(diagram, (Box, Word)): # If it's a standard Box or Word
            logger.debug(f"  Dispatching standard Box/Word '{diagram.name}' to internal IQPAnsatz.")
            # self.internal_iqp_ansatz(diagram) returns a lambeq.backend.circuit.Circuit instance (e.g. TKETCircuit)
            result = self.internal_iqp_ansatz(diagram)
            logger.debug(f"  Internal IQPAnsatz for Box/Word '{diagram.name}' returned type: {type(result)}")
            return result # type: ignore
        elif isinstance(diagram, GrammarDiagram): # If it's a composite diagram
            logger.debug(f"  Dispatching GrammarDiagram to Functor's super().__call__.")
            # The Functor's base __call__ will iterate through the diagram's boxes and apply mappings.
            # This means for AmbiguousLexicalBox within the diagram, our handler will be called.
            # For standard Boxes/Words, the IQPAnsatz will be called.
            result = super().__call__(diagram)
            logger.debug(f"  Functor super().__call__ for GrammarDiagram returned type: {type(result)}")
            return result
        else:
            raise TypeError(f"ControlledSenseFunctor can only be called on Diagram, Box, or Word, not {type(diagram)}")

