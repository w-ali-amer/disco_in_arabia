# common_qnlp_types.py (Corrected Type Checks & Dummies)
import logging
from typing import Union, Optional, Any 
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
class _DummyBox:
    def __init__(self, name: str, dom, cod, data=None, _dagger=False):
        self.name = name; self.dom = dom; self.cod = cod
        self.data = data if data is not None else {}; self._dagger = _dagger
    def __str__(self): return self.name
class _DummyGDiagram: pass
class _DummyCup: pass
class _DummyId: pass 
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

        # Check if N_ARABIC and S_ARABIC are instances of Ty and have correct string representation
        if isinstance(N_ARABIC, Ty) and str(N_ARABIC) == 'n' and \
           isinstance(S_ARABIC, Ty) and str(S_ARABIC) == 's': # type: ignore
            ARE_N_S_PROPER_ATOMIC_TYPES = True
            logger.info("common_qnlp_types: N_ARABIC and S_ARABIC are Ty('n') and Ty('s') respectively. Considered proper for Box creation.")
        else:
            logger.critical("common_qnlp_types: N_ARABIC or S_ARABIC are not proper Ty('n')/Ty('s') after assignment. This is problematic.")
            # Fallback to creating new Ty instances if the above failed
            N_ARABIC = Ty('n') if Ty is not _DummyTy else _DummyTy('n') # type: ignore
            S_ARABIC = Ty('s') if Ty is not _DummyTy else _DummyTy('s') # type: ignore
            logger.info(f"common_qnlp_types: Fallback N_ARABIC: {N_ARABIC}, S_ARABIC: {S_ARABIC}")
            # Re-check if these fallbacks are usable (they should be if Ty is not _DummyTy)
            if isinstance(N_ARABIC, Ty) and str(N_ARABIC) == 'n' and isinstance(S_ARABIC, Ty) and str(S_ARABIC) == 's': # type: ignore
                 ARE_N_S_PROPER_ATOMIC_TYPES = True # If fallback Ty objects are fine

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

# --- Standalone Diagnostic Block ---
if __name__ == "__main__":
    logger.info("\n" + "="*30 + "\nCOMMON_QNLP_TYPES DIAGNOSTIC RUN\n" + "="*30)
    logger.info(f"Python version: {sys.version}")
    try:
        import lambeq
        logger.info(f"Lambeq version: {lambeq.__version__}")
    except Exception as e:
        logger.error(f"Could not get lambeq version: {e}")

    logger.info(f"LAMBEQ_IMPORTED_SUCCESSFULLY: {LAMBEQ_IMPORTED_SUCCESSFULLY}")
    logger.info(f"Exported AtomicType: {AtomicType} (type: {type(AtomicType)})")
    logger.info(f"Exported Ty: {Ty} (type: {type(Ty)})")
    logger.info(f"Exported Box: {Box} (type: {type(Box)})")
    
    if LAMBEQ_IMPORTED_SUCCESSFULLY and AtomicType is not _DummyAtomicType: # type: ignore
        noun_attr, sentence_attr = None, None
        try:
            noun_attr = AtomicType.NOUN # type: ignore
            sentence_attr = AtomicType.SENTENCE # type: ignore
            logger.info(f"  AtomicType.NOUN: {noun_attr} (type: {type(noun_attr)})")
            logger.info(f"  isinstance(AtomicType.NOUN, AtomicType): {isinstance(noun_attr, AtomicType)}") # type: ignore
            logger.info(f"  isinstance(AtomicType.NOUN, Ty): {isinstance(noun_attr, Ty)}") # type: ignore
            logger.info(f"  AtomicType.SENTENCE: {sentence_attr} (type: {type(sentence_attr)})")
            logger.info(f"  isinstance(AtomicType.SENTENCE, AtomicType): {isinstance(sentence_attr, AtomicType)}") # type: ignore
            logger.info(f"  isinstance(AtomicType.SENTENCE, Ty): {isinstance(sentence_attr, Ty)}") # type: ignore
        except Exception as e_attr_check: 
            logger.error(f"  Exception accessing/checking NOUN/SENTENCE attributes: {e_attr_check}")

    logger.info("-" * 30)
    logger.info(f"N_ARABIC defined as: {N_ARABIC} (type: {type(N_ARABIC)})")
    logger.info(f"S_ARABIC defined as: {S_ARABIC} (type: {type(S_ARABIC)})")
    logger.info(f"ARE_N_S_PROPER_ATOMIC_TYPES: {ARE_N_S_PROPER_ATOMIC_TYPES}")
    
    logger.info("-" * 30)
    logger.info("Attempting to create a simple Box(N_ARABIC, S_ARABIC) using exported Box:")
    if ARE_N_S_PROPER_ATOMIC_TYPES and Box is not None and Box is not _DummyBox: # type: ignore
        try:
            test_box = Box("TestBox", N_ARABIC, S_ARABIC) 
            logger.info(f"  Successfully created TestBox: {test_box} (dom: {test_box.dom}, cod: {test_box.cod})")
        except Exception as e_box_test: 
            logger.error(f"  FAILED to create TestBox: {e_box_test}", exc_info=True)
    else: 
        logger.warning("  Skipping TestBox creation due to improper AtomicTypes or Box class.")

    logger.info("-" * 30)
    logger.info(f"VERB_TRANS_TYPE_ARABIC: {VERB_TRANS_TYPE_ARABIC} (type: {type(VERB_TRANS_TYPE_ARABIC)})")
    logger.info(f"LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY: {LAMBEQ_TYPES_INITIALIZED_SUCCESSFULLY}")
    logger.info("="*30 + "\nDIAGNOSTIC RUN FINISHED\n" + "="*30)

