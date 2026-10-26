# Arabic axis: where a qubit helps (exp90–133)

`proof_qrac_cap*.py` hold the certificate for the composition cap, and `pasqal_*.py` the neutral-atom emulation of the one-qubit game.

| exp | question | verdict |
|---|---|---|
| 90–91 | Case as an operator vs its classical twin | the twin is as fast |
| 92 | Does a learned wazn operator transfer? | no; a fixed swap does (0.87–0.90) |
| 93 | Supplied routing with a learned agreement control (pro-drop) | yes, 0.948 / 0.971 against an exact 0.5 floor |
| 94–98 | Structure and depth advantage over a transformer | no |
| 99 | One qubit per noun storing two features | 0.854 vs 0.75; composition cap proved for three features |
| 100 | Do Arabic agreement readers ask single features? | no, they nest; qubit = bit |
| 101 | Which-referent game, six agreement classes | 0.883 vs 0.800 (uniform); +0.029 on real frequencies |
| 102 | Does syntax select by root or pattern alone? | almost never (0.26% / 0.89%) |
| 103 | Two readers sharing a qubit | three classes yes, six no |
| 104–105 | Neutral-atom emulation of the game | all arms clear the classical value with fast pulses |
| 106 | Root × template store | uniform yes; real data no (0.001) |
| 108 | Fixed one-qubit router vs learned control | matches with zero parameters |
| 109 | Contextuality in Arabic homographs | none, in usage or in a language model |
| 110 | Adjective attachment as a storage game | real text: qubit = bit |
| 112 | Deflected agreement, contextuality | none (0/20) |
| 113, 116, 118, 132 | Two qubits vs two bits for three and four referents | exact classical values 13/18 and 812/1440; two qubits beat both |
| 114–115 | One pulse per word | beats the bit only at three and five classes |
| 117, 119 | The paper's tense and sense results, re-run | the 56% tense result was a parser artefact |
| 120–121 | Root–pattern factorisation, quantum vs fair classical comparators | no quantum-specific gain |
| 124 | Coreference as quantum optimisation | no value |
| 127 | Certified quantum upper bound (semidefinite relaxation) | K = 3 in [0.7781, 0.8203]; K = 4 ≤ 0.6608 |
| 128 | Two atoms under one global laser, noiseless | 0.739 vs 2/3 |
| 133 | Noise robustness of the two-qubit codes as dimension witnesses | critical visibility 0.874 (K = 3), 0.823 (K = 4) |

Scripts here import modules and read data files from other folders by bare name. To re-run one, use `tools/flatten_for_rerun.sh` from the repository root (see the main README).
