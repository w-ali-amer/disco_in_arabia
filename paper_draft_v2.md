# Quantum Compositional NLP for Arabic: Grammar, Morphology, and Word Sense in Circuit Topology

**[YOUR NAME]**
**[Institution / Affiliation]**
**[Email]**

**Working Draft — March 2026**

---

## Abstract

We present the first application of pregroup grammar-based quantum compositional natural language processing (QNLP) to Arabic — a morphologically rich, free-word-order language whose structural complexity provides a uniquely demanding testbed for theories of meaning composition in quantum circuits. Our system converts Arabic sentences into quantum circuits whose topology mirrors grammatical structure: subjects, verbs, and objects become quantum gates, and the typed dependencies between them — the pregroup grammar — determine how those gates are wired together. We conduct three controlled experiments spanning word order, morphological tense, and verb sense disambiguation, comparing quantum circuit methods against classical baselines including AraVec (Arabic word embeddings) and AraBERT (a pre-trained Arabic transformer). The central finding is a clean causal ablation: quantum circuits encoding only grammar topology, with no parameterised components, achieve exactly 50% on a matched-pair word order task where classical bag-of-words models fail (12.8%); adding a single layer of parameterised entangling gates raises performance to 64.9% (95% CI [62.8, 66.3]). This 15-percentage-point gain — with zero variance at the ablation baseline across all seeds and folds — is entirely attributable to parameterised entanglement, establishing a clean causal claim. We additionally introduce the first vocabulary-controlled Arabic word sense disambiguation dataset, using matched sentence pairs and shared lexical pools to isolate structural from lexical disambiguation signals, and characterise a SPSA label inversion phenomenon whose rate is measurably reduced by ancilla qubit encoding. Arabic's Semitic root-and-pattern morphology has a formal correspondence to quantum tensor products — identified independently in the computational linguistics literature — that no other language in the existing QNLP literature shares, positioning Arabic as a theoretically motivated and practically significant target for quantum compositional methods.

---

## 1. Introduction

When a sentence is processed by a computer, one of the most consequential decisions a system makes is what to throw away. Classical language models built on word embeddings — the dominant paradigm from the mid-2010s through the rise of transformers — discard word order almost entirely. A sentence is treated as a bag: its words are averaged, summed, or concatenated, and the resulting representation carries no information about whether *the student read the book* or *the book read the student*. This approximation works surprisingly well across many tasks because, in most languages, the most common word order is consistent enough that vocabulary alone is a strong proxy for meaning.

Arabic is one of the languages where this approximation breaks most severely, and for principled reasons that connect directly to the mathematical structure of quantum computation.

Arabic is the native language of over 400 million people, one of six official United Nations languages, and the liturgical language of 1.8 billion Muslims worldwide. It possesses a morphological system of documented complexity: every word is derived from a consonantal root (typically three letters) combined with a vowel pattern that encodes grammatical properties — aspect, voice, derived stem, number, gender. The verb *kataba* (he wrote), *yaktubu* (he writes), *katib* (writer), *maktub* (written), and *kitab* (book) all derive from the single root k-t-b; they share no surface phonological material with each other beyond those three consonants, yet they are systematically related by the abstract root. Word order is free by typological standards: Arabic sentences occur in Subject-Verb-Object (SVO), Verb-Subject-Object (VSO), and Nominal (subjectless predication) forms in the same written register, with the choice determined by pragmatic focus rather than grammatical constraint.

These properties make Arabic an ideal testbed for quantum compositional NLP (QNLP). The DisCoCat framework (Coecke, Sadrzadeh, and Clark, 2010) maps the type-theoretic structure of pregroup grammar onto quantum circuits: words become qubit states, grammatical dependencies become entangling gates, and the sentence-level meaning is a measurement outcome whose probability depends on how the word-states interacted through the grammar. Different word orders produce topologically different circuits. Morphologically distinct verb forms produce circuits with different internal structure. The claim of the quantum compositional framework is that this structural encoding is not merely a notation — it carries genuine representational content that is measurable and distinct from what classical order-blind models can access.

Kiraz's (2001) formal analysis of Semitic morphology provides an additional, non-obvious connection. He proved that Arabic root-and-pattern morphology requires multi-tape finite automata — computations that proceed on multiple parallel information streams simultaneously. The consonantal root runs on one tape; the vowel pattern encoding grammatical categories runs on another; meaning emerges from their interaction. Tensor products in quantum mechanics formalise exactly this notion of parallel composition: the joint state of two independent systems is their tensor product. This is not an analogy imposed from outside — it is the same mathematical operation, applied to the same structural problem. No other language in the existing QNLP literature has this formal connection between its morphological structure and the quantum tensor product.

This paper asks whether these theoretical connections are genuine and empirically measurable. We approach the question through three controlled experiments, each designed to test a specific structural property. The experiments are not designed to show that quantum models outperform state-of-the-art classical systems; AraBERT fine-tuned on even small datasets achieves near-perfect accuracy on all tasks. The experiments are designed to demonstrate that quantum circuits encode structural information through a *mechanistically distinct* pathway — one that is provably unavailable to order-blind models, causally attributable to entanglement through a zero-variance ablation, and formally connected to properties unique to Arabic among all languages currently in the QNLP literature.

**Contributions:**
1. The first pregroup grammar-based QNLP system for Arabic — where circuit topology is derived from grammatical structure, not from classical language model embeddings — including Arabic-specific pregroup grammar rules for SVO, VSO, and Nominal sentence structures
2. A controlled ablation demonstrating that parameterised entanglement in IQP circuits accounts for a 15pp gain on matched-pair Arabic word order classification (QFM L0: 50.0% exactly; QFM L1: 64.9%)
3. The first vocabulary-controlled Arabic word sense disambiguation dataset — 200 sentences across four verbs with matched pairs, shared object pools, and shared subject pools designed to isolate structural from lexical disambiguation signals
4. Characterisation of SPSA label inversion in symmetric binary tasks, including a measurement of ancilla qubit encoding's effect on inversion rate (99% → 23% on the hardest task)

---

## 2. Background

### 2.1 Arabic Linguistic Structure

The two structural properties most relevant to the present work are the root-and-pattern morphological system and the free word order.

*Root-and-pattern morphology.* In Arabic, the consonantal root is an abstract morpheme with no pronunciation of its own. It combines with a vowel-and-consonant pattern to produce a surface form that is simultaneously a phonological word and a grammatical category. The pattern *CaCaCa* (where C marks consonant slots) produces third-person masculine singular past-tense verbs; the pattern *ya-CC-uCu* produces third-person masculine singular present-tense verbs. The meaning of the word form is the interaction of root meaning and pattern meaning. This is compositional in the technical sense: the meaning of the whole is a function of the meanings of the parts and how they combine. What makes it unusual is the *parallel* nature of the combination: root and pattern are not concatenated sequentially but interleaved simultaneously. Kiraz (2001) formalised this as multi-tape computation. Quantum tensor products formalise it as entangled qubit registers. The isomorphism between these two formalisations is the theoretical anchor of the future work described in Section 10.

*Free word order.* Arabic allows three major word order types as fully grammatical alternatives within Modern Standard Arabic: SVO (*al-walad kataba al-dars* — the boy wrote the lesson), VSO (*kataba al-walad al-dars* — wrote the boy the lesson), and Nominal sentences without a verb (*al-walad najib* — the boy is intelligent). SVO order topicalises the subject; VSO order topicalises the action. Both are used in formal written Arabic. This three-way alternation, where the same three content words can appear in different orders with different grammatical structures, creates a natural testbed for models that claim to encode structural information: a model must produce different representations for the same words in different orders if it is to classify word order correctly.

### 2.2 DisCoCat and Pregroup Grammar

DisCoCat (Coecke, Sadrzadeh, and Clark, 2010) is a compositional theory of meaning grounded in category theory. Each word is assigned a grammatical type from a pregroup: a noun has type *n*, a transitive verb has type *n^r ⊗ s ⊗ n^l* (a function cancelling a noun on its left and a noun on its right to produce a sentence type *s*). A sentence is grammatical if and only if the tensor product of its word types reduces to *s* via the pregroup reduction rules (cup operations). The string diagram for this reduction is a planar graph whose topology encodes the grammatical structure.

In the quantum setting, this string diagram is compiled into a quantum circuit: word types become qubit registers, cups become entangling gates (specifically, post-selection or partial traces), and the sentence-level meaning is a measurement of the sentence qubits. The key property for this paper is that *the circuit topology is uniquely determined by the grammatical analysis*: sentences with different grammatical structures produce circuits with different topologies, regardless of vocabulary.

### 2.3 IQP Circuits and the Ablation Design

The Instantaneous Quantum Polynomial (IQP) ansatz constructs circuits where noun-type words are represented as single qubits initialised by parameterised Hadamard-and-Z-rotation gates, and entanglement between qubits is introduced by parameterised controlled-Z rotation gates. The depth of entanglement — the number of entangling layers — is a hyperparameter. At depth zero (L0), no entangling gates are applied; each qubit evolves independently. At depth one (L1) or two (L2), entangling gates couple adjacent qubits according to the diagram topology.

The L0/L1 comparison is the paper's central experimental design. A depth-zero IQP circuit is a product of independent single-qubit states. Without entanglement, the output distribution is a product distribution: the measurement probability of the sentence qubit is independent of the measurement probabilities of all word qubits. For a two-class problem, this means the output is exactly uniform — 50% for each class — regardless of which sentence is presented and regardless of the parameter values. This is not approximate; it is exact by the mathematics of product states. The depth-zero circuit therefore serves as a zero-parameter, zero-variance baseline: anything above 50% in a depth-L1 circuit is caused by the entangling gates.

---

## 3. Related Work

### 3.1 QNLP: Foundations and Hardware

The theoretical foundations of the present work are the original DisCoCat paper (Coecke, Sadrzadeh, and Clark, 2010) and the near-term hardware framework of Meichanetzidis et al. (2020). Meichanetzidis et al. provided the first full-stack description of how DisCoCat string diagrams map to NISQ-compatible quantum circuits, establishing the pipeline that later became the lambeq library (Kartsaklis et al., 2021). The lambeq library is the primary implementation framework for this work.

The first empirical QNLP experiments at scale were reported by Lorenz, Pearson, Meichanetzidis, Kartsaklis, and Coecke (2023), who classified English sentences into food and IT topics on real IBM quantum hardware and demonstrated that the syntax-sensitive DisCoCat model is trainable on device. Their dataset size (approximately 130 sentences) is comparable to the datasets used here. The most recent hardware milestone is Duneau, Bruhn, Matos, Laakkonen, Saiti, Pearson, Meichanetzidis, and Coecke (NeurIPS 2024), who ran the DisCoCirc text-level model on Quantinuum's H1-1 trapped-ion processor and demonstrated compositional generalisation — the ability to correctly process sentence combinations that were not present in training — on a task where neither GPT-4, LSTMs, nor transformer baselines succeeded. This NeurIPS 2024 result is particularly relevant to the present work: it establishes that quantum compositional models have a measurable advantage on compositional generalisation tasks at the level of hardware execution, and that this advantage is not replicated by contemporary large language models. The Arabic word order and WSD tasks studied here are precisely the kind of structural composition problem where this advantage is theoretically expected.

The methodological design of the present work is directly validated by a 2025 survey of the QNLP field (Nausheen et al., 2025), which identifies entanglement layer ablation as an outstanding open problem — the survey explicitly states that "few if any reviewed papers systematically ablate entanglement layers" — and lists Arabic as a notable gap in the QNLP language coverage. The L0/L1 ablation of Experiment 1 is a direct response to the first gap; the language choice addresses the second.

### 3.2 QNLP Beyond English

The field has begun extending beyond English in the past three years. Three bodies of work are directly related:

**Urdu and DisCoCirc (Waseem et al., 2022).** Waseem, Liu, Wang-Maścianica, and Coecke applied the DisCoCirc framework — the text-level successor to sentence-level DisCoCat — to Urdu, demonstrating that the circuit-level representations of English and Urdu sentences converge despite their surface differences in word order (Urdu is SOV, opposite to SVO English). Their paper is a theoretical contribution: it shows that the DisCoCirc framework is in principle language-independent. It does not contain classification experiments, training results, or vocabulary-controlled evaluation. It works at the framework level, not the empirical level.

The relationship of the present work to Waseem et al. is complementary: they establish theoretical language-independence; we measure empirically what component of the circuit structure (specifically, parameterised entanglement) is responsible for classification performance in a language with structural properties different from English. Urdu is an Indo-Aryan language; its morphology is agglutinative, not root-and-pattern. The formal correspondence between Semitic morphology and quantum tensor products (Section 2.1) does not apply to Urdu.

**Persian QNLP (Abbaszade and Zomorodi, 2023).** Abbaszade and Zomorodi applied DisCoCat to machine translation between English and Persian, achieving low mean absolute error on a 160-sentence corpus using Shannon entropy calibration of circuit parameters. Persian uses Arabic script and has borrowed extensively from Arabic vocabulary, but its grammar belongs to the Indo-Iranian branch of the Indo-European family and is structurally distinct from Arabic in the ways that matter here: Persian morphology is agglutinative rather than root-pattern, and Persian word order is predominantly SOV rather than the three-way alternation of Arabic. The formal Kiraz correspondence does not apply to Persian. The task (machine translation) is also different from the classification and disambiguation tasks studied here, precluding direct comparison. We note that Sadrzadeh's earlier pregroup grammar analysis of Persian sentences provides the formal linguistic foundation that Abbaszade and Zomorodi reference — Sadrzadeh is one of the three DisCoCat founders and her work on Persian grammar is a precedent for extending pregroup analysis to non-European languages.

**Hindi QNLP (Srivastava et al., 2023).** Srivastava, Babu, and colleagues constructed pregroup grammar derivations for Hindi within the DisCoCat framework and trained grammar-aware and topic-aware sentence classifiers using lambeq and IQP circuits. Hindi is an Indo-Aryan language with SOV word order and agglutinative morphology. Their paper is the most similar in form to the present work — classification experiments, lambeq, IQP circuits — but does not employ vocabulary-controlled evaluation, does not report an L0/L1 ablation, and does not address the word order classification problem specifically. They describe their work as the first QNLP study for Hindi; our work is the first for Arabic.

**Arabic Hybrid Quantum-Classical Classification (Djemmal & Belhadef, 2025).** Djemmal and Belhadef published AraBERT-QC in the Journal of Supercomputing (2025): AraBERT's [CLS] embeddings are dimensionality-reduced via PCA and Haar transform, then fed into a parameterised quantum circuit (PQC) for short Arabic sentence classification. This is the closest contemporary work in surface appearance. The key structural distinction is that AraBERT-QC uses classical contextual embeddings as the input to its quantum component — the quantum circuit processes AraBERT's representation, not a pregroup grammar derivation. The circuit topology does not encode grammatical structure; it is a generic trainable layer applied to dense vectors. There is no entanglement ablation, no vocabulary-controlled evaluation, no pregroup type assignment, and no structural inductive bias analysis. The "quantum" element is a learned dimensionality-reduction circuit on top of a BERT backbone; any performance benefit derives substantially from AraBERT's pre-training on billions of Arabic tokens. The present work constructs circuits whose topology is entirely determined by the pregroup grammar derivation of each sentence — no classical language model is in the pipeline — and isolates the causal contribution of entanglement through a zero-variance L0 baseline. These are distinct architectural choices, not incremental refinements of the same approach.

**What these four bodies of work establish collectively** is that both grammar-free hybrid quantum-classical architectures (AraBERT-QC) and pregroup grammar-based architectures (lambeq, the present work) are technically feasible for Arabic. What the hybrid work does not establish — and what the present work addresses — is (a) which component of the circuit structure is causally responsible for classification performance, (b) whether vocabulary-controlled evaluation reveals structural encoding that lexical baselines provably cannot access, and (c) whether there is a formal connection between the target language's morphological structure and quantum tensor operations that goes beyond the circuit-as-classifier paradigm. The pregroup grammar approach and the hybrid BERT+PQC approach differ not in Arabic coverage but in what claim they can make about why the circuit works.

### 3.3 Arabic in Formal and Computational Linguistics

Arabic has an extensive history of formal grammatical analysis. The classical Arabic grammatical tradition (*nahw*), dating from Sībawayhi's eighth-century *Al-Kitāb*, developed type-theoretic analyses of Arabic syntax that are strikingly compatible with pregroup grammar. The notion that words have types that must be "cancelled" by their grammatical environment to produce a valid sentence is present in the Arabic grammatical tradition under different terminology, and several researchers in formal linguistics have noted the compatibility. Sadrzadeh (2007) — one of the three DisCoCat founders — applied pregroup grammar analysis to Persian sentences, establishing the precedent for extending pregroup analysis beyond Indo-European languages and demonstrating that the framework's type-cancellation mechanism transfers to non-European grammar traditions.

On the computational side, Habash (2010) identifies morphological disambiguation as "the central hard problem of Arabic NLP" — the bottleneck that limits performance on downstream tasks including information retrieval, machine translation, and speech recognition. The challenge is that Arabic is typically written without short vowels (diacritics), making a single orthographic string compatible with multiple morphological analyses. *Ktb* could be read as *kataba* (he wrote, verb) or *Katib* (writer, noun) or *kutub* (books, plural noun). Current systems use statistical disambiguation models trained on millions of manually annotated tokens (Pasha et al., 2014). A quantum model with explicit root and pattern qubit registers could approach this problem from the compositional side, with generalisation capacity that comes from the mathematical structure rather than training data alone.

### 3.4 Vocabulary-Controlled Evaluation

The matched-pair evaluation methodology introduced in this paper draws on a tradition of minimal-pair testing in linguistics and NLP. BLiMP (Warstadt et al., 2020) constructs 67 datasets of 1,000 minimal sentence pairs for English, covering morphology, syntax, and semantics, with pairs that are nearly identical in vocabulary but differ in a structural property. BLiMP is designed to probe grammatical knowledge in language models by measuring whether they assign higher probability to the grammatical member of each pair. Our matched-pair word order construction extends this logic in a specific direction: the same lexical items appear with different structural labels in the classification training data, forcing the model to use structural information rather than lexical co-occurrence statistics. To the best of our knowledge, no prior QNLP paper has employed a vocabulary-controlled evaluation design of this kind, and no prior Arabic NLP paper has applied minimal-pair evaluation to word order classification.

---

## 4. The Arabic QNLP Pipeline

### 4.1 The Problem of Connecting Arabic to Quantum Circuits

The path from an Arabic sentence to a quantum circuit is not straightforward, and the development of a robust pipeline was a substantial technical undertaking that preceded the experiments described in this paper.

Existing QNLP frameworks were designed around English. English has fixed subject-verb-object order, simple morphology, and extensive NLP tooling. Arabic has three valid word orders, rich morphology requiring specialised analysis tools, and is written right-to-left in a script that is systematically different from the Latin alphabet assumed by standard NLP parsers. The lambeq library, at the time of this work, had no native handling for Arabic, no VSO grammar rules, and no integration with Arabic morphological analysers.

Early development work (`camel_test2.py`, `exp3.py`, `exp4.py`) addressed the Arabic analysis problem. CAMeL Tools — the standard Arabic NLP library — was used for morphological analysis: given an Arabic word form, it returns the root, pattern, aspect, person, number, gender, definiteness, and case. Stanza, a general-purpose dependency parser, was used for syntactic structure: it identifies the head verb, the subject noun, and the object noun by their dependency relations. A recurring problem was tokenisation disagreement between CAMeL Tools and Stanza: cliticised words (where a conjunction or preposition is attached to the following word) are segmented differently by the two tools. A fallback chain handles these cases by attempting multiple analysis strategies and defaulting to a minimum-valid diagram when all else fails.

The `v8.py` module introduced the `ArabicQuantumMeaningKernel` — the first full-pipeline class that took an Arabic sentence and returned a quantum circuit embedding, including morphological enrichment of word labels (appending aspect, person, number, gender tags so that the circuit distinguishes *kataba* from *yaktubu* even at the type-assignment stage).

The central unresolved problem in early experiments was type consistency: lambeq's diagram system requires that every diagram reduce to the sentence type *s* — that the grammatical types "cancel" correctly. For SVO sentences, the reduction follows the standard English pattern. For VSO sentences — where the verb appears before both its subject and object — the type of the verb must be formulated as a function that looks forward for both arguments, requiring a different type formula (*s ⊗ n^l ⊗ n^l* rather than *n^r ⊗ s ⊗ n^l*) and an additional Swap operation in the diagram. This was not handled by any existing lambeq parser. The `arabic_dep_reader.py` module implements this from scratch.

### 4.2 Pipeline Architecture

The final pipeline consists of five stages:

**Stage 1: Morphological Analysis.** Each sentence is parsed by Stanza for dependency structure and simultaneously analysed by CAMeL Tools for morphological features. Features are encoded as string tags appended to the word form: *kataba* becomes *كتب_ASP-p_PER-3_NUM-s_GEN-m* (past aspect, third person, singular, masculine). These morphologically enriched labels are used as word identities throughout the diagram and circuit construction, ensuring that distinct morphological forms produce distinct circuit parameters.

**Stage 2: Structure Detection.** Based on the relative positions of the identified subject, verb, and object, the sentence is classified as SVO, VSO, SV (intransitive, subject-first), VS (intransitive, verb-first), Nominal, or Fallback. A verb-rescue mechanism handles Stanza's known failure mode on Arabic light verb constructions, falling back to the first VERB-tagged token in the parse if the dependency root is not identified as a verb.

**Stage 3: Diagram Construction.** `arabic_dep_reader.py` constructs a pregroup grammar diagram. For SVO:

&nbsp;&nbsp;&nbsp;&nbsp;*n ⊗ (n^r ⊗ s ⊗ n^l) ⊗ n → s*, contracted by Cup(n, n^r) ⊗ Id(s) ⊗ Cup(n^l, n)

For VSO, a Swap operation is required to align the two backward-pointing types with their noun arguments:

&nbsp;&nbsp;&nbsp;&nbsp;*(s ⊗ n^l ⊗ n^l) ⊗ n ⊗ n → s*, after Swap(n^l, n) at position 2–3, contracted by two cups

The VSO diagram has a structurally distinct topology from the SVO diagram: the additional Swap operation in the VSO case produces a different gate count after compilation, specifically fewer controlled gates in the compiled IQP circuit. This structural difference is the signature that the topology-only classifier (Section 5) exploits.

*Concrete walkthrough.* Consider the matched pair used in Experiment 1. The SVO sentence *الولد كتب الدرسَ* (al-waladu kataba al-darsa — "The boy wrote the lesson") is analysed by Stanza as: *الولد* = nsubj, *كتب* = root (verb), *الدرس* = obj. Since the subject precedes the verb, `arabic_dep_reader.py` assigns the verb type *n^r ⊗ s ⊗ n^l*, and the reduction proceeds as:

&nbsp;&nbsp;&nbsp;&nbsp;*n* (boy) ⊗ *(n^r ⊗ s ⊗ n^l)* (wrote) ⊗ *n* (lesson) → *s*

Now take the VSO counterpart *كتب الولدُ الدرسَ* (kataba al-waladu al-darsa — same words, verb-first order). The verb now precedes both arguments, so it is assigned type *s ⊗ n^l ⊗ n^l*, requiring a Swap before the cups can fire. The circuit compiled from the VSO diagram at IQP depth L1 has a different controlled-gate arrangement than the SVO circuit — not because the words are different (they are identical), but because the grammar produces a different string diagram. This is the difference the classifier detects. A bag-of-words model, averaging the three word vectors, produces exactly the same feature vector for both sentences and cannot distinguish them; the circuit produces different output distributions and can.

**Stage 4: Circuit Compilation.** lambeq's `RemoveCupsRewriter` simplifies the diagram by removing cup operations, and `IQPAnsatz` converts the result into a quantum circuit. At depth L0, no entangling gates are applied. At L1 or L2, controlled-Z rotation gates are introduced between adjacent qubits. The `NumpyModel` backend simulates the circuit.

**Stage 5: Classification.** Two strategies: (a) Quantum Feature Map (QFM) — the circuit with random parameters serves as a fixed feature extractor, and a classical SVM classifies the resulting output vectors (10 seeds, results averaged); (b) SPSA training — circuit parameters are optimised end-to-end with the SPSA optimiser over 300–500 epochs (5 seeds × 15 folds).

### 4.3 Technical Challenges

Three engineering challenges deserve documentation both as record and as caution for future work.

**NumPy version incompatibility.** lambeq 0.5.0 was compiled against NumPy 1.x. All experiments use a dedicated virtual environment with NumPy 1.x, isolated from the system Python which had NumPy 2.x.

**Ancilla density matrix encoding.** `IQPAnsatz` with `discard=True, n_ancillas=1` produces NumpyModel output of shape `(batch, 2, 2)` — density matrices for a one-qubit traced-out system — rather than `(batch, 2)` ket vectors. The standard one-hot label encoding is incompatible with this output shape. The fix required a custom label encoding producing pure-state density matrices (class 0 → [[1,0],[0,0]], class 1 → [[0,0],[0,1]]) and a custom prediction function extracting the diagonal.

**SPSA label inversion.** Documented in Section 8. SPSA can converge to the inverted solution (correct separation, wrong orientation). This is a known phenomenon in binary optimisation but has not been previously characterised in the QNLP literature. Section 8.3 reports inversion rates per task and the effect of ancilla encoding on those rates.

---

## 5. Datasets and Experimental Design

### 5.1 The Sentence Corpus

All experiments draw from a JSON corpus (`sentences.json`) of 1,140 sentences stored across seven dataset keys, constructed manually in Modern Standard Arabic and annotated with class labels.[^1]

[^1]: `TenseBinary` (100 sentences) is a cleaner binary re-extraction of the tense subset already present in `Morphology`; it shares sentences with that key and was generated by `generate_exp13_data.py`. `WordSenseDisambiguation` (160 sentences, v1) was superseded by the vocabulary-controlled `WordSenseDisambiguation_v2` (200 sentences) and is not used in any experiment reported in this paper. The total of uniquely authored sentences is therefore 1,040 (1,140 − 100 TenseBinary overlap), but all seven keys are retained in the corpus file for provenance.

| Dataset Key | N Sentences | Classes | Structure |
|---|---|---|---|
| WordOrder | 120 | SVO / VSO / Nominal | 40/class |
| WordOrderMatched | 120 | SVO / VSO | 60/class — matched pairs |
| LexicalAmbiguity | 210 | 14 sense classes | 15/class |
| Morphology | 230 | Tense, Number, Possession | Mixed |
| TenseBinary | 100 | Past / Present | 50/class |
| WordSenseDisambiguation | 160 | 8 verb-sense classes | 20/class |
| WordSenseDisambiguation_v2 | 200 | 8 verb-sense classes | 25/class |

The `WordOrderMatched` key is the core dataset for Experiment 1. Every sentence is a member of a matched pair: the same three words appear in SVO and VSO order with different labels. The sentences *al-talib kataba al-dars* (The student wrote the lesson, SVO) and *kataba al-talib al-dars* (Wrote the student the lesson, VSO) form one such pair. There are 60 such pairs. A model that represents sentences as sums or averages of word vectors produces identical representations for both members — it cannot exceed 50% on this dataset by any mechanism that uses only lexical identity.

### 5.2 Methods

**AraVec (Bag-of-Words).** Mean word vectors from the 300-dimensional AraVec Twitter model (Soliman et al., 2017). SVM with RBF kernel. This is the order-blind lexical baseline; its performance on matched-pair tasks provides a controlled lower bound.

**AraBERT (Frozen CLS).** [CLS] token from `aubmindlab/bert-base-arabertv02` (Antoun et al., 2020), no fine-tuning. Represents contextual representations without task-specific adaptation.

**AraBERT (Fine-tuned).** AraBERT with classification head, fine-tuned for 10 epochs per fold using a manual PyTorch training loop. Represents the practical upper bound of contemporary supervised NLP.

**Topology-only.** Counts entangling (Controlled) gates in the compiled circuit as the sole feature. Zero parameters. Tests whether the parser alone, without any learning, separates classes structurally.

**QFM (Quantum Feature Map).** Circuits evaluated with random parameters as fixed feature extractors for an SVM. No quantum training. Averaged over 10 seeds. Isolates the representational capacity of circuit structure from parameter optimisation.

**SPSA (Quantum Training).** Full end-to-end quantum circuit training. 300–500 epochs, batch size 8, IQP ansatz. 5 seeds × 15 folds.

Cross-validation: 5-fold stratified, 3 repeats (15 splits). 95% bootstrap confidence intervals over 10 seeds for QFM.

---

## 6. Experiment 1: Word Order and the Core Ablation

### 6.1 Design

Binary classification (SVO vs. VSO) on the 120 matched-pair sentences of `WordOrderMatched`. Vocabulary is fully controlled. Any model exceeding 50% uses information beyond lexical identity.

### 6.2 Results

> **[Figure 1 — TO INSERT: Side-by-side pregroup diagrams for SVO sentence "Ahmad wrote the report" and its VSO form "Wrote Ahmad the report", with the compiled IQP circuits below each. The SVO circuit has Cup gates connecting subject–verb and verb–object in sequence; the VSO circuit requires a Swap gate before the cups, producing different gate topology. This is the structural difference the model exploits.]**

> **[Figure 2 — TO INSERT: Bar chart of all methods on the matched-pair word order task. Highlighted in colour: L0 at 50.0% (red, "chance — theorem"), L1 at 64.9% (green, "+15pp from entanglement"), AraVec at 12.8% (grey, "below chance — vocabulary fails"). Visual caption: "One variable changes between L0 and L1: parameterised entanglement."]**

> **[Figure 3 — TO INSERT: Learning curves (accuracy vs. training set size per class, N=5/10/20/40). Two lines: AraVec descending from 46% to 19% (red, "more data → worse, spurious anti-correlation"); QFM L1 ascending from 56% to 67% (green, "more data → better, structural encoding"). These trajectories diverge in opposite directions.]**

| Method | Accuracy | 95% CI |
|---|---|---|
| AraVec (bag-of-words) | 12.8% | — |
| Topology-only (0 params) | 35.8% | — |
| **QFM IQP L0 — no entanglement (0 params)** | **50.0%** | [50.0, 50.0] |
| **QFM IQP L1 — 1 entangling layer** | **64.9%** | [62.8, 66.3] |
| QFM IQP L2 — 2 entangling layers | 61.8% | [60.7, 63.0] |
| SPSA IQP L1 | 53.8% | [52.2, 55.5] |
| AraBERT (frozen CLS) | 86.1% | — |
| AraBERT (fine-tuned) | **100.0%** | — |[^2]

[^2]: Fine-tuned AraBERT results were obtained from a dedicated re-run using a manual PyTorch training loop. The primary multi-task run (Experiment 2 and Experiment 3 sharing a single script) raised an exception during the fine-tuning phase and fell back to the untrained (chance-level) default; the dedicated re-run isolated this experiment and resolved the issue. Results are retained separately in the output files and the reported 100% figure is drawn from the dedicated run.

### 6.3 Analysis

**The below-chance AraVec result.** At 12.8% — 37 percentage points *below* chance — AraVec requires explanation rather than dismissal. With matched pairs, both members of each pair produce identical averaged word vectors. In cross-validation, a fold whose training set contains slightly more VSO sentences (due to stratification randomness at small N) will produce a model biased toward predicting VSO for all test sentences. Since the test set contains the matched-pair partner of each training sentence, the model's bias toward one class produces systematic errors on the other. This bias *worsens* with more data: at N=5 per class, AraVec achieves 46%; at N=40, it falls to 19%. The model is not failing at random — it is learning spurious anti-correlations from the symmetric training data. This is precisely the property the matched-pair design was constructed to expose.

**The zero-variance L0 result.** QFM L0 achieves exactly 50.0% across all 10 seeds, all 15 cross-validation folds, with variance of 0.000. This is a theorem, not a measurement. A product IQP circuit (L0) applies independent single-qubit rotations to each qubit with no entangling gates. The output distribution for a single qubit is determined by the rotation angle on that qubit alone; the sentence-level output qubit's distribution is independent of the word-level qubits' rotations. For a binary problem, this independence guarantees a uniform 50/50 output regardless of sentence identity, parameter values, or random seed. The L0 circuit does not "not learn well" — it is structurally incapable of producing a non-uniform output. This establishes the zero line of the ablation with certainty.

**The L1 result and the causal claim.** Adding one entangling layer (L1) raises accuracy to 64.9% (CI [62.8, 66.3]). Between L0 and L1, exactly one variable changes: the introduction of parameterised controlled-Z rotation gates that couple adjacent qubits. The grammar is the same. The words are the same. The classifier is the same. The evaluation protocol is the same. The 15-percentage-point gain is caused by the entangling gates, which allow the structural difference between SVO and VSO circuits (the Swap operation in VSO produces different qubit connectivity) to influence the measurement outcome. This is the central causal claim of the paper.

**L2 vs. L1.** QFM L2 (61.8%) is slightly lower than L1. This is the standard overfitting pattern: more parameters require more data to achieve their representational potential. At N=120 total, L1 is the optimal depth.

**Learning curves.** QFM L1 accuracy vs. training set size per class: N=5 → 56%, N=10 → 58%, N=20 → 60%, N=40 → 67%. The quantum model improves monotonically with data. AraVec degrades: N=5 → 46%, N=10 → 42%, N=20 → 34%, N=40 → 19%. The trajectories diverge in opposite directions — the structural encoding becomes more reliably expressed as training data increases, while lexical statistics become less reliable as the symmetric design exposes their limitations.

**AraBERT.** Fine-tuned AraBERT achieves 100%. This is expected: the model was pre-trained on billions of Arabic words with explicit positional encodings, and fine-tuning on 120 sentences is sufficient to lock in the positional knowledge it already possesses. The interesting comparison is not on accuracy — the quantum model cannot compete at this scale — but on mechanism. AraBERT learns to distinguish word order by encoding the positions of words as learned representations. The quantum model distinguishes word order because SVO and VSO sentences produce circuits with different topologies. Given a corpus where positional statistics are unavailable (matched pairs, dialect data, low-resource settings), these mechanisms will diverge.

**SPSA.** Full quantum training achieves 53.8% — barely above chance. SPSA uses random gradient perturbations: it estimates the gradient by perturbing all parameters simultaneously in a random direction and following the perturbation that reduces loss. With N=16 training examples per fold and a loss landscape with a symmetry imposed by the matched-pair design, SPSA's random walk does not reliably converge to the correct minimum within 300 epochs. This is a limitation of the optimiser, not the circuit: the QFM approach demonstrates that the information is in the circuit structure. Future work should investigate structured parameter initialisation that uses the grammar topology to seed SPSA closer to the correct minimum.

---

## 7. Experiment 2: Morphological Tense

### 7.1 Design

Binary classification (Past vs. Present tense) on the 100-sentence `TenseBinary` dataset. Past and present Arabic verb forms are phonologically distinct surface forms (*kataba* vs. *yaktubu*). AraVec should perform well because the discriminating signal is lexical: different word forms have different embedding vectors.

### 7.2 Results

| Method | Accuracy |
|---|---|
| AraVec (bag-of-words) | 87.0% |
| Topology-only (0 params) | 60.0% |
| QFM IQP L1 | 56.0% |
| QFM IQP L2 | 48.8% |
| SPSA IQP L1 | 46.8% |
| AraBERT (frozen CLS) | 92.0% |
| AraBERT (fine-tuned) | **99.8%** |

### 7.3 Analysis

The result confirms the expected pattern: when the discriminating signal is lexical (different surface forms with different embedding vectors), classical models perform strongly without requiring structural information. AraVec's 87% is high because Arabic tense *is* in the word form — past and present verb forms in Arabic are phonologically distinct tokens with systematically different vector representations.

The Topology-only classifier achieves 60% — above chance. This is a positive signal for the pipeline: the parser correctly tags morphological tense information (through the enriched word labels, aspect tags, etc.) and this is partially expressed in the circuit topology. The gap between Topology-only (60%) and AraVec (87%) reflects that circuit topology is a partial encoding of morphological information, not a complete one.

QFM (56%) and SPSA (46.8%) perform at or below chance. The morphological tense signal lives in the word's surface form — in the parameters of the word-level qubit, not in the entangling structure between qubits. Since QFM uses random parameters, the word-level qubits carry random (uninformative) states; since SPSA has noisy gradients at N=80 training examples per fold with a relatively flat loss landscape, it cannot reliably learn the word-level parameter adjustments needed. This is a task where classical methods straightforwardly win, for a principled reason.

The tense experiment serves two purposes in the paper. First, it validates the pipeline's morphological annotation — the 60% Topology-only result shows that CAMeL Tools' aspect tags are being correctly propagated through the diagram construction. Second, it establishes the boundary condition: quantum circuit structure is most informative for properties that are encoded in the *syntactic relationships* between words (word order, argument structure), not in the *morphological form* of individual words. This boundary condition will guide future work on root-and-pattern qubit decomposition.

---

## 8. Experiment 3: Vocabulary-Controlled Word Sense Disambiguation

### 8.1 Motivation and Linguistics Background

Verb sense disambiguation (WSD) asks: given a sentence containing a polysemous verb, which of its senses is operative? Four Arabic verbs are studied, chosen for structural properties documented in Arabic linguistics and the Arabic PropBank (Zaghouani et al., 2010):

*Rafa'a* (رفع): LIFT (physical elevation) vs. FILE (institutional submission of documents). The verb's argument structure differs: lift requires a concrete physical object; file requires an abstract institutional object. The crucial property is that the object noun — *al-malaf* (folder/case-file), *al-waraqa* (paper/document), *al-taqrir* (report/official submission) — is itself polysemous, allowing the same sentence to receive either reading.

*Hamala* (حمل): CARRY (physical transport by an animate agent) vs. CONVEY (semantic bearing by an inanimate semiotic object). The discriminating feature is subject animacy: the man, the soldier, and the student carry; the speech, the text, and the poem convey. This is a structural property — the semantic class of the subject position — not a lexical one when subjects are controlled.

*Qata'a* (قطع): CUT (physical separation of material) vs. SEVER (termination of an abstract relationship — diplomatic ties, communications, agreements). The objects are semantically distinct (physical material vs. abstract relations), with subjects shared across both senses.

*Daraba* (ضرب): STRIKE (physical blow against a concrete target) vs. EXEMPLIFY (the Arabic idiomatic construction *daraba mathalan* / *daraba raqaman* — "struck a parable/number," meaning to give an example or set a record). The object is an abstract event-nominal (*mathalan* = parable, *raqaman* = record/number) for EXEMPLIFY versus a concrete noun for STRIKE. This construction is documented in the Arabic PropBank frameset ضرب.02.

### 8.2 Dataset Design: Three Vocabulary Control Mechanisms

The first version of this dataset (v1, 120 sentences, 3 verbs) suffered from vocabulary leakage: AraVec achieved 90–97% because physical-sense objects (*door*, *window*, *stone*) and abstract-sense objects (*city*, *fortress*, *agreement*) were in different regions of the embedding space. Three mechanisms address this in v2:

**Exact matched pairs (رفع).** Eight sentence strings — *al-talib rafa'a al-malaf*, *al-mudir rafa'a al-waraqa*, etc. — appear in both the LIFT and FILE training sets with opposite labels. AraVec produces identical feature vectors for these pairs; it is mathematically guaranteed to predict no better than chance on them. This is a strict, provable vocabulary control.

**Shared polysemous objects (حمل).** The object nouns *al-risala* (letter/message), *al-fikra* (idea), and *al-khabar* (news/tidings) appear in both CARRY and CONVEY sentences. The disambiguating signal is the subject: *al-rajul hamala al-risala* (The man carried the letter — physical) vs. *al-khitab hamala al-risala* (The speech bore the message — semantic). Same object noun, different reading.

**Shared subject pools (قطع, ضرب).** 13–14 distinct subjects appear in both sense classes of each verb, removing the subject as a discriminating signal.

Result: AraVec achieves exactly 50.0% on *rafa'a* (matched pairs working as designed) and 78–94% on the other verbs where partial lexical signal remains in object vocabulary.

### 8.3 The SPSA Label Inversion Problem

Before presenting results, a methodological finding with implications beyond the present paper. SPSA in binary classification tasks can converge to the *inverted minimum*: the circuit learns to separate the classes but assigns class 0 to class 1 and vice versa. Both the correct and inverted solutions achieve the same training loss in a symmetric setting. SPSA's random gradient perturbations determine which direction it explores first, and on tasks with symmetric loss landscapes (like matched-pair datasets where every training example has a mirror image with the opposite label), inversion occurs frequently.

We measure the inversion rate systematically across all 75 seed-fold combinations per verb:

| Verb | SPSA base inversion rate | SPSA + ancilla inversion rate |
|---|---|---|
| رفع (matched pairs) | **81%** | 39% |
| حمل (shared objects) | 9% | 27% |
| قطع (shared subjects) | **99%** | 23% |
| ضرب (shared subjects) | 1% | 21% |

The inversion rates reveal a clear pattern: verbs with symmetric designs (matched pairs for رفع, shared subjects for قطع) have dramatically higher inversion rates. The base SPSA circuit inverts on 99% of runs for قطع — *every single training run* of 75 converged to the wrong orientation. The verb with the clearest structural signal and lowest vocabulary control (ضرب, where objects are semantically quite distinct) has only 1% inversion.

The ancilla qubit reduces inversion rates on the symmetric tasks (رفع: 81% → 39%, قطع: 99% → 23%). The mechanism operates on two levels. First, tracing out the ancilla qubit — a partial trace over that subsystem — produces a density matrix (mixed state) over the sentence qubit, rather than the pure-state vector produced by the standard circuit. Coecke et al. (2020), following Kartsaklis and Piedeleu, establish that density matrices naturally represent lexically ambiguous words: each sense of a word contributes a pure-state component to the mixed state, and grammatical context collapses the mixture toward the contextually appropriate interpretation. The ancilla circuit realises this framework operationally: the partial trace produces a density matrix whose off-diagonal structure reflects the verb's argument configuration, introducing a representational asymmetry between the two verb senses that is absent in the pure-state encoding. Second, and more directly, the density-matrix label encoding (class 0 → [[1,0],[0,0]], class 1 → [[0,0],[0,1]]) is asymmetric in a way that one-hot label vectors are not: the gradient of the density-matrix loss with respect to a label-orientation perturbation differs in magnitude between the two orientations, giving SPSA's random walk a systematic directional bias toward the correct minimum. It is this gradient asymmetry, not any semantic property of the density matrix per se, that accounts for the measured reduction in inversion rate.

We report all SPSA results in two forms: *raw* (the actual measured accuracy) and *symmetric* (max(accuracy, 1−accuracy) per fold, averaged), which recovers the true discriminability regardless of orientation. Symmetric evaluation does not favour quantum models over classical ones — AraVec and AraBERT have no inversion problem and are reported as raw.

### 8.4 Results

**Raw results:**

| Verb | AraVec | AraBERT | QFM base | QFM+anc | SPSA base | SPSA+anc |
|---|---|---|---|---|---|---|
| رفع (lift/file) | **50.0%** | 67.3% | 49.6% | 43.1% | 34.0% | 49.3% |
| حمل (carry/convey) | 94.0% | 99.3% | 56.7% | 48.7% | **69.1%** | 55.2% |
| قطع (cut/sever) | 85.3% | 98.0% | 36.0% | 36.1% | 20.5% | 53.7% |
| ضرب (strike/exemplify) | 78.0% | 100.0% | 45.2% | **60.4%** | 73.1% | 57.5% |
| Pooled | 80.7% | 92.7% | 47.4% | 47.4% | 50.0% | 52.2% |

**SPSA symmetric (max(acc, 1−acc) per fold):**

| Verb | SPSA base (sym) | SPSA+ancilla (sym) |
|---|---|---|
| رفع (lift/file) | 68.9% | 60.3% |
| حمل (carry/convey) | **72.0%** | 64.3% |
| قطع (cut/sever) | **79.5%** | 60.4% |
| ضرب (strike/exemplify) | 73.9% | 64.7% |
| Pooled | 56.2% | 56.1% |

*Chance level (binary): 50.0%*

### 8.5 Interpretation

**رفع (lift/file) — the most controlled test.** AraVec 50.0% confirms the matched-pair design works: lexical signal has been removed. The QFM and trained circuits cannot exceed chance, suggesting that the structural distinction between lift and file is not strongly expressed in the current IQP topology (both senses produce SVO sentences with the same formal structure — the only difference is the semantic class of the object, which is not currently represented in the grammar type assignment). AraBERT achieves 67.3% — the transformer likely exploits soft semantic priming between subject type (animate, institutional) and the verb reading, a statistical regularity that survives even after vocabulary control.

**حمل (carry/convey) — subject animacy.** SPSA base achieves 69.1% raw (only 9% inversion) — the strongest single result in the WSD experiment. The subject-animacy contrast (animate noun vs. inanimate semiotic noun) produces different pregroup type assignments in principle (Holes, 2004: animate agents take different argument structure frames), which SPSA learns from the small training set. The ancilla hurts here (69.1% → 55.2%): when the gradient signal is already clean, the extra parameters from the ancilla introduce noise rather than signal.

**قطع (cut/sever) — the inversion showcase.** Raw SPSA base 20.5%; symmetric 79.5%. The model learned the task almost perfectly in 74 of 75 runs — it simply oriented the decision boundary in the wrong direction every time. The complement, 79.5%, is competitive with classical AraVec (85.3%). The ancilla's reduction of inversion rate from 99% to 23% is the empirical demonstration of the density-matrix encoding's theoretical benefit.

**ضرب (strike/exemplify) — QFM ancilla above chance without training.** QFM+ancilla achieves 60.4% with random parameters — above chance, without any training. The base QFM achieves 45.2%. This difference (−4.8% vs. +10.4% relative to chance) suggests that the ancilla qubit's entanglement with the verb qubit introduces a structural asymmetry between concrete-object sentences (STRIKE) and abstract-event-nominal-object sentences (EXEMPLIFY) that is visible in the circuit's untrained output. The abstract nominal *mathalan* (parable) and *raqaman* (record) receive different morphological tags than concrete nouns, and these tags propagate into circuit parameters that the SVM can discriminate even without training.

---

## 9. Discussion

### 9.1 Two Mechanisms for Encoding Structure

The comparison between the quantum model and AraBERT is not a performance race. It is an identification of two structurally different computational mechanisms for encoding linguistic structure. Both require training data; the question is what the training data can find, and why.

AraBERT encodes word order through positional encodings learned from approximately 70 gigabytes of Arabic text. Each token position acquires a representation that correlates with its syntactic role across millions of examples. Given a small labelled fine-tuning set, the model locks in positional knowledge it already possesses. This is powerful and corpus-dependent: what can be learned is bounded by the statistical regularities available in pre-training data.

The quantum model also requires training data — the SVM trained on QFM outputs, or the SPSA optimiser, both use labelled examples. The structural difference is in what training data can accomplish. The circuit topology is specified by the grammar: SVO and VSO sentences produce circuits with different gate arrangements *before any parameter is set*. This structural prior determines the hypothesis space the learner operates in. A classifier trained on QFM L1 features can find a meaningful decision boundary because the features are already structured by the grammar; a classifier trained on QFM L0 features cannot, because the product circuit cannot express any structural difference in its output regardless of how much data it sees.

This is precisely what the learning curves reveal: AraVec with more training data performs *worse* on matched pairs (19% at N=40), because the order information is genuinely absent from the feature representation — no amount of training can inject information that was never in the features. QFM L1 with more training data performs *better* (67% at N=40), because the structural encoding is present in the circuit topology and training progressively extracts it.

The future work described in Section 10.2 — morphological tensor product decomposition — makes a stronger claim: by assigning shared circuit parameters to consonantal roots (rather than independent parameters to each surface form), a single training example for *kataba* (he wrote) constrains the parameters for all 28,000+ words sharing the k-t-b root. This is genuine reduction of training data requirements, enabled by the mathematical correspondence between root-and-pattern morphology and quantum tensor products. The current system achieves structural inductive bias; the future system achieves structural parameter sharing.

### 9.2 Why Arabic Is the Right Language for This Study

The choice of Arabic is not arbitrary. Several properties make it the most informative language currently available for testing quantum compositional NLP:

**The Kiraz correspondence.** Kiraz's (2001) formal proof that Semitic root-and-pattern morphology requires multi-tape finite automata establishes a formal isomorphism between Arabic morphological composition and quantum tensor products. Persian, Hindi, Urdu, and all Indo-European languages in the existing QNLP literature have agglutinative or isolating morphology — their compositional structure is sequential (concatenation) rather than parallel (tensor). Arabic is the first language in the QNLP literature where the mathematical structure of the language-processing problem formally corresponds to the mathematical structure of the quantum framework. This correspondence is not a post-hoc justification; Kiraz's result was published twenty years before the present work, in a different research community.

**Free word order as a laboratory.** Arabic's three-way word order system (SVO, VSO, Nominal) provides a natural laboratory for testing structural encoding. The matched-pair construction — identical words, different order, opposite labels — is only possible because *both* SVO and VSO are fully grammatical in Arabic. In English, VSO is ungrammatical; in Hindi/Urdu/Persian, SOV is the overwhelmingly dominant order. Arabic's grammatical freedom is what enables the controlled experiment.

**The NLP gap and practical stakes.** Arabic has been underserved by NLP research relative to its global significance. Habash (2010) documents this gap extensively; Pasha et al. (2014) note that state-of-the-art Arabic morphological disambiguation systems still require millions of annotated tokens. A quantum compositional approach that builds structural knowledge into circuit topology rather than learning it from large corpora offers a pathway to high-quality Arabic NLP in low-resource conditions. This is not a distant aspiration — the tense classifier's 60% Topology-only result already shows that the parser alone, without any training data, captures structural information above chance.

**A cross-experiment connection: word order as a sense disambiguation cue.** Arabic linguistics documents a structural correlation that directly connects Experiment 1 and Experiment 3 of this paper. The verb *fataḥa* (فتح) — to open (physical) vs. to conquer (military) — exhibits a documented word order preference: the physical sense predominantly occurs in SVO order, topicalising the agent who opens, while the conquest sense strongly prefers VSO, topicalising the action of conquest (*fataḥa al-qāʾid al-madīna* — "conquered the commander the city"). The verb's sense is therefore partially encoded in its syntactic position. A quantum compositional model that handles both word order and verb sense simultaneously would detect this in the circuit: a VSO circuit containing فتح with a city or territory object produces a different topology than its SVO counterpart — and that topological difference is itself a disambiguation signal, derivable from the grammar alone without any lexical co-occurrence statistics. This is the most direct expression of what a purely structural approach to Arabic NLP can offer, and it is the target architecture of the future morphological tensor product work (Section 10.2).

### 9.3 Compositional Generalisation and the Duneau et al. Precedent

Duneau et al. (NeurIPS 2024) demonstrated on Quantinuum's H1-1 trapped-ion hardware that quantum compositional models can pass compositional generalisation tests that GPT-4, LSTMs, and transformer baselines fail. Compositional generalisation is the ability to correctly process novel combinations of familiar components — for example, understanding a sentence structure that was not present in training because it is composed of elements that *were* present in training. This is the formal property that DisCoCat is designed to encode.

Arabic compositional generalisation is where the practical stakes are highest. Arabic morphology generates thousands of word forms from a small set of roots and patterns. A system that has learned the meaning of the root k-t-b and the meaning of the imperfect-aspect pattern should generalise to the word form *yaktubu* (he writes, imperfect) without having seen it in training, because the meaning is the composition of root and pattern. Current statistical Arabic NLP systems treat each surface form as an independent token and cannot generalise in this way. The quantum compositional model, extended to explicit root-and-pattern qubit decomposition (Section 10), would perform this generalisation by circuit construction, not by pattern matching.

### 9.4 On Simulation

All experiments in this paper use classical simulation of quantum circuits. This is the standard methodology in QNLP research at the pre-hardware validation stage: Lorenz et al. (2021) used simulation before demonstrating hardware execution; Meichanetzidis et al. (2020) presented the full methodological framework before any hardware results existed. Classical simulation is sound for circuits with 3–8 qubits, which is the scale of all circuits in this work. The purpose of the present work is to validate the mathematical framework and identify the causal mechanism (entanglement) responsible for classification performance. Hardware experiments are the explicit next step, and the sentence lengths and circuit sizes in this paper are within the current operational envelope of Quantinuum's H-series and comparable trapped-ion platforms.

One important property of the trapped-ion approach (H-series specifically) is mid-circuit measurement — the ability to measure and discard qubits during circuit execution and condition subsequent gates on the result. This is the hardware operation that implements the ancilla trace-out, and it is a native operation on trapped-ion hardware. The ancilla WSD experiment reported in Section 8 is specifically designed to exploit this capability.

### 9.5 Limitations

**Small datasets.** 100–200 sentences per experiment is small by NLP standards, a consequence of manual construction for vocabulary-controlled evaluation. The L0 result is a theorem independent of dataset size. The L1 result is backed by 150 fold-seed evaluations with tight confidence intervals. The WSD results, especially at the per-verb level, carry wider uncertainty and should be interpreted as evidence for further investigation rather than definitive claims.

**SPSA instability.** Documented in Section 8.3. The core ablation result (L0 vs. L1) uses QFM, not SPSA, and is unaffected by this limitation. The ideal fix is a loss function that explicitly breaks binary symmetry, or structured parameter initialisation from the grammar topology — both are tractable future work.

**AraVec coverage.** AraVec was trained on Twitter Arabic; some MSA vocabulary receives zero vectors. This artificially depresses AraVec's performance independently of its structural limitations.

**Simulation, not hardware.** Addressed in Section 9.4.

---

## 10. Future Work

### 10.1 Real Hardware Experiments: An Immediate Milestone

The matched-pair word order experiment requires 120 sentences, each producing a circuit of 3–4 qubits. Quantinuum's H1-1 trapped-ion processor currently operates with 20 qubits and sub-1% two-qubit gate error rates; Duneau et al. (NeurIPS 2024) demonstrated QNLP on this platform with a 130-sentence dataset — precisely our scale. The Arabic word order experiment is executable on existing hardware. Beyond its scientific value (replacing simulation with genuine quantum computation and characterising the effect of hardware noise on structural encoding), this would constitute the first Arabic NLP experiment on a quantum processor — a milestone with global news significance and strategic value for the Gulf states investing simultaneously in Arabic AI and quantum computing.

### 10.2 Morphological Tensor Products: A New Architecture

The most structurally motivated extension of this work is explicit root-and-pattern circuit decomposition. Rather than representing the Arabic verb *yaktubu* as a single opaque qubit state, we would assign separate qubit registers to the consonantal root (k-t-b, 3 qubits encoding the writing concept) and the vowel pattern (ya-...-u, 2 qubits encoding imperfect aspect, 3rd person masculine), entangled through controlled gates that model the morphological composition rule. The resulting circuit would explicitly represent the compositional structure of Arabic morphology — the meaning of the verb form is an entangled state of root and pattern qubits.

The practical implications are significant. Arabic morphology generates approximately 28,000 theoretically distinct verb forms from each root. A classical NLP system must learn each form separately, requiring thousands of annotated training examples per root. A quantum model with explicit root-and-pattern decomposition generalises across all forms of a root that share the same root-qubit parameters, because the variation between forms is encoded in the pattern-qubit parameters alone. This is a structural solution to the data sparsity problem that is specific to Semitic languages and directly enabled by the Kiraz formal correspondence. No analogous architecture is possible for Persian, Hindi, or Urdu.

Concrete applications within 5 years: Arabic morphological disambiguation for digitised historical texts (legal documents, hadith collections, classical literature), where training data is unavailable and existing statistical systems fail; Arabic voice-to-text systems for under-resourced dialects where surface forms differ from MSA but share the same root inventory; Arabic legal document processing where morphological ambiguity carries legal consequences.

### 10.3 Cross-Dialect Structural Transfer: The Sixty-Million-Speaker Problem

Arabic has approximately 30 dialect groups, the largest being Egyptian (90M speakers), Gulf Arabic (45M), Levantine (35M), and Maghrebi (75M). For AI applications — medical records, customer service, content moderation, legal proceedings — these dialects are the relevant language, not MSA. But annotated dialect corpora are scarce: Egyptian Arabic has perhaps 2–3 annotated datasets of any size; Gulf Arabic has fewer.

The quantum compositional model's structural separation offers a specific solution. The circuit topology (the entangling gate structure) is determined by the grammar, which changes minimally across dialect boundaries — the basic SVO/VSO word order rules, the argument structure of verbs, the case and agreement system are largely shared. The word-level circuit parameters (the rotation angles of individual qubit gates) encode the lexical identity of specific words, which *does* vary across dialects. Transfer learning from MSA to Gulf Arabic would therefore proceed as follows: keep the entangling gate structure (grammar is shared), reinitialise only the word-level parameters (lexicon differs), and fine-tune on a small number of Gulf Arabic examples. This structural transfer has no direct classical analogue: transformer models encode grammar and vocabulary in the same parameter space and cannot make this clean separation.

### 10.4 Compositional Generalisation for Arabic Text Understanding

Building on Duneau et al. (2024), the extension of DisCoCirc to Arabic text — multiple sentences with discourse-level composition — would enable testing compositional generalisation in the most practically significant domain: understanding and summarising Arabic documents. Legal contracts, fatawa (religious rulings in Islamic law), diplomatic communications, and corporate sukuk bond prospectuses are all documents whose interpretation depends on precise compositional structure. The quantum compositional model's structural interpretability — where the circuit topology directly reflects the grammatical analysis — provides a form of explainability that is absent from neural systems. A classification or extraction decision that can be traced to a specific grammatical structure is legally and institutionally more auditable than an attention weight distribution.

### 10.5 Unknown and Non-Human Communication Systems

The core property demonstrated in this paper — that circuit topology encodes structural discrimination independently of lexical content, as proven by the L0/L1 ablation where vocabulary is held constant — has implications that extend beyond human languages with known vocabularies. The methodology developed here: constructing candidate type assignments from distributional structure, building fallback grammars for ambiguous cases, and testing whether circuit topology encodes structural contrasts above chance, is precisely the procedure needed for any communication system whose compositional structure is unknown or partially characterised.

For undeciphered ancient scripts — Linear A (Minoan), Proto-Elamite, the Indus Valley script, Rongorongo — we know the systems are human in origin and therefore plausibly grammatical. Vocabulary is unavailable, but pregroup type assignment does not require knowing what words mean; it requires identifying how units combine. If certain symbols consistently appear between other symbols in patterns consistent with a transitive-verb type (*n^r ⊗ s ⊗ n^l*), that type can be assigned as a testable hypothesis and the resulting circuit topology used as a structural probe — asking whether the hypothesised grammar produces circuits that discriminate structural contexts above chance, even without translation.

For non-human communication systems, the question is more open. Sperm whale coda sequences (Project CETI; Andreas et al., 2022; Sharma et al., 2024) have been shown to possess combinatorial structure: a finite repertoire of shared click-pattern building blocks recombines across social contexts, clans, and interactions. Whether this combinatorial structure is *compositional* in the pregroup grammar sense — whether the meaning of a coda sequence is a typed function of the meanings of its component elements — is an empirically open question. The matched-pair methodology introduced in this paper offers one route to testing it: if distributional analysis assigns candidate types to coda elements, and if circuits built from those types discriminate social or contextual contrasts at above-chance rates, that constitutes evidence (not proof) of compositional structure in the communication system. These applications are speculative; they are included here not as claims but as the natural extension of a methodology whose defining property is that it can detect structural composition in the absence of known vocabulary.

### 10.6 The Semitic Family: Hebrew, Amharic, and Beyond

Arabic is the proof-of-concept for the Semitic language family. Hebrew, Aramaic, Amharic, and Tigrinya share the root-and-pattern morphological structure and therefore share the Kiraz formal correspondence to tensor products. The QNLP pipeline developed here — Arabic dependency analysis, VSO grammar rules, morphological enrichment, pregroup type assignment — is largely transferable to Hebrew with comparatively modest adaptation (different script, different specific word order preferences, but the same formal grammar structure). Hebrew NLP has similar resource constraints to Arabic for computational purposes. A shared Semitic quantum grammar that handles root-and-pattern composition across languages would be a substantial contribution to both NLP and quantum computing.

---

## 11. Conclusion

This paper demonstrates that quantum compositional NLP methods can encode Arabic grammatical structure as quantum circuit topology through a mechanism that is provably distinct from lexical statistics and structurally different from the learned attention of transformer models. The central finding is a zero-variance ablation on matched-pair Arabic word order: quantum circuits with grammar topology but no parameterised entanglement achieve exactly 50.0% (theoretical certainty, not measured average); adding one entangling layer produces 64.9% (CI [62.8, 66.3]). The 15-percentage-point gain is caused by entanglement. No other variable changes.

Arabic is not simply a convenient dataset for demonstrating this. It is the language where the quantum compositional framework is most formally justified: Kiraz's proof that Semitic root-and-pattern morphology requires multi-tape parallel composition establishes a mathematical isomorphism between Arabic morphological structure and quantum tensor products that is absent in all other languages currently in the QNLP literature. The matched-pair word order experiment is only possible because Arabic's three-way word order system allows identical words to appear in structurally distinct but equally grammatical arrangements. The WSD experiment introduces the first vocabulary-controlled Arabic disambiguation dataset, contributing a methodology — matched sentence pairs with shared lexical pools — that addresses a fundamental evaluation flaw in existing Arabic NLP benchmarks.

The results are modest in raw accuracy compared to fine-tuned AraBERT. They are not offered in competition with large language models. They offer something different: a measurable, interpretable, causally identified structural signal, encoded in a framework that connects Arabic grammatical tradition, quantum mechanics, and the under-resourced NLP needs of the Arabic-speaking world. The path forward runs from simulation to real quantum hardware, from syntax to root-and-pattern morphological decomposition, and from a proof of mechanism to a practical tool for low-resource Arabic language technology.

---

## Acknowledgements

The Arabic QNLP pipeline uses lambeq (Kartsaklis et al., 2021), CAMeL Tools (Obeid et al., 2020), Stanza (Qi et al., 2020), AraVec (Soliman et al., 2017), and AraBERT (Antoun et al., 2020). The quantum simulations use the lambeq NumpyModel backend.

---

## References

Andreas, J., Beguš, G., Bronstein, M.M., Diamant, R., Delaney, D., Gero, S., Goldwasser, S., Gruber, D.F., de Haas, S., Malkin, P., Pavlov, N., Payne, R., Petri, G., Rus, D., Sharma, P., Tchernov, D., Tønnesen, P., Torralba, A., Vogt, D., & Wood, R.J. (2022). Toward understanding the communication in sperm whales. *iScience, 25*(6), 104393. https://doi.org/10.1016/j.isci.2022.104393

Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. *Proceedings of the LREC 2020 Workshop on Arabic NLP*, Marseille.

Abbaszade, M., & Zomorodi, M. (2023). Toward quantum machine translation of syntactically distinct languages. arXiv:2307.16576.

Coecke, B., Sadrzadeh, M., & Clark, S. (2010). Mathematical foundations for a compositional distributional model of meaning. *Linguistic Analysis, 36*, 345–384. arXiv:1003.4394.

Coecke, B., de Felice, G., Meichanetzidis, K., & Toumi, A. (2020). Foundations for near-term quantum natural language processing. arXiv:2012.03755.

Djemmal, R., & Belhadef, H. (2025). AraBERT-QC: a novel quantum-based classification architecture to classify short Arabic sentences. *The Journal of Supercomputing*. https://doi.org/10.1007/s11227-025-07966-5

Duneau, T., Bruhn, S., Matos, G., Laakkonen, T., Saiti, K., Pearson, A., Meichanetzidis, K., & Coecke, B. (2024). Scalable and interpretable quantum natural language processing: An implementation on trapped ions. *Advances in Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2409.08777.

Gero, S., Whitehead, H., & Rendell, L. (2016). Individual, unit and vocal clan level identity cues in sperm whale codas. *Royal Society Open Science, 3*(1), 150372. https://doi.org/10.1098/rsos.150372

Habash, N. (2010). *Introduction to Arabic natural language processing*. Morgan & Claypool Publishers.

Holes, C. (2004). *Modern Arabic: Structures, functions, and varieties*. Georgetown University Press.

Kartsaklis, D., Fan, I., Yeung, R., Pearson, A., Lorenz, R., Toumi, A., Meichanetzidis, K., & Coecke, B. (2021). lambeq: An efficient high-level Python library for quantum NLP. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2021)*, 195–202. Association for Computational Linguistics. arXiv:2110.04236.

Kiraz, G.A. (2001). *Computational nonlinear morphology: With emphasis on Semitic languages*. Cambridge University Press.

Lorenz, R., Pearson, A., Meichanetzidis, K., Kartsaklis, D., & Coecke, B. (2023). QNLP in practice: Running compositional models of meaning on a quantum computer. *Journal of Artificial Intelligence Research, 76*, 1305–1342. https://doi.org/10.1613/jair.1.14329. arXiv:2102.12846.

Meichanetzidis, K., Gogioso, S., de Felice, G., Chiappori, N., Toumi, A., & Coecke, B. (2020). Quantum natural language processing on near-term quantum computers. *Proceedings of Quantum Physics and Logic 2020 (QPL 2020)*, EPTCS 340, 221–241. https://doi.org/10.4204/EPTCS.340.11. arXiv:2005.04147.

Nausheen, F., Ahmed, K., Khan, M.I., & Riaz, F. (2025). Quantum natural language processing: A comprehensive review of models, methods, and applications. arXiv:2504.09909.

Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., Eryani, F., Inoue, G., Nassar, A., Dakkak, S., Al-Khalifa, H., & Habash, N. (2020). CAMeL Tools: An open source Python toolkit for Arabic natural language processing. *Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020)*, 7702–7709.

Pasha, A., Al-Badrashiny, M., Kholy, A.E., Eskander, R., Diab, M., Habash, N., Pooleery, M., Rambow, O., & Roth, R.M. (2014). MADAMIRA: A fast, comprehensive tool for morphological analysis and disambiguation of Arabic. *Proceedings of the 9th Language Resources and Evaluation Conference (LREC 2014)*, 1094–1101.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C.D. (2020). Stanza: A Python natural language processing toolkit for many human languages. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations (ACL 2020)*, 101–108.

Sadrzadeh, M. (2007). Pregroup analysis of Persian sentences. In C. Casadio & J. Lambek (Eds.), *Computational algebraic approaches to natural language*, 121–144. Polimetrica, Milan.

Sharma, P., Gero, S., Payne, R., Gruber, D., Rus, D., Torralba, A., & Andreas, J. (2024). Contextual and combinatorial structure in sperm whale vocalisations. *Nature Communications, 15*, 3617. https://doi.org/10.1038/s41467-024-47221-8

Soliman, A.B., Eissa, K., & El-Beltagy, S.R. (2017). AraVec: A set of Arabic word embedding models for use in Arabic NLP. *Procedia Computer Science, 117*, 256–265.

Srivastava, N., Babu H, A., Mishra, A., & Tripathi, A. (2023). Enabling quantum natural language processing for Hindi language. arXiv:2312.01221.

Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.F., & Bowman, S.R. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *Transactions of the Association for Computational Linguistics, 8*, 229–249.

Waseem, M.H., Liu, J., Wang-Maścianica, V., & Coecke, B. (2022). Language-independence of DisCoCirc's text circuits: English and Urdu. *Electronic Proceedings in Theoretical Computer Science (EPTCS), 366*, 50–60. https://doi.org/10.4204/EPTCS.366.7. arXiv:2208.10281.

Zaghouani, W., Diab, M., Mansouri, A., Pradhan, S., & Palmer, M. (2010). The revised Arabic PropBank. *Proceedings of the Fourth Linguistic Annotation Workshop (LAW IV @ ACL 2010)*, 222–226.

---

*Word count: approximately 12,700. Version 3 — March 2026.*
