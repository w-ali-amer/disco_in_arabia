"""One-shot repo reorganisation: archive the closed experiment eras, keep the
active trained-block line (exp40-43) flat at the repo root.

Design constraints:
  * The ACTIVE line stays at root. Every script in it resolves data by bare
    relative filename (exp42_compiler.DATA_PATH, EMB_PATH, "stories_exp43b_%s.json",
    exp43b_ckpt/, fresh_results_path) and the results JSONs are cited by name in
    private docs 22-24. Moving any of it would break paths or break citations for
    no gain.
  * Everything else moves with `git mv`, so history follows the file.
  * Archived eras keep code + data + results TOGETHER, so an era stays runnable
    after a cd (cross-era imports still need PYTHONPATH -- see archive/README.md).

Dry run by default. Pass --go to execute.
"""
import os
import re
import subprocess
import sys

GO = "--go" in sys.argv

KEEP_EXACT = {
    ".gitignore", "README.md", "requirements.txt", "requirements_mac.txt",
    "port_sync.sh", "port_parity_check.py", "port_speed_bench.py",
    "reorganize_repo.py",
    "torch_sv_sim.py", "duneau_task_gen.py",
    "duneau_mini_data.json", "duneau_mini_data_b.json",
    "exp40_mini_duneau.py", "exp41_story_generator.py",
    "validate_exp41_data.py",
}
KEEP_PREFIX = ("exp42_", "exp43a_", "exp43b_",
               "stories_exp41", "stories_exp42", "stories_exp43",
               "results_exp40", "results_exp41", "results_exp42",
               "results_exp43")

PARSER = {
    "arabic_dep_reader.py", "arabic_discocirc_pipeline.py",
    "arabic_morpho_lex_core.py", "camel_test2.py", "common_qnlp_types.py",
    "audit.py", "audit_enrich.py", "audit_before.json", "audit_after.json",
    "audit_enrich.json", "units.py", "probe_units.py", "test_pos_fusion.py",
    "patch_idafa.py", "fix_dataset_pair_ids.py",
}
PAPER = {
    "baseline_binary.py", "baseline_classical.py", "generate_exp13_data.py",
    "generate_exp14_data_v2.py", "generate_figures.py", "visualize_exp13.py",
    "visualize_results_v2.py", "reprocess_exp14_symmetric.py",
    "dump_aravec_words.py", "dump2.py", "inspect_noun_box.py",
    "twin_audit.py", "probe_ops.py", "sentences.json", "exp16_wordvecs.json",
}
GEOMETRY = {
    "verify_v2.py", "fix_swap.py", "fix_surgery.py", "fix30.py", "diag28.py",
    "rung25_bloch.py", "rung25_bloch_v2.py", "ERRATUM.md",
    "EXP23_FAMILY_REVIEW.md", "RESULTS_EXP21.md", "RESULTS_EXP21_v2.md",
    "exp23_roots.json", "texts_exp27.json", "texts_exp27_reviewed.json",
}
FRAMES = {"s1a_stream_corpus.py", "s1b_census_harvest.py",
          "s1c_solve_bank.py", "exp35_job.json", "exp35_meta.json"}

DIRS = {
    "figures": "02_discocat_paper",
    "qnlp_experiment_outputs_per_set_v2": "02_discocat_paper",
    "exp16_analog": "02_discocat_paper",
    "dev_history": "05_dev_history",
}

BUCKETS = ["01_parser_pipeline", "02_discocat_paper", "03_semantic_geometry",
           "04_frames_scaling_hardware", "05_dev_history"]


def era_of_expnum(n):
    if n <= 20:
        return "02_discocat_paper"
    if n <= 32:
        return "03_semantic_geometry"
    if n <= 39:
        return "04_frames_scaling_hardware"
    return None                                   # 40+ is active, stays at root


def classify(name):
    if name in KEEP_EXACT or name.startswith(KEEP_PREFIX):
        return None
    if name in PARSER:
        return "01_parser_pipeline"
    if name in PAPER:
        return "02_discocat_paper"
    if name in GEOMETRY:
        return "03_semantic_geometry"
    if name in FRAMES:
        return "04_frames_scaling_hardware"
    m = re.match(r"^(?:results_|stories_|fig_)?exp(\d+)", name)
    if m:
        return era_of_expnum(int(m.group(1)))
    if name.endswith(".npz"):                      # exp21/exp18 state dumps
        return "03_semantic_geometry"
    return "UNCLASSIFIED"


def main():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True).stdout.split("\n")
    root_files = [f for f in tracked if f and "/" not in f]

    plan, unclassified = {}, []
    for f in root_files:
        b = classify(f)
        if b == "UNCLASSIFIED":
            unclassified.append(f)
        elif b:
            plan.setdefault(b, []).append(f)
    for d, b in DIRS.items():
        if os.path.isdir(d):
            plan.setdefault(b, []).append(d + "/")

    kept = [f for f in root_files if classify(f) is None]
    print("PLAN — %d root entries move, %d stay at root\n"
          % (sum(len(v) for v in plan.values()), len(kept)))
    for b in BUCKETS:
        items = sorted(plan.get(b, []))
        if not items:
            continue
        print("archive/%s  (%d)" % (b, len(items)))
        print("   " + " ".join(items[:14]) + (" ..." if len(items) > 14 else ""))
    print("\nSTAYS AT ROOT (%d): active exp40-43 line + infra" % len(kept))
    print("   " + " ".join(sorted(kept)))
    if unclassified:
        print("\n** UNCLASSIFIED (%d) — refusing to guess, left at root:"
              % len(unclassified))
        print("   " + " ".join(unclassified))

    if not GO:
        print("\n(DRY RUN — nothing moved. Re-run with --go)")
        return

    for b, items in plan.items():
        dest = os.path.join("archive", b)
        os.makedirs(dest, exist_ok=True)
        for it in items:
            src = it.rstrip("/")
            subprocess.run(["git", "mv", src, os.path.join(dest, os.path.basename(src))],
                           check=True)
    print("\nmoved. `git status` now shows the renames.")


if __name__ == "__main__":
    main()
