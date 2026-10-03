# Secrets leak through decisions: an explicit outline removes involuntary secret leakage in LLM writing

**Question.** When a language model writes a story while holding a secret it was told not to reveal
(Holtzman & West 2026, arXiv:2605.10794), the story still betrays the secret. Is the leak a by-product
of the model making its *content decisions* while holding the secret? If so, giving it an explicit
outline to follow (written without the secret) should close the channel even though the secret stays in
context. And what does the secret look like inside the model while it writes?

**Writer:** `google/gemma-3-12b-it` (local, bf16, one RTX A6000). **Guessers:** the writer itself and
`Qwen/Qwen2.5-14B-Instruct`, both local, forced-choice via first-token log-probs (order-debiased).
The OpenRouter key's daily budget was exhausted before any guesser call succeeded and the OpenAI key was
invalid, so no API model is used anywhere.

## Main findings

| Writer condition (word secret, n=360 each) | 2AFC Gemma | 15-way ID Gemma | 2AFC Qwen | 15-way ID Qwen |
|---|---|---|---|---|
| Secret, "don't reveal" | 0.73 | 0.36 | 0.58 | 0.29 |
| Secret + **irrelevant** outline (length-matched, to be ignored) | 0.81 | 0.48 | 0.61 | 0.40 |
| Secret + **outline to follow** (written without the secret) | 0.39 | 0.08 | 0.33 | 0.07 |
| Secret + self-written outline (written while holding the secret) | 0.72 | 0.42 | 0.58 | 0.40 |
| Secret + decoy word | 0.85 | 0.24 | 0.78 | 0.24 |
| Secret, "actively hide" | 0.38 | 0.07 | 0.22 | 0.06 |

Chance: 0.50 (2AFC), 0.067 (15-way ID).

1. **An outline written without the secret removes the leak** (identification at chance, 2AFC below
   chance, like "actively hide"). A length-matched irrelevant outline does *not* reduce leakage, so the
   effect is not context dilution. A self-written outline does not help because the outline itself leaks
   (secret identifiable from the outline 36-41% of the time) and stories whose outline leaked leak most.
2. **Plot-level secrets (a twist that must not be foreshadowed) behave the same way, more weakly:**
   4-way twist identification from the opening scene 0.35 (chance 0.25) -> 0.26-0.27 with a twist-blind
   outline; irrelevant outline 0.30-0.33 (n.s. vs. no outline).
3. **The secret never leaves the residual stream.** The secret word is decodable with ~100% accuracy
   (15-way, held-out premises) from the contribution of the secret instruction to the residual stream
   at every layer >= 4 and in every quarter of the story, in *every* condition, including the outline
   condition where nothing reaches the text. Text-only decodability (same story re-read without the
   secret) is 16-29% in leaking conditions and at chance with an outline.
4. **What the outline changes is the coupling, not the representation.** Per-story strength of the
   secret's representation predicts per-story leakage within word (Spearman rho = 0.50 with the Gemma
   guesser's 2AFC log-odds) in the plain condition, and not at all (|rho| < 0.05) with an outline. Under
   "actively hide" the correlation is negative.
5. **Ablation closes most of the channel.** Mean-ablating the word's own difference-of-means direction
   (estimated on other premises) from every layer during generation cuts 15-way identification from
   0.33 to 0.09-0.14 (Gemma guesser) and 0.27 to 0.11-0.13 (Qwen), with unchanged rated quality and
   length; another word's direction or a random direction changes nothing (0.27-0.34).
6. **Logit lens.** The instruction primes the secret token at mid-depth (+0.3 logit at layers 32-40 vs.
   the other words) and suppresses it at the output (-0.8 nats). With an outline the mid-depth priming is
   absent although the word is still 100% decodable; under "actively hide" it is reversed.

Everything (numbers, tables, figures) in `paper_draft/main.pdf` is produced by the scripts below from
the raw outputs in `results/`.

## Layout

```
src/common.py          prompts, words, premises, local writer (with hook support), local forced-choice guesser
src/twists.py          12 twist premises x 4 candidate twists, twist prompts
src/exp1_generate.py   Experiment 1: word secrets, all conditions (plans, stories)       -> results/exp1/*.jsonl
src/exp2_generate.py   Experiment 2: plot twists, opening scenes                          -> results/exp2/*.jsonl
src/guess.py           leakage measurement (2AFC both orders, 15-way ID, 4-way twist, quality) -> results/*/guess_<guesser>_<cond>.jsonl
src/exp3_acts.py       residual-stream activations, teacher-forced, with/without the secret -> data/acts/*.npy
src/exp3_analyze.py    probes by layer/position, strength-vs-leak, cross-condition transfer, logit lens -> results/summary_mech.json
src/exp3_ablate.py     directional ablation during generation (held-out premises)          -> results/exp3/*.jsonl
src/analyze_behav.py   behavioural statistics (cluster bootstrap), figures                  -> results/summary_behav.json
src/make_tables.py     LaTeX tables from the summaries                                     -> paper_draft/tables/
src/run_chain*.sh      the sequential GPU pipelines that were actually run
src/pilot*.py          pilot used to choose the design
results/               all raw generations and judgments (jsonl) and the two summary files
paper_draft/           main.tex, refs.bib, figures/, tables/, main.pdf
logs/                  run logs
```

## Rerun

```bash
uv venv .venv --python $(which python3) && source .venv/bin/activate
uv pip install "torch==2.8.0" transformers accelerate huggingface_hub openai numpy scipy pandas matplotlib scikit-learn statsmodels sentencepiece protobuf
export HF_HOME=$PWD/models/hf        # weights are downloaded here (gated: needs HF_TOKEN with Gemma access)
python src/exp1_generate.py          # ~4 h on one 48 GB GPU
python src/guess.py gemma12 exp1 && python src/guess.py qwen14 exp1
python src/exp3_acts.py && python src/exp3_analyze.py
python src/exp3_ablate.py && python src/guess.py gemma12 exp3 && python src/guess.py qwen14 exp3
python src/exp2_generate.py && python src/guess.py gemma12 exp2 && python src/guess.py qwen14 exp2
python src/analyze_behav.py && python src/make_tables.py
cd paper_draft && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
All generation scripts are resumable (one jsonl per condition). `data/` and `models/` are git-ignored.
