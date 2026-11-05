# Frozen pipeline v2

Determinative proposals for an annotation assistant. The pipeline is frozen before it is evaluated on
new annotated sequences: from the freeze on, no component, prompt, table or parameter may change. A
change requires a new freeze with a new name.

## Setting and input

The annotator has already transliterated and translated the sequence (in this project no automatic
transliteration exists). For every sequence the pipeline needs:

| field | content |
|---|---|
| `id` | sequence identifier |
| `mdc` | the sequence in Manuel de Codage with Gardiner codes |
| `transliteration` | the annotator's transliteration (Egyptological characters, `⸗` before suffix pronouns) |
| `translation` | the annotator's translation |

Output: for every sign position, whether it is proposed as a determinative, with the evidence behind
the decision.

## Components

1. **Rules.** The rule scorer of v1.47 (`egydet.rules`, `egydet.scoring`, knowledge base `kb/*.yaml`).
   One parameter set, tuned on the 99 annotated sequences with the same objective, sampler, warm start
   and inner 5x2 procedure as the cross-validation (5 random halvings, seed 42; best validation wins).
2. **Phonetic veto.** `egydet.rules.align` assigns every sign to a word of the transliteration using the
   phonogram values of JSesh (`kb/jsesh_sign_values.json`, extracted from JSesh's
   `signs_description.xml`, S. Rosmorduc, CC-BY 4.0). A determinative proposed by the rules is removed
   when its sign writes letters of its word, or is a uniliteral phonetic complement.
3. **Slot rescue.** The alignment marks determinative slots: signs at the end of a word that write no
   new sound. A slot is added when the verifier agent judges it a determinative with probability
   p >= 0.5 (threshold fixed in advance). Verifier: prompt `verifier_v1`, context condition (MdC, sign
   indices, knowledge-base notes, transliteration, translation), model `gpt-6-luna`, one call per
   sequence, strict JSON output.

Final proposal = (rules minus vetoed) plus rescued slots. If the verifier call fails, the sequence gets
rules plus veto, and the record is flagged.

## Commands

```bash
python -m egydet.pipeline freeze                       # once: writes configs/pipeline/frozen_v2.json
python -m egydet.pipeline run --input new.xlsx --out runs/frozen_v2_new
python -m egydet.pipeline run --corpus-unannotated --out runs/frozen_v2_pool
python -m egydet.pipeline run --gold data/gold/gold_v1.jsonl --mode offline --out runs/frozen_v2_gold
python -m egydet.pipeline evaluate runs/frozen_v2_new --gold data/gold/<new gold file>.jsonl
```

`freeze` stores the rule parameters, the configuration and the SHA-1 of every file that defines the
pipeline (knowledge base, sign values, prompt, rule and scorer code, the pipeline module itself).
`run` stops if any of these files changed. `run` writes `predictions.jsonl` (decisions and evidence:
rule output, vetoed signs, slots, rescued signs, verifier probabilities and reasons, word spans) and
`proposals.xlsx` (one row per sequence, proposed determinatives in brackets). LLM calls are cached; a
`run` can be resumed.

## Evaluation on new sequences (protocol)

1. **Selection.** Draw the sequences to annotate at random (fixed seed, recorded) from the corpus
   sequences not yet annotated, before running the pipeline on them. Do not select by the pipeline's
   output.
2. **Blind annotation.** Annotators mark determinatives without seeing any system proposal, so that the
   reference does not anchor on the output being evaluated. The transliteration and translation are made
   as usual and are given to the pipeline unchanged.
3. **Agreement.** A subset (at least 30 sequences) is annotated independently by two annotators;
   report Cohen's kappa at the sign-position level and adjudicate disagreements before scoring.
4. **Primary comparison, stated in advance.** Frozen v2 against the rules alone, macro-F1 at the
   sign-position level, paired bootstrap over sequences (10,000 resamples) with a 95% interval and a sign
   test. Secondary: rules + veto, per-sign errors.
5. **Size.** With 99 sequences the 95% interval of a paired macro-F1 difference has a half-width of
   about 3.5 points; at least 100 new sequences, preferably 150 to 200.
6. **Optional second round** (usability): annotators correct the pipeline's proposals; measure time per
   sequence and the changes they make.
