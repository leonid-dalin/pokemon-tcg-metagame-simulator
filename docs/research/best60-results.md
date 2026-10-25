# Best-60 research results

The six-stage run `run-20261025T223336Z-a15a1e44` completed with overall status `inconclusive`. It locked protocol hash `9c31452bd5a3d299ecbce0d619d048b634afb3339be4734d237b30669ecdba59`, code revision `9bbc9c120a37e679fd7da698cb814cf89198e8a4`, and source database SHA-256 `170afff7528e6a38c77100c0d20212b601e0f845d57c9236abc50edde7e5cd59`

## Study design

The primary cutoff was 2 October 2026 with a seven-day embargo. Training, model selection and feature screening used pre-cutoff data. The locked holdout was evaluated once, after the protocol lock. Earlier work had examined corpus-wide support and Kangaskhan outcomes, so this is a retrospective locked historical evaluation

The CPU feasibility profile used FM rank 2, seed 1312, interaction penalty 0.01 and 90-second candidate budgets. The full proposed rank, seed and penalty search was not exhausted

The final validation reused 84 pre-test fits after the saved source, snapshot, configuration, split and field hashes matched. Provenance also records matching package files and an AST comparison outside `_evaluate_benchmark`; the reused validation checkpoint records that test outcomes had not been read

All estimates describe decided-pairing outcomes under the specified data and model procedure. This study does not identify causal card effects or the optimal unobserved list

## Hypothesis decisions

| Hypothesis | Status | Evidence |
| --- | --- | --- |
| H1, predictive improvement | Inconclusive | Interaction log-loss gain was 0.00701 nats per pairing; the 95% interval included zero and only 18 UTC date blocks were available |
| H2, interaction detection | Inconclusive design | No power replicates ran because the available measurement split had fewer than 20 date blocks |
| H3, policy value | Not identified | Exact-list logging propensities and cross-fitted outcome predictions were missing |
| H4, score uncertainty and coverage | Inconclusive | The fixed-list interval was wider than its precision target; calibration coverage was unmeasured |

## H1: Held-out prediction and ranking

On the Kangaskhan-Bouffalant holdout, the additive model had log loss 0.68585 and the selected pair-interaction model had log loss 0.67884. The estimated gain was 0.00701 nats per pairing, with a 95% date-block interval from -0.00703 to 0.01475. It includes zero, so the predictive result is inconclusive. The 248 measured pairing keys spanned 18 UTC date blocks

The ranking diagnostic compared 20 eligible lists, each with at least five matches, and 32 list pairs within the same UTC date block and archetype, with total pair weight 211. Weighted agreement with the outcome residual proxy was 0.6066 for the interaction model and 0.4028 for the additive model. The estimated difference was 0.2038, but the result had no interval at 18 blocks and remains descriptive

The evaluator received UTC date-block IDs for H1 resampling and ranking strata. Some saved objects use generic event labels, including `event_cluster`; the effective units were date blocks. The sensitivity interval that also resampled both player identities was [-0.02332, 0.03312] and does not correct unmeasured confounding

## H2: Interaction evidence and simulation

The planned simulations requested 500 replicates at each log-odds effect of 0, 0.08, 0.2 and 0.4 for both the focal and pooled benchmarks. They ran zero replicates. The pre-test context contained 70 atomic UTC date blocks, which yields at most 14 measurement blocks under the configured 60/20/20 split, below the 20-cluster H1 minimum. The simulation status is `inconclusive_design`; zero completed replicates is not a measured zero-power result

The separate Lumiose training-information audit covered 1,642 focal views, 1,631 unique pairings, 104 supported copy slots and 66 date blocks. Under its assumed probability of 0.5, it reported a local approximate 80% MDE of 4.138 log odds for Lumiose City copy 1 and 3.138 to 11.698 log odds for the selected copy interactions. These are very large local approximations relative to the protocol's useful 0.2 log-odds interaction. The audit used no prior precision, and its cluster count does not estimate dependence inflation or empirical detection power

The pooled validation grid had two candidate fit timeouts and no complete interaction selection for held-out evaluation. The available results do not establish an additive null or the absence of interactions

## H3: Policy value

The target exact-list policy value is `not_identified`. The 248 orientation views, representing 248 unique pairings, lacked exact-list logging propensities and cross-fitted outcome predictions, so no policy-value estimate, confidence interval or clipped sensitivity estimate is available. Catalogue scores are model-selection outputs and do not measure policy value

## H4: Fixed-list range and recommendation

For the selected exact list, the fixed-list bootstrap mean was 0.4977 with a 95% block-resampling interval from 0.3755 to 0.6155. All 200 refits completed. The interval width was 0.2400, above the 0.10 target. Latent-mean and future-aggregate coverage were unmeasured, and the conditional interval does not establish calibration or post-selection coverage

The interaction and additive catalogue selectors chose the same observed 60-card list, with three Lumiose City. Their catalogue scores were 0.49084 and 0.47009, respectively. The corresponding vote-based lists each had one Lumiose City, with scores of 0.43134 and 0.42784. The selected catalogue list had one supporting observed list. This window did not reproduce the brief's stadium-free vote. These scores provide no supported uplift claim

The catalogue contained 225 inputs; 39 passed the available copy, size, ACE SPEC and companion checks, while 186 were excluded. Format metadata was missing, so format-specific restrictions could not be checked. Treat the selected list as a research candidate, not as a format-validated or deployment-ready recommendation

## Sensitivity and software checks

The 14-day embargo audit compared retained event, pairing and list counts only. It did not measure predictive sensitivity. Nested selection coverage, latent-mean coverage, future-aggregate coverage and formal confounding bounds remain unmeasured

The full test suite passed 679 tests. The mutation sweep caught 103 targeted mutants at revision `0b97e6d95b8fb9c2069ace7827f468d09d3c7de2`; the later evaluation-only correction left the 17 mutation-target files unchanged. The quality audit reported no blocking findings and one pre-existing warning

The run reached `reported` with every hypothesis status recorded. The saved report, simulation output, validation provenance and locked protocol provide the detailed machine-readable evidence
