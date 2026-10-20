# M1.5 executed learning validation

Date:2026-10-20, America/New_York. Primary evaluation: **teacher-forced, time-conditioned category prediction**. This reduced-modality occurrence diagnostic supplies no unconditional forecasting, prospective clinical prediction, zero-shot rollout or paper-fidelity result.

The candidate learned restricted held-out category structure: validation NLL improved from3.419079 at initialization to0.611644, below the unigram1.219380 and transition0.783077 comparators. Selected validation accuracy is68.12% versus63.53% for transitions. Learning is incomplete across features: Enter ED recall is0/56 in validation and0/296 in train, despite being a common category; numeric44-D MSE0.433327 remains worse than its train-mean baseline0.430093. The cause of the category failure is not established by this experiment. No methods or hyperparameters were changed in response.

One completed run: `m15-real-20261020-01`; source commit `b7a848fcfdc6c7a3a12ebb6749e54ce5e4cf3e55`; source archive SHA256 `e349f5d1a7293791f1f3898f65fd6060d8233557c2c10f984e723a38ea0cfeec`. Resolved configuration SHA256 `8dc47ea43e77043e1c922e128ca897452c9f327ce2ea211fa34d81fb7cf5f7b3`. Full private evidence: `artifacts/reduced_pilot/m15/m15-real-20261020-01/`.

The fixed cohort supplies526 training windows/2335 owned targets and102 validation windows/436 targets; the test partition is quarantined. The unchanged4,760,468-parameter candidate starts from seed42, with train-only fixed class weights, M1 losses/Muon-AdamW operations, source order, clocks and dataset recipe. Retained next-gap features can reveal target arrival time; category scores must retain the time-conditioned label.

Executed command, in the clean worktree:

```bash
timeout --signal=TERM --kill-after=10s 3590s srun --overlap --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=threads --mem=16G --time=01:00:00 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m15_run.py m15-real-20261020-01
```

Completed 5260 retained optimizer updates/20 epochs; 5261 physical optimizer calls include one duplicate next-update recovery reference. Total elapsed through fresh-process replay: 1066.02s; CPU time including replay child: 2065.22s. Peak parent/child RSS: 1189.5/1189.5MiB. Allocation: SLURM job56795452, step3, two CPUs with affinity[9, 13]; memory cgroup16GiB, process address-space cap16GiB, no GPU. Run-parent output at completion: 737.6MiB; all resource caps remained satisfied.

Terminal status is COMPLETE/PASS in `completion.json`; `execution.json` is an immutable earlier phase receipt written before selected-checkpoint replay. The subsequent completion and fresh-replay receipts establish that replay finished successfully. Resource accounting includes the replay child; peak child RSS is the operating system's high-water accounting, not an independently sampled simultaneous memory total. The16GiB job cgroup bounds aggregate process memory. Report plots add less than1MiB; total new private output remains below2GiB.

Selected checkpoint: `epoch20.pt`, update5260, validation prior NLL0.611644; checkpoint SHA256 `bd359a16e4035a7dbc1abc8e76ba82c3542494fcf818faad442d47d1cd944e8c`. Selection used all21 prespecified evaluations, with initialization eligible and earliest-epoch ties. No mortality scores or test targets were evaluated.

| Comparator | Train NLL | Validation NLL | Train accuracy | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|---:|---:|
|Initialization|3.416783|3.419079|0.000857|0.000000|0.000000|
|Selected|0.570818|0.611644|0.697216|0.681193|0.519171|
|Final|0.570818|0.611644|0.697216|0.681193|0.519171|
|Train-only unigram|1.204984|1.219380|0.525054|0.511468|0.169196|
|Train-only transition|0.755282|0.783077|0.649251|0.635321|0.373333|

Eight fixed prior latent draws are averaged as probabilities across all31 logits. NLL/accuracy are event means; macro-F1 uses the four training-supported categories. Unigram smoothing is1 per class; transition smoothing is one unigram-distributed pseudocount per previous-category row. All baseline fitting is train-only.

| Epoch | Update | Train NLL | Validation NLL | Validation posterior reconstruction | Validation prior reconstruction | Validation KL |
|---:|---:|---:|---:|---:|---:|---:|
|0|0|3.416783|3.419079|6.552682|6.417694|624.087959|
|1|263|3.303802|3.305338|5.447269|5.515056|8.865163|
|2|526|1.686835|1.683170|2.376722|2.444957|1.098875|
|3|789|1.322441|1.329564|1.870901|1.975066|0.480144|
|4|1052|1.021367|1.043882|1.615184|1.746502|0.265569|
|5|1315|0.920513|0.947397|1.500938|1.643605|0.160500|
|6|1578|0.888506|0.914497|1.476916|1.604072|0.140741|
|7|1841|0.854730|0.880857|1.475561|1.578938|0.087983|
|8|2104|0.834767|0.859645|1.435434|1.526401|0.079983|
|9|2367|0.786897|0.810634|1.401341|1.486326|0.056450|
|10|2630|0.749262|0.769421|1.362232|1.447748|0.031536|
|11|2893|0.698209|0.711846|1.331885|1.403498|0.032125|
|12|3156|0.673064|0.682955|1.288760|1.357380|0.023655|
|13|3419|0.652647|0.664393|1.260984|1.325676|0.026981|
|14|3682|0.632116|0.639733|1.211415|1.287449|0.026457|
|15|3945|0.604740|0.620722|1.210750|1.290352|0.029716|
|16|4208|0.605593|0.623613|1.200884|1.279142|0.023332|
|17|4471|0.593660|0.616684|1.206530|1.265792|0.022338|
|18|4734|0.586558|0.616884|1.189925|1.253498|0.021823|
|19|4997|0.577294|0.612609|1.207092|1.278985|0.019084|
|20|5260|0.570818|0.611644|1.195711|1.266615|0.024321|

[All train/validation curves and component values](learning_curves.csv); [machine-readable summary](summary.json). Objective components use unchanged mean-window reductions; fixed-beta1 and actual-beta totals are both included. Posterior reconstruction sees the target and is not predictive evidence. Standalone plots remain in ignored output: [learning curves](../../../artifacts/reduced_pilot/m15/learning-report/learning_curves.png) and [all component curves](../../../artifacts/reduced_pilot/m15/learning-report/component_curves.png).

| Selected validation category | Support | Recall | F1 |
|---|---:|---:|---:|
|Enter Hospitalization|102|0.558824|0.506667|
|Leave ED|55|0.709091|0.742857|
|Enter ED|56|0.000000|0.000000|
|Vitals|223|0.901345|0.827160|

All four observed categories have at least100 owned training targets. Rare-category performance (1–99 train targets) and unseen-category performance are not estimable.

| Validation diagnostic | Initialization | Selected | Train-fitted constant |
|---|---:|---:|---:|
|time_gate_log_loss|0.656212|0.048606|0.065027|
|time_gate_brier|0.231545|0.011335|0.011836|
|positive_log1p_hours_rmse|0.771655|0.507884|0.525908|
|observed_gap_hours_mae|1.210232|0.977045|0.996303|
|positive_gap_hours_mae|1.223304|0.973328|0.997142|
|numeric44_mse|0.524093|0.433327|0.430093|

Time masks retain1784 positive/25 real-zero train gaps and330 positive/4 real-zero validation gaps, in hours. 526/102 unknown terminal gaps are excluded. The continuous error uses positive log1p-hours; hour MAE applies the declared nonnegative inverse and0.5 gate. The additional train positive-gap median reference has validation positive-hour MAE0.991864. Numeric targets are the44-D Fourier representation; this is not raw-bpm error.

Preclip gradient norm min/median/max: 1.457313/4.287398/58.640732; 289/5260 updates were clipped at10. Nonfinite loss/gradient/model/optimizer checks passed. Prespecified three-consecutive-epoch overfitting flag epochs: none. Full traces remain private; the train/validation gap and final-versus-selected scores should also inform interpretation.

The final train/validation NLL gap is0.040827; there are small late validation reversals, but no sustained divergence under the predeclared flag. Clipping occurred on5.49% of updates, so training was finite with occasional gradient spikes rather than uniformly small gradients. Validation unannealed KL fell from624.087959 to0.024321: posterior and prior become close under this objective. This alone does not establish useful latent information or prove posterior collapse. The fixed-beta1 curve and independent prior predictive scores prevent the KL/warmup decrease from being presented as sufficient evidence of category learning.

Verification:79 isolated tests passed; the real one-update fixture matched M1 bitwise. Epoch1 recovery and evaluation insertion matched model/optimizer/clocks/RNG bitwise. Initialization/epoch1/final repeated predictions, selected reload in-process and fresh-process predictions, and serialized metric replay passed. All105 run artifact hashes were checked by the reporting tool. M1 source/artifact integrity was rechecked during selected fresh-process replay. No M1 or historical artifact was overwritten. No alpha sweep, second training run, GPU, mortality optimization or test evaluation was performed.

This single-seed experiment measures learning only in the fixed narrow event stream. Unknown availability, target arrival-time conditioning, restricted modalities and lack of rare support limit scientific interpretation. Baseline comparisons must be reported whether favorable or negative; no post hoc protocol adjustment is permitted.

No scientific-validity blocker was found for the stated bounded engineering experiment. Its common-category and numeric failures are negative findings, not permission for another run. Unconditional forecasting, strictly prospective prediction, clinical efficacy, latent-mechanism utility and complete paper reproduction remain unsupported claims. No additional training is authorized by this completed run.
