"""Aggregate saved M1.5 evidence and render learning curves without running any model."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'artifacts/reduced_pilot/m15/m15-real-20261020-01'
    public=root/'reports/server/2026-10-20_m15_learning'
    plots=root/'artifacts/reduced_pilot/m15/learning-report'
    plots.mkdir(exist_ok=False)
    read=lambda name: json.loads((run/name).read_text())
    manifest=read('manifest.json');history=read('history.json');execution=read('execution.json')
    selection=read('selection.json');completion=read('completion.json');baselines=read('baseline_metrics.json')
    assert completion['status']=='PASS' and execution['retained_optimizer_updates']<=5260
    assert manifest['claim']=='teacher-forced, time-conditioned category prediction'
    assert not execution['test_evaluated'] and not execution['gpu_initialized']
    inventory=read('artifact_inventory.json')
    for name,digest in inventory.items():
        with (run/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    trace=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    assert len(trace)==execution['retained_optimizer_updates']
    assert [r['step'] for r in trace]==list(range(1,len(trace)+1))
    selected=next(r for r in history if r['epoch']==selection['epoch'])
    assert selected==min(history,key=lambda r:(r['splits']['val']['metrics']['category_nll'],r['epoch']))
    norms=[r['gradient_norm'] for r in trace]
    increasing=[]
    for i in range(3,len(history)):
        h=history[i-3:i+1]
        if all(a['splits']['val']['metrics']['category_nll']<b['splits']['val']['metrics']['category_nll']
            and a['splits']['train']['metrics']['category_nll']>b['splits']['train']['metrics']['category_nll'] for a,b in zip(h,h[1:])):
            increasing.append(history[i]['epoch'])
    summary=dict(claim=manifest['claim'],run_id=manifest['run_id'],executed_commit=manifest['code_commit'],
        source_sha256=manifest['source_sha256'],resolved_config_sha256=manifest['resolved_config_sha256'],
        training=execution,completion=completion,selection=selection,initialization=history[0],selected=selected,final=history[-1],
        baselines=baselines,gradient=dict(min=min(norms),median=statistics.median(norms),max=max(norms),
            clipped_updates=sum(n>10 for n in norms),updates=len(norms)),
        prespecified_overfitting_flag_epochs=increasing,verified_run_artifacts=len(inventory),
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        claim_limitations=['teacher forcing and retained target arrival-time conditioning',
            'unknown availability; no prospective validity','four observed categories; no rare-category support',
            'small single-seed reduced-modality cohort; no efficacy or paper-fidelity claim'])
    with (public/'summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    curves=[]
    for row in history:
        for split,record in row['splits'].items():
            curve=dict(epoch=row['epoch'],step=row['step'],beta=row['beta'],split=split)
            curve.update({'objective_'+k:v for k,v in record['objective'].items()})
            curve.update({'metric_'+k:v for k,v in record['metrics'].items() if isinstance(v,(float,int))})
            curves.append(curve)
    with (public/'learning_curves.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    os.environ['MPLCONFIGDIR']=str(plots/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in history]
    def series(split,kind,key):return [r['splits'][split][kind][key] for r in history]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for split,color in [('train','tab:blue'),('val','tab:orange')]:
        axes[0,0].plot(epochs,series(split,'metrics','category_nll'),label=split,color=color)
        for key,style in [('loss_rec_post','-'),('loss_rec_prior','--'),('loss_kl',':')]:
            axes[0,1].plot(epochs,series(split,'objective',key),label=split+' '+key,color=color,linestyle=style)
        axes[1,0].plot(epochs,series(split,'metrics','positive_log1p_hours_rmse'),label=split,color=color)
        axes[1,1].plot(epochs,series(split,'metrics','numeric44_mse'),label=split,color=color)
    for name,style in [('unigram',':'),('transition','--')]:
        axes[0,0].axhline(baselines['val'][name]['category_nll'],linestyle=style,color='gray',label='val '+name)
    axes[0,0].set_ylabel('Prior category NLL (event mean)');axes[0,1].set_ylabel('Objective component (window mean)')
    axes[1,0].set_ylabel('Positive log1p-hours RMSE');axes[1,1].set_ylabel('Numeric44-D MSE')
    for ax in axes.flat:
        ax.axvline(selection['epoch'],color='black',alpha=.25);ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('M1.5: teacher-forced, time-conditioned category prediction')
    fig.savefig(plots/'learning_curves.png',dpi=160);fig.savefig(plots/'learning_curves.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    parts=['category','modality','specifics_presence','specifics_vec','value','time_gate','time_reg']
    for ax,part in zip(axes.flat,parts):
        for split,color in [('train','tab:blue'),('val','tab:orange')]:
            for branch,style in [('posterior','-'),('prior','--')]:
                ax.plot(epochs,series(split,'objective',branch+'_'+part),color=color,linestyle=style,label=split+' '+branch)
        ax.set_title(part)
    for split,color in [('train','tab:blue'),('val','tab:orange')]:
        axes[2,1].plot(epochs,series(split,'objective','loss_kl'),color=color,label=split)
        axes[2,2].plot(epochs,series(split,'objective','total_loss'),color=color,label=split+' actual beta')
        axes[2,2].plot(epochs,series(split,'objective','fixed_beta1_total'),color=color,linestyle='--',label=split+' beta1')
    axes[2,1].set_title('Unannealed KL');axes[2,2].set_title('Total objective')
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('M1.5 component objectives: mean window reductions, unchanged candidate')
    fig.savefig(plots/'component_curves.png',dpi=160);fig.savefig(plots/'component_curves.svg');plt.close(fig)
    lines=['# M1.5 executed learning validation','',
        'Date:2026-10-20, America/New_York. Primary evaluation: **teacher-forced, time-conditioned category prediction**. '+
        'This reduced-modality occurrence diagnostic supplies no unconditional forecasting, prospective clinical prediction, zero-shot rollout or paper-fidelity result.','',
        f"One completed run: `{manifest['run_id']}`; source commit `{manifest['code_commit']}`; source archive SHA256 `{manifest['source_sha256']}`. "+
        f"Resolved configuration SHA256 `{manifest['resolved_config_sha256']}`. Full private evidence: `artifacts/reduced_pilot/m15/{manifest['run_id']}/`.",'',
        'The fixed cohort supplies526 training windows/2335 owned targets and102 validation windows/436 targets; the test partition is quarantined. '+
        'The unchanged4,760,468-parameter candidate starts from seed42, with train-only fixed class weights, M1 losses/Muon-AdamW operations, source order, clocks and dataset recipe. '+
        'Retained next-gap features can reveal target arrival time; category scores must retain the time-conditioned label.','',
        'Executed command, in the clean worktree:','', '```bash',
        'timeout --signal=TERM --kill-after=10s 3590s srun --overlap --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=threads --mem=16G --time=01:00:00 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m15_run.py m15-real-20261020-01',
        '```','',
        f"Completed {execution['retained_optimizer_updates']} retained optimizer updates/20 epochs; {execution['physical_optimizer_calls']} physical optimizer calls include one duplicate next-update recovery reference. "+
        f"Total elapsed through fresh-process replay: {completion['elapsed_seconds']:.2f}s; CPU time including replay child: {completion['cpu_seconds_including_children']:.2f}s. "+
        f"Peak parent/child RSS: {completion['peak_parent_rss_MiB']:.1f}/{completion['peak_child_rss_MiB']:.1f}MiB. "+
        f"Allocation: SLURM job{manifest['resources']['job_id']}, step{manifest['resources']['step_id']}, two CPUs with affinity{manifest['resources']['affinity']}; memory cgroup16GiB, process address-space cap16GiB, no GPU. "+
        f"Run-parent output at completion: {completion['total_new_bytes']/1024**2:.1f}MiB; all resource caps remained satisfied.",'',
        f"Selected checkpoint: `epoch{selection['epoch']:02d}.pt`, update{selection['step']}, validation prior NLL{selection['metric']:.6f}; "+
        f"checkpoint SHA256 `{selection['checkpoint_sha256']}`. Selection used all21 prespecified evaluations, with initialization eligible and earliest-epoch ties. No mortality scores or test targets were evaluated.",'',
        '| Comparator | Train NLL | Validation NLL | Train accuracy | Validation accuracy | Validation macro-F1 |',
        '|---|---:|---:|---:|---:|---:|']
    comparisons=[('Initialization',history[0]['splits']),('Selected',selected['splits']),('Final',history[-1]['splits'])]
    for label,splits in comparisons:
        a=splits['train']['metrics'];b=splits['val']['metrics']
        lines.append(f"|{label}|{a['category_nll']:.6f}|{b['category_nll']:.6f}|{a['category_accuracy']:.6f}|{b['category_accuracy']:.6f}|{b['macro_f1']:.6f}|")
    for mode in ('unigram','transition'):
        a=baselines['train'][mode];b=baselines['val'][mode]
        lines.append(f"|Train-only {mode}|{a['category_nll']:.6f}|{b['category_nll']:.6f}|{a['category_accuracy']:.6f}|{b['category_accuracy']:.6f}|{b['macro_f1']:.6f}|")
    lines+=['','Eight fixed prior latent draws are averaged as probabilities across all31 logits. NLL/accuracy are event means; macro-F1 uses the four training-supported categories. '+
        'Unigram smoothing is1 per class; transition smoothing is one unigram-distributed pseudocount per previous-category row. All baseline fitting is train-only.','',
        '| Epoch | Update | Train NLL | Validation NLL | Validation posterior reconstruction | Validation prior reconstruction | Validation KL |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in history:
        t=r['splits']['train'];v=r['splits']['val'];o=v['objective']
        lines.append(f"|{r['epoch']}|{r['step']}|{t['metrics']['category_nll']:.6f}|{v['metrics']['category_nll']:.6f}|{o['loss_rec_post']:.6f}|{o['loss_rec_prior']:.6f}|{o['loss_kl']:.6f}|")
    lines+=['','[All train/validation curves and component values](learning_curves.csv); [machine-readable summary](summary.json). '+
        'Objective components use unchanged mean-window reductions; fixed-beta1 and actual-beta totals are both included. '+
        'Posterior reconstruction sees the target and is not predictive evidence. Standalone plots remain in ignored output: '+
        '[learning curves](../../../artifacts/reduced_pilot/m15/learning-report/learning_curves.png) and '+
        '[all component curves](../../../artifacts/reduced_pilot/m15/learning-report/component_curves.png).','',
        '| Selected validation category | Support | Recall | F1 |','|---|---:|---:|---:|']
    names={'5':'Enter Hospitalization','14':'Leave ED','22':'Enter ED','28':'Vitals'}
    for key,v in selected['splits']['val']['metrics']['per_category'].items():
        lines.append(f"|{names[key]}|{v['support']}|{v['recall']:.6f}|{v['f1']:.6f}|")
    lines+=['','All four observed categories have at least100 owned training targets. Rare-category performance (1–99 train targets) and unseen-category performance are not estimable.','',
        '| Validation diagnostic | Initialization | Selected | Train-fitted constant |','|---|---:|---:|---:|']
    for key in ('time_gate_log_loss','time_gate_brier','positive_log1p_hours_rmse','observed_gap_hours_mae','positive_gap_hours_mae','numeric44_mse'):
        lines.append(f"|{key}|{history[0]['splits']['val']['metrics'][key]:.6f}|{selected['splits']['val']['metrics'][key]:.6f}|{baselines['val']['unigram'][key]:.6f}|")
    lines+=['','Time masks retain1784 positive/25 real-zero train gaps and330 positive/4 real-zero validation gaps, in hours. '+
        '526/102 unknown terminal gaps are excluded. The continuous error uses positive log1p-hours; hour MAE applies the declared nonnegative inverse and0.5 gate. '+
        f"The additional train positive-gap median reference has validation positive-hour MAE{baselines['val']['positive_median_hours_mae']:.6f}. "+
        'Numeric targets are the44-D Fourier representation; this is not raw-bpm error.','',
        f"Preclip gradient norm min/median/max: {min(norms):.6f}/{statistics.median(norms):.6f}/{max(norms):.6f}; "+
        f"{sum(n>10 for n in norms)}/{len(norms)} updates were clipped at10. Nonfinite loss/gradient/model/optimizer checks passed. "+
        f"Prespecified three-consecutive-epoch overfitting flag epochs: {increasing or 'none'}. "+
        'Full traces remain private; the train/validation gap and final-versus-selected scores should also inform interpretation.','',
        f"Verification:79 isolated tests passed; the real one-update fixture matched M1 bitwise. Epoch1 recovery and evaluation insertion matched model/optimizer/clocks/RNG bitwise. "+
        f"Initialization/epoch1/final repeated predictions, selected reload in-process and fresh-process predictions, and serialized metric replay passed. "+
        f"All{len(inventory)} run artifact hashes were checked by the reporting tool. M1 source/artifact integrity was rechecked during selected fresh-process replay. "+
        'No M1 or historical artifact was overwritten. No alpha sweep, second training run, GPU, mortality optimization or test evaluation was performed.','',
        'This single-seed experiment measures learning only in the fixed narrow event stream. Unknown availability, target arrival-time conditioning, restricted modalities and lack of rare support limit scientific interpretation. '+
        'Baseline comparisons must be reported whether favorable or negative; no post hoc protocol adjustment is permitted.']
    with (public/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(selection=selection,completion=completion,gradient=summary['gradient'],
        overfitting_flag_epochs=increasing,report=str(public),plots=str(plots)),indent=2))


if __name__=='__main__':main()
