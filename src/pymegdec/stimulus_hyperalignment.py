"""Cross-subject stimulus decoding with Procrustes hyperalignment."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from math import comb
from collections.abc import Sequence
import numpy as np
from reptrace.decoding.hyperalignment import CLASS_ALIGNMENT_SAMPLE_MODES, fit_class_hyperalignment
from reptrace.decoding.windowed import fit_window_model, predict_window_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import get_default_classifier_param, should_use_default_classifier_param, train_multiclass_classifier
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec.stimulus_cross_subject import DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, DEFAULT_CROSS_SUBJECT_CLASSIFIER, DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA, DEFAULT_CROSS_SUBJECT_NORMALIZATION, DEFAULT_CROSS_SUBJECT_PARTICIPANTS, DEFAULT_CROSS_SUBJECT_WINDOW_CENTER, DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, FEATURE_MODES, NORMALIZATION_MODES, CrossSubjectStimulusConfig, load_participant_stimulus_features

TARGET_CENTERING_MODES=("group_mean","target_unsupervised")

@dataclass(frozen=True)
class CrossSubjectHyperalignmentConfig:
    window_center: float=DEFAULT_CROSS_SUBJECT_WINDOW_CENTER; window_size: float=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE; baseline_window: tuple[float,float]=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str="sensor_flat"; normalization: str=DEFAULT_CROSS_SUBJECT_NORMALIZATION; classifier: str=DEFAULT_CROSS_SUBJECT_CLASSIFIER; classifier_param: object=float("nan")
    components_pca: int|float=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA; max_trials_per_class_per_participant: int|None=None; chance_classes: int=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES; random_state: int|None=0
    signflip_permutations: int=10000; signflip_seed: int|None=0; hyper_components: int|float=64; hyper_iterations: int=10; hyper_tolerance: float=1e-8; hyper_sample_mode: str="class_repetition"; hyper_repetitions_per_class: int|None=None; target_centering: str="target_unsupervised"

def evaluate_cross_subject_hyperalignment(data_folder, participants, *, config=None, outer_participants=None, progress=None, label_shuffle_control=False, label_shuffle_seed=0):
    cfg=_checked(config or CrossSubjectHyperalignmentConfig()); data_folder=resolve_data_folder(data_folder); participants=tuple(map(int,participants)); outer=tuple(participants if outer_participants is None else map(int,outer_participants))
    fcfg=CrossSubjectStimulusConfig(window_center=cfg.window_center,window_size=cfg.window_size,baseline_window=cfg.baseline_window,feature_mode=cfg.feature_mode,normalization=cfg.normalization,classifier=cfg.classifier,classifier_param=cfg.classifier_param,components_pca=cfg.components_pca,max_trials_per_class_per_participant=cfg.max_trials_per_class_per_participant,chance_classes=cfg.chance_classes,random_state=cfg.random_state,signflip_permutations=cfg.signflip_permutations,signflip_seed=cfg.signflip_seed)
    sets=[]
    for p in participants:
        if progress: progress(f"LOAD participant={p}")
        sets.append(load_participant_stimulus_features(data_folder,p,config=fcfg))
    param=cfg.classifier_param
    if should_use_default_classifier_param(param): param=get_default_classifier_param(cfg.classifier)
    rows=[]
    for test in outer:
        if progress: progress(f"START outer_test_participant={test}")
        train=[s for s in sets if s.participant!=test]; target=next(s for s in sets if s.participant==test)
        rows.append(_fold(train,target,cfg,param,label_shuffle_seed if label_shuffle_control else None))
        rows[-1].update(label_shuffle_control=bool(label_shuffle_control),label_shuffle_seed=int(label_shuffle_seed) if label_shuffle_control else "")
        if progress: progress(f"DONE outer_test_participant={test} balanced_accuracy={rows[-1]['balanced_accuracy']:.4f}")
    return {"outer":rows,"group_summary":summarize_cross_subject_hyperalignment(rows,cfg)}

def export_cross_subject_hyperalignment(data_folder, participants, *, outer_output_path, group_summary_output_path=None, config=None, outer_participants=None, progress=None, label_shuffle_control=False, label_shuffle_seed=0):
    a=evaluate_cross_subject_hyperalignment(data_folder,participants,config=config,outer_participants=outer_participants,progress=progress,label_shuffle_control=label_shuffle_control,label_shuffle_seed=label_shuffle_seed)
    write_alpha_metrics_csv(a["outer"],outer_output_path)
    if group_summary_output_path: write_alpha_metrics_csv(a["group_summary"],group_summary_output_path)
    return a

def _fold(train_sets,test_set,cfg,param,shuffle_seed):
    labels={s.participant:_labels(s.labels,shuffle_seed,test_set.participant,s.participant) for s in train_sets}; feats={s.participant:s.features for s in train_sets}
    model,align=fit_class_hyperalignment(feats,labels,sample_mode=cfg.hyper_sample_mode,n_repetitions_per_class=cfg.hyper_repetitions_per_class,n_components=cfg.hyper_components,n_iterations=cfg.hyper_iterations,template_tolerance=cfg.hyper_tolerance)
    train_x=np.vstack([model.transform(s.participant,s.features) for s in train_sets]); train_y=np.concatenate([labels[s.participant] for s in train_sets])
    mean=np.mean(test_set.features,axis=0) if cfg.target_centering=="target_unsupervised" else None
    test_x=model.transform_group(test_set.features,feature_mean=mean)
    bundle=fit_window_model(train_x,train_y,fit_model=lambda x,y: train_multiclass_classifier(x,y,cfg.classifier,param,random_state=cfg.random_state),components_pca=cfg.components_pca,train_window=(cfg.window_center-cfg.window_size/2,cfg.window_center+cfg.window_size/2))
    pred,_=predict_window_model(bundle,test_x); acc=float(accuracy_score(test_set.labels,pred)); bal=float(balanced_accuracy_score(test_set.labels,pred))
    return {**_meta(cfg),"test_participant":test_set.participant,"n_train_participants":len(train_sets),"n_train_trials":int(train_x.shape[0]),"n_test_trials":int(test_set.labels.shape[0]),"chance_accuracy":1/cfg.chance_classes,"accuracy":acc,"percent":100*acc,"balanced_accuracy":bal,"balanced_percent":100*bal,"hyper_actual_components":model.n_components,"hyper_alignment_rows":int(next(iter(align.aligned_by_subject.values())).shape[0]),"hyper_repetitions_per_class":align.n_repetitions_per_class,"classifier_param":param,"actual_components_pca":bundle.actual_components_pca,"pca_explained_variance_percent":bundle.explained_variance_percent}

def summarize_cross_subject_hyperalignment(rows,cfg=None):
    if not rows: return []
    bal=np.asarray([float(r["balanced_accuracy"]) for r in rows]); raw=np.asarray([float(r["accuracy"]) for r in rows]); chance=float(rows[0]["chance_accuracy"]); d=bal-chance; cfg=cfg or CrossSubjectHyperalignmentConfig()
    return [{**_meta(cfg),"n_outer_folds":len(rows),"n_test_participants":len(rows),"chance_accuracy":chance,"chance_percent":100*chance,"accuracy_mean":float(raw.mean()),"accuracy_median":float(np.median(raw)),"accuracy_sem":_sem(raw),"percent_mean":float(100*raw.mean()),"balanced_accuracy_mean":float(bal.mean()),"balanced_accuracy_median":float(np.median(bal)),"balanced_accuracy_sem":_sem(bal),"balanced_percent_mean":float(100*bal.mean()),"mean_above_chance":float(d.mean()),"percent_above_chance":float(100*d.mean()),"participants_above_chance":int(np.sum(d>0)),"participants_total":len(d),"participants_at_or_below_chance":int(np.sum(d<=0)),"one_sided_exact_sign_p_value":_exact(d),"one_sided_signflip_p_value":_signflip(d,cfg.signflip_permutations,cfg.signflip_seed),"label_shuffle_control":rows[0].get("label_shuffle_control",False),"label_shuffle_seed":rows[0].get("label_shuffle_seed","")}]

def _meta(c): return {"window_center_s":c.window_center,"window_size_s":c.window_size,"window_start_s":c.window_center-c.window_size/2,"window_stop_s":c.window_center+c.window_size/2,"baseline_window_start_s":c.baseline_window[0],"baseline_window_stop_s":c.baseline_window[1],"feature_mode":c.feature_mode,"normalization":c.normalization,"alignment":"procrustes_hyperalignment","hyper_sample_mode":c.hyper_sample_mode,"hyper_requested_components":c.hyper_components,"hyper_iterations":c.hyper_iterations,"hyper_tolerance":c.hyper_tolerance,"target_centering":c.target_centering,"classifier":c.classifier,"components_pca":c.components_pca,"max_trials_per_class_per_participant":c.max_trials_per_class_per_participant}
def _labels(labels,seed,test,train):
    y=np.asarray(labels).copy()
    if seed is not None: np.random.default_rng(abs(hash((seed,test,train)))%(2**32)).shuffle(y)
    return y
def _checked(c):
    if c.feature_mode not in FEATURE_MODES or c.normalization not in NORMALIZATION_MODES or c.hyper_sample_mode not in CLASS_ALIGNMENT_SAMPLE_MODES or c.target_centering not in TARGET_CENTERING_MODES: raise ValueError("unsupported hyperalignment configuration")
    return c
def _sem(v):
    v=np.asarray(v,dtype=float); return 0.0 if v.size<2 else float(v.std(ddof=1)/np.sqrt(v.size))
def _exact(d):
    d=np.asarray(d); d=d[d!=0]; n=d.size; p=int(np.sum(d>0)); return 1.0 if n==0 else float(sum(comb(n,k) for k in range(p,n+1))/(2**n))
def _signflip(d,n,seed):
    if n<=0: return np.nan
    d=np.asarray(d,dtype=float); rng=np.random.default_rng(seed); null=np.mean(rng.choice([-1.0,1.0],size=(int(n),d.size))*d[None,:],axis=1); return float((np.sum(null>=d.mean())+1)/(int(n)+1))
def _win(v):
    a,b=v.split(",",1); return float(a),float(b)
def _optint(v): return None if str(v).lower() in {"none","auto","null"} else parse_int_or_inf(v)

def _parser(prog=None):
    p=argparse.ArgumentParser(prog=prog,description="Run LOSO stimulus decoding with Procrustes hyperalignment."); p.add_argument("--data-dir",dest="data_folder",default=None); p.add_argument("--participants",default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS); p.add_argument("--outer-participants",default=None); p.add_argument("--window-center",type=float,default=DEFAULT_CROSS_SUBJECT_WINDOW_CENTER); p.add_argument("--window-size",type=float,default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE); p.add_argument("--baseline-window",type=_win,default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW); p.add_argument("--feature-mode",choices=FEATURE_MODES,default="sensor_flat"); p.add_argument("--normalization",choices=NORMALIZATION_MODES,default=DEFAULT_CROSS_SUBJECT_NORMALIZATION); p.add_argument("--classifier",default=DEFAULT_CROSS_SUBJECT_CLASSIFIER); p.add_argument("--classifier-param",default=None); p.add_argument("--components-pca",type=parse_int_or_inf,default=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA); p.add_argument("--hyper-components",type=parse_int_or_inf,default=64); p.add_argument("--hyper-iterations",type=int,default=10); p.add_argument("--hyper-tolerance",type=float,default=1e-8); p.add_argument("--hyper-sample-mode",choices=CLASS_ALIGNMENT_SAMPLE_MODES,default="class_repetition"); p.add_argument("--hyper-repetitions-per-class",type=_optint,default=None); p.add_argument("--target-centering",choices=TARGET_CENTERING_MODES,default="target_unsupervised"); p.add_argument("--max-trials-per-class-per-participant",type=int,default=None); p.add_argument("--chance-classes",type=int,default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES); p.add_argument("--random-state",type=int,default=0); p.add_argument("--label-shuffle-control",action="store_true"); p.add_argument("--label-shuffle-seed",type=int,default=0); p.add_argument("--signflip-permutations",type=int,default=10000); p.add_argument("--signflip-seed",type=int,default=0); p.add_argument("--outer-output",default="outputs/stimulus_cross_subject_hyperalignment_outer.csv"); p.add_argument("--summary-output",default="outputs/stimulus_cross_subject_hyperalignment_group_summary.csv"); return p

def stimulus_cross_subject_hyperalignment(argv: Sequence[str]|None=None, prog: str|None=None) -> int:
    a=_parser(prog).parse_args(normalize_argv(argv)); participants=parse_participant_spec(a.participants); outer=parse_participant_spec(a.outer_participants) if a.outer_participants else None
    cfg=CrossSubjectHyperalignmentConfig(window_center=a.window_center,window_size=a.window_size,baseline_window=a.baseline_window,feature_mode=a.feature_mode,normalization=a.normalization,classifier=a.classifier,classifier_param=parse_classifier_param(a.classifier_param),components_pca=a.components_pca,max_trials_per_class_per_participant=a.max_trials_per_class_per_participant,chance_classes=a.chance_classes,random_state=a.random_state,signflip_permutations=a.signflip_permutations,signflip_seed=a.signflip_seed,hyper_components=a.hyper_components,hyper_iterations=a.hyper_iterations,hyper_tolerance=a.hyper_tolerance,hyper_sample_mode=a.hyper_sample_mode,hyper_repetitions_per_class=a.hyper_repetitions_per_class,target_centering=a.target_centering)
    res=export_cross_subject_hyperalignment(a.data_folder,participants,outer_output_path=a.outer_output,group_summary_output_path=a.summary_output,config=cfg,outer_participants=outer,progress=lambda m: print(m,flush=True),label_shuffle_control=a.label_shuffle_control,label_shuffle_seed=a.label_shuffle_seed)
    print(f"Wrote {len(res['outer'])} held-out participant rows to {a.outer_output}"); return 0

def main(argv: Sequence[str]|None=None) -> int: return stimulus_cross_subject_hyperalignment(argv,prog="pymegdec stimulus-cross-subject-hyperalignment")
if __name__=="__main__": raise SystemExit(main())
