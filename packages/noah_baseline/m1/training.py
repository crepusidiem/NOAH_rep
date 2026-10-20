"""One CPU-bounded candidate training path, with exact state/cursor/RNG recovery."""
from __future__ import annotations

import copy
from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import torch

from noah_baseline.evaluation.contracts import ContractError
from noah_baseline.models.noah import NOAH
from noah_baseline.models.optimization import NOAHOptimizer

REVISION='clean-m1-candidate-v1'


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def class_weights(windows):
    counts=[torch.zeros(31),torch.zeros(6),torch.zeros(2)]
    for window in windows:
        b=window.batch
        mask=b['loss_mask'][:,1:] & b['attention_mask'][:,1:] & b['attention_mask'][:,:-1]
        for count,key in zip(counts,('category_ids','modality_ids','has_specifics')):
            count.add_(torch.bincount(b[key][:,1:][mask].long(),minlength=len(count)))
    weights=[(count+1).pow(-.5) for count in counts]
    return [weight/weight.mean() for weight in weights]


class Trainer:
    def __init__(self, config, windows, identity):
        self.config=copy.deepcopy(config);self.windows=windows;self.identity=copy.deepcopy(identity)
        if not windows or any(w.subject_id not in identity['train_subjects'] for w in windows):
            raise ContractError('Training requires explicit train-only windows')
        seed_all(config['seed'])
        self.model=NOAH(**config['model'])
        self.weights=class_weights(windows)
        self.model.category_class_weights,self.model.modality_class_weights,self.model.specifics_presence_class_weights=self.weights
        self.optimizer=NOAHOptimizer(self.model,config['learning_rate'],config['weight_decay'])
        self.step=0;self.cursor=0

    def clock(self):
        # LR counts optimizer updates; KL counts consumed windows / corpus length.
        updates_per_epoch=math.ceil(len(self.windows)/self.config['accumulation'])
        warm_updates=self.config['lr_warmup_epochs']*updates_per_epoch
        lr=self.config['learning_rate']*min((self.step+1)/max(warm_updates,1),1.)
        beta=min((self.cursor/len(self.windows))/self.config['kl_warmup_epochs'],1.)
        return lr,beta

    def update(self):
        if self.step>=16: raise ContractError('M1 bounded path forbids more than16 optimizer updates')
        lr,beta=self.clock()
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad();self.model.train()
        epoch_remaining=len(self.windows)-(self.cursor%len(self.windows))
        n=min(self.config['accumulation'],epoch_remaining)
        losses=[];ids=[]
        for offset in range(n):
            window=self.windows[(self.cursor+offset)%len(self.windows)]
            loss,_,_=self.model(**window.batch,beta=beta)
            if not torch.isfinite(loss):raise ContractError('Nonfinite candidate training objective')
            (loss/n).backward();losses.append(float(loss.detach()));ids.extend(window.owned_target_ids)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['gradient_clip'])
        if not torch.isfinite(norm):raise ContractError('Nonfinite gradient norm')
        self.optimizer.step();self.cursor+=n;self.step+=1
        return dict(step=self.step,cursor=self.cursor,lr=lr,beta=beta,loss=float(np.mean(losses)),
                    gradient_norm=float(norm),microbatches=n,target_ids=ids)

    def save(self,path:Path):
        if path.exists():raise ContractError('Checkpoint overwrite forbidden')
        torch.save(dict(revision=REVISION,config=self.config,identity=self.identity,step=self.step,cursor=self.cursor,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),weights=self.weights,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),path)

    def load(self,path:Path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['revision']!=REVISION or state['config']!=self.config or state['identity']!=self.identity:
            raise ContractError('Checkpoint revision/configuration/data identity mismatch')
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        # PyTorch replaces optimizer param-group dictionaries on load. Refresh the wrapper's
        # aliases so subsequent LR updates reach the actual restored optimizer groups.
        self.optimizer.param_groups=[group for optimizer in self.optimizer.optimizers for group in optimizer.param_groups]
        if any(not torch.equal(a,b) for a,b in zip(self.weights,state['weights'])):
            raise ContractError('Train-only weight identities changed')
        self.step=state['step'];self.cursor=state['cursor']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])


def check_recovery(config,windows,identity,directory):
    """The saved model becomes the actual bounded run; the duplicate branch is only a reference."""
    trainer=Trainer(config,windows,identity)
    first=trainer.update();trainer.save(directory/'step1.pt')
    reference=trainer.update()
    expected={key:value.detach().clone() for key,value in trainer.model.state_dict().items()}
    restored=Trainer(config,windows,identity);restored.load(directory/'step1.pt')
    actual=restored.update()
    if actual!=reference:raise ContractError('Resume operation trace diverged')
    for key,value in restored.model.state_dict().items():
        if not torch.equal(value,expected[key]):raise ContractError('Resume changed model tensor:'+key)
    # Optimizer internal buffers must match too, not merely the visible model state.
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not equal(trainer.optimizer.state_dict(),restored.optimizer.state_dict()):
        raise ContractError('Resume optimizer buffers diverged')
    return restored,[first,actual],dict(status='PASS',checkpoint='step1.pt',tensor_equality='bitwise',optimizer_equality=True,trace_equality=True)
