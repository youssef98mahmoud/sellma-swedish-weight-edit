"""Small, explicitly synthetic Swedish/English character-LM weight-edit experiment.
No external model, dataset, or API required. Run: python experiment.py
"""
import json, math, random, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
SEED = 20261119
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)

SV_SUBJ = ['barnet','läraren','fågeln','räven','vännen','sjömannen','kvinnan','pojken','grannen','hunden','flickan','målaren','båten','systern','kocken']
SV_VERB = ['läser','målar','köper','håller','öppnar','skriver','hämtar','ser','vänder','gömmer','väljer','söker']
SV_OBJ = ['en bok','ett äpple','en blå väska','ett grönt löv','en röd båt','en skål','en gåva','en dörr','ett brev','en mössa','en sång','ett hus']
SV_LOC = ['i köket','på ön','vid sjön','i skogen','på gården','under trädet','i staden','på vägen']
EN_SUBJ = ['child','teacher','bird','fox','friend','sailor','woman','boy','neighbor','dog','girl','painter','boat','sister','cook']
EN_VERB = ['reads','paints','buys','holds','opens','writes','fetches','sees','turns','hides','chooses','seeks']
EN_OBJ = ['a book','an apple','a blue bag','a green leaf','a red boat','a bowl','a gift','a door','a letter','a hat','a song','a house']
EN_LOC = ['in the kitchen','on the island','by the lake','in the forest','in the yard','under the tree','in the city','on the road']

def generate(lang, count, seed):
    r = random.Random(seed)
    a,b,c,d = (SV_SUBJ,SV_VERB,SV_OBJ,SV_LOC) if lang=='sv' else (EN_SUBJ,EN_VERB,EN_OBJ,EN_LOC)
    return [f"{r.choice(a)} {r.choice(b)} {r.choice(c)} {r.choice(d)}." for _ in range(count)]

# Independently generated samples; NOTE: templates and vocabulary overlap across splits.
train_sv=generate('sv',800,11); train_en=generate('en',3200,12)
val_sv=generate('sv',240,21); val_en=generate('en',240,22)
test_sv=generate('sv',400,31); test_en=generate('en',400,32)
all_vocab=sorted(set(''.join(train_sv+train_en+val_sv+val_en+test_sv+test_en)))
chars=['^','$']+all_vocab
stoi={c:i for i,c in enumerate(chars)}
accents=[stoi[x] for x in 'åäö']

def encode(texts):
    return [[stoi['^']]+[stoi[c] for c in s]+[stoi['$']] for s in texts]

def batches(texts, batchsize=64, shuffle=False, seed=0):
    ids=list(range(len(texts)))
    if shuffle: random.Random(seed).shuffle(ids)
    for start in range(0,len(ids),batchsize):
        seqs=[texts[i] for i in ids[start:start+batchsize]]
        T=max(len(s) for s in seqs)-1
        x=torch.zeros((len(seqs),T),dtype=torch.long); y=torch.zeros_like(x)
        mask=torch.zeros_like(x,dtype=torch.bool)
        for j,s in enumerate(seqs):
            x[j,:len(s)-1]=torch.tensor(s[:-1]); y[j,:len(s)-1]=torch.tensor(s[1:]); mask[j,:len(s)-1]=True
        yield x,y,mask

class CharLM(nn.Module):
    def __init__(self,vocab):
        super().__init__(); self.embedding=nn.Embedding(vocab,24); self.gru=nn.GRU(24,48,batch_first=True); self.output=nn.Linear(48,vocab)
    def forward(self,x): return self.output(self.gru(self.embedding(x))[0])

model=CharLM(len(chars)); opt=torch.optim.AdamW(model.parameters(),lr=0.004)
encoded_train=encode(train_sv+train_en)
for epoch in range(5):
    model.train(); losses=[]
    for x,y,mask in batches(encoded_train,64,True,SEED+epoch):
        opt.zero_grad(); logits=model(x)
        loss=F.cross_entropy(logits[mask],y[mask]); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
    print(f'epoch {epoch+1}: train CE {np.mean(losses):.4f}',flush=True)
model.eval()

@torch.no_grad()
def precompute(texts):
    logits=[]; targets=[]
    for x,y,mask in batches(encode(texts),64):
        pred=model(x)
        logits.append(pred[mask]); targets.append(y[mask])
    return torch.cat(logits),torch.cat(targets)

cache={k:precompute(v) for k,v in dict(val_sv=val_sv,val_en=val_en,test_sv=test_sv,test_en=test_en).items()}

def eval_cache(key, edit=None):
    l,y=cache[key]; l=l if edit is None else l+edit
    losses=F.cross_entropy(l,y,reduction='none')
    acc_mask=torch.isin(y,torch.tensor(accents))
    # correct classification is stringent for accents in a character model
    return {'nll':round(float(losses.mean()),5),
            'accent_nll':round(float(losses[acc_mask].mean()),5) if bool(acc_mask.any()) else None,
            'accent_accuracy':round(float((l[acc_mask].argmax(-1)==y[acc_mask]).float().mean()),5) if bool(acc_mask.any()) else None,
            'accent_count':int(acc_mask.sum()),'token_count':int(len(y))}

zero=torch.zeros(len(chars))
base_sv=eval_cache('val_sv'); base_en=eval_cache('val_en')
# Predeclare selection rule: minimize heldout-dev Swedish accented-character CE,
# subject to no more than +0.03 nats English CE, and +0.02 Swedish overall CE.
search=[]
for size in np.arange(0,2.51,0.1):
    e=zero.clone(); e[accents]=float(size)
    sv=eval_cache('val_sv',e); en=eval_cache('val_en',e)
    admissible=en['nll']-base_en['nll']<=0.03 and sv['nll']-base_sv['nll']<=0.02
    search.append((round(float(size),2),sv,en,admissible))
chosen=min((x for x in search if x[3]),key=lambda x:(x[1]['accent_nll'],x[0]))
size=chosen[0]; edit=zero.clone(); edit[accents]=size

# Actual weight edit; measured equivalence against inference-only offset.
model_edit=CharLM(len(chars)); model_edit.load_state_dict(model.state_dict())
with torch.no_grad(): model_edit.output.bias[accents]+=size
with torch.no_grad():
    x=torch.tensor([encode(['fågeln läser en bok i köket.'])[0][:-1]])
    max_diff=float((model_edit(x)-(model(x)+edit)).abs().max())

# Random controls: edit 3 distinct nonaccent output biases with same amplitude.
r=random.Random(SEED); eligible=[i for i in range(len(chars)) if i not in accents and chars[i] not in ['^','$']]
random_controls=[]
for j in range(20):
    ix=r.sample(eligible,3); e=zero.clone(); e[ix]=size
    random_controls.append({'characters':[chars[i] for i in ix],
                            'sv':eval_cache('test_sv',e),'en':eval_cache('test_en',e)})
results={
 'experiment':'Targeted output bias edit in small synthetic bilingual char GRU',
 'seed':SEED,'torch':torch.__version__,'python':platform.python_version(),
 'architecture':'24-dimensional char embedding, single-layer GRU hidden=48, linear output bias',
 'train_counts':{'sv':len(train_sv),'en':len(train_en)},
 'validation_counts':{'sv':len(val_sv),'en':len(val_en)},
 'test_counts':{'sv':len(test_sv),'en':len(test_en)},
 'data_disclosure':'Manually authored synthetic vocabulary/templates, random recombinations. Vocabulary/templates overlap between splits; not natural Swedish or generalization to novel lexical items.',
 'edit_description':'Add a constant to the three output bias entries for Swedish å ä ö.',
 'selected_delta':size,
 'selection_rule':'Minimize Swedish accented-token validation NLL; max +0.03 English validation overall NLL; max +0.02 Swedish validation overall NLL. Grid 0..2.5 step 0.1.',
 'validation_baseline':{'sv':base_sv,'en':base_en},
 'validation_selected':{'sv':chosen[1],'en':chosen[2]},
 'test':{
   'baseline':{'sv':eval_cache('test_sv'),'en':eval_cache('test_en')},
   'targeted_weight_edit':{'sv':eval_cache('test_sv',edit),'en':eval_cache('test_en',edit)},
   'inference_logit_bias':{'sv':eval_cache('test_sv',edit),'en':eval_cache('test_en',edit)}},
 'max_forward_logit_difference_edit_vs_logit_bias':max_diff,
 'random_controls':random_controls,
 'notes':'No pretrained model, no natural text, and no semantic quality test. Evaluation uses teacher-forced next-character prediction. An output-bias edit is analytically identical to a constant inference-time logit offset.'}
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

# Illustrative held-out next-character predictions; not independently sampled examples.
examples={'corrected_accent':[],'introduced_errors':[]}
with torch.no_grad():
 for language,sentences in [('sv',test_sv),('en',test_en)]:
  for sent in sentences:
   ids=encode([sent])[0]; x=torch.tensor([ids[:-1]])
   original=model(x)[0]; adjusted=original+edit
   for i,(old,new) in enumerate(zip(original.argmax(-1).tolist(),adjusted.argmax(-1).tolist())):
    expected=ids[i+1]
    item={'language':language,'context':('^'+sent[:i])[-46:],'expected':chars[expected], 'baseline_prediction':chars[old], 'edited_prediction':chars[new]}
    if old!=expected and new==expected and chars[expected] in 'åäö' and len(examples['corrected_accent'])<8:
     examples['corrected_accent'].append(item)
    if old==expected and new!=expected and len(examples['introduced_errors'])<8:
     examples['introduced_errors'].append(item)
   if all(len(v)>=8 for v in examples.values()): break
(OUT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))

torch.save({'state_dict':model.state_dict(),'chars':chars,'config':{'embed':24,'hidden':48,'seed':SEED}},OUT/'checkpoint.pt')
print('CHOSEN DELTA',size,'TEST',json.dumps(results['test'],ensure_ascii=False,indent=2),'MAX_DIFF',max_diff,flush=True)
