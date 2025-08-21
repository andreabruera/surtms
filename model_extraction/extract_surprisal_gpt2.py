import numpy
import os
import scipy
import torch
import transformers

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def check_ws(w_one, w_two):
    check = True
    if 'NA' in w_one:
        check = False
    if 'NA' in w_two:
        check = False
    if w_one == w_two:
        check = False
    return check

tokenizer = AutoTokenizer.from_pretrained(
                                 "benjamin/gerpt2",
                                 #"benjamin/gerpt2-large",
                                 cache_dir='../../hf_models',
                                 )
model = AutoModelForCausalLM.from_pretrained(
                                 "benjamin/gerpt2",
                                 #"benjamin/gerpt2-large",
                                 cache_dir='../../hf_models',
                                 ).to('cuda')
softmax = torch.nn.Softmax(dim=-1)
sents = list()
with open('transcriptions.txt') as i:
    for l in i:
        if 'cw' not in l:
            continue
        line = l.strip().split(';')
        sents.append(' '.join(line[1:]))

#final_vecs = {0 : dict(), 4 : dict(), 8 : dict()}
#final_vecs = {0 : dict(), 14 : dict(), 32 : dict()}
final_vecs = {'surprisal' : dict(), 'entropy' : dict()}
for s in tqdm(sents):
    inputs = tokenizer(s.strip(), return_tensors="pt").to("cuda")
    outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    )
    split_s = s.split()
    idxs = inputs['input_ids'].cpu()[0]
    vecs = dict()
    layer_vecs = dict()
    layer_entr = dict()
    curr_hidden = -1 * torch.log2(softmax(outputs['logits'])).detach().cpu().numpy()[0]
    assert curr_hidden.shape[0] == len(idxs)
    entropy_one = softmax(outputs['logits']).detach().cpu().numpy()[0]
    curr_entropy = -numpy.sum(entropy_one*numpy.log2(entropy_one), axis=1)
    assert curr_entropy.shape == (len(idxs),)
    curr_w = ''
    curr_counter = 0
    curr_vecs = list()
    curr_entr = list()
    for idx_i, idx in enumerate(idxs):
        curr_tok = tokenizer.convert_ids_to_tokens([idx])
        curr_text_tok = tokenizer.convert_tokens_to_string(curr_tok)
        curr_w += curr_text_tok
        curr_vecs.append(curr_hidden[idx_i][idx])
        curr_entr.append(curr_entropy[idx_i])
        if curr_w.strip() == split_s[curr_counter]:
            try:
                layer_vecs[curr_w.strip()].extend(curr_vecs)
                layer_entr[curr_w.strip()].extend(curr_entr)
            except KeyError:
                layer_vecs[curr_w.strip()] = curr_vecs
                layer_entr[curr_w.strip()] = curr_entr
            curr_w = ''
            curr_counter += 1
            curr_vecs = list()
            curr_entr = list()
    layer_vecs = {k : numpy.sum(v) for k, v in layer_vecs.items()}
    layer_entr = {k : numpy.sum(v) for k, v in layer_entr.items()}
    vecs['surprisal'] = layer_vecs
    vecs['entropy'] = layer_entr
    for layer, layer_vecs in vecs.items():
        if layer == 'surprisal':
            idx = 1
        else:
            idx = 0
        try:
            final_vecs[layer]['_'.join([split_s[0], split_s[1]])].append(layer_vecs[split_s[idx]])
        except KeyError:
            final_vecs[layer]['_'.join([split_s[0], split_s[1]])] = [layer_vecs[split_s[idx]]]
        if layer == 'surprisal':
            idx = 3
        else:
            idx = 2
        try:
            final_vecs[layer]['_'.join([split_s[1], split_s[3]])].append(layer_vecs[split_s[idx]])
        except KeyError:
            final_vecs[layer]['_'.join([split_s[1], split_s[3]])] = [layer_vecs[split_s[idx]]]

ws = list()
with open('transcriptions.txt') as i:
    for l in i:
        line = l.strip().split(';')
        ws.append((line[1], line[2]))
        ws.append((line[2], line[4]))
sims = {
        #'gpt2-small_0' : dict(),
        #'gpt2-small_4' : dict(),
        'gpt2-small_surprisal' : dict(),
        'gpt2-small_entropy' : dict(),
        #'gpt2-large_0' : dict(),
        #'gpt2-large_14' : dict(),
        #'gpt2-large_32' : dict(),
        }
vecs = {k : dict() for k in sims.keys()}
for layer, layer_vecs in final_vecs.items():
    layer_vecs = {k : numpy.average(v, axis=0) if len(v)>1 else v[0] for k, v in layer_vecs.items()}
    for _, w_one in ws:
        key = '{}_{}'.format(_, w_one)
        sims['gpt2-small_{}'.format(layer)][key] = [layer_vecs[key]]

'''
out_f = 'vecs'
os.makedirs(out_f, exist_ok=True)
for model, res in sims.items():
    with open(os.path.join(out_f, '{}.tsv'.format(model)), 'w') as o:
        for ks, sim in res.items():
            o.write('{}\t{}\t{}\n'.format(ks[0], ks[1], sim))
'''

out_f = 'vecs'
os.makedirs(out_f, exist_ok=True)
for model, res in sims.items():
    with open(os.path.join(out_f, '{}.tsv'.format(model)), 'w') as o:
        for ks, vec in res.items():
            o.write('{}\t'.format(ks))
            for dim in vec:
                o.write('{}\t'.format(dim))
            o.write('\n')
