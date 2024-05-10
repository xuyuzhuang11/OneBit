import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import BitLlamaForCausalLM, LlamaTokenizer
from sklearn.decomposition import NMF


model_name = sys.argv[1]
teacher_model_path = sys.argv[2]
student_start_path = sys.argv[3]

logging.getLogger("transformers").setLevel(logging.CRITICAL)
target_model = BitLlamaForCausalLM.from_pretrained(teacher_model_path)
tokenizer = LlamaTokenizer.from_pretrained(teacher_model_path)

modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']
n_layers = len(target_model.model.layers)
nmf_model = NMF(n_components=1, init='random', random_state=0)

modules_dict = {name: module for name, module in target_model.named_modules()}
print('scale...')
for layer in range(n_layers):
    for module in modules:
        module_name = f'model.layers.{layer}.{module}'
        module_obj = modules_dict[module_name]
        A = module_obj.weight.detach().numpy()
        W = nmf_model.fit_transform(np.abs(A))
        H = nmf_model.components_
        module_obj.input_factor = nn.Parameter(torch.from_numpy(H).squeeze(0))
        module_obj.weight_scale = nn.Parameter(torch.from_numpy(W).squeeze(1))
        module_obj.weight = nn.Parameter(torch.sign(torch.from_numpy(A)) * 0.01)

print('saving...')
target_model.save_pretrained(student_start_path)
tokenizer.save_pretrained(student_start_path)
