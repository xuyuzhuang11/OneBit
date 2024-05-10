import torch
from transformers import BitLlamaForCausalLM, BitLlamaForCausalLMInf, LlamaTokenizer, BitLinear

ckpt_path = '/home/xyz/scripts/onebit_llama_7b_trainckpt'
inf_ckpt_path = '/home/xyz/checkpoints/onebit_llama2_13b'

def fp16_to_int8(fp16_tensor):
    # assert torch.all((fp16_tensor == 1) | (fp16_tensor == -1)), "Tensor should only contain 1 or -1."
    # assert fp16_tensor.shape[1] % 8 == 0, "The width of the tensor must be a multiple of 8."
    int_tensor = ((0 - fp16_tensor + 1) / 2).to(torch.uint8)
    reshaped = int_tensor.view(int_tensor.shape[0], -1, 8)
    multiplier = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=int_tensor.device)
    packed_tensor = torch.matmul(reshaped, multiplier).type(torch.int8)
    packed_tensor = packed_tensor.view(int_tensor.shape[0], -1)
    return packed_tensor

model_fp16 = BitLlamaForCausalLM.from_pretrained(ckpt_path)
model_int8 = BitLlamaForCausalLMInf.from_pretrained(ckpt_path, ignore_mismatched_sizes=True)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
modules_dict = {name: module for name, module in model_int8.named_modules()}
for param in model_int8.parameters():
    param.requires_grad = False
for param in model_fp16.parameters():
    param.requires_grad = False

for name, module in model_fp16.named_modules():
    if isinstance(module, BitLinear):
        print(f"Converting {name}...")
        fp16_tensor = module.weight.data
        int8_tensor = fp16_to_int8(torch.sign(fp16_tensor))
        int8_module = modules_dict[name]
        int8_module.weight.data = int8_tensor
        int8_module.weight_scale.data = module.weight_scale.data
        int8_module.input_factor.data = module.input_factor.data


model_int8.save_pretrained(inf_ckpt_path)
tokenizer.save_pretrained(inf_ckpt_path)
print(f"Modified model saved to {inf_ckpt_path}")
