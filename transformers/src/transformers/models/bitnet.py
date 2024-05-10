import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (torch.clamp(x, min, max) - x).detach() + x

class SignSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (1.001 - torch.tanh(input) ** 2)
        # return grad_output * (1.01 - torch.tanh(input) ** 2)

class SignSTE(nn.Module):
    def forward(self, input):
        return SignSTEFunc.apply(input)


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups            # now omitted!
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_scale = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.sign = SignSTE()
        self.input_factor = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.layernorm = nn.LayerNorm(out_features, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight_scale, 1.0)
        init.constant_(self.input_factor, 1.0)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        input = input * self.input_factor.view(1, self.in_features)
        weight = self.sign(self.weight)
        output = F.linear(input, weight)
        output *= self.weight_scale.view(1, self.out_features)
        
        output = self.layernorm(output)
        if self.bias is not None:
            output += self.bias
        
        return output
    
    
class BitLinearInf(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups            # now omitted!
        self.weight = nn.Parameter(torch.empty((out_features, in_features // 8), device=factory_kwargs['device'], dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        self.input_factor = nn.Parameter(torch.empty(in_features, **factory_kwargs), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.layernorm = nn.LayerNorm(out_features, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight_scale, 1.0)
        init.constant_(self.input_factor, 1.0)
        init.constant_(self.weight, 0)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def int8_to_fp16(self, int8_tensor):
        dtype = self.weight_scale.dtype
        shifts = torch.arange(8, device=int8_tensor.device).view(1, 1, 8)
        # expand dimension of int8_tensor to suite for shifts to broadcast
        expanded_int8 = int8_tensor.unsqueeze(-1)
        
        # parallel bit operations
        unpacked_bits = ((expanded_int8 >> shifts) & 1).to(dtype)
        unpacked_bits = unpacked_bits.view(int8_tensor.shape[0], -1)
        
        # convert 0/1 to +1/-1
        fp16_tensor = -2 * unpacked_bits + 1
        return fp16_tensor
    
    def forward(self, input):
        input = input * self.input_factor.view(1, self.in_features)
        weight = self.int8_to_fp16(self.weight)
        output = F.linear(input, weight)
        output *= self.weight_scale.view(1, self.out_features)
        
        output = self.layernorm(output)
        if self.bias is not None:
            output += self.bias
        
        return output