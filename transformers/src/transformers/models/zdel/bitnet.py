import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math


bitlinear_config = {
    'activation_bits' : 16,
    'is_weight_train' : True,
    'is_weight_bias_train' : True,
    'is_input_scale_train' : True,
    'r_low_rank': 4,
}


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

class SignSTE(nn.Module):
    def forward(self, input):
        return SignSTEFunc.apply(input)


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # self.abits = bitlinear_config['activation_bits']
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        # self.layernorm = nn.LayerNorm(in_features, elementwise_affine=False)
        if bitlinear_config['is_weight_train']:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        else:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)        
        
        if bitlinear_config['is_weight_bias_train']:
            self.weight_scale = nn.Parameter(torch.empty((out_features, bitlinear_config['r_low_rank']), **factory_kwargs))
        else:
            self.weight_scale = nn.Parameter(torch.empty((out_features, bitlinear_config['r_low_rank']), **factory_kwargs), requires_grad=False)
        # self.layernorm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.sign = SignSTE()
        
        if bitlinear_config['is_input_scale_train']:
            self.input_factor = nn.Parameter(torch.empty((bitlinear_config['r_low_rank'], in_features), dtype=torch.float16) * 1.0)
        else:
            self.input_factor = nn.Parameter(torch.empty((bitlinear_config['r_low_rank'], in_features), dtype=torch.float16) * 1.0,  requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # randomly initialize weight
        # prob = 0.5
        # mask = torch.bernoulli(torch.full(self.weight.shape, prob)).bool()
        # self.weight.data[mask] = -1
        # init.constant_(self.weight, 1.0)
        init.constant_(self.weight_scale, 1.0)
        init.constant_(self.input_factor, 0.01)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def initial_weight(self, weight: torch.Tensor = None, weight_scale: torch.Tensor = None):
        if weight is not None:
            if self.weight.device != weight.device:
                weight = weight.to(self.weight.device)
            self.weight = nn.Parameter(weight, requires_grad=False) if not bitlinear_config['is_weight_train'] else nn.Parameter(weight, requires_grad=True)
        if weight_scale is not None:
            if self.weight_scale.device != weight_scale.device:
                weight_scale = weight_scale.to(self.weight_scale.device)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False) if not bitlinear_config['is_weight_bias_train'] else nn.Parameter(weight_scale, requires_grad=True)
        
    def dynamic_calibration(self, x):
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        # xmin = self.upbound_factor * xmin
        # xmax = self.upbound_factor * xmax
        abs_max = torch.max(xmin.abs(), xmax.abs())
        scale = abs_max / (2 ** (self.abits - 1) - 1)
        self.scale = clamp_ste(scale, min=1e-5, max=1e4)
        # self.zero = (2 ** (self.abits - 1) - 1) * torch.ones_like(self.scale)

    def forward(self, input):
        dtype = input.dtype
        input = input.to(torch.float16)
        # input = self.layernorm(input)
        # input = input * self.input_factor.view(1, self.in_features)
        
        weight_value = F.linear(self.weight_scale, self.input_factor.T)
        weight_sign = self.sign(self.weight)
        weight = weight_sign * weight_value
        output = F.linear(input, weight)
        # print(output_int.shape, self.scale.shape)
        # output *= self.weight_scale.view(1, self.out_features)
        
        return output


class BitLinearBaseline(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.layernorm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.sign = SignSTE()
        # self.clip = ClampSTE()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        dtype = input.dtype
        
        weight = self.weight - self.weight.mean()
        beta = torch.abs(weight).mean()
        input = self.layernorm(input)
        weight = self.sign(weight)                                 # not self.sign !!!!!!!
        output = F.linear(input, weight).to(dtype=dtype)

        output = output * beta
        return output


if __name__ == "__main__":
    bit = BitLinear(4096, 4096).to('cuda')
    inputs = torch.load('')
    weight = torch.load('')
    weight_scale = torch.load('')
    bit.initial_weight(weight, weight_scale)
    # tensor = torch.randn((8, 4096), dtype=torch.float16).to('cuda')
    bit(inputs)
