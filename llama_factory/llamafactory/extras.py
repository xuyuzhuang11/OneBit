import os
import json
import time
import sys
import logging
import tiktoken
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union, Dict, Optional, Tuple
from datetime import timedelta

import matplotlib.pyplot as plt

import torch
from transformers import TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import has_length
try:
    from transformers.utils import (
        is_torch_bf16_cpu_available,
        is_torch_bf16_gpu_available,
        is_torch_cuda_available,
        is_torch_npu_available
    )
    _is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
    _is_bf16_available = is_torch_bf16_gpu_available() or is_torch_bf16_cpu_available
except ImportError:
    _is_fp16_available = torch.cuda.is_available()
    _is_bf16_available = torch.cuda.is_bf16_supported()

from transformers import TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizer
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList


IGNORE_INDEX = -100
LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp", "ln_1", "ln_2"]
LOG_FILE_NAME = "trainer_log.jsonl"


'''logging
'''
class LoggerHandler(logging.Handler):

    def __init__(self):
        super().__init__()
        self.log = ""

    def reset(self):
        self.log = ""

    def emit(self, record):
        if record.name == "httpx":
            return
        log_entry = self.format(record)
        self.log += log_entry
        self.log += "\n\n"


def reset_logging():
    r"""
    Removes basic config of root logger
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))


def get_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

logger = get_logger(__name__)


'''callbacks
'''
class LogCallback(TrainerCallback):

    def __init__(self, runner=None):
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps
            if os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME)) and args.overwrite_output_dir:
                logger.warning("Previous log file in this folder will be deleted.")
                os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if not state.is_local_process_zero:
            return

        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time
        )
        if self.runner is not None:
            logger.info("{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                logs["loss"] or 0, logs["learning_rate"] or 0, logs["epoch"] or 0
            ))

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def on_prediction_step(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()


'''template
'''
@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return system, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else: # baichuan, qwen and gpt2 models have no bos token
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        if self.efficient_eos: # used in baichuan, qwen, chatglm, etc.
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(elem)))

        return token_ids


@dataclass
class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + query    resp + eos
        Turn t: bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0: # llama2 template has no sep_ids
                query = self.prefix[0].replace("{{system}}", system) + query
            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
    efficient_eos: Optional[bool] = False
) -> None:
    template_class = Llama2Template if "llama2" in name else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history,
        efficient_eos=efficient_eos
    )


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    return template


r"""
Supports language model inference without histories.
"""
register_template(
    name="vanilla",
    prefix=[],
    prompt=[
        "{{query}}"
    ],
    system="",
    sep=[],
    use_history=False
)


r"""
Default template.
"""
register_template(
    name="default",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    sep=[
        "\n"
    ]
)


r"""
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
"""
register_template(
    name="llama2",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    sep=[]
)


r"""
Supports: https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
          https://huggingface.co/ziqingyang/chinese-alpaca-2-13b
"""
register_template(
    name="llama2_zh",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    system="You are a helpful assistant. 你是一个乐于助人的助手。",
    sep=[]
)


r"""
Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
"""
register_template(
    name="alpaca",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "### Instruction:\n{{query}}\n\n### Response:\n"
    ],
    system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ),
    sep=[
        "\n\n"
    ]
)


r"""
Supports: https://huggingface.co/lmsys/vicuna-7b-v1.5
          https://huggingface.co/lmsys/vicuna-13b-v1.5
"""
register_template(
    name="vicuna",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "USER: {{query}} ASSISTANT:"
    ],
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    sep=[]
)


r"""
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_template(
    name="belle",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\n\nBelle: "
    ],
    system="",
    sep=[
        "\n\n"
    ]
)


r"""
Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
          https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
          https://huggingface.co/IDEA-CCNL/Ziya2-13B-Chat
"""
register_template(
    name="ziya",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        {"token": "<human>"},
        ":{{query}}\n",
        {"token": "<bot>"},
        ":"
    ],
    system="",
    sep=[
        "\n"
    ]
)


r"""
Supports: https://huggingface.co/BAAI/AquilaChat-7B
          https://huggingface.co/BAAI/AquilaChat2-7B
          https://huggingface.co/BAAI/AquilaChat2-34B
"""
register_template(
    name="aquila",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}###Assistant:"
    ],
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    sep=[
        "###"
    ],
    stop_words=[
        "</s>"
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/internlm/internlm-chat-7b
          https://huggingface.co/internlm/internlm-chat-20b
"""
register_template(
    name="intern",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "<|User|>:{{query}}",
        {"token": "<eoh>"},
        "\n<|Bot|>:"
    ],
    system="",
    sep=[
        {"token": "<eoa>"},
        "\n"
    ],
    stop_words=[
        "<eoa>"
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_template(
    name="baichuan",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        {"token": "<reserved_102>"}, # user token
        "{{query}}",
        {"token": "<reserved_103>"}  # assistant token
    ],
    system="",
    sep=[],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
          https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
"""
register_template(
    name="baichuan2",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        {"token": "<reserved_106>"}, # user token
        "{{query}}",
        {"token": "<reserved_107>"}  # assistant token
    ],
    system="",
    sep=[],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_template(
    name="starchat",
    prefix=[
        {"token": "<|system|>"},
        "\n{{system}}",
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n{{query}}",
        {"token": "<|end|>"},
        "\n",
        {"token": "<|assistant|>"}
    ],
    system="",
    sep=[
        {"token": "<|end|>"},
        "\n"
    ],
    stop_words=[
        "<|end|>"
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
          https://huggingface.co/Qwen/Qwen-14B-Chat
"""
register_template(
    name="chatml",
    prefix=[
        {"token": "<|im_start|>"},
        "system\n{{system}}"
    ],
    prompt=[
        {"token": "<|im_start|>"},
        "user\n{{query}}",
        {"token": "<|im_end|>"},
        "\n",
        {"token": "<|im_start|>"},
        "assistant\n"
    ],
    system="You are a helpful assistant.",
    sep=[
        {"token": "<|im_end|>"},
        "\n"
    ],
    stop_words=[
        "<|im_end|>"
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/THUDM/chatglm2-6b
"""
register_template(
    name="chatglm2",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{system}}"
    ],
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
    system="",
    sep=[
        "\n\n"
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/THUDM/chatglm3-6b
"""
register_template(
    name="chatglm3",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{system}}"
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n",
        "{{query}}",
        {"token": "<|assistant|>"}
    ],
    system="",
    sep=[],
    stop_words=[
        "<|user|>",
        "<|observation|>"
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/openchat/openchat_v3.2_super
"""
register_template(
    name="openchat",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "GPT4 User: {{query}}",
        {"token": "<|end_of_turn|>"},
        "GPT4 Assistant:"
    ],
    system="",
    sep=[
        {"token": "<|end_of_turn|>"}
    ],
    efficient_eos=True
)


r"""
Supports: https://huggingface.co/xverse/XVERSE-7B-Chat
          https://huggingface.co/xverse/XVERSE-13B-Chat
"""
register_template(
    name="xverse",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\n\nAssistant: "
    ],
    system="",
    sep=[]
)


'''misc
'''
def get_logits_processor() -> LogitsProcessorList:
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


'''ploting
'''
def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5) # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]) -> None:

    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(save_dictionary, "training_{}.png".format(key)), format="png", dpi=100)
        print("Figure saved:", os.path.join(save_dictionary, "training_{}.png".format(key)))
