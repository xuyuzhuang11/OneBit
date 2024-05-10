import sys
import os
import math
from typing import Any, Dict, Optional, Tuple, Literal, List

import torch
import datasets
from types import MethodType
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, PreTrainedModel
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.trainer import WEIGHTS_NAME
from transformers.utils.versions import require_version
from transformers.models.llama import modeling_llama as LlamaModule
from trl import AutoModelForCausalLMWithValueHead

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments
)
from llamafactory.extras import get_logger, reset_logging, count_parameters, infer_optim_dtype, LAYERNORM_NAMES
import llamafactory.llama_patch as LlamaPatches

logger = get_logger(__name__)


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def parse_train_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments,
        GeneratingArguments
    ))
    return _parse_args(parser, args)



def get_train_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    model_args, data_args, training_args, finetuning_args, generating_args = parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments
    data_args.init_for_training(training_args.seed)

    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type != "lora":
        raise ValueError("RM and PPO stages can only be performed with the LoRA method.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.resume_from_checkpoint is not None:
        raise ValueError("RM and PPO stages do not support `resume_from_checkpoint`.")

    if finetuning_args.stage in ["ppo", "dpo"] and not training_args.do_train:
        raise ValueError("PPO and DPO stages can only be performed at training.")

    if finetuning_args.stage in ["rm", "dpo"]:
        for dataset_attr in data_args.dataset_list:
            if not dataset_attr.ranking:
                raise ValueError("Please use ranked datasets for reward modeling or DPO training.")

    if finetuning_args.stage == "ppo" and model_args.reward_model is None:
        raise ValueError("Reward model is necessary for PPO training.")

    if finetuning_args.stage == "ppo" and data_args.streaming:
        raise ValueError("Streaming mode does not suppport PPO training currently.")

    if finetuning_args.stage == "ppo" and model_args.shift_attn:
        raise ValueError("PPO training is incompatible with S^2-Attn.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_train and finetuning_args.finetuning_type == "lora" and finetuning_args.lora_target is None:
        raise ValueError("Please specify `lora_target` in LoRA training.")

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora" and len(model_args.checkpoint_dir) != 1:
            raise ValueError("Only LoRA tuning accepts multiple checkpoints.")

        if model_args.quantization_bit is not None:
            if len(model_args.checkpoint_dir) != 1:
                raise ValueError("Quantized model only accepts a single checkpoint. Merge them first.")
            
            if not finetuning_args.resume_lora_training:
                raise ValueError("Quantized model cannot create new LoRA weight. Merge them first.")

    if training_args.do_train and model_args.quantization_bit is not None and (not finetuning_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    # postprocess training_args
    if (
        training_args.local_rank != -1
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(ddp_find_unused_parameters=False))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args_dict = training_args.to_dict()
            training_args_dict.update(dict(resume_from_checkpoint=last_checkpoint))
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
            logger.info(
                "Resuming from checkpoint. Change `output_dir` or use `overwrite_output_dir` to avoid."
            )

    # postprocess model_args
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    # Log on each process the small summary:
    logger.info("Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, compute dtype: {}".format(
        training_args.local_rank, training_args.device, training_args.n_gpu,
        bool(training_args.local_rank != -1), str(model_args.compute_dtype)
    ))
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args


def prepare_model_for_training(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layernorm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> "PreTrainedModel":
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """
    if finetuning_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in layernorm_names):
                param.data = param.data.to(torch.float32)
        logger.info("Upcasting weights in layernorm in float32.")

    if finetuning_args.neft_alpha > 1e-6:
        input_embed = model.get_input_embeddings()
        if isinstance(input_embed, torch.nn.Embedding):
            def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                embeddings = input_embed.__class__.forward(self, x)
                if self.training:
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = finetuning_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                return embeddings

            input_embed.forward = MethodType(noisy_forward, input_embed)
            logger.info("Using noisy embedding with alpha={:.2f}".format(finetuning_args.neft_alpha))
        else:
            logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled
        logger.info("Gradient checkpointing enabled.")

    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def forward_in_fp32(self, x: torch.Tensor) -> torch.Tensor:
                return output_layer.__class__.forward(self, x.to(output_layer.weight.dtype)).to(torch.float32)

            output_layer.forward = MethodType(forward_in_fp32, output_layer)

    return model


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    vhead_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(vhead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    vhead_params = torch.load(vhead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
    model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
    model.register_buffer("default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False)
    model.register_buffer("default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False)
    return True


def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None,
    output_layer_name: Optional[str] = "lm_head"
) -> List[str]:
    if quantization_bit is not None:
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:
        linear_cls = torch.nn.Linear

    module_names = set()
    for name, module in model.named_modules():
        if output_layer_name not in name and isinstance(module, linear_cls):
            module_names.add(name.split(".")[-1])

    if output_layer_name in module_names:
        module_names.pop(output_layer_name)

    return list(module_names)


def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    is_mergeable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        num_layers = getattr(model.config, "num_layers")
        if finetuning_args.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]

        trainable_layers = ["{:d}.{}".format(idx, finetuning_args.name_module_trainable) for idx in trainable_layer_ids]
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable): # continually fine-tuning
                checkpoints_to_merge, latest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        if is_trainable and latest_checkpoint is None: # create new lora weights while training
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model, model_args.quantization_bit)
            else:
                target_modules = finetuning_args.lora_target

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=finetuning_args.additional_target
            )
            model = get_peft_model(model, lora_config)
            if id(model.peft_config) != id(model.base_model.peft_config): # https://github.com/huggingface/peft/issues/923
                model.base_model.peft_config = model.peft_config

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    stage: Optional[Literal["pt", "sft", "rm", "ppo", "kd"]] = "sft"
) -> Tuple[PreTrainedModel, "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    # By Yuzhuang
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
    # tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs
    )

    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)

    # Fix tokenizer (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # Set model dtype
    if model_args.compute_dtype is not None: # for training
        setattr(config, "torch_dtype", model_args.compute_dtype)
    else: # for evaluation, priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # Fix config (for Qwen)
    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

    # Set RoPE scaling
    if model_args.rope_scaling is not None:
        if hasattr(config, "use_dynamic_ntk"): # for Qwen models
            if is_trainable:
                logger.warning("Qwen model does not support RoPE scaling in training.")
            else:
                setattr(config, "use_dynamic_ntk", True)
                setattr(config, "use_logn_attn", True)
                logger.info("Using dynamic NTK scaling.")

        elif hasattr(config, "rope_scaling"): # for LLaMA and Falcon models
            if is_trainable:
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )

                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and model_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
                else:
                    logger.warning("Input length is smaller than max length. Consider increase input length.")
                    scaling_factor = 1.0
            else:
                scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))

        else:
            logger.warning("Current model does not support RoPE scaling.")

    # Set FlashAttention-2
    if model_args.flash_attn:
        if getattr(config, "model_type", None) == "llama":
            LlamaModule.LlamaAttention = LlamaPatches.LlamaFlashAttention2
            LlamaModule.LlamaModel._prepare_decoder_attention_mask = LlamaPatches._prepare_decoder_attention_mask
            logger.info("Using FlashAttention-2 for faster training and inference.")
        elif getattr(config, "model_type", None) == "qwen":
            logger.info("Qwen models automatically enable FlashAttention if installed.")
        else:
            logger.warning("Current model does not support FlashAttention-2.")
    elif is_trainable and model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        LlamaModule.LlamaAttention = LlamaPatches.LlamaShiftShortAttention
        logger.warning("Using `--flash_attn` for faster training in large context length.")

    # Set shift short attention (S^2-Attn)
    if is_trainable and model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(config, "group_size_ratio", 0.25)
            logger.info("Using shift short attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shift short attention.")

    # Quantization configurations (using bitsandbytes library).
    is_mergeable = True
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # Load and prepare pre-trained models (without valuehead).
    # from Xu Yuzhuang
    from transformers import BitLlamaForCausalLM
    
    model = BitLlamaForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )
    
    if stage == "kd":
        assert model_args.teacher_model_name_or_path is not None, "teacher model path is None!"
        teacher_config = AutoConfig.from_pretrained(model_args.teacher_model_name_or_path, **config_kwargs)
        model.teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_name_or_path,
            config=teacher_config,
            torch_dtype=torch.float16,
            **config_kwargs
        )

    # Disable custom generate method (for Qwen and Baichuan2)
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # Initialize adapters
    model = prepare_model_for_training(model=model, finetuning_args=finetuning_args) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)
    model = model.train() if is_trainable else model.eval()
    
    if stage == "kd":
        model.kd_loss_scale = model_args.kd_loss_scale
        model.kd_alpha = model_args.kd_alpha
        model.kd_beta = model_args.kd_beta
        model.kd_gamma = model_args.kd_gamma
        # model.teacher_model = model.teacher_model.eval()
        for param in model.teacher_model.parameters():
            param.requires_grad = False

    # Prepare model with valuehead for RLHF
    if stage == "rm" or stage == "ppo":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        model._keys_to_ignore_on_save = None
        reset_logging()
        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            logger.warning("Only the last checkpoint containing valuehead will be loaded.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            logger.info("Load reward model from {}".format(model_args.reward_model))
            if getattr(model, "is_peft_model", False):
                model.pretrained_model.load_adapter(model_args.reward_model, "reward")
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer
