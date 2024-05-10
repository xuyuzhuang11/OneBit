from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llamafactory.extras import LogCallback
from llamafactory.core import get_train_args
from llamafactory.sft import run_sft
from llamafactory.kd import run_kd

if TYPE_CHECKING:
    from transformers import TrainerCallback


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "kd":
        run_kd(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def main():
    run_exp()


if __name__ == "__main__":
    main()
