from typing import TYPE_CHECKING, Optional, List, Dict, Union, Any, Tuple

import os
import random
import json
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from transformers import Trainer

from llamafactory.dsets import get_dataset, preprocess_dataset, split_dataset
from llamafactory.extras import IGNORE_INDEX
# from llama_factory.extras import get_logits_processor
from llamafactory.extras import plot_loss, get_logger
from llamafactory.core import load_model_and_tokenizer
# from llmtuner.tuner.sft.metric import ComputeMetrics
from transformers import DataCollatorForLanguageModeling

from transformers import TrainerCallback
from transformers.trainer import PredictionOutput
from llamafactory.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


logger = get_logger(__name__)

class KDTrainer(Trainer):
    r"""
    Mainly implement the kd-loss for supporting knowledge distllation from teacher model to student model.
    The other functions are relics from CustomSeq2SeqTrainer.
    """
    
    def ce_loss(self, student_logits, teacher_logits):

        model_output_log_prob = F.log_softmax(student_logits, dim=2)
        real_output_soft = F.softmax(teacher_logits, dim=2)

        loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        
        with torch.no_grad():
            # model.teacher_model.eval()
            teacher_outputs = model.teacher_model(
                **inputs,
                output_hidden_states=(model.kd_beta > 0),
                output_attentions=(model.kd_gamma > 0),
                # **inputs, output_hidden_states=True, output_attentions=True
            )
        teacher_logits = teacher_outputs.logits
        teacher_hidden_states = teacher_outputs.hidden_states           # still no detached!
        teacher_attn_weights = teacher_outputs.attentions               # still no detached!
        del teacher_outputs
        
        # print(inputs)
        outputs = model(**inputs, output_hidden_states=(model.kd_beta > 0), output_attentions=(model.kd_gamma > 0))
        student_loss = outputs.loss
        student_logits = outputs.logits
        student_hidden_states = outputs.hidden_states
        student_attn_weights = outputs.attentions

        if not return_outputs:
            del outputs
        
        kd_loss = 0.0
        
        if model.kd_loss_scale > 0.0:
            kd_loss = self.ce_loss(student_logits, teacher_logits)

        del teacher_logits
        del student_logits

        # print(model.kd_alpha, model.kd_loss_scale, kd_loss, student_loss)
        tok_loss = model.kd_alpha * model.kd_loss_scale * kd_loss + (1 - model.kd_alpha) * student_loss
        logger.info(f"kd_loss: {model.kd_loss_scale * kd_loss} kd_loss_after: {model.kd_alpha * model.kd_loss_scale * kd_loss} student_loss: {student_loss}")
        
        n_layers = len(model.model.layers)
        # n_layers = len(model.model.decoder.layers)
        if model.kd_beta > 0:
            pkd_loss = 0.0
            for i in range(n_layers):
                teacher_hidden_state = teacher_hidden_states[i]
                student_hidden_state = student_hidden_states[i]
                teacher_hidden_state = teacher_hidden_state.view(-1, teacher_hidden_state.shape[-1])
                student_hidden_state = student_hidden_state.view(-1, teacher_hidden_state.shape[-1])
                teacher_hidden_state = F.normalize(teacher_hidden_state, p=2, dim=1)
                student_hidden_state = F.normalize(student_hidden_state, p=2, dim=1)
                diff = teacher_hidden_state - student_hidden_state
                squared_diff = torch.mean(torch.norm(diff, p=2, dim=1) ** 2)
                pkd_loss += squared_diff
            tok_loss += model.kd_beta * pkd_loss
            logger.info(f"pkd_loss: {pkd_loss}")
            
        if model.kd_gamma > 0:
            attn_loss = 0.0
            for i in range(n_layers):
                teacher_attn_weight = teacher_attn_weights[i]
                student_attn_weight = student_attn_weights[i]
                teacher_attn_weight = teacher_attn_weight.view(-1, teacher_attn_weight.shape[-1])
                student_attn_weight = student_attn_weight.view(-1, student_attn_weight.shape[-1])
                diff = teacher_attn_weight - student_attn_weight
                squared_diff = torch.mean(torch.norm(diff, p=2, dim=1) ** 2)
                attn_loss += squared_diff
            tok_loss += model.kd_gamma * attn_loss
            logger.info(f"attn_loss: {attn_loss}")

        return (tok_loss, outputs) if return_outputs else tok_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = self._pad_tensors_to_target_len(
                        inputs["attention_mask"], inputs["labels"], pad_token_id=0
                    )
                if "position_ids" in inputs:
                    inputs["position_ids"] = self._pad_tensors_to_target_len(
                        inputs["position_ids"], inputs["labels"], pad_token_id=0
                    )

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


def run_kd(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="kd")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="kd")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = KDTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args)
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        # print(metrics)
        try:
            import math
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
