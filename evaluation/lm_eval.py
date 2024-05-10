import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pprint import pprint
from lm_eval import evaluator
from lm_eval.log import create_logger
from lm_eval.datautils import get_loaders
from lm_eval.categories import subcategories, categories
from lm_eval.parallel_utils import map_layers_to_multi_gpus
from lm_eval.LMClass import LMClass
# from transformers.models import LlamaForCausalLM, BitLlamaForCausalLM

args = {
    "multigpu": False,
    "batch_size": 32,
    "net": "onebitllama",
    "model_family": "onebitllama",
    "eval_ppl": True,
    "seed": 1234,
    "model": '/home/xyz/checkpoints/onebit_llama_7b',
    # "tasks": '',
    "tasks": 'hellaswag,winogrande,piqa,boolq,arc_easy,arc_challenge',
    "num_fewshot": 0,
    "limit": -1,
}

logger = create_logger('/home/xyz/evaluation')

@torch.no_grad()
def evaluate(lm):
    results = {}
    if args['multigpu']:
        if "opt" in args['net'].lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args['net'].lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args['net'].lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args['net'].lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args['net'].lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args['net'].lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)


    if args['eval_ppl']:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["c4", "wikitext2"]:
            cache_testloader = f'/home/xyz/evaluation/testloader_{args["model_family"]}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args['seed'],
                    model=args['model'],
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            # print(nsamples)
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                if "opt" in args['net'].lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args['net'].lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args['model']:
                    outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                # print(logits)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args['limit']:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            print(f'PPL {dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if args['tasks'] != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args['tasks'],
            num_fewshot=args['num_fewshot'],
            limit=None if args['limit'] == -1 else args['limit'],
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args['tasks']:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results


lm = LMClass(args)
lm.seqlen = 2048
lm.model.eval()
for param in lm.model.parameters():
    param.requires_grad = False
evaluate(lm)
