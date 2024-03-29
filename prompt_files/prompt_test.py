import argparse
import copy
import json
import os
import glob
import random
import re
import sys
from pprint import pprint
from collections import OrderedDict
from tqdm import tqdm


sys.path.append(os.getcwd())


from prompt_files.prompt_dataset import MemT5PromptDSTDataset

import torch
DEVICE='cuda' if torch.cuda.device_count() > 0 else 'cpu'


def predict_t5(model, tokenizer, batch, batch_ds, batch_dials):
    dst_input_ids = batch['input_ids_womask'].to(DEVICE)
    dst_input_mask = batch['attention_mask_womask'].to(DEVICE)
    dst_decoder_input_ids = batch.get('decoder_inputs_womask', None)

    if dst_decoder_input_ids is not None:
        dst_decoder_input_ids = dst_decoder_input_ids.to(DEVICE)

    # prepare batch graph for test
    batch_graph = batch['batch_graph'].to(DEVICE)
    #x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
    #print(batch_dials[0]['state'] )

    if dst_decoder_input_ids is not None:
        outputs = model.generate(
            input_ids=dst_input_ids,
            attention_mask=dst_input_mask,
            decoder_input_ids=dst_decoder_input_ids,
            use_cache=False,
            return_dict_in_generate=True,
            max_length=100,
            batch_graph=batch_graph,
        )
    else:
        outputs = model.generate(
            input_ids=dst_input_ids,
            attention_mask=dst_input_mask,
            use_cache=False,
            return_dict_in_generate=True,
            max_length=100,
            batch_graph=batch_graph,
        )

    dst_predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=True)
    dst_predictions = [re.sub('\<(prompt_)(.*?)\>', '', s) for s in dst_predictions]
    dst_predictions = [_.strip() for _ in dst_predictions]

    # debug
    # if small_sample_run:
    # for _ds, inp, p in zip(batch_ds[:2], batch['input_ids_womask'], dst_predictions):
    #     print('===========================')
    #     print(_ds)
    #     print(tokenizer.decode(inp, skip_special_tokens=False,
    #                            clean_up_tokenization_spaces=True))
    #     print(p)
    #     input()

    # process predicted values
    for i, (pred_str, ds_str) in enumerate(zip(dst_predictions, batch_ds)):
        ds_list = ds_str.split()
        # assert 'pred_state' not in batch_dials[i]
        batch_dials[i]['pred_state'] = {}
        for j, ds in enumerate(ds_list):
            left_token = '<extra_id_{}>'.format(j)
            right_token = '<extra_id_{}>'.format(j + 1)
            if right_token not in pred_str:
                right_token = '</s>'
            try:
                value = pred_str.split(left_token)[1].split(right_token)[0].strip()
            except:
                value = 'none'
            if value == 'none' or value == '':
                continue
            batch_dials[i]['pred_state'][ds] = value


def predict_gpt2(model, tokenizer, batch, batch_ds, batch_dials):
    gen_input_ids = batch['gen_input_ids'].to(DEVICE)

    outputs = model.generate(
        input_ids=gen_input_ids,
        return_dict_in_generate=True,
        max_length=800
    )
    outputs.sequences = outputs.sequences[:, gen_input_ids.shape[1]:]
    dst_predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
    dst_predictions = [_.replace(tokenizer.eos_token, '') for _ in dst_predictions]
    # dst_predictions = [_ if _ != 'don\'t care' else 'dontcare' for _ in dst_predictions]

    for i, (pred_str, ds_str) in enumerate(zip(dst_predictions, batch_ds)):
        ds_list = ds_str.split()
        value_list = pred_str.split(',')
        assert 'pred_state' not in batch_dials[i]
        batch_dials[i]['pred_state'] = {}
        for j, ds in enumerate(ds_list):
            if j >= len(value_list):
                value = 'none'
            else:
                value = value_list[j].strip()
            if value == 'none' or value == '':
                continue
            batch_dials[i]['pred_state'][ds] = value


def predict(model, dataset, tokenizer, args, test_domain_ids=None):
    model.to(DEVICE)
    pred_metrics = OrderedDict()
    for cl_domain_id, dials in dataset.dialogs.items():
        if test_domain_ids is not None and cl_domain_id not in test_domain_ids:
            continue
        cl_domain = tuple(eval(cl_domain_id))
        if cl_domain not in dataset.cl_domain_to_ds:
            print('DOMAIN {} not predicted'.format(cl_domain))
            continue
        dss = dataset.cl_domain_to_ds[cl_domain]

        num_dialog_per_batch = args.pred_batch_size
        assert num_dialog_per_batch > 0

        # prepare batches
        if args.small_sample_run:
            dials = dials[:10]

        for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=str(cl_domain)):
            test_batch = []
            batch_dials = dials[s: s + num_dialog_per_batch]
            for dial in batch_dials:
                test_batch.append(dataset.prepare_for_generation(dial, cl_domain))
            batch = dataset.collate_fn(test_batch)
            batch_ds = [_['ds'] for _ in test_batch]
            # if args.model_type == 't5':
            predict_t5(model, tokenizer, batch, batch_ds, batch_dials)
            # elif args.model_type == 'gpt2':
            #     predict_gpt2(model, tokenizer, batch, batch_ds, batch_dials)

        # compute metric
        slot_acc = {ds: 0 for ds in dss}
        joint_total = 0
        joint_acc = 0
        for dial in dials:
            joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
            for ds in dss:
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        joint_accuracy = joint_acc / joint_total
        slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
        avg_acc = sum(slot_acc.values()) / len(slot_acc)
        pred_metrics[cl_domain] = joint_accuracy
        print('{} acc: {}'.format(cl_domain, joint_accuracy))

    pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
    print('===================metrics===========')
    pprint(pred_metrics)
    json.dump(dataset.dialogs, open(os.path.join(args.output_dir, 'predict_results.json'), 'w'), indent=4)
    print('saving predictions to {}'.format(os.path.join(args.output_dir, 'predict_results.json')))

    return pred_metrics


def prompt5_predict(model, dataset:MemT5PromptDSTDataset, tokenizer, args, test_domain_ids=None):
    model.to(DEVICE)
    pred_metrics = OrderedDict()
    for domain, dials in dataset.dialogs.items():
        if test_domain_ids is not None and domain not in test_domain_ids:
            continue
        slots = dataset.domain2slot[domain]

        num_dialog_per_batch = args.pred_batch_size
        assert num_dialog_per_batch > 0

        # prepare batches
        if args.small_sample_run:
            dials = dials[:10]

        for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=domain):
            test_batch = []
            batch_dials = dials[s: s + num_dialog_per_batch]
            for dial in batch_dials:
                test_batch.append(dataset.prepare_for_generation(dial, domain))
            batch = dataset.collate_fn(test_batch)
            batch_ds = [_['ds'] for _ in test_batch]
            if args.model_type == 't5':
                predict_t5(model, tokenizer, batch, batch_ds, batch_dials)
            elif args.model_type == 'gpt2':
                predict_gpt2(model, tokenizer, batch, batch_ds, batch_dials)

        # compute metric
        slot_acc = {ds: 0 for ds in slots}
        joint_total = 0
        joint_acc = 0
        for dial in dials:
            joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
            for ds in slots:
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        joint_accuracy = joint_acc / joint_total
        slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
        avg_acc = sum(slot_acc.values()) / len(slot_acc)
        pred_metrics[domain] = joint_accuracy
        print('{} acc: {}'.format(domain, joint_accuracy))

    pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
    print('===================metrics===========')
    pprint(pred_metrics)
    json.dump(dataset.dialogs, open(os.path.join(args.output_dir, 'predict_results.json'), 'w'), indent=4)
    print('saving predictions to {}'.format(os.path.join(args.output_dir, 'predict_results.json')))

    return pred_metrics


def prompt5_predict_fwt(model, dataset:MemT5PromptDSTDataset, tokenizer, args, dataset_order):
    model.to(DEVICE)
    pred_metrics = OrderedDict()

    for domain_idx, domain in enumerate(dataset_order):
        if domain_idx == 0:
            continue
        dials = dataset.dialogs[domain]
        slots = dataset.domain2slot[domain]

        num_dialog_per_batch = args.pred_batch_size
        assert num_dialog_per_batch > 0

        # prepare batches
        if args.small_sample_run:
            dials = dials[:10]

        for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=domain):
            test_batch = []
            batch_dials = dials[s: s + num_dialog_per_batch]
            for dial in batch_dials:
                # test_batch.append(dataset.prepare_for_generation(dial, domain))
                test_batch.append(dataset.convert_dial_to_example(dial, domain, dataset.domain2soft_prompt(dataset_order[domain_idx-1])))

            batch = dataset.collate_fn(test_batch)
            batch_ds = [_['ds'] for _ in test_batch]
            if args.model_type == 't5':
                predict_t5(model, tokenizer, batch, batch_ds, batch_dials)
            elif args.model_type == 'gpt2':
                predict_gpt2(model, tokenizer, batch, batch_ds, batch_dials)

        slot_acc = {ds: 0 for ds in slots}
        joint_total = 0
        joint_acc = 0
        for dial in dials:
            joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
            for ds in slots:
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        joint_accuracy = joint_acc / joint_total
        slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
        avg_acc = sum(slot_acc.values()) / len(slot_acc)
        pred_metrics[domain] = joint_accuracy
        print('{} acc: {}'.format(domain, joint_accuracy))

    pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
    print('===================metrics===========')
    pprint(pred_metrics)
    json.dump(dataset.dialogs, open(os.path.join(args.output_dir, 'predict_results.json'), 'w'), indent=4)
    print('saving predictions to {}'.format(os.path.join(args.output_dir, 'predict_results.json')))

    return pred_metrics



def prompt5_predict_select_fwt(model, dataset:MemT5PromptDSTDataset, tokenizer, args, target_domain_to_init_domain):
    # calculate FWT for selectinit
    model.to(DEVICE)
    pred_metrics = OrderedDict()

    for dial_domain, init_domain in target_domain_to_init_domain.items():

        dials = dataset.dialogs[dial_domain]
        slots = dataset.domain2slot[dial_domain]

        num_dialog_per_batch = args.pred_batch_size
        assert num_dialog_per_batch > 0

        # prepare batches
        if args.small_sample_run:
            dials = dials[:10]

        for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=dial_domain):
            test_batch = []
            batch_dials = dials[s: s + num_dialog_per_batch]
            for dial in batch_dials:
                # test_batch.append(dataset.prepare_for_generation(dial, domain))
                test_batch.append(dataset.convert_dial_to_example(dial, dial_domain, dataset.domain2soft_prompt(init_domain)))

            batch = dataset.collate_fn(test_batch)
            batch_ds = [_['ds'] for _ in test_batch]
            if args.model_type == 't5':
                predict_t5(model, tokenizer, batch, batch_ds, batch_dials)
            elif args.model_type == 'gpt2':
                predict_gpt2(model, tokenizer, batch, batch_ds, batch_dials)

        slot_acc = {ds: 0 for ds in slots}
        joint_total = 0
        joint_acc = 0
        for dial in dials:
            joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
            for ds in slots:
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        joint_accuracy = joint_acc / joint_total
        slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
        avg_acc = sum(slot_acc.values()) / len(slot_acc)
        pred_metrics[dial_domain] = joint_accuracy
        print('{} acc: {}'.format(dial_domain, joint_accuracy))

    pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
    print('===================metrics===========')
    pprint(pred_metrics)
    json.dump(dataset.dialogs, open(os.path.join(args.output_dir, 'predict_results.json'), 'w'), indent=4)
    print('saving predictions to {}'.format(os.path.join(args.output_dir, 'predict_results.json')))

    return pred_metrics



def parse_fullstate_str(text):
    state = {}
    while '(' in text and ')' in text:
        startpos = text.index('(')
        text = text[startpos+1:]
        endpos = text.index(')')
        service_str = text[:endpos]
        text = text[endpos+1:]

        parameters = sum([s.split('",') for s in service_str.split("=")],[])
        parameters = [_.strip() for _ in parameters]
        if len(parameters)>1:
            if len(parameters) % 2 != 0:
                parameters = parameters[:-1]

            for i in range(0,len(parameters),2):
                state[parameters[i]] = parameters[i+1].replace('"',"")
    return state


def predict_batch_fullstate(model, tokenizer, batch, batch_dials):
    dst_input_ids = batch['input_ids_womask'].to(DEVICE)
    dst_input_mask = batch['attention_mask_womask'].to(DEVICE)

    outputs = model.generate(
        input_ids=dst_input_ids,
        attention_mask=dst_input_mask,
        use_cache=False,
        return_dict_in_generate=True,
        max_length=100,
    )

    dst_predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=True)
    dst_predictions = [re.sub('\<(prompt_)(.*?)\>', '', s) for s in dst_predictions]
    dst_predictions = [_.strip() for _ in dst_predictions]
    for i, pred_str in enumerate(dst_predictions):
        batch_dials[i]['pred_state'] = parse_fullstate_str(pred_str)
        batch_dials[i]['pred_str'] =  pred_str
        batch_dials[i]['state'] = parse_fullstate_str(batch_dials[i]['reply'])


def prompt5_predict_fullstate(model, dataset:MemT5PromptDSTDataset, tokenizer, args, test_domain_ids=None):
    model.to(DEVICE)
    pred_metrics = OrderedDict()
    for domain, dials in dataset.dialogs.items():
        if test_domain_ids is not None and domain not in test_domain_ids:
            continue
        slots = dataset.domain2slot[domain]

        num_dialog_per_batch = args.pred_batch_size
        assert num_dialog_per_batch > 0

        # prepare batches
        if args.small_sample_run:
            dials = dials[:10]

        for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=domain):
            test_batch = []
            batch_dials = dials[s: s + num_dialog_per_batch]
            for dial in batch_dials:
                test_batch.append(dataset.prepare_for_generation(dial, domain))
            batch = dataset.collate_fn(test_batch)
            predict_batch_fullstate(model, tokenizer, batch, batch_dials)

        slot_acc = {ds: 0 for ds in slots}
        joint_total = 0
        joint_acc = 0
        for dial in dials:
            joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
            for ds in slots:
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        joint_accuracy = joint_acc / joint_total
        slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
        avg_acc = sum(slot_acc.values()) / len(slot_acc)
        pred_metrics[domain] = joint_accuracy
        print('{} acc: {}'.format(domain, joint_accuracy))

    pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
    print('===================metrics===========')
    pprint(pred_metrics)
    json.dump(dataset.dialogs, open(os.path.join(args.output_dir, 'predict_results.json'), 'w'), indent=4)
    print('saving predictions to {}'.format(os.path.join(args.output_dir, 'predict_results.json')))

    return pred_metrics


def prompt5_predict_fullstate_fwt(model, dataset:MemT5PromptDSTDataset, tokenizer, args, dataset_order):
    model.to(DEVICE)
    pred_metrics = OrderedDict()
    for domain_idx, domain in enumerate(dataset_order):
        if domain_idx == 0:
            continue
        dials = dataset.dialogs[domain]
        slots = dataset.domain2slot[domain]

        num_dialog_per_batch = args.pred_batch_size
        assert num_dialog_per_batch > 0

        # prepare batches
        if args.small_sample_run:
            dials = dials[:10]

        for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=domain):
            test_batch = []
            batch_dials = dials[s: s + num_dialog_per_batch]
            for dial in batch_dials:
                # test_batch.append(dataset.prepare_for_generation(dial, domain))
                test_batch.append(dataset.convert_dial_to_example(dial, domain, dataset.domain2soft_prompt(dataset_order[domain_idx-1])))

            batch = dataset.collate_fn(test_batch)
            predict_batch_fullstate(model, tokenizer, batch, batch_dials)

        slot_acc = {ds: 0 for ds in slots}
        joint_total = 0
        joint_acc = 0
        for dial in dials:
            joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
            for ds in slots:
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        joint_accuracy = joint_acc / joint_total
        slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
        avg_acc = sum(slot_acc.values()) / len(slot_acc)
        pred_metrics[domain] = joint_accuracy
        print('{} acc: {}'.format(domain, joint_accuracy))

    pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
    print('===================metrics===========')
    pprint(pred_metrics)
    json.dump(dataset.dialogs, open(os.path.join(args.output_dir, 'predict_results.json'), 'w'), indent=4)
    print('saving predictions to {}'.format(os.path.join(args.output_dir, 'predict_results.json')))

    return pred_metrics



# def evaluate_checkpoints(args):
#     if args.checkpoint is not None:
#         checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoints', args.checkpoint))
#         print(os.path.join(args.output_dir, 'checkpoints', args.checkpoint))
#     else:
#         all_checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoints', "*.ckpt"), recursive=True)
#         step_checkpoints = [c for c in all_checkpoints if ('step_count=' in c)]
#         step_ckpt_dict = {int(c.split('step_count=')[1].split('.ckpt')[0]): c for c in step_checkpoints}
#         steps = list(sorted(list(step_ckpt_dict.keys())))[-5:]
#         checkpoints = [step_ckpt_dict[k] for k in steps]
#
#         # all_checkpoints = list(sorted(all_checkpoints))
#         # checkpoints += all_checkpoints[:2]
#
#     checkpoints = list(set(checkpoints))
#     if args.small_sample_run:
#         checkpoints = checkpoints[-1:]
#     print('checkpoints to generate: {}'.format(checkpoints))
#
#     if args.generate_path is None:
#         args.generate_path = args.output_dir
#
#     args.generate_path = os.path.join(args.generate_path, 'generation_prompt_dst')
#
#     metric_file_name = 'metrics.json'
#     if args.small_sample_run:
#         metric_file_name = 'smallsample_' + metric_file_name
#     elif args.medium_sample_run:
#         metric_file_name = 'mediumsample_' + metric_file_name
#
#     metric_file = os.path.join(args.generate_path, args.dataset_split, metric_file_name)
#     print('writing metrics to {}'.format(metric_file))
#     if os.path.exists(metric_file):
#         inp = input('metrics.json exists, overwrite?')
#         if inp == 'y' or inp == 'Y':
#             os.remove(metric_file)
#
#     args.generate_path = os.path.join(args.generate_path, args.dataset_split)
#
#     val_checkpoints_metric = {}
#     for checkpoint in checkpoints:
#         # load dataset and model
#         print('====================checkpoint: {}==================='.format(checkpoint.split('/')[-1]))
#         module = CLPromptDSTModule.load_from_checkpoint(checkpoint)
#         tokenizer = module.tokenizer
#         module.freeze()
#         module.to(DEVICE)
#         module.eval()
#         dataset = module.get_dataset('val')
#         model = module.model
#         pred_metrics = predict(model, dataset, tokenizer, args)
#         val_checkpoints_metric[checkpoint] = pred_metrics['avg_joint_acc']
#
#     checkpoint_metrics = list(sorted(val_checkpoints_metric.items(), key=lambda x: x[1]))
#     test_ckpt, metric = checkpoint_metrics[-1]
#     print('====================best checkpoint on val: {} ========================\n'
#           '==================joint acc {}==================='.format(test_ckpt.split('/')[-1], metric))
#     module = CLPromptDSTModule.load_from_checkpoint(test_ckpt)
#     tokenizer = module.tokenizer
#     module.freeze()
#     module.to(DEVICE)
#     module.eval()
#     dataset = module.dataset_class(
#         tokenizer=tokenizer,
#         data_dir=module.hparams.data_dir,
#         type_path='test',
#         small_sample_run=args.small_sample_run,
#         prompt_style=module.hparams.prompt_style,
#         test_file_path=args.test_file_path
#     )
#     model = module.model
#     pred_metrics = predict(model, dataset, tokenizer, args)
#     print('=============final joint acc on test===================')
#     pprint(pred_metrics)


# def predict_todcl(args):
#     if args.checkpoint is not None:
#         checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoints', args.checkpoint))
#         print(os.path.join(args.output_dir, 'checkpoints', args.checkpoint))
#     else:
#         all_checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoints', "*.ckpt"), recursive=True)
#         step_checkpoints = [c for c in all_checkpoints if ('step_count=' in c)]
#         step_ckpt_dict = {int(c.split('step_count=')[1].split('.ckpt')[0]): c for c in step_checkpoints}
#         steps = list(sorted(list(step_ckpt_dict.keys())))[-5:]
#         checkpoints = [step_ckpt_dict[k] for k in steps]
#
#         # all_checkpoints = list(sorted(all_checkpoints))
#         # checkpoints += all_checkpoints[:2]
#
#     checkpoints = list(set(checkpoints))
#     if args.small_sample_run:
#         checkpoints = checkpoints[-1:]
#     print('checkpoints to generate: {}'.format(checkpoints))
#
#     if args.generate_path is None:
#         args.generate_path = args.output_dir
#
#     args.generate_path = os.path.join(args.generate_path, 'generation_prompt_dst')
#
#     metric_file_name = 'metrics.json'
#     if args.small_sample_run:
#         metric_file_name = 'smallsample_' + metric_file_name
#     elif args.medium_sample_run:
#         metric_file_name = 'mediumsample_' + metric_file_name
#
#     metric_file = os.path.join(args.generate_path, args.dataset_split, metric_file_name)
#     print('writing metrics to {}'.format(metric_file))
#     if os.path.exists(metric_file):
#         inp = input('metrics.json exists, overwrite?')
#         if inp == 'y' or inp == 'Y':
#             os.remove(metric_file)
#
#     args.generate_path = os.path.join(args.generate_path, args.dataset_split)
#
#     val_checkpoints_metric = {}
#     for checkpoint in checkpoints:
#         # load dataset and model
#         print('====================checkpoint: {}==================='.format(checkpoint.split('/')[-1]))
#         model = PromptDSTModule.load_from_checkpoint(checkpoint)
#         tokenizer = model.tokenizer
#         model.freeze()
#         model.to(DEVICE)
#         model.eval()
#         dataset = T5DomainPromptDSTDatasetTodcl(
#             tokenizer=tokenizer,
#             data_dir=model.hparams.data_dir,
#             type_path='test',
#             small_sample_run=args.small_sample_run,
#             prompt_style=model.hparams.prompt_style,
#             test_file_path=args.test_file_path,
#         )
#         model = model.model
#
#         # prepare prediction path
#         os.makedirs(os.path.join(args.generate_path, checkpoint.split('/')[-1]), exist_ok=True)
#         pred_file = os.path.join(args.generate_path, checkpoint.split('/')[-1], 'predictions.json')
#         if args.small_sample_run:
#             pred_file = os.path.join(args.generate_path, checkpoint.split('/')[-1], 'small_sample_predictions.json')
#         elif args.medium_sample_run:
#             pred_file = os.path.join(args.generate_path, checkpoint.split('/')[-1], 'medium_sample_predictions.json')
#
#         pred_metrics = {}
#         for cl_domain_id, dials in dataset.dialogs.items():
#             cl_domain = tuple(eval(cl_domain_id))
#             dss = dataset.cl_domain_to_ds[cl_domain]
#
#             num_dialog_per_batch = args.pred_batch_size
#             assert num_dialog_per_batch > 0
#
#             # prepare batches
#             if args.small_sample_run:
#                 dials = dials[:10]
#
#             for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=str(cl_domain)):
#                 test_batch = []
#                 batch_dials = dials[s: s+num_dialog_per_batch]
#                 for dial in batch_dials:
#                     test_batch.append(dataset.prepare_for_generation(dial, cl_domain))
#                 batch = dataset.collate_fn(test_batch)
#                 batch_ds = [_['ds'] for _ in test_batch]
#
#                 dst_input_ids = batch['input_ids_womask'].to(DEVICE)
#                 dst_input_mask = batch['attention_mask_womask'].to(DEVICE)
#                 dst_decoder_input_ids = batch.get('decoder_inputs_womask', None)
#
#                 if dst_decoder_input_ids is not None:
#                     dst_decoder_input_ids = dst_decoder_input_ids.to(DEVICE)
#
#                 if dst_decoder_input_ids is not None:
#                     outputs = model.generate(
#                         input_ids=dst_input_ids,
#                         attention_mask=dst_input_mask,
#                         decoder_input_ids=dst_decoder_input_ids,
#                         use_cache=False,
#                         return_dict_in_generate=True,
#                         max_length=80,
#                     )
#                 else:
#                     outputs = model.generate(
#                         input_ids=dst_input_ids,
#                         attention_mask=dst_input_mask,
#                         use_cache=False,
#                         return_dict_in_generate=True,
#                         max_length=80,
#                     )
#
#                 dst_predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False,
#                                                          clean_up_tokenization_spaces=True)
#                 dst_predictions = [re.sub('\<(prompt_)(.*?)\>', '', s) for s in dst_predictions]
#                 dst_predictions = [_.strip() for _ in dst_predictions]
#                 if args.small_sample_run:
#                     for _ds, inp, p in zip(batch_ds, batch['input_seqs_womask'], dst_predictions):
#                         print('===========================')
#                         print(_ds)
#                         print(inp)
#                         print(p)
#                 dst_predictions = [_ if _ != 'don\'t care' else 'dontcare' for  _ in dst_predictions]
#
#                 for i, (pred_str, ds_str) in enumerate(zip(dst_predictions, batch_ds)):
#                     ds_list = ds_str.split()
#                     assert 'pred_state' not in batch_dials[i]
#                     batch_dials[i]['pred_state'] = {}
#                     for j, ds in enumerate(ds_list):
#                         left_token = '<extra_id_{}>'.format(j)
#                         right_token = '<extra_id_{}>'.format(j+1)
#                         if right_token not in pred_str:
#                             right_token = '</s>'
#                         try:
#                             value = pred_str.split(left_token)[1].split(right_token)[0].strip()
#                         except:
#                             # print(pred_str)
#                             # print(left_token)
#                             # print(right_token)
#                             # input()
#                             value = 'none'
#                         if value == 'none' or value == '':
#                             continue
#                         batch_dials[i]['pred_state'][ds] = value
#                     # print(pred_str)
#                     # print(batch_dials[i]['pred_state'])
#                     # input()
#
#             slot_acc = {ds: 0 for ds in dss}
#             joint_total = 0
#             joint_acc = 0
#             for dial in dials:
#                 joint_total += 1
#                 if set(dial['state'].items()) == set(dial['pred_state'].items()):
#                     joint_acc += 1
#                 for ds in dss:
#                     golden = dial['state'].get(ds, 'none')
#                     pred = dial['pred_state'].get(ds, 'none')
#                     if golden == pred:
#                         slot_acc[ds] += 1
#             joint_accuracy = joint_acc / joint_total
#             slot_acc = {ds: acc / joint_total for ds, acc in slot_acc.items()}
#             avg_acc = sum(slot_acc.values()) / len(slot_acc)
#             pred_metrics[cl_domain] = joint_accuracy
#
#         val_checkpoints_metric[checkpoint] = sum(pred_metrics.values()) / len(pred_metrics)
#
#         pred_metrics['avg_joint_acc'] = sum(pred_metrics.values()) / len(pred_metrics)
#         print('===================metrics===========')
#         pprint(pred_metrics)
#         # json.dump({
#         #     'checkpoint': checkpoint,
#         #     'joint_slot_acc': {str(k): v for k, v in pred_metrics.items()},
#         #     # 'slot_acc': slot_acc,
#         #     # 'avg_acc': avg_acc,
#         # }, open(metric_file, 'a+'), indent=4)
#         # pred_metrics['prediction_path'] = pred_file
#         # json.dump(dataset.dialogs, open(pred_file, 'w'), indent=4)
#         # print('writing predictions to {}'.format(pred_file))




# if __name__ == "__main__":
#     parser = prepare_prompt_parser()
#     parser.add_argument('--dataset_split', type=str)
#     parser.add_argument('--pred_batch_size', type=int)
#     parser.add_argument('--checkpoint', type=str, default=None)
#     parser.add_argument('--generate_path', type=str, default=None)
#     parser.add_argument('--medium_sample_run', action='store_true')
#     parser.add_argument('--test_file_path', type=str)
#
#     args = parser.parse_args()
#     print(args)
#
#     # predict_todcl(args)
#     evaluate_checkpoints(args)