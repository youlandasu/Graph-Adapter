import json
import os
from tabulate import tabulate
import collections
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Select from sgd or mwoz.")

args = parser.parse_args()

pred_dir = "output_temp/MULTI_PROMPT/prompt100_order1_vocab_sample_mem50_lr0.5_epoch20_firstlr0.5_firstEpoch20_metalr0.5_metaEpoch10_BZ8_ACC2_seed2/"
pred_dir = "output_temp/MWOZ_MULTI_PROMPT/Log_softmax_order1_vocab_sample_mem50_lr0.5_epoch20_firstlr0.5_firstEpoch20_metalr0.5_metaEpoch10_BZ8_ACC2_seed1/"
pred_dir = "output_temp/MULTI_PROMPT/log_softmax_graphlr1E-2_graphwd5E-4_order32_vocab_sample_mem50_lr0.5_epoch20_BZ8_ACC2_seed1"
with open(os.path.join(pred_dir,"predict_results.json")) as f:
    pred_dials = json.load(f)

if args.dataset == 'sgd':
    schema = json.load(
                open('data/dstc8-schema-guided-dialogue/train/schema.json')) + json.load(
                open('data/dstc8-schema-guided-dialogue/dev/schema.json')) + json.load(
                open('data/dstc8-schema-guided-dialogue/test/schema.json'))
    slots = []
    for service in schema:
        service_name = service['service_name'].lower() #'weather_1'
        for task in pred_dials: #"['sgd_services_4']":
            task = "sgd_" + task.split('\'')[1]
            if service_name in task:
                slot_names = [x["name"] for x in service['slots']]
                for result_slot in slot_names:
                    slot_name = 'sgd_{}-{}'.format(service_name, result_slot)
                    slots.append(slot_name) 
    res = []
    [res.append(x) for x in slots if x not in res]
    slots = res # 'sgd_trains_1-from_station'

    domain2slot = {} # 'trains_1': ['sgd_trains_1-from', 'sgd_trains_1-to', 'sgd_trains_1-from_station'...]
    for service in schema:
        service_name = service['service_name'].lower()#'weather_1'
        task_name = '[\'' + 'sgd_{}'.format(service_name) + '\']'
        domain2slot[task_name] = []
        for slot in slots:
            if service_name in slot:
                domain2slot[task_name].append(slot)

elif args.dataset == 'mwoz':
    slots = []
    schema = json.load(open('data/multiwoz/data/MultiWOZ_2.2/schema.json'))
    for service in schema:
        service_name = service['service_name'].lower()
        for task in pred_dials: #"['sgd_services_4']":
            task = "MWOZ_" + task.split('\'')[1]
            if service_name in task:
                slot_names = [x["name"] for x in service['slots']]
                for result_slot in slot_names:
                    slot_name = 'MWOZ_{}-{}'.format(service_name, result_slot)
                    slots.append(slot_name) 
    res = []
    [res.append(x) for x in slots if x not in res]
    slots = res # 'sgd_trains_1-from_station'

    domain2slot = {} # 'trains_1': ['sgd_trains_1-from', 'sgd_trains_1-to', 'sgd_trains_1-from_station'...]
    for service in schema:
        service_name = service['service_name'].lower()#'weather_1'
        task_name = '[\'' + 'MWOZ_{}'.format(service_name) + '\']'
        domain2slot[task_name] = []
        for slot in slots:
            if service_name in slot:
                domain2slot[task_name].append(slot)

#print(domain2slot)
domain_combined_metrics = {}
joint_total = 0
joint_acc = 0
pred_metrics = {}
for task in pred_dials:
    domain_joint_acc = 0
    domain_joint_total = 0
    for dial in pred_dials[task]:
        joint_total += 1
        domain_joint_total += 1
        if set(dial['state'].items()) == set(dial['pred_state'].items()):
                joint_acc += 1
                domain_joint_acc += 1
    domain_joint = domain_joint_acc / domain_joint_total
    pred_metrics[task] = domain_joint
    print("{} joint accuracy:".format(task), domain_joint)  

joint_accuracy = joint_acc / joint_total
print("joint accuracy:", joint_accuracy)

avg_joint = sum(pred_metrics.values()) / len(pred_metrics)
print("avg joint accuracy:", avg_joint)

print('='*40)

slot_total = 0
correct_slot = 0
slot_metrics = {}
for task in pred_dials:
    domain_slots = domain2slot[task]
    slot_acc = {ds: 0 for ds in domain_slots}
    domain_slot_total = 0
    for dial in pred_dials[task]:
        for ds in domain_slots:
            domain_slot_total += 1
            slot_total += 1
            golden = dial['state'].get(ds, 'none')
            pred = dial['pred_state'].get(ds, 'none')
            if golden == pred:
                slot_acc[ds] += 1
                correct_slot += 1
    domain_slot_accuracy = sum([slot_acc[ds] for ds in domain_slots]) / domain_slot_total
    slot_metrics[task] = domain_slot_accuracy
    print("{} slot accuracy:".format(task), domain_slot_accuracy)  

slot_accuracy = correct_slot / slot_total
print("slot accuracy:", slot_accuracy)

avg_slot = sum(slot_metrics.values()) / len(slot_metrics)
print("avg slot accuracy:", avg_slot)

slot_total = 0
correct_slot = 0
slot_metrics = {}
for task in pred_dials:
    domain_slots = domain2slot[task]
    slot_acc = {ds: 0 for ds in domain_slots}
    domain_slot_total = 0
    for dial in pred_dials[task]:
        for ds in dial['state']:
            domain_slot_total += 1
            slot_total += 1
            golden = dial['state'][ds]
            pred = dial['pred_state'].get(ds, 'none')
            if golden == pred:
                slot_acc[ds] += 1
                correct_slot += 1
    domain_slot_accuracy = sum([slot_acc[ds] for ds in domain_slots]) / domain_slot_total
    slot_metrics[task] = domain_slot_accuracy
    print("{} average goal accuracy:".format(task), domain_slot_accuracy)  

if args.dataset == 'sgd':
    print('='*40)


    domain_combined_metrics = {}
    for task in pred_dials:
        domain_joint_acc = 0
        domain_joint_total = 0
        domain_name = task.split('_')[1]
        if  domain_name not in domain_combined_metrics:
            domain_combined_metrics[domain_name] = [0,0]
        for dial in pred_dials[task]:
            domain_joint_total += 1
            if set(dial['state'].items()) == set(dial['pred_state'].items()):
                    domain_joint_acc += 1
        
        domain_combined_metrics[domain_name][0] += domain_joint_acc
        domain_combined_metrics[domain_name][1] += domain_joint_total

    for key, value in domain_combined_metrics.items():  
        print("{} joint accuracy:".format(key), value[0]/value[1])  

    print('='*40)

    domain_combined_slot_metrics = {}
    for task in pred_dials:
        domain_slots = domain2slot[task]
        slot_acc = {ds: 0 for ds in domain_slots}
        domain_slot_total = 0
        domain_name = task.split('_')[1]
        if domain_name not in domain_combined_slot_metrics.keys():
            domain_combined_slot_metrics[domain_name] = [0,0]
        for dial in pred_dials[task]:
            for ds in domain_slots:
                domain_slot_total += 1
                slot_total += 1
                golden = dial['state'].get(ds, 'none')
                pred = dial['pred_state'].get(ds, 'none')
                if golden == pred:
                    slot_acc[ds] += 1
        domain_slot_accuracy = sum([slot_acc[ds] for ds in domain_slots]) / domain_slot_total
        domain_combined_slot_metrics[domain_name][0] += sum([slot_acc[ds] for ds in domain_slots])
        domain_combined_slot_metrics[domain_name][1] += domain_slot_total

    for key, value in domain_combined_slot_metrics.items():  
        print("{} slot accuracy:".format(key), value[0]/value[1])  