import re
from typing import List
import json
import random

from prompt_files.p_tuning.todcl.todcl_domain_dataset import t5_shift_tokens_right, Sampler, SubsetRandomSampler, SequentialSampler, BatchSampler, Dataset, RandomSampler
from prompt_files.prompts_config import PROMPT_TOKENS, META_PROMPT_TOKENS, GRAPH_PROMPT_TOKENS, UNUSED_TOKENS
from torch.utils.data import Dataset, Sampler, BatchSampler, Dataset, RandomSampler
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils.loop import add_self_loops
import numpy as np
from prompt_files.transformer_utils import logging
logger = logging.get_logger(__name__)
# logger.setLevel(logging.INFO)
np.random.seed(42)
random.seed(42)

class RangeIndexSampler(Sampler):
    def __init__(self, start, end):
        self.indexes = list(range(start, end))
        np.random.shuffle(self.indexes)

    def __iter__(self):
        yield from self.indexes

    def __len__(self):
        return len(self.indexes)


class RangeIndexSeqSampler(Sampler):
    def __init__(self, start, end, max_sample=None):
        if max_sample is None:
            self.indexes = list(range(start, end))
        else:
            self.indexes = list(range(start, end, max(1, (end-start)//max_sample)))

    def __iter__(self):
        yield from self.indexes

    def __len__(self):
        return len(self.indexes)


class MemT5PromptDSTDataset(Dataset):
    """
    A unified dataset which support:
    - Multi-task:
        set training_prompt_name='meta' and resample for training,
        set prepare_for_generation(domain='meta') for testing.
        make_full_data_sampler
    - RandInit:
        directly use
        make_domain_sampler
    - CLInit & SelectInit:
        set training_prompt_name='meta_prompt'/'prompt', resample dataset for training.
        copy prompt, resample every epoch if perm_desc
        make_domain_sampler
    - Memory replay:
        set training_prompt_name='meta_prompt'/'prompt', resample dataset for training.
        prepare forward_domains, sample replay_memory
        make_domain_sampler
    - Backward transfer:
        set training_prompt_name='meta_prompt'/'prompt', resample dataset for training.
        prepare backward_domain, sample replay_memory
        make_domain_sampler for current domain data + make_domain_sampler(domain='memory') for backward domain memory
    """
    def __init__(self,
                 tokenizer,
                 type_path,
                 dialogs,
                 domain2slot,
                 num_domain_prompt=100,
                 num_graph_prompt=0, # graph_prompt
                 small_sample_run=False,
                 permute_desc=False,
                 multitask=False,
                 dataset_order=None,
                 graph_data_config=None,
                 ):
        self.permute_desc = permute_desc
        self.dialogs = dialogs
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.small_sample_run = small_sample_run
        self.type_path = type_path
        self.multitask = multitask
        self.dataset_order = dataset_order

        # prepare prompt template
        self.num_domain_prompt = num_domain_prompt
        self.num_graph_prompt = num_graph_prompt # graph_prompt

        if graph_data_config: # prepare graph batch
            self.undirected = graph_data_config["undirected"]
            self.self_loops = graph_data_config["self_loops"]

        self.slot2description = {}
        schema = json.load(
            open('data/dstc8-schema-guided-dialogue/train/schema.json')) + json.load(
            open('data/dstc8-schema-guided-dialogue/dev/schema.json')) + json.load(
            open('data/dstc8-schema-guided-dialogue/test/schema.json'))
        #extra_id_num = 0
        for service in schema:
            service_name = service['service_name'].lower()
            slots = service['slots']
            for slot in slots:
                slot_name = 'sgd_{}-{}'.format(service_name, slot['name'])
                desc = slot['description'] + ': <extra_id_0> . '
                #desc = slot['description'] + ': <extra_id_{}> . '.format(extra_id_num)
                self.slot2description[slot_name] = desc
                #extra_id_num += 1

        self.domain2slot = domain2slot
        self.domains = list(sorted(list(self.domain2slot.keys())))

        self.all_slots = set()
        if self.num_graph_prompt != 0:
            for domain in self.domains:
                if self.dataset_order and domain not in self.dataset_order:
                    continue
                self.all_slots.update(self.domain2slot[domain])
            self.slot_connect = [[0 for i in range(len(self.all_slots))] for j in range(len(self.all_slots))] 
            for k, domain_slotk in enumerate(self.all_slots):
                kdomain = domain_slotk.split('-')[0]
                for l, domain_slotl in enumerate(self.all_slots):
                    ldomain = domain_slotl.split('-')[0]
                    if kdomain == ldomain:
                        self.slot_connect[k][l] = 1
                        self.slot_connect[l][k] = 1
        graph_prompt = self.domain2graph_prompt(self.all_slots) if self.num_graph_prompt != 0 else '' #<graph_prompt_i>
        #ds_sequence = self.domain2slot_token(self.all_slots) if self.num_graph_prompt != 0 else ''
        assert graph_prompt != ''


        self.replay_memory = {d: [] for d in self.domains}  # dials
        # prepare examples
        if type_path in ['train', 'val']:
            self.domain2samples = {k: [] for k in self.domains}
            self.domain2samples['memory'] = []
            for domain, dials in self.dialogs.items():
                assert domain in self.domains
                soft_prompt = self.domain2soft_prompt(domain)
                #graph_prompt = self.domain2graph_prompt(domain) # graph_prompt
                for dial in dials:
                    self.domain2samples[domain].append(self.convert_dial_to_example(dial, domain, soft_prompt, graph_prompt)) # graph_prompt
            if small_sample_run:
                self.domain2samples = {k: v[:10] for k, v in self.domain2samples.items()}

            self.domain2numsamples = {k: len(self.domain2samples[k]) for k in self.domain2samples.keys()}
            self.dataset_len = sum(self.domain2numsamples.values())
            print('domain2numsamples', self.domain2numsamples)
            print('total', self.dataset_len)

        self.aug_metatrain_data = {d: [] for d in self.domains}  # dials
        #self.random_generator = random.Random(52)

    def convert_dial_to_example(self, dial, domain, soft_prompt, graph_prompt='', do_augment=False, extra_aug_slots=[]):# graph_prompt
        # extra_aug_domains: use slots from these domains to query dial, slot values should be 'none'
        slots = self.domain2slot[domain]
        domain_enc_prompt = ''
        target_seq = ''
        extra_id_num = 0
        if do_augment:
            assert self.type_path == 'train'
            # num_slots = self.random_generator.randint(len(slots)//2, len(slots))
            num_slots = random.randint(1, len(slots))
            slots = random.sample(slots, num_slots)
            if len(extra_aug_slots) > 0:
                # num_extra_slots = self.random_generator.randint(1, min(len(slots)//2, len(extra_aug_slots)))
                num_extra_slots = random.randint(1, min(len(slots), len(extra_aug_slots)))
                extra_slots = random.sample(extra_aug_slots, num_extra_slots)
                slots += extra_slots
        if self.type_path == 'train' and do_augment:
            slots = np.random.permutation(slots)

        for i, slot in enumerate(slots):
            enc_p = self.slot2description[slot]
            domain_enc_prompt += enc_p.replace('<extra_id_0>', '<extra_id_{}>'.format(extra_id_num))
            #domain_enc_prompt += self.slot2description[slot]
            #extra_id_num = self.slot2description[slot].split(':')[1].replace('.','').strip()
            value = dial['state'].get(slot, 'none')
            #target_seq += extra_id_num + value
            target_seq += '<extra_id_{}>{}'.format(extra_id_num, value)
            extra_id_num += 1

        # add active slots property
        active_slots = [] # active slots in a turn: 1 * len(self.all_slots)
        for slot in self.all_slots:
            if slot in dial['state'].keys():
                active_slots.append(1)
            else:
                active_slots.append(0)

        input_dict = {
            'dst_input_sequence': (dial['history'] + ' </s> ' + domain_enc_prompt + graph_prompt + soft_prompt).lower(), ## graph_prompt
            'dst_target_sequence': target_seq.lower(),
            'ds': ' '.join(slots),
            'dial_id': dial['dataset'] + '_' + dial['dial_id'] + '_' + str(dial['turn_id']),
            'dst_generation_decoder_inputs': '',
            'slot_connect': self.slot_connect,
            'active_slots': active_slots,
        }
        return input_dict

    def collate_fn(self, batch):
        dst_input_seqs = [x['dst_input_sequence'] for x in batch]
        dst_ds = [x['ds'] for x in batch]
        slot_connect = [x['slot_connect']for x in batch]
        active_slots = [x['active_slots']for x in batch]
        #dst_ds_seqs = [x['ds_sequence']for x in batch]
        #dst_ds_dict = self.tokenizer(dst_ds_seqs,
        #                                max_length=self.num_graph_prompt, #same as prompts_config
        #                                padding=True,
        #                                truncation=True,
        #                                return_tensors='pt')
        #dst_ds_ids = dst_ds_dict['input_ids']
        #dst_ds_mask = dst_ds_dict['attention_mask']

        dst_input_dict = self.tokenizer(dst_input_seqs,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']

        ''' #CUDA OUT OF MEMORY
        pad_ds_ids = self.pad_with_mask(dst_input_ids) #numpy
        graph_list = []
        for i in range(len(batch)):
            graph_list += self.convert_ontology_to_graph(pad_ds_ids[i], slot_connect[i], active_slots[i])
        batch_graph = Batch.from_data_list(graph_list).to(dst_input_ids.device)
        '''
        dst_ds_ids = torch.clone(dst_input_ids).to(dst_input_ids.device)
        graph_prompt_ids = dst_input_ids - self.tokenizer.vocab_size - len(PROMPT_TOKENS) - len(UNUSED_TOKENS)
        dst_ds_ids = dst_ds_ids[(graph_prompt_ids < self.num_graph_prompt)&(graph_prompt_ids>=0)].view(dst_input_ids.size(0), -1)
        graph_list = []
        for i in range(len(batch)):
            graph_list += self.convert_ontology_to_graph(dst_ds_ids[i], slot_connect[i], active_slots[i])
        batch_graph = Batch.from_data_list(graph_list).to(dst_input_ids.device) # 103 * batch

        input_batch = {
            "input_ids_womask": dst_input_ids,
            "attention_mask_womask": dst_input_mask,
            'input_seqs_womask': dst_input_seqs,
            'ds': dst_ds,
            'batch_graph': batch_graph,
            #'slot_connect': slot_connect,
            #'ds_ids': pad_ds_ids,
            #'ds_mask': dst_ds_mask,
        }
        if 'dst_target_sequence' in batch[0]:
            # training mode
            dst_target_seqs = [x['dst_target_sequence'] for x in batch]
            dst_target_dict = self.tokenizer(dst_target_seqs,
                                             max_length=1024,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt')
            dst_target_ids = dst_target_dict['input_ids']

            input_batch.update({
                'target_ids_womask': dst_target_ids,
                'target_seqs_womask': dst_target_seqs,
                'decoder_input_ids_womask': t5_shift_tokens_right(dst_target_ids),
            })

        if batch[0]['dst_generation_decoder_inputs'] != '':
            dst_decoder_input_seqs = [x['dst_generation_decoder_inputs'] for x in batch]
            dst_decoder_input_dict = self.tokenizer(dst_decoder_input_seqs,
                                                    max_length=1024,
                                                    padding=True,
                                                    truncation=True,
                                                    return_tensors='pt')
            dst_decoder_input_ids = dst_decoder_input_dict['input_ids']
            dst_decoder_input_ids = t5_shift_tokens_right(dst_decoder_input_ids)
            input_batch.update({
                "decoder_inputs_womask": dst_decoder_input_ids,
            })

        return input_batch

    def get_prompt_init_dict(self, from_cl_domain=None, to_cl_domain=None):
        assert from_cl_domain is not None or to_cl_domain is not None
        if from_cl_domain is None:
            from_prompt_idxs = list(range(self.num_domain_prompt))
        else:
            from_idx = self.domains.index(from_cl_domain)
            from_prompt_idxs = list(range(from_idx * self.num_domain_prompt, (from_idx + 1) * self.num_domain_prompt))
        if to_cl_domain is None:
            to_prompt_idxs = list(range(self.num_domain_prompt))
        else:
            to_idx = self.domains.index(to_cl_domain)
            to_prompt_idxs = list(range(to_idx * self.num_domain_prompt, (to_idx + 1) * self.num_domain_prompt))
        ret_dict = {b: a for a, b in zip(from_prompt_idxs, to_prompt_idxs)}
        return ret_dict

    def domain2soft_prompt(self, domain):
        if domain == 'meta' or self.multitask:
            soft_prompt = ''.join(['<meta_prompt_{}>'.format(i) for i in range(self.num_domain_prompt)])
        else:
            domain_idx = self.domains.index(domain)
            soft_prompt = ''.join(['<prompt_{}>'.format(i + domain_idx * self.num_domain_prompt) for i in
                                   range(self.num_domain_prompt)])
        return soft_prompt

    # convert tokens as graph_prompt as placeholder: <graph_prompt_{0}> ... <graph_prompt_{slot_n}>
    def domain2graph_prompt(self, all_slots): #meta graph prompt
        #graph_prompt = ''.join([s.split(':')[1] for s in self.slot2description[slot]])
        graph_prompt = ''.join(['<graph_prompt_{}>'.format(i) for i in range(len(all_slots))])
        return graph_prompt

    # construct input tokens to GCN: <graph_prompt_{slot_0}> ... <graph_prompt_{slot_n}>
    def domain2slot_token(self, all_slots): #meta graph prompt
        ds_sequence = ''
        for slot in all_slots:
            ds_sequence += self.slot2description[slot].split(':')[1].replace('.','').strip()
        return ds_sequence

    def pad_with_mask(self, dst_input_ids):
        device = dst_input_ids.device
        dst_ds_ids = torch.clone(dst_input_ids).to(device)
        graph_prompt_ids = dst_input_ids - self.tokenizer.vocab_size - len(PROMPT_TOKENS) - len(UNUSED_TOKENS)
        dst_ds_ids[(graph_prompt_ids >= self.num_graph_prompt) | (graph_prompt_ids < 0)] = 0
        
        return dst_ds_ids


    def convert_ontology_to_graph(self, ds_ids, slot_connect, active_slots):
        device = ds_ids.device
        # length of ds_ids is history + </s> + slot_in_turn + 103 + 100
        '''
        pad_slot_connect = torch.zeros((len(ds_ids), len(ds_ids))).to(device) # [103,103] for dataset-order 1
        begin, end = torch.nonzero(ds_ids)[0].item(), torch.nonzero(ds_ids)[-1].item()
        pad_slot_connect[begin: end+1,begin: end+1] = torch.Tensor(slot_connect).to(device)
        edge_index = self.get_edge_index_from_adj_matrix(pad_slot_connect)
        pad_active_slots = torch.zeros(len(ds_ids)).to(device)
        pad_active_slots[ds_ids>0] = torch.Tensor(active_slots).to(device)

        graph_data = []
        for i in range(len(pad_active_slots)):
            if pad_active_slots[i].item() == 1:
                graph_data.append(Data(x=ds_ids.unsqueeze(-1), edge_index=edge_index)) #TODO: change slot_connect and y=y
            else:
                graph_data.append(Data(x=torch.zeros_like(ds_ids).unsqueeze(-1), edge_index=edge_index))
        '''
        edge_index = self.get_edge_index_from_adj_matrix(torch.Tensor(slot_connect).to(device))
        active_slots = torch.Tensor(active_slots).to(device)
        graph_data = [Data(x=active_slots.unsqueeze(-1), edge_index=edge_index)]
        #graph_ids = torch.where(active_slots > 0, ds_ids, 0)
        #graph_data = [Data(x=graph_ids.unsqueeze(-1), edge_index=edge_index)]

        #for i in range(len(active_slots)):
        #    if active_slots[i].item() == 1:
        #        graph_data.append(Data(x=ds_ids.unsqueeze(-1), edge_index=edge_index)) #TODO: change slot_connect and y=y
        #    else:
        #        graph_data.append(Data(x=torch.zeros_like(ds_ids).unsqueeze(-1), edge_index=edge_index))
        return graph_data

    def get_edge_index_from_adj_matrix(self, adj_matrix):
        edge_index, edge_value = dense_to_sparse(adj_matrix)
        undirected = self.undirected 
        self_loops = self.self_loops
        if edge_index.shape[1] != 0 and undirected:
            edge_index = to_undirected(edge_index)
        if edge_index.shape[1] != 0 and self_loops:
            edge_index, _ = add_self_loops(edge_index)
        return edge_index

    def prepare_for_generation(self, dial, domain):
        soft_prompt = self.domain2soft_prompt(domain)
        graph_prompt = self.domain2graph_prompt(self.all_slots) if self.num_graph_prompt != 0 else ''
        return self.convert_dial_to_example(dial, domain, soft_prompt, graph_prompt) # graph_prompt

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        example = None
        for domain in ['memory'] + self.domains:
            domain_num = self.domain2numsamples[domain]
            if index < domain_num:
                example = self.domain2samples[domain][index]
                break
            index -= domain_num
        assert example is not None
        return example

    def make_domain_sampler(self, batch_size, target_domain):
        # target domain could be 'memory'
        assert self.type_path in ['train', 'val']
        start_sample_idx = 0
        end_sample_idx = -1
        # only sample in the target domain
        for domain in ['memory'] + self.domains:
            domain_num = self.domain2numsamples[domain]
            if target_domain == domain:
                end_sample_idx = start_sample_idx + domain_num
                break
            else:
                start_sample_idx += domain_num
        assert end_sample_idx > -1
        return BatchSampler(RangeIndexSampler(start_sample_idx, end_sample_idx), batch_size, drop_last=False)

    def make_full_data_sampler(self, batch_size):
        # For multi-task learning
        return BatchSampler(RangeIndexSampler(0, self.dataset_len), batch_size, drop_last=False)

    def make_sequential_domain_sampler(self, batch_size, target_domain):
        # target domain could be 'memory'
        assert self.type_path in ['train', 'val']
        start_sample_idx = 0
        end_sample_idx = -1
        for domain in ['memory'] + self.domains:
            domain_num = self.domain2numsamples[domain]
            if target_domain == domain:
                end_sample_idx = start_sample_idx + domain_num
                break
            else:
                start_sample_idx += domain_num
        assert end_sample_idx > -1
        return BatchSampler(RangeIndexSeqSampler(start_sample_idx, end_sample_idx), batch_size, drop_last=False)

# Evlauate the overall generation performance
class T5PromptGenFullStateDataset(MemT5PromptDSTDataset):
    def convert_dial_to_example(self, dial, domain, soft_prompt, graph_prompt='', do_augment=False, extra_aug_slots=[]):
        # add active slots property
        active_slots = [] # active slots in a turn: 1 * len(self.all_slots)
        for slot in self.all_slots:
            if slot in dial['state'].keys():
                active_slots.append(1)
            else:
                active_slots.append(0)

        input_dict = {
            'dst_input_sequence': (dial['history'] + soft_prompt).lower(),
            'dst_target_sequence': dial['reply'].replace('</s>', '').lower(),
            'dial_id': dial['dataset'] + '_' + dial['dial_id'] + '_' + str(dial['turn_id']),
            'slot_connect': self.slot_connect,
            'active_slots': active_slots,
        }
        return input_dict


    def collate_fn(self, batch):
        dst_input_seqs = [x['dst_input_sequence'] for x in batch]

        dst_input_dict = self.tokenizer(dst_input_seqs,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']
        slot_connect = [x['slot_connect']for x in batch]
        active_slots = [x['active_slots']for x in batch]
        #dst_ds_seqs = [x['ds_sequence']for x in batch]
        #dst_ds_dict = self.tokenizer(dst_ds_seqs,
        #                                max_length=50, #same as prompts_config
        #                                padding=True,
        #                                truncation=True,
        #                                return_tensors='pt')
        #dst_ds_ids = dst_ds_dict['input_ids']
        #dst_ds_mask = dst_ds_dict['attention_mask']
        pad_ds_ids = self.pad_with_mask(dst_input_ids) #numpy
        graph_list = []
        for i in range(len(batch)):
            graph_list += self.convert_ontology_to_graph(pad_ds_ids[i], slot_connect[i], active_slots[i])
        batch_graph = Batch.from_data_list(graph_list).to(dst_input_ids.device)

        input_batch = {
            "input_ids_womask": dst_input_ids,
            "attention_mask_womask": dst_input_mask,
            'input_seqs_womask': dst_input_seqs,
            'batch_graph': batch_graph,
            #'slot_connect': slot_connect,
        }
        if 'dst_target_sequence' in batch[0]:
            # training mode
            dst_target_seqs = [x['dst_target_sequence'] for x in batch]
            dst_target_dict = self.tokenizer(dst_target_seqs,
                                             max_length=1024,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt')
            dst_target_ids = dst_target_dict['input_ids']

            input_batch.update({
                'target_ids_womask': dst_target_ids,
                'target_seqs_womask': dst_target_seqs,
                'decoder_input_ids_womask': t5_shift_tokens_right(dst_target_ids),
            })
        return input_batch
