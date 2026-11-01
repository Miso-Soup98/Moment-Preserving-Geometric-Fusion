import os
import json
import pickle
import argparse
from typing import List
import numpy as np
import torch
from utils.logger import print_log
from tqdm import tqdm
from .pdb_utils import AgAbComplex, VOCAB

def _generate_chain_data(residues, start, paper_contract=False):
    backbone_atoms = VOCAB.backbone_atoms
    X, S, res_pos, xloss_mask = ([], [], [], [])
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    S.append(VOCAB.symbol_to_idx(start))
    res_pos.append(0)
    xloss_mask.append([0 for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    for residue in residues:
        residue_xloss_mask = [0 for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        if paper_contract:
            candidates = [bb_atom_coord['CA']] if 'CA' in bb_atom_coord else []
            candidates += list(bb_atom_coord.values()) + list(sc_atom_coord.values())
            ca_x = next((coord for coord in candidates if np.asarray(coord).shape == (3,) and np.isfinite(coord).all()), [0.0, 0.0, 0.0])
        elif 'CA' not in bb_atom_coord:
            for atom in bb_atom_coord:
                ca_x = bb_atom_coord[atom]
                print_log('no ca, use {0}'.format(atom), level='DEBUG')
                break
        else:
            ca_x = bb_atom_coord['CA']
        x = [ca_x for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        i = 0
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1
        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])
        xloss_mask.append(residue_xloss_mask)
    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center
    if start == VOCAB.BOA:
        res_pos = [0 for _ in res_pos]
    data = {'X': X, 'S': S, 'residue_pos': res_pos, 'xloss_mask': xloss_mask}
    return data

class E2EDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, save_dir=None, cdr=None, paratope='H3', full_antigen=False, num_entry_per_file=-1, random=False, read_only_cache=False, paper_contract=False, condition='P2-B', epitope_policy='auto', epitope_size=48, template_map=None):
        super().__init__()
        self.read_only_cache = read_only_cache
        self.cdr = cdr
        self.paratope = paratope
        self.full_antigen = full_antigen
        self.paper_contract = bool(paper_contract)
        self.condition = condition
        self.paper_template = None
        if template_map is not None:
            from .paper_template import PaperTemplateGenerator
            self.paper_template = PaperTemplateGenerator(template_map=template_map)
        from utils.epitope_config import resolve_epitope_config
        self.epitope_config = resolve_epitope_config(paratope, epitope_policy, epitope_size)
        if full_antigen:
            self.epitope_config = dict(self.epitope_config, policy='full_antigen', selection='all cached antigen residues in source order')
        if condition not in ('P2-B', 'P1'):
            raise ValueError('Unknown MPGF input condition: ' + str(condition))
        if condition == 'P1':
            cdr_list = cdr if isinstance(cdr, list) else [cdr]
            paratope_list = paratope if isinstance(paratope, list) else [paratope]
            if not paper_contract or cdr_list != ['H3'] or paratope_list != ['H3']:
                raise ValueError('P1 data requires paper_contract and H3 design/paratope')
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, '{0}_processed'.format(prefix))
        if not os.path.exists(save_dir):
            if read_only_cache:
                raise FileNotFoundError('Read-only dataset cache is missing: {0}'.format(save_dir))
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AgAbComplex] = []
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = [os.path.join(save_dir, os.path.basename(name)) for name in metainfo['file_names']]
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            if read_only_cache:
                raise FileNotFoundError('Read-only cache metadata is missing: {0}'.format(metainfo_file))
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            if read_only_cache:
                raise ValueError('Invalid read-only cache metadata: {0}'.format(metainfo_file)) from e
            print_log('Faild to load file {0}, error: {1}'.format(metainfo_file, e), level='WARN')
            need_process = True
        if read_only_cache:
            if not self.file_names or len(self.file_names) != len(self.file_num_entries):
                raise ValueError('Read-only cache has no parts or inconsistent part counts')
            if any((not isinstance(n, int) or n <= 0 for n in self.file_num_entries)):
                raise ValueError('Read-only cache has invalid part entry counts')
            if sum(self.file_num_entries) != self.num_entry:
                raise ValueError('Read-only cache total disagrees with part entry counts')
            for part_path in self.file_names:
                if not os.path.isfile(part_path):
                    raise FileNotFoundError('Read-only cache part is missing: {0}'.format(part_path))
        if need_process:
            self.file_names, self.file_num_entries = ([], [])
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)
            metainfo = {'num_entry': self.num_entry, 'file_names': [os.path.basename(name) for name in self.file_names], 'file_num_entries': self.file_num_entries}
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)
        if not self.file_num_entries:
            raise ValueError('Dataset contains no valid complexes')
        self.random = random
        self.cur_file_idx, self.cur_idx_range = (0, (0, self.file_num_entries[0]))
        self._load_part()
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'

    def _save_part(self, save_dir, num_entry):
        if self.read_only_cache:
            raise RuntimeError('Read-only dataset cannot write cache parts')
        file_name = os.path.join(save_dir, 'part_{0}.pkl'.format(len(self.file_names)))
        print_log('Saving {0} ...'.format(file_name))
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log('Loading preprocessed file {0}, {1}/{2}'.format(f, self.cur_file_idx + 1, len(self.file_names)))
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        if self.read_only_cache and len(self.data) != self.file_num_entries[self.cur_file_idx]:
            raise ValueError('Read-only cache part length disagrees with metadata')
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx

    def __len__(self):
        return self.num_entry

    def iter_complexes(self):
        state = (self.data, self.cur_file_idx, self.cur_idx_range, self.access_idx)
        rng = np.random.get_state()
        try:
            for index in sorted(self.idx_mapping):
                local_index = self._check_load_part(index)
                yield self.data[local_index]
        finally:
            self.data, self.cur_file_idx, self.cur_idx_range, self.access_idx = state
            np.random.set_state(rng)

    def preprocess(self, file_path, save_dir, num_entry_per_file):
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            try:
                cplx = AgAbComplex.from_pdb(os.path.join(os.path.dirname(os.path.abspath(file_path)), item['pdb_data_path']), item['heavy_chain'], item['light_chain'], item['antigen_chains'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                raise ValueError('Invalid complex: {0}'.format(item['pdb'])) from e
            self.data.append(cplx)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    def get_epitope(self, item):
        if self.full_antigen:
            return [(residue, chain_id, index) for chain_id, chain in item.get_antigen() for index, residue in enumerate(chain)]
        from .epitope import select_epitope
        config = getattr(self, 'epitope_config', None)
        if config is None:
            return select_epitope(item, self.paratope, 'cached_h3')
        return select_epitope(item, config['paratope'], config['policy'], config['per_cdr_residues'])

    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]
        ag_residues = [residue for residue, _, _ in self.get_epitope(item)]
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA, getattr(self, 'paper_contract', False))
        hc, lc = (item.get_heavy_chain(), item.get_light_chain())
        hc_residues, lc_residues = ([], [])
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH, getattr(self, 'paper_contract', False))
        for i in range(len(lc)):
            lc_residues.append(lc.get_residue(i))
        lc_data = _generate_chain_data(lc_residues, VOCAB.BOL, getattr(self, 'paper_contract', False))
        data = {key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0) for key in hc_data}
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]] + [0] + [1 for _ in lc_data['S'][1:]]
        if self.cdr is None:
            smask = cmask
        else:
            smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
            cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1
        data['cmask'], data['smask'] = (cmask, smask)
        paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        paratope = [self.paratope] if type(self.paratope) == str else self.paratope
        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1
        data['paratope_mask'] = paratope_mask
        if getattr(self, 'paper_contract', False):
            from .paper_inputs import prepare_paper_item
            try:
                prepare_paper_item(data, hc_residues, lc_residues, len(ag_residues), self.condition, item.get_id())
            except (ValueError, IndexError) as error:
                failure_log = getattr(self, 'entry_failure_log', None)
                if failure_log:
                    with open(failure_log, 'a', encoding='utf-8') as handle:
                        handle.write(json.dumps(dict(complex_id=item.get_id(), condition=self.condition, status='entry_failed', error=str(error))) + '\n')
                raise
            data['complex_id'] = item.get_id()
        if getattr(self, 'condition', 'P2-B') != 'P1':
            generator = getattr(self, 'paper_template', None)
            if generator is None:
                raise ValueError('A training template is required for P2-B')
            data['template'] = generator.construct_template(item, align=False)
        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float, torch.bool]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        has_bonds = ['backbone_bonds' in item for item in batch]
        if any(has_bonds):
            if not all(has_bonds):
                raise ValueError('Cannot mix paper-contract and incomplete items in one batch')
            bonds, offset = ([], 0)
            for item, length in zip(batch, lengths):
                bonds.append(torch.as_tensor(item['backbone_bonds'], dtype=torch.long).reshape(-1, 2) + offset)
                offset += length * VOCAB.MAX_ATOM_NUMBER
            res['backbone_bonds'] = torch.cat(bonds, dim=0)
            res['complex_ids'] = [item.get('complex_id') for item in batch]
        return res
