import json
from pathlib import Path
import torch
from data.dataset import E2EDataset
from data.paper_template import validate_template_map
from data.pdb_utils import VOCAB
from models.mpgf_model import MPGFModel

def device_for(value):
    device = torch.device(value)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA is unavailable; use --device cpu')
    return device

def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

def build_model(config):
    return MPGFModel(config['embed_size'], config['hidden_size'], VOCAB.MAX_ATOM_NUMBER, VOCAB.get_num_amino_acid_type(), k_neighbors=config['k_neighbors'], n_layers=config['n_layers'], iter_round=config['iter_round'], dropout=config['dropout'], cdr_type=config['cdr'], paratope=config['paratope'], fusion_variant='mpgf', fusion_config=config['fusion'], paper_contract=True, condition=config['condition'], loss_mode=config['loss_mode'])

def load_template(directory):
    document = json.loads((Path(directory) / 'template.json').read_text(encoding='utf-8'))
    return validate_template_map(document['template_map'])

def make_dataset(path, config, template):
    return E2EDataset(str(path), cdr=config['cdr'], paratope=config['paratope'], paper_contract=True, condition=config['condition'], template_map=template, epitope_policy=config['epitope_policy'], epitope_size=config['epitope_size'])

def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'mpgf-runtime-v1':
        raise ValueError('Unsupported checkpoint format')
    model = build_model(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device)
    return (model, checkpoint)
