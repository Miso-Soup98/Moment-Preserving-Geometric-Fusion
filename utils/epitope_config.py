from copy import deepcopy
from numbers import Integral
from data.epitope import EPITOPE_POLICIES, normalize_paratope

def resolve_epitope_config(paratope, policy='auto', per_cdr=48):
    loops = list(normalize_paratope(paratope))
    if policy == 'auto':
        policy = 'cached_h3' if loops == ['H3'] else 'paratope_union'
    if policy not in EPITOPE_POLICIES:
        raise ValueError('Unknown epitope selection policy: ' + str(policy))
    if isinstance(per_cdr, bool) or not isinstance(per_cdr, Integral) or per_cdr < 1:
        raise ValueError('epitope_size must be a positive integer')
    if policy == 'cached_h3' and per_cdr != 48:
        raise ValueError('cached_h3 preserves the original cached patch; epitope_size must remain 48')
    return {'schema_version': 1, 'policy': policy, 'paratope': loops, 'per_cdr_residues': int(per_cdr), 'reference_source': 'declared benchmark epitope from reference structure before prediction', 'selection': 'cached H3 patch and ordering' if policy == 'cached_h3' else 'union of per-CDR nearest antigen residues by observed canonical heavy-atom distance', 'ordering': 'cache order' if policy == 'cached_h3' else 'source antigen chain and residue order'}

def model_epitope_config(model):
    saved = getattr(model, 'epitope_config', None)
    paratope = getattr(model, 'paratope', 'H3')
    if saved is None:
        return resolve_epitope_config(paratope, 'cached_h3', 48)
    if not isinstance(saved, dict) or saved.get('schema_version') != 1:
        raise ValueError('Invalid checkpoint epitope_config schema')
    if saved.get('policy') not in EPITOPE_POLICIES:
        raise ValueError('Checkpoint epitope_config must store a resolved policy, not auto')
    actual = resolve_epitope_config(paratope, saved['policy'], saved.get('per_cdr_residues'))
    if saved.get('paratope') != actual['paratope']:
        raise ValueError('Checkpoint epitope_config conflicts with model.paratope')
    return deepcopy(actual)
