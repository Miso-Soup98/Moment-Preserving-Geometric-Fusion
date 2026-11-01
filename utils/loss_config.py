LOSS_MODES = ('full', 'seq_coord_dist')
LOSS_TERMS = ('seq', 'coord', 'bb', 'sc', 'shadow', 'dist')

def resolve_loss_config(mode='full'):
    if mode not in LOSS_MODES:
        raise ValueError('Unknown loss mode: ' + str(mode))
    active = LOSS_TERMS if mode == 'full' else ('seq', 'coord', 'dist')
    return {'mode': mode, 'active_terms': list(active), 'weights': {term: float(term in active) for term in LOSS_TERMS}}
