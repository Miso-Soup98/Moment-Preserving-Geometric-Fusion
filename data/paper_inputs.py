from numbers import Integral
import numpy as np
from configs import IMGT
from .pdb_utils import VOCAB
_BACKBONE_PAIRS = ((0, 1), (1, 2), (2, 3))

class PaperInputError(ValueError):

    def __init__(self, complex_id, message, reason_code='invalid_paper_input'):
        self.complex_id = complex_id
        self.reason_code = reason_code
        super().__init__('Paper input %s: %s' % (complex_id, message))

def _fail(complex_id, message, reason_code='invalid_paper_input'):
    raise PaperInputError(complex_id, message, reason_code)

def _source_adjacent(left, right):
    for name in ('segment_id', 'polymer_segment', 'segment'):
        lhs, rhs = (getattr(left, name, None), getattr(right, name, None))
        if lhs is not None and rhs is not None and (lhs != rhs):
            return False
    for name in ('source_index', 'polymer_index', 'source_residue_index'):
        lhs, rhs = (getattr(left, name, None), getattr(right, name, None))
        if lhs is not None or rhs is not None:
            if not isinstance(lhs, Integral) or not isinstance(rhs, Integral):
                return False
            return int(rhs) == int(lhs) + 1
    return True

def _backbone_topology(chains, atom_slots):
    edges = []
    for residues, offset in chains:
        for index, residue in enumerate(residues):
            base = (offset + index) * atom_slots
            edges.extend(((base + source, base + target) for source, target in _BACKBONE_PAIRS))
            if index and _source_adjacent(residues[index - 1], residue):
                edges.append(((offset + index - 1) * atom_slots + 2, base))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)

def prepare_paper_item(data, hc_residues, lc_residues, ag_count, condition, complex_id):
    if condition not in ('P1', 'P2-B'):
        _fail(complex_id, 'condition must be P1 or P2-B')
    if isinstance(ag_count, bool) or not isinstance(ag_count, Integral) or ag_count < 1:
        _fail(complex_id, 'at least one real antigen/epitope residue is required')
    hc_residues, lc_residues = (list(hc_residues), list(lc_residues))
    heavy_start = int(ag_count) + 2
    light_start = heavy_start + len(hc_residues) + 1
    expected_nodes = int(ag_count) + len(hc_residues) + len(lc_residues) + 3
    c = VOCAB.MAX_ATOM_NUMBER
    if c != 14:
        _fail(complex_id, 'paper atom layout requires 14 slots')
    required = ('X', 'S', 'cmask', 'smask', 'paratope_mask', 'xloss_mask')
    missing = [key for key in required if key not in data]
    if missing:
        _fail(complex_id, 'missing fields: ' + ', '.join(missing))
    x = np.asarray(data['X'])
    tokens = np.asarray(data['S'])
    if x.shape != (expected_nodes, c, 3) or not np.issubdtype(x.dtype, np.floating):
        _fail(complex_id, 'X must follow full BOA/antigen/BOH/heavy/BOL/light 14-slot layout')
    if tokens.shape != (expected_nodes,) or not np.issubdtype(tokens.dtype, np.integer):
        _fail(complex_id, 'S must be an integer identity vector matching the full node layout')
    markers = ((0, VOCAB.BOA), (heavy_start - 1, VOCAB.BOH), (light_start - 1, VOCAB.BOL))
    if any((tokens[index] != VOCAB.symbol_to_idx(symbol) for index, symbol in markers)):
        _fail(complex_id, 'global marker positions do not match the source chain layout')
    masks = {}
    for key in ('cmask', 'smask', 'paratope_mask', 'xloss_mask'):
        value = np.asarray(data[key])
        expected_shape = (expected_nodes, c) if key == 'xloss_mask' else (expected_nodes,)
        if value.shape != expected_shape or not np.isin(value, (0, 1)).all():
            _fail(complex_id, key + ' has invalid shape or non-boolean entries')
        masks[key] = value.astype(bool, copy=False)
    smask, observed = (masks['smask'], masks['xloss_mask'])
    antibody = np.zeros(expected_nodes, dtype=bool)
    antibody[heavy_start:heavy_start + len(hc_residues)] = True
    antibody[light_start:] = True
    for key in ('cmask', 'smask', 'paratope_mask'):
        if np.any(masks[key] & ~antibody):
            _fail(complex_id, key + ' may contain only real antibody residues')
    real = np.ones(expected_nodes, dtype=bool)
    real[[index for index, _ in markers]] = False
    canonical_ids = {VOCAB.symbol_to_idx(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    if any((int(token) not in canonical_ids for token in tokens[real])):
        _fail(complex_id, 'all real reference identities must be canonical amino acids')
    if not smask.any():
        _fail(complex_id, 'a nonempty sequence-design region is required')
    for residues, offset in ((hc_residues, heavy_start), (lc_residues, light_start)):
        seen = set()
        for index, residue in enumerate(residues):
            key = residue.get_id()
            if not isinstance(key, tuple) or len(key) != 2 or (not isinstance(key[0], Integral)) or isinstance(key[0], bool) or (not isinstance(key[1], str)) or (key in seen):
                _fail(complex_id, 'original chain residue IDs must be legal and unique')
            seen.add(key)
            if int(tokens[offset + index]) != VOCAB.symbol_to_idx(residue.get_symbol()):
                _fail(complex_id, 'reference identities must match original chain residue mappings')
    visible = np.zeros(expected_nodes, dtype=bool)
    visible[1:int(ag_count) + 1] = True
    if condition == 'P1':
        visible |= antibody & ~smask
    if not np.isfinite(x[visible]).all():
        _fail(complex_id, 'permitted visible coordinates contain NaN/Inf')
    occupied = np.zeros((expected_nodes, c), dtype=bool)
    for index in np.flatnonzero(real):
        symbol = VOCAB.idx_to_symbol(int(tokens[index]))
        occupied[index, :len(VOCAB.backbone_atoms) + len(VOCAB.get_sidechain_info(symbol))] = True
    if not (observed[visible] & occupied[visible]).any(axis=1).all():
        _fail(complex_id, 'each permitted visible residue requires an observed canonical atom')
    p1_template = None
    if condition == 'P1':
        design = np.flatnonzero(smask)
        if not len(design):
            _fail(complex_id, 'P1 requires a nonempty H3 design mask')
        if design[0] < heavy_start or design[-1] >= heavy_start + len(hc_residues) or (not np.array_equal(design, np.arange(design[0], design[-1] + 1))):
            _fail(complex_id, 'P1 requires one continuous design interval within the heavy chain')
        lo, hi = IMGT.H3
        h3 = np.asarray([heavy_start + index for index, residue in enumerate(hc_residues) if lo <= int(residue.get_id()[0]) <= hi], dtype=np.int64)
        if not np.array_equal(design, h3):
            _fail(complex_id, 'P1 design mask must equal the complete reference IMGT H3 interval')
        if not np.array_equal(masks['paratope_mask'], smask):
            _fail(complex_id, 'P1 paratope_mask must equal the H3 design mask')
        left, right = (int(design[0]) - 1, int(design[-1]) + 1)
        if left < heavy_start or right >= heavy_start + len(hc_residues):
            _fail(complex_id, 'P1 H3 requires visible flanking residues on both sides', 'p1_flanking_residue_missing')
        if not observed[[left, right], :4].all() or not np.isfinite(x[[left, right], :4]).all():
            missing_flanks = []
            for side, index in (('left', left), ('right', right)):
                missing_atoms = [atom for slot, atom in enumerate(VOCAB.backbone_atoms) if not observed[index, slot] or not np.isfinite(x[index, slot]).all()]
                if missing_atoms:
                    missing_flanks.append('%s %s missing %s' % (side, hc_residues[index - heavy_start].get_id(), '/'.join(missing_atoms)))
            _fail(complex_id, 'P1 flanks require observed finite N/CA/C/O coordinates; ' + '; '.join(missing_flanks), 'p1_flank_backbone_missing')
        fraction = np.arange(1, len(design) + 1, dtype=np.float64) / (len(design) + 1)
        backbone = (1 - fraction[:, None, None]) * x[left, :4] + fraction[:, None, None] * x[right, :4]
        p1_template = np.repeat(backbone[:, 1:2], c, axis=1)
        p1_template[:, :4] = backbone
        p1_template = p1_template.astype(x.dtype, copy=False)
    topology = _backbone_topology(((hc_residues, heavy_start), (lc_residues, light_start)), c)
    data['backbone_bonds'] = topology
    if p1_template is not None:
        data['cmask'] = smask.copy()
        data['template'] = p1_template
    return data
