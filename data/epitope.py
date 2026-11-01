from copy import copy, deepcopy
from numbers import Integral
import numpy as np
from scipy.spatial import cKDTree
from .pdb_utils import VOCAB
EPITOPE_POLICIES = ('cached_h3', 'paratope_union')
_CDRS = ('H1', 'H2', 'H3', 'L1', 'L2', 'L3')
_CANONICAL_SYMBOLS = frozenset('ACDEFGHIKLMNPQRSTVWY')

def normalize_paratope(paratope):
    if isinstance(paratope, str):
        paratope = [paratope]
    try:
        names = list(paratope)
    except TypeError:
        raise ValueError('paratope must be a nonempty CDR name or iterable of CDR names') from None
    if not names:
        raise ValueError('paratope must contain at least one CDR')
    normalized = []
    for name in names:
        if not isinstance(name, str) or name.strip().upper() not in _CDRS:
            raise ValueError('Invalid paratope CDR %r; expected one of %s' % (name, _CDRS))
        normalized.append(name.strip().upper())
    if len(set(normalized)) != len(normalized):
        raise ValueError('paratope must not contain duplicate CDRs')
    return tuple((name for name in _CDRS if name in normalized))

def _observed_atoms(residue):
    symbol = residue.get_symbol().upper()
    if symbol not in _CANONICAL_SYMBOLS:
        return []
    coordinates = residue.get_coord_map()
    atoms = []
    for name in VOCAB.backbone_atoms + VOCAB.get_sidechain_info(symbol):
        if name not in coordinates:
            continue
        try:
            xyz = np.asarray(coordinates[name], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if xyz.shape == (3,) and np.isfinite(xyz).all():
            atoms.append(xyz)
    return atoms

def _nearest_indices(distances, count):
    finite = np.flatnonzero(np.isfinite(distances))
    if len(finite) <= count:
        return finite
    cutoff = np.partition(distances[finite], count - 1)[count - 1]
    tolerance = 64 * np.finfo(np.float64).eps * max(1.0, abs(cutoff))
    before = finite[distances[finite] < cutoff - tolerance]
    boundary = finite[np.abs(distances[finite] - cutoff) <= tolerance]
    return np.concatenate((before, boundary[:count - len(before)]))

def select_epitope(complex, paratope, policy, per_cdr=48):
    paratope = normalize_paratope(paratope)
    if policy not in EPITOPE_POLICIES:
        raise ValueError('Unknown epitope policy %r; expected %s' % (policy, EPITOPE_POLICIES))
    if isinstance(per_cdr, bool) or not isinstance(per_cdr, Integral) or per_cdr < 1:
        raise ValueError('per_cdr must be a positive integer')
    if policy == 'cached_h3':
        return copy(complex).get_epitope()
    antigen = getattr(complex, 'antigen', None)
    if antigen is None:
        antigen = complex.get_antigen()
    entries, atom_coordinates, owners = ([], [], [])
    for chain_name, chain in antigen:
        for index, residue in enumerate(chain):
            owner = len(entries)
            entries.append((residue, chain_name, index))
            observed = _observed_atoms(residue)
            atom_coordinates.extend(observed)
            owners.extend([owner] * len(observed))
    if not atom_coordinates:
        raise ValueError('Epitope selection requires observed canonical finite antigen atoms')
    atom_coordinates = np.asarray(atom_coordinates, dtype=np.float64)
    owners = np.asarray(owners, dtype=np.intp)
    selected = np.zeros(len(entries), dtype=bool)
    for cdr in paratope:
        try:
            loop = complex.get_cdr(cdr)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError('Epitope selection cannot resolve paratope CDR %s' % cdr) from exc
        loop_atoms = [] if loop is None else [atom for residue in loop for atom in _observed_atoms(residue)]
        if not loop_atoms:
            raise ValueError('Epitope selection requires observed canonical finite atoms in CDR %s' % cdr)
        tree = cKDTree(np.asarray(loop_atoms, dtype=np.float64))
        nearest_atom, _ = tree.query(atom_coordinates, k=1)
        distances = np.full(len(entries), np.inf, dtype=np.float64)
        np.minimum.at(distances, owners, nearest_atom)
        selected[_nearest_indices(distances, int(per_cdr))] = True
    return [(deepcopy(residue), chain, index) for keep, (residue, chain, index) in zip(selected, entries) if keep]
