from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from utils.hashing import json_hash
BACKBONE = ('N', 'CA', 'C', 'O')

def cache_complex_id(complex_):
    return str(complex_.get_id()).split('(')[0]

def validate_template_map(template_map):
    if not isinstance(template_map, dict) or set(template_map) != {'H', 'L'}:
        raise ValueError('Template must contain H and L anchor dictionaries')
    output = {}
    for chain in ('H', 'L'):
        anchors = template_map[chain]
        if not isinstance(anchors, dict) or len(anchors) < 2:
            raise ValueError('Each template chain requires at least two numbered anchors')
        output[chain] = {}
        for position, coordinates in anchors.items():
            try:
                number = int(position)
            except (TypeError, ValueError):
                raise ValueError('Template anchor numbering must be integral') from None
            if str(number) != str(position):
                raise ValueError('Template anchor numbering must use canonical integer strings')
            values = np.asarray(coordinates, dtype=np.float64)
            if values.shape != (4, 3) or not np.isfinite(values).all():
                raise ValueError('Template anchors require finite [N,CA,C,O] coordinates')
            output[chain][str(number)] = values.tolist()
    return output

class PaperTemplateGenerator:

    def __init__(self, template_map=None, template_path=None):
        if (template_map is None) == (template_path is None):
            raise ValueError('Provide exactly one template_map or template_path')
        if template_path is not None:
            document = json.loads(Path(template_path).read_text())
            template_map = document.get('template_map', document)
        self.template_map = validate_template_map(template_map)

    def _chain(self, chain, label, n_channel):
        values, hits = ([], [])
        for index, residue in enumerate(chain):
            position, insertion = residue.get_id()
            anchor = self.template_map[label].get(str(position))
            if anchor is not None:
                backbone = np.asarray(anchor, dtype=np.float64)
                value = np.concatenate((backbone, np.repeat(backbone[1:2], n_channel - 4, axis=0)))
                hits.append(index)
            else:
                value = np.zeros((n_channel, 3), dtype=np.float64)
            values.append(value)
        if len(hits) < 2:
            raise ValueError('Template initialization requires two observed numbered anchor hits on chain ' + label)
        values = np.asarray(values)
        for left, right in zip(hits[:-1], hits[1:]):
            step = (values[right] - values[left]) / (right - left)
            for index in range(left + 1, right):
                values[index] = values[index - 1] + step
        if hits[0]:
            step = values[hits[0]] - values[hits[0] + 1]
            for index in range(hits[0] - 1, -1, -1):
                values[index] = values[index + 1] + step
        if hits[-1] != len(values) - 1:
            step = values[hits[-1]] - values[hits[-1] - 1]
            for index in range(hits[-1] + 1, len(values)):
                values[index] = values[index - 1] + step
        return values

    def construct_template(self, cplx, n_channel=14, align=False):
        if align:
            raise ValueError('Paper templates never align initialization to a target antibody')
        if type(n_channel) is not int or n_channel < 4:
            raise ValueError('Template atom layout must include four backbone slots')
        return np.concatenate((self._chain(cplx.get_heavy_chain(), 'H', n_channel), self._chain(cplx.get_light_chain(), 'L', n_channel)))

def _coordinates(residue):
    values = residue.get_backbone_coord_map()
    if any((atom not in values for atom in BACKBONE)):
        return None
    result = np.asarray([values[atom] for atom in BACKBONE], dtype=np.float64)
    return result if result.shape == (4, 3) and np.isfinite(result).all() else None

def _align(moving, fixed):
    moving, fixed = (moving.reshape(-1, 3), fixed.reshape(-1, 3))
    if len(moving) < 3:
        raise ValueError('Template fitting needs three non-collinear common backbone points')
    centered = moving - moving.mean(0)
    other = fixed - fixed.mean(0)
    if np.linalg.matrix_rank(centered) < 2 or np.linalg.matrix_rank(other) < 2:
        raise ValueError('Template fitting needs three non-collinear common backbone points')
    u, _, vt = np.linalg.svd(centered.T @ other)
    correction = np.eye(3)
    correction[2, 2] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = u @ correction @ vt
    translation = fixed.mean(0) - moving.mean(0) @ rotation
    return (rotation, translation)

def fit_training_template(complexes, retained_manifest_sha256, allowed_complex_ids, conserve_threshold=0.95, min_position_coverage=0.5):
    if not isinstance(retained_manifest_sha256, str) or len(retained_manifest_sha256) != 64 or any((c not in '0123456789abcdef' for c in retained_manifest_sha256.lower())):
        raise ValueError('Retained training manifest SHA256 is required')
    allowed = list(allowed_complex_ids)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError('Retained training complex IDs must be nonempty and unique')
    if not (0 < conserve_threshold <= 1 and 0 <= min_position_coverage < 1):
        raise ValueError('Invalid frozen template fitting configuration')
    frequencies = {'H': defaultdict(Counter), 'L': defaultdict(Counter)}
    insertions = {'H': set(), 'L': set()}
    records, seen = ([], set())
    for complex_ in complexes:
        ident = cache_complex_id(complex_)
        if ident not in allowed or ident in seen:
            raise ValueError('Template input is excluded, non-training, or duplicated: ' + ident)
        seen.add(ident)
        record = {'complex_id': ident, 'H': {}, 'L': {}}
        for chain in ('H', 'L'):
            for segment in (1, 2, 3, 4):
                framework = complex_.get_framework(chain + 'FR' + str(segment))
                if framework is None:
                    continue
                for residue in framework:
                    position, insertion = residue.get_id()
                    if str(insertion).strip():
                        insertions[chain].add(position)
                        continue
                    if position in record[chain]:
                        raise ValueError('Duplicate numbered framework residue in template training input')
                    frequencies[chain][position][residue.get_symbol()] += 1
                    record[chain][position] = _coordinates(residue)
        records.append(record)
    if seen != set(allowed):
        raise ValueError('Template source does not cover the exact retained training ID manifest')
    positions = {chain: sorted((position for position, counts in frequencies[chain].items() if position not in insertions[chain] and sum(counts.values()) > len(records) * min_position_coverage and (max(counts.values()) / sum(counts.values()) > conserve_threshold))) for chain in ('H', 'L')}
    if any((len(positions[chain]) < 2 for chain in ('H', 'L'))):
        raise ValueError('Insufficient retained-training conserved framework anchors')
    order = [(chain, position) for chain in ('H', 'L') for position in positions[chain]]
    arrays, masks = ([], [])
    for record in records:
        mask = np.asarray([record[chain].get(position) is not None for chain, position in order])
        coordinates = np.zeros((len(order), 4, 3), dtype=np.float64)
        for index, (chain, position) in enumerate(order):
            if mask[index]:
                coordinates[index] = record[chain][position]
        arrays.append(coordinates)
        masks.append(mask)
    reference_index = min(range(len(records)), key=lambda i: (-int(masks[i].sum()), records[i]['complex_id']))
    reference, reference_mask = (arrays[reference_index], masks[reference_index])
    total, counts = (np.zeros_like(reference), np.zeros(len(order), dtype=np.int64))
    fitted, skipped = ([], [])
    for record, values, mask in zip(records, arrays, masks):
        try:
            common = mask & reference_mask
            rotation, translation = _align(values[common], reference[common])
        except ValueError as error:
            skipped.append({'complex_id': record['complex_id'], 'reason': str(error)})
            continue
        total[mask] += values[mask] @ rotation + translation
        counts[mask] += 1
        fitted.append(record['complex_id'])
    if not fitted or not counts.all():
        raise ValueError('No finite retained-training template can be fitted at every anchor')
    coordinates = total / counts[:, None, None]
    template = {'H': {}, 'L': {}}
    for index, (chain, position) in enumerate(order):
        template[chain][str(position)] = coordinates[index].tolist()
    template = validate_template_map(template)
    metadata = dict(schema_version=1, fit_sources='retained_training_only', source_training_manifest_sha256=retained_manifest_sha256, source_complex_ids=sorted(seen), fitted_complex_ids=sorted(fitted), skipped_fitting_inputs=skipped, excluded_source_check=True, fit_configuration={'conserve_threshold': conserve_threshold, 'min_position_coverage': min_position_coverage, 'alignment': 'proper_Kabsch_to_most_observed_training_reference'}, reference_complex_id=records[reference_index]['complex_id'], anchor_observation_counts={chain: {str(position): int(counts[order.index((chain, position))]) for position in positions[chain]} for chain in ('H', 'L')}, template_map_sha256=json_hash(template))
    return {'schema_version': 'mpgf-paper-template-v1', 'template_map': template, 'metadata': metadata}
