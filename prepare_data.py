import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
from data.dataset import E2EDataset
from data.paper_template import cache_complex_id, fit_training_template
from data.pdb_utils import AgAbComplex
from utils.hashing import json_hash

def read_summary(path):
    if path.suffix.lower() in ('.json', '.jsonl'):
        text = path.read_text(encoding='utf-8').strip()
        rows = json.loads(text) if text.startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
        for row in rows:
            yield {key: row[key] for key in ('pdb', 'heavy_chain', 'light_chain', 'antigen_chains')}
        return
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            heavy, light = (row['Hchain'].strip(), row['Lchain'].strip())
            if heavy.lower() in ('', 'nan', 'na') or light.lower() in ('', 'nan', 'na'):
                continue
            chains = [value.strip() for value in row['antigen_chain'].split('|')]
            types = [value.strip() for value in row['antigen_type'].split('|')]
            if len(chains) != len(types):
                raise ValueError('Antigen chains and types have different lengths')
            antigen = [chain for chain, kind in zip(chains, types) if kind in ('protein', 'peptide') and chain not in (heavy, light, '', 'NA', 'nan')]
            if antigen:
                yield {'pdb': row['pdb'].lower(), 'heavy_chain': heavy, 'light_chain': light, 'antigen_chains': antigen}

def prepare_structures(args):
    records, seen = ([], set())
    structures = args.output / 'pdb'
    structures.mkdir()
    for row in tqdm(list(read_summary(args.summary)), desc='Prepare structures'):
        identifier = str(row['pdb']).lower()
        if identifier in seen:
            continue
        if not re.fullmatch('[a-z0-9_-]+', identifier):
            raise ValueError('Invalid structure identifier')
        if not row['heavy_chain'] or not row['light_chain'] or (not row['antigen_chains']):
            continue
        if row['heavy_chain'] == row['light_chain']:
            if args.skip_invalid:
                print('Skip {0}: separate input chains are required'.format(identifier))
                continue
            raise ValueError('Separate input chains are required: ' + identifier)
        source = next((args.pdb_dir / (name + '.pdb') for name in (identifier, identifier.upper()) if (args.pdb_dir / (name + '.pdb')).is_file()), None)
        target = structures / (identifier + '.pdb')
        try:
            if source is None:
                raise FileNotFoundError('Structure is missing: ' + identifier)
            if args.renumber:
                from utils.renumber import renumber_pdb
                renumber_pdb(str(source), str(target), scheme='imgt', mute=True)
            else:
                shutil.copyfile(source, target)
            complex_ = AgAbComplex.from_pdb(str(target), row['heavy_chain'], row['light_chain'], row['antigen_chains'])
            if not complex_.get_epitope():
                raise ValueError('No input interface residues')
            complex_.to_pdb(str(target))
        except (AssertionError, ValueError, KeyError, OSError) as error:
            target.unlink(missing_ok=True)
            if not args.skip_invalid:
                raise ValueError('Invalid input {0}: {1}'.format(identifier, error)) from error
            print('Skip {0}: {1}'.format(identifier, error))
            continue
        seen.add(identifier)
        record = dict(row, pdb=identifier, pdb_data_path='pdb/' + target.name, numbering='imgt')
        record['heavy_chain_seq'] = complex_.get_heavy_chain().get_seq()
        record['light_chain_seq'] = complex_.get_light_chain().get_seq()
        for cdr in ('H1', 'H2', 'H3', 'L1', 'L2', 'L3'):
            record['cdr' + cdr.lower() + '_seq'] = complex_.get_cdr(cdr).get_seq()
        records.append(record)
    if len(records) < 3:
        raise ValueError('At least three valid structures are required')
    return records

def cluster_records(records, args):
    executable = shutil.which(args.mmseqs)
    if executable is None:
        raise FileNotFoundError('MMseqs2 is required; set --mmseqs to its executable')
    with tempfile.TemporaryDirectory(prefix='mpgf-clusters-') as directory:
        directory = Path(directory)
        fasta = directory / 'sequences.fasta'
        with fasta.open('w', encoding='utf-8') as handle:
            for row in records:
                sequence = ''.join((row['cdr' + cdr.lower() + '_seq'] for cdr in args.cluster_cdr))
                if not sequence:
                    raise ValueError('Empty clustering sequence: ' + row['pdb'])
                handle.write('>' + row['pdb'] + '\n' + sequence + '\n')
        prefix = directory / 'clusters'
        subprocess.run([executable, 'easy-cluster', str(fasta), str(prefix), str(directory / 'work'), '--min-seq-id', str(args.identity), '-c', str(args.coverage), '--cov-mode', '0', '--cluster-mode', '1', '--threads', str(args.threads), '-v', '1'], check=True)
        mapping = {}
        with (directory / 'clusters_cluster.tsv').open(encoding='utf-8') as handle:
            for line in handle:
                representative, member = line.rstrip('\n').split('\t')
                mapping[member] = representative
        if set(mapping) != {row['pdb'] for row in records}:
            raise ValueError('Clustering output does not cover all inputs')
    clusters = {}
    for row in records:
        cluster = mapping[row['pdb']]
        row['cluster'] = cluster
        clusters.setdefault(cluster, []).append(row)
    return clusters

def split_records(clusters, args):
    rng = random.Random(args.seed)
    test_ids = set()
    if args.test_ids:
        test_ids = set(args.test_ids.read_text(encoding='utf-8').lower().split())
    elif args.benchmark == 'rabd':
        test_ids = set(json.loads((Path(__file__).parent / 'data' / 'benchmark.json').read_text()))
    if test_ids:
        test_clusters = {key for key, rows in clusters.items() if any((row['pdb'] in test_ids for row in rows))}
        test = [row for key in sorted(test_clusters) for row in clusters[key] if row['pdb'] in test_ids]
        remaining = sorted(set(clusters) - test_clusters)
        if not test or len(remaining) < 2:
            raise ValueError('Benchmark split requires test structures and at least two independent remaining clusters')
        rng.shuffle(remaining)
        valid_count = max(1, int(len(remaining) * args.valid_ratio))
        valid_keys, train_keys = (remaining[:valid_count], remaining[valid_count:])
    else:
        keys = sorted(clusters)
        if len(keys) < 3:
            raise ValueError('At least three sequence clusters are required')
        rng.shuffle(keys)
        valid_count = max(1, int(len(keys) * args.valid_ratio))
        test_count = max(1, int(len(keys) * args.test_ratio))
        if valid_count + test_count >= len(keys):
            raise ValueError('Split ratios leave no training clusters')
        test_keys = keys[:test_count]
        valid_keys = keys[test_count:test_count + valid_count]
        train_keys = keys[test_count + valid_count:]
        test = [row for key in test_keys for row in clusters[key]]
    train = [row for key in train_keys for row in clusters[key]]
    valid = [row for key in valid_keys for row in clusters[key]]
    if not train or not valid or (not test):
        raise ValueError('All splits must be nonempty')
    return {'train': train, 'valid': valid, 'test': test}

def main():
    parser = argparse.ArgumentParser(description='Prepare MPGF datasets from a local summary and PDB files')
    parser.add_argument('--summary', type=Path, required=True, help='SAbDab TSV or JSON/JSONL with pdb, heavy_chain, light_chain, antigen_chains')
    parser.add_argument('--pdb_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--renumber', action='store_true', help='Apply IMGT numbering with ANARCI; otherwise inputs must already use IMGT')
    parser.add_argument('--skip_invalid', action='store_true')
    parser.add_argument('--benchmark', choices=('rabd', 'random'), default='rabd')
    parser.add_argument('--test_ids', type=Path, help='Whitespace-separated structure identifiers reserved for testing')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--identity', type=float, default=0.4)
    parser.add_argument('--coverage', type=float, default=0.8)
    parser.add_argument('--cluster_cdr', nargs='+', choices=('H1', 'H2', 'H3', 'L1', 'L2', 'L3'), default=['H3'])
    parser.add_argument('--mmseqs', default='mmseqs')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cache_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if any((not math.isfinite(value) or not 0 < value < 1 for value in (args.valid_ratio, args.test_ratio))):
        parser.error('Split ratios must be in (0,1)')
    if args.valid_ratio + args.test_ratio >= 1:
        parser.error('Validation and test ratios must sum to less than 1')
    if any((not math.isfinite(value) or not 0 < value <= 1 for value in (args.identity, args.coverage))):
        parser.error('Identity and coverage must be in (0,1]')
    if args.threads < 1 or args.cache_size < 1:
        parser.error('threads and cache_size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('The output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    splits = split_records(cluster_records(prepare_structures(args), args), args)
    datasets = {}
    for name, rows in splits.items():
        manifest = args.output / (name + '.json')
        with manifest.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        datasets[name] = E2EDataset(str(manifest), num_entry_per_file=args.cache_size, paper_contract=True)
        if len(datasets[name]) != len(rows):
            raise ValueError('Prepared dataset count differs from split manifest')
    training_ids = [cache_complex_id(item) for item in datasets['train'].iter_complexes()]
    template = fit_training_template(datasets['train'].iter_complexes(), json_hash(splits['train']), training_ids)
    with (args.output / 'template.json').open('w', encoding='utf-8') as handle:
        json.dump({'template_map': template['template_map']}, handle, indent=2)
    print(json.dumps({name: len(rows) for name, rows in splits.items()}))
if __name__ == '__main__':
    main()
