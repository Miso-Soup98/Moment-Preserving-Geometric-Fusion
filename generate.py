import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import E2EDataset
from data.pdb_utils import AgAbComplex, Peptide, Protein, Residue, VOCAB
from runtime import device_for, load_model, make_dataset, move_batch
from utils.random_seed import setup_seed

def to_complex(original, coordinates, sequence):
    heavy, light, chain = ([], [], None)
    for index, atoms in zip(sequence, coordinates):
        symbol = VOCAB.idx_to_symbol(index)
        if symbol == VOCAB.BOA:
            continue
        if symbol == VOCAB.BOH:
            chain = heavy
            continue
        if symbol == VOCAB.BOL:
            chain = light
            continue
        if chain is not None:
            names = VOCAB.backbone_atoms + VOCAB.get_sidechain_info(symbol)
            chain.append(Residue(symbol, dict(zip(names, atoms)), (len(chain), ' ')))
    for generated, reference in ((heavy, original.get_heavy_chain()), (light, original.get_light_chain())):
        if len(generated) != len(reference):
            raise ValueError('Generated chain length differs from the input')
        for residue, source in zip(generated, reference):
            residue.id = source.id
    antibody = Protein(original.pdb_id, {original.heavy_chain: Peptide(original.heavy_chain, heavy), original.light_chain: Peptide(original.light_chain, light)})
    result = AgAbComplex(original.antigen, antibody, original.heavy_chain, original.light_chain, skip_epitope_cal=True, skip_validity_check=True)
    result.cdr_pos = original.cdr_pos
    return result

def main():
    parser = argparse.ArgumentParser(description='Generate antibody sequences and PDB structures with MPGF')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.samples < 1:
        parser.error('batch_size and samples must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('The output directory must be empty')
    device = device_for(args.device)
    setup_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval()
    dataset = make_dataset(args.dataset, checkpoint['model_config'], checkpoint['template_map'])
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=E2EDataset.collate_fn)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), (args.output / 'sequences.fasta').open('w', encoding='utf-8') as sequences:
        for sample in range(args.samples):
            entry_index = 0
            for batch in tqdm(loader, desc='Sample {0}/{1}'.format(sample + 1, args.samples)):
                lengths = batch['lengths'].tolist()
                batch = move_batch(batch, device)
                batch.pop('xloss_mask')
                coordinates, identities, scores = model.sample(**batch)
                if not torch.isfinite(coordinates).all() or not torch.isfinite(scores).all():
                    raise FloatingPointError('Generated output is nonfinite')
                start = 0
                for length in lengths:
                    local = dataset._check_load_part(dataset.idx_mapping[entry_index])
                    original = dataset.data[local]
                    result = to_complex(original, coordinates[start:start + length].cpu().tolist(), identities[start:start + length].cpu().tolist())
                    name = '{0:06d}_{1:03d}'.format(entry_index, sample + 1)
                    result.to_pdb(str(args.output / (name + '.pdb')))
                    for label, chain in (('H', result.get_heavy_chain()), ('L', result.get_light_chain())):
                        sequences.write('>{0}_{1}\n{2}\n'.format(name, label, chain.get_seq()))
                    start += length
                    entry_index += 1
if __name__ == '__main__':
    main()
