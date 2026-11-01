import argparse
import math
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import E2EDataset
from runtime import build_model, device_for, load_model, load_template, make_dataset, move_batch
from utils.epitope_config import resolve_epitope_config
from utils.loss_config import LOSS_MODES
from utils.random_seed import setup_seed
from utils.training_numerics import check_and_clip_gradients, require_finite_loss

def parse():
    parser = argparse.ArgumentParser(description='Train MPGF')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=('P1', 'P2-B'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--final_lr', type=float, default=1e-05)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cdr', nargs='+', default=['H3'], choices=('H1', 'H2', 'H3', 'L1', 'L2', 'L3'))
    parser.add_argument('--paratope', nargs='+', default=['H3'], choices=('H1', 'H2', 'H3', 'L1', 'L2', 'L3'))
    parser.add_argument('--epitope_policy', choices=('auto', 'cached_h3', 'paratope_union'), default='auto')
    parser.add_argument('--epitope_size', type=int, default=48)
    parser.add_argument('--loss_mode', choices=LOSS_MODES, default='full')
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--iter_round', type=int, default=3)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error('epochs and batch_size must be positive; num_workers must be nonnegative')
    if any((not math.isfinite(rate) or rate <= 0 for rate in (args.lr, args.final_lr))):
        parser.error('Learning rates must be finite and positive')
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error('grad_clip must be finite and nonnegative')
    if args.condition == 'P1' and (args.cdr != ['H3'] or args.paratope != ['H3']):
        parser.error('P1 requires H3 design and paratope')
    if args.embed_size < 4 or args.hidden_size < 1 or min(args.n_layers, args.iter_round, args.k_neighbors) < 1:
        parser.error('Invalid model dimensions')
    if not 0 <= args.dropout < 1:
        parser.error('dropout must be in [0,1)')
    return args

def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.tmp')
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def main(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('The output directory must be empty')
    device = device_for(args.device)
    setup_seed(args.seed)
    epitope = resolve_epitope_config(args.paratope, args.epitope_policy, args.epitope_size)
    config = {name: getattr(args, name) for name in ('embed_size', 'hidden_size', 'n_layers', 'iter_round', 'k_neighbors', 'dropout', 'cdr', 'paratope', 'condition', 'loss_mode')}
    config.update(epitope_policy=epitope['policy'], epitope_size=epitope['per_cdr_residues'], fusion={'epsilon': 0.0001, 'distance_epsilon': 0.0001, 'length_scale': 1.0, 'node_chunk_size': 256, 'edge_chunk_size': 2048})
    template = load_template(args.data)
    train_set = make_dataset(args.data / 'train.json', config, template)
    valid_set = make_dataset(args.data / 'valid.json', config, template)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=E2EDataset.collate_fn, generator=generator)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=E2EDataset.collate_fn)
    state = None
    if args.resume:
        model, state = load_model(args.resume, device)
        if state['model_config'] != config or state['template_map'] != template:
            raise ValueError('Resume configuration or training template differs')
    else:
        model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    decay = math.log(args.final_lr / args.lr) / total_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: math.exp(decay * (step + 1)))
    training_config = {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'final_lr': args.final_lr, 'seed': args.seed, 'grad_clip': args.grad_clip, 'steps_per_epoch': len(train_loader)}
    start_epoch, best_loss = (0, float('inf'))
    best_source = None
    if state:
        if state['training_config'] != training_config:
            raise ValueError('Resume requires the same total epoch budget and training configuration')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_loss = (state['epoch'], state['best_loss'])
        best_source = args.resume if state['validation_loss'] == best_loss else args.resume.with_name('best.pt')
        best_state = torch.load(best_source, map_location='cpu', weights_only=True)
        if best_state['model_config'] != config or best_state['template_map'] != template or best_state['validation_loss'] != best_loss:
            raise ValueError('The best checkpoint for this run is missing or inconsistent')
        del best_state
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        numpy_rng = state['numpy_rng']
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        generator.set_state(state['loader_rng'])
    if start_epoch >= args.epochs:
        raise ValueError('Checkpoint already reached the requested epoch budget')
    args.output.mkdir(parents=True, exist_ok=True)
    if best_source is not None:
        shutil.copyfile(best_source, args.output / 'best.pt')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train, train_count = (0.0, 0)
        for batch in tqdm(train_loader, desc='Train {0}/{1}'.format(epoch + 1, args.epochs)):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, context_ratio=0.0)[0]
            require_finite_loss(loss)
            loss.backward()
            check_and_clip_gradients(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            total_train += float(loss.detach())
            train_count += 1
        model.eval()
        total_valid, valid_count = (0.0, 0)
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Validation'):
                loss = model(**move_batch(batch, device), context_ratio=0.0)[0]
                require_finite_loss(loss)
                total_valid += float(loss)
                valid_count += 1
        valid_loss = total_valid / valid_count
        improved = valid_loss < best_loss
        best_loss = min(best_loss, valid_loss)
        numpy_rng = np.random.get_state()
        checkpoint = {'format': 'mpgf-runtime-v1', 'model_config': config, 'model_state': model.state_dict(), 'template_map': template, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'training_config': training_config, 'epoch': epoch + 1, 'best_loss': best_loss, 'validation_loss': valid_loss, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], 'python_rng': random.getstate(), 'numpy_rng': [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]], 'loader_rng': generator.get_state()}
        save_checkpoint(args.output / 'last.pt', checkpoint)
        if improved:
            save_checkpoint(args.output / 'best.pt', checkpoint)
        print('epoch={0} train={1:.6f} valid={2:.6f} best={3:.6f}'.format(epoch + 1, total_train / train_count, valid_loss, best_loss))
if __name__ == '__main__':
    main(parse())
