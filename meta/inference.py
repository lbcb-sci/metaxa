import torch
import torch.nn.functional as F
import torch.utils
from torch.utils.data import IterableDataset, DataLoader
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from more_itertools import grouper

from pathlib import Path
import argparse
import sys
from contextlib import ExitStack
from itertools import chain

from lightning_module import MetaLightningModule
from datasets import one_hot_encoding

FASTA_EXTENSIONS = {'.fasta', '.fa', '.fna'}
FASTQ_EXTENSIONS = {'.fastq', '.fq'}


class InferenceDataset(IterableDataset):
    def __init__(self, path: Path, chunk_len: int, n_workers: int):
        self.path = path
        self.chunk_len = chunk_len
        self.n_workers = n_workers

        ext = path.suffix.lower()
        if ext in FASTA_EXTENSIONS:
            self.reads_format = 'fasta'
        elif ext in FASTQ_EXTENSIONS:
            self.reads_format = 'fastq'
        else:
            raise ValueError(
                f'Reads file have to be either in FASTA or FASTQ format {ext}.'
            )

    def __iter__(self):
        for i, record in enumerate(SeqIO.parse(self.path, self.reads_format)):
            if self.n_workers == 0 or i % self.n_workers == self.worker_id:
                sequence = ''.join(
                    [c if c in 'ACGT' else 'N' for c in str(record.seq).upper()]
                )

                for start in range(
                    0, len(sequence) - self.chunk_len + 1, self.chunk_len
                ):
                    seq = sequence[start : start + self.chunk_len]
                    x = one_hot_encoding(seq)

                    """g = grouper(
                        seq, 5, incomplete='ignore'
                    )  # Ignore partial (last) kmer
                    x = torch.tensor([KMER_ENCODING[k] for k in chain(['CLS'], g)])"""

                    yield record.id, start // self.chunk_len, x


def infer_collate_fn(batch):
    ids, indices, x = zip(*batch)

    # CNN Model
    lens = torch.tensor([b.shape[0] for b in x])  # +1 for CLS token

    x = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=0.0
    )  # B x L x 4

    return ids, indices, x, lens


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_id = worker_id


@torch.inference_mode
def main(args: argparse.Namespace):
    device = torch.device(args.device)

    model = (
        MetaLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
        .eval()
        .to(dtype=torch.bfloat16)
    )

    ds = InferenceDataset(args.reads, args.chunk_len, args.n_workers)
    dl = DataLoader(
        ds,
        args.batch_size,
        num_workers=args.n_workers,
        collate_fn=infer_collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    with ExitStack() as stack:
        if isinstance(args.output, str):
            output = stack.enter_context(open(args.output, 'w'))
        else:
            output = args.output

        # Header
        print('read_id', 'window_id', 'logits', 'class', sep='\t', file=output)

        for ids, indices, x, lens in tqdm(dl):
            x = x.to(dtype=model.dtype, device=device)
            lens = lens.to(device)
            y = model(x, lens)

            for id, idx, probs in zip(
                ids, indices, y.to(device='cpu', dtype=torch.float16).numpy()
            ):
                print(
                    id,
                    idx,
                    ','.join([f'{p:.5f}' for p in probs]),
                    np.argmax(probs),
                    sep='\t',
                    file=output,
                )


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('reads', type=Path)
    parser.add_argument('--checkpoint', '-c', type=Path, required=True)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--chunk_len', type=int, default=1000)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--output', '-o', default=sys.stdout)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
