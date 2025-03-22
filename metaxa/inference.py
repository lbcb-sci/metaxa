import torch
import torch.utils
from torch.utils.data import IterableDataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd

import torch.multiprocessing as mp
from queue import Empty
from collections import defaultdict
from pathlib import Path
import argparse
import sys

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
                sequence = str(record.seq).upper()

                for start in range(
                    0, len(sequence) - self.chunk_len + 1, self.chunk_len
                ):
                    seq = sequence[start : start + self.chunk_len]
                    try:
                        x = one_hot_encoding(seq)
                    except KeyError as e:
                        print(
                            f'Found invalid character {e} for record {record.id}.',
                            file=sys.stderr,
                        )
                        continue

                    yield record.id, start // self.chunk_len, x


def infer_collate_fn(batch):
    ids, indices, x = zip(*batch)

    # CNN Model
    lens = torch.tensor([b.shape[0] for b in x])

    x = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=0.0
    )  # B x L x 4

    return ids, indices, x, lens


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_id = worker_id


def preds_consumer(checkpoint_path, output_path, in_queue, out_queue):
    data = torch.load(checkpoint_path, map_location='cpu')
    idx_to_species = data['idx_to_species']
    idx_to_genus = data['idx_to_genus']
    del data

    species_logits_dict = defaultdict(
        lambda: torch.zeros(len(idx_to_species), dtype=torch.float32)
    )
    genus_logits_dict = defaultdict(
        lambda: torch.zeros(len(idx_to_genus), dtype=torch.float32)
    )

    while (batch := in_queue.get()) is not None:
        ids, y_species, y_genus = batch

        for rid, logits_species, logits_genus in zip(ids, y_species, y_genus):
            species_logits_dict[rid] += logits_species
            genus_logits_dict[rid] += logits_genus

        out_queue.put((y_species, y_genus))

    results = []
    for read_id in species_logits_dict:
        species_logits = species_logits_dict[read_id]
        genus_logits = genus_logits_dict[read_id]

        species_idx = species_logits.argmax().item()
        genus_idx = genus_logits.argmax().item()

        results.append(
            {
                'read_id': read_id,
                'species_taxid': idx_to_species[species_idx],
                'genus_taxid': idx_to_genus[genus_idx],
            }
        )

    df = pd.DataFrame(results)
    if output_path is None:
        df.to_csv(sys.stdout, sep='\t', index=False)
    else:
        df.to_csv(output_path, sep='\t', index=False)


@torch.inference_mode()
def main(args: argparse.Namespace):
    device = torch.device(args.device)

    model = MetaLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).eval()

    ds = InferenceDataset(args.reads, args.chunk_len, args.n_workers)
    dl = DataLoader(
        ds,
        args.batch_size,
        num_workers=args.n_workers,
        collate_fn=infer_collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    in_queue, out_queue = mp.Queue(), mp.Queue()
    consumer = mp.Process(
        target=preds_consumer,
        args=(args.checkpoint, args.output, in_queue, out_queue),
    )
    consumer.start()

    if device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 8:
        half_dtype = torch.bfloat16
    else:
        half_dtype = torch.float16  # Fallback to float16

    tqdm.write('Inference started.')
    for ids, _, x, lens in tqdm(dl):
        x, lens = x.to(device), lens.to(device)
        with torch.autocast(device.type, dtype=half_dtype):
            y_species, y_genus = model(x, lens)

        y_species, y_genus = y_species.float(), y_genus.float()
        try:
            y_species_cpu, y_genus_cpu = out_queue.get(block=False)

            if y_species_cpu.shape != y_species.shape:
                y_species_cpu = y_species.to(device='cpu')
                y_genus_cpu = y_genus.to(device='cpu')
            else:
                y_species_cpu.copy_(y_species)
                y_genus_cpu.copy_(y_genus)
        except Empty:
            y_species_cpu = y_species.to(device='cpu')
            y_genus_cpu = y_genus.to(device='cpu')

        in_queue.put((ids, y_species_cpu, y_genus_cpu))

    in_queue.put(None)

    tqdm.write('Inference finished. Aggregating results...')
    consumer.join()


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('reads', type=Path)
    parser.add_argument('--checkpoint', '-c', type=Path, required=True)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--chunk_len', type=int, default=1000)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--output', '-o', default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
