import torch.distributed
from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn.functional as F
import lightning as L
from Bio import SeqIO

from badread.identities import Identities
from badread.error_model import ErrorModel
from badread.simulate import add_glitches
from badread.simulate import sequence_fragment

import os
import sys
from pathlib import Path
import random
from collections import Counter, defaultdict
from itertools import product, chain
import importlib.util
from more_itertools import grouper
import math
import re
import json

from typing import List, Tuple, Dict

BASES_ENCODING = {b: i for i, b in enumerate('ACGT')}
BASES_DECODING = {i: b for i, b in enumerate('ACGT')}
KMER_ENCODING = {k: i for i, k in enumerate(product('ACGT', repeat=5))}
KMER_ENCODING['CLS'] = len(KMER_ENCODING)
KMER_ENCODING['PAD'] = len(KMER_ENCODING)

COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


NCBI_ACCESSION_PATTERN = re.compile(r'(GCF_\d+\.\d+)')
FASTA_EXTS = {'.fna', '.fa', '.fasta', '.ffn', '.faa', '.frn'}


class MetaDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: Path,
        metadata: Path,
        kmer_len: int,
        train_batch_size: int = 65_536,
        val_batch_size: int = 65_536,
    ):
        super().__init__()

        with metadata.open('r') as f:
            self.metadata = json.load(f)

        self.n_species = len(self.metadata['sidx_to_gidx'].keys())
        self.n_genus = len(set(self.metadata['sidx_to_gidx'].values()))

        # TODO: This one should be sorted based on indices
        self.files = sorted(
            [file for file in root.rglob('*') if file.suffix in FASTA_EXTS]
        )

        self.kmer_len = kmer_len

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        log_dir = self.trainer.loggers[0].save_dir
        name = self.trainer.loggers[0].name
        version = self.trainer.loggers[0].version
        f_ids = Path(log_dir, name, version, 'mappings.csv')

        f_ids.parent.mkdir(parents=True)
        with f_ids.open('w') as f:
            f.write('id\tname\n')
            for i, path in enumerate(self.files):
                f.write(f'{i}\t{path.stem}\n')

    def train_dataloader(self):
        return DataLoader(
            RefSeqDataset(self.files, self.metadata, self.kmer_len, 'train'),
            batch_size=self.train_batch_size,
            num_workers=16,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            RefSeqDataset(self.files, self.metadata, self.kmer_len, 'val'),
            batch_size=self.val_batch_size,
            num_workers=16,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )


def reverse_complement(seq: str) -> str:
    seq = [COMPLEMENT[b] for b in reversed(seq)]
    return ''.join(seq)


class RefSeqDataset(IterableDataset):
    def __init__(
        self, files: List[Path], metadata: Dict[str, Dict], kmer_len: int, mode: str
    ) -> None:
        super().__init__()
        self.badread = BadReadTransform()
        self.kmer_len = kmer_len
        self.mode = mode

        acc_to_species = {
            k: int(v) for k, v in metadata['accession_to_species'].items()
        }
        species_to_idx = {int(k): int(v) for k, v in metadata['species_to_idx'].items()}

        self.sidx_to_gidx = {
            int(k): int(v) for k, v in metadata['sidx_to_gidx'].items()
        }

        # TODO: Populate this based on indices passed as metadata
        self.sequences = [[] for _ in range(len(species_to_idx))]
        for path in files:
            records = list(SeqIO.parse(path, 'fasta'))

            try:
                accession = NCBI_ACCESSION_PATTERN.search(str(path)).group(1)
            except AttributeError:
                print(f'Cannot parse acession {str(path)}.', file=sys.stderr)

            # TODO: Just taking the longest one
            seqs = []
            for record in records:
                seq = ''.join(
                    [c if c in 'ACGT' else 'N' for c in str(record.seq).upper()]
                )
                seqs.append(seq)

            idx = species_to_idx[acc_to_species[accession]]
            self.sequences[idx].append((accession, seqs))

    def __iter__(self):
        while True:
            # Get all assemblies for some species
            sidx = random.randrange(len(self.sequences))
            species_asms = self.sequences[sidx]

            # Get specific assembly
            if len(species_asms) > 1:
                asm_idx = random.randrange(len(species_asms))
                acc, asm_seqs = species_asms[asm_idx]
            else:
                acc, asm_seqs = species_asms[0]

            # Get specific contig
            if len(asm_seqs) > 1:
                weights = [len(seq) for seq in asm_seqs]
                seq = random.choices(asm_seqs, weights, k=1)[0]
            else:
                seq = asm_seqs[0]

            start = random.randrange(len(seq))
            original = seq[start : start + self.kmer_len]
            if len(original) < self.kmer_len:
                # Handling circular genome
                diff = self.kmer_len - len(original)
                original += seq[:diff]

            # Transforms
            # TODO N's are replaced randomly
            original = ''.join(
                [
                    BASES_DECODING[random.randint(0, 3)] if b == 'N' else b
                    for b in original
                ]
            )

            if random.random() < 0.5:
                original = reverse_complement(original)

            # BadRead transfomrm -> make sure it is long enough
            s = self.badread(original)
            if len(s) < 500:
                continue

            # Cap sequence to 1000 bp
            s = s[:1000]
            s = one_hot_encoding(s)

            yield acc, s, sidx, self.sidx_to_gidx[sidx]


def kmer_encoding_fn(seq):
    g = grouper(seq, 5, incomplete='ignore')  # Ignore partial (last) kmer
    s = torch.tensor([KMER_ENCODING[k] for k in chain(['CLS'], g)])

    return s


def one_hot_encoding(seq):
    x = torch.tensor([BASES_ENCODING[b] for b in seq])
    return F.one_hot(x, len(BASES_ENCODING)).float()  # L x 4


class BadReadTransform:
    def __init__(
        self,
        identity_params: Tuple[float, float, float] = (90, 98, 5),
        error_model: str = 'nanopore2020',
        glitch_params: Tuple[float, float, float] = (10000, 25, 25),
    ):
        folder = Path(importlib.util.find_spec('badread').origin).parent

        with open('/dev/null', 'w') as null:
            self.identities = Identities(
                identity_params[0], identity_params[2], identity_params[1], output=null
            )
            self.error_model = ErrorModel(
                Path(folder, 'error_models', error_model + '.gz'), output=null
            )

        self.glitch_rate = glitch_params[0]
        self.glitch_size = glitch_params[1]
        self.glitch_skip = glitch_params[2]

    def __call__(self, sequence: str) -> str:
        if not isinstance(sequence, str):
            raise TypeError('Input to the BadReadTransform should be a string.')

        sequence = add_glitches(
            sequence, self.glitch_rate, self.glitch_size, self.glitch_skip
        )

        tgt_iden = self.identities.get_identity()
        sequence = sequence_fragment(sequence, tgt_iden, self.error_model)

        return sequence


def train_collate_fn(batch):
    _, x, species_cls, genus_cls = zip(*batch)
    # x, y = zip(*batch)

    # KMER Encoding
    """x = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=KMER_ENCODING['PAD']
    )
    attn_mask = x != KMER_ENCODING['PAD']"""

    # CNN Model
    lens = torch.tensor([b.shape[0] for b in x])

    x = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=0.0
    )  # B x L x 4

    species_cls = torch.tensor(species_cls)
    genus_cls = torch.tensor(genus_cls)

    return x, lens, species_cls, genus_cls


if __name__ == '__main__':
    dataset = MetaDataModule(
        Path(
            '/charonfs/scratch/users/astar/gis/stanojevicd/data/meta/databases/MetageNN_main/main_database/'
        )
    )
    dl = dataset.train_dataloader()

    counter = Counter()
    for i, b in enumerate(dl):
        if i == 10:
            break
        for s in b[0]:
            counter.update(list(s))

    print(counter)
