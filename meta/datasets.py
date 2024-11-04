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
import edlib

from pathlib import Path
import random
from collections import Counter
from itertools import product, chain
import importlib.util
from more_itertools import grouper
import re
from collections import deque
import math

from typing import List, Tuple

BASES_ENCODING = {b: i for i, b in enumerate('ACGT')}
BASES_DECODING = {i: b for i, b in enumerate('ACGT')}
KMER_ENCODING = {k: i for i, k in enumerate(product('ACGT', repeat=5))}
KMER_ENCODING['CLS'] = len(KMER_ENCODING)
KMER_ENCODING['PAD'] = len(KMER_ENCODING)

COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


class MetaDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: Path,
        kmer_len: int,
        train_batch_size: int = 65_536,
        val_batch_size: int = 65_536,
    ):
        super().__init__()

        self.root = root
        self.files = sorted(Path(self.root).glob('*.fasta'))
        self.n_classes = len(self.files)
        self.kmer_len = kmer_len

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        f_ids = self.root / 'ids.txt'

        with f_ids.open('w') as f:
            for id in self.files:
                f.write(f'{id}\n')

    def train_dataloader(self):
        return DataLoader(
            RefSeqDataset(self.files, self.kmer_len, 'train'),
            batch_size=self.train_batch_size,
            num_workers=16,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            RefSeqDataset(self.files, self.kmer_len, 'val'),
            batch_size=self.val_batch_size,
            num_workers=16,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )


def reverse_complement(seq: str) -> str:
    if random.random() < 0.5:
        seq = [COMPLEMENT[b] for b in reversed(seq)]

    return ''.join(seq)


class RefSeqDataset(IterableDataset):
    def __init__(self, files: List[Path], kmer_len: int, mode: str) -> None:
        super().__init__()
        self.badread = BadReadTransform()
        self.kmer_len = kmer_len
        self.mode = mode

        self.ids, self.sequences = [], []
        for path in files:
            records = list(SeqIO.parse(path, 'fasta'))

            # TODO: Just taking the longest one
            # TODO: Not accounting for circular genomes
            # TODO: Not accounting for reverse complement
            longest = max(records, key=lambda r: len(r))

            self.ids.append(longest.id)

            seq = ''.join([c if c in 'ACGT' else 'N' for c in str(longest.seq).upper()])
            self.sequences.append(seq)

        self.n_classes = len(self.ids)

    def __iter__(self):
        while True:
            seq_id = random.randrange(len(self.sequences))
            seq = self.sequences[seq_id]

            start = random.randrange(len(seq) - self.kmer_len + 1)
            original = seq[start : start + self.kmer_len]

            # Transforms
            # TODO N's are replaced randomly
            original = [
                BASES_DECODING[random.randint(0, 3)] if b == 'N' else b
                for b in original
            ]
            original = reverse_complement(original)

            # BadRead transfomrm -> make sure it is long enough
            s = self.badread(original)
            if len(s) < 500:
                continue

            # Cap sequence to 1000 bp
            s = s[:1000]

            matches = {m[0] // 5 for m in get_matches(s, original)}
            length = math.floor((len(s) - 31) / 5 + 1)
            y = torch.tensor(
                [seq_id if i in matches else self.n_classes for i in range(length)]
            )

            yield seq_id, one_hot_encoding(s), y


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
        identity_params: Tuple[float, float, float] = (95, 100, 2.5),
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
    ids, x, y = zip(*batch)
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

    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)

    return ids, x, lens, y


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


def get_matches(query, target, kmer_len=31, step=1):
    qpos, rpos = 0, 0
    d = deque(maxlen=kmer_len)
    matches = []

    cigar = edlib.align(query, target, task='path')['cigar']
    for m in re.finditer(r'(\d+)([=XDI])', cigar):
        l, op = int(m.group(1)), m.group(2)

        for _ in range(l):
            d.append(op)

            if op == '=' or op == 'X':
                qpos += 1
                rpos += 1
            elif op == 'I':
                qpos += 1
            elif op == 'D':
                rpos += 1
            else:
                raise ValueError('Invalid cigar op')

            if qpos >= kmer_len:
                if qpos % step == 0:
                    match = all(map(lambda o: o == '=', d))
                    if match:
                        kmer = query[qpos - kmer_len : qpos]
                        matches.append((qpos - kmer_len, kmer))

    return matches
