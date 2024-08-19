from Bio import SeqIO
from tqdm import tqdm
import polars as pl

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from pathlib import Path


def extract_kmers_for_fasta(path: Path, kmer_len: int, out_folder: Path):
    unique = set()
    for record in SeqIO.parse(path, 'fasta'):
        sequence = str(record.seq)
        u = set(
            sequence[s : s + args.kmer_len]
            for s in range(len(sequence) - args.kmer_len + 1)
        )
        unique.update(u)

    out_path = (out_folder / path.stem).with_suffix('.kmers')
    with out_path.open('w') as out:
        for kmer in unique:
            out.write(f'{kmer}\n')


def main(args: argparse.Namespace) -> None:
    files = sorted(args.input.glob('*.fasta'))

    with ProcessPoolExecutor(args.n_workers) as pool:
        tqdm.write('Extract kmers for every fasta file...')

        futures = []
        for file in files:
            f = pool.submit(extract_kmers_for_fasta, file, args.kmer_len, args.out)
            futures.append(f)

        for result in tqdm(as_completed(futures), total=len(futures)):
            result.result()

        df = pl.scan_csv(
            list(args.out.glob('*.kmers')), has_header=False, new_columns=['kmer']
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kmer_len', type=int, default=35)
    parser.add_argument('--n_workers', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
