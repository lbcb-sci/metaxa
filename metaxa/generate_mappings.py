from pathlib import Path
from collections import defaultdict
import json

import argparse


def parse_metadata(path):
    species_to_id = defaultdict(lambda: len(species_to_id))
    id_to_accessions = []
    id_to_genus = []
    with open(path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            data = line.strip().split(',')
            accession, stid, gtid = data[0], int(data[2]), int(data[3])

            id = species_to_id[stid]
            if id == len(id_to_accessions):
                id_to_accessions.append([accession])
                id_to_genus.append(gtid)
            else:  # Species already exists
                id_to_accessions[id].append(accession)
                # No need to update genus, should be the same

    id_to_species = [k for k, _ in species_to_id.items()]
    return id_to_species, id_to_accessions, id_to_genus


def parse_metadata2(path):
    species, genus = set(), set()
    accession_to_species = {}
    species_to_genus = {}
    with open(path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            data = line.strip().split(',')
            accession, stid, gtid = data[0], int(data[2]), int(data[3])

            species.add(stid)
            genus.add(gtid)

            accession_to_species[accession] = stid
            species_to_genus[stid] = gtid

    species_to_idx = {stid: idx for idx, stid in enumerate(species)}
    idx_to_species = {idx: stid for stid, idx in species_to_idx.items()}

    genus_to_idx = {gtid: idx for idx, gtid in enumerate(genus)}
    idx_to_genus = {idx: gtid for gtid, idx in genus_to_idx.items()}

    sidx_to_gidx = {
        species_to_idx[k]: genus_to_idx[v] for k, v in species_to_genus.items()
    }

    train_dict = {
        'accession_to_species': accession_to_species,
        'species_to_idx': species_to_idx,
        'sidx_to_gidx': sidx_to_gidx,
    }

    inference_dict = {'idx_to_species': idx_to_species, 'idx_to_genus': idx_to_genus}

    return train_dict, inference_dict


def main(args):
    train_dict, inference_dict = parse_metadata2(args.input)

    with open('train_mappings.json', 'w') as f:
        json.dump(train_dict, f, indent=4)

    with open('inference_mappings.json', 'w') as f:
        json.dump(inference_dict, f, indent=4)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=Path)
    # parser.add_argument('-o', '--output', type=Path)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
