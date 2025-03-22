# Metaxa

**Metaxa** is a deep learning–based classifier for metagenomic data that predicts taxonomic labels at the species and genus levels. 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lbcb-sci/metaxa.git
   cd metaxa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   ```

3. Download model:
   ```bash
   wget ...
   ```

## Usage
Once installed, you can run Metaxa from the command line:

```bash
python metaxa/inference.py -c model.ckpt -d cuda:0 -b 1024 --n_workers 16 -o output.tsv reads.fastq
```

### Arguments

| Argument         | Description                                   | Example            |
|------------------|-----------------------------------------------|--------------------|
| --checkpoint, -c | Path to model checkpoint                      | -c checkpoint.ckpt |
| --device, -d     | Device to run inference on                    | -d cuda:0          |
| --batch_size, -b | Batch size                                    | -b 1024            |
| --n_workers      | Number of data loading workers                | --n_workers 16     |
| --output, -o     | Path to output classification file            | -o output.tsv      |
|                  | Input FASTQ/A file with sequences to classify | reads.fastq        |

### Output

The output is a TSV file where each row contains:

- Read identifier (`read_id`),
- Predicted species-level taxonomic ID (`species_taxid`), 
- Predicted genus-level taxonomic ID (`genus_taxid`).

## Acknowledgements

This research is supported by the Singapore Ministry of Health’s National Medical Research Council under its Open Fund – Individual Research Grants (NMRC/OFIRG/MOH-000649-00).