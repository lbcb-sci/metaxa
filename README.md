# Metaxa

**Metaxa** is a deep-learning-based metagenomic taxonomy classification model that operates at the species and genus level. 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/metaxa.git
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
python metaxa/inference.py -m metadata.json -c model.ckpt -d cuda:0 -b 1024 --n_workers 16 -o output.tsv reads.fastq
```

### Arguments

| Argument         | Description                                   | Example            |
|------------------|-----------------------------------------------|--------------------|
| --metadata, -m   | Path to taxis metadata JSON                   | -m metadata.json   |
| --checkpoint, -c | Path to model checkpoint                      | -c checkpoint.ckpt |
| --device, -d     | Device to run inference on                    | -d cuda:0          |
| --batch_size, -b | Batch size                                    | -b 1024            |
| --n_workers      | Number of data loading workers                | --n_workers 16     |
| --output, -o     | Path to output classification file            | -o output.tsv      |
|                  | Input FASTQ/A file with sequences to classify | reads.fastq        |

### Output

The output is a TSV file where each row contains the `read_id`, the predicted species-level taxonomic ID (`species_taxid`), and the predicted genus-level taxonomic ID (`genus_taxid`).