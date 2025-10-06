# MuseTok

This is the official code implementation for the paper: 

**MuseTok: Symbolic Music Tokenization for Hierarchical Generation and Semantic Understanding.**

## Environment

* Python 3.10 and torch==2.5.1 used for the experiments
* Install dependencies

```
pip install -r requirements.txt
```

## Quick Start

Download and unzip [best weights](https://drive.google.com/file/d/1HK534lEVdHYl3HMRkKvz8CWYliXRmOq_/view?usp=sharing) in the root directory. 

### Music Generation

Generate music pieces by continuing the prompts from our test sets with the two-stage music generation framework:

```
python test_generation.py \
        --configuration=config/generation.yaml \
        --model=ckpt/best_generator/model.pt \
        --use_prompt \
        --primer_n_bar=4 \
        --n_pieces=20 \
        --output_dir=samples/generation
```

Or, generate music pieces from scratch:

```
python test_generation.py \
        --configuration=config/generation.yaml \
        --model=ckpt/best_generator/model.pt \
        --n_pieces=20 \
        --output_dir=samples/generation
```

TODO: 
1. encode midi data into musetok tokens
2. colab file for generation by continuing user inputs


## Train the model

### Data Preparation
Download the datasets used in the paper from [link] and unzip in the root directory `MuseTok`. To train with customized datasets, please refer to the [steps](https://github.com/Yuer867/MuseTok/tree/main/data_processing#readme).

### Music Tokenization

Train a music tokenization model from scratch:

```
python train_tokenizer.py config/tokenization.yaml
```

Test the reconstruction quality with music pieces in the test sets:

```
python test_reconstruction.py config/tokenization.yaml ckpt/best_tokenizer/model.pt samples/reconstruction 20
```

### Hierarchical music generation

1. Encode REMI sequences to RVQ tokens offline with data augmentation. Skip this step if you would like to use the tokens encoded with provided tokenizer weights for training **and** have downloaded the datasets in the `Data Preparation` step. 

```
python remi2tokens.py config/remi2tokens.yaml ckpt/best_tokenizer/model.pt
```

2. Train a music generation model with learn tokens.

```
python train_generator.py config/generation.yaml
```

3. Generate music pieces with new checkpoints.

```
python test_generation.py \
        --configuration=config/generation.yaml \
        --model=ckpt/best_generator/model.pt \  # change the checkpoints here
        --use_prompt \
        --primer_n_bar=4 \
        --n_pieces=20 \
        --output_dir=samples/generation
```
