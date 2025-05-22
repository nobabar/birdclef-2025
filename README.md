# BirdCLEF 2025: Bird Song Classification

## Project Overview

This project focuses on the BirdCLEF 2025 competition, a machine learning challenge for bird song classification. The goal is to develop algorithms that can identify bird species from audio recordings, with applications in biodiversity monitoring and ecological research.

## Data Structure

The dataset consists of several components:

### Training Data

- **train_audio/**: Individual bird sound recordings
  - Clean, single-species recordings
  - Labeled with species information
  - Format: `.ogg` files at 32 kHz
  - Filenames: `[collection][file_id_in_collection].ogg`

- **train_soundscapes/**: Full 1-minute environmental recordings
  - Contains background noise and multiple species
  - Similar format to test data
  - Filenames: `[site]_[date]_[local_time].ogg`

### Metadata Files

- **train.csv**: Contains metadata for training recordings
  - `primary_label`: Species code
  - `secondary_labels`: Additional species in recording
  - `latitude` & `longitude`: Recording location
  - `author`: Recording provider
  - `rating`: Quality rating (1-5)
  - `collection`: Source collection (XC, iNat, or CSA)

- **taxonomy.csv**: Species information
  - iNaturalist taxon ID
  - Class name (Aves, Amphibia, Mammalia, Insecta)

## Audio Processing Pipeline

Our preprocessing pipeline follows the BirdNET paper approach:

1. **Spectrogram Generation**
   - Mel-spectrograms with 64 bands
   - Frequency range: 150 Hz to 15 kHz
   - FFT window size: ~32ms at 32kHz
   - 25% overlap between frames

2. **Data Augmentation**
   - Frequency shifts
   - Time shifts
   - Spectrogram warping
   - Ambient noise addition

3. **Signal Processing**
   - 3-second chunks
   - Signal strength-based detection
   - Log scaling for magnitude

## To test the project

create a uv environment

```
uv venv
```

then activate it

```
source .venv/bin/activate
```

install the necessary packages from the `pyproject.toml`

```
uv pip install -r pyproject.toml
```

and finally run the scripts

```
python preprocessing.py
python augmentation.py
python training.py
```

## References

- [BirdNET Paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)
- [BirdCLEF 2025 Competition](https://www.kaggle.com/competitions/birdclef-2025)

## License

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

Licensed under a [MIT License](https://opensource.org/license/mit).
