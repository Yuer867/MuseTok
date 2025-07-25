# Data Preparation

This folder provides the scripts of converting midi data into REMI event sequences, building vocabulary and splitting training / validation / test sets. 

Alternatively, you can download processed datasets in [link] for quickly training and inference.

## Convert MIDI to events

```
python data_processing/midi2events.py
```

## Build Vocabulary

```
python data_processing/events2words.py
```

## Data Splits

```
python data_processing/data_split.py
```