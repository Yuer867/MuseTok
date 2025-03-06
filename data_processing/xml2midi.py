from music21 import converter, midi
from glob import glob
import multiprocessing
import os
from tqdm import tqdm
from utils import pickle_dump
import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ProcessPoolExecutor

def xml_to_midi(file):
    
    midi_dir = os.path.dirname(file).replace('data_mxl', 'data_midi')
    os.makedirs(midi_dir, exist_ok=True)
    
    file_name = os.path.basename(file).split('.')[0]
    midi_path = os.path.join(midi_dir, file_name + '.mid')
    if os.path.exists(midi_path):
        return 0, None

    try:
        score = converter.parse(file)
        
        # Add each part to the combined MIDI stream
        combined_midi = midi.MidiFile()
        for part in score.parts:
            mf = midi.translate.streamToMidiFile(part)
            for track in mf.tracks:
                combined_midi.tracks.append(track)
        combined_midi.open(midi_path, 'wb')
        combined_midi.write()
        combined_midi.close()
        
        return 0, None
    except Exception as e:
        # print(f"Error processing {f}: {e}")
        return 1, file


if __name__ == "__main__":
    files = glob('/deepfreeze/jingyue/data/PDMX/data_mxl/*/*/*.mxl')
    print('# files:', len(files))

    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(xml_to_midi, files), desc='Preprocess', total=len(files)))
        
    bad_files = sum([i[0] for i in results])
    discards = round(100*bad_files / float(len(files)),2)
    print(f'Successfully processed {len(files) - bad_files} / {len(files)} files (discarded {discards}%)')
    
    error_files = []
    for i in results:
        if results[i][1] is not None:
            error_files.append(results[i][1])
    pickle_dump(error_files, '/deepfreeze/jingyue/data/PDMX/error_files.pkl')