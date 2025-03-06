import numpy as np
from glob import glob
from tqdm import tqdm
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

from remiplus.input_representation import InputRepresentation, remi2midi
from remiplus.vocab import RemiVocab, Tokens

def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
def pickle_load(f):
    return pickle.load(open(f, 'rb'))

def midi2events(file):
    try:
        vocab = RemiVocab()
        rep = InputRepresentation(file, strict=True, do_extract_chords=False)
        events = rep.get_remi_events()
        seq = vocab.encode(events)
        
        unk = np.where(np.array(seq) == 1)[0]
        if len(unk) > 0:
            for i in unk:
                if 'Time Signature' in events[i]:
                    print(events[i])
                    return 1
        
        new_events = []
        for i in range(len(events)):
            if ('Position' in events[i] and 'Tempo' in events[i+1]) or ('Tempo' in events[i]):
                continue
            if 'Time Signature' in events[i] or 'Velocity' in events[i]:
                continue
            else:
                new_events.append(events[i])
        new_events.append('EOS_None')
        
        new_events = [{'name': e.split('_')[0], 'value': e.split('_')[1]} for e in new_events]
        bar_pos = [i for i, v in enumerate(new_events) if v['name'] == 'Bar']
        num_notes = sum([1 for i in new_events if i['name'] == 'Pitch'])

        # data_name = '_'.join(file.split('/')[-2:]).split('.')[0]
        # print(data_name)
        data_name = os.path.basename(file).split('.')[0]
        output_path = os.path.join(output_dir, data_name + '.pkl')
        pickle_dump((bar_pos, new_events), output_path)
        
    except Exception:
        return 1, 0, 0

    return 0, len(bar_pos), num_notes


def test_remi2midi(file, out_path):
    _, events = pickle_load(file)
    events = ['{}_{}'.format(e['name'], str(e['value'])) for e in events]
    pm = remi2midi(events, bpm=110, has_velocity=True)
    pm.write(out_path)
    

if __name__ == "__main__":
    # EMOPIA
    midi_dir = '/deepfreeze/jingyue/data/EMOPIA/midi_quantized_480'
    output_dir = '/deepfreeze/jingyue/data/EMOPIA/data_events_figaro'
    files = glob(os.path.join(midi_dir, '*.mid'), recursive=True)
    
    # Pop1k7
    # midi_dir = '/deepfreeze/jingyue/data/Pop1k7/midi_synchronized'
    # output_dir = '/deepfreeze/jingyue/data/Pop1k7/data_events'
    # files = glob(os.path.join(midi_dir, '*/*.mid'), recursive=True)
    
    # POP909
    # midi_dir = '/deepfreeze/jingyue/data/POP909/data_midi'
    # output_dir = '/deepfreeze/jingyue/data/POP909/data_events'
    # files = glob(os.path.join(midi_dir, '*.mid'), recursive=True)
    
    # MAESTRO
    # midi_dir = '/deepfreeze/jingyue/data/MAESTRO/maestro-v3.0.0'
    # output_dir = '/deepfreeze/jingyue/data/MAESTRO/data_events'
    # files = glob(os.path.join(midi_dir, '*/*.midi'), recursive=True)
    
    # Pianist8
    # midi_dir = '/deepfreeze/jingyue/data/Pianist8/midi'
    # output_dir = '/deepfreeze/jingyue/data/Pianist8/data_events'
    # files = glob(os.path.join(midi_dir, '*/*.mid'), recursive=True)
    
    # file = os.path.join(output_dir, os.listdir(output_dir)[12])
    # print(file)
    # filename = os.path.basename(file)
    # test_remi2midi(file, os.path.join('/deepfreeze/jingyue/data/tmp', filename.replace('.pkl', '_test.mid')))
    # input()
    
    os.makedirs(output_dir, exist_ok=True)
    print('# midi files', len(files))

    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(midi2events, files), desc='Preprocess', total=len(files)))
        
    bad_files = sum([i[0] for i in results])
    discards = round(100*bad_files / float(len(files)),2)
    print(f'Successfully processed {len(files) - bad_files} files (discarded {discards}%)')
    
    avg_bar = sum([i[1] for i in results]) / (len(files) - bad_files)
    print('Average number of bars is', avg_bar)
    
    ave_note = sum([i[2] for i in results]) / (len(files) - bad_files)
    print('Average number of notes is', ave_note)
