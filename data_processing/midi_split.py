import miditoolkit
import os
from tqdm import tqdm
import shutil
from miditoolkit.midi import containers
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

BEAT_RESOL = 480
MIN_BAR = 8


def split_midi(file):
    try:
        filename = os.path.basename(file).split('.')[0]
        file_subdir = '/'.join(file.split('/')[-3:-1])
        os.makedirs(os.path.join(output_dir, file_subdir), exist_ok=True)

        orig_midi_obj = miditoolkit.midi.parser.MidiFile(file)
        if len(orig_midi_obj.instruments) == 0:
            return '4'
        if len(orig_midi_obj.instruments) > 1:
            merge_tracks(orig_midi_obj)
        assert len(orig_midi_obj.instruments) == 1
        # assert len(orig_midi_obj.tempo_changes) == 1
        tempo = orig_midi_obj.tempo_changes[0]
        
        time_signatures = orig_midi_obj.time_signature_changes
        if len(time_signatures) == 0 or time_signatures[0].time != 0:
            orig_midi_obj.time_signature_changes.insert(0, containers.TimeSignature(numerator=4, denominator=4, time=0))
        
        if len(time_signatures) == 1:
            output_path = os.path.join(output_dir, file_subdir, filename + '.mid')
            assert len(orig_midi_obj.time_signature_changes) == 1
            orig_midi_obj.dump(output_path)
            return '0'
        else:
            merge_time_sig(orig_midi_obj)
            time_signatures = orig_midi_obj.time_signature_changes
            if len(time_signatures) == 1:
                output_path = os.path.join(output_dir, file_subdir, filename + '.mid')
                len(orig_midi_obj.time_signature_changes) == 1
                orig_midi_obj.dump(output_path)
                return '1'
            else:
                max_tick = orig_midi_obj.max_tick - 1
                segment_number = 0
                segments = 0
                for i in range(len(time_signatures)):
                    if i < len(time_signatures)-1:
                        seg_start = time_signatures[i].time
                        seg_end = time_signatures[i+1].time
                    else:
                        seg_start = time_signatures[i].time
                        seg_end = max_tick
                        
                    quarters_per_bar = 4 * time_signatures[i].numerator / time_signatures[i].denominator
                    if seg_end - seg_start > BEAT_RESOL * quarters_per_bar * (MIN_BAR - 1):
                        new_midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=BEAT_RESOL)
                        new_midi_obj.time_signature_changes.append(containers.TimeSignature(numerator=time_signatures[i].numerator, 
                                                                                            denominator=time_signatures[i].denominator, 
                                                                                            time=0))
                        new_midi_obj.tempo_changes.append(tempo)
                        
                        new_midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='piano'))
                        notes = [note for note in orig_midi_obj.instruments[0].notes if note.start >= seg_start and note.start < seg_end]
                        for note in notes:
                            note.start = note.start - seg_start
                            note.end = note.end - seg_start
                        new_midi_obj.instruments[0].notes = notes
                        
                        output_path = os.path.join(output_dir, file_subdir, filename + '_{}.mid'.format(segment_number))
                        assert len(new_midi_obj.time_signature_changes) == 1
                        new_midi_obj.dump(output_path)
                        segment_number += 1
                        segments += 1
                    else:
                        continue
                if segments == 0:
                    return '3'
                else:
                    return '2'
    except:
        print(file)
            

class TimeSig(object):
    def __init__(self, numerator, denominator, start, end):
        self.numerator = numerator
        self.denominator = denominator
        self.quarters_per_bar = 4 * numerator / denominator
        self.start = start
        self.end = end
        self.duration = end-start
        self.to_be_check = (end - start == BEAT_RESOL * self.quarters_per_bar)
        self.sig = '{}/{}'.format(numerator, denominator)


def merge_time_sig(midi_obj):
    """
    Merge time signatures of midi files
    """
    time_signatures = midi_obj.time_signature_changes
    max_tick = midi_obj.max_tick - 1
    note_offset = 0
    
    # build new time signature object to keep necessary information
    time_sigs = []
    for i in range(len(time_signatures)):
        if i < len(time_signatures)-1:
            seg_start = time_signatures[i].time
            seg_end = time_signatures[i+1].time
        else:
            seg_start = time_signatures[i].time
            seg_end = max_tick
        time_sigs.append(TimeSig(time_signatures[i].numerator, time_signatures[i].denominator, seg_start, seg_end))
    
    # iterate through time signature changes
    new_time_signatures = []
    last_time = None
    skip_next = False
    for i in range(len(time_sigs)):
        if time_sigs[i].to_be_check == True and time_sigs[i].start == 0:
            if time_sigs[i+1].to_be_check == False:
                note_offset = time_sigs[i+1].quarters_per_bar * BEAT_RESOL - time_sigs[i+1].start
                if note_offset < 0:
                    return False
                new_time_signatures.append(containers.TimeSignature(
                    numerator=time_sigs[i+1].numerator, denominator=time_sigs[i+1].denominator, time=0)
                )
                last_time = time_sigs[i+1].sig
            else:
                # bad midi file, skip
                return False
        elif time_sigs[i].to_be_check == True and time_sigs[i].start != 0:
            if skip_next or (i == len(time_sigs) - 1 and time_sigs[i-1].to_be_check == False):
                continue
            else:
                if time_sigs[i].sig != last_time:
                    new_time_signatures.append(containers.TimeSignature(
                        numerator=time_sigs[i].numerator, denominator=time_sigs[i].denominator, time=int(time_sigs[i].start + note_offset))
                    )
                    last_time = time_sigs[i].sig
        elif time_sigs[i].to_be_check == False:
            j = i + 1
            while j < len(time_sigs) and time_sigs[j].to_be_check != False:
                j += 1
                continue
            
            if j < len(time_sigs):
                if time_sigs[j].sig == time_sigs[i].sig and time_sigs[j].start - time_sigs[i].end == time_sigs[i].quarters_per_bar * BEAT_RESOL:
                    skip_next = True
                else:
                    skip_next = False
            else:
                skip_next = False
                
            if time_sigs[i].sig != last_time:
                new_time_signatures.append(containers.TimeSignature(
                    numerator=time_sigs[i].numerator, denominator=time_sigs[i].denominator, time=int(time_sigs[i].start + note_offset))
                )
                last_time = time_sigs[i].sig
    midi_obj.time_signature_changes = new_time_signatures
    
    if note_offset != 0:
        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                note.start = int(note.start + note_offset)
                note.end = int(note.end + note_offset)
        for tempo in midi_obj.tempo_changes:
            tempo.time = int(tempo.time + note_offset)
    
    return True
    
    
def merge_tracks(midi_obj):
    new_notes = []
    for instrument in midi_obj.instruments:
        new_notes += instrument.notes
    midi_obj.instruments = [containers.Instrument(program=0, is_drum=False, name='piano')]
    midi_obj.instruments[0].notes = new_notes
    
    
if __name__ == '__main__':
    """
    split midi files based on time signature (keep at least 8 bars)
    """
    data_home = '/deepfreeze/jingyue/data/PDMX/data_midi_piano'
    output_dir = '/deepfreeze/jingyue/data/PDMX/data_midi_piano_split'
    
    midi_files = glob('/deepfreeze/jingyue/data/PDMX/data_midi_piano/*/*/*.mid', recursive=True)
    os.makedirs(output_dir, exist_ok=True)
    
    
    for file in midi_files:
        if 'QmbL4zWgUpUHZBR2xakYBWSCrRHmb3tmHUC4RhGTfLLkRN.mid' in file:
            print(split_midi(file))
        else:
            continue
    input()
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(split_midi, midi_files), desc='Preprocess', total=len(midi_files)))
        
    print('number of files with single time signature:', results.count('0'))
    print('number of files with single time signature after merging:',results.count('1'))
    print('number of files split into segments after merging:', results.count('2'))
    print('number of files with no segments after merging:', results.count('3'))
    print('number of files with no tracks:', results.count('4'))