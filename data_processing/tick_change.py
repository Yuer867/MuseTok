import mido
import pretty_midi
import os
from tqdm import tqdm
import miditoolkit
from miditoolkit.midi import containers
from copy import deepcopy
import shutil

BEAT_RESOL = 480

def tick_scaling(obj, orig_ticks, is_note=False):
    new_obj = deepcopy(obj)
    if is_note:
        new_obj.start = int(obj.start / orig_ticks * BEAT_RESOL)
        duration = obj.end - obj.start
        new_duration = int(duration / orig_ticks * BEAT_RESOL)
        new_obj.end = new_obj.start + new_duration
    else:
        new_obj.time = int(obj.time / orig_ticks * BEAT_RESOL)
    return new_obj


if __name__ == '__main__':
    """
    change the value of ticks_per_beat (default: 480)
    """
    data_home = '/deepfreeze/jingyue/data/Ragtime-perfect-Jazz/midi_quantized'
    output_dir = '/deepfreeze/jingyue/data/Ragtime-perfect-Jazz/midi_quantized_480'
    midi_files = os.listdir(data_home)
    os.makedirs(output_dir, exist_ok=True)
    num_copy = 0

    for file in tqdm(midi_files):
        filename = file.split('.')[0]
        midi_path = os.path.join(data_home, filename + '.mid')
        output_path = os.path.join(output_dir, filename + '.mid')
        
        orig_midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
        orig_ticks = orig_midi_obj.ticks_per_beat
        if orig_ticks == BEAT_RESOL:
            shutil.copy(midi_path, output_path)
            num_copy += 1
        else:
            new_midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=BEAT_RESOL)
            
            orig_time_changes = orig_midi_obj.time_signature_changes
            # print(orig_time_changes)
            new_midi_obj.time_signature_changes = [tick_scaling(i, orig_ticks) for i in orig_time_changes]
            # print(new_midi_obj.time_signature_changes)
            
            orig_tempo_changes = orig_midi_obj.tempo_changes
            # print(orig_tempo_changes)
            new_midi_obj.tempo_changes = [tick_scaling(i, orig_ticks) for i in orig_tempo_changes]
            # print(new_midi_obj.tempo_changes)
            
            notes = []
            new_midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='piano'))
            for track in orig_midi_obj.instruments:
                # print(track.notes[:10])
                notes += [tick_scaling(note, orig_ticks, is_note=True) for note in track.notes]
                # print(notes[:10])
            notes = sorted(notes, key=lambda x: (x.start, x.pitch))
            new_midi_obj.instruments[0].notes = notes
        
            assert new_midi_obj.ticks_per_beat == 480, f"{new_midi_obj.ticks_per_beat}"
            new_midi_obj.dump(output_path)
            
    print(num_copy)