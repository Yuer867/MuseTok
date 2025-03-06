# midi quantizer
# quantize the midi to the fix bpm (i.e. 120)
import os
import mido
from glob import glob
import pretty_midi as pyd
import numpy as np
from tqdm import tqdm

class MidiQuantizer:

    '''
    bpm: the bpm to fix
    threshold: two thresholds to half or double the tempo to balance the final quantized tracks
    '''
    def __init__(self, bpm = 120, threshold = [60, 200]):
        self.bpm = bpm
        self.beat_time = 60.0 / bpm
        self.threshold = threshold
    
    '''
    process the midi file to quantize
    midi_file: the processing file
    output_file: output file address
    '''
    def process(self, midi_file, output_file, resolution=480):
        try:
            x = mido.MidiFile(midi_file)
            m_midi = pyd.PrettyMIDI(initial_tempo=120, resolution=resolution)
        except:
            return
        threshold = self.threshold
        beat_time = self.beat_time
        tpb = x.ticks_per_beat

        t_sign = []
        k_sign = []
        ins_notes = []
        tempos = []
        track_names = []

        print(midi_file + ":")
        for i, track in enumerate(x.tracks):
            acc_ticks = 0
            error_on = 0
            error_off = 0
            notes = []
            dict_note = {}
            track_name = ""
            for j in range(130):
                dict_note[j] = -1
            for msg in track:
                acc_ticks += msg.time
                if msg.type == "track_name":
                    track_name = msg.name
                if msg.type == "time_signature":
                    t_sign.append([acc_ticks, msg.numerator, msg.denominator])
                if msg.type == "key_signature":
                    k_sign.append([acc_ticks, msg.key])
                if msg.type == "set_tempo":
                    if mido.tempo2bpm(msg.tempo) < 300:
                        tempos.append(msg.tempo)
                if msg.type == "note_on":
                    if msg.velocity > 0:
                        if dict_note[msg.note] == -1:
                            dict_note[msg.note] = [acc_ticks, msg.velocity]
                        else:
                            error_on += 1
                            # print("find a note does not end but a same-pitch note appears")
                    else:
                        if dict_note[msg.note] != -1:
                            notes.append([dict_note[msg.note][0], acc_ticks, msg.note, dict_note[msg.note][1]])
                            dict_note[msg.note] = -1
                if msg.type == "note_off":
                    if dict_note[msg.note] != -1:
                        notes.append([dict_note[msg.note][0], acc_ticks, msg.note, dict_note[msg.note][1]])
                        dict_note[msg.note] = -1
                    else:
                        error_off += 1
                    
            if len(notes) > 0:
                print("%d notes in %d track, error-on: %d, error-off: %d" %(len(notes), i, error_on, error_off))
                ins_notes.append(notes)
                track_names.append(track_name)

        # initial changes
        m_midi.time_signature_changes = []
        m_midi.key_signature_changes = []    

        if len(tempos) == 0:
            print("no tempo")
            tempo = mido.bpm2tempo(120)
            # return
        else:
            tempo = sum(tempos) / len(tempos)
        if mido.tempo2bpm(tempo) <= 60:
            beat_time *= 2.0
        if mido.tempo2bpm(tempo) >= 200:
            beat_time /= 2.0


        for ts in t_sign:
            tick, num, denom = ts
            tick = tick / tpb * beat_time
            m_midi.time_signature_changes.append(pyd.TimeSignature(num, denom, tick))

        for ks in k_sign:
            tick, key = ks
            tick = tick / tpb * beat_time
            key = pyd.key_name_to_key_number(key)
            m_midi.key_signature_changes.append(pyd.KeySignature(key, tick))

        for i, notes in enumerate(ins_notes):
            ins_name = track_names[i] if track_names[i] != "" else "Piano_" + str(i)
            new_piano = pyd.Instrument(program=0, name = ins_name)
            new_piano.notes = []
            for note in notes:
                sta, end, pitch, vel = note
                sta = sta / tpb * beat_time
                end = end / tpb * beat_time
                new_piano.notes.append(pyd.Note(vel, pitch, sta, end))
            m_midi.instruments.append(new_piano)
        m_midi.write(output_file)
        print("*****end*****")
        
        
if __name__ == '__main__':
    # EMOPIA
    # dataset_path = '/deepfreeze/jingyue/data/EMOPIA/midi_synchronized'
    # output_path = '/deepfreeze/jingyue/data/EMOPIA/midi_quantized_480'
    # midi_files = glob(os.path.join(dataset_path, '*.mid'), recursive=True)
    
    # Pop1k7
    # dataset_path = '/deepfreeze/jingyue/data/Pop1k7/midi_synchronized'
    # output_dir = '/deepfreeze/jingyue/data/Pop1k7/midi_quantized_480'
    # midi_files = glob(os.path.join(dataset_path, '*/*.mid'), recursive=True)
    
    # POP909
    midi_files = glob('/deepfreeze/jingyue/data/POP909/POP909/*/*.mid')
    output_dir = '/deepfreeze/jingyue/data/POP909/midi_quantized_480'
    
    os.makedirs(output_dir, exist_ok=True)

    processor = MidiQuantizer(bpm=120, threshold=[60,200])
    for file in tqdm(midi_files):
        data_name = os.path.basename(file)
        # data_name = '_'.join(file.split('/')[-2:])
        output_path = os.path.join(output_dir, data_name)
        processor.process(file, output_path, resolution=480)
        assert mido.MidiFile(output_path).ticks_per_beat == 480
    