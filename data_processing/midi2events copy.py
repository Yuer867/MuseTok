import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import collections
from concurrent.futures import ProcessPoolExecutor

import miditoolkit
from miditoolkit.midi import containers
import copy

# ================================================== #
#  Configuration                                     #
# ================================================== #  
BEAT_RESOL = 480
# BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 12
TRIPLET_RESOL = BEAT_RESOL // 24
INSTR_NAME_MAP = {'piano': 0}
MIN_VELOCITY = 40
NOTE_SORTING = 1  # 0: ascending / 1: descending

DEFAULT_TEMPO = 110
DEFAULT_VELOCITY_BINS = np.linspace(4, 127, 42, dtype=int)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
# DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60 + 1, dtype=int)
DEFAULT_SHIFT_BINS = np.linspace(-TICK_RESOL, TICK_RESOL, TICK_RESOL + 1, dtype=int)
# DEFAULT_DURATION_BINS = np.arange(BEAT_RESOL / 8, BEAT_RESOL * 8 + 1, BEAT_RESOL / 8)
# DEFAULT_DURATION_BINS = np.arange(BEAT_RESOL / 12, BEAT_RESOL * 8 + 1, BEAT_RESOL / 12)


class NoteEvent(object):
    def __init__(self, start, end, pitch, velocity, bar_resol, default_onset):
        self.start = start
        self.quantized_start = start // bar_resol * bar_resol + default_onset[np.argmin(abs(default_onset - start % bar_resol))]
        self.is_valid_start = self.quantized_start % (TICK_RESOL * 3) == 0
        self.is_triplet_candidate = self.quantized_start % (TRIPLET_RESOL * 4) == 0
        self.is_triplet = None
        self.end = end
        self.pitch = pitch
        self.duration = None
        self.velocity = velocity
        
    def __repr__(self):
        return f'Note(quantized_start={self.quantized_start:d}, end={self.end:d}, pitch={self.pitch}, velocity={self.velocity})'


def check_triplet(start, quantized_timing, bar_resol):
    bar_idx = start // bar_resol
    if (start + TRIPLET_RESOL * 4) in quantized_timing and \
        (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 12) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 20) in quantized_timing and \
        (start + TRIPLET_RESOL * 20) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 4, start + TRIPLET_RESOL * 8, 
                    start + TRIPLET_RESOL * 12, start + TRIPLET_RESOL * 16, start + TRIPLET_RESOL * 20]
    elif (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 24) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 40) in quantized_timing and \
        (start + TRIPLET_RESOL * 40) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 8, start + TRIPLET_RESOL * 16, 
                    start + TRIPLET_RESOL * 24, start + TRIPLET_RESOL * 32, start + TRIPLET_RESOL * 40]
    elif (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 48) in quantized_timing and \
        (start + TRIPLET_RESOL * 64) in quantized_timing and \
        (start + TRIPLET_RESOL * 80) in quantized_timing and \
        (start + TRIPLET_RESOL * 80) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 16, start + TRIPLET_RESOL * 32, 
                    start + TRIPLET_RESOL * 48, start + TRIPLET_RESOL * 64, start + TRIPLET_RESOL * 80]
    elif (start + TRIPLET_RESOL * 4) in quantized_timing and \
        (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 8) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 4, start + TRIPLET_RESOL * 8]
    elif (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 8, start + TRIPLET_RESOL * 16]
    elif (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 16, start + TRIPLET_RESOL * 32]
    elif (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 64) in quantized_timing and \
        (start + TRIPLET_RESOL * 64) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 32, start + TRIPLET_RESOL * 64]
    else:
        return False



def analyzer(midi_path):
    """
    get melody and chord(marker) tracks for lead sheet
    """
    # load midi obj
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    assert midi_obj.ticks_per_beat == 480, f"{midi_obj.ticks_per_beat}"
    notes = []
    for instrument in midi_obj.instruments:
        notes += instrument.notes
    max_tick = midi_obj.max_tick
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    # assert len(midi_obj.time_signature_changes) == 1, midi_obj.time_signature_changes
    time_sig = midi_obj.time_signature_changes[0]
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
    bar_resol = int(BEAT_RESOL * quarters_per_bar)

    # new midi obj
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.time_signature_changes.append(containers.TimeSignature(numerator=time_sig.numerator, denominator=time_sig.denominator, time=0))
    new_midi_obj.tempo_changes.append(containers.TempoChange(tempo=float(DEFAULT_TEMPO), time=0))
    new_midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='piano'))
    new_midi_obj.ticks_per_beat = BEAT_RESOL
    new_midi_obj.instruments[0].notes = notes

    # --- global tempo --- #
    tempos = [b.tempo for b in midi_obj.tempo_changes][:40]
    tempo_median = np.median(tempos)
    global_bpm = int(tempo_median)
    new_midi_obj.markers.insert(0, containers.Marker(text='global_bpm_' + str(int(global_bpm)), time=0))

    # save
    new_midi_obj.instruments[0].name = 'piano'
    return new_midi_obj, bar_resol


def midi2corpus_strict(midi_obj, bar_resol, remove_overlap=True):
    """
    quantize midi data
    """ 
    new_midi_obj = copy.deepcopy(midi_obj)

    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # skip 
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx = instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    # load global bpm
    global_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
                marker.text.split('_')[1] == 'bpm':
            global_bpm = int(marker.text.split('_')[2])
    
    # --- step 1: adjust onset values --- #
    
    # valid onset values (triplet 80*n or non-triplet 120*n)
    default_onset = np.unique(np.concat([np.arange(0, bar_resol, TRIPLET_RESOL*4), np.arange(0, bar_resol, TICK_RESOL*3), np.array([bar_resol])]))
    default_normal_onset = np.unique(np.concat([np.arange(0, bar_resol, TICK_RESOL*3), np.array([bar_resol])])) # non-triplet
    
    instr_quantized_notes = collections.defaultdict(list)
    instr_quantized_timing = collections.defaultdict(list)
    for key in instr_notes.keys():
        # quantize onsets of all notes to 60*n or 80*n ticks
        notes = instr_notes[key]
        for note in notes:
            quantized_note = NoteEvent(note.start, note.end, note.pitch, note.velocity, bar_resol, default_onset)
            instr_quantized_notes[key].append(quantized_note)
            instr_quantized_timing[key].append(quantized_note.quantized_start)
        
        # keep onsets of normal notes or valid triplets, adjust onsets of invalid triplet
        valid_timing = []
        for quantized_note in instr_quantized_notes[key]:
            # if not triplet notes or already belong to some triplets
            if not quantized_note.is_triplet_candidate:
                quantized_note.is_triplet = False
                continue
            else:
                if quantized_note.quantized_start in valid_timing:
                    quantized_note.is_triplet = True
                    continue
                if check_triplet(quantized_note.quantized_start, instr_quantized_timing[key], bar_resol):
                    valid_timing.append(quantized_note.quantized_start)
                    valid_timing += check_triplet(quantized_note.quantized_start, instr_quantized_timing[key], bar_resol)
                    quantized_note.is_triplet = True
                else:
                    quantized_note.quantized_start = quantized_note.start // bar_resol * bar_resol + \
                        default_normal_onset[np.argmin(abs(default_normal_onset - quantized_note.start % bar_resol))]
                    quantized_note.is_triplet = False
        
        # --- step 2: adjust duration --- #
        if TICK_RESOL == 20:
            default_normal_duration = np.array([BEAT_RESOL // 8,                   # 1/32       - 60
                                                BEAT_RESOL // 4,                   # 1/16       - 120
                                                BEAT_RESOL // 4 + BEAT_RESOL // 8, # 1/16 + 1/32 - 180
                                                BEAT_RESOL // 2,                   # 1/8        - 240
                                                BEAT_RESOL // 2 + BEAT_RESOL // 4, # 1/8 + 1/16 - 360
                                                BEAT_RESOL,                        # 1/4        - 480
                                                BEAT_RESOL + BEAT_RESOL // 2,      # 1/4 + 1/8  - 720
                                                2 * BEAT_RESOL,                    # 1/2        - 960
                                                2 * BEAT_RESOL + BEAT_RESOL,       # 1/2 + 1/4  - 1440
                                                4 * BEAT_RESOL])                   # 1          - 1920
        elif TICK_RESOL == 40:
            default_normal_duration = np.array([BEAT_RESOL // 4,                   # 1/16       - 120
                                                BEAT_RESOL // 2,                   # 1/8        - 240
                                                BEAT_RESOL // 2 + BEAT_RESOL // 4, # 1/8 + 1/16 - 360
                                                BEAT_RESOL,                        # 1/4        - 480
                                                BEAT_RESOL + BEAT_RESOL // 2,      # 1/4 + 1/8  - 720
                                                2 * BEAT_RESOL,                    # 1/2        - 960
                                                2 * BEAT_RESOL + BEAT_RESOL,       # 1/2 + 1/4  - 1440
                                                4 * BEAT_RESOL])                   # 1          - 1920
        else:
            raise ValueError('invalid tick resolution {}'.format(TICK_RESOL))
        
        default_triplet_duration = np.array([BEAT_RESOL // 6,                  # 1/8 // 3   - 80
                                            BEAT_RESOL // 3,                   # 1/4 // 3   - 160
                                            2 * BEAT_RESOL // 3,               # 1/2 // 3   - 320
                                            4 * BEAT_RESOL // 3])              # 1 // 3     - 640
        
        default_duration = np.concat([default_normal_duration, default_triplet_duration])
        for quantized_note in instr_quantized_notes[key]:
            assert quantized_note.is_triplet is not None
            if quantized_note.is_triplet:
                quantized_note.duration = default_duration[np.argmin(abs(default_duration - (quantized_note.end - quantized_note.quantized_start)))]
            if not quantized_note.is_triplet:
                quantized_note.duration = default_normal_duration[np.argmin(abs(default_normal_duration - (quantized_note.end - quantized_note.quantized_start)))]
            quantized_note.end = quantized_note.quantized_start + quantized_note.duration
        
        # --- step 3: remove note overlap --- #
        if remove_overlap:
            # remove all overlap between two consecutive notes if
            #   - their ends is not the same
            #   - or one note is not covered by another
            onsets2notes = collections.defaultdict(list)
            onsets2ends = collections.defaultdict(int)
            for quantized_note in instr_quantized_notes[key]:
                onsets2notes[int(quantized_note.quantized_start)].append(quantized_note)
                onsets2ends[int(quantized_note.quantized_start)] = max(onsets2ends[int(quantized_note.quantized_start)], quantized_note.end)
            onsets2notes = sorted(onsets2notes.items(), key=lambda x: x[0])
            onsets2ends = sorted(onsets2ends.items(), key=lambda x: x[0])
            
            for i in range(len(onsets2notes[:-1])):
                for quantized_note in onsets2notes[i][1]:
                    j = i + 1
                    while j < len(onsets2notes[:-1]) and quantized_note.end > onsets2ends[j][0]:
                        if quantized_note.end >= onsets2ends[j][1]:
                            j += 1
                            continue
                        else:
                            if onsets2ends[j][0] - quantized_note.quantized_start in default_duration:
                                quantized_note.end = onsets2ends[j][0]
                            break
        else:
            # remove only the overlap between two notes with the same pitch
            note2onsets = collections.defaultdict(list)
            for quantized_note in instr_quantized_notes[key]:
                note2onsets[int(quantized_note.pitch)].append(quantized_note)
            for quantized_note in instr_quantized_notes[key]:
                larger_note_start = [note.quantized_start for note in note2onsets[int(quantized_note.pitch)] if note.quantized_start > quantized_note.quantized_start]
                if len(larger_note_start) > 0:
                    closet_note_start = np.min(larger_note_start)
                    if quantized_note.end > closet_note_start and closet_note_start - quantized_note.quantized_start >= TICK_RESOL * 2:
                        duration_diff = default_duration - (closet_note_start - quantized_note.quantized_start)
                        quantized_note.duration = default_duration[np.where(duration_diff <= 0)[0][np.argmax(duration_diff[duration_diff <= 0])]]
                        quantized_note.end = quantized_note.quantized_start + quantized_note.duration
    
        new_notes = []
        for quantized_note in instr_quantized_notes[key]:
            new_note = containers.Note(quantized_note.velocity, quantized_note.pitch, quantized_note.quantized_start, quantized_note.end)
            new_notes.append(new_note)
        assert len(new_notes) ==  len(notes)
        instr_notes[key] = new_notes
    
    for key in instr_notes.keys():
        new_midi_obj.instruments[key].notes = instr_notes[key]

    return new_midi_obj


def midi2events(file):
    try:
        filename = os.path.basename(file)
        output_path = os.path.join(quantized_midi_dir, filename)
        
        midi_obj, bar_resol = analyzer(file)
        new_midi_obj = midi2corpus_strict(midi_obj, bar_resol, 
                                        remove_overlap=remove_overlap)
        new_midi_obj.dump(output_path)
    except Exception:
        return 1

    return 0


if __name__ == '__main__':
    # two arguments that can be changed
    remove_overlap = False
    TICK_RESOL = BEAT_RESOL // 12

    midi_dir = '/classical_by_composer'
    quantized_midi_dir = '/classical_by_composer_quantized'
    midi_files = glob(os.path.join(midi_dir, '*.mid'), recursive=True)
    
    os.makedirs(quantized_midi_dir, exist_ok=True)
    print('# midi files', len(midi_files))

    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(midi2events, midi_files, chunksize=16), desc='Preprocess', total=len(midi_files)))
    
    bad_files = sum(results)
    discards = round(100*bad_files / float(len(midi_files)),2)
    print(f'Successfully processed {len(midi_files) - bad_files} files (discarded {discards}%)')