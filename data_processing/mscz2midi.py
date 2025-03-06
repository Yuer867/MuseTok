import os
from tqdm import tqdm
from glob import glob
import multiprocessing

# mscz_files = glob('data_mscz_254k/*/*/*.mscz')
# print(len(mscz_files))
# error_files = 0

# for file in tqdm(mscz_files):
#     input_path = 'data_mscz_254k/' + file
#     output_path = 'data_midi_254k/' + file.replace('.mscz', '.mid')
#     os.makedirs(file[:3], exist_ok=True)
    
    # try: 
    #     os.system('/Applications/MuseScore\ 4.app/Contents/MacOS/mscore -o {} {}'.format(output_path, input_path))
    # except:
    #     error_files += 1
    
def mscz_to_midi(sub_dir_path):
    all_score = 0
    good_score = 0
    bad_score = 0

    error_list = []
    
    for sub_sub_dir in tqdm(os.listdir(sub_dir_path)):
        sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)

        sub_sub_midi_path = sub_sub_dir_path.replace('mscz_254k', 'midi_254k')
        os.makedirs(sub_sub_midi_path, exist_ok=True)

        for f in os.listdir(sub_sub_dir_path)[:2]:
            if '.mscz' not in f:
                continue
            all_score += 1

            score_name = f.split('.')[0]
            score_path = os.path.join(sub_sub_dir_path, f)
            midi_path = os.path.join(sub_sub_midi_path, score_name + '.mid')

            try:
                os.system("/Applications/MuseScore\ 4.app/Contents/MacOS/mscore -o '{}' '{}'".format(midi_path, score_path))
            except Exception as e:
                error_list.append(e)
                bad_score += 1
                continue

            good_score += 1
            
    return all_score, good_score, bad_score

if __name__ == '__main__':
    dir_path = 'data_mscz_254k'
    sub_dir_list = [os.path.join(dir_path, sub_dir) for sub_dir in os.listdir(dir_path)]
    print(sub_dir_list)
    with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
        results = list(pool.map(func = mscz_to_midi, iterable = tqdm(iterable = sub_dir_list, desc = f"json2midi", total = len(sub_dir_list)), chunksize = 1))

    print("=========== Final Results ===========")
    total_score = sum([i[0] for i in results])
    total_good_score = sum([i[1] for i in results])
    total_bad_score = sum([i[2] for i in results])
    print('Good: {}/{}'.format(total_good_score, total_score), total_good_score / total_score)
    print('Bad: {}/{}'.format(total_bad_score, total_score), total_bad_score / total_score)