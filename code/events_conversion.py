# This script contains functions that convert psychopy log files into a single CSV.

# target format is similar to this https://gin.g-node.org/lnnrtwttkhn/highspeed-bids/src/master/sub-01/ses-01/func/sub-01_ses-01_task-highspeed_rec-prenorm_run-03_events.tsv
import pandas as pd
import datetime
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict
# import psychopy  # you should be running python 3.8 or 3.10
# from psychopy.misc import fromFile
# assert psychopy.__version__.startswith('2024.2'), f'psychopy needs to be version 2024.2 to load the files but {psychopy.__version__=}'

def extract_datetime(file):
    """extract datetime from psychopy timestamped filenames"""
    date, time = file.split('_')[-2:]  # Only get the last two parts for date-time
    year, month, day = map(int, date.split('-'))
    hour, minute_second_micro = time.split('h')
    minute, second, *_ = minute_second_micro.split('.')
    return datetime.datetime(year, month, day, int(hour), int(minute), int(second))

def add_row(df, **new_row):
    """add a row to a dataframe, inplace with varying arguments.
    all keywords that don't exist will be be added as columns to the df

    Parameters
    ----------
    df : dataframe to add a row to
    onset : only required argument
    kwargs: keyword argument to add as columns

    Returns None
    """
    # make sure these are present
    assert 'onset' in new_row
    assert 'condition' in new_row
    if 'duration' in new_row:
        new_row['duration'] = round(new_row['duration'], 4)
    # if column doesnt exist, create it
    for name in new_row:
        if name not in df.columns:
            df[name]= [np.nan]*len(df)
    df.loc[len(df)] = new_row
    return None  # it works in-place, so no return of df

def json_decode(cell):
    """convert strings to lists , e..g "[1,2]" => [1, 2]"""
    try:
        return json.loads(cell)
    except (TypeError, ValueError):
        # print(f'not json: {cell}')
        return cell  # Return the cell unchanged if it's not a JSON string



#%% actual conversion code
output_dir = 'Z:/Fast-Replay-7T/output/'
psychopy_data = 'Z:/Fast-Replay-7T/data_cleaned' # directory where psychopy log files for the experiment are stored

# some definitions
intervals = np.array([32, 64, 128, 512])
stimuli = ['gesicht', 'haus', 'katze', 'schuh', 'stuhl']
key_mapping = {'r': 'right', 'y': 'down', 'b': 'right', 'g': 'left'}


def convert_psychopy_to_bids(csv_file):
    # load the log file (only there we have the image labels, what?!)
    with open(csv_file[:-3] + 'log', 'r') as f:
        loglines = f.readlines()
        # build lookup dictionary
        # this is increasingly ugly, but it creates a nested dict of a nested dict
        log_dict = defaultdict(lambda: defaultdict(dict))
        for line in loglines:
            t, lvl, msg = [x.strip() for x in line.split('\t', 3)]
            if msg.startswith('Created'): continue  # ignore creation events
            if not ' = ' in msg: continue  # only record property assignments
            if not ': ' in msg: continue  # only record property assignments
            if lvl=='DATA': continue  # ignore keypresses
            cmp_name, msg = msg.split(': ', 1)
            prop, val = msg.split(' = ', 1)
            log_dict[float(t)][cmp_name][prop] = val

    # load the CSV file
    df_run = pd.read_csv(csv_file)

    df_run_bids = pd.DataFrame(columns=['onset', 'duration', 'subject', 'session', 'condition', 'trial_type'])

    # loop over all trials

    for i, line in tqdm(df_run.iterrows(), total=len(df_run), desc='extracting events'):
        line = line[~line.isna()]  # filter all NaN entries, makes it a bit easier to debug
        line = line.apply(json_decode)  # convert strings to python datatypes

        condition = 'other'  # default condition
        # parse which trial this currently is
        if 'language_selection_screen.started' in line:
            condition = 'instruction'
            component = 'language_selection_screen'
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'instruction',
                    stim_label='language_selection',
                    key_down='german' if line['choice_key.keys']=='g' else 'english',
                    )
            continue

        # ignore instructions
        elif 'instruct_pre1.started' in line:
            continue

        elif 'instruct_pre2.started' in line:
            continue

        elif 'localizer.started' in line:  # this row is a localizer trial
            condition = 'localizer'

            # there are several events within this trial that we need to transfer
            # 1 fixation dot pre, only once before the entire block
            #### somehow missing

            # 2 fixation dot pre, before every image
            # component = 'localizer_fixation'
            if f'{component}.stopped' in line and f'{component}.started' in line:
                add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'fixation',
                    stim_label='dot')

            # 3 image itself
            component = 'localizer_img'
            onset = line[f'{component}.started']
            orientation = log_dict[round(onset, 4)][component]['ori']
            stim_label = log_dict[round(onset, 4)][component]['image'].split('/')[-1][:-5]
            stim_index = stimuli.index(stim_label.lower())+1

            add_row(df_run_bids,
                    onset=onset,
                    duration=line[f'{component}.stopped'] - onset,
                    condition=condition,
                    trial_type= 'stimulus',
                    stim_label= stim_label,
                    orientation=orientation,
                    interval_time=line[f'localizer_isi.stopped'] - line[f'localizer_isi.started'],
                    response_time=line['key_resp_localizer.rt'] if 'key_resp_localizer.rt' in line else np.nan,
                    accuracy= ('key_resp_localizer.rt' in line)==(orientation=='180')
                    )
                    # accuracy=
            del onset, orientation, stim_label, stim_index  # for safety, not to accidentially reuse it later

            # 4 ISI
            component = 'localizer_isi'
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'blank',
                    stim_label='interval')

            # 5 feedback, if given
            component = 'loc_feedback'
            if f'{component}.started' in line:
                onset = line[f'{component}.started']
                add_row(df_run_bids,
                        onset=onset,
                        duration=line[f'{component}.stopped'] - onset,
                        condition=condition,
                        trial_type= 'feedback',
                        stim_label=log_dict[round(onset, 4)][component]['foreColor'])
                del onset


        elif 'sequence.started' in line:
            condition = 'sequence'

            # 1 Cue of which item to look out for
            component = 'cue'
            onset = line[f'{component}_text.started']
            cue_label = log_dict[round(onset, 4)][component + '_text']['text'].replace("'", "")
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'cue',
                    stim_label=cue_label)
            del onset

            # 2 empty blank for 1500 ms
            component = 'blank1500'
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'blank',
                    stim_label='blank')

            # 3 fixation dot before sequence
            component = 'fixation_dot'
            add_row(df_run_bids,
                    onset= line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] -  line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'fixation',
                    stim_label='dot')

            # 4 sequences of five images
            onset_seq1 = round(line['sequence_img_1.started'], 4)
            correct_idx = -1  # save which press would have been correct
            for seq in range(1, 6):

                elapsed = (line[f'sequence_isi_{seq}.stopped'] - line[f'sequence_isi_{seq}.started'])*1000
                interval = intervals[np.argmin(abs(intervals - elapsed))]

                # 4.1 image
                component = f'sequence_img_{seq}'
                stim_label = log_dict[onset_seq1][component]['image'].split('/')[-1][:-5]
                stim_index = stimuli.index(stim_label.lower())+1
                if stim_label==cue_label:
                    assert correct_idx==-1  # make sure we're not overwriting something
                    correct_idx=seq

                add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'stimulus',
                    stim_index=stim_index,
                    stim_label=stim_label,
                    serial_position=seq,
                    interval_time=interval)
                del stim_label, stim_index  # for safety, remove

                # 4.2 ISI
                component = f'sequence_isi_{seq}'
                add_row(df_run_bids,
                        onset=line[f'{component}.started'],
                        duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                        condition=condition,
                        trial_type= 'interval',
                        stim_label= 'dot')
            assert correct_idx>0, 'no target stimulus found? error in code.'

            component = f'buffer_fixation'
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'delay',
                    stim_label= 'dot')

            component = f'question'
            t_feedback = round(line['text_feedback__answer.started'], 4)
            choices = log_dict[round(line[f'{component}_text.started'], 4)][f'{component}_text']['text']
            choices = [int(x.strip()) for x in choices.split('\\n')[-1][:-1].split('?')]

            key_expected = 'left' if choices.index(correct_idx)==0 else 'right'
            key_down = key_mapping[line['question_key_resp.keys']] if 'question_key_resp.rt' in line else np.nan

            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'choice',
                    stim_label= 'choice',
                    response_time=line['question_key_resp.rt'] if 'question_key_resp.rt' in line else np.nan,
                    key_down=key_down,
                    key_expected=key_expected,
                    key_pressed=bool('question_key_resp.rt' in line),
                    choice_left=choices[0],
                    choice_right=choices[1],
                    choice_correct=correct_idx,
                    accuracy=key_down==key_expected,
                    )

            component = f'feedback'
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'feedback',
                    )

        elif 'buffer_2.started' in line:
            condition = 'fixation'
            component = 'buffer_2'
            add_row(df_run_bids,
                    onset=line[f'{component}.started'],
                    duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                    condition=condition,
                    trial_type= 'pre-fixation',
                    stim_label= 'dot',
                    )
        elif 'break_2.started' in line:
            component = 'break_2'
            add_row(df_run_bids,
                onset=line[f'{component}.started'],
                duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                condition=condition,
                trial_type= 'break',
                stim_label= 'break',
                )

        elif 'instruct_end.started' in line:
            component = 'instruct_end'
            add_row(df_run_bids,
                onset=line[f'{component}.started'],
                duration=line[f'{component}.stopped'] - line[f'{component}.started'],
                condition=condition,
                trial_type= 'instruction',
                stim_label= 'end-of-experiment',
                )
        elif len(line)<=10:
            continue  # empty line

        else:
            raise ValueError('this is unknown')
    assert len(df_run_bids)>100, f'to few rows in {csv_file=}'
    return df_run_bids
