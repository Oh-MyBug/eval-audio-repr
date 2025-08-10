import os
import re
import yaml
import hashlib
import logging
import pandas as pd
import scipy.io.wavfile

from pathlib import Path
from easydict import EasyDict

# Folders
WORK = 'work'
METADATA_DIR = 'evar/metadata'
RESULT_DIR = 'results'
LOG_DIR = './MurmurGraphNet/logs'

_defs = {
    # folds, unit_sec, data_folder (None if task name is the folder name), balanced training when fine-tining
    'us8k': [10, 4.0, None, False],
    'esc50': [5, 5.0, None, False],
    'fsd50k': [1, 7.6358, None, False], ## Changed to NOT balanced: to make it the same as PaSST.
    'fsdnoisy18k': [1, 8.25, None, False],
    'gtzan': [1, 30.0, None, False],
    'nsynth': [1, 4.0, None, False],
    'cremad': [1, 2.5, None, False],
    'spcv1': [1, 1.0, None, False],
    'spcv2': [1, 1.0, None, False],
    'surge': [1, 4.0, None, False],
    'vc1': [1, 8.2, None, False],
    'vocalsound': [1, 4.18, None, False],
    'voxforge': [1, 5.8, None, False],
    'as20k': [1, 10.0, 'as', False],
    'as': [1, 10.0, 'as', True],
    'audiocaps': [1, 10.0, None, False],
    'ja_audiocaps': [1, 10.0, 'audiocaps', False],
    'clotho': [1, 30.0, None, False],
    'circor1': [1, 5.0, None, False],
    'circor2': [1, 5.0, None, False],
    'circor3': [1, 5.0, None, False],
    'bmdhs1': [1, 20.0, 'bmdhs', False],
    'bmdhs2': [1, 20.0, 'bmdhs', False],
    'bmdhs3': [1, 20.0, 'bmdhs', False],
}

_fs_table = {
    16000: '16k',
    22000: '22k', # Following COALA that uses 22,000 Hz
    32000: '32k',
    44100: '44k',
    48000: '48k',
}

def kwarg_cfg(**kwargs):
    cfg = EasyDict(kwargs)
    cfg.id = hash_text(str(cfg), L=8)
    return cfg

def append_to_csv(csv_filename, data):
    filename = Path(csv_filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(filename) if filename.exists() else pd.DataFrame()
    df = pd.concat([df, data], ignore_index=True).to_csv(filename, index=False)

# Get murmur from patient data.
def get_murmur(data):
    murmur = None
    for text in data.split("\n"):
        if text.startswith("#Murmur:"):
            murmur = text.split(": ")[1]
    if murmur is None:
        raise ValueError(
            "No murmur available. Is your code trying to load labels from the hidden data?"
        )
    return murmur


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for text in data.split("\n"):
        if text.startswith("#Outcome:"):
            outcome = text.split(": ")[1]
    if outcome is None:
        raise ValueError(
            "No outcome available. Is your code trying to load labels from the hidden data?"
        )
    return outcome


# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()
    

# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = scipy.io.wavfile.read(filename)
    return recording, frequency

# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            num_locations = int(l.split(" ")[1])
        else:
            break
    return num_locations

# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data

# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1 : num_locations + 1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(" ")
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings

def hash_text(text, L=128):
    hashed = hashlib.shake_128(text.encode()).hexdigest(L//2 + 1)
    return hashed[:L]

def setup_logger(name='', filename=None, level=logging.INFO):
    # Thanks to https://stackoverflow.com/a/53553516/6528729
    from imp import reload
    reload(logging)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=level, filename=filename)
    logger = logging.getLogger(name)
    console = logging.StreamHandler()
    console.setLevel(level)
    logger.addHandler(console)

def app_setup_logger(cfg, level=logging.INFO):
    logpath = Path(LOG_DIR)/cfg.id
    logpath.mkdir(parents=True, exist_ok=True)
    setup_logger(filename=logpath/'log.txt', level=level)
    print('Logging to', logpath/'log.txt')
    logging.info(str(cfg))
    return logpath

def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file(), f'{path_to_config} not found, cwd={Path(".").resolve()}'
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg

# Regular expression to check string can be converted into variables
# Thanks to -- https://stackoverflow.com/a/385597/6528729
re_valuable = re.compile("""(?x)
   ^
      (               # int|float|double
        [+-]?\ *      # first, match an optional sign *and space*
        (             # then match integers or f.p. mantissas:
            \d+       # start out with a ...
            (
                \.\d* # mantissa of the form a.b or a.
            )?        # ? takes care of integers of the form a
            |\.\d+     # mantissa of the form .b
        )
        ([eE][+-]?\d+)?  # finally, optionally match an exponent
      )
      |(              # bool
        False|True
      )
   $""")

def split_camma(text):
    flag = None
    elements = []
    cur = []
    for c in text:
        if flag is not None:
            cur.append(c)
            if flag == '[' and c == ']': flag = None
            if flag == '(' and c == ')': flag = None
            if flag == '"' and c == '"': flag = None
            if flag == "'" and c == "'": flag = None
            continue
        if c in ['[', '(', '"', "'"]:
            cur.append(c)
            flag = c
            continue
        if c == ',':
            elements.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
            elements.append(''.join(cur))
    return elements

def eval_if_possible(text):
    for pat in [r'\[.*\]', r'\(.*\)']:
        if re.search(pat, text):
            return eval(text)
    if re_valuable.match(text):
        return eval(text)
    return text

def complete_cfg(cfg, options, no_id=False):
    # Override parameter values with given "options".
    if 'name' not in cfg or not isinstance(cfg['name'], str):
        cfg['name'] = ''
    print(options)
    for item in split_camma(options):
        if item == '': continue
        keyvalues = item.split('=')
        assert len(keyvalues) == 2, f'An option need one and only one "=" in the option {item} in {options}.'
        key, value = keyvalues
        value = eval_if_possible(value)
        if key[0] == '+':
            key = key[1:]
            cfg[key] = None
        if key not in cfg.keys():
            raise Exception(f'Cannot find a setting named: {key} of the option {item}')
        cfg[key] = value
    # Set ID.
    if not no_id:
        task = Path(cfg.task_metadata).stem if 'task_metadata' in cfg else ''
        if 'name' in cfg and len(cfg['name']) > 0:
            name = cfg.name
        elif 'weight_file' in cfg and len(str(cfg['weight_file'])) > 0:
            weight_path = Path(str(cfg['weight_file']))
            parent = weight_path.parent.stem if len(weight_path.parent.stem) > 0 else str(cfg.audio_repr.split(',')[-1])
            name = f'{parent}-{weight_path.stem}'
        else:
            name = str(cfg.audio_repr.split(',')[-1])
        cfg.id = name + '_' + task + '_' + hash_text(str(cfg), L=8)
    return cfg

def get_original_folder(task, folder):
    orgs = {
        'us8k': 'UrbanSound8K',
        'esc50': 'ESC-50-master',
        'as20k': 'AudioSet',
        'as': 'AudioSet',
        'vocalsound': 'vocalsound_44k/data_44k',
    }
    return orgs[task] if task in orgs else folder


def get_defs(cfg, task, original_data=False):
    """Get task definition parameters.

    Returns:
        pathname (str): Metadata .csv file path.
        wav_folder (str): "work/16k/us8k" for example.
        folds (int): Number of LOOCV folds or 1. 1 means no cross validation.
        unit_sec (float): Unit duration in seconds.
        weighted (bool): True if the training requires a weighted loss calculation.
        balanced (bool): True if the training requires a class-balanced sampling.
    """
    folds, unit_sec, folder, balanced = _defs[task]
    folder = folder or task
    workfolder = f'{WORK}/original/{get_original_folder(task, folder)}' if original_data else f'{WORK}/{_fs_table[cfg.sample_rate]}/{folder}'
    return f'{METADATA_DIR}/{task}.csv', workfolder, folds, unit_sec, balanced

def make_cfg(config_file, task, options, extras={}, cancel_aug=False, abs_unit_sec=None, original_data=False):
    cfg = load_yaml_config(config_file)
    cfg = complete_cfg(cfg, options, no_id=True)
    task_metadata, task_data, n_folds, unit_sec, balanced = get_defs(cfg, task, original_data=original_data)
    # cancel augmentation if required
    if cancel_aug:
        cfg.freq_mask = None
        cfg.time_mask = None
        cfg.mixup = 0.0
        cfg.rotate_wav = False
    # unit_sec can be configured at runtime
    if abs_unit_sec is not None:
        unit_sec = abs_unit_sec
    # update some parameters.
    update_options = f'+task_metadata={task_metadata},+task_data={task_data}'
    update_options += f',+unit_samples={int(cfg.sample_rate * unit_sec)}'
    cfg = complete_cfg(cfg, update_options, no_id=True)
    # overwrite by extra command line
    options = []
    for k, v in extras.items():
        if v is not None:
            options.append(f'{k}={v}')
    options = ','.join(options)
    cfg = complete_cfg(cfg, options)
    # Set task name
    if 'task_name' not in cfg:
        cfg['task_name'] = task
    # Return file_name instead of waveform when loading an audio
    if 'return_filename' not in cfg:
        cfg['return_filename'] = False
    # Statistics for normalization
    if 'mean' not in cfg:
        cfg['mean'] = cfg['std'] = None
    return cfg, n_folds, balanced