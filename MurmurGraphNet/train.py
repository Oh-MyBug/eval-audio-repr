"""Main solver program for CirCor evaluation.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import fire
import torch
import random
import logging
import torchaudio
import numpy as np
import pandas as pd
sys.path.append('app/circor/heart-murmur-detection')
sys.path.append('app/circor/heart-murmur-detection/ModelEvaluation')

from tqdm import tqdm
from pathlib import Path
from sklearn import utils
import evar.ar_m2d
from MurmurGraphNet.dataloader.data import create_dataloader
from MurmurGraphNet.model.finetune import TaskNetwork, finetune_main   
from MurmurGraphNet.utils.utils import make_cfg, load_patient_data, load_recordings, append_to_csv, kwarg_cfg, RESULT_DIR
from MurmurGraphNet.model.evaluate_model import evaluate_model


def infer_and_eval(cfg, model, test_root, eval_mode='follow_prior_work'):
    model.eval()

    pids = sorted(list(set([f.stem.split('_')[0] for f in Path(test_root).glob('*.wav')])))
    txt_files = [test_root+pid+'.txt' for pid in pids]
    print('Test file folder:', test_root)
    print('Test files:', pids[:2], txt_files[:2])
    softmax_fn = torch.nn.Softmax(dim=1)
    probabilities, wav_probabilities = [], []
    
    # ✅ 定义四个位置的顺序
    POSITIONS = ['MV', 'AV', 'PV', 'TV']

    for txt in tqdm(txt_files):
        # Load recordings
        data = load_patient_data(txt)
        recordings, frequencies = load_recordings(test_root, data, get_frequencies=True)
        recordings = [torch.tensor(r / 32768.).to(torch.float) for r in recordings]

        # Resample all recordings to target sample rate
        wavs = [torchaudio.transforms.Resample(f, cfg.sample_rate)(r) for r, f in zip(recordings, frequencies)]
        
        # ✅ 组织成位置-录音的映射
        position_recordings = organize_recordings_by_position(txt, wavs, test_root, data)
        
        # ✅ 生成四位置组合样本
        multi_position_samples = generate_multi_position_samples(position_recordings, cfg.unit_samples)
        
        # Process each multi-position sample
        L = cfg.unit_samples
        sample_logits = []
        
        for multi_wav in multi_position_samples:
            # multi_wav: (5, L) tensor - [MV, AV, PV, TV, mask_channel]
            with torch.no_grad():
                x = multi_wav.unsqueeze(0)  # (1, 5, L)
                logit = model(x)  # Forward through your updated model
                if isinstance(logit, tuple):
                    logit = logit[0]  # Take first element if tuple returned
            sample_logits.append(logit)
        
        # Average logits across all samples for this patient
        if sample_logits:
            avg_logits = torch.stack(sample_logits).mean(0)  # [1, num_classes]
        else:
            # Fallback: create zero logits
            avg_logits = torch.zeros(1, 3)  # [1, 3] for 3 classes
        
        # Reorder classes from ["Absent", "Present", "Unknown"] -> ["Present", "Unknown", "Absent"]
        avg_logits = avg_logits[:, [1, 2, 0]]
        
        # Get probabilities
        probs = avg_logits.softmax(1).detach().to('cpu')[0]
        probabilities.append(probs)
        
        # For wav_probabilities, we'll use the same result (since we're now processing as groups)
        wav_probabilities.append(probs.unsqueeze(0))  # Keep same format as original

    probabilities = torch.stack(probabilities)
    
    # Rest of the function remains the same...
    def label_decision_rule(wav_probs):
        cidxs = torch.argmax(wav_probs, dim=1)
        PRESENT, UNKNOWN, ABSENT = 0, 1, 2
        if PRESENT in cidxs:
            final_label = PRESENT
        elif UNKNOWN in cidxs:
            final_label = UNKNOWN
        else:
            final_label = ABSENT
        return final_label

    if eval_mode is None or eval_mode == 'follow_prior_work':
        print('Label decision follows: Panah et al. "Exploring Wav2vec 2.0 Model for Heart Murmur Detection." EUSIPCO, 2023, pp. 1010–14.')
        cidxs = torch.tensor([label_decision_rule(wav_probs) for wav_probs in wav_probabilities])
    elif eval_mode == 'normal':
        print('Label decision is: torch.argmax(probabilities, dim=1)')
        cidxs = torch.argmax(probabilities, dim=1)
    else:
        assert False, f'Unknown eval_mode: {eval_mode}'
        
    labels = torch.nn.functional.one_hot(cidxs, num_classes=3)

    wav_probabilities = [p.numpy() for p in wav_probabilities]
    probabilities = probabilities.numpy()
    labels = labels.numpy()
    return evaluate_model(test_root, probabilities, labels), (wav_probabilities, probabilities)


def organize_recordings_by_position(txt_file, wavs, test_root, data):
    """
    组织录音文件按位置分类
    Returns: dict {position: [wav_tensors]}
    """
    POSITIONS = ['MV', 'AV', 'PV', 'TV']
    position_recordings = {pos: [] for pos in POSITIONS}
    
    # 从文件名中解析位置信息
    wav_files = [f for f in Path(test_root).glob('*.wav') if f.stem.split('_')[0] == Path(txt_file).stem]
    
    for wav_file in wav_files:
        parts = wav_file.stem.split('_')
        if len(parts) >= 2:
            position = parts[1]  # 假设格式为 patientID_position_segment.wav
            if position in POSITIONS:
                # 找到对应的wav tensor
                # 这里需要根据你的具体数据加载逻辑来匹配wav_file和wavs
                # 简化版本：按顺序匹配（你可能需要调整）
                for i, wav in enumerate(wavs):
                    if i < len(wav_files) and wav_file == sorted(wav_files)[i]:
                        position_recordings[position].append(wav)
                        break
    
    return position_recordings


def generate_multi_position_samples(position_recordings, unit_samples, num_samples=10):
    """
    生成多位置组合样本
    Args:
        position_recordings: {position: [wav_tensors]}
        unit_samples: 每个位置的目标样本长度
        num_samples: 生成的样本数量
    Returns:
        List of (5, unit_samples) tensors
    """
    POSITIONS = ['MV', 'AV', 'PV', 'TV']
    samples = []
    
    for _ in range(num_samples):
        position_audios = []
        mask_values = []
        
        for position in POSITIONS:
            recordings = position_recordings[position]
            
            if recordings:
                # 随机选择一个录音
                selected_wav = random.choice(recordings)
                
                # 长度处理
                if len(selected_wav) >= unit_samples:
                    start = random.randint(0, len(selected_wav) - unit_samples)
                    wav_segment = selected_wav[start:start + unit_samples]
                else:
                    wav_segment = torch.nn.functional.pad(selected_wav, (0, unit_samples - len(selected_wav)))
                
                mask_values.append(1.0)  # 有数据
            else:
                # 缺失位置用零填充
                wav_segment = torch.zeros(unit_samples)
                mask_values.append(0.0)  # 缺失
            
            position_audios.append(wav_segment)
        
        # ✅ 创建mask通道（与训练时格式一致）
        mask_channel = torch.zeros(unit_samples)
        segment_length = unit_samples // 4
        
        for i, mask_val in enumerate(mask_values):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < 3 else unit_samples
            mask_channel[start_idx:end_idx] = mask_val
        
        position_audios.append(mask_channel)
        
        # 构建样本: (5, unit_samples)
        sample = torch.stack(position_audios, dim=0)
        samples.append(sample)
    
    return samples


def eval_main(config_file, task, checkpoint, options='', seed=42, lr=None, hidden=(128,), epochs=None, early_stop_epochs=None, warmup_epochs=None,
              mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
              optim='sgd', unit_sec=None, verbose=False, data_path='work', eval_mode=None, reweight=True, save_prob=None):
    
    cfg, n_folds, balanced = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
    lr = lr or cfg.ft_lr
    cfg.mixup = mixup if mixup is not None else cfg.mixup
    cfg.ft_early_stop_epochs = early_stop_epochs if early_stop_epochs is not None else cfg.ft_early_stop_epochs
    cfg.warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg.warmup_epochs
    cfg.ft_epochs = epochs or cfg.ft_epochs
    cfg.ft_freq_mask = freq_mask if freq_mask is not None else cfg.ft_freq_mask
    cfg.ft_time_mask = time_mask if time_mask is not None else cfg.ft_time_mask
    cfg.ft_rrc = rrc if rrc is not None else (cfg.ft_rrc if 'ft_rrc' in cfg else False)
    cfg.training_mask = training_mask if training_mask is not None else (cfg.training_mask if 'training_mask' in cfg else 0.0)
    cfg.ft_bs = batch_size or cfg.ft_bs
    cfg.optim = optim
    cfg.unit_sec = unit_sec
    cfg.data_path = data_path

    train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg, fold=n_folds-1, seed=seed, batch_size=cfg.ft_bs,
        always_one_hot=True, balanced_random=balanced)
    print('Classes:', train_loader.dataset.classes)
    cfg.eval_checkpoint = checkpoint

    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, mixup=cfg.mixup, bs=cfg.ft_bs,
                                freq_mask=cfg.ft_freq_mask, time_mask=cfg.ft_time_mask, rrc=cfg.ft_rrc, epochs=cfg.ft_epochs,
                                early_stop_epochs=cfg.ft_early_stop_epochs, n_class=len(train_loader.dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make a fresh model
    ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
    if hasattr(train_loader, 'lms_mode') and train_loader.lms_mode:
        ar.precompute_lms(device, train_loader)
    else:
        ar.precompute(device, train_loader)
    task_model = TaskNetwork(cfg, ar).to(device)
    task_model_dp = torch.nn.DataParallel(task_model).to(device)
    # Load checkpoint
    print('Using checkpoint', checkpoint)
    print(task_model_dp.load_state_dict(torch.load(checkpoint, map_location=device)))
    task_model_dp.eval()

    # Run evaluation
    circor_no = task[-1]  # ex) '1' of 'circor1'
    stratified_data = f'app/circor/heart-murmur-detection/data/stratified_data{circor_no}/test_data/'
    results, probs = infer_and_eval(cfg, task_model_dp, stratified_data, eval_mode=eval_mode)
    (   classes,
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        f_measure,
        f_measure_classes,
        accuracy,
        accuracy_classes,
        weighted_accuracy,
        uar,
    ) = results

    name  = f'{cfg.id}{"" if cfg.weight_file != "" else "/rnd"}-'
    report = f'Circor MLP solution {name} on {task} -> weighted_accuracy: {weighted_accuracy:.5f}. UAR: {uar:.5f}, recall per class: {accuracy_classes}'
    report += f', best weight: {checkpoint}, config: {cfg}'
    logging.info(report)

    extra = f'-lr{lr}-h{hidden}-e{cfg.ft_epochs}' + ('-rew' if reweight else '')
    result_df = pd.DataFrame({
        'representation': [(cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]) + extra], # AR name
        'task': [task],
        'wacc': [weighted_accuracy],
        'uar': [uar],
        'r_Present': [accuracy_classes[0]],
        'r_Unknown': [accuracy_classes[1]],
        'r_Absent': [accuracy_classes[2]],
        'weight_file': [cfg.weight_file],
        'run_id': [cfg.id],
        'report': [report],
    })
    csv_name = {
        None: 'circor-scores.csv',
        'follow_prior_work': 'circor-scores.csv',
        'normal': 'circor-scores-wo-rule.csv',
    }[eval_mode]
    append_to_csv(f'{RESULT_DIR}/{csv_name}', result_df)

    if save_prob is not None:
        for i, var in zip(['_1', '_2'], probs):
            prob_name = Path(save_prob)/str(checkpoint).replace('/', '-').replace('.pth', i + '.npy')
            #probs = [p.numpy() for p in probs]
            prob_name.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_name, np.array(var, dtype=object))
            print('Probabilities saved as:', prob_name)


class WeightedCE:
    def __init__(self, labels, circor_reweighting=False) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = utils.class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        if circor_reweighting:
            weights = weights * np.sqrt(np.array([1, 3, 5])).T
        weights = weights / weights.sum()
        self.celoss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
        self.__name__ = f'CrossEntropyLoss(weight={weights})'
    def __call__(self, preds, gts):
        loss = self.celoss(preds, gts)
        return loss


def solve_graph_circor(config_file, task, options='', seed=42, lr=None, hidden=(128,), epochs=None, early_stop_epochs=None, warmup_epochs=None,
                  mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
                  optim='sgd', unit_sec=None, verbose=False, data_path='work', eval_only=f'MurmurGraphNet\logs\m2d_vit_base-80x608p16x16-221006-mr7-checkpoint-300_circor1_88c4bcf4\weights_ep22it47-0.73368_loss7.9553.pth', eval_mode=None, reweight=True, save_prob=None):

    assert task in [f'circor{n}' for n in range(1, 3+1)]
    # We train a model using the original fine-tuner from the EVAR (finetune_main), and the best_path holds the path of the best weight.
    # This part is the same training process as what we have been doing in BYOL-A and M2D.
    if eval_only is None:
        cfg, __, balanced = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
        train_loader, _, _, _ = create_dataloader(cfg, fold=0, seed=0, batch_size=1, balanced_random=balanced, pin_memory=False)
        #labels = np.argmax(train_loader.dataset.labels, axis=1)  # One-hot to numbers
        labels = train_loader.dataset.labels.numpy()
        weighted_ce = WeightedCE(labels, circor_reweighting=reweight)
        report, scores, best_path, name, cfg, logpath = finetune_main(config_file, task, options=options, seed=seed, lr=lr, hidden=hidden, epochs=epochs,
            early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
            mixup=mixup, freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, batch_size=batch_size,
            optim=optim, unit_sec=unit_sec, freeze_ar=True, loss_fn=weighted_ce, verbose=verbose, data_path=data_path)
        del report, scores, name, cfg, logpath
    else:
        best_path = eval_only

    # Then, we evaluate the trained model specifically for the CirCor problem setting.
    return eval_main(config_file, task, best_path, options=options, seed=seed, lr=lr, hidden=hidden, epochs=epochs,
        early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
        mixup=mixup, freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, batch_size=batch_size,
        optim=optim, unit_sec=unit_sec, verbose=verbose, data_path=data_path, eval_mode=eval_mode, reweight=reweight, save_prob=save_prob)


if __name__ == '__main__':
    fire.Fire(solve_graph_circor)
