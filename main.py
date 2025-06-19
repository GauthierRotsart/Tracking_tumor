import hydra
import torch.nn as nn
import torch
import math

import os
import sys
import pandas as pd
from tqdm.auto import tqdm
import random
import numpy as np
from refactor_functional import load_data_structure, data_array_creation, subsequent_mask, CosineAnnealingWarmupRestarts,\
    to_device, association_image, association, get_default_device
from refactor_TF import Net


def leave_one_out_split(patients, K):
    index = K - 1
    test_patient = [patients[index]]
    train_patients = [patients[i] for i in range(len(patients)) if i != index and i != 4 and i != 28 and i != 31]
    return train_patients, test_patient


def get_ps_data(base_path, translate_path, amplitude_path, cfg):
    # version du 10/02/2025
    # Training dataset contains series of 1500 images
    serie_list = [list(range(i * 1400, (i + 1) * 1400)) for i in range(math.ceil(cfg.data_quantity / 1400))]

    # Samples selection for one patient
    selected_index = []
    count_sample = 0
    count_serie = 0
    if len(serie_list) == 1:
        selected_index.extend(serie_list[0][:cfg.data_quantity])
    else:
        while count_sample + 1400 < cfg.data_quantity:
            selected_index.extend(serie_list[count_serie][:1400])
            count_sample += 1400
            count_serie += 1
        selected_index.extend(serie_list[count_serie][:cfg.data_quantity - count_sample])

    shuffle_batch = True
    if shuffle_batch:
        random.shuffle(selected_index)

    x_cache = {}
    y_cache = {}
    for nbr in range(count_serie + 1):
        if cfg.patient < 16:
            root_training = f'{base_path}FDGorFAZA_study/Patient_{cfg.patient}/1/FDG1/{cfg.training.params.regularity}/' \
                            f'wo_ief_training/resized/{cfg.training.params.width}/normalised/'
            translate_data = pd.read_excel(translate_path, header=None, sheet_name='FDG1').values
            amplitude_data = pd.read_excel(amplitude_path, header=None, sheet_name='FDG1').values
            translation = [translate_data[cfg.patient, 1], translate_data[cfg.patient, 2], translate_data[cfg.patient, 3]]
            amplitude = [amplitude_data[cfg.patient, 1], amplitude_data[cfg.patient, 2], amplitude_data[cfg.patient, 3]]
            data_path = f'{root_training}serie{nbr}/Patient_{cfg.patient}_1500_DRRMasksAndCOM_serie_{nbr}.p'
        else:
            root_training = f'{base_path}CPAP_study/Patient_{cfg.patient - 15}/1/NO_CPAP/{cfg.training.params.regularity}/' \
                            f'wo_ief_training/resized/{cfg.training.params.width}/normalised/'
            translate_data = pd.read_excel(translate_path, header=None, sheet_name='NO_CPAP1').values
            amplitude_data = pd.read_excel(amplitude_path, header=None, sheet_name='NO_CPAP1').values
            translation = [translate_data[cfg.patient - 15, 1], translate_data[cfg.patient - 15, 2],
                           translate_data[cfg.patient - 15, 3]]
            amplitude = [amplitude_data[cfg.patient - 15, 1], amplitude_data[cfg.patient - 15, 2],
                         amplitude_data[cfg.patient - 15, 3]]
            data_path = f'{root_training}serie{nbr}/Patient_{cfg.patient - 15}_1500_DRRMasksAndCOM_serie_{nbr}.p'
        dynamic_sequence = load_data_structure(data_path)
        x_cache[nbr], y_cache[nbr] = data_array_creation(dynamic_sequence, cfg, translation, amplitude)

    return x_cache, y_cache, selected_index


def get_mp_data(base_path, translate_path, amplitude_path, cfg):
    # version du 10/02/2025
    patient_list, _ = leave_one_out_split(patients=list(range(1, 36)), K=cfg.patient)
    # Balanced dataset for each patient
    data_patient = cfg.data_quantity // len(patient_list)
    # Training dataset contains series of 1500 images
    serie_list = [list(range(i * 1400, (i + 1) * 1400)) for i in range(math.ceil(data_patient / 1400))]

    # Samples selection for one patient
    selected_index2 = []
    count_sample = 0
    count_serie = 0
    if len(serie_list) == 1:
        selected_index2.extend(serie_list[0][:data_patient])
    else:
        while count_sample + 1400 < data_patient:
            selected_index2.extend(serie_list[count_serie][:1400])
            count_sample += 1400
            count_serie += 1
        selected_index2.extend(serie_list[count_serie][:data_patient - count_sample])

    # Give a new ID for samples coming from two patients
    selected_index = []
    for pat in patient_list:
        for d in selected_index2:
            selected_index.append(((pat * data_patient) + d))

    shuffle_batch = True
    if shuffle_batch:
        random.shuffle(selected_index)

    x_cache = {}
    y_cache = {}
    for nbr in range(count_serie + 1):
        for patient in patient_list:
            if patient < 16:
                if cfg.training.params.strategy == 'T1':
                    root_training = f'{base_path}FDGorFAZA_study/Patient_{patient}/1/FDG1/{cfg.training.params.regularity}/' \
                                    f'wo_ief_training/resized/{cfg.training.params.width}/normalised/'
                    data_path = f'{root_training}serie{nbr}/Patient_{patient}_1500_DRRMasksAndCOM_serie_{nbr}.p'
                elif cfg.training.params.strategy == 'T2':
                    if nbr < 10:
                        # Comme on fait du leave one out, on peut prendre les images du test set pour eviter de
                        # generer de nouvelles images
                        root_training = f'{base_path}FDGorFAZA_study/Patient_{patient}/2/FDG2/{cfg.training.params.regularity}/' \
                                        f'test_images_21/resized/{cfg.training.params.width}/normalised/'
                        data_path = f'{root_training}serie{nbr}/Patient_{patient}_1500_DRRMasksAndCOM_serie_{nbr}.p'
                    else:
                        root_training = f'{base_path}FDGorFAZA_study/Patient_{patient}/2/FDG2/{cfg.training.params.regularity}/' \
                                        f'validation_images_21/resized/{cfg.training.params.width}/normalised/'
                        data_path = f'{root_training}serie{nbr - 10}/Patient_{patient}_1000_DRRMasksAndCOM_serie_{nbr - 10}.p'
                else:
                    print(f"You use {cfg.training.params.strategy} strategy !")
                    raise NotImplementedError

                # On utilise la translation T1 et l amplitude T1 pour etre dans les memes conditions que lors du test
                translate_data = pd.read_excel(translate_path, header=None, sheet_name='FDG1').values
                amplitude_data = pd.read_excel(amplitude_path, header=None, sheet_name='FDG1').values
                translation = [translate_data[patient, 1], translate_data[patient, 2], translate_data[patient, 3]]
                amplitude = [amplitude_data[patient, 1], amplitude_data[patient, 2], amplitude_data[patient, 3]]
            else:
                if cfg.training.params.strategy == 'T1':
                    root_training = f'{base_path}CPAP_study/Patient_{patient - 15}/1/NO_CPAP/{cfg.training.params.regularity}/' \
                                    f'wo_ief_training/resized/{cfg.training.params.width}/normalised/'
                    data_path = f'{root_training}serie{nbr}/Patient_{patient - 15}_1500_DRRMasksAndCOM_serie_{nbr}.p'
                elif cfg.training.params.strategy == 'T2':
                    if nbr < 10:
                        root_training = f'{base_path}CPAP_study/Patient_{patient - 15}/2/NO_CPAP/{cfg.training.params.regularity}/' \
                                        f'test_images_21/resized/{cfg.training.params.width}/normalised/'
                        data_path = f'{root_training}serie{nbr}/Patient_{patient - 15}_1500_DRRMasksAndCOM_serie_{nbr}.p'
                    else:
                        root_training = f'{base_path}CPAP_study/Patient_{patient - 15}/2/NO_CPAP/{cfg.training.params.regularity}/' \
                                        f'validation_images_21/resized/{cfg.training.params.width}/normalised/'
                        data_path = f'{root_training}serie{nbr - 10}/Patient_{patient - 15}_1000_DRRMasksAndCOM_serie_{nbr - 10}.p'
                else:
                    print(f"You use {cfg.training.params.strategy} strategy !")
                    raise NotImplementedError

                translate_data = pd.read_excel(translate_path, header=None, sheet_name='NO_CPAP1').values
                amplitude_data = pd.read_excel(amplitude_path, header=None, sheet_name='NO_CPAP1').values
                translation = [translate_data[patient - 15, 1], translate_data[patient - 15, 2],
                               translate_data[patient - 15, 3]]
                amplitude = [amplitude_data[patient - 15, 1], amplitude_data[patient - 15, 2],
                             amplitude_data[patient - 15, 3]]

            dynamic_sequence = load_data_structure(data_path)
            x_cache[patient + nbr * 35], y_cache[patient + nbr * 35] = data_array_creation(dynamic_sequence,
                                                                                           cfg, translation,
                                                                                           amplitude)
    return x_cache, y_cache, selected_index


def train_loop(model, cfg):
    base_path = '/Benson_DATA1/grotsart/'
    translate_path = '/linux/grotsartdehe/translate.xlsx'
    amplitude_path = '/linux/grotsartdehe/amplitude.xlsx'
    if cfg.multi is False:
        x_cache, y_cache, selected_index = get_ps_data(base_path, translate_path, amplitude_path, cfg)
        saving_path = f'/Benson_DATA2/grotsart/Medical_physics/PS-models/{cfg.data_quantity}/{cfg.training.params.width}-' \
                      f'{cfg.training.params.patch_size}/'
    else:
        x_cache, y_cache, selected_index = get_mp_data(base_path, translate_path, amplitude_path, cfg)
        saving_path = f'/Benson_DATA2/grotsart/Medical_physics/MP-models/{cfg.data_quantity}/{cfg.training.params.width}-' \
                      f'{cfg.training.params.patch_size}/'

    n_epochs = cfg.training.params.epoch
    warmup_epoch = cfg.training.params.warmup
    learning_rate = cfg.training.params.learning_rate
    batch_size = cfg.training.params.batch_size
    width = cfg.training.params.width
    height = cfg.training.params.height
    patch_size = cfg.training.params.patch_size
    img_sequence = cfg.training.params.img_sequence
    horizon = cfg.training.params.horizon
    patches_sequence = (height // patch_size) * (width // patch_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = math.ceil(cfg.data_quantity / batch_size) * n_epochs
    warmup_steps = math.ceil(cfg.data_quantity / batch_size) * warmup_epoch
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=total_steps, cycle_mult=1.0,
                                              max_lr=learning_rate, min_lr=learning_rate / 100,
                                              warmup_steps=warmup_steps, gamma=1.0)

    if not os.path.exists(saving_path):
        os.umask(0)  # for the file permission
        os.makedirs(saving_path)  # Create a new directory because it does not exist
        print("New directory created to save the data: ", saving_path)

    lrs = [optimizer.param_groups[0]["lr"]]
    if cfg.training.params.load_weights is True:
        start_epoch = cfg.inference.params.epoch
        epoch_list = [start_epoch + i + 1 for i in range(n_epochs - start_epoch)]
    else:
        start_epoch = 0
        epoch_list = [i + 1 for i in range(n_epochs)]
    for epoch in epoch_list:
        model.train()
        epoch_loss = []
        with tqdm(range(cfg.data_quantity), initial=start_epoch,  file=sys.stdout, leave=False, dynamic_ncols=True,
                  total=cfg.data_quantity, unit="serie", desc=f"Training time: {np.round(epoch/n_epochs * 100,2)} %") as tepoch:

            for idx in range(0, len(selected_index), batch_size):
                serie_loss = []
                batch_idx = selected_index[idx:idx + batch_size]
                batch_x = np.zeros((len(batch_idx), img_sequence, patches_sequence, patch_size ** 2))
                batch_y = np.zeros((len(batch_idx), horizon, cfg.training.params.output_dimension))
                b_i = 0
                for i in batch_idx:
                    if cfg.multi is False:
                        idx_serie = i // 1400
                        idx_image = i - (idx_serie * 1400)

                        x_set = x_cache[idx_serie]
                        x = association_image(x_set, idx_image, cfg.training.params.img_sequence)

                        y_set = y_cache[idx_serie]
                        y = association(y_set, idx_image + cfg.training.params.img_sequence, cfg.training.params.horizon)

                        batch_x[b_i, :, :, :] = x
                        batch_y[b_i, :, :] = y
                    else:
                        data_patient = cfg.data_quantity // 31
                        idx_patient = i // data_patient
                        idx_serie = (i - (idx_patient * data_patient)) // 1400
                        idx_image = i - (idx_patient * data_patient) - (idx_serie * 1400)

                        x_set = x_cache[idx_patient + idx_serie * 35]
                        x = association_image(x_set, idx_image, cfg.training.params.img_sequence)

                        y_set = y_cache[idx_patient + idx_serie * 35]
                        y = association(y_set, idx_image + cfg.training.params.img_sequence, cfg.training.params.horizon)

                        batch_x[b_i, :, :, :] = x
                        batch_y[b_i, :, :] = y
                    b_i += 1

                enc_inp = torch.from_numpy(batch_x).to(cfg.device).float()
                target = torch.from_numpy(batch_y).to(cfg.device).float()
                # Le SOS doit etre différent des données qui pourraient apparaitre dans le training set
                start_of_sequence = torch.Tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1).to(cfg.device)
                start_of_sequence = start_of_sequence.repeat(len(batch_idx), 1, 1)

                target_c = torch.zeros((target.shape[0], target.shape[1], 1)).to(cfg.device)
                target = torch.cat((target, target_c), -1)

                dec_inp = torch.cat((start_of_sequence, target[:, :-1, :]), 1)
                # si on utilise des sequences de taille differente, on utilise du padding pour que toutes les
                # sequences aient la meme taille. Dans ce cas, il faut masquer certaine partie de l encodeur
                enc_mask = torch.ones((enc_inp.shape[0], 1, enc_inp.shape[1]*enc_inp.shape[2])).to(cfg.device)
                tgt_mask = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(cfg.device)

                # Compute prediction and loss
                pred = model(enc_inp, enc_mask, dec_inp, tgt_mask)
                loss = torch.sqrt(loss_function(pred[:, :, 0:3], target[:, :, 0:3]))
                epoch_loss = np.append(epoch_loss, loss.item())
                serie_loss = np.append(serie_loss, loss.item())

                # Gradient backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]
                scheduler.step()

                tepoch.set_postfix(loss=np.mean(serie_loss))
                tepoch.update(len(batch_idx))

        lrs.append(lr)

        if epoch == 1 or epoch == 5 or epoch == 10 or epoch == 25 or epoch == 50 or epoch == 75 or epoch == 100:
            print(f"Epoch: {epoch}, Training loss: {np.mean(epoch_loss)}")
            if cfg.training.params.strategy == 'T1':
                model_name = f'Patient_{cfg.patient}-LR-{cfg.training.params.learning_rate}-epoch-{epoch}-{cfg.seed}'
            elif cfg.training.params.strategy == 'T2':
                model_name = f'T2-Patient_{cfg.patient}-LR-{cfg.training.params.learning_rate}-epoch-{epoch}-{cfg.seed}'
            else:
                raise NotImplementedError
            
            torch.save({
                'model_state_dict': model.state_dict(),
            }, saving_path + model_name + '.pt')


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    torch.set_default_dtype(torch.float32)

    # un grand batch permet un entrainement plus rapide mais apporte moins de précision
    # un petit batch permet un entrainement moins rapide mais apporte plus de précision
    # dans le cas de petit dataset, on peut donc se permettre d'utiliser un petit batch
    # Stochastic Gradient Descent (SGD) = Gradient Descent with batch size equal to 1
    # Mini batch Gradient Descent = Gradient Descent with batch size bigger than 1

    # ALL is seeded for reproducibility reasons
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.training.params.load_weights is True:
        if cfg.multi is False:
            data_path = f'/Benson_DATA2/grotsart/Medical_physics/PS-models/{cfg.data_quantity}/{cfg.training.params.width}-' \
                        f'{cfg.training.params.patch_size}/'
        else:
            data_path = f'/Benson_DATA2/grotsart/Medical_physics/MP-models/{cfg.data_quantity}/{cfg.training.params.width}-' \
                        f'{cfg.training.params.patch_size}/'
        if cfg.training.params.strategy == 'T1':
            model_path = f'Patient_{cfg.patient}-LR-{cfg.training.params.learning_rate}-epoch-{cfg.inference.params.epoch}-{cfg.seed}.pt'
        elif cfg.training.params.strategy == 'T2':
            model_path = f'T2-Patient_{cfg.patient}-LR-{cfg.training.params.learning_rate}-epoch-{cfg.inference.params.epoch}-{cfg.seed}.pt'
        else:
            raise NotImplementedError
        net = Net(cfg=cfg)
        checkpoint = torch.load(data_path + model_path, map_location=get_default_device(cfg.device))
        net.load_state_dict(checkpoint['model_state_dict'])
        net = to_device(net, cfg.device)  # utile ?
    else:
        net = Net(cfg)
        net = to_device(net, cfg.device)

        # Initialize parameters with Glorot / fan_avg.
        # Les poids qui ont une dimension plus petite que 2 utilisent l initialisation par défaut de pytorch
        # Cette initialisation depend de la couche du réseau
        for p in net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # nn.init.kaiming_uniform_(p)

    with torch.cuda.device(cfg.device):
        torch.cuda.empty_cache()

    train_loop(net, cfg)


if __name__ == "__main__":
    main()
