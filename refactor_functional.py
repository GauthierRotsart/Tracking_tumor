import torch
import numpy as np
import bz2
import _pickle as cPickle
import os
import pickle
import math
import copy


def get_default_device(device):
    if torch.cuda.is_available():
        return torch.device('cuda:' + str(device))
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def association(x, index, dim):
    dim = abs(dim)
    N = x.shape[0]
    seq = np.zeros((dim, x.shape[1]), dtype=np.float32)

    for j in range(dim):
        idx = min(index + j, N - 1)
        seq[j] = x[idx]
    return seq


def association_image(x, index, dim):
    # Resultats: Idem que association image
    # Seul le dataset de taille X x W x H est mis en RAM
    # Plus lent, mais permets de traiter un dataset plus gros

    dim = abs(dim)
    n = x.shape[0]
    seq = np.zeros((dim, x.shape[1], x.shape[2]), dtype=np.float32)
    for j in range(dim):
        idx = min(index + j, n - 1)
        seq[j] = x[idx]

    return seq


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def load_data_structure(file_path):
    if file_path.endswith('.p'):
        # option using basic pickle function
        # self.Patients.list.append(pickle.load(open(dictFilePath, "rb")).list[0])

        # option for large files
        max_bytes = 2 ** 31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        data = pickle.loads(bytes_in)

    elif file_path.endswith('.pbz2'):
        data = bz2.BZ2File(file_path, 'rb')
        data = cPickle.load(data)
    else:
        print("Extension not found")
        raise NotImplementedError

    return data


def divide_into_patch(image, patch_size):
    x = image
    nbr_patch = x.shape[0]//patch_size  # nombre de patch dans une direction de l image
    patch = []
    for i in range(nbr_patch):
        for j in range(nbr_patch):
            p = x[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch.append(p)
    return patch


def patch_to_vector(image, patch_size):
    x = image
    nbr_of_patch = (x.shape[0]*x.shape[1])//patch_size**2
    data = np.zeros((nbr_of_patch, patch_size*patch_size))
    p = divide_into_patch(image, patch_size)
    for i in range(len(p)):
        data[i, :] = p[i].flatten()
    return data


def dataset_image(image_list, patch_size):
    x = image_list[0]  # on suppose que toutes les images ont la même dimension
    nbr_of_patch = (x.shape[0]*x.shape[1])//patch_size**2
    data = np.zeros((len(image_list), nbr_of_patch, patch_size*patch_size))
    for i in range(len(image_list)):
        data[i, :, :] = patch_to_vector(image_list[i], patch_size)
    return data


def data_array_creation(dynamic_sequence, cfg, translation, amplitude):
    assert cfg.training.params.output_dimension == 3, \
        f'You are using an output of dimension {cfg.training.params.output_dimension == 3}'
    dynamic_sequence = np.array(dynamic_sequence, dtype=object)
    dynmod = copy.deepcopy(dynamic_sequence)
    cropped_image = dynamic_sequence[:, 0]
    mask = dynamic_sequence[:, 1]
    center_of_mass = dynmod[:, 3]
    for i in range(center_of_mass.shape[0]):
        center_of_mass[i][0] = (center_of_mass[i][0] - translation[0]) / amplitude[0]
        center_of_mass[i][1] = (center_of_mass[i][1] - translation[1]) / amplitude[1]
        center_of_mass[i][2] = (center_of_mass[i][2] - translation[2]) / amplitude[2]
    center_of_mass = np.array(center_of_mass.tolist())

    if cfg.mask is True:
        x_set_image = mask
    else:
        x_set_image = cropped_image

    x_set = dataset_image(x_set_image, cfg.training.params.patch_size)
    pos_x = center_of_mass[:, 0]
    pos_y = center_of_mass[:, 1]
    pos_z = center_of_mass[:, 2]
    y_set = np.zeros((len(x_set_image), cfg.training.params.output_dimension))

    y_set[:, 0] = pos_x
    y_set[:, 1] = pos_y
    y_set[:, 2] = pos_z
    return x_set, y_set


def data_array_creation_no_normalisation(dynamic_sequence, cfg, translation, amplitude):
    assert cfg.training.params.output_dimension == 3, \
        f'You are using an output of dimension {cfg.training.params.output_dimension == 3}'
    dynamic_sequence = np.array(dynamic_sequence, dtype=object)
    dynmod = copy.deepcopy(dynamic_sequence)
    cropped_image = dynamic_sequence[:, 0]
    mask = dynamic_sequence[:, 1]
    center_of_mass = dynmod[:, 3]
    center_of_mass = np.array(center_of_mass.tolist())

    if cfg.mask is True:
        x_set_image = mask
    else:
        x_set_image = cropped_image

    x_set = dataset_image(x_set_image, cfg.training.params.patch_size)
    pos_x = center_of_mass[:, 0]
    pos_y = center_of_mass[:, 1]
    pos_z = center_of_mass[:, 2]
    y_set = np.zeros((len(x_set_image), cfg.training.params.output_dimension))

    y_set[:, 0] = pos_x
    y_set[:, 1] = pos_y
    y_set[:, 2] = pos_z
    return x_set, y_set


def data_array_creation_amplitude(dynamic_sequence, cfg, translation, amplitude):
    assert cfg.training.params.output_dimension == 3, \
        f'You are using an output of dimension {cfg.training.params.output_dimension == 3}'
    dynamic_sequence = np.array(dynamic_sequence, dtype=object)
    dynmod = copy.deepcopy(dynamic_sequence)
    cropped_image = dynamic_sequence[:, 0]
    mask = dynamic_sequence[:, 1]
    center_of_mass = dynmod[:, 3]
    for i in range(center_of_mass.shape[0]):
        center_of_mass[i][0] = (center_of_mass[i][0]) / amplitude[0]
        center_of_mass[i][1] = (center_of_mass[i][1]) / amplitude[1]
        center_of_mass[i][2] = (center_of_mass[i][2]) / amplitude[2]
    center_of_mass = np.array(center_of_mass.tolist())

    if cfg.mask is True:
        x_set_image = mask
    else:
        x_set_image = cropped_image

    x_set = dataset_image(x_set_image, cfg.training.params.patch_size)
    pos_x = center_of_mass[:, 0]
    pos_y = center_of_mass[:, 1]
    pos_z = center_of_mass[:, 2]
    y_set = np.zeros((len(x_set_image), cfg.training.params.output_dimension))

    y_set[:, 0] = pos_x
    y_set[:, 1] = pos_y
    y_set[:, 2] = pos_z
    return x_set, y_set


def data_array_creation_translation(dynamic_sequence, cfg, translation, amplitude):
    assert cfg.training.params.output_dimension == 3, \
        f'You are using an output of dimension {cfg.training.params.output_dimension == 3}'
    dynamic_sequence = np.array(dynamic_sequence, dtype=object)
    dynmod = copy.deepcopy(dynamic_sequence)
    cropped_image = dynamic_sequence[:, 0]
    mask = dynamic_sequence[:, 1]
    center_of_mass = dynmod[:, 3]
    for i in range(center_of_mass.shape[0]):
        center_of_mass[i][0] = (center_of_mass[i][0] - translation[0])
        center_of_mass[i][1] = (center_of_mass[i][1] - translation[1])
        center_of_mass[i][2] = (center_of_mass[i][2] - translation[2])
    center_of_mass = np.array(center_of_mass.tolist())

    if cfg.mask is True:
        x_set_image = mask
    else:
        x_set_image = cropped_image

    x_set = dataset_image(x_set_image, cfg.training.params.patch_size)
    pos_x = center_of_mass[:, 0]
    pos_y = center_of_mass[:, 1]
    pos_z = center_of_mass[:, 2]
    y_set = np.zeros((len(x_set_image), cfg.training.params.output_dimension))

    y_set[:, 0] = pos_x
    y_set[:, 1] = pos_y
    y_set[:, 2] = pos_z
    return x_set, y_set


def denormalisation(data, translation, amplitude):
    for i in range(3):
        data[:, :, i] = (data[:, :, i] * amplitude[i]) + translation[i]
    return data


def denormalisation_amplitude(data, translation, amplitude):
    for i in range(3):
        data[:, :, i] = (data[:, :, i] * amplitude[i])
    return data


class NoamOpt(object):
    """
    Optim wrapper that implements rate.
    """

    def __init__(self, optimizer, warmup_steps, t_total):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = 0  # warmup_multiplier
        self.cycles = 0.5
        self.t_total = t_total
        self._rate = 0
        self.epoch=0

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self):
        """
        Implement `lrate` above
        """
        if self._step < self.warmup_steps:
            f = float(self._step) / float(max(1e-4, self.warmup_steps))
            return self.warmup_multiplier + (1e-4 - self.warmup_multiplier) * f
        progress = float(self._step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class CosineAnnealingWarmupRestarts(object):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle
        self.last_epoch = 0

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__()

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




