from typing import Iterable, Callable
from hash import *
import torch, numpy as np, math
from scipy.special import erfinv

g_device = torch.device('cuda:1')

class modulebase:
    
    default_config = {
        'device': None
    }
    
    def __init__(self, **kwargs):
        
        kwargs['device'] = g_device

        self.print: Callable[[str], None] = None
        
        config_list = [kwargs]
        clazz = type(self)
        while clazz != modulebase:
            config_list.append(clazz.default_config)
            clazz = clazz.__base__
        config_list.append(modulebase.default_config)
        config_list.reverse()
        
        self.config = {}
        
        for config in config_list:
            self.config.update(config)
        
        self.device = self.config['device']
    
    # search mse
    def once(self, state):
        pass

    # get qmask candidate
    def preprocess(self, state):
        pass
    
    # faultmap initialized
    def process_hash(self, state):
        pass
    
    # qmask done
    def process(self, state):
        pass
    
    # qtensor done
    def process_error(self, state):
        pass
    
    # error injected
    def postprocess(self, state):
        pass

_quan_point = {
    'int4': list(range(-8, 8)),
    'int8': list(range(-128, 128)),
    #'flint': [-16, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 16]
    'flint': [-32, -24, -16, -12, -8, -6, -4, -3, 0, 3, 4, 6, 8, 12, 16, 24]
}

_quan_point_multi = {}

_quan_point_arg = {}

MULTIPLIER = 256

for k, v in _quan_point.items():
    m = max(abs(x) for x in v)
    c = np.array(v) / m * MULTIPLIER
    
    _quan_point_arg[k] = [
        np.argmin(np.abs(c - (i + .5)))
        for i in range(-MULTIPLIER, MULTIPLIER + 1)
    ]

class qbase(modulebase):

    default_config = {
        'scale_1_max': 1.2, # outlier
        'scale_1_min': 0.05,
        'scale_2_max': 1.2, # non-outlier
        'scale_2_min': 0.05,
        'scale_1_step': 0.05,
        'scale_2_step': 0.05,
        'encoding_1': 'int4',
        'encoding_2': 'int4',
    }
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.scale_1 = torch.arange(self.config['scale_1_min'], self.config['scale_1_max'], self.config['scale_1_step'], device=self.device)
        self.scale_2 = torch.arange(self.config['scale_2_min'], self.config['scale_2_max'], self.config['scale_2_step'], device=self.device)
        
        self.point_1 = torch.tensor(_quan_point[self.config['encoding_1']], device=self.device).float()
        self.point_2 = torch.tensor(_quan_point[self.config['encoding_2']], device=self.device).float()
        
        self.point_1 /= self.point_1.max()
        self.point_2 /= self.point_2.max()
        
        self.point_1_a = torch.tensor(_quan_point_arg[self.config['encoding_1']], device=self.device, dtype=torch.int16)
        self.point_2_a = torch.tensor(_quan_point_arg[self.config['encoding_2']], device=self.device, dtype=torch.int16)
        
        self.scale_1_count = self.scale_1.numel() # scale 1 搜索数量
        self.scale_2_count = self.scale_2.numel() # scale 2 搜索数量
        
        scale_1_repeated = self.scale_1.view(self.scale_1_count, 1).repeat(1, self.scale_2_count).flatten()
        scale_2_repeated = self.scale_2.view(1, self.scale_2_count).repeat(self.scale_1_count, 1).flatten()
        
        self.scales = torch.stack([scale_1_repeated, scale_2_repeated], dim=1).reshape(1, -1, 2)
        self.all_scales = self.scale_1_count * self.scale_2_count
    
    def get_qmask_(self, flattened_abs, scale_1, scale_2):
        return self.get_qmask_with_zero(flattened_abs, scale_1, scale_2)

    def get_qmask_with_zero(self, flattened_abs, scale_1, scale_2):
        mask = self.get_qmask(flattened_abs, scale_1, scale_2)
        return mask, torch.zeros_like(mask)
    
    def get_qmask(self, flattened_abs, scale_1, scale_2):
        raise NotImplementedError
    
    def once(self, state):
        
        tensor: torch.Tensor = state['tensor']
        
        global_scale = torch.max(torch.abs(tensor))
        
        if global_scale == 0: global_scale = 1
        
        flattened = (tensor / global_scale).view(-1, 1)
        flattened_abs = torch.abs(flattened)
        
        # print(f'flatten shape: {flattened.shape}')
        
        size = flattened.numel()
        
        batch = math.ceil(128 * 1024 * 1024 / size)
        
        start = 0
        
        g_min_mse = torch.tensor(float('inf'), device=self.device), None
        
        while start < self.all_scales:
            end = min(start + batch, self.all_scales)
            scale_1_batch = self.scales[:, start:end, 0]
            scale_2_batch = self.scales[:, start:end, 1]
            
            encode_1 = (torch.clamp(flattened / scale_1_batch + 1, 0, 2) * MULTIPLIER).long()
            encode_2 = (torch.clamp(flattened / scale_2_batch + 1, 0, 2) * MULTIPLIER).long()
            
            encode_1 = self.point_1_a[encode_1].long()
            encode_2 = self.point_2_a[encode_2].long()
            
            outlier_mask, zero_mask = self.get_qmask_(flattened_abs, scale_1_batch, scale_2_batch)
            
            quan = torch.where(
                outlier_mask,
                self.point_1[encode_1] * scale_1_batch,
                self.point_2[encode_2] * scale_2_batch
            )
            
            quan = torch.where(
                zero_mask,
                torch.zeros_like(quan),
                quan
            )

            del outlier_mask, zero_mask
            #encode = torch.where(
            #    outlier_mask,
            #    encode_1,
            #    encode_2
            #)
            
            loss_min_sum = torch.sum((quan - flattened) ** 2, dim=0)
            
            del quan
            
            # print(f'loss_min_sum shape: {loss_min_sum.shape}')
            
            loss_min = torch.min(loss_min_sum, dim=0)
            
            if loss_min.values < g_min_mse[0]:
                # print(f'indice: {loss_min.indices + start}')
                g_min_mse = loss_min.values, loss_min.indices + start # , encode[:, loss_min.indices]
                
            start = end
            
            del loss_min, loss_min_sum
        
        min_mse = g_min_mse[0]
        min_index = g_min_mse[1]
        # result = g_min_mse[2]
        chosen_scale_1 = min_index // self.scale_2_count
        chosen_scale_2 = min_index % self.scale_2_count
        
        state['scale_1'] = self.scale_1.view(-1)[chosen_scale_1].item()
        state['scale_2'] = self.scale_2.view(-1)[chosen_scale_2].item()
        
        # self.print(f'chosen scale 1: {state["scale_1"]}, chosen scale 2: {state["scale_2"]}')
                   
        state['global_scale'] = global_scale
        # state['qtensor_bit'] = torch.ones_like(result) * 4
        state['qmse'] = min_mse / tensor.numel()
    
    def preprocess(self, state):
        
        tensor: torch.Tensor = state['tensor']
        
        global_scale = state['global_scale']
        
        flattened = (tensor / global_scale).view(-1, 1)
        flattened_abs = torch.abs(flattened)
        
        qmask, zero_mask = self.get_qmask_(flattened_abs.view(-1, 1), state['scale_1'], state['scale_2'])
        state['qmask'], state['zeromask'] = qmask.view(-1), zero_mask.view(-1)

    def process(self, state):
        tensor: torch.Tensor = state['tensor']
        flattened = (tensor / state['global_scale']).view(-1)
        
        encode_1 = (torch.clamp(flattened / state['scale_1'] + 1, 0, 2) * MULTIPLIER).long()
        encode_2 = (torch.clamp(flattened / state['scale_2'] + 1, 0, 2) * MULTIPLIER).long()
    
        encode_1 = self.point_1_a[encode_1]
        encode_2 = self.point_2_a[encode_2]
        
        encode = torch.where(
            state['qmask'],
            encode_1,
            encode_2
        )
        
        state['qtensor'] = encode

    def postprocess(self, state):
        qtensor: torch.Tensor = state['qtensor']
        
        qtensor_capped1 = torch.clamp(qtensor, 0, self.point_1.shape[0] - 1)
        qtensor_capped2 = torch.clamp(qtensor, 0, self.point_2.shape[0] - 1)
        
        quan = torch.where(
            state['qmask'],
            self.point_1[qtensor_capped1.long()] * state['scale_1'],
            self.point_2[qtensor_capped2.long()] * state['scale_2']
        )
        
        quan = torch.where(
            state['zeromask'],
            torch.zeros_like(quan),
            quan
        )
        
        state['tensor'] = quan.view_as(state['tensor']) * state['global_scale']

class quan(qbase):
    
    def get_qmask(self, flattened_abs, scale_1, scale_2):
        return flattened_abs > scale_2

class rquan(qbase):
    
    default_config = {
        'ratio': 0.01,
    }
    
    def get_qmask(self, flattened_abs, scale_1, scale_2):
        
        # size = flattened_abs.numel()
        
        sigma = (flattened_abs ** 2).mean().sqrt()
        
        threshold = sigma * erfinv(1 - self.config['ratio']) * (2 ** 0.5)
        
        # threshold = -torch.kthvalue(-flattened_abs, num, dim=0)[0]
        
        return flattened_abs >= threshold

class sigmaq(qbase):
    
    def get_qmask(self, flattened_abs, scale_1, scale_2):
        
        sigma = (flattened_abs ** 2).mean().sqrt()
        
        return flattened_abs >= sigma * 3

class olivewise(qbase):
    
    def __init__(self, parent: qbase, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
    
    def get_qmask_with_zero(self, flattened_abs, scale_1, scale_2):
        
        mask = self.parent.get_qmask(flattened_abs, scale_1, scale_2)

        tensor = flattened_abs
        
        numel, batch = mask.shape
        
        col = 2
        row = math.ceil(numel / col)
        
        padding = row * col - numel
        padded = torch.cat([mask, torch.zeros((padding, batch), device=self.device, dtype=torch.bool)], dim=0)
        padded_tensor = torch.cat([tensor, torch.zeros((padding, 1), device=self.device, dtype=torch.float32)], dim=0)
        reshaped = padded.view(row, col, batch)
        reshaped_tensor = padded_tensor.view(row, col, 1)
        
        row_count = reshaped.sum(dim=1)
        
        double_outlier = (row_count == 2)
        
        #reshaped &= ~double_outlier
        
        victim = row_count == 1
        
        #print(f'row_count = 1: {victim.half().mean()}')
        #print(f'row_count = 2: {double_outlier.half().mean()}')
        
        prune_first = reshaped_tensor[:, 0, :] < reshaped_tensor[:, 1, :]
        
        reshaped_zeromask = torch.zeros_like(reshaped)
        reshaped_zeromask[:, 0, :] = double_outlier & prune_first
        reshaped_zeromask[:, 1, :] = double_outlier & ~prune_first
        reshaped_zeromask |= victim.view(row, 1, batch) & ~reshaped
        
        #reshaped_zeromask[victim_nonzero[:, 0], :, victim_nonzero[:, 1]] = ~reshaped[victim_nonzero[:, 0], :, victim_nonzero[:, 1]]
        
        return (
            reshaped.view(-1, batch)[:numel, :].view_as(mask),
            reshaped_zeromask.view(-1, batch)[:numel, :].view_as(mask),
        )
        
        #reshaped_tensor[victim, 0] *= ~reshaped[victim, 1] # victim-outlier
        #reshaped_tensor[victim, 1] *= ~reshaped[victim, 0] # outlier-victim

        #state['tensor'] = reshaped_tensor.view(-1)[:numel].view_as(state['tensor'])

class rowreduce(modulebase):
    default_config = {
        'emu_col': 64,
        'reduce_ratio': 0.5
    }
    
    def preprocess(self, state):
        
        mask = state['qmask']
        numel = mask.numel()
        
        col = self.config['emu_col']
        row = math.ceil(numel / col)
        
        padding = row * col - numel
        padded = torch.cat([mask, torch.zeros(padding, device=self.device, dtype=torch.bool)], dim=0)
        reshaped = padded.view(row, col)
        
        row_count = reshaped.sum(dim=1)
        
        nonzero = row_count > 0
        
        row_nonzero = nonzero.sum()
        
        reduced = math.ceil(row_nonzero * self.config['reduce_ratio'])
        
        row_count += 2 * col * ~nonzero
        
        row_count_sorted = torch.sort(row_count)
        
        row_to_reduce = row_count_sorted.indices[:reduced]
        
        reshaped[row_to_reduce, :] = False
        
        state['qmask'] = reshaped.view(-1)[:numel].view_as(mask)

class rowreduce_new(modulebase):
    default_config = {
        'emu_col': 128,
        'reduce_threshold': 0.5
    }
    
    def preprocess(self, state):
        
        mask = state['qmask']
        tensor = state['tensor'].flatten() / state['global_scale']
        reduce_threshold = self.config['reduce_threshold']
        
        numel = mask.numel()
        
        col = self.config['emu_col']
        row = math.ceil(numel / col)
        
        padding = row * col - numel
        padded = torch.cat([mask, torch.zeros(padding, device=self.device, dtype=torch.bool)], dim=0)
        padded_tensor = torch.cat([tensor, torch.zeros(padding, device=self.device, dtype=torch.float32)], dim=0)
        reshaped = padded.view(row, col)
        reshaped_tensor = padded_tensor.view(row, col)
        
        outlier_tensor = reshaped_tensor * reshaped
        
        row_count = reshaped.sum(dim=1)
        outlier_max = outlier_tensor.abs().max(dim=1).values
        
        to_reduce = (row_count == 1) & (outlier_max < reduce_threshold)
        
        reshaped[to_reduce, :] = False
        
        state['qmask'] = reshaped.view(-1)[:numel].view_as(mask)

class adderror(modulebase):
    
    default_config = {
        'error_percent': 0,
        'error_low': 0,
        'error_high': 4
    }
        
    def preprocess(self, state):
        qtensor: torch.Tensor = state['tensor'].view(-1)

        n = qtensor.numel()

        bit_slice = torch.arange(self.config['error_low'], self.config['error_high'], device=self.device, dtype=torch.short).view(1, -1)

        bit_total = bit_slice.numel()

        err_mask = torch.rand((n, bit_total), device=self.device) < self.config['error_percent']

        err_mask = (err_mask * torch.pow(2, bit_slice)).sum(dim=1, dtype=torch.short).view_as(qtensor)
        
        err_fault = torch.randint_like(qtensor, 2).bool()

        #vmask = (1 << self.config['error_high']) - (1 << self.config['error_low'])

        state['err_xor'] = err_mask * err_fault
        state['err_xor2'] = err_mask * ~err_fault # 一半是stuck at right
    
    def process_error(self, state):
        qtensor: torch.Tensor = state['qtensor']
        err_xor: torch.Tensor = state['err_xor']
        state['qtensor'] = qtensor ^ err_xor

class adderrorq(adderror):
        
    default_config = {
        'error_on': 'outlier', # non-ourlier, all
        'error_total': 4
    }
    
    def preprocess(self, state):
        qmask = state['qmask']
        err_on = self.config['error_on']
        
        if err_on == 'non-outlier':
            qmask = ~qmask
        elif err_on == 'all':
            qmask = torch.ones_like(qmask)
            
        qprob = qmask.float().mean()
        
        qprob *= self.config['error_total'] / (self.config['error_high'] - self.config['error_low'])
        
        qtensor: torch.Tensor = state['tensor'].view(-1)

        n = qtensor.numel()

        bit_slice = torch.arange(self.config['error_low'], self.config['error_high'], device=self.device, dtype=torch.short).view(1, -1)

        bit_total = bit_slice.numel()

        err_mask = torch.rand((n, bit_total), device=self.device) < self.config['error_percent'] / qprob

        err_mask = (err_mask * torch.pow(2, bit_slice)).sum(dim=1, dtype=torch.short).view_as(qtensor)
        
        err_fault = torch.randint_like(qtensor, 2).bool()
        # 这里bug

        #vmask = (1 << self.config['error_high']) - (1 << self.config['error_low'])

        state['err_xor'] = err_mask * err_fault
    
    
class hashmodule(modulebase):
    
    default_config = {
        'hash': MinCI16384,
        'use_hash': 4,
        'faultmap_key': 'faultmap'
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        hash = self.config['hash']
        
        self.key_prefix = self.__class__.__name__ + '_'
        
        self.masking_bit = torch.tensor(hash, dtype=torch.int, device=self.device)
        self.maxbit: int = self.masking_bit.max().item() + 1
        self.fullhashcount, self.hashbit =  self.masking_bit.shape
        
        self.height = 1 << self.maxbit
        self.map_height = 1 << self.hashbit
        
        self.map_height_range = torch.arange(self.map_height, device=self.device, dtype=torch.int)
        self.height_range = torch.arange(self.height, device=self.device, dtype=torch.int)
        
        self.hashbit_range = torch.arange(self.hashbit, device=self.device, dtype=torch.int)
        
        one = torch.tensor(1, device=self.device, dtype=torch.int)
        
        self.bit_slice_hash = one << self.hashbit_range
        
        masking_bit_slice = one << self.masking_bit
        
        masking = masking_bit_slice.sum(dim=1)
        
        # (map_height, hashbit)
        sliced_bit = (self.map_height_range.view(-1, 1) & self.bit_slice_hash.view(1, -1)).bool()
        
        # (fullhashcount, map_height)
        map_mask = (masking_bit_slice.view(self.fullhashcount, 1, self.hashbit) * sliced_bit.view(1, self.map_height, self.hashbit)).sum(dim=2)
        
        # (fullhashcount, height)
        masked_height_range = self.height_range.view(1, self.height) & masking.view(self.fullhashcount, 1)
        
        # (fullhashcount, map_height, height)
        row_to_hash_table = map_mask.view(self.fullhashcount, self.map_height, 1) == masked_height_range.view(self.fullhashcount, 1, self.height)
        
        # _, _, fullhashcount * height
        _, _, hash_to_row = row_to_hash_table.nonzero(as_tuple=True)
        # _, _, fullhashcount * height
        _, _, row_to_hash = row_to_hash_table.transpose(1, 2).nonzero(as_tuple=True)
        
        self.hashcount = self.config['use_hash']
        
        self.row_to_hash = row_to_hash.view(self.fullhashcount, self.height)[:self.hashcount]
        self.hash_to_row = hash_to_row.view(self.fullhashcount, self.map_height, -1)[:self.hashcount]
    
        self.map_range = torch.arange(self.hashcount, device=self.device, dtype=torch.long)
        
    def get_mask(self, state) -> torch.BoolTensor:
        raise NotImplementedError

    def set_mask(self, state, value: torch.BoolTensor):
        raise NotImplementedError

    @property
    def faultmap_key(self):
        return self.config['faultmap_key']

    # update qmask or faultmask
    def preprocess(self, state):
        mask = self.get_mask(state)
        
        state[self.key_prefix + 'mask_shape'] = mask.shape
        state[self.key_prefix + 'mask_numel'] = numel = mask.numel()
        
        mask = mask.view(-1)
        
        row = self.height
        col = int(np.ceil(numel / row))
        
        padding = row * col - numel
        padded = torch.cat([mask, torch.zeros(padding, device=self.device, dtype=torch.bool)], dim=0)
        
        if not self.faultmap_key in state:
            state[self.faultmap_key] = torch.zeros((self.fullhashcount, self.map_height, col), device=self.device, dtype=torch.bool)

        # (hashcount, map_height, ?, col)
        update = padded.view(row, col)[self.hash_to_row, :]
        update = update.sum(dim=2, dtype=torch.bool)
        
        state[self.faultmap_key][:self.hashcount] |= update
        
    def process_hash(self, state):
        
        mask = state[self.faultmap_key][
            self.map_range.view(-1, 1, 1),
            self.row_to_hash.view(self.hashcount, self.height, 1)
        ]
        
        mask = mask.prod(dim=0, dtype=torch.bool)
        
        mask = mask.view(-1)[:state[self.key_prefix + 'mask_numel']]
        self.set_mask(state, mask.view(state[self.key_prefix + 'mask_shape']))

class hashq(hashmodule):
    def __init__(self, **kwargs):
        kwargs['use_hash'] = 3
        super().__init__(**kwargs)
    
    def get_mask(self, state) -> torch.BoolTensor:
        return state['qmask']
    
    def set_mask(self, state, value: torch.BoolTensor):
        print(f'hashq mask: {state["qmask"].half().mean()} -> {value.half().mean()}')
        state['qmask'] = value

class hashqwise(qbase):
    pass # todo

class hasherr(hashmodule):
    
    default_config = {
        'use_hash': 4
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_mask(self, state) -> torch.BoolTensor:
        return state['err_xor'].bool()
    
    def set_mask(self, state, value: torch.BoolTensor):
        state['err_xor'] *= ~value

class fullerr(hasherr):
    
    default_config = {
        'correct_error_low': 0,
        'correct_error_high': 4,
        'col': 512,
        'fame_count': 4
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.row_range = torch.arange(self.config['col'], device=self.device, dtype=torch.long).view(1, -1)
        self.bit_mask = 2 ** self.config['correct_error_high'] - 2 ** self.config['correct_error_low']
    
    def preprocess(self, state):
        state['qmask_0'] = state['qmask']
        return super().preprocess(state)

    def get_mask(self, state) -> torch.BoolTensor:
        return (state['err_xor'] & self.bit_mask).bool() | (state['qmask_0'] & state['err_xor'].bool())
    
    def set_mask(self, state, value: torch.BoolTensor):
        #self.print(f'fullerr: all={value.numel()}, sum={value.sum()}')

        # sort (outlier, 1~4)

        numel = value.numel()
        
        line_width = self.config['col']

        row_count = math.ceil(numel / line_width)

        pad = row_count * line_width - numel

        padded = torch.cat([value, torch.zeros(pad, dtype=torch.bool, device=self.device)]).view(row_count, line_width)
        qmask_padded = torch.cat([state['qmask_0'], torch.zeros(pad, dtype=torch.bool, device=self.device)]).view(row_count, line_width)
        #error_padded = torch.cat([state['err_xor'], torch.zeros(pad, dtype=torch.int16, device=self.device)]).view(row_count, line_width)

        def get_error_to_correct():
            yield padded & qmask_padded
            yield padded & ~qmask_padded
            #for k in range(self.config['correct_error_high'] - 1, self.config['correct_error_low'] - 1, -1):
            #    yield padded & (error_padded & (1 << k)).bool()
        
        correct_mask = torch.zeros_like(padded)

        fame_available = torch.ones((row_count, 1), dtype=torch.short, device=self.device) * self.config['fame_count']

        for to_correct in get_error_to_correct():
            error_to_use = (to_correct & ~correct_mask).sum(dim=1).view(-1, 1) # how many error we're going to correct in this stage?
            exact_fame_to_use = torch.minimum(error_to_use, fame_available)

            avialable_mask = line_width * exact_fame_to_use >= self.row_range * error_to_use

            correct_mask |= to_correct & avialable_mask

            fame_available -= exact_fame_to_use

        uncorrected_outlier_count = (qmask_padded & padded & ~correct_mask).sum(dim=1)

        correct_mask = correct_mask.view(-1)[:numel]


        max_uncorrected = uncorrected_outlier_count.max()
        uncorrected_count = uncorrected_outlier_count.bool().sum()

        self.print(f'map shape={padded.shape}, uncorrected={max_uncorrected}x{uncorrected_count} overhead={uncorrected_count * max_uncorrected / padded.numel()}')

        correct_mask |= state['qmask_0']
        
        state['zeromask'] = value & ~correct_mask & ~state['qmask_0']

        prev = (state['err_xor'] & self.bit_mask).bool().sum()
        prev_ot = ((state['err_xor'] & self.bit_mask).bool() & state['qmask_0']).sum()

        state['err_xor'] *= ~correct_mask
        
        bitxor = (state['err_xor'] & self.bit_mask).bool()

        z = state['zeromask'].sum()

        self.print(f'fullerr: {prev}({prev_ot}) => {bitxor.sum()}({(bitxor & state["qmask_0"]).sum()}), zero={z}')

        #reshaped = padded.view(row, col)
        
        #max_value_row = reshaped.sum(dim=1).max()
        
        #self.print(f'fullerr: max_value_row={max_value_row}')
        
class weightzeroerr(hasherr):
    def set_mask(self, state, value: torch.BoolTensor):
        state['zeromask'] = value

class weightzeroerr_v2(hasherr):
    def set_mask(self, state, value: torch.BoolTensor):
        state['zeromask'] = value & ~state['qmask_0']
        mean = (state['err_xor'].bool() & value & state['qmask_0']).float().mean()
        self.print(f'weightzero: total corrected={mean}')
        state['err_xor'] *= ~(value & state['qmask_0'])
        
    def preprocess(self, state):
        state['qmask_0'] = state['qmask']
        return super().preprocess(state)

    
class showmehasherr(hasherr):
    def set_mask(self, state, value: torch.BoolTensor):
        state['hasherr'] = value

class wesco(modulebase):
    
    default_config = {
        'encoding': [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [2, 3, 0, 1],
            [1, 0, 3, 2]
        ],
        'col': 64
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.encoding = torch.tensor(self.config['encoding'], device=self.device, dtype=torch.long)
        
        self.encoding_count, self.bit_width = self.encoding.shape
        
        one = torch.tensor(1, device=self.device, dtype=torch.int16)
        
        bit_range = torch.arange(self.bit_width, device=self.device, dtype=torch.int16)
        
        self.bit_slicer = one << bit_range
        
        range = torch.arange(1 << self.bit_width, device=self.device, dtype=torch.int16)
        
        # (2^bitwidth) x bitwidth
        self.sliced_range = (self.bit_slicer.view(1, -1) & range.view(-1, 1)).bool()
        
        # encoding_range = torch.arange(self.encoding_count, device=self.device, dtype=torch.int)
        
        # encoding_count x bitwidth x (2^bitwidth)
        # self.exchanged_slice = self.sliced_range[:, self.encoding]
        
        # encoding_count x (2^bitwidth)
        # self.encoding_table = (self.exchanged_slice * self.bit_slicer).sum(dim=1)
        
        
    def process(self, state):
        
        tensor: torch.Tensor = state['qtensor']
        err: torch.Tensor = state['err_xor']
        err2: torch.Tensor = state['err_xor2']
        
        numel = tensor.numel()
        
        col = self.config['col']
        row = math.ceil(numel / col)
        
        padding = row * col - numel
        padded_tensor = torch.cat([tensor.flatten(), torch.zeros(padding, device=self.device, dtype=torch.short)], dim=0)
        padded_err = torch.cat([err.flatten(), torch.zeros(padding, device=self.device, dtype=torch.short)], dim=0)
        padded_err2 = torch.cat([err2.flatten(), torch.zeros(padding, device=self.device, dtype=torch.short)], dim=0)
        
        del err
        
        reshaped_tensor = padded_tensor.view(row, col)
        reshaped_err = padded_err.view(row, col) # stuck at fault
        reshaped_err2 = padded_err2.view(row, col) # stuck at right
        
        stuck_at = ~reshaped_tensor & reshaped_err | reshaped_tensor & reshaped_err2 # stuck at value

        # tensor = 0: stuck at 1 <===> stuck at fault
        # tensor = 1: stuck at 1 <===> stuck at right

        stuck_at_mask = reshaped_err | reshaped_err2 # stuck at (fault or right)

        #del reshaped_err, reshaped_err2
        
        bit_slicer = self.bit_slicer.view(-1, 1, 1)

        # (bitwidth, row, col)
        sliced_tensor = (bit_slicer & reshaped_tensor).bool()
        
        #del reshaped_tensor
        
        # (bitwidth, row, col)
        sliced_stuck_at = (bit_slicer & stuck_at).bool()
        
        #del stuck_at

        # (bitwidth, row, col)
        sliced_stuck_at_mask = (bit_slicer & stuck_at_mask).bool()

        #del stuck_at_mask
        
        # (encoding_count, bitwidth, row, col)
        exchanged_stuck_at_mask = sliced_stuck_at_mask[self.encoding]
        
        #del sliced_stuck_at_mask
        
        # (encoding_count, bitwidth, row, col)
        exchanged_stuck_at = sliced_stuck_at[self.encoding]
        
        #del sliced_stuck_at
        
        # unmatches must be stuck-ats whose value differ from tensor at the same position

        # (encoding_count, bitwidth, row, col)
        unmatch_sum = exchanged_stuck_at_mask & (sliced_tensor.view(1, self.bit_width, row, col) ^ exchanged_stuck_at)

        # to save place, bitwidth won't be more than 256
        unmatch_sum = unmatch_sum.sum(dim=1, dtype=torch.int8).sum(dim=2, dtype=torch.int32)
        
        row_encoding = unmatch_sum.min(dim=0).indices
        
        #del unmatch_sum
        
        row_range = torch.arange(row, dtype=torch.long, device=self.device)
        
        encoded_stuck_at_mask = exchanged_stuck_at_mask[row_encoding, :, row_range]
        
        #del exchanged_stuck_at_mask
        
        encoded_stuck_at = exchanged_stuck_at[row_encoding, :, row_range]
        
        #del exchanged_stuck_at
        
        # (row, bitwidth, col)
        injected_sliced_tensor = torch.where(encoded_stuck_at_mask, encoded_stuck_at, sliced_tensor.transpose(0, 1))
        
        injected_tensor = (injected_sliced_tensor * self.bit_slicer.view(1, -1, 1)).sum(dim=1, dtype=torch.int16)
        
        state['err_xor'] = injected_tensor.view(-1)[:numel].view_as(tensor) ^ tensor
        
    # 1 2 3 4
    # 4 3 2 1
    # 3 4 1 2
    # 2 1 4 3
    # => min(bit sum)

class bitzeroerr(hasherr):
    
    default_config = {
        'correct_error_low': 0,
        'correct_error_high': 3
    }
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        i_map = torch.arange(16, device=self.device).unsqueeze(dim=1)
        j_map = torch.arange(16, device=self.device).unsqueeze(dim=0)
        xor_map = j_map ^ i_map
        
        self.table = torch.where(
            (i_map < 8) ^ (j_map > 8) | (torch.abs(i_map - 8) < torch.abs(j_map - 8)),
            torch.zeros_like(xor_map),
            xor_map
        ).short()

    def set_mask(self, state, value: torch.BoolTensor):
        val = state['qtensor']
        other = val ^ (state['err_xor'] & value)
        
        correct_mask = 2 ** self.config['correct_error_high'] - 2 ** self.config['correct_error_low']
        
        state['err_xor'] = (correct_mask & self.table[val, other.long()]) | (correct_mask & state['err_xor'])

class qmgr:
    
    def __init__(self, modules: Iterable[modulebase], printer):
        self.modules = list(modules)
        
        for module in modules:
            module.print = printer
        self.state = None
        
    def process(self, tensor: torch.Tensor):
        if not self.state:
            # print(f'initializing: {tensor.shape}')
            state = {
                'tensor': tensor
            }
            for module in self.modules:
                module.once(state)
            
            state.pop('tensor')
            self.state = state
        
        state = self.state.copy()
        state['tensor'] = tensor
        
        # print(f'processing: {tensor.shape}')
        
        for module in self.modules:
            module.preprocess(state)
        
        for module in self.modules:
            module.process_hash(state)
            
        for module in self.modules:
            module.process(state)
        
        for module in self.modules:
            module.process_error(state)
        
        for module in self.modules:
            module.postprocess(state)
        
        # print(f'result: {state}')
        
        return state
        
'''
class wrapper:
    
    def __init__(self, modules: Iterable[modulebase]):
        self.modules = list(modules)
    
    def process(self, module: torch.nn.Module, state_handler = None):
        # statically wrap weight
        
        for param in module.parameters():
            if param.dim() & 1: continue
            q = qmgr(self.modules)
            state = q.process(param.data)
            if state_handler is not None: state_handler(state)
            param.data = state['tensor']
        
    def wrap(self, module: torch.nn.Module):
        
        self.process(module)
        # dynamically wrap activation
        
        old = module.forward
        mgr = qmgr(self.modules)
        
        def wrapped_forward(self, **args):
            return mgr.process(old(self, **args))
        
        setattr(module, 'forward', wrapped_forward)
'''

'''
设计:
1. 按bit_width分组权重的bit作为mask输入
2. mask输出再补0变成nx512
3. 按照每行的总数量平均取值
4. 恢复到原来的数据
'''
class flower(hashmodule):

    default_config = {
        'bit_width': 4,
        'line_width': 512,
        'fame_count': 4
    }

    def __init__(self, **kwargs):
        kwargs['use_hash'] = 4
        kwargs['faultmap_key'] = 'flowermap'
        kwargs['hash'] = MinCI16384_flower

        super().__init__(**kwargs)

        self.bit_slicer = torch.pow(2, torch.arange(0, self.config['bit_width'], device=self.device, dtype=torch.long)).view(1, -1)
        self.row_range = torch.arange(self.config['line_width'], device=self.device, dtype=torch.long).view(1, -1)

    def get_mask(self, state) -> torch.BoolTensor:
        err_xor = state['err_xor'].view(-1, 1) # n*1

        sliced_err = (err_xor & self.bit_slicer).bool().view(-1) # n*bit_width

        return sliced_err
    
    def set_mask(self, state, value: torch.BoolTensor):
        
        bitn = value.numel() # bitn = n * bit_width

        line_width = self.config['line_width']

        row_count = math.ceil(bitn / line_width)

        pad = row_count * line_width - bitn

        padded = torch.cat([value, torch.zeros(pad, dtype=torch.bool, device=self.device)]).view(row_count, line_width)

        line_error_count = padded.sum(dim=1).view(-1, 1)

        correct_mask = line_width * self.config['fame_count'] >= self.row_range * line_error_count

        sliced_correct = (value & correct_mask.view(-1))[:bitn].view(state['err_xor'].numel(), self.config['bit_width'])

        corrected = (sliced_correct * self.bit_slicer).sum(dim=1).view_as(state['err_xor'])

        prev = state['err_xor'].bool().sum()
        state['err_xor'] &= ~corrected 

        self.print(f'flower: {prev} -> {state["err_xor"].bool().sum()}')

class int4rquan(rquan):
    default_config = {
        'ratio': 0.0034,
        'encoding_1': 'int4'
    }

class flintquan(sigmaq):
    default_config = {
        'encoding_1': 'flint'
    }
