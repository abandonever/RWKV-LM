import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset

from .binidx import MMapIndexedDataset


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        if args.my_qa_mask == 1:
            self.data_mask = MMapIndexedDataset(args.data_file.replace("_text_", "_mask_"))
            self.data_mask_size = len(self.data_mask._bin_buffer) // self.data_mask._index._dtype_size
            assert self.data_size == self.data_mask_size
            assert len(self.data) == len(self.data_mask)
        else:
            self.data_mask = None
            self.data_mask_size = 0

        self.data_count = len(self.data)
        self.data_offsets = []

        epoch_count = self.args.epoch_count
        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        self.sample_total = epoch_count * self.samples_per_epoch

        rank_zero_info(f"Sample per epoch = {self.samples_per_epoch} (epoch_steps * real_bsz)")
        rank_zero_info(f"Sample total = {self.sample_total} (epoch_count * epoch_steps * real_bsz), epoch count = {epoch_count}")

        self.ctx_len = args.ctx_len
        self.req_len = self.ctx_len + 1

        # 第一段训练数据区间
        buf_offset = 0
        buf_end = buf_offset + self.req_len
        self.data_offsets.append(buf_offset)

        for d_ptr, d_size in self.data._index:
            # 缓冲区的截止位置已经超过了数据的尾部
            if buf_end >= self.data_size:
                break

            # 数据的头位置
            d_offset = d_ptr // self.data._index._dtype_size

            # 数据的头部不在缓冲区头部的后面，前进到下一个数据
            if d_offset <= buf_offset:
                continue

            # 数据的尾位置
            d_end = d_offset + d_size

            # 数据的尾部在缓冲区尾部的前面，前进到下一个数据
            if d_end < buf_end:
                continue

            # 新的一段训练数据区间
            buf_offset = d_offset
            buf_end = buf_offset + self.req_len
            self.data_offsets.append(buf_offset)

        rank_zero_info(f"Unique sample size = {len(self.data_offsets)} (real sample count in dataset)")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        args = self.args

        data = self.data
        data_mask = self.data_mask

        # 是否做随机采样
        do_random_sample = False

        if idx >= len(self.data_offsets):
            do_random_sample = True
        elif self.req_len > (self.data_size - self.data_offsets[idx]):
            do_random_sample = True

        if self.samples_per_epoch < self.sample_total:
            do_random_sample = True

        ii = []
        ll = []
        dd = []

        if do_random_sample:
            dd_len = 0
            while dd_len < self.req_len:
                d_idx = np.random.choice(self.data_count)
                d_pointer, d_size = data._index[d_idx]
                d_i = d_pointer // data._index._dtype_size
                d_len = min(d_size, self.req_len-dd_len)

                ii.append(d_i)
                ll.append(d_len)
                dd_len += d_len
        else:
            i = self.data_offsets[idx]
            ii.append(i)
            ll.append(self.req_len)

        for i, l in zip(ii, ll):
            dix = data.get(idx=0, offset=i, length=l).astype(int)
            dd.append(dix)

        dix = np.concatenate(dd)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        if args.my_qa_mask == 1:
            mask_dd = []
            for i, l in zip(ii, ll):
                mask_ix = data_mask.get(idx=0, offset=i, length=l).astype(int)
                mask_dd.append(mask_ix)

            mask_ix = np.concatenate(mask_dd)
            mask_ix[-1] = 1

            z = torch.tensor(mask_ix[1:], dtype=torch.bfloat16)

            return x, y, z

        return x, y
