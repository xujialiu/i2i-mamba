import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert opt.resize_or_crop == "resize_and_crop"

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize(
            (self.opt.loadSize, self.opt.loadSize), Image.BICUBIC
        )
        B = AB.crop((w2, 0, w, h)).resize(
            (self.opt.loadSize, self.opt.loadSize), Image.BICUBIC
        )
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[
            :,
            h_offset : h_offset + self.opt.fineSize,
            w_offset : w_offset + self.opt.fineSize,
        ]
        B = B[
            :,
            h_offset : h_offset + self.opt.fineSize,
            w_offset : w_offset + self.opt.fineSize,
        ]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == "BtoA":
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            tmp = A[1, ...]
            A = tmp.unsqueeze(0)
        elif input_nc == 2:
            tmp = A[0:2, ...]
            A = tmp

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            tmp = B[1, ...]
            B = tmp.unsqueeze(0)
        elif output_nc == 2:
            tmp = B[0:2, ...]
            B = tmp
        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return "AlignedDataset"


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert opt.resize_or_crop == "resize_and_crop"

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        B = AB
        A = Image.fromarray(gen_move_img(np.array(B)))

        # w, h = AB.size

        # w2 = int(w / 2)
        A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[
            :,
            h_offset : h_offset + self.opt.fineSize,
            w_offset : w_offset + self.opt.fineSize,
        ]
        B = B[
            :,
            h_offset : h_offset + self.opt.fineSize,
            w_offset : w_offset + self.opt.fineSize,
        ]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == "BtoA":
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            tmp = A[1, ...]
            A = tmp.unsqueeze(0)
        elif input_nc == 2:
            tmp = A[0:2, ...]
            A = tmp

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            tmp = B[1, ...]
            B = tmp.unsqueeze(0)
        elif output_nc == 2:
            tmp = B[0:2, ...]
            B = tmp
        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return "AlignedDataset"


def gen_move_img(img, shift_counts=[3, 10], shift_size=[1, 30], shift_num_cols=[3, 30]):
    """
    生成带有运动伪影的OCTA图像.

    参数:
    - img: np.array, 输入的灰度图像.
    - shift_count: int, 发生偏移的次数.
    - shift_size: int或包含两个元素的可迭代对象, 偏移的大小值或范围 (像素) .
    - shift_num: int或包含两个元素的可迭代对象, 每次连续偏移的行数值或范围.

    返回值:
    - np.array, 带有运动伪影的图像.
    """
    # 复制输入图像, 避免修改原图像
    image = img.copy()
    h, w, c = image.shape

    shift_counts = random.randint(*shift_counts)

    for _ in range(shift_counts):
        # 随机选择起始行索引, 确保不超出图像边界
        max_start_idx = h - 1
        start_idx = random.randint(0, max_start_idx)

        # 确定偏移的行数shift_num
        num_rows_to_shift = random.randint(shift_num_cols[0], shift_num_cols[1])

        # 调整num_rows_to_shift, 确保不超出图像高度
        num_rows_to_shift = min(num_rows_to_shift, h - start_idx)

        # 获取当前需要偏移的行索引列表
        rows_to_shift = list(range(start_idx, start_idx + num_rows_to_shift))

        # 确定偏移大小shift_size
        shift_pixels = random.randint(shift_size[0], shift_size[1])

        # 随机选择偏移方向
        shift_direction = random.choice([-1, 1])  # -1表示左移, 1表示右移
        shift = shift_pixels * shift_direction

        # 对选定的行进行偏移
        for row_idx in rows_to_shift:
            shifted_row = np.zeros_like(image[row_idx])
            if shift > 0:
                # 向右移动
                if shift < w:
                    shifted_row[shift:] = image[row_idx][:-shift]
                else:
                    # 如果偏移量超过宽度, 整行置零
                    shifted_row[:] = 0
            elif shift < 0:
                # 向左移动
                shift_abs = -shift
                if shift_abs < w:
                    shifted_row[:-shift_abs] = image[row_idx][shift_abs:]
                else:
                    # 如果偏移量超过宽度, 整行置零
                    shifted_row[:] = 0
            else:
                # 不移动
                shifted_row = image[row_idx].copy()

            # 将偏移后的行替换回图像
            image[row_idx] = shifted_row

    return image
