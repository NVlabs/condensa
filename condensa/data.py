# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
import torch
import torch.utils.data as data
import PIL

def fast_collate(batch):
    """Fast batch collation. Based on version from
       NVIDIA Apex: https://github.com/NVIDIA/apex."""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

class GPUDataLoader(object):
    """Custom data loader with support for prefetching and fast collation.
       Based on version from NVIDIA Apex: https://github.com/NVIDIA/apex."""
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 num_workers,
                 sampler=None,
                 meanstd=None):
        if isinstance(dataset[0][0], PIL.Image.Image):
            nc = len(dataset[0][0].getbands())
        else:
            raise RuntimeError(
                '[Condensa] GPUDataLoader only supports PIL image datasets')

        if not torch.cuda.is_available():
            raise RuntimeError(
                '[Condensa] GPUDataLoader requires PyTorch CUDA support')

        if nc == 3:
            loader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     pin_memory=True,
                                     sampler=sampler,
                                     collate_fn=fast_collate,
                                     num_workers=num_workers)
        else:
            raise NotImplementedError(
                '[Condensa] GPUDataLoader currently only supports 3-channel images'
            )

        self.base_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if meanstd is not None:
            mean, std = meanstd
            self.mean = torch.tensor([x * 255
                                      for x in mean]).cuda().view(1, nc, 1, 1)
            self.std = torch.tensor([x * 255
                                     for x in std]).cuda().view(1, nc, 1, 1)
        self.preload()

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        self.loader = iter(self.base_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is None and target is None:
            raise StopIteration
        input.record_stream(torch.cuda.current_stream())
        target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

            self.next_input = self.next_input.float()
            if self.mean is not None and self.std is not None:
                self.next_input = self.next_input.sub_(self.mean).div_(
                    self.std)
