import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
import pdb
from numpy.random import randint
from temporal_transforms import ReverseFrames, ShuffleFrames

from multiprocessing.dummy import Pool as ThreadPool


class VideoRecord(object):

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet_3D(data.Dataset):

    def __init__(self,
                 root_path,
                 list_file,
                 num_segments_lst=[3],
                 new_length=1,
                 modality='RGB',
                 image_tmpl='img_{:05d}.jpg',
                 temp_transform=None,
                 transform=None,
                 random_shift=True,
                 gap=2,
                 dataset='something',
                 dense_sample=False,
                 shift_val=None):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments_lst[0]
        self.num_segments_lst = num_segments_lst
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.temp_transform = temp_transform
        self.transform = transform
        self.random_shift = random_shift
        self.gap = gap
        self.dataset = dataset
        self.dense_sample = dense_sample
        self.shift_val = shift_val

        if self.dense_sample:
            print('using dense sample for training!')
        else:
            print('using sparse sample for training!')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.dataset == 'something_v2':
            return [
                Image.open(os.path.join(
                    directory, self.image_tmpl.format(idx))).convert('RGB')
            ]
        elif self.modality == 'Flow':
            x_img = Image.open(
                os.path.join(directory,
                             self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(
                os.path.join(directory,
                             self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [
            VideoRecord(x.strip().split(' ')) for x in open(self.list_file)
        ]

    def _sample_indices(self, record):
        # ==== new version
        if record.num_frames >= self.num_segments + self.new_length - 1:
            avg_dur = (record.num_frames - self.new_length + 1) / float(
                self.num_segments)
            indices = np.multiply(
                list(range(self.num_segments)),
                avg_dur).round().astype(int) + \
                randint(avg_dur, size=self.num_segments)
        else:
            indices = np.sort(
                randint(record.num_frames - self.new_length + 1,
                        size=self.num_segments))
        return indices + 1

    def _sample_dense_indices(self, record, gap=2):
        average_duration = (record.num_frames - \
                self.new_length + 1) // self.num_segments
        if average_duration > (gap - 1):  # num_frames >= 2*num_segments
            start_f = randint(record.num_frames-\
                    gap*self.new_length*self.num_segments+1)
            offsets = np.array(
                list(
                    range(start_f,
                          start_f + gap * self.new_length * self.num_segments,
                          gap * self.new_length)))
        elif record.num_frames >= self.num_segments:
            offsets = np.sort(randint(record.num_frames - \
                    self.new_length + 1, size=self.num_segments))
        else:
            pre_n_zeros = randint((self.num_segments - record.num_frames))
            post_n_zeros = self.num_segments - record.num_frames - pre_n_zeros
            offsets = np.array([0]*pre_n_zeros+list(range(record.num_frames))+\
                    [record.num_frames-self.new_length]*post_n_zeros)
        return offsets + 1

    def _sample_dense_val_indices(self, record, gap=2):
        average_duration = (record.num_frames - \
                self.new_length + 1) // self.num_segments
        if average_duration > (gap - 1):  # num_frames >= 2*num_segments
            start_f = (record.num_frames-\
                    gap*self.new_length*self.num_segments+1)//2
            offsets = np.array(
                list(
                    range(start_f,
                          start_f + gap * self.new_length * self.num_segments,
                          gap * self.new_length)))
        elif record.num_frames > self.num_segments:
            offsets = np.array(
                list(
                    range(0, self.new_length * self.num_segments,
                          self.new_length)))
        else:
            pre_n_zeros = (self.num_segments - record.num_frames) // 2
            post_n_zeros = self.num_segments - record.num_frames - pre_n_zeros
            offsets = np.array([0]*pre_n_zeros+list(range(record.num_frames))+\
                    [record.num_frames-1]*post_n_zeros)
        return offsets + 1

    def _get_shift_dense_val_indices(self, record, gap=2, shift_val=4):
        average_duration = (record.num_frames - \
                self.new_length + 1) // self.num_segments
        if average_duration > (gap - 1):  # num_frames >= 2*num_segments
            shift_len = (record.num_frames - \
                    gap*self.new_length*self.num_segments)//shift_val
            if shift_len > 0:
                offsets = []
                for i in range(shift_val):
                    offsets.extend(
                        list(
                            range(
                                i * shift_len, i * shift_len +
                                gap * self.new_length * self.num_segments,
                                gap * self.new_length)))
                offsets = np.array(offsets)
            else:
                start_f = (record.num_frames-\
                        gap*self.new_length*self.num_segments+1)//2
                offsets = np.array((list(
                    range(start_f,
                          start_f + gap * self.new_length * self.num_segments,
                          gap * self.new_length))) * shift_val)
        elif record.num_frames > self.num_segments:
            offsets = np.array(
                list(
                    range(0, self.new_length * self.num_segments,
                          self.new_length)) * shift_val)
        else:
            pre_n_zeros = (self.num_segments - record.num_frames) // 2
            post_n_zeros = self.num_segments - record.num_frames - pre_n_zeros
            offsets = np.array(shift_val*([0]*pre_n_zeros+list(
                range(record.num_frames))+\
                    [record.num_frames-1]*post_n_zeros))
        return offsets + 1

    def _get_val_indices(self, record, n_cframes):
        if record.num_frames >= n_cframes + self.new_length - 1:
            tick = (record.num_frames - self.new_length + \
                    1) / float(n_cframes)
            offsets = np.array([int(tick / 2.0 + \
                    tick * x) for x in range(n_cframes)])
        elif record.num_frames >= n_cframes:
            offsets = np.array(list(range(n_cframes)))
        else:
            pre_n_zeros = (n_cframes - record.num_frames) // 2
            post_n_zeros = n_cframes - record.num_frames - pre_n_zeros
            offsets = np.array([0]*pre_n_zeros+list(range(record.num_frames))+\
                    [record.num_frames-1]*post_n_zeros)
        return offsets + 1

    def _get_shift_val_indices(self, record, n_cframes, n_shift=1):
        if record.num_frames >= n_cframes * n_shift + self.new_length - 1:
            tick = (record.num_frames - self.new_length + \
                    1) / float(n_cframes) # >n_shift
            offsets = []
            gap = int(tick // n_shift)
            if (n_shift - 1) * (gap + 1) < int(tick):
                gap = gap + 1
            else:
                gap = gap
            for start_i in range(0, gap * n_shift, gap):
                offsets.extend(
                    [int(start_i + tick * x) for x in range(n_cframes)])
            offsets = np.array(offsets)
        elif record.num_frames >= n_cframes + self.new_length - 1:
            tick = (record.num_frames - self.new_length + \
                    1) / float(n_cframes)
            offsets = np.array(n_shift*[int(tick / 2.0 + \
                    tick * x) for x in range(n_cframes)])
        elif record.num_frames >= n_cframes:
            offsets = np.array(n_shift * list(range(n_cframes)))
        else:
            offsets = np.tile(
                np.array(range(record.num_frames - self.new_length + 1)),
                ((n_cframes - self.new_length + 1) //
                 (record.num_frames - self.new_length + 1), 1))
            offsets = offsets.flatten('F')
            offsets = np.pad(offsets, (0, n_cframes - len(offsets)), 'maximum')
            offsets = np.tile(offsets, n_shift)
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.random_shift:
            if not self.dense_sample:
                segment_indices = self._sample_indices(record)
            else:
                segment_indices = self._sample_dense_indices(record, self.gap)
        else:
            if not self.dense_sample:
                if self.shift_val:
                    segment_indices = self._get_shift_val_indices(
                        record, self.num_segments_lst[0], self.shift_val)
                else:
                    segment_indices = np.array([], dtype=int)
                    for i in self.num_segments_lst:
                        segment_indices = np.append(segment_indices,
                                                    self._get_val_indices(
                                                        record, i),
                                                    axis=0)
            else:
                if self.shift_val:
                    segment_indices = self._get_shift_dense_val_indices(
                        record, self.gap, self.shift_val)
                else:
                    segment_indices = self._sample_dense_val_indices(
                        record, self.gap)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        idx_list = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                idx_list.append(p)
                if p < record.num_frames:
                    p += 1

        process_idx_list = self.temp_transform(idx_list)
        for p in process_idx_list:
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
        process_data = self.transform(images)
        record_name = record.path.strip().split('/')[-1]
        return record_name, process_data, record.label, process_idx_list

    def __len__(self):
        return len(self.video_list)
