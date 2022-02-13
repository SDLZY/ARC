import torch
from torch.utils.data import Dataset, DataLoader
from allennlp.nn.util import get_text_field_mask
from allennlp.data.dataset import Batch


class MultiDataset(Dataset):
    def __init__(self, *args):
        super(MultiDataset, self).__init__()
        self.datasets = args

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        return [dset[index] for dset in self.datasets]

    @property
    def is_train(self):
        return self.datasets[0].split == 'train'


def collate_fn(data_list, to_gpu=False):
    """Creates mini-batch tensors
    """
    tds = []
    # print(data_list)
    for data in zip(*data_list):
        images, instances = zip(*data)
        images = torch.stack(images, 0)
        batch = Batch(instances)
        td = batch.as_tensor_dict()
        if 'question' in td:
            td['question_mask'] = get_text_field_mask(td['question'], num_wrapping_dims=1)
            td['question_tags'][td['question_mask'] == 0] = -2  # Padding

        td['answer_mask'] = get_text_field_mask(td['answers'], num_wrapping_dims=1)
        td['answer_tags'][td['answer_mask'] == 0] = -2

        td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
        td['images'] = images
        tds.append(td)

    return tds


class VCRLoaderMultiDataset(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, batch_size=3, num_workers=6, num_gpus=3, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            # shuffle=False,
            shuffle=data.is_train,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, to_gpu=False),
            pin_memory=False,
            drop_last=data.is_train,
            **kwargs,
        )
        return loader