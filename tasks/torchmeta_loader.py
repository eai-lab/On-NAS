""" Script for On-NAS & Two-Fold Meta-learning(TFML) & On-NAS

This code have been written for a research purpose. 

Licenses and code references will be added at camera-ready version of the code. 

"""
import torchvision.datasets as dset
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import miniimagenet

from tasks.core import TaskDistribution, Task


def sample_meta_batch(
    batch_iter, meta_batch_size, task_batch_size, shots, ways, task_train_sampler=None
):
    """Sample a meta batch using a torchmeta :class:`BatchMetaDataLoader`

    Args:
        batch_iter: Iterator for the :class:`BatchMetaDataLoader`
        meta_batch_size: Number of tasks per meta-batch
        task_batch_size: Number of elements in a task batch
        shots: Number of samples per class
        ways: Number of classes per task
        task_train_sampler: Samples for the meta batch train dataset

    Returns:
        A list of data loaders for training, validation and testing for one task.
        Currently, the validation loader is the same as the training loader.
    """
    
    batch = next(batch_iter)
    train_batch_x, train_batch_y = batch["train"]
    test_batch_x, test_batch_y = batch["test"]
    num_tasks = meta_batch_size
    meta_train_batch = list()
    for task_idx in range(num_tasks):
        dset_train = TensorDataset(train_batch_x[task_idx], train_batch_y[task_idx])
        dset_val = TensorDataset(test_batch_x[task_idx], test_batch_y[task_idx])
        train_loader = DataLoader(
            dset_train, batch_size=task_batch_size, shuffle=True #sampler=task_train_sampler
        )
        test_loader = DataLoader(dset_val, batch_size=shots * ways)
        meta_train_batch.append(Task(train_loader, train_loader, test_loader))
    return meta_train_batch

def use_whole_batch(
    batch_iter, meta_batch_size, task_batch_size, shots, ways, task_train_sampler=None
):
    """Sample a meta batch using a torchmeta :class:`BatchMetaDataLoader`

    Args:
        batch_iter: Iterator for the :class:`BatchMetaDataLoader`
        meta_batch_size: Number of tasks per meta-batch
        task_batch_size: Number of elements in a task batch
        shots: Number of samples per class
        ways: Number of classes per task
        task_train_sampler: Samples for the meta batch train dataset

    Returns:
        A list of data loaders for training, validation and testing for one task.
        Currently, the validation loader is the same as the training loader.
    """
    
    batch = next(batch_iter)
    train_batch_x, train_batch_y = batch["train"]
    test_batch_x, test_batch_y = batch["test"]
    num_tasks = meta_batch_size

    meta_train_batch = list()
    for task_idx in range(num_tasks):
        dset_train = TensorDataset(train_batch_x[task_idx], train_batch_y[task_idx])
        dset_val = TensorDataset(test_batch_x[task_idx], test_batch_y[task_idx])
        train_loader = DataLoader(
            dset_train, batch_size=task_batch_size, sampler=task_train_sampler
        )
        test_loader = DataLoader(dset_val, batch_size=shots * ways)
        
        meta_train_batch.append(Task(train_loader, train_loader, test_loader))
    return meta_train_batch


def create_og_data_loader(
    root,
    meta_split,
    k_way,
    n_shot,
    input_size,
    n_query,
    batch_size,
    num_workers,
    download=True,
    use_vinyals_split=False,
    seed=None,
):
    """Create a torchmeta BatchMetaDataLoader for Omniglot

    Args:
        root: Path to Omniglot data root folder (containing an 'omniglot'` subfolder with the
            preprocess json-Files or downloaded zip-files).
        meta_split: see torchmeta.datasets.Omniglot
        k_way: Number of classes per task
        n_shot: Number of samples per class
        input_size: Images are resized to this size.
        n_query: Number of test images per class
        batch_size: Meta batch size
        num_workers: Number of workers for data preprocessing
        download: Download (and dataset specific preprocessing that needs to be done on the
            downloaded files).
        use_vinyals_split: see torchmeta.datasets.Omniglot
        seed: Seed to be used in the meta-dataset

    Returns:
        A torchmeta :class:`BatchMetaDataLoader` object.
    """
    dataset = Omniglot(
        root,
        num_classes_per_task=k_way,
        transform=Compose([Resize(input_size), ToTensor()]),
        target_transform=Categorical(num_classes=k_way),
        class_augmentations=[Rotation([90, 180, 270])],
        meta_split=meta_split,
        download=download,
        use_vinyals_split=use_vinyals_split,
    )
    dataset = ClassSplitter(
        dataset, shuffle=True, num_train_per_class=n_shot, num_test_per_class=n_query
    )
    dataset.seed = seed
    dataloader = BatchMetaDataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return dataloader


def create_miniimagenet_data_loader(
    root,
    meta_split,
    k_way,
    n_shot,
    n_query,
    batch_size,
    num_workers,
    download=True,
    seed=None,
):
    """Create a torchmeta BatchMetaDataLoader for MiniImagenet

    Args:
        root: Path to mini imagenet root folder (containing an 'miniimagenet'` subfolder with the
            preprocess json-Files or downloaded tar.gz-file).
        meta_split: see torchmeta.datasets.MiniImagenet
        k_way: Number of classes per task
        n_shot: Number of samples per class
        n_query: Number of test images per class
        batch_size: Meta batch size
        num_workers: Number of workers for data preprocessing
        download: Download (and dataset specific preprocessing that needs to be done on the
            downloaded files).
        seed: Seed to be used in the meta-dataset

    Returns:
        A torchmeta :class:`BatchMetaDataLoader` object.
    """
    dataset = miniimagenet(
        root,
        n_shot,
        k_way,
        meta_split=meta_split,
        test_shots=n_query,
        download=download,
        seed=seed,
    )
    dataloader = BatchMetaDataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return dataloader


class TorchmetaTaskDistribution(TaskDistribution):
    """Class to create tasks for meta learning using torchmeta data loaders"""

    def __init__(self, config, n_channels, input_size, download=True):
        self.n_input_channels = n_channels
        self.input_size = input_size
        self.n_classes = config.k  # k-way classification
        self.data_path = config.data_path
        self.download = download

        self.k_way = config.k
        self.n_query = config.q  # number of query points (= test points)
        self.n_shot_test = config.n  # shots during meta-training
        self.n_shot_train = config.n_train  # shots during meta-testing
        self.meta_batch_size_train = config.meta_batch_size
        self.meta_batch_size_test = config.test_meta_batch_size
        self.num_workers = config.workers
        self.task_batch_size = config.batch_size  # batch size during task training
        self.task_batch_size_test = config.batch_size_test
        self.train_it = None
        self.train_sampler = None
        self.val_it = None
        self.val_sampler = None
        self.test_it = None
        self.test_sampler = None
        self.seed = config.seed

    def sample_meta_train(self):
        return sample_meta_batch(
            self.train_it,
            self.meta_batch_size_train,
            self.task_batch_size,
            self.n_shot_train,
            self.k_way,
            self.train_sampler,
        )

    def sample_meta_valid(self):
        return sample_meta_batch(
            self.val_it,
            self.meta_batch_size_test,
            self.task_batch_size_test,
            self.n_shot_test,
            self.k_way,
            self.val_sampler,
        )

    def sample_meta_test(self):
        return sample_meta_batch(
            self.test_it,
            self.meta_batch_size_test,
            self.task_batch_size_test,
            self.n_shot_test,
            self.k_way,
            self.test_sampler,
        )


class OmniglotFewShot(TorchmetaTaskDistribution):
    def __init__(self, config, download=True):
        super().__init__(config, 1, 28, download)
        self.use_vinyals_split = config.use_vinyals_split
        self.train_loader = create_og_data_loader(
            self.data_path,
            "train",
            self.k_way,
            self.n_shot_train,
            self.input_size,
            self.n_query,
            self.meta_batch_size_train,
            self.num_workers,
            self.download,
            self.use_vinyals_split,
            seed=self.seed,
        )
        self.train_it = iter(self.train_loader)

        if self.use_vinyals_split:
            self.val_loader = create_og_data_loader(
                self.data_path,
                "val",
                self.k_way,
                self.n_shot_test,
                self.input_size,
                self.n_query,
                self.meta_batch_size_test,
                self.num_workers,
                self.download,
                self.use_vinyals_split,
                seed=self.seed,
            )
            self.val_it = iter(self.test_loader)

        self.test_loader = create_og_data_loader(
            self.data_path,
            "test",
            self.k_way,
            self.n_shot_test,
            self.input_size,
            self.n_query,
            self.meta_batch_size_test,
            self.num_workers,
            self.download,
            self.use_vinyals_split,
            seed=self.seed,
        )
        self.test_it = iter(self.test_loader)

        self.train_sampler = None
        if self.task_batch_size != self.n_shot_train * self.k_way:
            self.train_sampler = RandomSampler(
                range(self.n_shot_train * self.k_way),
                replacement=True,
                num_samples=self.task_batch_size,
            )                                                                                                                                                                                                                                                                                                                                               
        self.test_sampler = None
        if self.task_batch_size_test != self.n_shot_test * self.k_way:
            self.val_sampler = RandomSampler(
                range(self.n_shot_test * self.k_way),
                replacement=True,
                num_samples=self.task_batch_size_test,
            )
            self.test_sampler = RandomSampler(
                range(self.n_shot_test * self.k_way),
                replacement=True,
                num_samples=self.task_batch_size_test,
            )


class MiniImageNetFewShot(TorchmetaTaskDistribution):
    """Class to create mini-imagenet-based tasks for meta learning"""

    def __init__(self, config, download=True):
        super().__init__(config, 3, 84, download)

        self.train_loader = create_miniimagenet_data_loader(
            self.data_path,
            "train",
            self.k_way,
            self.n_shot_train,
            self.n_query,
            self.meta_batch_size_train,
            self.num_workers,
            self.download,
            seed=self.seed,
        )
        self.train_it = iter(self.train_loader)

        self.val_loader = create_miniimagenet_data_loader(
            self.data_path,
            "val",
            self.k_way,
            self.n_shot_test,
            self.n_query,
            self.meta_batch_size_test,
            self.num_workers,
            self.download,
            seed=self.seed,
        )
        self.val_it = iter(self.val_loader)

        self.test_loader = create_miniimagenet_data_loader(
            self.data_path,
            "test",
            self.k_way,
            self.n_shot_test,
            self.n_query,
            self.meta_batch_size_test,
            self.num_workers,
            self.download,
            seed=self.seed,
        )
        self.test_it = iter(self.test_loader)

        self.train_sampler = None
        if self.task_batch_size != self.n_shot_train * self.k_way:
            self.train_sampler = RandomSampler(
                range(self.n_shot_train * self.k_way),
                replacement=True,
                num_samples=self.task_batch_size,
            )

        self.test_sampler = None
        if self.task_batch_size_test != self.n_shot_test * self.k_way:
            self.val_sampler = RandomSampler(
                range(self.n_shot_test * self.k_way),
                replacement=True,
                num_samples=self.task_batch_size_test,
            )
            self.test_sampler = RandomSampler(
                range(self.n_shot_test * self.k_way),
                replacement=True,
                num_samples=self.task_batch_size_test,
            )



    
        