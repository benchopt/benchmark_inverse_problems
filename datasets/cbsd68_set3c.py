from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from torchvision import transforms
    from datasets import load_dataset
    from benchmark_utils.hugging_face_torch_dataset import (
        HuggingFaceTorchDataset
    )
    from benchmark_utils.helper import get_task_physic, get_device


class Dataset(BaseDataset):

    name = "CBSD68_Set3c"

    parameters = {
        'task': [
            'denoising',
            'gaussian-debluring',
            'motion-debluring',
            'SRx4',
            'inpainting',
            'demosaicing'
        ],
        'img_size': [256],
    }

    requirements = ["datasets"]

    def get_data(self):
        device = get_device()

        n_channels = 3
        image_size = (n_channels, self.img_size, self.img_size)

        physics = get_task_physic(self.task, image_size, device)

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        dataset_CBSD68 = load_dataset("deepinv/CBSD68")
        train_dataset = HuggingFaceTorchDataset(
            dataset_CBSD68["train"],
            key="png",
            physics=physics,
            device=device,
            transform=transform
        )

        dataset_Set3c = load_dataset("deepinv/set3c")
        test_dataset = HuggingFaceTorchDataset(
            dataset_Set3c["train"],
            key="image",
            physics=physics,
            device=device,
            transform=transform
        )

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="Set3c",
            task_name=self.task,
            image_size=image_size
        )
