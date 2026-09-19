import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
from datasets import load_dataset

from ..genlip.config import DataConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CaptionDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        tokenizer,
        split: str | None = None,
    ):
        split = split or config.train_split
        self.dataset = load_dataset(
            config.dataset_name,
            cache_dir=config.cache_dir,
            split=split,
        )
        self.tokenizer = tokenizer
        self.image_size = config.image_size
        self.caption_column = config.caption_column
        self.max_text_length = config.max_text_length

        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        pixel_values = self._process_image(row["image"])
        caption = row[self.caption_column]

        encoding = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors=None,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

    max_len = max(item["input_ids"].numel() for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = item["attention_mask"]
        labels[i, :length] = item["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataloader(
    config: DataConfig,
    tokenizer,
    split: str | None = None,
) -> DataLoader:
    split = split or config.train_split
    dataset = CaptionDataset(config, tokenizer, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    return dataloader
