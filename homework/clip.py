from pathlib import Path
from typing import Any
import math

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset


processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct"
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load(model_name: str = "clip_model"):
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()

    if device == "cuda":
        clip = clip.to(dtype=torch.bfloat16)

    return clip


def clip_data_collator(
    features: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    max_length = max(
        f["input_ids"].shape[0]
        for f in features
    )

    def pad_tensor(tensor, pad_value):
        return torch.cat(
            [
                tensor,
                torch.full(
                    (max_length - tensor.shape[0],),
                    pad_value,
                    dtype=tensor.dtype,
                ),
            ]
        )

    input_ids = torch.stack(
        [
            pad_tensor(
                f["input_ids"],
                pad_value=processor.tokenizer.eos_token_id,
            )
            for f in features
        ]
    )

    attention_mask = torch.stack(
        [
            pad_tensor(
                f["attention_mask"],
                pad_value=0,
            )
            for f in features
        ]
    )

    pixel_values = torch.stack(
        [f["pixel_values"] for f in features]
    )

    labels = torch.stack(
        [
            pad_tensor(
                f["labels"],
                pad_value=-100,
            )
            for f in features
        ]
    )

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(
        self,
        dataset: CaptionDataset,
        processor: AutoProcessor,
    ):
        self.dataset = dataset

        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(
                    192,
                    scale=(0.5, 1.0),
                ),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]

        image = Image.open(
            item["image_path"]
        ).convert("RGB")

        pixel_values = self.image_processor(image)

        text = (
            item["caption"]
            + self.processor.tokenizer.eos_token
        )

        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = (
            text_inputs["input_ids"]
            .squeeze(0)
            .long()
        )

        attention_mask = (
            text_inputs["attention_mask"]
            .squeeze(0)
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


class CLIP(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        proj_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        vision_dim = vision_encoder.config.hidden_size
        text_dim = text_encoder.config.hidden_size

        self.vision_projection = nn.Linear(
            vision_dim,
            proj_dim,
            bias=False,
        )

        self.text_projection = nn.Linear(
            text_dim,
            proj_dim,
            bias=False,
        )

        # CLIP learns log(1 / temperature).
        # Initial temperature = 0.07.
        self.temperature = nn.Parameter(
            torch.tensor(math.log(1 / temperature))
        )

    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        param = next(
            self.vision_encoder.parameters()
        )

        image = image.to(
            device=param.device,
            dtype=param.dtype,
        )

        outputs = self.vision_encoder(
            pixel_values=image
        )

        hidden_states = outputs.last_hidden_state

        # Use the first vision token as the image representation.
        features = hidden_states[:, 0]

        features = self.vision_projection(features)

        # Normalize to unit length.
        features = features / features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        return features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        param = next(
            self.text_encoder.parameters()
        )

        input_ids = input_ids.to(
            device=param.device
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device=param.device
            )

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            # Index of the final non-padding token.
            lengths = (
                attention_mask.sum(dim=1).long() - 1
            )

            batch_indices = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
            )

            features = hidden_states[
                batch_indices,
                lengths,
            ]
        else:
            features = hidden_states[:, -1]

        features = self.text_projection(features)

        # Normalize to unit length.
        features = features / features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        return features

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs,
    ):
        """Save additional projection/temperature parameters."""

        additional_state_dict = {}

        for name, param in self.named_parameters():
            if (
                "vision_encoder." in name
                or "text_encoder." in name
            ):
                continue

            additional_state_dict[name] = param.data

        torch.save(
            additional_state_dict,
            Path(save_directory)
            / "additional_weights.pt",
        )

    def load_pretrained(
        self,
        load_directory: str,
        **kwargs,
    ):
        """Load projection/temperature parameters."""

        additional_weights_path = (
            Path(load_directory)
            / "additional_weights.pt"
        )

        if additional_weights_path.exists():
            additional_state_dict = torch.load(
                additional_weights_path,
                map_location="cpu",
            )

            for name, param in self.named_parameters():
                if (
                    "vision_encoder." in name
                    or "text_encoder." in name
                ):
                    continue

                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if (
                "vision_encoder." in name
                or "text_encoder." in name
            ):
                continue

            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision
        and text backbones.
        """
        self.vision_encoder.gradient_checkpointing_enable(
            **kwargs
        )

        self.text_encoder.gradient_checkpointing_enable(
            **kwargs
        )

    def enable_input_require_grads(self):
        """
        Enable input gradients for the vision and text
        backbones.
        """

        def make_inputs_require_grads(
            module,
            input,
            output,
        ):
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(
            make_inputs_require_grads
        )

        self.text_encoder.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        image_features = self.encode_image(
            pixel_values
        )

        text_features = self.encode_text(
            input_ids,
            attention_mask,
        )

        # Cosine similarity because features are normalized.
        logits = image_features @ text_features.T

        # CLIP uses exp(log_temperature) as the scale.
        logits = logits * torch.exp(
            self.temperature
        )

        return (
            image_features,
            text_features,
            logits,
        )


def compute_clip_loss(
    outputs,
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Symmetric CLIP contrastive loss.
    """

    image_features, text_features, logits = outputs

    batch_size = image_features.shape[0]

    # Correct image/text pairs are on the diagonal.
    targets = torch.arange(
        batch_size,
        device=logits.device,
    )

    # Image -> text.
    image_to_text_loss = torch.nn.functional.cross_entropy(
        logits,
        targets,
    )

    # Text -> image.
    text_to_image_loss = torch.nn.functional.cross_entropy(
        logits.T,
        targets,
    )

    # Symmetric CLIP objective.
    loss = (
        image_to_text_loss
        + text_to_image_loss
    ) / 2

    return loss


def get_target_modules_for_lora(
    model: nn.Module,
) -> list[str]:

    target_modules = []

    for name, module in model.named_modules():

        if (
            isinstance(module, nn.Linear)
            and (
                "vision_encoder" in name
                or "text_encoder" in name
            )
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip_model",
    num_train_epochs: float = 0.05,
    per_device_train_batch_size: int = 1024,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 16,
):

    vlm = BaseVLM()

    output_dir = (
        Path(__file__).parent
        / output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensorboard_dir = output_dir / "tensorboard"

    tensorboard_dir.mkdir(
        exist_ok=True
    )

    writer = SummaryWriter(
        log_dir=tensorboard_dir
    )

    # Initialize encoders.
    vision_encoder = (
        vlm.model.model.vision_model
    )

    text_encoder = (
        vlm.model.model.text_model
    )

    model = CLIP(
        vision_encoder,
        text_encoder,
    ).to(device)

    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(
            model
        ),
        bias="none",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.print_trainable_parameters()

    model.to(device)

    if device == "cuda":
        model = model.to(
            dtype=torch.bfloat16
        )

    model.train()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load caption dataset.
    train_dataset = CaptionDataset(
        "train",
        data_dir,
    )

    train_dataset = CaptionDatasetForTraining(
        train_dataset,
        processor,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    # Save model.
    trainer.save_model(output_dir)
    model.model.save_pretrained(
        output_dir
    )

    writer.close()

    return model, processor


def demo_train():
    train(
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(
    ckpt_path: str,
    val_dataset: str = "valid_grader",
):

    import tqdm

    testset = MultiChoiceQADataset(
        val_dataset
    )

    clip = load(ckpt_path)

    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):

        image = Image.open(
            pair["image_path"]
        ).convert("RGB")

        pixel_values = (
            image_processor(image)
            .unsqueeze(0)
            .to(device)
            .bfloat16()
        )

        text_inputs = processor(
            text=[
                s + processor.tokenizer.eos_token
                for s in pair["candidates"]
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = (
            text_inputs["input_ids"]
            .long()
            .to(device)
        )

        attention_mask = (
            text_inputs["attention_mask"]
            .to(device)
        )

        vision_feature, text_feature, _ = clip(
            pixel_values,
            input_ids,
            attention_mask,
        )

        prediction = torch.matmul(
            vision_feature,
            text_feature.T,
        ).argmax(dim=-1)

        if prediction == pair["correct_index"]:
            correct_count += 1

        total_count += 1

    print(
        f"Accuracy: "
        f"{correct_count / total_count}"
    )


def main():
    from fire import Fire

    Fire(
        {
            "train": train,
            "test": test,
        }
    )


if __name__ == "__main__":
    main()