from __future__ import annotations

import copy
import math


CNN_RECIPE = {
    "version": "reference-cnn-epochs-v1",
    "reference": "Z:/THU/增量生成/test_wlcnn.py; experiments/hit_sqm_ours_rep02_20260407_164048/effective_config.yaml",
    "max_epochs": 1000,
    "batch_size": 64,
    "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.01},
    "early_stopping": {"metric": "source_exact_set_accuracy", "mode": "max", "patience": 10, "min_delta": 0.0001},
    "sampling": "shuffle_without_replacement; include_last_partial_batch",
    "validation_every_epochs": 1,
    "checkpoint_selection": "highest_source_exact_set_accuracy; earliest_tie",
}


def auxiliary(name, epochs, batch_size, learning_rate, milestones, gamma):
    return {"name": name, "max_epochs": epochs, "batch_size": batch_size,
            "optimizer": {"name": "SGD", "learning_rate": learning_rate, "momentum": 0.9, "weight_decay": 0.0001},
            "schedule": {"name": "multistep", "milestones": milestones, "gamma": gamma}}


GENERATOR_RECIPES = {
    "diffucd": {
        "reference": "Measurement 242 (2025) 115951, Section 4; Zotero WZKZMQKR",
        "source_url": "https://doi.org/10.1016/j.measurement.2024.115951",
        "provenance": "paper_reconstruction",
        "max_epochs": 1000, "batch_size": 64, "microbatch_size": 8,
        "optimizer": {"name": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "schedule": {"name": "step", "every_epochs": 20, "gamma": 0.1},
        "validation_every_epochs": 1, "checkpoint_selection": "last", "sampling_steps": 1200,
        "auxiliaries": [],
        "port_choices": ["The article reports 1000 epochs, Adam and a 0.1 decay every 20 epochs; it does not specify the initial learning rate or batch size. The 1D port uses lr=1e-4 and batch=64."],
    },
    "ccgan": {
        "reference": "improved_CcGAN 106ec275901de001b44beb238992debd89e74002, RC-49/RC-49_64x64/CcGAN-improved/scripts/run_train.sh; main.py; Train_net_for_label_embed.py",
        "source_url": "https://github.com/UBCDingXin/improved_CcGAN/tree/106ec275901de001b44beb238992debd89e74002/RC-49/RC-49_64x64/CcGAN-improved",
        "provenance": "author_RC49_schedule_1d_port",
        "max_updates": 30000, "batch_size": 256, "discriminator_batch_size": 256, "microbatch_size": 256,
        "optimizer": {"name": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0, "betas": [0.5, 0.999]},
        "schedule": {"name": "constant"}, "validation_every_updates": 2000,
        "checkpoint_selection": "last", "sampling_steps": 1,
        "auxiliaries": [auxiliary("condition_signal_to_label", 200, 256, 0.01, [60, 120, 160], 0.2),
                        auxiliary("condition_label_to_embedding", 500, 128, 0.01, [100, 200, 400], 0.2)],
        "port_choices": ["RC-49 is the selected author preset; the paper uses different budgets on other datasets. Auxiliary epochs traverse the source windows or unique source condition vectors, respectively. Keep the full GAN batch because its generator contains BatchNorm."],
    },
    "iccdm": {
        "reference": "iCCDM Appendix F, Table S.1, RC-49; CCDM a63aa63010255b0b3107411c2d53e7869b8af9c3, iCCDM/config/RC64/run_train.sh; label_embedding.py; trainer.py",
        "source_url": "https://arxiv.org/html/2602.02114v1#A6",
        "provenance": "author_RC49_schedule_1d_port",
        "max_updates": 100000, "batch_size": 128, "microbatch_size": 8,
        "optimizer": {"name": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "schedule": {"name": "constant"}, "gradient_clip": 1.0,
        "validation_every_updates": 5000, "checkpoint_selection": "last", "sampling_steps": 32,
        "auxiliaries": [auxiliary("condition_signal_to_label", 10, 256, 0.0001, [80, 140], 0.1),
                        auxiliary("condition_label_to_embedding", 500, 128, 0.01, [150, 250, 350], 0.1),
                        auxiliary("covariance_signal_to_label", 10, 256, 0.0001, [80, 140], 0.1),
                        auxiliary("covariance_label_to_embedding", 500, 128, 0.001, [150, 250, 350], 0.1)],
        "port_choices": ["Use FP32 accumulated gradients for the 1D port instead of the author's fp16 Accelerator execution; the effective batch remains 128. Auxiliary label embeddings use unique source condition vectors. No added generator early stopping or EMA (disabled by the author trainer default)."],
    },
    "time_weaver": {
        "reference": "ConTSG-Bench 6c74409a6bc24825d8f08d878229485c68d41d67, configs/generators/timeweaver_ettm1.yaml",
        "source_url": "https://github.com/seqml/ConTSG-Bench/blob/6c74409a6bc24825d8f08d878229485c68d41d67/configs/generators/timeweaver_ettm1.yaml",
        "provenance": "third_party_benchmark_schedule",
        "max_epochs": 700, "batch_size": 256, "microbatch_size": 8,
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]},
        "schedule": {"name": "cosine"}, "gradient_clip": 1.0,
        "validation_every_epochs": 1, "checkpoint_selection": "best_source_loss",
        "early_stopping": {"metric": "source_validation_loss", "mode": "min", "patience": 50, "min_delta": 0.0},
        "sampling_steps": 50, "auxiliaries": [],
        "port_choices": ["The checked original paper does not specify a complete training duration/batch/early-stop schedule. This is the existing third-party implementation and its ETTm1 preset, not an author-official training configuration."],
    },
    "flowts": {
        "reference": "FlowTS arXiv:2411.07506v2, Sections 4.1-4.2; FlowTS 1ec35fb1d3d89d91a1607a9f949a515347d54c8c, FMTS/run.sh and Models/interpretable_diffusion/FMTS.py",
        "source_url": "https://arxiv.org/html/2411.07506v2",
        "provenance": "published_iteration_range_with_author_learning_rate",
        "max_updates": 25000, "batch_size": 64, "microbatch_size": 8,
        "optimizer": {"name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.01, "betas": [0.9, 0.999]},
        "schedule": {"name": "constant"}, "validation_every_updates": 1000,
        "checkpoint_selection": "last", "sampling_steps": 32, "time_sampling": "logitnorm",
        "auxiliaries": [],
        "port_choices": ["25000 is the upper training iteration in the paper's efficiency comparison (2500-25000), not a universal author default. The author solar script sets lr=3e-4 and logit-normal time sampling. Batch=64, AdamW weight_decay=0.01 and final-checkpoint selection are explicit 1D task-port choices. The released solar YAML has max_epochs=1000 and accumulation=2 but its imported engine/solver.py is absent, so its counter cannot be treated as verified epochs or updates."],
    },
}


def generator_recipe(method, *, smoke_updates=None):
    recipe = copy.deepcopy(GENERATOR_RECIPES[method])
    recipe.update({"version": "external-method-recipes-v1", "native_length": 1024,
                   "sampling": "shuffled_source_epochs" if "max_epochs" in recipe else "method_native_random_batches"})
    if smoke_updates is not None:
        if smoke_updates < 2:
            raise ValueError("A short execution check requires at least two updates")
        recipe.pop("max_epochs", None)
        recipe.pop("early_stopping", None)
        recipe.update({"max_updates": smoke_updates, "batch_size": 8, "discriminator_batch_size": 8,
                       "microbatch_size": 8, "validation_every_updates": 1, "checkpoint_selection": "best_source_loss",
                       "validation_samples": 8,
                       "schedule": {"name": "constant"}, "sampling": "method_native_random_batches"})
        for phase in recipe["auxiliaries"]:
            phase.pop("max_epochs")
            phase.update({"max_updates": smoke_updates, "batch_size": 8, "schedule": {"name": "constant"}})
    return recipe


def phase_budget(recipe, sample_count):
    if sample_count < 1 or recipe["batch_size"] < 1:
        raise ValueError("Training requires positive sample and batch counts")
    per_epoch = math.ceil(sample_count / recipe["batch_size"])
    return {"training_windows": sample_count, "updates_per_epoch": per_epoch,
            "max_epochs": recipe.get("max_epochs"),
            "max_updates": recipe.get("max_updates", recipe.get("max_epochs", 0) * per_epoch)}


def classifier_window_count(real_count, additional_count, job):
    from .common import METHODS

    if job.get("arm") == "generated_only":
        return additional_count
    return real_count + (additional_count if job["method"] in (*METHODS, "augmentation", "superposition") else 0)


class EarlyStopping:
    def __init__(self, mode, patience, min_delta, **_):
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best = None
        self.bad_epochs = 0

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError("Early stopping requires a finite validation score")
        improvement = (self.best is None or (score > self.best + self.min_delta if self.mode == "max"
                                            else score < self.best - self.min_delta))
        if improvement:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self):
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.best, self.bad_epochs = state["best"], state["bad_epochs"]


def epoch_learning_rate(recipe, epoch):
    rate = recipe["optimizer"]["learning_rate"]
    schedule = recipe.get("schedule", {"name": "constant"})
    elapsed = epoch - 1
    if schedule["name"] == "step":
        rate *= schedule["gamma"] ** (elapsed // schedule["every_epochs"])
    elif schedule["name"] == "multistep":
        rate *= schedule["gamma"] ** sum(elapsed >= boundary for boundary in schedule["milestones"])
    elif schedule["name"] == "cosine":
        rate *= (1 + math.cos(math.pi * elapsed / recipe["max_epochs"])) / 2
    return rate
