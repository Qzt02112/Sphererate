"""Leak-free declared-start adapter for the official ScanDMM implementation.

The official model initializes its latent state from ``scanpaths[:, 0]`` and
also supervises ``obs_x_1`` with that same value.  For our task, the declared
start is known before the first one-second gaze target, so those two tensors
must be independent.  These factories preserve all official module/parameter
names while changing only the source of the latent-state initialization.
"""

from __future__ import annotations

import math
from typing import Any

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
import torch


def declared_start_xyz(
    pitch_deg: float,
    yaw_deg: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert ScanDMM's (latitude, longitude) convention to unit xyz."""

    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    return torch.tensor(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
        dtype=dtype,
        device=device,
    )


def _expand_initial_gaze(initial_gaze: torch.Tensor | None, scanpaths: torch.Tensor) -> torch.Tensor:
    if initial_gaze is None:
        raise ValueError(
            "DeclaredStartDMM requires initial_gaze; using scanpaths[:, 0] would leak the first target"
        )
    gaze = torch.as_tensor(initial_gaze, dtype=scanpaths.dtype, device=scanpaths.device)
    if gaze.ndim == 1:
        gaze = gaze.unsqueeze(0)
    if gaze.shape == (1, 3):
        gaze = gaze.expand(scanpaths.size(0), -1)
    if gaze.shape != (scanpaths.size(0), 3):
        raise ValueError(
            f"initial_gaze must have shape [3], [1, 3], or [B, 3]; got {tuple(gaze.shape)}"
        )
    return gaze


def build_declared_start_dmm(official_dmm_class: type) -> type:
    """Return a state-dict-compatible DMM using an independent start tensor."""

    class DeclaredStartDMM(official_dmm_class):
        def model(
            self,
            scanpaths,
            scanpaths_reversed,
            mask,
            scanpath_lengths,
            images=None,
            annealing_factor=1.0,
            predict=False,
            initial_gaze=None,
        ):
            t_max = scanpaths.size(1)
            pyro.module("dmm", self)

            start = _expand_initial_gaze(initial_gaze, scanpaths)
            z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
            z_prev = self.tanh(
                self.twoZ_to_z_dim(
                    torch.cat((z_prev, self.tanh(self.input_to_z_dim(start))), dim=1)
                )
            )
            img_features = self.cnn(images)

            with pyro.plate("z_minibatch", len(scanpaths)):
                for t in pyro.markov(range(1, t_max + 1)):
                    z_mu, z_sigma = self.trans(z_prev, img_features)
                    with poutine.scale(scale=annealing_factor):
                        z_t = pyro.sample(
                            f"z_{t}",
                            dist.Normal(z_mu, z_sigma)
                            .mask(mask[:, t - 1 : t])
                            .to_event(1),
                        )
                    x_mu, x_sigma = self.emitter(z_t)
                    observation = None if predict else scanpaths[:, t - 1, :]
                    pyro.sample(
                        f"obs_x_{t}",
                        dist.Normal(x_mu, x_sigma)
                        .mask(mask[:, t - 1 : t])
                        .to_event(1),
                        obs=observation,
                    )
                    z_prev = z_t

        def guide(
            self,
            scanpaths,
            scanpaths_reversed,
            mask,
            scanpath_lengths,
            images=None,
            annealing_factor=1.0,
            initial_gaze=None,
        ):
            t_max = scanpaths.size(1)
            pyro.module("dmm", self)

            h_0_contig = self.h_0.expand(
                1, scanpaths.size(0), self.rnn.hidden_size
            ).contiguous()
            rnn_output, _ = self.rnn(scanpaths_reversed, h_0_contig)
            rnn_output = poly.pad_and_reverse(rnn_output, scanpath_lengths)

            start = _expand_initial_gaze(initial_gaze, scanpaths)
            z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
            z_prev = self.tanh(
                self.twoZ_to_z_dim(
                    torch.cat((z_prev, self.tanh(self.input_to_z_dim(start))), dim=1)
                )
            )

            with pyro.plate("z_minibatch", len(scanpaths)):
                for t in pyro.markov(range(1, t_max + 1)):
                    z_mu, z_sigma = self.combiner(z_prev, rnn_output[:, t - 1, :])
                    z_distribution = dist.Normal(z_mu, z_sigma)
                    assert z_distribution.event_shape == ()
                    assert z_distribution.batch_shape[-2:] == (
                        len(scanpaths),
                        self.z_0.size(0),
                    )
                    with poutine.scale(scale=annealing_factor):
                        z_t = pyro.sample(
                            f"z_{t}",
                            z_distribution.mask(mask[:, t - 1 : t]).to_event(1),
                        )
                    z_prev = z_t

    DeclaredStartDMM.__name__ = "DeclaredStartDMM"
    DeclaredStartDMM.__qualname__ = "DeclaredStartDMM"
    return DeclaredStartDMM


def build_declared_start_trainer(
    official_train_class: type,
    *,
    pitch_deg: float,
    yaw_deg: float,
) -> type:
    """Return an official-trainer adapter that passes a separate start tensor."""

    class DeclaredStartTrain(official_train_class):
        def prepare_train_data(self) -> dict[str, Any]:
            data = super().prepare_train_data()
            start = declared_start_xyz(pitch_deg, yaw_deg, dtype=data["sequences"].dtype)
            self.initial_gazes = start.unsqueeze(0).repeat(len(data["sequence_lengths"]), 1)
            return data

        def process_minibatch(self, epoch, which_mini_batch, shuffled_indices):
            if self.args.annealing_epochs > 0 and epoch < self.args.annealing_epochs:
                minimum = self.args.minimum_annealing_factor
                annealing_factor = minimum + (1.0 - minimum) * (
                    float(which_mini_batch + epoch * self.N_mini_batches + 1)
                    / float(self.args.annealing_epochs * self.N_mini_batches)
                )
            else:
                annealing_factor = 1.0

            start_index = which_mini_batch * self.args.bs
            end_index = min((which_mini_batch + 1) * self.args.bs, self.N_sequences)
            mini_batch_indices = shuffled_indices[start_index:end_index]
            lengths = self.seq_lengths[mini_batch_indices]
            _, order = torch.sort(lengths, descending=True)
            sorted_indices = mini_batch_indices[order]
            sorted_starts = self.initial_gazes[sorted_indices]

            batch = self.get_mini_batch(
                mini_batch_indices,
                self.sequences,
                self.seq_lengths,
                self.images,
                cuda=self.dmm.use_cuda,
            )
            mini_batch, reversed_batch, batch_mask, sorted_lengths, batch_images = batch
            if self.dmm.use_cuda:
                sorted_starts = sorted_starts.cuda()

            return self.svi.step(
                scanpaths=mini_batch,
                scanpaths_reversed=reversed_batch,
                mask=batch_mask,
                scanpath_lengths=sorted_lengths,
                images=batch_images,
                annealing_factor=annealing_factor,
                initial_gaze=sorted_starts,
            )

    DeclaredStartTrain.__name__ = "DeclaredStartTrain"
    DeclaredStartTrain.__qualname__ = "DeclaredStartTrain"
    return DeclaredStartTrain
