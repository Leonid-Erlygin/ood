
import torch
from torch import Tensor, nn, optim


from model import FastflowModel, get_logp

class FastflowTrainer:
    def __init__(self, hparams):

        self.hparams = hparams
        self.model: FastflowModel = FastflowModel(hparams)
        self.loss_val = 0
        self.automatic_optimization = False

    # def configure_callbacks(self):
    #     """Configure model-specific callbacks."""
    #     early_stopping = EarlyStopping(
    #         monitor=self.hparams.model.early_stopping.metric,
    #         patience=self.hparams.model.early_stopping.patience,
    #         mode=self.hparams.model.early_stopping.mode,
    #     )
    #     return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(
                list(self.model.decoders[decoder_idx].parameters())
            )

        self.optimizer = optim.Adam(
            params=decoders_parameters,
            lr=self.hparams.model.lr,
        )

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of CFLOW.

        For each batch, decoder layers are trained with a dynamic fiber batch size.
        Training step is performed manually as multiple training steps are involved
            per batch of input images

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Loss value for the batch

        """
        opt = self.optimizers()
        self.model.encoder.eval()

        images = batch["image"]
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(images.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            (
                batch_size,
                dim_feature_vector,
                im_height,
                im_width,
            ) = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = (
                batch_size * image_size
            )  # number of rows in the conditional vector
            # e_r = einops.rearrange(encoder_activations, "b c h w -> b h w c")

            height.append(im_height)
            width.append(im_width)
            decoder = self.model.decoders[layer_idx].to(images.device)

            opt.zero_grad()
            p_u, log_jac_det = decoder(encoder_activations)

            #
            decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
            self.manual_backward(decoder_log_prob.mean())
            opt.step()
            avg_loss += decoder_log_prob.sum()

        return {"loss": avg_loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of CFLOW.

            Similar to the training step, encoder features
            are extracted from the CNN for each batch, and anomaly
            map is computed.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        batch["anomaly_maps"] = self.model(batch["image"])

        return batch
