import torch
from torch import Tensor, nn, optim


from model import FastflowModel, get_logp
from tqdm import tqdm


class FastflowTrainer:
    def __init__(self, hparams):

        self.hparams = hparams
        self.model: FastflowModel = FastflowModel(hparams)
        self.loss_val = 0
        self.automatic_optimization = False
        self.device = torch.device(self.hparams.device)
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(
                list(self.model.decoders[decoder_idx].parameters())
            )

        self.optimizer = optim.Adam(
            params=decoders_parameters,
            lr=self.hparams.model.lr,
        )

    def eval_model(self, test_loader):
        pass

    def train(self, train_loader, test_loader):

        sum_loss = 0
        for epoch in range(self.hparams.trainer.epochs):
            for activation_batch in tqdm(train_loader):
                avg_loss = self.training_step(activation_batch)
                sum_loss += avg_loss.detach().cpu().numpy()[0]
                print(sum_loss)
        print(sum_loss / len(train_loader.dataset))
        # self.eval_model(test_loader)

    def training_step(self, activation_batch):
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
        opt = self.optimizer

        avg_loss = torch.zeros([1], dtype=torch.float64).to(self.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation_batch[layer_idx]  # BxCxHxW
            encoder_activations = encoder_activations.to(self.device)
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
            decoder = self.model.decoders[layer_idx].to(self.device)

            opt.zero_grad()
            p_u, log_jac_det = decoder(encoder_activations)

            decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
            loss = decoder_log_prob.mean()
            loss.backward()
            opt.step()
            avg_loss += decoder_log_prob.sum()

        return avg_loss
