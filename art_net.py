import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import copy

# Content loss between two images is simply the Mean squared error
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# The style loss of an image depends on a more complex relationship
# as the GRAM matrix 
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = StyleLoss.gram_matrix(target_feature).detach()

    def forward(self, input):
        mat = StyleLoss.gram_matrix(input)
        self.loss = F.mse_loss(mat, self.target)
        return input

    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        mat = torch.mm(features, features.t())
        return mat.div(a * b * c * d)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class ArtNet:
    def __init__(self, device):
        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()
        # the normalization mean and std are taken directly from the original paper
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.device = device


    @staticmethod
    def get_optimizer(image, lr):
        optimizer = optim.LBFGS([image.requires_grad_()], lr=lr)
        return optimizer

    def get_losses(self, style_image, content_image):
        self.cnn = copy.deepcopy(self.cnn)
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(self.device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def train(self, content_weight, style_weight, n_steps, content_image, style_image, input_image, lr, experiment):
        model, style_losses, content_losses = self.get_losses( style_image, content_image)
        optimizer = ArtNet.get_optimizer(input_image, lr)
        step = 0
        with experiment.train():
            while step <= n_steps:

                # LBFGS optimizer requires a closure function since it
                # needs to evaluate the function multiple times
                def closure():
                    input_image.data.clamp_(0, 1)
                    optimizer.zero_grad()
                    model(input_image)
                    style_score = 0
                    content_score = 0

                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss

                    style_score *= style_weight
                    content_score *= content_weight

                    loss = style_score + content_score
                    loss.backward()

                    step += 1
                    if step % 50 == 0:
                        print("run {}:".format(step))
                        experiment.log_metric("style_loss", style_score.item())
                        experiment.log_metric("content_loss", content_score.item())
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                            style_score.item(), content_score.item()))
                        print()

                    return style_score + content_score

                optimizer.step(closure)

        input_image.data.clamp_(0, 1)

        return input_image
