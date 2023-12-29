import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import cv2

from PIL import Image
from matplotlib import pyplot as plt
from specvqgan.modules.losses.vggishish.transforms import Crop
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
from specvqgan.data.transforms import \
    SpectrogramTorchAudio, \
    MelScaleTorchAudio, \
    LowerThresh, \
    Log10, Multiply, \
    Subtract, \
    Add, \
    Divide, \
    Clip

# ---------------------------
# Utils
# ---------------------------
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def load_specs_as_img(
    spec,
    device,
    spec_take_first=192,
):
    loader = transforms.Compose([
        transforms.Resize((80, spec_take_first)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    spec = spec[:, :spec_take_first]
    spec_img = Image.fromarray((spec * 255).astype(np.uint8)).convert('RGB')
    spec_img = loader(spec_img).unsqueeze(0)
    return spec_img.to(device, torch.float)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def attach_audio_to_video(
    video_path, 
    audio_path,
    dest,  
    FPS=15, 
    recon_only=False, 
    put_text=False,
    text_start_frame=0, 
    video_start_in_seconds=0,
    video_duration_in_seconds=2
):
    clip = VideoFileClip(video_path)
    clip = clip.subclip(
        video_start_in_seconds, 
        video_start_in_seconds + video_duration_in_seconds
    )
    clip = clip.set_fps(FPS)

    if put_text:
        frames = [f for f in clip.iter_frames()]
        H, W, _ = frames[0].shape
        for i in range(len(frames)):
            text = 'Original Audio' if i < text_start_frame else 'Generated Audio'
            if recon_only:
                text = 'Reconstructed Sound'
            img_w_text = cv2.putText(frames[i], text, (W//50, H//6),
                                     cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                     color=(255, 0, 0), thickness=3)
        clip = ImageSequenceClip(frames, fps=FPS)
    clip = clip.set_audio(AudioFileClip(audio_path))
    clip.write_videofile(dest, audio=True, fps=FPS, verbose=False, logger=None, write_logfile=False)
    return clip

# def attach_audio_to_video(video_path, audio_path, dest, start_step, FPS=15, recon_only=False, put_text=False, v_duration=2):
#     clip = VideoFileClip(video_path).set_fps(FPS)
#     if put_text:
#         frames = [f for f in clip.iter_frames()]
#         H, W, _ = frames[0].shape
#         for i in range(len(frames)):
#             text = 'Original Audio' if i < start_step else 'Generated Audio'
#             if recon_only:
#                 text = 'Reconstructed Sound'
#             img_w_text = cv2.putText(frames[i], text, (W//50, H//6),
#                                      cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
#                                      color=(255, 0, 0), thickness=3)
#         clip = ImageSequenceClip(frames, fps=FPS)
#     clip = clip.subclip(0, v_duration)
#     clip = clip.set_audio(AudioFileClip(audio_path))
#     clip.write_videofile(dest, audio=True, fps=FPS, verbose=False, logger=None)
#     return clip


def draw_spec(spec, dest, cmap='magma'):
    plt.imshow(spec, cmap=cmap, origin='lower')
    plt.axis('off')
    plt.savefig(dest, bbox_inches='tight', pad_inches=0.)
    plt.close()


# ---------------------------
# Transforms
# ---------------------------
class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


TRANSFORMS = transforms.Compose([
    SpectrogramTorchAudio(nfft=1024, hoplen=1024//4, spec_power=1),
    MelScaleTorchAudio(sr=22050, stft=513, fmin=125, fmax=7600, nmels=80),
    LowerThresh(1e-5),
    Log10(),
    Multiply(20),
    Subtract(20),
    Add(100),
    Divide(100),
    Clip(0, 1.0),
])


def get_melspectrogram_torch(audio, save_dir, length):
    y = torch.zeros(length)
    if audio.shape[0] < length:
        y[:len(audio)] = audio
    else:
        y = audio[:length]

    mel_spec = TRANSFORMS(y).numpy()

    return mel_spec


# ---------------------------
# Losses
# ---------------------------
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# ---------------------------
# Style Transfer
# ---------------------------
# desired depth layers to compute style/content losses
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    device,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
    device='cpu'
):
    """Run the style transfer."""
    print('Building the style transfer model..')
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
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

            run[0] += 1
            if run[0] % 150 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
