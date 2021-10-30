### Author    :  J. Yang
### Date      :  10/11/2021
### Function  :  Image reconstruction

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from .modules import VectorQuantizedVAE, to_scalar
from progress.bar import Bar
from PIL import Image

def save_image(tensor, dir):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(dir)

class Recon:
    def __init__(self, model_dir, device, data_form, image_num, batch_size):
        """
        :param model_dir:       Dir of recon_model.pt, e.g. './api_recon/recon_model.pt'
        :param device:      Cuda device, e.g. 'cuda:1'
        :param data_form:   1,2,3 or 4
                                1 means Reading & Saving .pt data, e.g. "/workspace/data.pt"
                                2 means Reading & Saving fold data, e.g. "/workspace/data/"
                                3 means Reading & Saving 1 picture, e.g. "/workspace/data.jpg" or png/bmp
                                4 means Reading & Saving 1 image tensor

        :param image_num:   The number of images, e.g. 1, 10 or 100
        :param image_in:    Input image, images or tensor, corresponds to the data_form.
                                e.g. "/workspace/data.pt"
                                     "/workspace/data/"
                                     "/workspace/data.jpg" or png/bmp
        :param image_out:   Output image, images or tensor, corresponds to the data_form.
                                e.g. "/workspace/recon.pt"
                                     "/workspace/recon/"
                                     "/workspace/recon.jpg" or png/bmp
        """

        self.model_dir = model_dir    
        self.data_form = data_form
        self.image_num = image_num
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        ### Load Model and Parameters
        num_channels = 3
        hidden_size  = 256
        k            = 512
        model = VectorQuantizedVAE(num_channels, hidden_size, k).to(self.device)
        model.load_state_dict(torch.load(self.model_dir, map_location=torch.device('cpu')))
        model.eval()
        self.model = model

    def recon(self, image_in, image_out=''):
        self.image_in = image_in
        self.image_out = image_out

        ### Loading image data
        if self.data_form == 1:
            """
            Mode 1:  Reading & Saving .pt data,
                     e.g. self.image_in  = "/workspace/yjt/gc10_dsets/data.pt"
                          self.image_out = "/workspace/yjt/gc10_dsets/recon.pt"
            """
            test_kwargs = {'batch_size': self.batch_size}
            cuda_kwargs = {'num_workers': 0,
                           'pin_memory': False,
                           'shuffle': False}
            test_kwargs.update(cuda_kwargs)
            trigger_set = torch.load(self.image_in)
            trigger_sample, trigger_label = trigger_set['data'], trigger_set['target']
            dataset2 = torch.utils.data.TensorDataset(trigger_sample, trigger_label)
            test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        elif self.data_form == 2:
            """
            Mode 2:  Reading & Saving fold data
                     e.g. self.image_in = "/workspace/yjt/gc10_dsets/"
            """
            test_dataset = datasets.ImageFolder(self.image_in, transform=transforms.ToTensor())
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        elif self.data_form == 3:
            """
            Mode 3:  Reading & Saving 1 picture
                     e.g. self.image_in = "/workspace/yjt/gc10_dsets/data.jpg"
            """
            img_path = self.image_in
            transform_test = transforms.Compose([
                transforms.Resize(224, 224),
                transforms.ToTensor()
            ]
            )
            img = Image.open(img_path)
            img_ = transform_test(img).unsqueeze(0)

        elif self.data_form == 4:
            """
            Mode 4:  Reading & Saving 1 image tensor
                     e.g. self.image_in = image_tensor
            """
            img_ = self.image_in

        else:
            print('Error: Data Format Error! Please check your input!')

        ### Reconstruct image data
        if self.data_form == 1 or self.data_form == 2:
            bar = Bar('Processing', max=self.image_num / self.batch_size, fill='@', suffix='%(percent)d%% - [ %(elapsed_td)s / %(eta_td)s ]')
            picnum = 0
            with torch.no_grad():
                for images, y in test_loader:
                    images = images.to(self.device)
                    reconstruction, _, _ = self.model(images)
                    bar.next()
                    if picnum == 0:
                        new_x = reconstruction.cpu()
                        new_y = y.cpu()
                    else:
                        new_x = torch.cat([new_x, reconstruction.cpu()], dim=0)
                        new_y = torch.cat([new_y, y.cpu()], dim=0)
                    picnum = picnum + 1
                torch.save({'data': new_x, 'target': new_y}, self.image_out)
                bar.finish()
            print(picnum)
            return 1, 1

        elif self.data_form == 3 or self.data_form == 4:
            img_ = img_.to(self.device)
            reconstruction, _, _ = self.model(img_)
            new_x = reconstruction.cpu()
            ### Tensor to PIL image
            p_image = reconstruction.cpu().clone()
            # p_image = p_image.squeeze(0)
            # pil_image = transforms.ToPILImage()(p_image)
            pil_image=None

            ### Save as .pt
            # torch.save({'data': new_x}, self.image_out)
            ### Save as .jpg bmp png
            # save_image(new_x, self.image_out)
            return reconstruction.cpu(), pil_image

        else:
            print('Error: Data Format Error! Please check your input!')
            return 0, 0

        ### Ending
        print("Recon Complete 100%.")


def main():
    test = Recon(model='./api_recon/recon_model.pt', device='cuda:1', data_form=1, image_num=2000, batch_size=1,
                 image_in="/workspace/yjt/gc10_dsets/clean.pt", image_out="/workspace/yjt/gc10_dsets/recon_clean.pt")
    test.recon()

if __name__ == '__main__':
    main()
