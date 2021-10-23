import os
import tqdm
from PIL import Image
from torch import Tensor
from torchvision import transforms

from api_robustModel import RobustResnet34


DEVICE = 'cpu'


def test_model(
    model_path='api_robustModel/models/resnet34.pt',
    dataset_path='E:/datasets/gc10_none_mask_divided/test',
    is_progress=True
):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    model = RobustResnet34(
        # model_weight_path='api_robustModel/models/resnet34.pt',
        model_weight_path='../server/checkpoints/resnet34-adv-2_0.03_10-best.pt',
        # model_weight_path='../server/checkpoints/resnet34-best.pt',
        device=DEVICE)

    correct_count = 0
    total_count = 0
    class_id = 0
    for class_dir in os.listdir(dataset_path):
        pbar = tqdm.tqdm(os.listdir(os.path.join(dataset_path, class_dir)))
        for image in pbar:
            pbar.set_description('%d %s: %s' % (class_id, class_dir, image))
            image_path = os.path.join(dataset_path, class_dir, image)
            image_tensor: Tensor = transform_test(Image.open(image_path))
            prob_list, index_list, class_name = model.top_k(image_tensor, k=1)
            if class_name[0] == class_dir:
                correct_count += 1
            total_count += 1
        class_id += 1
    print('Test Accu = %.5f   ' % (correct_count/total_count), end='')
    print('model:', model_path)

    # image_tensor:Tensor = transform_test(Image.open(img_path))
    # model = RobustResnet34(
    #     model_weight_path=model_path,
    #     device=DEVICE)


if __name__ == '__main__':
    test_model()
