from PIL import Image
from torch import Tensor
from torchvision import transforms


from api_robustModel import RobustResnet34


img_path = "./airplane_0003.jpg"
img_path = "C:/Users/29650/datasets/gc10_none_mask_divided/train/airplane/airplane_0020.jpg"
DEVICE = 'cpu'



def main():
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    image_tensor:Tensor = transform_test(Image.open(img_path))
    model = RobustResnet34(
        # model_weight_path='api_robustModel/models/resnet34.pt',
        model_weight_path='../server/checkpoints/resnet34-adv-2_0.03_10-best.pt',
        # model_weight_path='../server/checkpoints/resnet34-best.pt',
        device=DEVICE)
    prob_list,index_list,class_name=model.top_k(image_tensor,k=5)
    print(prob_list)
    print(index_list)
    print(class_name)

def test_model(model_path='api_robustModel/models/resnet34.pt',dataset_path='E:/datasets/'):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    image_tensor:Tensor = transform_test(Image.open(img_path))
    model = RobustResnet34(
        model_weight_path=model_path,
        device=DEVICE)


if __name__ == '__main__':
    main()
