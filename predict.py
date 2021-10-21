from PIL import Image
from torchvision import transforms


from api_robustModel import RobustResnet34



img_path = "./airplane_0003.jpg"

def main():
    transform_test = transforms.Compose([
        transforms.Resize(224, 224),
        transforms.ToTensor()
    ]
    )
    image_pil = Image.open(img_path)
    image_tensor = transform_test(image_pil)

if __name__ == '__main__':
    main()