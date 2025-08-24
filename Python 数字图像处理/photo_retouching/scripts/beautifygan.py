import torch
from PIL import Image
from torchvision import transforms
from models.beautifygan.model import BeautifyGAN

def beautify_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_image = transform(image).unsqueeze(0)

    model = BeautifyGAN()
    model.load_state_dict(torch.load('models/beautifygan/beautifygan.pth'))
    model.eval()

    with torch.no_grad():
        output_image = model(input_image)

    output_image = output_image.squeeze().cpu().numpy()
    output_image = ((output_image + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(output_image)
