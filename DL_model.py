import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

cifar100_classes = {
    0: "яблоко",
    1: "аквариумная рыба",
    2: "ребенок",
    3: "медведь",
    4: "бобр",
    5: "кровать",
    6: "пчела",
    7: "жук",
    8: "велосипед",
    9: "бутылка",
    10: "миска",
    11: "мальчик",
    12: "мост",
    13: "автобус",
    14: "бабочка",
    15: "верблюд",
    16: "банка",
    17: "замок",
    18: "гусеница",
    19: "крупный рогатый скот",
    20: "стул",
    21: "шимпанзе",
    22: "часы",
    23: "облако",
    24: "таракан",
    25: "диван",
    26: "краб",
    27: "крокодил",
    28: "чашка",
    29: "динозавр",
    30: "дельфин",
    31: "слон",
    32: "камбала",
    33: "лес",
    34: "лиса",
    35: "девочка",
    36: "хомяк",
    37: "дом",
    38: "кенгуру",
    39: "клавиатура",
    40: "лампа",
    41: "газонокосилка",
    42: "леопард",
    43: "лев",
    44: "ящерица",
    45: "лобстер",
    46: "мужчина",
    47: "клен",
    48: "мотоцикл",
    49: "гора",
    50: "мышь",
    51: "гриб",
    52: "дуб",
    53: "апельсин",
    54: "орхидея",
    55: "выдра",
    56: "пальма",
    57: "груша",
    58: "пикап",
    59: "сосна",
    60: "равнина",
    61: "тарелка",
    62: "мак",
    63: "дикобраз",
    64: "опоссум",
    65: "кролик",
    66: "енот",
    67: "скат",
    68: "дорога",
    69: "ракета",
    70: "роза",
    71: "море",
    72: "тюлень",
    73: "акула",
    74: "землеройка",
    75: "скунс",
    76: "небоскреб",
    77: "улитка",
    78: "змея",
    79: "паук",
    80: "белка",
    81: "трамвай",
    82: "подсолнух",
    83: "сладкий перец",
    84: "стол",
    85: "танк",
    86: "телефон",
    87: "телевизор",
    88: "тигр",
    89: "трактор",
    90: "поезд",
    91: "форель",
    92: "тюльпан",
    93: "черепаха",
    94: "гардероб",
    95: "кит",
    96: "ива",
    97: "волк",
    98: "женщина",
    99: "червь"
}

def init_model(model_path='files/model/model_acc-76.92.pth'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(512, 100)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


def predict_image(image, model, device, top_k=5):
    global cifar100_classes

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, top_k)

        return [
            {'class' : cifar100_classes[idx.item()], 'proba_percent' : round(prob.item() * 100, 2)}
            for idx, prob in zip(top_indices, top_probs)
        ]