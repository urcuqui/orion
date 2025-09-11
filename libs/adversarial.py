
from art.attacks.evasion import CarliniL2Method, HopSkipJump
from art.estimators.classification import PyTorchClassifier, BlackBoxClassifier
import torch
import timm
from torchvision import transforms
from PIL import Image

def generate_advimage(weights, noutputs, file):
    """ Generate adversarial image using Carlini & Wagner L2 Method """

    #device = (torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu"))
    device = torch.device("cpu") # Force to use CPU
    print(device)

    target_model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    target_model.head = torch.nn.Linear(target_model.head.in_features, int(noutputs))
    target_model = target_model.to(device)
    target_model.load_state_dict(
        torch.load(
            "weights/"+weights.filename, map_location=device
        )
    )
    target_model.eval()

    # DO NOT CHANGE
    labels = {}
    labels[0] = "fake"
    labels[1] = "real"
    classifier = PyTorchClassifier(
        model = target_model,
        loss = torch.nn.CrossEntropyLoss(),
        nb_classes = len(labels), # 1000
        input_shape = (3, 224, 224)
    )
    attack = CarliniL2Method(classifier)

    img = Image.open(file)
    preprocess = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]);

    unnormalize = transforms.Normalize(
    mean= [-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std= [1/s for s in [0.229, 0.224, 0.225]]
    )

    img_tensor = preprocess(img).unsqueeze(0)
    # run the attacks
    adversarial_example = attack.generate(img_tensor.numpy())

    adv_tensor = torch.from_numpy(adversarial_example).to(device)
    output = target_model(adv_tensor)

    print(f"Output index:\n---------------\n{output[0].argmax()}\n")
    print(f"Output label:\n---------------\n{labels[output[0].argmax().item()]}\n")

    def tensor_to_pil(img_tensor):    
        unnormed_tensor = unnormalize(img_tensor)
        return transforms.functional.to_pil_image(unnormed_tensor[0])

    masked_pil= tensor_to_pil(adv_tensor)
    masked_pil.save(fp="static/adversarial/output_art.png")
    return True

    