import os
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


def select_device(device_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    return device


def build_models(dino_config, dino_checkpoint, sam_encoder_version, sam_checkpoint, device):
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=dino_config, model_checkpoint_path=dino_checkpoint)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    return grounding_dino_model, sam_predictor


def process_images(image_files, source_folder, result_folder, grounding_dino_model, sam_predictor, classes, thresholds,
                   class_colors):
    # Create a list to store the names of images where no detections were made
    no_detections = []
    for image_file in image_files:
        image_path = os.path.join(source_folder, image_file)
        image = cv2.imread(image_path)
        detections = process_dino(grounding_dino_model, image, classes, thresholds, result_folder, image_file)

        # Only process with SAM if there are detections
        if len(detections.xyxy) > 0:
            process_sam(sam_predictor, detections, image, image_file, classes, result_folder, thresholds, class_colors)
        else:
            # If there are no detections, add the image name to the list
            no_detections.append(image_file)

    # Write the names of images with no detections to a text file
    if no_detections:
        txt_path = os.path.join(result_folder['no_detections_txt_path'], "no_detections.txt")
        with open(txt_path, "w") as file:
            for image_name in no_detections:
                file.write(f"{image_name}\n")


def process_dino(grounding_dino_model, image, classes, thresholds, result_folder, image_file):
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=thresholds['box'],
        text_threshold=thresholds['text']
    )

    # Annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # Save the annotated grounding dino image
    output_path = os.path.join(result_folder['groundingdino'], f"groundingdino_annotated_{image_file}")
    cv2.imwrite(output_path, annotated_frame)

    return detections


def process_sam(sam_predictor, detections, image, image_file, classes, result_folder, thresholds, class_colors):
    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        thresholds['nms']
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Convert detections to masks
    detections.mask, Masks = segment(sam_predictor, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy,
                                     detections.class_id)

    save_results(image, image_file, classes, detections, Masks, result_folder, class_colors)


def segment(sam_predictor, image, xyxy, class_id):
    # Prompting SAM with detected boxes
    sam_predictor.set_image(image)
    result_masks, value_masks = [], []
    for idx, box in enumerate(xyxy):
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
        value_masks.append(masks[index] * (class_id[idx] + 1))
    return np.array(result_masks), np.array(value_masks)


def save_results(image, image_file, classes, detections, Masks, result_folder, class_colors):
    # Annotate image with detections and segmentations
    annotated_image = annotate_image(image, detections, classes)

    # Save the annotated grounded-sam image
    output_path = os.path.join(result_folder['grounded_sam'], f"grounded_sam_annotated_{image_file}")
    cv2.imwrite(output_path, annotated_image)

    combined_mask = combine_masks(image, detections, Masks, class_colors)

    # Save the combined mask
    combined_mask_path = os.path.join(result_folder['masks'], f"combined_masks_{image_file}")
    cv2.imwrite(combined_mask_path, combined_mask)


def annotate_image(image, detections, classes):
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image


def combine_masks(image, detections, Masks, class_colors):
    combined_mask = np.zeros_like(image)  # Create an empty image to combine the masks
    for idx, mask in enumerate(Masks):
        class_id = detections.class_id[idx]
        class_color = class_colors[class_id]
        # Apply color to the mask pixels
        mask_image = np.uint8(mask[:, :, np.newaxis] * class_color)
        # Combine the mask with the combined_mask
        combined_mask = cv2.addWeighted(combined_mask, 1, mask_image, 0.7, 0)
    return combined_mask



device_id = 0  # Set this to the ID of the GPU you want to use
device = select_device(device_id)

dino_config = "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
dino_checkpoint = "../../../ground_model/groundingdino_swint_ogc.pth"
sam_encoder_version = "vit_b"
sam_checkpoint = "../../../sam_model/sam_vit_b_01ec64.pth"

# Define the source image folder path and result folder path
source_folder = "../images/input_images/"
result_folder = {
    'groundingdino': "../images/output_results/Groundingdino",
    'grounded_sam': "../images/output_results/Grounded_sam",
    'masks': "../images/output_results/Masks",
    'no_detections_txt_path': "../images"
}
# no_detections_txt_path = "../images"

os.makedirs(result_folder['groundingdino'], exist_ok=True)
os.makedirs(result_folder['grounded_sam'], exist_ok=True)
os.makedirs(result_folder['masks'], exist_ok=True)

# Predict classes and hyper-param for GroundingDINO
classes = ["car"]  # Modify the classes as needed
thresholds = {
    'box': 0.25,
    'text': 0.25,
    'nms': 0.8
}
class_colors = [(0, 0, 255), (0, 255, 0)]  # Modify with colors for each class

# Get the list of image files in the source image folder
image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg") or f.endswith(".png")]

# Build models
grounding_dino_model, sam_predictor = build_models(dino_config, dino_checkpoint, sam_encoder_version, sam_checkpoint,
                                                   device)

# Process images
process_images(image_files, source_folder, result_folder, grounding_dino_model, sam_predictor, classes, thresholds,
               class_colors)
