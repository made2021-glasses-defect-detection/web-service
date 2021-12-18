from argparse import ArgumentParser
from redis_storage import RedisStorage, current_time_ms
from serialize import json_to_torch, torch_to_json

INPUT_VALIDATION_HANDLER = "validator"
SEGMENTATION_HANDLER = "segmentation"
CLASSIFIER_HANDLER = "classification"

HANDLERS = [
    INPUT_VALIDATION_HANDLER,
    SEGMENTATION_HANDLER,
    CLASSIFIER_HANDLER,
]

INPUT_MODEL_PATH = "validation/efficientnet-b0.pch"
SEGMENTATION_MODEL_PATH = "segmentation/unet_resnet34_whole.pth"
CLASSIFICATION_MODEL_PATH = "classification/clf_whole.pth"
UPLOAD_DIR = "./uploads"
CLASSIFICATION_THRESHOLD = 0.5

# Storage format:
# {
#   "uid": "uuidv4",
#   "path": "file_path",
#   "input_validation": True|False, True - валидация успешно пройдена
#   "segmented_image": "file_path",
#   "segmentation_result": "JSON with Torch Tensor",
# }

storage = RedisStorage()

def handler_callback(arguments):
    if arguments.handler == INPUT_VALIDATION_HANDLER:
        from validation import classification_efficientnet

        input_validator = classification_efficientnet.Evaluator(INPUT_MODEL_PATH)

        def input_handler(payload):
            uid = payload["uid"]
            path = payload["path"]

            result = input_validator.predict(path)
            payload["input_validation"] = result
            payload["validated_at"] = current_time_ms()

            storage.update(uid, payload)
            if result == True:
                storage.publish(RedisStorage.SEGMENTATION, payload)

            print(f"Update input_validation for {path} = {result}")

        storage.subscribe(RedisStorage.INPUT_VALIDATION, input_handler)
    elif arguments.handler == SEGMENTATION_HANDLER:
        from segmentation import segmentor

        segmentation = segmentor.Evaluator(SEGMENTATION_MODEL_PATH, UPLOAD_DIR)

        def segmentation_handler(payload):
            uid = payload["uid"]
            path = payload["path"]

            predicted, image_pred_path = segmentation.predict(path)
            payload["segmentation_result"] = torch_to_json(predicted)
            payload["segmented_image"] = image_pred_path
            payload["segmented_at"] = current_time_ms()
            storage.update(uid, payload)
            storage.publish(RedisStorage.CLASSIFICATION, payload)
            print(f"Image {uid} segmeted {image_pred_path}")

        storage.subscribe(RedisStorage.SEGMENTATION, segmentation_handler)
    elif arguments.handler == CLASSIFIER_HANDLER:
        from classification import classifier

        classifier_model = classifier.Evaluator(CLASSIFICATION_MODEL_PATH)

        def classification_handler(payload):
            uid = payload["uid"]
            segmentation_result = json_to_torch(payload["segmentation_result"])

            proba = classifier_model.predict(segmentation_result)
            payload["proba"] = round(proba * 100, 2)
            payload["result"] = proba > CLASSIFICATION_THRESHOLD
            payload["classified_at"] = current_time_ms()
            storage.update(uid, payload)
            print(f"Image {uid} classified, probability of defect={proba}")

        storage.subscribe(RedisStorage.CLASSIFICATION, classification_handler)
    else:
        raise f"Unknown worker type {arguments.handler}"

def setup_parser(parser):
    """Setup CLI arg parsers"""
    parser.set_defaults(callback=handler_callback)

    parser.add_argument(
        "--handler",
        dest="handler",
        choices=HANDLERS,
        required=True,
    )

if __name__ == "__main__":
    """Entry point"""
    parser = ArgumentParser(
        description="Worker for process queues",
    )

    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)
