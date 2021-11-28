from argparse import ArgumentParser
from input_classifier import classification_efficientnet
from redis_storage import RedisStorage

INPUT_VALIDATION_HANDLER = "validator"
DEFECT_DETECTION_HANDLER = "detector"

HANDLERS = [
	INPUT_VALIDATION_HANDLER,
	DEFECT_DETECTION_HANDLER,
]

INPUT_MODEL_PATH = "input_classifier/efficientnet-b0.pch"

# Storage format:
# {
# 	"uid": "uuidv4",
# 	"path": "file_path",
#   "input_validation": True|False, True - валидация успешно пройдена
# }

storage = RedisStorage()

def handler_callback(arguments):
	if arguments.handler == INPUT_VALIDATION_HANDLER:
		input_validator = classification_efficientnet.Evaluator(INPUT_MODEL_PATH)
		
		def input_handler(payload):
			uid = payload["uid"]
			path = payload["path"]

			result = input_validator.predict(path)
			payload["input_validation"] = result
			storage.update(uid, payload)
			print(f"Update input_validation for {path} = {result}")

		storage.subscribe(RedisStorage.INPUT_VALIDATION, input_handler)
	elif arguments.handler == DEFECT_DETECTION_HANDLER:
		print("TBD...")

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