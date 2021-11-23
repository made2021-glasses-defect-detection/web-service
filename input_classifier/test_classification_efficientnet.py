from classification_efficientnet import Evaluator

GLASS_IMAGE_PATH = "images/glasses/1.jpg"
OTHER_IMAGE_PATH = "images/other/0.jpg"

def test_with_glasses_image():
	e = Evaluator()
	assert True == e.predict(GLASS_IMAGE_PATH)

def test_with_other_image():
	e = Evaluator()
	assert False == e.predict(OTHER_IMAGE_PATH)
