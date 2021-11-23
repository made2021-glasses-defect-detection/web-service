# Test

```sh
python -m pytest test_classification_efficientnet.py
```

# Build Image

```sh
docker build -t input-classifier .
```

# Test Image

```sh
docker run -it -v /E//Soft/MADE/semester_3/diploma/model/input_classifier:/app/images input-classifier bash
```