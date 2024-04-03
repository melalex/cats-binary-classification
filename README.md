# cats-binary-classification

Logistic regression based app that determines whether provided image contains cat or not. It uses [microsoft-catsvsdogs-dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) and [random-image-sample-dataset](https://www.kaggle.com/datasets/pankajkumar2002/random-image-sample-dataset) datasets for training and testing.

Various parameters of this app could be modified in a [definitions.py](src/definitions.py)

## Run

```bash
make predict filename='/path/to/the/image.jpg'
```

## Run tests

```bash
make test
```
