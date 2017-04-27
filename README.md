# Convolutinal Neural Network for Digit and Character Recognition using Tensor Flow

Project consist in the development of an AI model that
can be able to recognize handwritten digits and characters 
of a socioeconomic survey [Link]().

In order to achive the highest possible accuracy, we use
a convolutional neural network and we take advantage of shape
detection.

To train our models we've used the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19).

We've implemented using [Tensor Flow](https://www.tensorflow.org/) framework
in python.



Table below shows performance of several convNet models we've built:

| Model							| Accuracy | Comments |
|-------------------------------|----------|----------|
| Model C1						| 0.953    |          |
| Model C2                      | 0.942    |     	  |
| Model C3                      | 0.968    |     	  |
|                               |          |     	  |
| Model D1                      | 0.974    |     	  |
| Model D2_                     | 0.978    | Slow     |
| Model D2                      | 0.981    | Fast     |


## How the project is organised

Project consist of three main directories:

* Extraction / Contain the scripts that extract individual characters of the survey form
	** Detect page,
	** Feature extractor
* Modeling / Contains convnet models designed
* Api / Script that integrates the above.
	** Engine
