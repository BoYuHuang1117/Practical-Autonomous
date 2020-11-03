# Practical Autonomous

### All the code is programmed in Python

* Lane detection

<img width="910" alt="Capture" src="https://user-images.githubusercontent.com/38172621/98052459-ed4a5300-1dfb-11eb-9c94-881afd7603e9.PNG">


* traffic sign recognition

- load the complete dataset with labels

- transform the size of the image into 32x32

- build the Convolutional Neural Netwrok

- train the model

- testing with the image found on the websites

## Sample image
### Image in size of 32x32 (RGB)
<img width="128" alt="Capture1" src="https://user-images.githubusercontent.com/38172621/98053044-4ff01e80-1dfd-11eb-8900-0bbcbf29c50b.PNG">

### Image in size of 32x32 (HSV)
<img width="125" alt="Capture2" src="https://user-images.githubusercontent.com/38172621/98053042-4ff01e80-1dfd-11eb-82d3-0a32f86a1aa2.PNG">

## Sample output
| label # |  labelname  |	softmax Prob|
| ------- | ----------- | ------- |
| 14 | Stop |	0.998944 |
| 33 | Turn right ahead |	0.000532
| 29 | Bicycles crossing | 0.000311
| 34 | Turn left ahead | 0.000118
| 36 | Go straight or right | 0.000095
