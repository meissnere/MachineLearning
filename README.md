# Machine Learning Graduate Studies
The purpose of these notes are to
- Identify different types of machine learning problems
- Know what a machine learning model is
- Know the general workflow of building and applying a machine learning model
- Know the advantages and disadvantages about machine learning problems

## Machine Learning Model
To start off, machine learning nowadays does not go beyond a computer program that
performs the predefined procedures. What truly distinguished a machine learning algorithm
from a non machine learning algorithm, such as a program that controls traffic lights,
is its ability to **adapt** its behaviours to new input. This adaption seems to imply
that the machine is actually **learning**. However, under the hood, this perceived
adaption of behaviours is as rigid as every bit of machine instructions.

A machine learning algorithm is the process that uncovers the underlying relationship within
the data. The outcome is called the **machine learning model**. The model changes when fed
different data, meaning the function is not predefined and fixed.

The input of the generated model here would be a digital photo, and the output is a boolean
value indicating the existence of a cat in the photo.

![image_recogntions](assets/image_recognition.png)

The model for the above case is a function that maps multiple dimensional pixel values
to a binary value. For example, assume we have a photo of 3 pixels, and the value of each
pixel ranges from 0 to 255. The mapping space between the input and output would be
**(256 X 256 X 256) X 2**, which is around 33 million. This task of learning the mapping
might be daunting; however,
```
The task of machine learning is to learn the function from the vast mapping space
```
The results of the machine learning model is often not 100% accurate. Before the wide
application of deep learning in 2012, the best machine learning model could only achieve
75% accuracy in the [ImageNet visual recognition challenge](http://www.image-net.org/).

### Supervised VS. Unsupervised
When first given a machine learning problem, one can determine whether it is a **supervised**
or **unsupervised** problem.
