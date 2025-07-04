#  PyTorch MNIST Neural Network – School Project

This is a hands-on school project where I implemented, trained, and tested both a **feedforward neural network (FFN)** and a **convolutional neural network (CNN)** using **PyTorch** on the **MNIST handwritten digits dataset**.

---

## Project Highlights

-  Tensors: Creation, shaping, and operations
-  Autograd: Tracked gradients and computation graph with `requires_grad=True`
-  FFN: Built using `torch.nn.Module` and trained with SGD
-  CNN: Built custom convolutional model, trained with Adam optimizer
-  Achieved **98% test accuracy**
-  Visualized predictions and learned features

---

##  Key Files

| File                               | Description                                   |
|------------------------------------|-----------------------------------------------|
| `07-pytorch_cnn-exercise.ipynb`    | Tensors, autograd, and gradient walkthrough   |
| `07-pytorch_cnn_exercise.ipynb`    | Neural net training, CNN building, evaluation |
| `requirements.txt`                 | Dependencies to run the project               |
| `.github/workflows/python-app.yml` | CI workflow to test notebook execution      |

---

## What This Project Does (Detailed Overview)

This Jupyter-based school project demonstrates the core concepts of neural networks using PyTorch through a hands-on implementation on the MNIST handwritten digit dataset.

It is split into two major parts:

⸻

# Part 1: PyTorch Basics & Autograd

This section walks through the foundation of PyTorch:
	•	Tensor operations: Creating and manipulating tensors using torch.Tensor, torch.rand(), .view(), and more
	•	Gradient tracking: Using requires_grad=True to compute derivatives automatically
	•	Computation graphs: Understanding how PyTorch builds and traverses the graph when computing gradients
	•	Backpropagation by hand: Calculating gradients using .backward() and verifying them with the chain rule

Objective: Learn the internal mechanics of deep learning step-by-step.

⸻

# Part 2: Building & Training Neural Networks

This section builds a complete digit recognition system from scratch.

- Feedforward Neural Network (FFN)
	•	Custom Net class using torch.nn.Module
	•	Uses 2 linear layers with Sigmoid activation
	•	Trained with:
	•	nn.CrossEntropyLoss for classification
	•	torch.optim.SGD optimizer
	•	Achieved up to 97–98% accuracy on the MNIST test set

- Convolutional Neural Network (ConvNet)
	•	Built a custom ConvNet class with:
	•	nn.Conv2d, nn.MaxPool2d, and nn.Linear
	•	ReLU and MaxPooling layers for feature extraction
	•	Trained with Adam optimizer for faster convergence
	•	Model achieves 98%+ accuracy with fewer epochs
	•	Tested with unseen data using model.eval() for predictions

⸻

- Extra Features
	•	Filter analysis: Visualizes what the convolution layers learned
	•	Test evaluation: Calculates prediction accuracy on unseen data
	•	Visual debugging: Uses matplotlib to plot input digits and predictions
	•	Batch training: Implements data batching using torch.utils.data.DataLoader

⸻

# Final Outcome

-By the end of the notebook, you’ll have:
	•	Trained two models (FFN + CNN) for handwritten digit classification
	•	Achieved 98% accuracy
	•	Understood key neural network building blocks in PyTorch
	•	Learned how to debug, train, and evaluate models end-to-end



##  How to Run This Project

```bash
# 1. Clone the repository
git clone https://github.com/Stanleymikky00/pytorch-mnist-neural-network.git
cd pytorch-mnist-neural-network

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook

