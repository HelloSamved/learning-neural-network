# Neural Ink: MNIST Classifier

An interactive handwritten-digit classifier powered by a neural network built
from scratch with NumPy. The Streamlit interface supports uploaded images,
confidence visualization, real MNIST sample exploration, and an explanation of
the learning pipeline.

## Highlights

- Implements forward propagation, backpropagation, and mini-batch gradient descent
- Uses a `784 -> 28 -> 28 -> 10` dense neural-network architecture
- Achieves 94.7% accuracy on a held-out validation set
- Packages pretrained weights for fast inference
- Preprocesses uploaded images into centered, normalized 28x28 inputs

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run web_interface.py
```

To retrain the model using `train.csv`:

```powershell
python mnist_prediction.py
```

## Project Files

- `mnist_prediction.ipynb`: original model exploration and learning notebook
- `mnist_prediction.py`: corrected reusable neural-network and preprocessing code
- `web_interface.py`: portfolio-ready Streamlit application
- `mnist_weights.npz`: pretrained model parameters and accuracy metadata
- `mnist_samples.npz`: lightweight sample set used by the live explorer