# tuesdayGAN
A generative adversarial network that loves sound art.

## How to use

1. Preprocess the data: set paths to the .wav and .npy files in `prepare_n_process/wav2npy.py`, then run this script &ndash; it converts your music file to a dataset.
2. Uncomment one option in `nn/T_launcher.py` (depending on if you wish to *train* or *test* the model) and comment the other one, then run this script.

## Dependencies

1. TensorFlow (v1.9 and above works fine) 
2. NumPy
3. SciPy 
4. Matplotlib

I apologize for the mess! Documentation in progress...
