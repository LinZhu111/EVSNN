
# Event-based Video Reconstruction via Potential-assisted Spiking Neural Network
Minimal code for running inference on spiking neural network trained for Event-based Video Reconstruction via Potential-assisted Spiking Neural Network, CVPR2022.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [EVSNN](#evsnn)
	* [Requirements](#requirements)
	* [Folder Structure](#folder-structure)
	* [Network](#network)
	* [Usage](#usage)
	* [Dataset](#dataset)
	* [License](#license)

<!-- /code_chunk_output -->

=======================================================================
## Requirements

* Python >= 3.7 (3.9 recommended)
* PyTorch >= 1.6 (1.9 recommended)
* Spikingjelly >= 0.0.0.0.4 (lastest version)
======================================================================
## Running with Anaconda
cuda_version=10.2
conda create -n snnrec 
conda activate snnrec 
conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
conda install pandas

## Install Spikingjelly
pip install spikingjelly

=====================================================================
## Inference
Usage:
python rec_snn.py [-network NETWORK] [-path_to_pretrain_models PATH_TO_PRETRAIN_MODELS] [-path_to_event_files PATH_TO_EVENT_FILES] [-save_path SAVE_PATH] [-height HEIGHT] [-width WIDTH] [-num_events_per_pixel NUM_EVENTS_PER_PIXEL]

For example, to run EVSNN:
python rec_snn.py -network EVSNN_LIF_final -path_to_pretrain_models ./pretrained_models/EVSNN.pth

To run PA-EVSNN
python rec_snn.py -network PAEVSNN_LIF_AMPLIF_final -path_to_pretrain_models ./pretrained_models/PAEVSNN.pth

======================================================================
## Folder Structure
  minimal_code_snn/
  |
  ├── rec_snn.py - evaluation of trained model
  |
  ├── data/ - default directory for storing input data
  |
  ├── model/ - models, losses, and metrics
  |   ├── dataset.py
  |   ├── snn_network.py
  |
  ├── neurons/  
  |	 ├── spiking_neuron.py - spiking neurons, MP neurons
  |
  ├── results/  - generated results are saved here
  |  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  


