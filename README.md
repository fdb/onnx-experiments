# ONNX Experiments

This repository contains experiments on working with the ONNX format, and specifically using onnxruntime-web to run ONNX models in the browser.

## Setup

I'm using Miniconda to manage my Python environment. I've created a new environment for this course:

```bash
conda create -y -n onnx-experiments python=3.11
conda activate onnx-experiments
conda install -y -c fastai -c conda-forge fastai jupyterlab onnx onnxruntime
```

To activate the environment:

```bash
conda activate onnx-experiments
jupyter lab
```

## Training

I trained a simple image recognition model using the [fast.ai](https://course.fast.ai) appoach. Code is in the [not-hotdog-training.ipynb](not-hotdog/not-hotdog-training.ipynb) notebook.


## References
- [Converting a fastai model to ONNX](https://dev.to/tkeyo/export-fastai-resnet-models-to-onnx-2gj7) ([Jupyter notebook](https://github.com/tkeyo/fastai-onnx/blob/main/fastai_to_onnx.ipynb))
- [Deploy ONNX Runtime on the web](https://onnxruntime.ai/docs/tutorials/web/)
- [Loading the images and converting them to tensors](https://github.com/microsoft/onnxruntime-nextjs-template/blob/main/utils/imageHelper.ts)
