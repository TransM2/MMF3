# MMF3: Neural Code Summarization Based on Multi-Modal Fine-Grained Feature Fusion
We public the source code and datasets for MMF3.

## Datasets
In the MMF3, we use two large-scale datasets for experiments, including one Java and one Python datasets. In data file, we give the two datasets, which obtain from following paper. If you want to train the model, you must download the datasets.

### JHD(Java Hu Dataset)
 * paper: [https://xin-xia.github.io/publication/ijcai18.pdf](https://xin-xia.github.io/publication/ijcai18.pdf)
 * data: [https://github.com/xing-hu/TL-CodeSum](https://github.com/xing-hu/TL-CodeSum)

### PBD(Python Barone Dataset)
 * paper: [https://arxiv.org/abs/1707.02275](https://arxiv.org/abs/1707.02275)
 * data: [https://github.com/EdinburghNLP/code-docstring-corpus](https://github.com/EdinburghNLP/code-docstring-corpus)

## Data preprocessing
MMF3 uses source code and ASTs modalities, which uses the [JDK](http://www.eclipse.org/jdt/) compiler to parse java methods as ASTs, and the [Treelib](https://treelib.readthedocs.io/en/latest/) toolkit to prase Python functions as ASTs. In addition, before embedding ASTs, we use BERT pre-training to embed the information of nodes.

## Get ASTs
In Data_pred file, the `get_Java_ast.py` generates ASTs for a Java dataset and `get_python_ast.py` generates ASTs for Python functions. You can run the following command：<br>


## Requirements
  * Pytorch 1.8.0
  * Python 3.6.5
  * Network 2.3
  * Numpy 1.19.5
  * Nltk 3.6.2
  * Pandas 1.1.5
  * Tensorflow 1.15.0
  * bert-serving-client 1.10.0
  * bert-serving-server 1.10.0
