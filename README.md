# Image Token Matters: Mitigating Hallucination in LVLMs via Latent Editing

This repository contains the official implementation of the paper "Image Token Matters: Mitigating Hallucination in Discrete Tokenizer-based Large Vision-Language Models via Latent Editing".

## Overview
Large Vision-Language Models (LVLMs) with discrete image tokenizers unify multimodal representations by encoding visual inputs into a finite set of tokens. Despite their effectiveness, we find that these models still hallucinate non-existent objects. We hypothesize that one reason is due to visual priors induced during training: when certain image tokens frequently co-occur in the same spatial regions and represent shared objects, they become strongly associated with the verbalizations of those objects. As a result, the model may hallucinate by evoking visually absent tokens that often co-occur with present ones. To test this assumption, we construct a co-occurrence graph of image tokens using a segmentation dataset and employ a Graph Neural Network (GNN) with contrastive learning followed by a clustering method to group tokens that frequently co-occur in similar visual contexts. We find that hallucinations predominantly correspond to clusters whose tokens dominate the input, and more specifically, that the visually absent tokens in those clusters show much higher correlation with hallucinated objects compared to tokens present in the image. Based on this observation, we propose a hallucination mitigation method that suppresses the influence of visually absent tokens by modifying latent image embeddings during generation. Experiments show our method reduces hallucinations while preserving expressivity.
## Environment Setup

Please use conda to create the experiment environment and install additional packages with `pip`

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cgc-vtd



# install pyg
conda install pytorch-sparse -c pyg
conda install pyg -c pyg

# Install additional dependencies
pip install git+https://github.com/cocodataset/panopticapi.git
pip install 'accelerate>=0.26.0'
cd transformers
pip install -e .
```

## Prepare Data
To train CGC, we need the MSCOCO image data with panoptic segmentation annotations. Please download the MSCOCO 2017 val set from [HERE](http://images.cocodataset.org/zips/val2017.zip) as well as the Panoptic annotations from [HERE](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip).
organize the file structure as follows:
```bash
COCO_root/
├── annotations/
│   ├── panoptic_val2017/
│   ├── panoptic_val2017.json
│   └── panoptic_val2017.zip
├── val2017/
```
## Implementation
To build the co-occurrence graph, simply run 
```bash
python src/cgc/build_codebook_graph_coco.py --model_cache_dir ~/.huggingface --coco_root COCO_root --save_dir path/to/save/graph --model_path deepseek-ai/Janus-Pro-7B --model_type janus 
```

After finishing the graph building, you can start to train the GNN model for graph-based embeddings.
```bash
python arc/cgc/train_gnn_codebook.py --model_path deepseek-ai/Janus-Pro-7B --model_type janus --graph_data_dir path/to/save/graph --save_dir path/to/save/GNN/parameters --hidden_dim GNN_dimension --output_dim output_dimension --num_layers number_of_GNN_layer 
```
Finally, perform the modified K-means clustering to group codebook tokens.
```bash
python src/cgc/train_codebook_clustering.py --model_type janus --output_dir clustering/dir/path  --model_path deepseek-ai/Janus-Pro-7B --n_clusters number_of_total_clusters --cache_dir ~/.huggingface --balanced --gnn_model_path  path/to/save/GNN/parameters
```
## Evaluation
### AMBER
[AMBER](https://github.com/junyangwang0410/AMBER) is An LLM-free Multi-dimensional Benchmark for MLLMs hallucination evaluation, which can be used to evaluate both generative task and discriminative task including existence, attribute and relation hallucination.

For perform evaluation, you need to download the image data, query files, and annotations as instructed [HERE](https://github.com/junyangwang0410/AMBER)

To run CGC+VTD:
```bash
src/amber_generate.py  --image_dir path/to/AMBER/image/ --output_file output/path/amber.json --gen_type gnn --cluster_results_path clustering/dir/path/clustering_results.pkl --model_cache_dir ~/.huggingface --model_type janus --model_id deepseek-ai/Janus-Pro-7B --find_non_input_indices --weight editing_weight --layer editing_layer --num_clusters number_of_dominant_clusters
```

available generation method are: `--gen_type [vcd, sid, confidence, opera]`, correspond to [[VCD](https://github.com/DAMO-NLP-SG/VCD),[SID](https://github.com/huofushuo/SID),[PROJECTAWAY](https://github.com/nickjiang2378/vl-interp/tree/main),[OPERA](https://github.com/shikiw/OPERA)]

### Object HalBench
Object HalBench is a widely used LLM-assisted benchmark for object hallucination. We follow the implementations from [HERE](https://github.com/RLHF-V/RLAIF-V). We use the 300 carefully selected samples for evaluation.
To run the evalaution:
```bash
python src/objhal_generate.py  --query_file path/to/obj_halbench_300_with_image.jsonl --image_dir path/to/coco2014/images/val2014 --model_type janus --model_id deepseek-ai/Janus-Pro-7B --answers_file path/to/answer/file  --gen_type gnn --cluster_results_path clustering/dir/path/clustering_results.pkl --weight edit_weight --layer editing_layer --find_non_input_indices --model_cache_dir ~/.huggingface --num_clusters number_of_dominant_clusters
```
### MME
[MME]([https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation/tools](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)) measures both perception and cognition abilities on a total of 14 subtasks. To use this benchmark, please first download the images from [HERE](https://huggingface.co/datasets/lmms-lab/MME) and put it to `paht/to/MME_Release`. Also download the query files and put it under the `root/to/mme/results` dir. To run the evaluation:
```bash
python src/mme_generate.py --model_id deepseek-ai/Janus-Pro-7B --model_type janus --image_dir path/to/MME_Release --operation subtract  --root_dir root/to/mme/results --gen_type gnn --cluster_results_path clustering/dir/path/clustering_results.pkl --model_cache_dir ~/.huggingface  --find_non_input_indices --weight edit_weight --layer editing_layer --find_non_input_indices --model_cache_dir ~/.huggingface --num_clusters number_of_dominant_clusters
```
