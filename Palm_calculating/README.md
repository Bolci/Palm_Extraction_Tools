# Palm calculation algorihm

This folder contains Palm calculation algorithm described in paper:
Spatial–spectral Transformer-based Semantic Segmentation for a City-level Surveying of Individual Date Palm Trees using Very-High-Resolution Satellite Data

![alt text](http://url/to/img.png)

## Requirements


## Usage

<pre>
python3 main.py input_folder config checkpoint
</pre>

#### Parameters

- `input_folder`: path to folder with input images to inference, where the palm should be segmented and calculated
- `config`: path to config file for mmsegmentation model
- `checkpoint`: path to trained checkpoint of mmsegmentation model
- `--out `: path to folder where to save output files (default value: `./`)
- `--georeference` / `--no-georeference` : Set if the results should be saved in georefereced verison also (default value: `True`)
- `--vectorize` / `--no-vectorize`: Set of the results should be converted into .geojson file also (default value: `False`)



