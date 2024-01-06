# OSTrack With Template Update 


## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f ostrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```



## Similarity Prediction Model
The similarity prediction model is located at 
```
ostrack_with_template/lib/models/ostrack/similarity_model.py
```

## Inference
For inference you have to download the pretrained file from [Drive](https://drive.google.com/drive/folders/1B1IBkSZ3JXUFH9X5R7Lz-ae_0n8WCVQL)
Locate the pretrained file at 
```
output\checkpoints\train\ostrack\vitb_256_mae_32x4_ep300
```
run the demo with your own video 
```
python tracking/video_demo.py --video_file <YOUR_OWN_VIDEO>
```

