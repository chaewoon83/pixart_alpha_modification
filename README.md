# PixArt Alpha - Base vs Routed FFN Model Comparison

A comprehensive toolkit for comparing PixArt base models with Routed FFN optimization variants. Includes inference capabilities, web-based GUI, and training utilities.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [GUI Inference](#gui-inference)
3. [Training](#training)

---

## Environment Setup

### Option 1: Using Conda (Recommended)

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate PixArt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Option 2: Using pip with requirements.txt

1. **Create a virtual environment:**
   ```bash
   python -m venv pixart_env
   ```

2. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     pixart_env\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source pixart_env/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## GUI Inference

### Launch the Streamlit Web Interface

1. **Ensure your environment is activated:**
   ```bash
   conda activate PixArt
   # or
   source pixart_env/bin/activate
   ```

2. **Start the Streamlit app:**
   ```bash
   streamlit run GUI/streamlit.py
   ```

3. **Access the UI:**
   - Open your browser and navigate to: `http://localhost:8501`

### Using the GUI

1. **Enter a prompt** in the text area (e.g., "A child in a red raincoat holding a yellow umbrella")

2. **Set the seed** for reproducibility

3. **Click Generate** to:
   - Run inference on both Base and Routed models
   - Compare generated images side-by-side
   - View inference time and token routing statistics

### Output Interpretation

- **Base Model**: Original PixArt model results
- **Routed Model**: PixArt with Routed FFN optimization
- **Token Routing Ratio**: Percentage of tokens processed by Light vs Heavy FFN modules
  - **Light FFN**: Faster, lower-quality computation path
  - **Heavy FFN**: Slower, higher-quality computation path
- **Inference Time**: Total time to generate the image
- **Token Counts**: Number of tokens routed to each FFN type

---

## Training

### Train Routed FFN Model

1. **Prepare your dataset** in the appropriate format (see `diffusion/data/datasets/`)

2. **Configure training parameters** in your chosen config file:
   ```bash
   configs/pixart_config/PixArt_xl2_img256_small_Routed.py
   ```

3. **Run training:**
   ```bash
   python train_scripts/train_test.py \
     --config configs/pixart_config/PixArt_xl2_img256_small_Routed.py \
     --output_dir ./output/Routed_Model \
     --num_train_epochs 100 \
     --train_batch_size 8 \
     --learning_rate 1e-4 \
     --mixed_precision fp16
   ```

4. **Monitor training:**
   - Check logs in the output directory
   - Use TensorBoard to visualize metrics:
     ```bash
     tensorboard --logdir ./output/Routed_Model
     ```

### Training Script Options

- `train_scripts/train.py`: Standard training
- `train_scripts/train_test.py`: Test/development training
- `train_scripts/train_pixart_lcm.py`: LCM-specific training
- `train_scripts/train_dreambooth.py`: DreamBooth fine-tuning
- `train_scripts/train_selective_ffn.py`: Train selective FFN routing

### LCM (Latent Consistency Model) Training

For faster inference during training:

```bash
python train_scripts/train_pixart_lcm.py \
  --config configs/pixart_config/PixArt_xl2_img256_small_Routed.py \
  --output_dir ./output/Routed_LCM \
  --num_train_epochs 50
```
---

## Common Commands

### Inference via CLI

Generate a single image using the command line:

```bash
python tools/infer_pixart.py \
  configs/pixart_config/PixArt_xl2_img256_small.py \
  output/Base_Model/Base256x256.pth \
  --prompt "A beautiful landscape" \
  --output_dir ./InferenceDatas/Original \
  --seed 42 \
  --collect_stats
```