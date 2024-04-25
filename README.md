# SFDF-Nets
Deep learning-based image reconstruction for photonic integrated interferometric imaging.（Optics Express2022）

 
# README.md

## 数据集路径地址

**① Gopro dense:** `oss://pjlab-3090-sail/zhangziran/Syn_DATASET/GOPRO/`

**② 二阶稀疏采样Event数据集:** `oss://pjlab-3090-sail/zhangziran/workspace_sail/SynData4event/gray_event_10_new_fliter_2rd/`

**③ 实拍数据集路径:** `oss://pjlab-3090-sail/zhangziran/Self_collected_DATASET/imgs/2024_04_10/`

实拍数据中有效的5个dict：

```python
dataset_dict = {
    "2024-04-10-21-38": [10,180],
    "2024-04-10-21-43": [10,180],
    "2024-04-10-21-47": [10,180],
    "2024-04-10-21-51": [10,180],
    "2024-04-10-21-55": [10,180],
}
test_key = [
    "2024-04-10-21-38",
    "2024-04-10-21-43",
    "2024-04-10-21-47",
    "2024-04-10-21-51",
    "2024-04-10-21-55",
]
```

## 如何self-tuning

**① 下载code，地址如下:** `oss://pjlab-3090-sail/zhangziran/workspace_sail/base_code/E-TRFNet/`

**② 安装依赖库:**
```bash
pip install requirements.txt
```

**③ 运行bash  self_rgbclean.sh即可self-tuning**
需要:
- 注释`# FLAGS2="--skip_training "`
- 设置权重的路径`MODEL_PRETRAINED1`
- 设置结果保存路径`PARAM_NAME1="timelens_RC_x8_lpips_lpfv2_selftuning"`

## selftuning的shell代码

通过调用test_key中的idx依次对模型进行self-tuning，0..4为test_key中的5个key。

```bash
MODEL_NAME1="TimeLens"
PARAM_NAME1="timelens_RC_x8_lpips_lpfv2_selftuning"
MODEL_PRETRAINED1="/mnt/data/oss_beijing/zhangziran/Experiment_result/output/timelens_tuning_rgbclean_event_10_new_fliter_2rd_local/GOPRO_TimeLens_tuning/weights/TimeLens_12.pt"
FLAGS1="--save_flow True"
# FLAGS2="--skip_training "
# Loop from 1 to 5 and execute the python command with the current index
for idx in {0..4}
    echo "$idx"
    FLAGS3="--STN $idx"
    # Run the Python script with the current settings
    python run_network.py --model_name "$MODEL_NAME1" --param_name "$PARAM_NAME1" --model_pretrained "$MODEL_PRETRAINED1" $FLAGS1 $FLAGS2 $FLAGS3
done
```

## 注意事项

- **a. 二阶稀疏采样v2e的预训练权重为:** `/mnt/data/oss_beijing/zhangziran/Experiment_result/output/timelens_tuning_rgbclean_event_10_new_fliter_2rd_local/GOPRO_TimeLens_tuning/weights/TimeLens_12.pt`
- **b. 参数为：timelens_RC_x8_lpips_lpfv2_selftuning，参数中可设置结果的保存路径:** `/mnt/workspace/zhangziran/base_code/E-TRFNet/Real_results/event_lowlight_lpf_2rd_local/selftuning_nonlinear_x2tuning_rgbx4_12`
- **c. 权重TimeLens_27.pt为times直接在仿真数据上端到端训的，效果表现不如导入预训练再在二阶稀疏采样仿真数据上tuning的TimeLens_12.pt** `/mnt/data/oss_beijing/zhangziran/Experiment_result/output/timelens_training_rgbclean_event_10_new_fliter_2rd_local/GOPRO_TimeLens_tuning/weights/TimeLens_27.pt`
