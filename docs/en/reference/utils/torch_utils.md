<<<<<<< HEAD
---
description: Explore Ultralytics-tailored torch utility features like Model EMA, early stopping, smart inference, image scaling, get_flops, and many more.
keywords: Ultralytics, Torch Utils, Model EMA, Early Stopping, Smart Inference, Get CPU Info, Time Sync, Fuse Deconv and bn, Get num params, Get FLOPs, Scale img, Copy attr, Intersect dicts, De_parallel, Init seeds, Profile
---

# Reference for `ultralytics/utils/torch_utils.py`

!!! Note

    This file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py). If you spot a problem please help fix it by [contributing](https://docs.ultralytics.com/help/contributing/) a [Pull Request](https://github.com/ultralytics/ultralytics/edit/main/ultralytics/utils/torch_utils.py) 🛠️. Thank you 🙏!

<br><br>

## ::: ultralytics.utils.torch_utils.ModelEMA

<br><br>

## ::: ultralytics.utils.torch_utils.EarlyStopping

<br><br>

## ::: ultralytics.utils.torch_utils.torch_distributed_zero_first

<br><br>

## ::: ultralytics.utils.torch_utils.smart_inference_mode

<br><br>

## ::: ultralytics.utils.torch_utils.get_cpu_info

<br><br>

## ::: ultralytics.utils.torch_utils.select_device

<br><br>

## ::: ultralytics.utils.torch_utils.time_sync

<br><br>

## ::: ultralytics.utils.torch_utils.fuse_conv_and_bn

<br><br>

## ::: ultralytics.utils.torch_utils.fuse_deconv_and_bn

<br><br>

## ::: ultralytics.utils.torch_utils.model_info

<br><br>

## ::: ultralytics.utils.torch_utils.get_num_params

<br><br>

## ::: ultralytics.utils.torch_utils.get_num_gradients

<br><br>

## ::: ultralytics.utils.torch_utils.model_info_for_loggers

<br><br>

## ::: ultralytics.utils.torch_utils.get_flops

<br><br>

## ::: ultralytics.utils.torch_utils.get_flops_with_torch_profiler

<br><br>

## ::: ultralytics.utils.torch_utils.initialize_weights

<br><br>

## ::: ultralytics.utils.torch_utils.scale_img

<br><br>

## ::: ultralytics.utils.torch_utils.make_divisible

<br><br>

## ::: ultralytics.utils.torch_utils.copy_attr

<br><br>

## ::: ultralytics.utils.torch_utils.get_latest_opset

<br><br>

## ::: ultralytics.utils.torch_utils.intersect_dicts

<br><br>

## ::: ultralytics.utils.torch_utils.is_parallel

<br><br>

## ::: ultralytics.utils.torch_utils.de_parallel

<br><br>

## ::: ultralytics.utils.torch_utils.one_cycle

<br><br>

## ::: ultralytics.utils.torch_utils.init_seeds

<br><br>

## ::: ultralytics.utils.torch_utils.strip_optimizer

<br><br>

## ::: ultralytics.utils.torch_utils.profile

<br><br>
=======
---
description: Explore valuable torch utilities from Ultralytics for optimized model performance, including device selection, model fusion, and inference optimization.
keywords: Ultralytics, torch utils, model optimization, device selection, inference optimization, model fusion, CPU info, PyTorch tools
---

# Reference for `ultralytics/utils/torch_utils.py`

!!! note

    This file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py). If you spot a problem please help fix it by [contributing](https://docs.ultralytics.com/help/contributing/) a [Pull Request](https://github.com/ultralytics/ultralytics/edit/main/ultralytics/utils/torch_utils.py) 🛠️. Thank you 🙏!

<br>

## ::: ultralytics.utils.torch_utils.ModelEMA

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.EarlyStopping

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.FXModel

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.torch_distributed_zero_first

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.smart_inference_mode

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.autocast

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_cpu_info

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_gpu_info

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.select_device

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.time_sync

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.fuse_conv_and_bn

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.fuse_deconv_and_bn

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.model_info

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_num_params

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_num_gradients

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.model_info_for_loggers

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_flops

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_flops_with_torch_profiler

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.initialize_weights

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.scale_img

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.copy_attr

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.get_latest_opset

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.intersect_dicts

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.is_parallel

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.de_parallel

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.one_cycle

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.init_seeds

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.unset_deterministic

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.strip_optimizer

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.convert_optimizer_state_dict_to_fp16

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.cuda_memory_usage

<br><br><hr><br>

## ::: ultralytics.utils.torch_utils.profile_ops

<br><br>
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
