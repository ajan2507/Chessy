<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from .tasks import (BaseModel, ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight,
                    attempt_load_weights, guess_model_scale, guess_model_task, parse_model, torch_safe_load,
                    yaml_model_load)

__all__ = ('attempt_load_one_weight', 'attempt_load_weights', 'parse_model', 'yaml_model_load', 'guess_model_task',
           'guess_model_scale', 'torch_safe_load', 'DetectionModel', 'SegmentationModel', 'ClassificationModel',
           'BaseModel')
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    attempt_load_one_weight,
    attempt_load_weights,
    guess_model_scale,
    guess_model_task,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

__all__ = (
    "attempt_load_one_weight",
    "attempt_load_weights",
    "parse_model",
    "yaml_model_load",
    "guess_model_task",
    "guess_model_scale",
    "torch_safe_load",
    "DetectionModel",
    "SegmentationModel",
    "ClassificationModel",
    "BaseModel",
)
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
