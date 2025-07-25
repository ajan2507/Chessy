---
comments: true
description: دليل لاختبار نماذج YOLOv8 الصحيحة. تعرف على كيفية تقييم أداء نماذج YOLO الخاصة بك باستخدام إعدادات ومقاييس التحقق من الصحة مع أمثلة برمجية باللغة البايثون وواجهة سطر الأوامر.
keywords: Ultralytics, YOLO Docs, YOLOv8, التحقق من الصحة, تقييم النموذج, المعلمات الفرعية, الدقة, المقاييس, البايثون, واجهة سطر الأوامر
---

# التحقق من النماذج باستخدام Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="بيئة النظام البيئي والتكاملات لـ Ultralytics YOLO">

## مقدمة

يعتبر التحقق من النموذج خطوة حاسمة في خط أنابيب التعلم الآلي، حيث يتيح لك تقييم جودة النماذج المدربة. يوفر وضع الـ Val في Ultralytics YOLOv8 مجموعة أدوات ومقاييس قوية لتقييم أداء نماذج الكشف عن الكائنات الخاصة بك. يعمل هذا الدليل كمصدر كامل لفهم كيفية استخدام وضع الـ Val بشكل فعال لضمان أن نماذجك دقيقة وموثوقة.

## لماذا يوفر Ultralytics YOLO التحقق من الصحة

هنا هي الأسباب التي تجعل استخدام وضع الـ Val في YOLOv8 مفيدًا:

- **الدقة:** الحصول على مقاييس دقيقة مثل mAP50 و mAP75 و mAP50-95 لتقييم نموذجك بشكل شامل.
- **الراحة:** استخدم الميزات المدمجة التي تتذكر إعدادات التدريب، مما يبسط عملية التحقق من الصحة.
- **مرونة:** قم بالتحقق من النموذج باستخدام نفس المجموعات البيانات وأحجام الصور أو مجموعات بيانات وأحجام صور مختلفة.
- **ضبط المعلمات الفرعية:** استخدم المقاييس التحقق لضبط نموذجك لتحسين الأداء.

### الميزات الرئيسية لوضع الـ Val

هذه هي الوظائف المميزة التي يوفرها وضع الـ Val في YOLOv8:

- **الإعدادات التلقائية:** يتذكر النماذج إعدادات التدريب الخاصة بها للتحقق من الصحة بسهولة.
- **دعم متعدد المقاييس:** قيم نموذجك بناءً على مجموعة من مقاييس الدقة.
- **واجهة سطر الأوامر وواجهة برمجة Python:** اختر بين واجهة سطر الأوامر أو واجهة برمجة Python حسب تفضيلك للتحقق من الصحة.
- **توافق البيانات:** يعمل بسلاسة مع مجموعات البيانات المستخدمة خلال مرحلة التدريب بالإضافة إلى مجموعات البيانات المخصصة.

!!! Tip "نصيحة"

    * تتذكر نماذج YOLOv8 إعدادات التدريب تلقائيًا، لذا يمكنك التحقق من النموذج بنفس حجم الصورة وعلى مجموعة البيانات الأصلية بسهولة باستخدام "yolo val model=yolov8n.pt" أو "model('yolov8n.pt').val()"

## أمثلة الاستخدام

تحقق من دقة النموذج المدرب YOLOv8n على مجموعة بيانات COCO128. لا يلزم تمرير أي وسيطة كوسيطة يتذكر الـ model التدريب والوسيطات كسمات النموذج. انظر الجدول أدناه للحصول على قائمة كاملة من وسيطات التصدير.

!!! Example "مثال"

    === "البايثون"

        ```python
        from ultralytics import YOLO

        # تحميل النموذج
        model = YOLO('yolov8n.pt')  # تحميل النموذج الرسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مخصص

        # التحقق من النموذج
        metrics = model.val()  # لا يلزم أي وسيطات، يتذكر التكوين والوسيطات
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # قائمة تحتوي على map50-95 لكل فئة
        ```
    === "واجهة سطر الأوامر"

        ```bash
        yolo detect val model=yolov8n.pt  # تجريب نموذج رسمي
        yolo detect val model=path/to/best.pt  # تجٌَرب نموذج مخصص
        ```

## الوسيطات

تشير إعدادات التحقق بالنسبة لنماذج YOLO إلى المعلمات الفرعية والتكوينات المختلفة المستخدمة لتقييم أداء النموذج على مجموعة بيانات التحقق. هذه الإعدادات يمكن أن تؤثر على أداء النموذج وسرعته ودقته. تشمل بعض إعدادات التحقق الشائعة في YOLO حجم الدفعة وتكرارات تنفيذ التحقق أثناء التدريب والمقاييس المستخدمة لتقييم أداء النموذج. العوامل الأخرى التي قد تؤثر على العملية الخاصة بالتحقق تشمل حجم وتركيب مجموعة البيانات التحقق والمهمة المحددة التي يتم استخدام النموذج فيها. من المهم ضبط هذه الإعدادات وتجربتها بعناية لضمان أداء جيد للنموذج على مجموعة بيانات التحقق وكشف ومنع الحالة التي يتم فيها ضبط الطراز بشكل جيد.

| مفتاح         | القيمة  | الوصف                                                                              |
|---------------|---------|------------------------------------------------------------------------------------|
| `data`        | `None`  | مسار إلى ملف البيانات، على سبيل المثال coco128.yaml                                |
| `imgsz`       | `640`   | حجم الصور الداخلية باعتبارها عدد صحيح                                              |
| `batch`       | `16`    | عدد الصور لكل دفعة (-1 للدفع الآلي)                                                |
| `save_json`   | `False` | حفظ النتائج في ملف JSON                                                            |
| `save_hybrid` | `False` | حفظ النسخة المختلطة للتسميات (التسميات + التنبؤات الإضافية)                        |
| `conf`        | `0.001` | حد الثقة في كشف الكائن                                                             |
| `iou`         | `0.6`   | حد تداخل على المتحدة (IoU) لعملية الجمع والطرح                                     |
| `max_det`     | `300`   | العدد الأقصى من الكشفات لكل صورة                                                   |
| `half`        | `True`  | استخدم التنصت نصف الدقة (FP16)                                                     |
| `device`      | `None`  | الجهاز الذي يتم تشغيله عليه، على سبيل المثال جهاز Cuda=0/1/2/3 أو جهاز=معالج (CPU) |
| `dnn`         | `False` | استخدم OpenCV DNN لعملية التنصت الأمثل                                             |
| `plots`       | `False` | إظهار الرسوم البيانية أثناء التدريب                                                |
| `rect`        | `False` | تحقق صيغة *rectangular* مع تجميع كل دفعة للحصول على الحد الأدنى من التعبئة         |
| `split`       | `val`   | اختر تقسيم البيانات للتحقق من الصحة، على سبيل المثال "val"، "test" أو "train"      |
|
