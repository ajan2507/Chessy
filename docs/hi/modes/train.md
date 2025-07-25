---
comments: true
description: Ultralytics YOLO के साथ YOLOv8 मॉडल ट्रेन करने के लिए चरणबद्ध मार्गदर्शिका, एकल-GPU और बहु-GPU ट्रेनिंग के उदाहरणों के साथ।
keywords: Ultralytics, YOLOv8, YOLO, ऑब्जेक्ट डिटेक्शन, ट्रेन मोड, कस्टम डेटासेट, GPU ट्रेनिंग, बहु-GPU, हाइपरपैरामीटर, CLI उदाहरण, Python उदाहरण
---

# Ultralytics YOLO के साथ मॉडल ट्रेनिंग

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO इकोसिस्टम और इंटीग्रेशन">

## परिचय

एक गहरी यान्त्रिकी मॉडल को ट्रेनिंग देना उसे डेटा खिलाते हुए और इसके पैरामीटर्स को समायोजित करके सही पूर्वानुमान करने की सामर्थ्य को शामिल करता है। YOLOv8 मॉडल में Ultralytics YOLO के ट्रेन मोड ने ऑब्जेक्ट डिटेक्शन मॉडल्स को प्रभावी और दक्ष ट्रेनिंग के लिए इंजीनियरिंग किया गया है, जिससे आधुनिक हार्डवेयर क्षमताओं का पूरी तरह से उपयोग किया जा सके। यह मार्गदर्शिका उन सभी विवरणों को कवर करने का उद्देश्य रखती है जो आपको YOLOv8 के मजबूत सेट ऑफ़ सुविधाओं का उपयोग करके अपने खुद के मॉडल्स को ट्रेनिंग शुरू करने के लिए चाहिए।

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube वीडियो प्लेयर" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> Google Colab में अपने कस्टम डेटासेट पर एक YOLOv8 मॉडल को ट्रेन करने का तरीका।
</p>

## प्रशिक्षण के लिए Ultralytics YOLO का चयन क्यों करें?

यहां YOLOv8 के ट्रेन मोड को चुनने के कुछ प्रमुख कारण हैं:

- **दक्षता:** अपने हार्डवेयर से सबसे अधिक लाभ उठाएं, चाहे आप सिंगल-GPU सेटअप पर हों या कई GPU पर स्केल कर रहें हों।
- **प्राक्तिशिल्ता:** COCO, VOC और ImageNet जैसे तत्परता उपलब्ध डेटासेटों के अलावा कस्टम डेटासेट पर ट्रेन करें।
- **उपयोगकर्ता मित्रपूर्णता:** सीधे और शक्तिशाली CLI और Python इंटरफ़ेस का उपयोग एक सीधी ट्रेनिंग अनुभव के लिए।
- **हाइपरपैरामीटर लचीलापन:** मॉडल प्रदर्शन को सुधारने के लिए वैश्विक स्तर पर अनुकूलन योग्य हाइपरपैरामीटरों की एक व्यापक श्रृंखला।

### ट्रेन मोड की प्रमुख सुविधाएं

निम्नलिखित YOLOv8 के ट्रेन मोड की कुछ महत्वपूर्ण सुविधाएं हैं:

- **स्वत: डेटासेट डाउनलोड:** COCO, VOC और ImageNet जैसे मानक डेटासेट्स को पहली बार के उपयोग पर स्वत: डाउनलोड किया जाता है।
- **बहु-GPU समर्थन:** प्रक्रिया की गति को तेज करने के लिए अनुप्रयोग में कई जीपीयू का उपयोग करें।
- **हाइपरपैरामीटर कॉन्फ़िगरेशन:** हाइपरपैरामीटर को यामल कॉन्फ़िगरेशन फ़ाइल या CLI तर्कों के माध्यम से संशोधित करने का विकल्प।
- **दृश्यीकरण और मॉनिटरिंग:** प्रशिक्षण मैट्रिक्स के वास्तविक समय ट्रैकिंग और सीखने की प्रक्रिया के दृश्यीकरण के लिए बेहतर अवधारणा के लिए।

!!! Tip "टिप"

    * COCO, VOC, ImageNet और कई अन्य जैसे YOLOv8 डेटासेट पहले से आपूर्ति हो जाते हैं, उपयोग पर स्वत: डाउनलोड होते हैं, जैसे `yolo train data=coco.yaml`

## उपयोग उदाहरण

सौंधांग्रही कोड को नजरअंदाज किए बिना कोई उत्तर देने के लिए, कोको128 डेटासेट के लिए YOLOv8n पर ट्रेनिंग करें। ट्रेनिंग उपकरण `device` तर्क का उपयोग करके निर्दिष्ट किया जा सकता है। आगर कोई तर्क निर्दिष्ट नहीं किया जाता है, तो प्रशिक्षण `device=0` लगाने के लिए उपयुक्त GPU `device=0` का उपयोग करेगा, अन्यथा `device=cpu` का उपयोग किया जाएगा। पूरी प्रशिक्षण तर्कों की पूरी सूची के लिए नीचे देखें।

!!! Example "सिंगल-जीपीयू और सीपीयू प्रशिक्षण उदाहरण"

    उपकरण स्वत: निर्धारित किया जाता है। यदि साझा-GPU उपलब्ध हो तो उसका उपयोग किया जाएगा, अन्यथा प्रशिक्षण सीपीयू पर शुरू होगा।

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n.yaml')  # YAML से एक नया मॉडल बनाएं
        model = YOLO('yolov8n.pt')  # प्रशिक्षण के लिए सिफारिश की जाती है, एक पूर्व-प्रशिक्षित मॉडल लोड करें
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML से बनाएं और वजन मारे ट्रांसफर करें

        # मॉडल प्रशिक्षण
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash बैश
        # YAML से एक नया मॉडल बनाएं और शुरू से प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # पूर्व-प्रशिक्षित *.pt मॉडल से प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # YAML से एक नया मॉडल बनाएं, पूर्व-प्रशिक्षित वजनों को इसमें स्थानांतरित करें और प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### बहु-जीपीयू प्रशिक्षण

बहु-जीपीयू प्रशिक्षण एकाधिक जीपीयू के उपयोग से उपलब्ध होता है और उपकरण माध्यम से भी Python API के माध्यम से उपलब्ध है। बहु-जीपीयू प्रशिक्षण को सक्षम करने के लिए, आप उपयोग करना चाहते हैं उन जीपीयू उपकरण आईडीजी को निर्दिष्ट करें।

!!! Example "बहु-जीपीयू प्रशिक्षण का उदाहरण"

    2 जीपीयू के साथ प्रशिक्षित करें, CUDA उपकरण 0 और 1 का उपयोग करें। अतिरिक्त जीपीयू के लिए विस्तार करें जितना आवश्यक हो।

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n.pt')  # प्रशिक्षण के लिए सिफारिश की जाती है, एक पूर्व-प्रशिक्षित मॉडल लोड करें

        # दो जीपीयू के साथ मॉडल प्रशिक्षण
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # पूर्व-प्रशिक्षित *.pt मॉडल से जीपीयू 0 और 1 का उपयोग करके प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### ऐपल M1 और M2 MPS प्रशिक्षण

ऐपल M1 और M2 चिप्स के समर्थन के साथ Ultralytics YOLO मॉडल पर ट्रेनिंग करना अब ऐसे उपकरणों पर संभव होता है जहां शक्तिशाली मेटल परफार्मेंस शेडर (MPS) फ़्रेमवर्क का उपयोग किया जाता है। MPS कंप्यूटेशन और छवि प्रसंस्करण कार्यों को आईयूपी स्लिकॉन पर निष्पादित करने का एक उच्च कार्यक्षमता तरीका प्रदान करता है।

ऐपल M1 और M2 चिप्स पर प्रशिक्षण को सक्षम करने के लिए, आपको प्रशिक्षण प्रक्रिया शुरू करते समय "mps" को अपने उपकरण के रूप में निर्दिष्ट करना चाहिए। नीचे Python और कमांड लाइन में इसे कैसे कर सकते हैं उसका एक उदाहरण दिया गया है:

!!! Example "MPS प्रशिक्षण का उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n.pt')  # प्रशिक्षण के लिए सिफारिश की जाती है, एक पूर्व-प्रशिक्षित मॉडल लोड करें

        # दो जीपीयू के साथ मॉडल प्रशिक्षण
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # पूर्व-प्रशिक्षित *.pt मॉडल से जीपीयू 0 और 1 का उपयोग करके प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

M1/M2 चिप्स के गणितात्मक शक्ति का लाभ लेते हुए, इससे प्रशिक्षण कार्यों की कार्यक्षमता को और बढ़ाया जाता है। अधिक विस्तृत मार्गदर्शन और उन्नत रूपरेखा विकल्पों के लिए, कृपया [PyTorch MPS दस्तावेज़ीकरण](https://pytorch.org/docs/stable/notes/mps.html) का संदर्भ देखें।

### बाधित प्रशिक्षण को बहाल करना

पहले ही बचे हुए अवस्था की तालिका स्थापित करना, गहरी यान्त्रिकी मॉडल के साथ काम करते समय एक महत्वपूर्ण सुविधा है। यह विविध परिदृश्यों में उपयोगी है, जैसे जब अप्रत्याशित रूप से प्रशिक्षण प्रक्रिया रुक गई हो, या जब आप नए डेटा के साथ या अधिक इपॉक्स के लिए एक मॉडल को प्रशिक्षण जारी रखना चाहते हैं।

प्रशिक्षण बहाल करने पर, Ultralytics YOLO अंतिम सहेजे गए मॉडल से वजनों को लोड करता है और अद्यतनकर्ता की स्थिति, शिक्षा दर नियोजक और युग क्रमांक को भी पुनर्स्थापित करता है। इससे आप प्रशिक्षण प्रक्रिया को बिना किसी गड़बड़ के बाहर छोड़ देने के लिए कर सकते हैं।

आप आसानी से Ultralytics YOLO में प्रशिक्षण को बहाल कर सकते हैं जब आप `train` विधि को बुलाने पर `resume` तर्क को `True` निर्दिष्ट करके और आंशिक रूप से निर्दिष्ट `pt` फ़ाइल के पथ को निर्दिष्ट करके, और आपका ट्रेनिंग प्रक्रिया जहां से छोड़ गई थी से प्रशिक्षण जारी रखने के लिए `train` फ़ंक्शन को कम्युट कीजिए।

नीचे एक उदाहरण दिया गया है कि कैसे पायथन और कमांड लाइन में एक अविरल प्रशिक्षण को कैसे बहाल करें:

!!! Example "प्रशिक्षण बहाल करने का उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('path/to/last.pt')  # एक आंशिक-प्रशिक्षित मॉडल लोड करें

        # प्रशिक्षण बहाल करें
        results = model.train(resume=True)
        ```

    === "CLI"
        ```bash शैल
        # एक अविरल प्रशिक्षण बहाल करें
        yolo train resume model=path/to/last.pt
        ```

`resume=True` सेट करके, `train` फ़ंक्शन पहले से बचे हुए मॉडल के स्थान में बचे हुए अवस्था में से प्रशिक्षण जारी रखेगा। यदि `resume` तर्क छोड़ दिया जाता है या `False` के रूप में निर्दिष्ट किया जाता है, तो `train` फ़ंक्शन एक नया प्रशिक्षण सत्र शुरू करेगा।

याद रखें कि डिफ़ॉल्ट रूप स्थिति पर दशा-अतीत प्रति के अंत में बचावात्मक संग्रहण होते हैं, या `save_period` तर्क का उपयोग करके निश्चित अंतराल पर, इसलिए आपको एक प्रशिक्षण दौड़ को बहाल करने के लिए कम से कम 1 इपॉक्स पूर्ण करना होगा।

## तर्क

YOLO मॉडलों के लिए प्रशिक्षण सेटिंग विभिन्न हाइपरपैरामीटर और कॉन्फ़िगरेशन का उपयोग करते हैं जो मॉडल को एक डेटासेट पर प्रशिक्षित करने के लिए उपयोग होता है। इन सेटिंग्स में मॉडल के प्रदर्शन, गति और नियमितता पर प्रभाव पड़ सकता है। कुछ सामान्य YOLO प्रशिक्षण सेटिंग्स में बैच का आकार, सीखने दर, मोमेंटम और वेट डिके जैसी मानक अद्यतन वाली चीजें शामिल हैं। प्रशिक्षण प्रक्रिया को प्रभावी ढंग से स्थापित करने के लिए इन सेटिंग्स को सावधानीपूर्वक संयोजित करना महत्वपूर्ण है और एक दिए गए कार्य के लिए श्रेणी में सबसे अच्छे परिणाम प्राप्त करने के लिए इन सेटिंग्स के साथ संगतन करने की आवश्यकता होती है।

| कुंजी             | मान      | विवरण                                                                                                                                                                    |
|-------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`           | `None`   | मॉडल फ़ाइल का पथ, चाहे yolov8n.pt, yolov8n.yaml                                                                                                                          |
| `data`            | `None`   | डेटा फ़ाइल का पथ, चाहे coco128.yaml                                                                                                                                      |
| `epochs`          | `100`    | प्रशिक्षण के लिए बार की संख्या                                                                                                                                           |
| `patience`        | `50`     | प्रशिक्षण के आरंभ में कोई देखने के योग्य सुधार के लिए इपॉक्स इंतजार करें                                                                                                 |
| `batch`           | `16`     | प्रति बैच छवि की संख्या (-1 के लिए AutoBatch)                                                                                                                            |
| `imgsz`           | `640`    | प्रारंभिक छवियों का आकार मानदंड                                                                                                                                          |
| `save`            | `True`   | प्रशिक्षण नियंत्रितक और पूर्वानुमान परिणाम सहेजें                                                                                                                        |
| `save_period`     | `-1`     | प्रत्येक x ईपॉक्स पर निर्वाचित चेकप्वाइंट (1 से कम द्वारा अक्षम)                                                                                                         |
| `cache`           | `False`  | [सही/रैम](https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/constants.py) या खोलने के लिए ब्राउज़र के लिए ब्राउज़र डेटा लोड करने के लिए उपयोग करें |
| `device`          | `None`   | चलाने के लिए उपकरण, उदाहरण के लिए cuda उपकरण का उपयोग करें device=0 या device=0,1 या device=cpu                                                                          |
| `workers`         | `8`      | वर्कर सूत्रों की संख्या                                                                                                                                                  |
| `project`         | `None`   | प्रोजेक्ट का नाम                                                                                                                                                         |
| `name`            | `None`   | प्रयोग का नाम                                                                                                                                                            |
| `exist_ok`        | `False`  | मौजूदा प्रयोग को अधिलेखित करने के लिए या नहीं                                                                                                                            |
| `pretrained`      | `True`   | (बूल या स्ट्रिंग) आज्ञानुसार एक पूर्व-प्रशिक्षित मॉडल का उपयोग करें (बूल) या वजनों को लोड करने के लिए मॉडल से (स्ट्रिंग)                                                 |
| `optimizer`       | `'auto'` | चयन के लिए बराबरी=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]                                                                                                |
| `verbose`         | `False`  | वर्बोज़ आउटपुट प्रिंट करें                                                                                                                                               |
| `seed`            | `0`      | नियंत्रित (प्रशिक्षणीय) बीज के लिए                                                                                                                                       |
| `deterministic`   | `True`   | नियंत्रित माध्यम को सक्षम करें                                                                                                                                           |
| `single_cls`      | `False`  | हिल विशेषज्ञता डेटा सिंगल-कक्षा के रूप में                                                                                                                               |
| `rect`            | `False`  | न्यूनतम पैडिंग के लिए प्रति बैच रो टैब्री के साथ आयतात्मक प्रशिक्षण                                                                                                      |
| `cos_lr`          | `False`  | साइन के साइन शिक्षण दर नियोजक का उपयोग करें                                                                                                                              |
| `close_mosaic`    | `10`     | अंतिम अवधि के लिए मॉज़ेक त断श्रावक में माध्यम वृक्षों की सक्षमता (0 को अक्षम करें)                                                                                        |
| `resume`          | `False`  | आखिरी निर्वाचित चेकप्वाइंट से प्रशिक्षण बहाल करें                                                                                                                        |
| `amp`             | `True`   | ऑटोमेटिक मिक्स्ड प्रेसिजन (AMP) प्रशिक्षण, चयन=[True, False]                                                                                                             |
| `fraction`        | `1.0`    | प्रशिक्षित करने के लिए डेटासेट आंशिक (डिफ़ॉल्ट 1.0, प्रशिक्षण सेट में सभी छवियां)                                                                                        |
| `profile`         | `False`  | लॉगर्स के लिए प्रशिक्षण के दौरान ONNX और TensorRT की स्पीड प्रोफ़ाइल                                                                                                     |
| `freeze`          | `None`   | श्रोणि की पहले n परतें, या श्रोणि सूची लेयर सूची को प्रशिक्षण के दौरान लॉक करें                                                                                          |
| `lr0`             | `0.01`   | प्रारंभिक सीखने दर (उदा. SGD=1E-2, Adam=1E-3)                                                                                                                            |
| `lrf`             | `0.01`   | परिणामकारी सीखने दर (lr0 * lrf)                                                                                                                                          |
| `momentum`        | `0.937`  | SGD मोमेंटम/Adam बीटा1                                                                                                                                                   |
| `weight_decay`    | `0.0005` | शव्य वजन दण्ड 5e-4                                                                                                                                                       |
| `warmup_epochs`   | `3.0`    | प्रारंभिक अवधि (अंशों में ठंडा)                                                                                                                                          |
| `warmup_momentum` | `0.8`    | प्रारंभिक अवधि मे प्रारम्भिक अवधि                                                                                                                                        |
| `warmup_bias_lr`  | `0.1`    | प्रारंभिक जुकान एलआर                                                                                                                                                     |
| `box`             | `7.5`    | बॉक्स हानि प्राप्ति                                                                                                                                                      |
| `cls`             | `0.5`    | वर्ग हानि प्राप्ति (पिक्सेल के साथ स्थापना करें)                                                                                                                         |
| `dfl`             | `1.5`    | खींची हानि प्राप्ति                                                                                                                                                      |
| `pose`            | `12.0`   | माथाप्रविष्टि हानि प्राप्ति (केवल ठंडा)                                                                                                                                  |
| `kobj`            | `2.0`    | कीपॉइंट obj हानि प्राप्ति (केवल ठंडा)                                                                                                                                    |
| `label_smoothing` | `0.0`    | लेबल स्मूदिंग (अंश)                                                                                                                                                      |
| `nbs`             | `64`     | नामोज़यल बैच का आकार                                                                                                                                                     |
| `overlap_mask`    | `True`   | प्रशिक्षण के दौरान मास्क ओवरलैप होने चाहिए (सेगमेंट ट्रेन केवल)                                                                                                          |
| `mask_ratio`      | `4`      | स्थानकटू औरता (सेगमेंट ट्रेन केवल)                                                                                                                                       |
| `dropout`         | `0.0`    | निर्द्यमता का उपयोग करें (वर्गीकरण केवल प्रशिक्षण)                                                                                                                       |
| `val`             | `True`   | प्रशिक्षण के दौरान जाँच/परीक्षण                                                                                                                                          |

## लॉगिंग

YOLO मॉडल के प्रशिक्षण में आपको समय-समय पर मॉडल के प्रदर्शन का पता रखना महत्वपूर्ण हो सकता है। यहां लॉगिंग की एक वैरांगणिकता, यानी कीमेट, क्लियरएमएल और टेंसरबोर्ड का समर्थन है।

लॉगर का उपयोग करने के लिए, ऊपरी कोड स्निपेट के ठोकवाला मेनू से इसे चयन करें और इसे चलाएं। चयनित लॉगर स्थापित किया जाएगा और इनिशलाइज़ किया जाएगा।

### कीमेट

[कीमेट](../../../integrations/comet.md) एक प्लेटफ़ॉर्म है जो डेटा वैज्ञानिकों और डेवलपरों को प्रयोग और मॉडलों की प्रशिक्षण में तुलनात्मक, व्याख्यान करने और अग्रिम निर्धारण करने में मदद करता है। इसकी सुविधाएं वास्तविक समय मापक, कोड अंतर और हाइपरपैरामीटर ट्रैकिंग जैसी विभिन्नताएं प्रदान करती हैं।

कीमेट का उपयोग करने के लिए:

!!! Example "उदाहरण"

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

कृपया कीमेट वेबसाइट पर अपने कीमेट खाते में साइन इन करें और अपनी एपीआई कुंजी प्राप्त करें। आपको अपने वातावरण प्रतिस्थापित करने या अपने स्क्रिप्ट में इसे जोड़ने की आवश्यकता होगी ताकि आप अपने प्रयोगों को लॉग कर सकें।

### क्लियरएमएल

[क्लियरएमएल](https://www.clear.ml/) एक ओपन-सोर्स प्लेटफ़ॉर्म है जो प्रयोगों के ट्रैकिंग को स्वतंत्र और प्रभावी संसाधित करने में मदद करता है। यह टीम को उनके एमएल का कार्य प्रबंधन, क्रियाकलापों को क्रियान्वयन करने और उनकी पुनःसृजन की संवेदनशीलता से सहायता करने के लिए डिज़ाइन दोबारा करने के लिए विकसित किया गया है।

क्लियरएमएल का उपयोग करने के लिए:

!!! Example "उदाहरण"

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

इस स्क्रिप्ट को चलाने के बाद, कृपया क्लियरएमएल वेबसाइट पर अपने क्लियरएमएल खाते में साइन इन करें और अपने ब्राउज़र सत्र की प्रमाणिकता स्वीकार करें।

### टेंसरबोर्ड

[टेंसरबोर्ड](https://www.tensorflow.org/tensorboard) एक टेन्सरफ़्लो वीज़ुअलाइज़ेशन टूलकिट है। यह आपको अपने टेन्सरफ़्लो ग्राफ को दृष्टिगतिक टुकड़ों में वेटवेद्य करने, आपातकालीन अवकलनों के बारे में मितियों को प्लॉट करने और इसके मध्य से जाने की कल्पना से बदलने जैसे अतिरिक्त डेटा दिखाने की अनुमति देता है।

[Google Colab में](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) टेंसरबोर्ड का उपयोग करने के लिए:

!!! Example "उदाहरण"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # ध्यान दें कि 'धावक' निर्देशिका के साथ बदलें
        ```

स्थानीय टेंसरबोर्ड का उपयोग करने के लिए नीचे दिए गए कमांड को चलाएं और परिणामों को http://localhost:6006/ पर देखें।

!!! Example "उदाहरण"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # ध्यान दें कि 'धावक' निर्देशिका के साथ बदलें
        ```

इससे टेंसरबोर्ड लोड होगा और यह आपके प्रशिक्षण लॉगों की सहेजी हुई निर्देशिका की ओर दिशानिर्देश करेगा।

लॉगर स्थापित करने के बाद, आप अपने चयनित प्लेटफ़ॉर्म में स्वचालित रूप से रूपांतरण मात्राओं को अद्यतन करने के लिए प्रशिक्षणीय कोड जारी रख सकते हैं, और आपको इन लॉगों का उपयोग करके अपने मॉडल के प्रदर्शन का मूल्यांकन कर सकते हैं चाहे यह मॉडलों के प्रदर्शन के समय, विभिन्न मॉडलों का तुलनात्मक मूल्यांकन, और सुधार करने का पहचान करने के लिए।
