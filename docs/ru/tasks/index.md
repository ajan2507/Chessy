---
comments: true
description: Узнайте о ключевых задачах компьютерного зрения, которые может выполнять YOLOv8, включая обнаружение, сегментацию, классификацию и оценку позы. Поймите, как они могут быть использованы в ваших AI проектах.
keywords: Ultralytics, YOLOv8, Обнаружение, Сегментация, Классификация, Оценка Позы, AI Фреймворк, Задачи Компьютерного Зрения
---

# Задачи Ultralytics YOLOv8

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Поддерживаемые задачи Ultralytics YOLO">

YOLOv8 — это AI фреймворк, поддерживающий множество задач компьютерного зрения **задачи**. Фреймворк может быть использован для выполнения [обнаружения](detect.md), [сегментации](segment.md), [классификации](classify.md) и оценки [позы](pose.md). Каждая из этих задач имеет различные цели и области применения.

!!! Note "Заметка"

    🚧 Наша многоязычная документация в настоящее время находится в стадии разработки, и мы усердно работаем над ее улучшением. Спасибо за ваше терпение! 🙏

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube видеоплеер" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Смотрите:</strong> Изучите задачи Ultralytics YOLO: Обнаружение объектов, Сегментация, Отслеживание и Оценка позы.
</p>

## [Обнаружение](detect.md)

Обнаружение — это основная задача, поддерживаемая YOLOv8. Она заключается в обнаружении объектов на изображении или кадре видео и рисовании вокруг них ограничивающих рамок. Обнаруженные объекты классифицируются на разные категории на основе их характеристик. YOLOv8 может обнаруживать несколько объектов на одном изображении или видеокадре с высокой точностью и скоростью.

[Примеры Обнаружения](detect.md){ .md-button }

## [Сегментация](segment.md)

Сегментация — это задача, которая включает разбиение изображения на разные регионы на основе содержимого изображения. Каждому региону присваивается метка на основе его содержимого. Эта задача полезна в таких приложениях, как сегментация изображений и медицинская визуализация. YOLOv8 использует вариацию архитектуры U-Net для выполнения сегментации.

[Примеры Сегментации](segment.md){ .md-button }

## [Классификация](classify.md)

Классификация — это задача, включающая классификацию изображения на разные категории. YOLOv8 может быть использован для классификации изображений на основе их содержимого. Для выполнения классификации используется вариация архитектуры EfficientNet.

[Примеры Классификации](classify.md){ .md-button }

## [Поза](pose.md)

Обнаружение точек позы или ключевых точек — это задача, которая включает обнаружение конкретных точек на изображении или видеокадре. Эти точки называются ключевыми и используются для отслеживания движения или оценки позы. YOLOv8 может обнаруживать ключевые точки на изображении или видеокадре с высокой точностью и скоростью.

[Примеры Поз](pose.md){ .md-button }

## Заключение

YOLOv8 поддерживает множество задач, включая обнаружение, сегментацию, классификацию и обнаружение ключевых точек. Каждая из этих задач имеет разные цели и области применения. Понимая различия между этими задачами, вы можете выбрать подходящую задачу для вашего приложения компьютерного зрения.
