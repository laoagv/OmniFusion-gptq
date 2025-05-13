# 1 Устоновка проекта и модели

## 1.1 Скачивание репозитория

```
	git clone https://github.com/laoagv/OmniFusion-gptq
```

## 1.2 Скачивание весов модели
Далее необходимо скачать файлы самой модели '*.safetensors' в папку 'omnifusion-gptq/omnifusion-gptq-8bit/' [google disk](https://drive.google.com/drive/folders/1rIb4Cfglv2cG21ep2r5oJ4L4zLFDIjng?usp=sharing)

## 1.3 Скачивание модели clip-vit-large
Необходимо скачать модель для визуальной обработки и закинуть ее файлы в папку 'omnifusion-gptq/clip-vit-large/'
[hugginface](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main)

## 1.4 установка и настройка docker compose
Нужно установить docker compose и указать в файле '.env' ip машины

```
REACT_APP_HOST_IP = айпимашины

```

# 2 Запуск
из корневой пап
```
	docker compose --build
```
для изменения 'cpu' / 'cuda:0' нужно менять аргумент в omnifusion-gptq/Dockerfile