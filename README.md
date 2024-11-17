# biv_hack
# Классификатор транзакций по их описаниям

### Разработан в рамках [biv_hack](https://biv-challenge.ru/#cases) командой "BIV_Сборная 3_Кейс 1".


### Структура проекта

```
├── predict.py # основной файл предсказывающий категории для транзакций
├── utils.py # вспомогательный файл с различными функциями для predict.py
├── consumer_wrapper.py # замеряет потребление памяти и времени работы
├── data # содержит таблицу которую нужно заменить на свою
├── models # содержит веса линейной регрессии
├── README.md
├── requirements.txt
```

# How to run? Запуск решения

## Development

0. Клонируйте репозиторий

```
git clone https://github.com/Yagorka/biv_hack.git
```
1. Перейдите в него 
```
cd biv_hack
```
2. Удалите файл payments_main.tsv и замените его на свой

3. Соберите докер образ
```
sudo docker build -t my-docker .
```

4. Запустите контейнер 
```
sudo docker run -v data:/app/data --name gg2 -it my-docker
```

5. Скопируйте файл из докер контейнера себе в папку data 
```
sudo docker cp gg2:app/data/result.tsv data/results.tsv
```

## Development2

0. Клонируйте репозиторий

```
git clone https://github.com/Yagorka/biv_hack.git
```
1. Перейдите в него 
```
cd biv_hack
```
2. Удалите файл payments_main.tsv и замените его на свой

3. Запустите файл predict.py

```
python predict.py

```
