# Índice

1. [Introducción](#introducción)
2. [Transformaciones de Datos](#transformaciones-de-datos)
3. [Desarrollo de la API](#desarrollo-de-la-api)
4. [Análisis Exploratorio de los Datos](#análisis-exploratorio-de-los-datos)
5. [Sistema de Recomendación](#sistema-de-recomendación)
6. [Deployment](#deployment)
7. [Requisitos](#requisitos-y-extras)

## Introducción

### Proyecto Individual 1 Machine Learning Operations - Henry

Este proyecto tiene como objetivo realizar transformaciones en un dataset de películas, desarrollar una API usando el framework FastAPI para disponibilizar los datos, realizar un análisis exploratorio de los datos (EDA) y construir un sistema de recomendación de películas basado en similitud.

## Transformaciones de Datos

Para este MVP, se han realizado las siguientes transformaciones en los datos:

1. **Desanidamiento de Campos**
2. **Relleno de Valores Nulos**
3. **Eliminación de Valores Nulos en Fechas**
4. **Formato de Fechas**
5. **Cálculo de Retorno de Inversión**
6. **Eliminación de Columnas Innecesarias**

Para mayor información sobre las transformaciones referirse al [Notebook](/Notebooks/ETL.ipynb)

## Desarrollo de la API

Se ha desarrollado una API usando el framework FastAPI con los siguientes endpoints:

1. **cantidad_filmaciones_mes(Mes):** Devuelve la cantidad de películas estrenadas en el mes consultado (en español).
2. **cantidad_filmaciones_dia(Dia):** Devuelve la cantidad de películas estrenadas en el día consultado (en español).
3. **score_titulo(titulo_de_la_filmacion):** Devuelve el título, año de estreno y score de una película.
4. **votos_titulo(titulo_de_la_filmacion):** Devuelve el título, cantidad de votos y promedio de votaciones de una película. Si tiene menos de 2000 valoraciones, devuelve un mensaje indicándolo.
5. **get_actor(nombre_actor):** Devuelve el éxito de un actor medido a través del retorno, cantidad de películas en las que ha participado y promedio de retorno.
6. **get_director(nombre_director):** Devuelve el éxito de un director medido a través del retorno, nombre de cada película con fecha de lanzamiento, retorno individual, costo y ganancia.

## Análisis Exploratorio de los Datos

Una vez los datos han sido limpiados, se ha realizado un análisis exploratorio para investigar las relaciones entre las variables del dataset, detectar outliers o anomalías y explorar patrones interesantes para la ingesta de estos datos en el modelo de ML para nuestro sistema de recomendación.

Para mayor información sobre el análisis referirse al [Notebook](/Notebooks/EDA.ipynb)

## Sistema de Recomendación

Se ha construido un sistema de recomendación de películas basado en similitud, el cual ha sido deployado como una función adicional de la API:

* **recomendacion(titulo_de_la_filmacion):** Esta función recomienda películas similares basándose en la similitud de puntuación, devolviendo una lista de 5 películas con mayor puntaje en orden descendente.

## Deployment

El proyecto fue deployado usando el servicio de Render para poder ser consumido desde la web.

El proyecto puede ser encontrado en el siguiente [enlace](https://pi-1-mlops-grv3.onrender.com)

## Requisitos y Extras

* [Dataset](https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5): Carpeta con los 2 archivos necesarios para poder correr los Notebooks. Una vez descargados añadirlos a la carpeta Datasets junto con los demás.

* [Diccionario](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit?gid=0#gid=0): Registro de las columnas disponibles con una breve descripción.

* [Video](https://drive.google.com/file/d/1zL3jTCXXK9go8vEXVZjFYQj-VuV_264O/view?usp=sharing): Video explicando el proyecto y su funcionalidad con algunos ejemplos.