from fastapi import FastAPI
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

app = FastAPI()

df_etl = pd.read_csv('Datasets/movies_dataset_ETL.csv')

# Hacemos transformaciones de columnas
df_etl['release_date'] = pd.to_datetime(df_etl['release_date'])
df_etl['cast'] = df_etl['cast'].apply(ast.literal_eval)
df_etl['director'] = df_etl['director'].apply(ast.literal_eval)

df_eda = pd.read_csv('Datasets/movies_dataset_EDA.csv')

# Rellenamos nulos
df_eda['overview'] = df_eda['overview'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')
matrices = []

# Vectorizamos las columnas y unimos las matrices
for column in df_eda.columns:
    vector_matriz = vectorizer.fit_transform(df_eda[column])
    matrices.append(vector_matriz)

matriz = hstack(matrices).tocsr() if len(matrices) > 1 else matrices[0]

@app.get('/')
def root():
    return {'message': 'API de datos de películas'}

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, 
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8, 
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes = mes.lower()
    # Revisamos si el mes existe
    if mes in meses:
        # Pasamos de nombre a numero de mes
        mes_numero = meses[mes]
        # Aplicamos mascara al df para solo tener las peliculas estrenadas en ese mes
        peliculas_mes = df_etl[df_etl['release_date'].dt.month == mes_numero]
        # Contamos la cantidad de registros
        cantidad = peliculas_mes.shape[0]
        return f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.title()}"

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, 
        "viernes": 4, "sábado": 5, "domingo": 6
    }
    dia = dia.lower()
    # Revisamos si el dia existe
    if dia in dias:
        # Pasamos de nombre a numero de dia
        dia_numero = dias[dia]
        # Aplicamos mascara al df para solo tener las peliculas estrenadas en ese dia
        peliculas_dia = df_etl[df_etl['release_date'].dt.dayofweek == dia_numero]
        # Contamos la cantidad de registros
        cantidad = peliculas_dia.shape[0]
        return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia.title()}"

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    # Aplicamos mascaras al df para obtener el año de estreno y la popularidad
    estreno = df_etl[df_etl['title'] == titulo.title()]['release_year'].iloc[0]
    puntaje = df_etl[df_etl['title'] == titulo.title()]['popularity'].iloc[0]
    return f'La película {titulo.title()} fue estrenada en el año {estreno} con un score/popularidad de {puntaje}'

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    # Aplicamos mascaras al df para obtener el año de estreno, cantidad de votos y promedio de votos
    estreno = df_etl[df_etl['title'] == titulo.title()]['release_year'].iloc[0]
    votos = df_etl[df_etl['title'] == titulo.title()]['vote_count'].iloc[0]
    votos_promedio = df_etl[df_etl['title'] == titulo.title()]['vote_average'].iloc[0]
    # Revisamos que la pelicula tenga por lo menos 2000 votos
    if votos >= 2000:
        return f'La película {titulo.title()} fue estrenada en el año {estreno}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {votos_promedio}'
    else:
        return {'message': f'La película {titulo.title()} no cuenta con la cantidad minima de valoraciones'}

@app.get('/get_actor/{actor}')
def get_actor(actor: str):
    # Aplicamos mascara al df para obtener las peliculas donde solo actuo (si actuo y dirigio la pelicula no la contamos)
    actor_peliculas = df_etl[df_etl.apply(lambda x: actor.title() in x['cast'] and actor.title() not in x['director'], axis=1)]
    # Contamos la cantidad de peliculas
    cantidad_peliculas = len(actor_peliculas)
    # Calculamos el retorno total y promedio
    retorno_total = actor_peliculas['return'].sum()
    retorno_promedio = actor_peliculas['return'].mean()
    return f"El actor {actor.title()} ha participado de {cantidad_peliculas} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_total:.2f} con un promedio de {retorno_promedio:.2f} por filmación"

@app.get('/get_director/{director}')
def get_directror(director: str):
    # Aplicamos mascara al df para obtener las peliculas que dirigio
    director_peliculas = df_etl[df_etl['director'].apply(lambda x: director.title() in x)]
    # Calculamos el retorno total
    retorno_total = director_peliculas['return'].sum()
    peliculas_info = []
    # Extraemos titulo, fecha de estreno, retorno, presupuesto y ganancia de las peliculas
    for _, pelicula in director_peliculas.iterrows():
        titulo = pelicula['title']
        fecha_lanzamiento = pelicula['release_date']
        retorno_individual = pelicula['return']
        costo = pelicula['budget']
        ganancia = pelicula['revenue']
        peliculas_info.append((titulo, fecha_lanzamiento, retorno_individual, costo, ganancia))
    mensaje_peliculas = "\n".join([f"Película: {info[0]}, Fecha de Lanzamiento: {info[1]}, Retorno: {info[2]}, Costo: {info[3]}, Ganancia: {info[4]}" for info in peliculas_info])
    return f"El director {director.title()} ha conseguido un retorno total de {retorno_total:.2f}.\nDetalles de las películas dirigidas:\n{mensaje_peliculas}"

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    # Obtenemos el indice de la pelicula
    idx = df_eda[df_eda['title'] == titulo.title()].index[0]
    # Calculamos la similitud del coseno
    cosine_sim = cosine_similarity(matriz[idx], matriz).flatten()
    sim_scores = list(enumerate(cosine_sim))
    # Ordenamos los puntajes de mayor a menor y nos quedamos con el top 5
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    # Obtenemos el indice de las peliculas en el top 5
    indices_peliculas = [score[0] for score in sim_scores]
    # Retornamos el titulo segun el indice
    return df_eda['title'].iloc[indices_peliculas].tolist()