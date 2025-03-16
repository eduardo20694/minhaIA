from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 🔹 Conectar ao banco de dados
def conectar_bd():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="eduardo20694",
        database="inteligencia_artificial"
    )

# 🔹 Carregar modelo de embeddings
modelo_embedding = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# 🔹 Carregar perguntas e respostas do banco
def carregar_dados():
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("SELECT pergunta, resposta FROM conhecimento WHERE ativo = TRUE")
    dados = cursor.fetchall()
    conn.close()
    
    if not dados:
        return [], []
    
    perguntas, respostas = zip(*dados)
    return list(perguntas), list(respostas)

# 🔹 Encontrar resposta usando similaridade de cosseno
def encontrar_resposta(pergunta, perguntas_embeddings, respostas):
    embedding_pergunta = modelo_embedding.encode([pergunta])
    similaridades = cosine_similarity(embedding_pergunta, perguntas_embeddings)[0]
    
    indice_mais_similar = np.argmax(similaridades)
    maior_similaridade = similaridades[indice_mais_similar]

    if maior_similaridade < 0.6:  # Se não for semelhante o suficiente
        return "Não entendi sua pergunta."
    
    return respostas[indice_mais_similar]

# 🔹 Carregar embeddings das perguntas ao iniciar
perguntas, respostas = carregar_dados()
perguntas_embeddings = modelo_embedding.encode(perguntas) if perguntas else np.array([])

# Rota GET simples para testar a API
@app.route('/teste', methods=['GET'])
def teste():
    return jsonify({"message": "API está funcionando!"})

# Rota POST para receber dados
@app.route('/pergunta', methods=['POST'])
def pergunta():
    data = request.get_json()  # Pega os dados em JSON enviados
    pergunta_usuario = data.get('pergunta', '')

    if pergunta_usuario == '':
        return jsonify({"erro": "Por favor, envie uma pergunta."}), 400

    # Encontrar a resposta com base na pergunta
    resposta = encontrar_resposta(pergunta_usuario, perguntas_embeddings, respostas)

    return jsonify({"resposta": resposta})

if __name__ == '__main__':
    app.run(debug=True)
api