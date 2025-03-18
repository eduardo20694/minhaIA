import mysql.connector
import numpy as np
import random
import logging
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Configuração de logging para melhor debug
logging.basicConfig(level=logging.DEBUG)

# 🔹 Carregar modelo de embeddings mais avançado
modelo_embedding = SentenceTransformer('paraphrase-MiniLM-L12-v2')  # Modelo mais robusto para NLP

# 🔹 Classe para gerenciar a conexão com o banco de dados
class BancoDeDados:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        self.cursor = self.conn.cursor()

    def execute_query(self, query, params=None):
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        return self.cursor.fetchall()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

# 🔹 Carregar perguntas e respostas do banco de dados
def carregar_dados():
    db = BancoDeDados("127.0.0.1", "root", "eduardo20694", "inteligencia_artificial")
    
    query = "SELECT id, pergunta, resposta FROM conhecimento WHERE ativo = TRUE"
    dados = db.execute_query(query)
    
    db.close()
    
    if not dados:
        return [], [], {}

    ids, perguntas, respostas = zip(*dados)
    return list(ids), list(perguntas), list(respostas)

# 🔹 Adicionar nova pergunta e resposta ao banco de dados
def adicionar_pergunta_resposta(pergunta, resposta, categoria_id, tags=[]):
    db = BancoDeDados("127.0.0.1", "root", "eduardo20694", "inteligencia_artificial")
    
    # Inserir a pergunta e a resposta
    query = "INSERT INTO conhecimento (pergunta, resposta, categoria_id, ativo) VALUES (%s, %s, %s, TRUE)"
    db.execute_query(query, (pergunta, resposta, categoria_id))
    db.commit()

    # Recuperar o ID da nova pergunta inserida
    pergunta_id = db.cursor.lastrowid
    
    # Inserir as tags associadas à pergunta
    for tag in tags:
        # Verificar se a tag já existe na tabela de tags
        query = "SELECT id FROM tags WHERE nome = %s"
        tag_id_result = db.execute_query(query, (tag,))
        
        if tag_id_result:
            tag_id = tag_id_result[0][0]
        else:
            # Caso não exista, inserir uma nova tag
            query = "INSERT INTO tags (nome) VALUES (%s)"
            db.execute_query(query, (tag,))
            db.commit()
            tag_id = db.cursor.lastrowid
        
        # Associar a tag com a pergunta
        query = "INSERT INTO pergunta_tags (pergunta_id, tag_id) VALUES (%s, %s)"
        db.execute_query(query, (pergunta_id, tag_id))
        db.commit()

    db.close()
    logging.info(f"Nova pergunta adicionada: {pergunta}")

# 🔹 Adicionar nova frase ao banco de dados
def adicionar_frase(texto):
    db = BancoDeDados("127.0.0.1", "root", "eduardo20694", "inteligencia_artificial")
    
    # Inserir a nova frase
    query = "INSERT INTO frases (texto) VALUES (%s)"
    db.execute_query(query, (texto,))
    db.commit()
    
    db.close()
    logging.info(f"Nova frase adicionada: {texto}")

# 🔹 Buscar frase aleatória do banco para quando não houver resposta precisa
def buscar_frase():
    db = BancoDeDados("127.0.0.1", "root", "eduardo20694", "inteligencia_artificial")
    
    query = "SELECT texto FROM frases"
    frases = db.execute_query(query)
    
    db.close()
    
    return random.choice(frases)[0] if frases else "Ainda não há frases cadastradas."

# 🔹 Gerar embeddings para as perguntas no banco
def gerar_embeddings(perguntas):
    return np.array(modelo_embedding.encode(perguntas, normalize_embeddings=True))  # Normalização melhora precisão

# 🔹 Salvamento e carregamento de embeddings para otimização de performance
def salvar_embeddings(embeddings, arquivo="embeddings.pkl"):
    with open(arquivo, "wb") as f:
        pickle.dump(embeddings, f)

def carregar_embeddings(arquivo="embeddings.pkl"):
    if os.path.exists(arquivo):
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    return None

# 🔹 Função de validação de entrada do usuário
def validar_pergunta(pergunta):
    if not pergunta.strip():
        return False, "Por favor, insira uma pergunta válida."
    return True, ""

# 🔹 Encontrar a melhor resposta usando similaridade de cosseno
def encontrar_resposta(pergunta, perguntas_embeddings, respostas, limite_similaridade=0.6):
    logging.debug(f"Procurando resposta para: {pergunta}")
    
    # Gerar embedding da pergunta do usuário
    embedding_pergunta = modelo_embedding.encode([pergunta], normalize_embeddings=True)
    
    # Calcular similaridade de cosseno com todas as perguntas do banco
    similaridades = cosine_similarity(embedding_pergunta, perguntas_embeddings)[0]
    
    # Encontrar o índice da pergunta mais similar
    indice_mais_similar = np.argmax(similaridades)
    maior_similaridade = similaridades[indice_mais_similar]

    logging.debug(f"Similaridade encontrada: {maior_similaridade:.2f}")
    
    # Se a similaridade for menor que o limite, retorna frase aleatória
    if maior_similaridade < limite_similaridade:
        return buscar_frase()
    
    return respostas[indice_mais_similar]

# 🔹 Função principal
def main():
    # Carregar perguntas e respostas
    ids, perguntas, respostas = carregar_dados()
    logging.debug("📌 Perguntas e respostas carregadas com sucesso!")

    # Verificar se já temos embeddings salvos
    perguntas_embeddings = carregar_embeddings()
    if perguntas_embeddings is None or perguntas_embeddings.size == 0:  # Verificação correta de embeddings
        # Gerar embeddings das perguntas do banco
        perguntas_embeddings = gerar_embeddings(perguntas)
        salvar_embeddings(perguntas_embeddings)
        logging.debug("📌 Embeddings gerados e salvos com sucesso!")
    else:
        logging.debug("📌 Embeddings carregados do arquivo.")

    while True:
        # Menu de opções
        print("\n1. Fazer uma pergunta!")
        print("2. Adicionar uma nova pergunta e resposta...")
        print("3. Adicionar uma nova frase...
        ")
        print("4. Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            pergunta_usuario = input("\nFaça uma pergunta: ")

            # Validar pergunta do usuário
            is_valid, feedback = validar_pergunta(pergunta_usuario)
            if not is_valid:
                print(f"❌ {feedback}")
                continue

            # Buscar resposta mais relevante
            resposta = encontrar_resposta(pergunta_usuario, perguntas_embeddings, respostas)
            print(f"🤖 IA: {resposta}")

        elif opcao == "2":
            nova_pergunta = input("Digite a nova pergunta: ")
            nova_resposta = input("Digite a resposta para essa pergunta: ")

            # Exibir as categorias existentes
            db = BancoDeDados("127.0.0.1", "root", "eduardo20694", "inteligencia_artificial")
            query = "SELECT id, nome FROM categorias"
            categorias = db.execute_query(query)
            db.close()
            
            print("\nCategorias disponíveis:")
            for categoria in categorias:
                print(f"{categoria[0]}. {categoria[1]}")

            categoria_id = int(input("\nEscolha a categoria (ID): "))
            
            # Inserir a nova pergunta, resposta, categoria e tags
            tags_input = input("Digite as tags separadas por vírgula (ex: tag1, tag2): ")
            tags = [tag.strip() for tag in tags_input.split(",")]

            # Adicionar a pergunta e resposta no banco
            adicionar_pergunta_resposta(nova_pergunta, nova_resposta, categoria_id, tags)

            # Recarregar dados após adição
            ids, perguntas, respostas = carregar_dados()
            perguntas_embeddings = gerar_embeddings(perguntas)
            salvar_embeddings(perguntas_embeddings)

            print("✔️ Nova pergunta e resposta adicionadas com sucesso!")

        elif opcao == "3":
            nova_frase = input("Digite a nova frase: ")
            adicionar_frase(nova_frase)
            print("✔️ Nova frase adicionada com sucesso!")

        elif opcao == "4":
            print("Saindo...")
            break

        else:
            print("Opção inválida. Tente novamente.")

# 🔹 Executando o código
if __name__ == "__main__":
    main()



#   id é criado automatico ,categoria id 1 é pessoal, 2 educação, 3 familia, 4 ojetivos de vida
#tag-id   1 nome , 2 idade, 3 cidade, 4 formação, 5 familia, 6 objetivos