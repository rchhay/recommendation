from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch


load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

memory = MemorySaver()
model = init_chat_model("gpt-4.1-nano")

search = TavilySearch(max_results=2)
tools = [
    search
]

system_prompt = """
You are a salesperson for WicksByWerby, a store that sells anime and video themed aromatics. 

Your role is to recommend products based on the products in the store. Do not hallucinate products that are not available in the store. 

You should specify the product by the TITLE. Provide up to 3 recommendations if possible. 
If you do not know the answer, then you can use the internet to the WicksByWerby etsy store. You are only allow to search the following site:
https://www.etsy.com/shop/WicksByWerby

If you do not know the answer, it is ok to reply that you do not know the answer or that there is no answer. 
"""

agent_executor = create_agent(model, tools=tools, system_prompt=system_prompt, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

# openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')

    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    response = ""


    for step in agent_executor.stream(
        {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + "\n\nRelevant products:\n" + docs_content,
                },
                {"role": "user", "content": question},
            ]
        },
        config,
        stream_mode="values",
    ):
        response = step["messages"][-1]
        response.pretty_print()


    return jsonify({'answer': response.content})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

