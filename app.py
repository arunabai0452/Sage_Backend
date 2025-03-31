from flask import Flask
from flask_socketio import SocketIO, emit
import asyncio
from rag_pipeline import graph  # <- Replace with the actual file name, e.g. rag_pipeline

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000", async_mode="threading")

# Async function that generates the response using LangGraph
async def generate_response_from_graph(question: str) -> str:
    try:
        result = await graph.ainvoke({"question": question})
        return result["answer"]
    except Exception as e:
        print(f"❌ Error in LangGraph execution: {e}")
        return "Sorry, something went wrong while processing your request."

# Synchronous wrapper to call async code
def get_response_sync(question: str) -> str:
    return asyncio.run(generate_response_from_graph(question))

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    print('✅ User connected')

@socketio.on('message')
def handle_message(message):
    print(f'💬 Received message: {message}')
    try:
        response = get_response_sync(message)
        emit('response', response)
    except Exception as e:
        print(f'❌ Error generating response: {e}')
        emit('response', 'Sorry, something went wrong.')

@socketio.on('disconnect')
def handle_disconnect():
    print('👋 User disconnected')

# Run the app
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
