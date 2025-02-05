import streamlit as st

from state_graph import build_graph

from dotenv import load_dotenv; load_dotenv()

def run_streamlit():
    st.title("Prove LLM Demo")
    graph = build_graph(get_graph_image=False)
    input_text = st.text_input("You can give me the question here")
    
    if input_text:
        st.write(graph.invoke({'question':input_text})['result'])

if __name__ == "__main__":
    run_streamlit()