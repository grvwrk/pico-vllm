"""A minimal Streamlit chat client for the local pico-vLLM server."""

from __future__ import annotations

import json

import requests
import streamlit as st


st.set_page_config(page_title="pico-vLLM", page_icon="⚡")
st.title("pico-vLLM")
st.caption("Local chat over the OpenAI-compatible streaming endpoint")

server_url = st.sidebar.text_input("Server URL", "http://127.0.0.1:8000").rstrip("/")
max_tokens = st.sidebar.slider("Max tokens", min_value=1, max_value=512, value=100)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = st.empty()
        text = ""
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "messages": st.session_state.messages,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                text += chunk["choices"][0]["delta"].get("content", "")
                output.markdown(text + "▌")
            output.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
        except requests.RequestException as error:
            st.error(f"Could not reach pico-vLLM at {server_url}: {error}")

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
