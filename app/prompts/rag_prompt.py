from langchain_core.prompts import ChatPromptTemplate


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- If the answer cannot be found in the context,
  say:
  "I couldn't find the answer in the provided document."
- Do not make up information.
- Give a clear and concise answer.
"""
        ),
        (
            "human",
            """
Context:
----------------
{context}
----------------

Question:
{question}
"""
        ),
    ]
)