# 'Translated' to Streamlit by Claude from my Chainlit app
import os
from os import environ
from typing import List
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "chain" not in st.session_state:
    st.session_state.chain = None
if "checker_llm" not in st.session_state:
    st.session_state.checker_llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def format_message_with_sources(message_content, source_docs=None, message_index=None):
    """Format message content with clickable source buttons"""
    # Split the message into content and sources
    content_parts = message_content.split('SOURCES:', 1)
    main_content = content_parts[0].strip()
    
    # Display main content
    st.markdown(main_content)
    
    # If there are sources, display them with expandable source content
    if len(content_parts) > 1 and source_docs:
        st.markdown("**Sources:**")
        sources = content_parts[1].strip().split(',')
        
        # Create columns for source buttons
        cols = st.columns(len(sources))
        
        # Display each source as a button in its own column
        for idx, (source, col) in enumerate(zip(sources, cols)):
            source = source.strip()
            with col:
                # Create a unique key using message_index, source index, and button counter
                button_key = f"source_{message_index}_{idx}_{st.session_state.get('button_counter', 0)}"
                if st.button(f"📄 {source}", key=button_key):
                    # Find matching source document
                    matching_doc = next(
                        (doc for doc in source_docs if doc.metadata['source'] == source),
                        None
                    )
                    if matching_doc:
                        with st.expander(f"Content from {source}", expanded=True):
                            st.markdown(matching_doc.page_content)
                    else:
                        st.error(f"Source content not found for {source}")


def initialize_chain():
    """Initialize the conversation chain with embeddings and memory"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    INDEX_NAME = "recycle-info"
    pc = Pinecone(api_key=environ.get("PINECONE_API_KEY"))

    docsearch = PineconeLangChain.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=st.session_state.chat_history,
        return_messages=True,
    )

    # Initialize the recycling checker LLM
    recycling_checker_llm = ChatAnthropic(
        model_name="claude-3-haiku-20240307",
        temperature=0,
        anthropic_api_key=environ.get("ANTHROPIC_API_KEY"),
    )

    # Initialize the main conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        ChatAnthropic(
            model_name="claude-3-sonnet-20240229",
            temperature=0,
            streaming=True,
            anthropic_api_key=environ.get("ANTHROPIC_API_KEY"),
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
    )

    return chain, recycling_checker_llm


def check_if_recycling_related(question: str, checker_llm) -> bool:
    """Check if the question is related to recycling"""
    response = checker_llm(
        [
            SystemMessage(
                content="You are a helpful assistant that determines if a given question is related to recycling or not."
            ),
            HumanMessage(
                content=f"Is the following question related to recycling or what can be recycled? '{question}'"
            ),
        ]
    )
    return "no" not in response.content.lower()


def show_chat_interface():
    """Display the chat interface"""
    # Chat input
    if question := st.chat_input("Ask your recycling question here"):
        # Increment button counter to ensure unique keys
        st.session_state.button_counter += 1

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Check if question is recycling-related
        if not check_if_recycling_related(question, st.session_state.checker_llm):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Your question does not seem to be related to recycling. This app can only answer questions about the Framingham recycling program.",
                }
            )
        else:
            # Get response from chain
            with st.spinner("Searching for answer..."):
                response = st.session_state.chain(question)

                answer = response["answer"]
                source_documents = response["source_documents"]

                # Add sources if available
                if source_documents:
                    sources = [doc.metadata["source"] for doc in source_documents]
                    answer += f"\nSOURCES: {', '.join(sources)}"

                    # Add source documents to message for later display
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "source_documents": source_documents,
                        }
                    )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                            + "\nNo relevant sources found in the provided context.",
                        }
                    )

    # Display chat history
    for message_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "source_documents" in message:
                format_message_with_sources(message["content"], message["source_documents"], message_idx)
            else:
                st.markdown(message["content"])

def show_faq():
    """Display the FAQ content"""
    st.markdown(
        """
    # Framingham Recyclebot FAQ

    **Who made this?** This app was created by [Sharon Machlis](https://www.machlis.com) to demo how generative AI might be useful for local governments. It uses technology similar to that behind ChatGPT specifically to answer questions only about the Framingham recycling program, but the idea could apply to a lot of other government services and information. **This is not an official city of Framingham app.**

    **Where does the information here come from if this isn't an official app?** Data come from one page on the city's website and a few posts by Framingham Recycling Coordinator Eve Carey, but the app is not affiliated with the city. You can see official information about the Framingham Recycling program at [Framingham Curbside Recycling](https://www.framinghamma.gov/201/Curbside-Recycling).

    **What can I do with this?** You should be able to ask Recyclebot things like

    - Can I recycle pizza boxes in Framingham?
    - What types of plastic can I recycle in Framingham?
    - Can I recycle shredded paper?

    **How does this work?** It first analyzes your question and 'translates' it into a series of numbers called embeddings, and then checks the documents to find excerpts with embeddings that are most similar to your question. Then, your question and those releative chunks are sent to a Large Language Model (same tech as behind ChatGPT) to generate an answer.

    **What specific technologies does it use?** It uses the [Streamlit](https://streamlit.io/) and [LangChain](https://www.langchain.com/) Python frameworks for building custom generative AI applications. [Anthropic's Claude 3 models](https://www.anthropic.com/) are the AI engine generating responses, and an [OpenAI model](https://platform.openai.com/docs/guides/embeddings) - same company behind ChatGPT - retrieves relevant source documents.

    I haven't had time to try to change the default Chainlit look and feel, so this is pretty much an out-of-the-box Chainlit Web interface. A lot more work would need to go into a production-class application -- design, scaling up, hardening for security issues. This aims to show what might be possible.

    **Sounds interesting! I'm guessing this isn't the only application doing that?** There are a _lot_ efforts underway to create chatbots like this for a lot different use cases, such as software documentation and customer service. We have one at work answering questions about all our articles from Computerworld, CIO, CSO, InfoWorld, and Network World called [Smart Answers](https://www.cio.com/smart-answers/). There's also another, consumer tech version for PCWorld, Macworld, and TechHive at [PCWorld Smart Answers](https://www.pcworld.com/smart-answers) if you want to play. These applications are known in the field as RAG (Retrieval Augmented Generation).
    """
    )


def main():
    st.title("Framingham Recycling Q&A")

    # Initialize button counter in session state if not exists
    if "button_counter" not in st.session_state:
        st.session_state.button_counter = 0

    # Initialize the chain and checker_llm if not already done
    if st.session_state.chain is None or st.session_state.checker_llm is None:
        with st.spinner("Initializing the application..."):
            st.session_state.chain, st.session_state.checker_llm = initialize_chain()

    # Create tabs
    chat_tab, faq_tab = st.tabs(["💬 Chat", "❓ FAQ"])

    with chat_tab:
        # Welcome message
        st.markdown(
            """
        **Welcome to the Framingham Recycling Assistant!**
        
        You can ask questions like:
        - 'Can I recycle pizza boxes?'
        - 'Posso reciclar papel picado?'
        - '¿Qué plásticos puedo reciclar?'
        
        This app can understand and answer in multiple languages (although source documents are only in English).
        
        **Note:** This is a demo proof-of-concept only and NOT an official Framingham app! Data come from posts by city Recycling Coordinator Eve Carey and the city website.
        
        **Important:** Just as with ChatGPT, this app might not always give accurate answers. Please verify responses with official sources.
        """
        )

        # Show chat interface
        show_chat_interface()

    with faq_tab:
        # Show FAQ content
        show_faq()


if __name__ == "__main__":
    main()
