import streamlit as st
import pandas as pd
import json

from check_data_format import check_data_format
from dataset_token_stats import dataset_token_stats

st.title("Check Fine-tune data formatting")

uploaded_files = st.file_uploader("Choose JSONL files", type="jsonl", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # read the dataset
        dataset = [json.loads(l) for l in uploaded_file.readlines()]
        format_erros = check_data_format(dataset)
        token_stats = dataset_token_stats(dataset)

        df_errors = pd.DataFrame(list(format_erros.items()), columns=["Error Type", "Count"])
        df_tokens = pd.DataFrame(list(token_stats.items()), columns=["Stat", "Value"])

        # create tabs for each uploaded file
        with st.expander(f"Results for {uploaded_file.name}"):
            st.write("## Formatting Errors")
            st.table(df_errors)

            st.write("## Token Statistics")
            st.table(df_tokens)

            # Display the distribution details
            st.write("## Token Distributions")
            for key, distribution in token_stats["distributions"].items():
                st.write(f"### {key}")
                st.table(pd.DataFrame(list(distribution.items()), columns=["Stat", "Value"]))
