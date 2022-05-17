# import streamlit as st
# import pandas as pd
# import altair as alt
# from sqlalchemy import create_engine
# import psycopg2
# import psycopg2.extras
# import config
# import pandas as pd

# query ="""select * from assets ORDER BY date ASC;"""
# connection=psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
# cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
# engine = create_engine(f'postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}')

# df = pd.read_sql_query(query,con=engine)
# SYMBOLS = ['MSFT', 'ETH-USD', 'GOOG', 'AAPL', 'BTC-USD']
# IS_INVESTED = False

# try:
#     # countries = st.multiselect(
#     #     "Choose countries", list(df.index), ["China", "United States of America"]
#     # )
#     # if not countries:
#     #     st.error("Please select at least one country.")
#     # else:
#     #     data = df.loc[countries]
#     #     data /= 1000000.0
#     #     st.write("### Gross Agricultural Production ($B)", data.sort_index())

#     #     data = data.T.reset_index()
#     #     data = pd.melt(data, id_vars=["index"]).rename(
#     #         columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
#     #     )
#     #     chart = (
#     #         alt.Chart(data)
#     #         .mark_area(opacity=0.3)
#     #         .encode(
#     #             x="year:T",
#     #             y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
#     #             color="Region:N",
#     #         )
#     #     )
#         st.altair_chart(df, use_container_width=True)
# except URLError as e:
#     st.error(
#         """
#         **This demo requires internet access.**

#         Connection error: %s
#     """
#         % e.reason
#     )