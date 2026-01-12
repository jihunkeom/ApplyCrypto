import streamlit as st
import os

def render_table_detail(table_data):
    if not table_data:
        st.error("선택된 테이블 데이터가 없습니다.")
        return

    table_name = table_data.get('table_name', 'Unknown')
    st.header(f"Table: {table_name}")
    
    # Columns
    columns = table_data.get('columns', [])
    if columns:
        st.markdown("### Columns")
        col_list = []
        for col in columns:
            name = col.get('name')
            is_new = col.get('new_column', False)
            status = "(New)" if is_new else ""
            col_list.append(f"- **{name}** {status}")
        st.markdown("\n".join(col_list))
    
    st.divider()

    # Access Files
    access_files = table_data.get('access_files', [])
    if access_files:
        st.markdown("### Access Files")
        for file in access_files:
            st.text(f"- {os.path.basename(file)}") # Show formatted text for files

    st.divider()

    # SQL Queries Summary
    st.markdown("### 관련 SQL 쿼리")
    sql_queries = table_data.get('sql_queries', [])
    
    if sql_queries:
        st.write(f"총 쿼리 수: {len(sql_queries)}")
        # We can list them simply here, as the sidebar allows navigation
        for q in sql_queries:
            qid = q.get('id', 'Unknown')
            qtype = q.get('query_type', 'Unknown')
            st.caption(f"{qid} ({qtype})")
    else:
        st.info("이 테이블 항목에 대해 정의된 특정 SQL 쿼리가 없습니다.")

    # Raw Data
    with st.expander("Raw Dictionary Data"):
        st.json(table_data)
