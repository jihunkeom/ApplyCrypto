import streamlit as st
import os

def render_table_list(data):
    st.header("테이블 접근 정보")
    
    if not data:
        st.info("테이블 접근 정보를 사용할 수 없습니다.")
        return

    # Sidebar Summary
    table_names = [table.get('table_name', 'Unknown') for table in data]
    st.sidebar.markdown("### Tables")
    for name in table_names:
        st.sidebar.text(f"- {name}")

    # Main List
    for table in data:
        table_name = table.get('table_name', 'Unknown')
        with st.expander(f"Table: {table_name}", expanded=True):
            st.subheader(f"Table: {table_name}")
            
            # Columns
            columns = table.get('columns', [])
            if columns:
                st.markdown("**Columns:**")
                col_names = [col.get('name') for col in columns]
                st.write(", ".join(col_names))
            
            # Access Files
            access_files = table.get('access_files', [])
            if access_files:
                st.markdown("**Access Files:**")
                for file in access_files:
                    st.text(f"- {os.path.basename(file)}")

            # SQL Queries Summary Link
            st.markdown("### SQL Queries")
            sql_queries = table.get('sql_queries', [])
            
            if sql_queries:
                for query in sql_queries:
                    col1, col2 = st.columns([0.8, 0.2])
                    query_id = query.get('id', 'Unknown')
                    query_type = query.get('query_type', 'Unknown')
                    
                    with col1:
                        st.markdown(f"**{query_id}** ({query_type})")
                        # Show a snippet of SQL
                        sql_snippet = query.get('sql', '')[:100].replace('\n', ' ')
                        if len(query.get('sql', '')) > 100:
                            sql_snippet += "..."
                        st.caption(f"`{sql_snippet}`")

                    with col2:
                        # Unique key for button using table name and query id
                        btn_key = f"btn_{table_name}_{query_id}"
                        if st.button("상세 보기", key=btn_key):
                            st.session_state["selected_query"] = query
                            st.session_state["selected_table_name"] = table_name
                            st.session_state["current_tab"] = "SQL Details"
                            st.rerun()
                    
                    st.divider()
            else:
                st.info("이 테이블 항목에 대해 정의된 특정 SQL 쿼리가 없습니다.")
