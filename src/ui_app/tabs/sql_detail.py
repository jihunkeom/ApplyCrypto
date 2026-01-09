import streamlit as st

def render_sql_detail():
    st.header("SQL Query Details")
    
    selected_query = st.session_state.get("selected_query")
    selected_table_name = st.session_state.get("selected_table_name")
    
    if not selected_query:
        st.info("Please select a SQL query from the 'Table Access Info' tab to view details here.")
        return

    st.subheader(f"Query: {selected_query.get('id', 'Unknown')}")
    if selected_table_name:
        st.caption(f"Belongs to Table: {selected_table_name}")
    
    # Metadata
    st.markdown("### Metadata")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f"**Type:** {selected_query.get('query_type', 'Unknown')}")
    with col2:
        st.markdown(f"**ID:** {selected_query.get('id', 'Unknown')}")
    with col3:
        if st.button("🔍 View in Call Graph"):
            st.session_state["target_sql_id"] = selected_query.get('id')
            st.session_state["view_mode"] = "call_graph"
            st.rerun()
        
    # SQL Content
    st.markdown("### SQL Code")
    sql_content = selected_query.get('sql', '')
    if sql_content:
        st.code(sql_content, language='sql')
    else:
        st.text("No SQL content available.")
        
    # Strategy Specific
    st.markdown("### Strategy Specific Info")
    strategy = selected_query.get('strategy_specific')
    if strategy:
        st.json(strategy)
    else:
        st.text("No strategy specific information.")
    
    # Raw JSON
    with st.expander("View Raw JSON"):
        st.json(selected_query)
