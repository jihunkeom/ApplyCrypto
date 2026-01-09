import json
import os
import streamlit as st

# Import components
try:
    from tabs import table_detail, sql_detail, call_graph_view
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))
    import table_detail
    import sql_detail
    import call_graph_view

@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from: {file_path}")
        return None

def main():
    st.set_page_config(page_title="ApplyCrypto", layout="wide")
    
    # Path to JSON
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "example_for_prompt", "table_access_info.json")
    cg_path = os.path.join(current_dir, "example_for_prompt", "call_graph.json")
    
    data = load_data(json_path)
    if not data:
        st.error("Failed to load table access info.")
        return

    # Load Call Graph if not already loaded (cache it roughly)
    if "call_graph_data" not in st.session_state:
        cg_data = load_data(cg_path)
        if cg_data:
            st.session_state["call_graph_data"] = cg_data
        else:
            st.warning("Failed to load call graph data. Call graph features will be disabled.")
            st.session_state["call_graph_data"] = {}

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a **Table** or specific **SQL Query**.")

    # Initialize default state if needed
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "welcome" # welcome, table, sql

    for table in data:
        table_name = table.get('table_name', 'Unknown')
        
        # We use an expander for each table to group its queries
        # Note: 'expanded' state is not easily persistent without extra logic, 
        # so they might close on rerun unless we manage IDs carefully.
        # For a simple version, we let them operate naturally.
        with st.sidebar.expander(f"📁 {table_name}", expanded=False):
            # Table Overview Button
            if st.button("Overview", key=f"btn_overview_{table_name}"):
                st.session_state["view_mode"] = "table"
                st.session_state["selected_table"] = table
                st.session_state["selected_table_name"] = table_name # for context in sql view
                st.rerun()
            
            # List SQL Queries
            sql_queries = table.get('sql_queries', [])
            for query in sql_queries:
                qid = query.get('id', 'Unknown')
                # Use a unique key for every button
                if st.button(f"📄 {qid}", key=f"btn_sql_{table_name}_{qid}"):
                    st.session_state["view_mode"] = "sql"
                    st.session_state["selected_table"] = table # Context
                    st.session_state["selected_table_name"] = table_name
                    st.session_state["selected_query"] = query
                    st.rerun()

    # --- Main Content Area ---
    if st.session_state["view_mode"] == "welcome":
        st.title("ApplyCrypto UI")
        st.info("👈 Please select a Table or SQL Query from the sidebar to view details.")
        
    elif st.session_state["view_mode"] == "table":
        table_data = st.session_state.get("selected_table")
        table_detail.render_table_detail(table_data)
        
    elif st.session_state["view_mode"] == "sql":
        # Ensure sql_detail uses the correct state key
        # sql_detail currently looks for 'selected_query' and 'selected_table_name', which we set above.
        sql_detail.render_sql_detail()
        
    elif st.session_state["view_mode"] == "call_graph":
        call_graph_view.render_call_graph_view()

if __name__ == "__main__":
    main()
