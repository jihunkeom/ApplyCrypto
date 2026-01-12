import json
import os
import streamlit as st
from pathlib import Path
from config.config_manager import load_config, ConfigurationError

from ui_app.tabs import table_detail, sql_detail, call_graph_view

@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"JSON 디코딩 오류: {file_path}")
        return None

def main():
    st.set_page_config(page_title="ApplyCrypto", layout="wide")
    
    # Load configuration
    try:
        # Assuming config.json is in the project root
        config_path = os.path.join(os.getcwd(), "config.json")
        config = load_config(config_path)
    except Exception as e:
        st.error(f"설정을 불러오는데 실패했습니다: {e}")
        st.info(f"{os.getcwd()}에 'config.json'이 존재하는지 확인해주세요.")
        return

    target_project = Path(config.target_project)
    results_dir = target_project / ".applycrypto" / "results"
    
    json_path = results_dir / "table_access_info.json"
    cg_path = results_dir / "call_graph.json"
    
    if not json_path.exists():
        st.error("분석 결과를 찾을 수 없습니다 (table_access_info.json).")
        st.warning("'analyze' 명령어를 실행하여 필요한 데이터를 생성해주세요.")
        return

    data = load_data(str(json_path))
    if not data:
        st.error("테이블 접근 정보를 불러오는데 실패했습니다.")
        return

    # Load Call Graph if not already loaded (cache it roughly)
    if "call_graph_data" not in st.session_state:
        if cg_path.exists():
            cg_data = load_data(str(cg_path))
        else:
            cg_data = None
        if cg_data:
            st.session_state["call_graph_data"] = cg_data
        else:
            st.warning("콜 그래프 데이터를 불러오는데 실패했습니다. 콜 그래프 기능이 비활성화됩니다.")
            st.session_state["call_graph_data"] = {}

    # --- Sidebar Navigation ---
    st.sidebar.title("탐색")
    st.sidebar.markdown("**테이블** 또는 특정 **SQL 쿼리**를 선택하세요.")

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
            if st.button("개요", key=f"btn_overview_{table_name}"):
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
        st.info("👈 사이드바에서 테이블이나 SQL 쿼리를 선택하여 상세 정보를 확인하세요.")
        
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
