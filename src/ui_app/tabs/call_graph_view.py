import streamlit as st
import os

@st.cache_data
def get_code_snippet(file_path, start_line, end_line):
    """
    Reads the file at file_path and extracts lines from start_line to end_line.
    Returns the snippet as a string or None if file not found.
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 0-indexed adjustment for list access
        # start_line and end_line are typically 1-based from the AST
        s_idx = max(0, start_line - 1)
        # Limit end index
        e_idx = min(len(lines), end_line)
        
        snippet = "".join(lines[s_idx:e_idx])
        return snippet
    except Exception as e:
        return f"파일 읽기 오류: {e}"

def find_paths_to_method(node, target_method_name, current_path, results):
    # node is a dict representing a method call in the tree
    # current_path represents the stack of calls leading to this node
    
    # We create a lightweight snapshot of the current node info to store in the path
    node_info = {
        "signature": node.get('method_signature', 'Unknown'),
        "layer": node.get('layer', 'Unknown'),
        "class": node.get('class_name', ''),
        "method": node.get('method_name', ''),
        "file_path": node.get('file_path'),
        "line_number": node.get('line_number'),
        "end_line_number": node.get('end_line_number')
    }
    
    new_path = current_path + [node_info]
    
    # Check match based on user rule: *Mapper.[sql_id]
    sig = node.get('method_signature', '')
    
    is_match = False
    # Check if signature matches pattern like "SomethingMapper.sql_id"
    # We check if it ends with "Mapper.{target_method_name}" to align with the rule.
    if sig and sig.endswith(f"Mapper.{target_method_name}"):
        is_match = True
    
    if is_match:
        results.append(new_path)
    
    # Recurse
    # children might be key 'children' or empty
    children = node.get('children', [])
    for child in children:
        find_paths_to_method(child, target_method_name, new_path, results)

def render_call_graph_view():
    target_id = st.session_state.get("target_sql_id")
    cg_data = st.session_state.get("call_graph_data")
    
    if not target_id:
        st.error("대상 SQL ID가 선택되지 않았습니다.")
        return
        
    if not cg_data:
        st.error("콜 그래프 데이터가 없습니다.")
        return
        
    st.header(f"Call Flows for: `{target_id}`")
    
    if st.button("← SQL 상세로 돌아가기"):
        st.session_state["view_mode"] = "sql"
        st.rerun()
    
    st.divider()
    
    # Search for paths
    trees = cg_data.get('call_trees', [])
    paths = []
    
    with st.spinner("콜 그래프 검색 중..."):
        for tree in trees:
            find_paths_to_method(tree, target_id, [], paths)
            
    if not paths:
        st.warning(f"알려진 콜 그래프에서 `{target_id}`에 도달하는 실행 경로를 찾을 수 없습니다.")
        st.info("팁: 콜 그래프가 이 매퍼 메서드를 사용하는 엔드포인트를 포함하는지 확인하세요.")
        return
        
    st.success(f"{len(paths)}개의 고유한 실행 경로를 찾았습니다.")
    
    for i, path in enumerate(paths):
        if i > 0:
            st.markdown(
                """
                <hr style="
                    border: 0; 
                    height: 3px; 
                    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(13, 110, 253, 0.75), rgba(0, 0, 0, 0)); 
                    margin: 30px 0;
                ">
                """, 
                unsafe_allow_html=True
            )

        # Entry point is the first element
        entry = path[0]
        entry_sig = entry['signature']
        
        with st.expander(f"Path {i+1}: from...{entry_sig[-50:] if len(entry_sig)>50 else entry_sig}", expanded=True):
            # Render the stack
            for stage_idx, step in enumerate(path):
                sig = step['signature']
                layer = step['layer']
                
                # Check if it's the target
                is_target = (stage_idx == len(path) - 1)
                
                # Styling
                if is_target:
                    box_style = "background-color: #ffeeba; border: 2px solid #ffc107; padding: 10px; border-radius: 5px;"
                    icon = "🎯"
                elif stage_idx == 0:
                    box_style = "background-color: #d1e7dd; border: 1px solid #198754; padding: 10px; border-radius: 5px;"
                    icon = "🚀"
                else:
                    box_style = "background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; border-radius: 5px;"
                    icon = "⬇️"
                
                # Build the visible box
                st.markdown(
                    f"""
                    <div style="{box_style}">
                        <strong>{icon} {layer}</strong><br>
                        <code>{sig}</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Expandable code snippet
                fp = step.get('file_path')
                sl = step.get('line_number')
                el = step.get('end_line_number')
                
                if fp and sl and el:
                    # Provide a unique key using path index and step index
                    with st.expander("Code Snippet", expanded=False):
                        snippet = get_code_snippet(fp, sl, el)
                        if snippet:
                            st.code(snippet, language='java')
                            st.caption(f"Source: {os.path.basename(fp)} (Line {sl}-{el})")
                        else:
                            st.text("코드 스니펫을 사용할 수 없습니다 (파일을 로컬에서 찾을 수 없거나 범위가 잘못되었습니다).")
                
                if not is_target:
                    st.markdown("<div style='text-align: center; font-size: 20px;'>↓</div>", unsafe_allow_html=True)


