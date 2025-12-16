#!/usr/bin/env python3
"""
PlantUML 다이어그램을 이미지로 렌더링하는 스크립트
"""

from pathlib import Path

from plantuml import PlantUML


def render_diagram(puml_file: Path, output_format: str = "png"):
    """
    PlantUML 파일을 이미지로 렌더링

    Args:
        puml_file: PlantUML 파일 경로
        output_format: 출력 형식 (png, svg 등)
    """
    try:
        # PlantUML 서버 사용 (공개 서버)
        server = PlantUML(url="http://www.plantuml.com/plantuml/img/")

        # 출력 파일 경로
        output_file = puml_file.with_suffix(f".{output_format}")

        print(f"렌더링 중: {puml_file.name} -> {output_file.name}")

        # 파일 내용 읽기
        with open(puml_file, "r", encoding="utf-8") as f:
            puml_content = f.read()

        # 다이어그램 렌더링 (processes 메서드 사용)
        diagram_data = server.processes(puml_content)

        # 응답 검증
        if not isinstance(diagram_data, bytes):
            print(f"✗ 오류: 예상치 못한 응답 타입: {type(diagram_data)}")
            return False

        # HTML 오류 페이지인지 확인
        if diagram_data.startswith(b"<") or diagram_data.startswith(b"<!DOCTYPE"):
            print("✗ 오류: PlantUML 서버가 HTML 오류 페이지를 반환했습니다.")
            error_text = diagram_data[:500].decode("utf-8", errors="ignore")
            print(f"   오류 내용: {error_text[:200]}...")
            return False

        # PNG 파일인지 확인 (PNG 시그니처: 89 50 4E 47)
        if not diagram_data.startswith(b"\x89PNG"):
            print("✗ 오류: 생성된 데이터가 PNG 형식이 아닙니다.")
            print(f"   헤더: {diagram_data[:10]}")
            return False

        # 파일 저장
        with open(output_file, "wb") as f:
            f.write(diagram_data)

        # 파일 크기 확인
        file_size = output_file.stat().st_size
        print(f"✓ 완료: {output_file} ({file_size:,} bytes)")
        return True

    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    # 스크립트가 있는 디렉터리
    script_dir = Path(__file__).parent

    # PlantUML 파일 목록
    puml_files = [
        script_dir / "component_diagram.puml",
        script_dir / "class_diagram.puml",
    ]

    print("PlantUML 다이어그램 렌더링 시작...\n")

    success_count = 0
    for puml_file in puml_files:
        if puml_file.exists():
            if render_diagram(puml_file, "png"):
                success_count += 1
        else:
            print(f"✗ 파일을 찾을 수 없습니다: {puml_file}")

    print(f"\n완료: {success_count}/{len(puml_files)}개 파일 렌더링 성공")


if __name__ == "__main__":
    main()
