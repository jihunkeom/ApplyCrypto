"""
SourceFileCollector 사용 예제

이 예제는 SourceFileCollector를 사용하여 프로젝트 내의 소스 파일을 수집하는 방법을 보여줍니다.
"""

from pathlib import Path


from collector.source_file_collector import SourceFileCollector
from config.config_manager import ConfigurationManager


CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def main():
    config_manager = ConfigurationManager(CONFIG_PATH)
    print("설정 로드 완료")


    # 3. Collector 초기화
    collector = SourceFileCollector(config_manager)
    print("Collector 초기화 완료")

    # 4. 파일 수집
    collected_files = list(collector.collect())
    
    # 5. 결과 출력
    print(f"\n[수집 결과] 총 {len(collected_files)}개 파일 발견")
    for file in collected_files:
        print(f"- {file.filename} ({file.size} bytes) : {file.relative_path}")
            
    print("\n=== 예제 종료 ===")


if __name__ == "__main__":
    main()
