"""
ApplyCrypto CLI 진입점

프로젝트 루트에서 실행:
  python main.py [command] [options]

또는 설치 후:
  applycrypto [command] [options]
"""

from dotenv import load_dotenv
import sys

from cli.cli_controller import CLIController


def main():
    load_dotenv(".env")

    """CLI 메인 함수"""
    controller = CLIController()
    exit_code = controller.execute()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
