"""
Call Chain Diff Generator 모듈

호출 체인(Controller → Service → Repository) 단위로 LLM을 호출하여
가장 적절한 레이어에 암복호화 코드를 삽입하는 Diff를 생성합니다.
"""


from ..base_diff_generator import BaseDiffGenerator, DiffGeneratorInput


class CallChainDiffGenerator(BaseDiffGenerator):
    """
    Call Chain Diff 생성기

    호출 체인 단위로 암복호화 코드 Diff를 생성합니다.
    레이어별 배치 처리 대신, 하나의 호출 체인에 포함된 모든 파일을
    한 번의 LLM 호출로 처리하여 가장 적절한 레이어에 암복호화를 적용합니다.
    """

    def create_prompt(self, input_data: DiffGeneratorInput) -> str:
        """
        Call Chain 전용 프롬프트를 생성합니다.

        Args:
            input_data: Diff 생성 입력

        Returns:
            str: 생성된 프롬프트
        """
        # extra_variables에서 file_list 추출
        file_list = input_data.extra_variables.get("file_list", "")

        # 배치 프롬프트 생성 (절대 경로 포함)
        batch_variables = {
            "table_info": input_data.table_info,
            "source_files": "\n\n".join(
                [
                    f"=== File Path (Absolute): {snippet.path} ===\n{snippet.content}"
                    for snippet in input_data.code_snippets
                ]
            ),
            "file_count": len(input_data.code_snippets),
            "file_list": file_list,
            **(input_data.extra_variables or {}),
        }

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        from jinja2 import Template

        template = Template(template_str)
        return template.render(**batch_variables)
