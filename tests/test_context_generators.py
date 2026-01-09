import unittest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.append(str(src_dir))

from config.config_manager import Configuration
from models.table_access_info import TableAccessInfo
from models.modification_context import CodeSnippet
from modifier.code_generator.base_code_generator import BaseCodeGenerator
from modifier.context_generator.jdbc_context_generator import JdbcContextGenerator
from modifier.context_generator.mybatis_context_generator import MybatisContextGenerator

class TestContextGenerators(unittest.TestCase):

    def setUp(self):
        # Common usage basic mock config
        self.mock_config = MagicMock(spec=Configuration)
        self.mock_config.target_project = "D:/RES/workspace/gam-pjt"
        self.mock_config.max_tokens_per_batch = 10000

        self.mock_code_generator = MagicMock(spec=BaseCodeGenerator)
        self.mock_code_generator.calculate_token_size.return_value = 10
        self.mock_code_generator.create_prompt.return_value = "Prompt"

    def test_jdbc_context_generator(self):
        print("\nStarting JDBC Context Generator Test...")
        
        generator = JdbcContextGenerator(self.mock_config, self.mock_code_generator)

        # Prepare Mock TableAccessInfo with JDBC DATA
        layer_files = {
            "Repository": [
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bc\\comm\\dao\\dem\\dvo\\TGAH00017DVO.java"
            ],
            "dem_daq": [
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bc\\comm\\dao\\dem\\TGAH00017DEM.java"
            ],
            "biz": [
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bc\\comm\\biz\\BCGaPesnInfoInqrBIZ.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bz\\mng\\biz\\BZMngSalesMngBIZ.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bz\\mnger\\biz\\BZMngerInvwBIZ.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\tgt\\biz\\TGTrgtBrncBIZ.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\tgt\\biz\\TGTrgtFpBIZ.java"
            ],
            "svc": [
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bc\\comm\\svc\\IBCGaPesnInqrSVC.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bc\\comm\\svc\\impl\\IBCGaPesnInqrSVCImpl.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bz\\mng\\svc\\IBZMngSalesMngSVC.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bz\\mng\\svc\\impl\\IBZMngSalesMngSVCImpl.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bz\\mnger\\svc\\IBZMngerInvwSVC.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\bz\\mnger\\svc\\impl\\IBZMngerInvwSVCImpl.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\tgt\\svc\\ITGTrgtBrncSVC.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\tgt\\svc\\ITGTrgtFpSVC.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\tgt\\svc\\impl\\ITGTrgtBrncSVCImpl.java",
                "D:\\RES\\workspace\\gam-pjt\\src\\main\\java\\sli\\gam\\tgt\\svc\\impl\\ITGTrgtFpSVCImpl.java"
            ]
        }

        table_access_info = TableAccessInfo(
            table_name="TEST_TABLE",
            columns=[],
            access_files=[],
            query_type="SELECT",
            layer_files=layer_files
        )

        with patch.object(generator, '_read_files', side_effect=lambda file_paths, root: [MagicMock(path=p, content=f"Content of {Path(p).name}") for p in file_paths]):
            contexts = generator.generate(table_access_info)

        print(f"Generated {len(contexts)} contexts.")

        for ctx in contexts:
            layer = ctx.layer
            count = len(ctx.code_snippets)
            print(f"Context Layer: {layer}, File Count: {count}")
            # Show last up to 4 parts to give context (keyword/layer/...)
            files = [os.path.join("...", *Path(s.path).parts[-4:]) for s in ctx.code_snippets]
            
            print(f"  Files: [")
            for f in files:
                print(f"    {f},")
            print(f"  ]")

        expected_counts = {
            "Repository": 1,
            "dem_daq": 1,
            "comm": 3,
            "mng": 3,
            "mnger": 3,
            "tgt": 6
        }
        
        found_layers = {}
        for ctx in contexts:
            found_layers[ctx.layer] = len(ctx.code_snippets)
            
        for layer, expected in expected_counts.items():
            self.assertIn(layer, found_layers, f"Missing layer '{layer}'")
            self.assertEqual(found_layers[layer], expected, f"Layer '{layer}' count mismatch. Expected {expected}, got {found_layers[layer]}.")

        print("JDBC Context Generator Test Passed.")


    def test_mybatis_context_generator(self):
        print("\nStarting Mybatis Context Generator Test...")
        
        # Setup specific config for Mybatis test if needed, or reuse self.mock_config
        self.mock_config.target_project = "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung"
        
        generator = MybatisContextGenerator(self.mock_config, self.mock_code_generator)

        layer_files = {
            "service": [
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\service\\CocoonAcctAthntServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\service\\CocoonAcctDpstServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\service\\CocoonAcctWdrServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\service\\CocoonArsAthntServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\api\\common\\point\\PntCommService.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\service\\PntCustomersBankInfoService.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\service\\PntCustomersBankInfoServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\service\\PntCustomersCreateServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\issue\\service\\PntIssueChargeServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\pg\\service\\PgArsServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\pg\\service\\PgAuthServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\use\\service\\PntUseCreditServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\use\\service\\PntUseDirectServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\use\\service\\PntUseHealServiceImpl.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Web_samsung\\src\\main\\java\\com\\cps\\web\\point\\custaccn\\service\\PointCustAccnService.java"
            ],
            "controller": [
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\controller\\PntCustomersBankInfoController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\controller\\PntCustomersCreateController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\issue\\controller\\PntIssueChargeController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\pg\\controller\\PgArsController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\pg\\controller\\PgAuthController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\use\\controller\\PntUseCreditController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\use\\controller\\PntUseDirectController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\use\\controller\\PntUseHealController.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Web_samsung\\src\\main\\java\\com\\cps\\web\\point\\custaccn\\controller\\PointCustAccnController.java"
            ],
            "repository": [
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\dao\\CocoonAcctAthntDao.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\dao\\CocoonAcctDpstDao.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\dao\\CocoonAcctWdrDao.java",
                "D:\\PII-EncryptEx\\PrimusCPS_Eai_samsung\\src\\main\\java\\com\\cps\\eai\\cocoon\\dao\\CocoonCommonDao.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\dao\\PntCustomersBankInfoDao.java",
                "D:\\PII-EncryptEx\\PrimusCPS_ExternalApi_samsung\\src\\main\\java\\com\\cps\\api\\point\\customers\\dao\\PntCustomersCreateDao.java"
            ]
        }

        table_access_info = TableAccessInfo(
            table_name="TEST_TABLE",
            columns=[{"name": "COL1", "new_column": False}],
            access_files=[],
            query_type="SELECT",
            layer_files=layer_files
        )

        # Mock _read_files
        def mock_read_files(file_paths, project_root):
            snippets = []
            for p in file_paths:
                snippets.append(CodeSnippet(path=p, content=f"Content of {p}"))
            return snippets
        
        with patch.object(generator, '_read_files', side_effect=mock_read_files):
            contexts = generator.generate(table_access_info)

        print(f"Generated {len(contexts)} contexts.")
        
        for ctx in contexts:
            print(f"Context Layer (Group): {ctx.layer}")
            print(f"File Count: {ctx.file_count}")
            for snippet in ctx.code_snippets:
                # Print filename (split by backslash)
                filename = snippet.path.split(os.sep)[-1]
                if '\\' in snippet.path and os.sep != '\\': # Handle windows paths if running on non-windows or mixed
                     filename = snippet.path.split('\\')[-1]
                elif '/' in snippet.path and os.sep != '/':
                     filename = snippet.path.split('/')[-1]
                
                # Fallback to simple split if path separator is ambiguous or match exact code
                if '\\' in snippet.path:
                    filename = snippet.path.split('\\')[-1]
                else:
                    filename = snippet.path.split('/')[-1]

                print(f" - {filename}") 
            print("-" * 30)

        # Check 'PntCustomersBankInfo' group
        pnt_bank_group = next((c for c in contexts if c.layer == "PntCustomersBankInfo"), None)
        self.assertIsNotNone(pnt_bank_group, "Group 'PntCustomersBankInfo' not found.")
        
        files = [Path(s.path).name for s in pnt_bank_group.code_snippets]
        self.assertIn("PntCustomersBankInfoService.java", files)
        self.assertIn("PntCustomersBankInfoServiceImpl.java", files)
        self.assertIn("PntCustomersBankInfoController.java", files)
        self.assertIn("PntCustomersBankInfoDao.java", files)
        
        print("Mybatis Context Generator Test Passed.")

if __name__ == "__main__":
    unittest.main()
