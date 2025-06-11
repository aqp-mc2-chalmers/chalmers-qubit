import unittest
import subprocess
import pathlib

class test_mkdocs_build(unittest.TestCase):
    def test_build_docs_without_warnings(self):
        """Test that MkDocs builds the documentation without warnings or errors."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()
        
        result = subprocess.run(
            ["mkdocs", "build", "--strict"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = result.stdout

        # Check that the build completed successfully
        self.assertEqual(result.returncode, 0, msg=f"MkDocs build failed:\n{output}")

        # Check for presence of warnings or errors
        self.assertNotIn("WARNING", output, msg=f"MkDocs build produced warnings:\n{output}")
        self.assertNotIn("ERROR", output, msg=f"MkDocs build produced errors:\n{output}")


if __name__ == "__main__":
    unittest.main()