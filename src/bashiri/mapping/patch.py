import importlib.resources
from typing import Tuple, Dict, List

from tests4py.projects import Project, resources
from unidiff import PatchSet


def get_patch(file_path: str) -> PatchSet:
    with open(file_path, "r") as f:
        return PatchSet(f)


class PatchTranslator:
    def __init__(self):
        self.translation: Dict[Tuple[str, int], Tuple[str, List[int]]] = dict()
        self.modifiers: Dict[str, List[Tuple[int, int], int]] = dict()
        self.files: Dict[str, str] = dict()

    def translate(self, line: Tuple[str, int]) -> Tuple[str, List[int]]:
        file, line = line
        if (file, line) in self.translation:
            return self.translation[(file, line)]
        modifier = 0
        if file in self.modifiers:
            for m in self.modifiers[file]:
                if m[0] <= line >= m[1]:
                    modifier += m[2]
        return self.files.get(file, file), [line + modifier]

    def add_translation(
        self,
        last_removed_chunk: List[int],
        src_file: str,
        last_added_chunk: List[int],
        target_file: str,
    ):
        self.translation.update(
            {
                (target_file, line): (src_file, last_removed_chunk)
                for line in last_added_chunk
            }
        )

    @staticmethod
    def build_translator(patch: PatchSet) -> "PatchTranslator":
        translator = PatchTranslator()
        last_removed_chunk = list()
        last_added_chunk = list()

        for file in patch:
            src_file = file.source_file
            if src_file and (src_file.startswith("a/") or src_file.startswith("b/")):
                src_file = src_file[2:]
            target_file = file.target_file
            if target_file and (
                target_file.startswith("a/") or target_file.startswith("b/")
            ):
                target_file = target_file[2:]
            translator.files[target_file] = src_file
            if target_file not in translator.modifiers:
                translator.modifiers[target_file] = list()
            for hunk in file:
                translator.modifiers[target_file].append(
                    (
                        hunk.target_start,
                        hunk.target_start + hunk.target_length - 1,
                        hunk.removed - hunk.added,
                    )
                )
                for line in hunk:
                    if line.is_context:
                        if last_removed_chunk or last_added_chunk:
                            translator.add_translation(
                                last_removed_chunk,
                                src_file,
                                last_added_chunk,
                                target_file,
                            )
                            last_removed_chunk = list()
                            last_added_chunk = list()
                        translator.translation[(target_file, line.target_line_no)] = (
                            src_file,
                            [line.source_line_no],
                        )
                    elif line.is_removed:
                        last_removed_chunk.append(line.source_line_no)
                    elif line.is_added:
                        last_added_chunk.append(line.target_line_no)
                if last_removed_chunk or last_added_chunk:
                    translator.add_translation(
                        last_removed_chunk, src_file, last_added_chunk, target_file
                    )
                    last_removed_chunk = list()
                    last_added_chunk = list()
        return translator

    @staticmethod
    def build_t4p_translator(project: Project) -> "PatchTranslator":
        project_resources = importlib.resources.files(
            getattr(getattr(resources, project.project_name), f"bug_{project.bug_id}")
        )
        with importlib.resources.as_file(
            project_resources.joinpath(
                "fix.patch",
            )
        ) as resource:
            patch = get_patch(resource)
            return PatchTranslator.build_translator(patch)

    @staticmethod
    def build_translator_from_file(patch_file: str) -> "PatchTranslator":
        patch = get_patch(patch_file)
        return PatchTranslator.build_translator(patch)
