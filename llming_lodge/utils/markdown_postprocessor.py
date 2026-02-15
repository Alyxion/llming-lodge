import re

class LlmMarkdownPostProcessor:
    """Helper class to post-process markdown text to convert LaTeX math to markdown math"""

    def __init__(self, text: str):
        self._text = text

    def filter_text_markdown(self) -> str:
        """
        Convert LaTeX math to markdown math

        :param text: Input text
        :return: Text with LaTeX math converted to markdown math
        """
        text = self._text
        # Convert LaTeX inline math: \( ... \) -> $...$
        text = re.sub(r'\\\(\s*([\s\S]+?)\s*\\\)', lambda m: f"${m.group(1).strip()}$", text)
        # Convert LaTeX display math: \[ ... \] -> $$...$$
        text = re.sub(r'\\\[\s*([\s\S]+?)\s*\\\]', lambda m: f"$$ {m.group(1).strip()} $$", text)

        return text

    def process(self) -> str:
        self._text = self.filter_text_markdown()
        return self._text

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value