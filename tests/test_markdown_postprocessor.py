import pytest
from llming_lodge.utils.markdown_postprocessor import LlmMarkdownPostProcessor

def check_md(input_md, expected_md):
    processor = LlmMarkdownPostProcessor(input_md)
    output = processor.process()
    if output.strip() != expected_md.strip():
        print("\n=== ACTUAL OUTPUT ===\n", output)
        print("\n=== EXPECTED OUTPUT ===\n", expected_md)
    assert output.strip() == expected_md.strip(), f"\n=== ACTUAL OUTPUT ===\n{output}\n=== EXPECTED OUTPUT ===\n{expected_md}\n"

def test_quadratic_formula():
    check_md(
        r"""
1. **Quadratic Formula:**
   \[
   x = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{2a}
   \]
""",
        r"""
1. **Quadratic Formula:**
   $$ x = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{2a} $$
"""
    )

def test_arithmetic_sequence_inline():
    check_md(
        r"""
3. **Arithmetic Sequence (nth term):**
   \( a_n = a_1 + (n-1)d \)
""",
        r"""
3. **Arithmetic Sequence (nth term):**
   $a_n = a_1 + (n-1)d$
"""
    )

def test_area_of_circle_inline():
    check_md(
        r"""
5. **Area of a Circle:**
   \(\pi r^2\)
""",
        r"""
5. **Area of a Circle:**
   $\pi r^2$
"""
    )

def test_pythagorean_theorem_block():
    check_md(
        r"""
4. **Pythagorean Theorem:**
   \[
   a^2 + b^2 = c^2
   \]
""",
        r"""
4. **Pythagorean Theorem:**
   $$ a^2 + b^2 = c^2 $$
"""
    )

def test_sine_cosine_rule_two_blocks():
    check_md(
        r"""
9. **Sine and Cosine Rule:**
   \[
   a^2 = b^2 + c^2 - 2bc \cdot \cos(A)
   \]

   \[
   \frac{a}{\sin(A)} = \frac{b}{\sin(B)} = \frac{c}{\sin(C)}
   \]
""",
        r"""
9. **Sine and Cosine Rule:**
   $$ a^2 = b^2 + c^2 - 2bc \cdot \cos(A) $$

   $$ \frac{a}{\sin(A)} = \frac{b}{\sin(B)} = \frac{c}{\sin(C)} $$
"""
    )
