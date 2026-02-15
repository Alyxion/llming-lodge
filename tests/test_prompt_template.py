import pytest
from llming_lodge.prompt.prompt_template import PromptTemplate

# --- Normal cases ---
def test_simple_variable():
    tpl = PromptTemplate("Hello {{ name }}!")
    assert tpl.render(parameters={"name": "World"}) == "Hello World!"

def test_for_loop_small():
    tpl = PromptTemplate("{% for x in items %}{{ x }}, {% endfor %}")
    out = tpl.render(parameters={"items": [1, 2, 3]})
    assert out.strip() == "1, 2, 3,"

def test_conditionals():
    tpl = PromptTemplate("{% if flag %}YES{% else %}NO{% endif %}")
    assert tpl.render(parameters={"flag": True}) == "YES"
    assert tpl.render(parameters={"flag": False}) == "NO"

# --- Safety/limit cases ---
def test_large_list_creation_blocked():
    tpl = PromptTemplate("{{ [1]*1000001 }}")
    with pytest.raises(ValueError):
        tpl.render()

def test_large_string_multiplication_blocked():
    tpl = PromptTemplate("{{ 'a'*1000001 }}")
    with pytest.raises(ValueError):
        tpl.render()

def test_large_range_in_for_blocked():
    tpl = PromptTemplate("{% for i in range(1000001) %}{{ i }}{% endfor %}")
    with pytest.raises(ValueError):
        tpl.render()

def test_timeout_in_loop():
    tpl = PromptTemplate("{% for i in range(9999) %}{{ i }}{% endfor %}")
    # Should raise TimeoutError due to time-aware thread check
    with pytest.raises(TimeoutError):
        tpl.render(max_time_s=0.001)
