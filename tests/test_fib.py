import pytest
from fib import fibR

def test_fib_1_equals_1():
    assert fibR(1) == 1

def test_fib_2_equals_1():
    assert fibR(2) == 1

def test_fib_6_equals_8():
    assert fibR(6) == 8
