from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="hello", ext_modules=[cpp_extension.CppExtension("hello_cpp", [])])