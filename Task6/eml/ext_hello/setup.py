import setuptools
import torch.utils.cpp_extension
import os

setuptools.setup( name        = "eml_ext",
                  ext_modules = [ torch.utils.cpp_extension.CppExtension( 'eml_ext_hello_world_cpp',
                                                                          ['eml/ext/HelloWorld.cpp'] ) ],
                  cmdclass = { 'build_ext': torch.utils.cpp_extension.BuildExtension }
)