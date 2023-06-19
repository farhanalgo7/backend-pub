#!/bin/bash
cd docs/
printf 'n\n\nDocForFabric\n\nPaddy\n\n1\n\n' | sphinx-quickstart
cd ..
cp ./DocMaterial/index.rst ./DocMaterial/conf.py ./docs
sphinx-apidoc -o docs ./src_for_doc./AutoDocAction
tree
cat docs/conf.py
# rm docs/conf.rst docs/crawler.rst
cd docs/
make html
