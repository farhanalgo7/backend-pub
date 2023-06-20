#!/bin/bash
cd docs/
printf 'n\n\nDocForFabric\n\nABC\n\n1\n\n' | sphinx-quickstart
cd ..
cp ./DocMaterial/index.rst ./DocMaterial/conf.py ./docs
sphinx-apidoc -o docs ./src_for_doc./backend_pub
tree
cat docs/conf.py
# rm docs/conf.rst docs/crawler.rst
cd docs/
make html
