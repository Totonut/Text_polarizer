#!/bin/sh

export CLASSPATH="`pwd`/stanford-parser-full-2014-08-27/stanford-parser:`pwd`/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models:`pwd`/stanford-postagger-full-2014-08-27/stanford-postagger.jar:`pwd`/stanford-ner-2014-08-27/stanford-ner.jar"
python extract_components.py
