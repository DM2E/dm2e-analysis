#!/bin/bash

content=$(mktemp)
echo "" > $content
index=out/index.html
cp tpl/index.html $index
for i in out/*;do
    i=$(basename "$i")
    echo "<li><a href='$i'>$i</a><br/></li>" >> $content
done
echo $content
sed -i "/\${content}/r $content" $index

OUTPUT_DIR=/var/lib/tomcat7/webapps/ROOT/visualize

rm $OUTPUT_DIR/*
cp out/* $OUTPUT_DIR/
chown tomcat7 $OUTPUT_DIR/*

rm $content
