#!/usr/bin/zsh

var="""This
is a multiline
string.
<emphasis>209.204.146.22</emphasis>
""";

echo $(sed 's!<emphasis>([0-9]+(\.[0-9]+){3})</emphasis>!<inet>$1</inet>!' $var);
