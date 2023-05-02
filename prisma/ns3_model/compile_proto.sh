#!/bin/sh

protoc -I=. --python_out=. ./messages.proto