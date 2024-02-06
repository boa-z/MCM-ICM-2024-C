#!/bin/bash

# 获取输入参数
old_value=$1
new_value=$2

# 使用sed命令替换config.json中的值
sed -i.bak "s|$old_value|$new_value|g" config.json

echo "替换完成！"
