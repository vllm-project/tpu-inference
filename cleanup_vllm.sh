#!/bin/bash
current_pid=$$
pids=$(ps -ef | grep -i "vllm\|EngineCore\|APIServer" | grep -v grep | grep -v $current_pid | awk '{print $2}')
if [ -n "$pids" ]; then
    kill -9 $pids
fi
