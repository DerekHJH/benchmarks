# benchmarks

## ShareGPT

## Mooncake trace

优先mooncake trace，因为他自带时间戳。按照时间戳来发送就可以了。

# Metrics

如果我们把每一次all-to-all通信看成一次事件，那么
- 每一次事件，一定有最早结束和最晚结束的DP group，他们的结束时间分别是time_early和time_late。我们记录下time_range = time_late - time_early。
- 所有事件的time_range的分布图，平均值，方差，都可以辅助我们决定一个调度算法的好坏。