
[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

------


[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)


![](images/322.%20Coin%20Change%20and%20518.%20Coin%20Change%202.png)

![](images/322-1.png)
![](images/322-2.png)


#### iterate amount

``` go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1) // 需要的最小硬币数量
	dp[0] = 0                   // 无法组成0的硬币
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1 //初始化需要最小硬币数量为最大值（不可能取到）
	}
	for i := 1; i <= amount; i++ { // 自底向上，遍历所有状态
		for _, coin := range coins { //求所有选择的最小值 min(dp[4],dp[3],dp[0])+1
			if i-coin >= 0 {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if amount < dp[amount] { // 不可能兑换成比自己更大的硬币
		return -1
	}
	return dp[amount]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```



[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

![](images/518.png)

#### iterate coins

``` go
func change(amount int, coins []int) int {
	dp := make([]int, amount+1) // dp[x] 表示金额之和等于 xx 的硬币组合数
	dp[0] = 1                   // 当不选取任何硬币时，金额之和才为 0，只有 1 种硬币组合
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			// 如果存在一种硬币组合的金额之和等于 i - coin，则在该硬币组合中增加一个面额为 coin 的硬币，
			dp[i] += dp[i-coin] // 即可得到一种金额之和等于 i 的硬币组合。
		}
	}
	return dp[amount]
}
```


[参考视频](https://www.bilibili.com/video/BV1kX4y1P7M3?spm_id_from=333.999.0.0&vd_source=c42cfd612643754cd305aa832e64afe1)

[代码](https://happygirlzt.com/codelist.html)