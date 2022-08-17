
[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

------


[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)


![](images/322.%20Coin%20Change%20and%20518.%20Coin%20Change%202.png)

![](images/322-1.png)
![](images/322-2.png)


#### iterate amount

```go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0 //base case
	for i := 1; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	for i := 1; i <= amount; i++ { //遍历所有状态的所有值
		for _, coin := range coins { //求所有选择的最小值 min(dp[4],dp[3],dp[0])+1
			if i-coin >= 0 {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if amount-dp[amount] < 0 {
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

```go
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}
```


[参考视频](https://www.bilibili.com/video/BV1kX4y1P7M3?spm_id_from=333.999.0.0&vd_source=c42cfd612643754cd305aa832e64afe1)

[代码](https://happygirlzt.com/codelist.html)