## 动态规划 

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

[120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

[63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

[91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) (小米)

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

[664. 奇怪的打印机](https://leetcode-cn.com/problems/strange-printer/)

[10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

### 补充：
------

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) (AI)

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

方法一：贪心

若当前指针所指元素之前的和小于0， 则丢弃当前元素之前的数列

```go
func maxSubArray(nums []int) int {
	if len(nums) == 0 {
		return math.MinInt64
	}
	cur_sum, max_sum := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		cur_sum = max(cur_sum, cur_sum+nums[i])
		max_sum = max(cur_sum, max_sum)
	}
	return max_sum
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

方法二：动态规划

若前一个元素大于0，则将其加到当前元素上



```go
func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] < nums[i-1]+nums[i] {
			nums[i] += nums[i-1]
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}
```

```go
func maxSubArray(nums []int) int {
	res, last := math.MinInt64, 0
	for i := 0; i < len(nums); i++ {
		cur_sum := max(last, 0) + nums[i]
		res = max(res, cur_sum)
		last = cur_sum
	}
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

```go
func maxSubArray(nums []int) int {
	for i := 1; i < len(nums); i++ {
		if nums[i-1] > 0 {
			nums[i] += nums[i-1]
		}
	}
	return max(nums)
}
func max(nums []int) int {
	res := math.MinInt64
	for i := 0; i < len(nums); i++ {
		if nums[i] > res {
			res = nums[i]
		}
	}
	return res
}
```

[120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

```go
func minimumTotal(triangle [][]int) int {
	bottom := triangle[len(triangle)-1]
	dp := make([]int, len(bottom))
	for i := range dp {
		dp[i] = bottom[i]
	}
	for i := len(dp) - 2; i >= 0; i-- {
		for j := 0; j < len(triangle[i]); j++ {
			dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]
		}
	}
	return dp[0]
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

思路
尝试把大问题拆分为子问题，它们的区别在于问题的规模。
规模在这里是：层高。
base case 是当矩阵行高只有 1 时，它的最优路径是显而易见的。

![](https://pic.leetcode-cn.com/13dafa7efab2287884a901a99c04c1f7b7ef2dcf5e6f2a8dfa477e2ac6890e8b-image.png)

有了一层高的「最优路径」，我们推出两层高的「最优路径」
有了两层高的「最优路径」，我们推出三层高的「最优路径」
……
triangle.length-1层高的「最优路径」推出 triangle.length 层高的「最优路径」




[63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

```go

```

[91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

```go

```

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```go

```

[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) (小米)