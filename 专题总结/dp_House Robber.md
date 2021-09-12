
[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

[337.打家劫舍III](https://leetcode-cn.com/problems/house-robber-iii/)

------

### 思路1：

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

There is some frustration when people publish their perfect fine-grained algorithms without sharing any information abut how they were derived. This is an attempt to change the situation. There is not much more explanation but it's rather an example of higher level improvements. Converting a solution to the next step shouldn't be as hard as attempting to come up with perfect algorithm at first attempt.

This particular problem and most of others can be approached using the following sequence:

    1. Find recursive relation
    2. Recursive (top-down)
    3. Recursive + memo (top-down)
    4. Iterative + memo (bottom-up)
    5. Iterative + N variables (bottom-up)

#### Step 1. Figure out recursive relation.
A robber has 2 options: a) rob current house i; b) don't rob current house.
If an option "a" is selected it means she can't rob previous i-1 house but can safely proceed to the one before previous i-2 and gets all cumulative loot that follows.
If an option "b" is selected the robber gets all the possible loot from robbery of i-1 and all the following buildings.
So it boils down to calculating what is more profitable:

- robbery of current house + loot from houses before the previous
- loot from the previous house robbery and any loot captured before that

rob(i) = Math.max( rob(i - 2) + currentHouseValue, rob(i - 1) )

#### Step 2. Recursive (top-down)
Converting the recurrent relation from Step 1 shound't be very hard.

```go
func rob(nums []int) int {
	return Rob(nums, len(nums)-1)
}
func Rob(nums []int, i int) int {
	if i < 0 {
		return 0
	}
	return max(Rob(nums, i-2)+nums[i], Rob(nums, i-1)) //max(抢 去下下家，不抢 去下家)
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```
Time Limit Exceeded
55/68 cases passed (N/A)

This algorithm will process the same i multiple times and it needs improvement. Time complexity: [to fill]


#### Step 3. Recursive + memo (top-down).

```go
var memo []int

func rob(nums []int) int {
	n := len(nums)
	memo = make([]int, n+1)
	for i := 0; i <= len(nums); i++ {
		memo[i] = -1
	}
	return Rob(nums, n-1)
}
func Rob(nums []int, i int) int {
	if i < 0 {
		return 0
	}
	if memo[i] >= 0 {
		return memo[i]
	}
	result := max(Rob(nums, i-2)+nums[i], Rob(nums, i-1)) //max(抢 去下下家，不抢 去下家)
	memo[i] = result
	return result
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


Much better, this should run in O(n) time. Space complexity is O(n) as well, because of the recursion stack, let's try to get rid of it.


#### Step 4. Iterative + memo (bottom-up)

```go
func rob(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	memo := make([]int, n+1)
	memo[0], memo[1] = 0, nums[0]
	for i := 1; i < n; i++ {
		val := nums[i]
		memo[i+1] = max(memo[i], memo[i-1]+val)
	}
	return memo[n]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

#### Step 5. Iterative + 2 variables (bottom-up)
We can notice that in the previous step we use only memo[i] and memo[i-1], so going just 2 steps back. We can hold them in 2 variables instead. This optimization is met in Fibonacci sequence creation and some other problems [to paste links].

```go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	prev1, prev2 := 0, 0
	for _, num := range nums {
		tmp := prev1
		prev1 = max(prev2+num, prev1)
		prev2 = tmp
	}
	return prev1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

```

[From good to great. How to approach most of DP problems.](https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.)









------
### 思路2：

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

### 方法一：动态规划

解题思路：


状态定义：

设动态规划列表 dp ，dp[i] 代表前 i 个房子在满足条件下的能偷窃到的最高金额。
转移方程：

设： 有 n 个房子，前 n 间能偷窃到的最高金额是 dp[n] ，前 n−1 间能偷窃到的最高金额是 dp[n−1] ，此时向这些房子后加一间房，此房间价值为 num ；

加一间房间后： 由于不能抢相邻的房子，意味着抢第 n+1 间就不能抢第 n 间；那么前 n+1 间房能偷取到的最高金额 dp[n+1] 一定是以下两种情况的 较大值 ：

不抢第 n+1 个房间，因此等于前 n 个房子的最高金额，即 dp[n+1] = dp[n] ；
抢第 n+1 个房间，此时不能抢第 n 个房间；因此等于前 n−1 个房子的最高金额加上当前房间价值，即 dp[n+1]=dp[n−1]+num ；
细心的我们发现： 难道在前 n 间的最高金额 dp[n] 情况下，第 n 间一定被偷了吗？假设没有被偷，那 n+1 间的最大值应该也可能是 dp[n+1] = dp[n] + num 吧？其实这种假设的情况可以被省略，这是因为：

假设第 n 间没有被偷，那么此时 dp[n] = dp[n-1] ，此时 dp[n+1] = dp[n] + num = dp[n-1] + num ，即两种情况可以 合并为一种情况 考虑；
假设第 n 间被偷，那么此时 dp[n+1] = dp[n] + num 不可取 ，因为偷了第 nn 间就不能偷第 n+1 间。
最终的转移方程： dp[n+1] = max(dp[n],dp[n-1]+num)
初始状态：

前 0 间房子的最大偷窃价值为 0 ，即 dp[0] = 0。
返回值：

返回 dp 列表最后一个元素值，即所有房间的最大偷窃价值。
简化空间复杂度：

我们发现 dp[n] 只与 dp[n−1] 和 dp[n−2] 有关系，因此我们可以设两个变量 cur和 pre 交替记录，将空间复杂度降到 O(1) 。

复杂度分析：
- 时间复杂度 O(N) ： 遍历 nums 需要线性时间；
- 空间复杂度 O(1) ： cur和 pre 使用常数大小的额外空间。


```go
func rob(nums []int) int {
	cur, pre := 0, 0
	for _, num := range nums {
		cur, pre = max(pre+num, cur), cur
	}
	return cur
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是数组长度。只需要对数组遍历一次。

- 空间复杂度：O(1)。使用滚动数组，可以只存储前两间房屋的最高总金额，而不需要存储整个数组的结果，因此空间复杂度是 O(1)。

[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

解题思路：


此题是 198. 打家劫舍 的拓展版： 唯一的区别是此题中的房间是环状排列的（即首尾相接），而 198.198. 题中的房间是单排排列的；而这也是此题的难点。

环状排列意味着第一个房子和最后一个房子中只能选择一个偷窃，因此可以把此环状排列房间问题约化为两个单排排列房间子问题：

在不偷窃第一个房子的情况下（即 nums[1:]），最大金额是 p1；

在不偷窃最后一个房子的情况下（即 nums[:n−1]），最大金额是 p2。
综合偷窃最大金额： 为以上两种情况的较大值，即 max(p1,p2) 。




```go
func rob(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	} else {
		return max(Rob(nums[:n-1]), Rob(nums[1:]))
	}
}
func Rob(nums []int) int {
	pre, cur := 0, 0
	for _, num := range nums {
		cur, pre = max(pre+num, cur), cur
	}
	return cur
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[337.打家劫舍III](https://leetcode-cn.com/problems/house-robber-iii/)






------

### 思路3：

