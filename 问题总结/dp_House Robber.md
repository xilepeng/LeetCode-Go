
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

``` go
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

``` go
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

``` go
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

``` go
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



``` go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]               // 只有一间房屋，则偷窃该房屋
	dp[1] = max(nums[0], nums[1]) // 只有两间房屋，选择其中金额较高的房屋进行偷窃
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-2]+nums[i], dp[i-1]) // dp[i] 前i间房屋能偷窃到的最高总金额 = max(抢第1间房子，不抢第1件房子)
	}
	return dp[len(dp)-1]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

上述方法使用了数组存储结果。考虑到每间房屋的最高总金额只和该房屋的前两间房屋的最高总金额相关，因此可以使用滚动数组，在每个时刻只需要存储前两间房屋的最高总金额。


``` go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	first := nums[0]                // 只有一间房屋，则偷窃该房屋
	second := max(nums[0], nums[1]) // 只有两间房屋，选择其中金额较高的房屋进行偷窃
	for i := 2; i < len(nums); i++ {
		first, second = second, max(first+nums[i], second) // dp[i] 前i间房屋能偷窃到的最高总金额 = max(抢第1间房子，不抢第1件房子)
	}
	return second
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


``` go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	first := nums[0]                // 只有一间房屋，则偷窃该房屋
	second := max(nums[0], nums[1]) // 只有两间房屋，选择其中金额较高的房屋进行偷窃
	for i := 2; i < len(nums); i++ {
		tmp := second
		second = max(first+nums[i], second) // dp[i] 前i间房屋能偷窃到的最高总金额 = max(抢第1间房子，不抢第1件房子)
		first = tmp
	}
	return second
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


``` go
func rob(nums []int) int {
	first, second := 0, 0
	for i := 0; i < len(nums); i++ {
		first, second = second, max(first+nums[i], second) 
	}
	return second
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



[官方题解](https://leetcode.cn/problems/house-robber/solution/da-jia-jie-she-by-leetcode-solution/)

[参考视频](https://www.bilibili.com/video/BV1qf4y1i7Mx)



[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

解题思路：


此题是 198. 打家劫舍 的拓展版： 唯一的区别是此题中的房间是环状排列的（即首尾相接），而 198.198. 题中的房间是单排排列的；而这也是此题的难点。

环状排列意味着第一个房子和最后一个房子中只能选择一个偷窃，因此可以把此环状排列房间问题约化为两个单排排列房间子问题：

在不偷窃第一个房子的情况下（即 nums[1:]），最大金额是 p1；

在不偷窃最后一个房子的情况下（即 nums[:n−1]），最大金额是 p2。
综合偷窃最大金额： 为以上两种情况的较大值，即 max(p1,p2) 。




``` go
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

